"""
Genetic training: evolve the NN controller by mutating the best car each generation.

Each generation races a population of n cars on a group of tracks at once. The
n x tracks-per-group car-races are split evenly over the worker processes (75%
of the CPU threads by default), each job racing a slice of the cars on one track; the
cars are ghosts, so the split does not change any result. A car's score for the generation is the
average of its scores across the tracks in the group. The best car then seeds
the next generation:

    car 0        the best car, unchanged (so the best never gets worse)
    cars 1..n-1  copies of the best car with small Gaussian noise on the weights

Tracks: the 23 tracks in start_poses.csv are split three ways:
    test        2 tracks (--test-tracks, default YasMarina and Zandvoort), never trained on
    validation  --val-tracks N (default 3) picked at random with --val-seed, never trained
                on; used to compare hyperparameters (see grid_search.py)
    training    the rest (18 by default), or a random --num-train-tracks subset of them
The training tracks are cut into groups of --tracks-per-group (default 3, so 6 groups).
Each group is raced for m generations (-m), then the next group takes over. One pass
through all the groups is a cycle, and the run does p cycles (-p, default 2):

    cycle 1: gens 0..m-1    first group of tracks
             gens m..2m-1   second group
             ...
             VAL + TEST     best car races each validation and test track; weights untouched
    cycle 2: same groups again, then VAL + TEST again
    ...
    END      the renderer opens and the final best car drives --show-track
             (default: the first test track)

Generation 0 is n random nets, or the best car of a saved run with --load.

Outputs, in runs/genetic_<timestamp>/ (rewritten every generation, so a run
stopped with Ctrl+C keeps everything up to the last finished generation):
    config.json        the settings the run used
    history.csv        one row per generation: tracks, best/mean score, per-track
                       score of the best car
    generations.csv    one row per car per generation, with per-track scores
    best.npz           best car of the latest generation, in the weights.npz
                       format, so it can be watched with
                       python testing_env.py --load runs/<run>/best.npz --map Monza
    best_per_gen.npz   params (G, P) best car of every generation, layer_sizes
    population.npz     the whole latest population (weights.npz format)
    val.csv            one row per validation track per cycle
    test.csv           one row per test track per cycle
    best_cycleC.npz    best car at the end of cycle C

    python run.py                                      # n=100, m=3, p=2: 36 generations
    python run.py -n 200 -m 5 -p 4
    python run.py --show-track Monza                   # what to render at the end
    python run.py --show-track none                    # no window at the end
    python run.py --load runs/genetic_<ts>/best.npz --sigma 0.02
"""

import argparse
import csv
import json
import os
import time
import warnings
from multiprocessing import Pool

# One BLAS thread per process: the nets are tiny, and with a worker per core
# extra threads only fight each other. Set before numpy loads; workers inherit it.
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import numpy as np

from nn_controller import HERE, LIDAR_BINS, NNController, load_tracks
from testing_env import (TIMESTEP, add_race_view, is_window_closed, load_population,
                         make_ghost_env, make_population, race, save_weights)


NUM_AGENTS = 100
GENS_PER_GROUP = 3     # m: generations on one group of tracks before moving on
TRACKS_PER_GROUP = 3     # tracks raced in parallel each generation
SIGMA = 0.05              # std of the Gaussian noise added to mutated weights
MUTATION_RATE = 1.0       # fraction of weights that get noise in each mutant
CYCLES = 2                # p: passes through all the training groups
TEST_TRACKS = ['YasMarina', 'Zandvoort']  # held out of training; the rest are training/validation
VAL_TRACKS = 3            # random tracks held out of training to compare hyperparameters


# ---------------------------------------------------------------- worker side
# Each worker process keeps its envs between generations so a track's map is
# only loaded once per group. The cache is dropped when the group changes.
_envs = {}
_envs_group = None


def _quiet_worker():
    # Every env build warns that RK4 is the integrator; once is plenty.
    warnings.filterwarnings('ignore', message='Chosen integrator is RK4')


def _race_track(job):
    """Race a chunk of the population on one track. Runs in a worker process."""
    global _envs_group
    group, track, lo, map_path, start_pose, params, hidden, max_steps = job
    if group != _envs_group:
        _envs.clear()
        _envs_group = group
    n = len(params)
    if (map_path, n) not in _envs:
        _envs[(map_path, n)] = make_ghost_env(map_path, n)
    env = _envs[(map_path, n)]

    results, _ = race(env, params, hidden, start_pose, max_steps, render=False)
    return track, lo, results


# ---------------------------------------------------------------- main side
def split_jobs(n_cars, n_tracks, workers):
    """Slices (lo, hi) of the population that each track is split into.

    The work is n_cars * n_tracks car-races, shared as evenly as possible over
    the workers: each job races cars_per_job cars on one track (a job cannot
    span tracks, since it runs in one env).
    """
    cars_per_job = -(-n_cars * n_tracks // workers)  # ceil
    bounds = np.linspace(0, n_cars, -(-n_cars // cars_per_job) + 1).astype(int)
    return list(zip(bounds[:-1], bounds[1:]))


def race_tracks(pool, group, tracks, all_tracks, params, hidden, max_steps, workers):
    """Race every car on every track; returns {track: [result per car]}.

    The cars are ghosts to each other, so splitting them across processes does
    not change any car's result, only how many cores share the work.
    """
    jobs = [(group, t, lo, *all_tracks[t], params[lo:hi], hidden, max_steps)
            for t in tracks for lo, hi in split_jobs(len(params), len(tracks), workers)]
    per_track = {t: [None] * len(params) for t in tracks}
    for t, lo, results in pool.imap_unordered(_race_track, jobs):
        per_track[t][lo:lo + len(results)] = results
    return per_track



def mutate(parent, n, sigma, rate, rng):
    """n-1 mutants of parent, with the unchanged parent first."""
    children = np.repeat(parent[None, :], n, axis=0)
    noise = rng.normal(0.0, sigma, size=(n - 1, parent.size))
    if rate < 1.0:
        noise *= rng.random(noise.shape) < rate
    children[1:] += noise
    return children


def track_group(track_names, g, size):
    """The g-th group of `size` consecutive tracks, wrapping round the list."""
    return [track_names[(g * size + k) % len(track_names)] for k in range(size)]


def best_from_saved(path):
    """Params of the best car in a saved weights.npz (by results.csv next to it, if any)."""
    controllers, _ = load_population(path)
    hidden = tuple(np.load(path)['layer_sizes'][1:-1])
    best = 0
    results_csv = os.path.join(os.path.dirname(path), 'results.csv')
    if len(controllers) > 1 and os.path.exists(results_csv):
        with open(results_csv, newline='') as f:
            scores = [float(r['score']) for r in csv.DictReader(f)]
        best = int(np.argmax(scores))
    return controllers[best].net.get_params(), hidden


def write_population(path, params, hidden):
    controllers = []
    for p in params:
        c = NNController(hidden=hidden)
        c.net.set_params(p)
        controllers.append(c)
    save_weights(path, controllers, list(range(len(controllers))))


RESULT_COLUMNS = ['cycle', 'gen', 'track', 'score', 'crashed', 'sim_time_s',
                  'distance_m', 'median_speed_mps', 'laps']


def eval_best(pool, label, cycle, gen, params, hidden, tracks, all_tracks, max_steps, path):
    """Race the best car on each of `tracks` and append the results to `path`; no weights change."""
    per_track = race_tracks(pool, (label, cycle), tracks, all_tracks, params[None, :], hidden,
                            max_steps, workers=len(tracks))
    print(f'----- {label.upper()} after cycle {cycle + 1} (best car of gen {gen}, weights not updated)')
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        for t in tracks:
            r = per_track[t][0]
            print(f'      {t:14s} score {r["score"]:8.1f}  {"crashed" if r["crashed"] else "survived":8s}  '
                  f't={r["sim_time_s"]:6.2f}s  dist={r["distance_m"]:7.1f}m  laps={r["laps"]}')
            w.writerow([cycle + 1, gen, t, r['score'], int(r['crashed']), r['sim_time_s'],
                        r['distance_m'], r['median_speed_mps'], r['laps']])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('-n', type=int, default=NUM_AGENTS, help='cars per generation')
    parser.add_argument('-m', '--gens-per-group', type=int, default=GENS_PER_GROUP,
                        help='generations on each group of tracks before moving to the next')
    parser.add_argument('--num-train-tracks', type=int, default=0,
                        help='train on only this many tracks, a seeded random subset of the '
                             'training tracks (--val-seed); 0 = all of them')
    parser.add_argument('--tracks-per-group', type=int, default=TRACKS_PER_GROUP,
                        help='tracks raced in parallel each generation')
    parser.add_argument('--tracks', nargs='+',
                        help='training tracks, in order (default: every track in start_poses.csv '
                             'that is not a test track)')
    parser.add_argument('--test-tracks', nargs='+', default=TEST_TRACKS,
                        help='held-out tracks, raced by the best car at the end of every cycle')
    parser.add_argument('--val-tracks', type=int, default=VAL_TRACKS,
                        help='number of random tracks held out of training for validation '
                             '(0 to disable)')
    parser.add_argument('--val-seed', type=int, default=0,
                        help='seed for picking the validation tracks (separate from --seed so '
                             'every run of a grid search validates on the same tracks)')
    parser.add_argument('--out-dir', help='write the run here instead of runs/genetic_<timestamp>')
    parser.add_argument('--shuffle-tracks', action='store_true', help='shuffle the training track order once at the start')
    parser.add_argument('-p', '--cycles', type=int, default=CYCLES,
                        help='passes through all the training groups')
    parser.add_argument('--sigma', type=float, default=SIGMA, help='std of mutation noise')
    parser.add_argument('--mutation-rate', type=float, default=MUTATION_RATE,
                        help='fraction of weights mutated in each mutant')
    parser.add_argument('--max-time', type=float, default=60.0, help='sim seconds per race')
    parser.add_argument('--seed', type=int, default=0, help='seed for gen-0 nets and mutation noise')
    parser.add_argument('--load', help='start from the best car in this weights.npz / best.npz')
    parser.add_argument('--show-track',
                        help='track to render the final best car on at the end (default: first test '
                             'track; "none" to skip)')
    parser.add_argument('--workers', type=int, default=0,
                        help='worker processes; the n x tracks-per-group car-races are split evenly '
                             'across them (default: 75%% of CPU threads)')
    args = parser.parse_args()
    if args.n < 2:
        parser.error('-n must be at least 2')

    all_tracks = load_tracks()
    test_tracks = args.test_tracks
    unknown = [t for t in (args.tracks or []) + test_tracks if t not in all_tracks]
    if unknown:
        parser.error(f'unknown track(s) {unknown}; choose from: {", ".join(all_tracks)}')
    # Validation tracks: a seeded random pick from the tracks not used for testing (or, with
    # --tracks, not named there), then taken out of the training list.
    candidates = [t for t in all_tracks if t not in test_tracks]
    train_candidates = args.tracks or candidates
    val_pool = [t for t in candidates if t not in train_candidates] if args.tracks else candidates
    if args.val_tracks > len(val_pool) - (0 if args.tracks else 1):
        parser.error(f'--val-tracks {args.val_tracks} is too many: only {len(val_pool)} tracks are '
                     f'available for validation')
    picked = set(np.random.default_rng(args.val_seed).choice(len(val_pool), args.val_tracks,
                                                              replace=False).tolist())
    val_tracks = [t for i, t in enumerate(val_pool) if i in picked]
    track_names = [t for t in train_candidates if t not in val_tracks]
    if args.num_train_tracks:
        if not 0 < args.num_train_tracks <= len(track_names):
            parser.error(f'--num-train-tracks must be between 1 and {len(track_names)}')
        keep = set(np.random.default_rng(args.val_seed + 1).choice(
            len(track_names), args.num_train_tracks, replace=False).tolist())
        track_names = [t for i, t in enumerate(track_names) if i in keep]
    if args.show_track not in (None, 'none') and args.show_track not in all_tracks:
        parser.error(f'unknown --show-track {args.show_track!r}')
    overlap = (set(track_names) | set(val_tracks)) & set(test_tracks)
    if overlap:
        parser.error(f'track(s) {sorted(overlap)} are both training and test tracks')
    rng = np.random.default_rng(args.seed)
    if args.shuffle_tracks:
        track_names = [str(t) for t in rng.permutation(track_names)]

    k = args.tracks_per_group
    n_groups = -(-len(track_names) // k)  # ceil; a short last group wraps round to the start
    if len(track_names) % k:
        print(f'Note: {len(track_names)} training tracks do not split evenly into groups of {k}; '
              f'the last group wraps round to the first tracks.')
    gens_per_cycle = n_groups * args.gens_per_group
    total_gens = args.cycles * gens_per_cycle

    # Generation 0.
    if args.load:
        best_params, hidden = best_from_saved(args.load)
        population = mutate(best_params, args.n, args.sigma, args.mutation_rate, rng)
    else:
        controllers, _ = make_population(args.n, args.seed)
        hidden = tuple(W.shape[1] for W in controllers[0].net.weights[:-1])
        population = np.stack([c.net.get_params() for c in controllers])
    hidden = tuple(int(h) for h in hidden)

    run_dir = args.out_dir or os.path.join(HERE, 'runs', f'genetic_{time.strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(run_dir, exist_ok=bool(args.out_dir))
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump({**vars(args), 'track_order': track_names, 'test_tracks': test_tracks,
                   'val_track_list': val_tracks,
                   'groups': n_groups, 'total_generations': total_gens, 'hidden': hidden}, f, indent=2)

    val_path = os.path.join(run_dir, 'val.csv')
    test_path = os.path.join(run_dir, 'test.csv')
    for path in (val_path, test_path):
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow(RESULT_COLUMNS)
    history_path = os.path.join(run_dir, 'history.csv')
    gens_path = os.path.join(run_dir, 'generations.csv')
    with open(history_path, 'w', newline='') as f:
        csv.writer(f).writerow(['gen', 'group', 'tracks', 'best_car', 'best_score', 'mean_score',
                                'best_track_scores', 'best_laps', 'best_crashes', 'wall_time_s'])
    with open(gens_path, 'w', newline='') as f:
        csv.writer(f).writerow(['gen', 'car', 'avg_score'] + [f'track{i}' for i in range(k)] +
                               [f'score{i}' for i in range(k)] + [f'crashed{i}' for i in range(k)])

    max_steps = int(args.max_time / TIMESTEP)
    layer_sizes = np.array([LIDAR_BINS + 4, *hidden, 2])
    best_history = []
    print(f'Run: {run_dir}\n{args.n} cars, {len(track_names)} training tracks in {n_groups} groups of {k}, '
          f'{args.gens_per_group} gens per group, {args.cycles} cycles = {total_gens} generations\n'
          f'Validation tracks: {", ".join(val_tracks) or "none"}\n'
          f'Test tracks: {", ".join(test_tracks) or "none"}\n'
          f'sigma={args.sigma}, mutation rate={args.mutation_rate}\n')

    # Default to 75% of the CPU threads (at least 1); more workers than car-races would only sit idle.
    default_workers = max(1, (os.cpu_count() or 1) * 3 // 4)
    workers = min(args.workers or default_workers, args.n * k)
    slices = split_jobs(args.n, k, workers)
    print(f'{workers} worker processes: {args.n} cars x {k} tracks = {args.n * k} car-races per generation, '
          f'as {len(slices) * k} jobs of up to {max(hi - lo for lo, hi in slices)} cars\n')
    pool = Pool(workers, initializer=_quiet_worker)
    gen = 0
    try:
        while gen < total_gens:
            group = gen // args.gens_per_group  # counts up across cycles, so env caches reset per group
            cycle = gen // gens_per_cycle
            tracks = track_group(track_names, group % n_groups, k)
            if gen % gens_per_cycle == 0:
                print(f'===== cycle {cycle + 1}/{args.cycles}')
            start = time.time()
            per_track = race_tracks(pool, group, tracks, all_tracks, population, hidden, max_steps, workers)

            scores = np.array([[r['score'] for r in per_track[t]] for t in tracks])  # (k, n)
            avg = scores.mean(axis=0)
            best = int(np.argmax(avg))
            best_params = population[best].copy()
            best_history.append(best_params)
            wall = time.time() - start

            best_runs = [per_track[t][best] for t in tracks]
            print(f'gen {gen:4d}  [{", ".join(tracks)}]  best car {best:3d}  avg {avg[best]:8.1f}  '
                  f'(mean {avg.mean():7.1f})  per track: '
                  + '  '.join(f'{s:7.1f}{"x" if r["crashed"] else " "}'
                              for s, r in zip(scores[:, best], best_runs))
                  + f'  {wall:5.1f}s')

            with open(history_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    gen, group, '|'.join(tracks), best, round(avg[best], 3), round(avg.mean(), 3),
                    '|'.join(f'{s:.3f}' for s in scores[:, best]),
                    '|'.join(str(r['laps']) for r in best_runs),
                    '|'.join(str(int(r['crashed'])) for r in best_runs),
                    round(wall, 2)])
            with open(gens_path, 'a', newline='') as f:
                w = csv.writer(f)
                for car in range(args.n):
                    w.writerow([gen, car, round(avg[car], 3)] + tracks +
                               [round(s, 3) for s in scores[:, car]] +
                               [int(per_track[t][car]['crashed']) for t in tracks])

            write_population(os.path.join(run_dir, 'population.npz'), population, hidden)
            write_population(os.path.join(run_dir, 'best.npz'), best_params[None, :], hidden)
            np.savez(os.path.join(run_dir, 'best_per_gen.npz'), params=np.stack(best_history),
                     layer_sizes=layer_sizes)

            if (gen + 1) % gens_per_cycle == 0:
                write_population(os.path.join(run_dir, f'best_cycle{cycle + 1}.npz'),
                                 best_params[None, :], hidden)
                for label, eval_tracks, path in (('val', val_tracks, val_path),
                                                 ('test', test_tracks, test_path)):
                    if eval_tracks:
                        eval_best(pool, label, cycle, gen, best_params, hidden, eval_tracks,
                                  all_tracks, max_steps, path)
            elif (gen + 1) % args.gens_per_group == 0:
                print(f'--- group {group % n_groups} done; next: '
                      f'{", ".join(track_group(track_names, (group + 1) % n_groups, k))}')
            population = mutate(best_params, args.n, args.sigma, args.mutation_rate, rng)
            gen += 1
    except KeyboardInterrupt:
        print('\nStopped.')
        pool.terminate()
        pool.join()
        print(f'{gen} generations saved to {run_dir}')
        return
    pool.close()
    pool.join()
    print(f'{gen} generations saved to {run_dir}')

    if args.show_track != 'none':
        show_track = args.show_track or (test_tracks[0] if test_tracks else track_names[0])
        show_best(best_params, hidden, show_track, all_tracks, max_steps)


def show_best(params, hidden, track, all_tracks, max_steps):
    """Render the final best car on one track; the window stays up until closed."""
    print(f'\nShowing the final best car on {track}. Close the window to exit.')
    map_path, start_pose = all_tracks[track]
    env = make_ghost_env(map_path, 1)
    add_race_view(env)
    (r,), window_open = race(env, params[None, :], hidden, start_pose, max_steps, render=True)
    print(f'{track}: score {r["score"]:.1f}  {"crashed" if r["crashed"] else "survived"}  '
          f'dist={r["distance_m"]:.1f}m  laps={r["laps"]}')
    if window_open:
        try:
            while True:
                env.render(mode='human')
        except Exception as ex:
            if not is_window_closed(ex):
                raise


if __name__ == '__main__':
    main()
