"""
Grid search over run.py hyperparameters, scored on the validation tracks.

Every combination of the values in --grid is trained by one run.py call (one after
another, since each run already uses all the worker processes). run.py holds out
--val-tracks random tracks (the same ones for every run, chosen by --val-seed) and the
best car races them at the end of every cycle. A config's score here is the mean of that
car's score over the validation tracks after the last cycle, averaged over --seeds.
The test tracks are not looked at.

    python grid_search.py                                   # sigma x mutation_rate, 9 runs
    python grid_search.py --grid sigma=0.02,0.05,0.1 n=100,300 -- -m 3 -p 2
    python grid_search.py --grid sigma=0.05,0.1 --seeds 0 1 2 --dry-run

Grid keys are run.py options with dashes written as underscores: sigma, mutation_rate,
n, tracks_per_group, num_train_tracks, gens_per_group (m), cycles (p), max_time, ...
The network size is fixed. Everything after a bare -- is passed to every run.py call
unchanged (e.g. -- -p 1 --val-tracks 5).

Outputs, in runs/grid_<timestamp>/:
    grid_results.csv   one row per run: the grid values, seed, final val score, run folder
    summary.csv        one row per config (seeds averaged), best first
    <config>_s<seed>/  the run.py output folder, with log.txt next to its files
"""

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time

import numpy as np

from nn_controller import HERE

DEFAULT_GRID = ['sigma=0.02,0.05,0.1', 'mutation_rate=0.1,0.5,1.0']
SHORT_FLAGS = {'n': '-n', 'm': '-m', 'p': '-p'}


def parse_grid(items):
    grid = {}
    for item in items:
        key, _, values = item.partition('=')
        if not values:
            raise SystemExit(f'bad --grid entry {item!r}; expected key=v1,v2,...')
        grid[key] = values.split(',')
    return grid


def flag(key):
    return SHORT_FLAGS.get(key, '--' + key.replace('_', '-'))


def final_val_score(run_dir):
    """Mean validation score of the last cycle in run_dir/val.csv, or nan if there is none."""
    path = os.path.join(run_dir, 'val.csv')
    if not os.path.exists(path):
        return float('nan')
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return float('nan')
    last = max(int(r['cycle']) for r in rows)
    return float(np.mean([float(r['score']) for r in rows if int(r['cycle']) == last]))


def total_gens(run_dir):
    try:
        with open(os.path.join(run_dir, 'config.json')) as f:
            return json.load(f)['total_generations']
    except (OSError, KeyError, ValueError):
        return ''


def main():
    argv = sys.argv[1:]
    extra = []
    if '--' in argv:
        i = argv.index('--')
        argv, extra = argv[:i], argv[i + 1:]
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--grid', nargs='+', default=DEFAULT_GRID, metavar='KEY=V1,V2',
                        help='values to try for each run.py option; every combination is run')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0],
                        help='run.py --seed values; each config is run once per seed and averaged')
    parser.add_argument('--dry-run', action='store_true', help='print the runs and exit')
    args = parser.parse_args(argv)

    grid = parse_grid(args.grid)
    if 'val_tracks' in grid or 'val_seed' in grid:
        parser.error('keep the validation set fixed across runs: pass --val-tracks / --val-seed after --')
    configs = [dict(zip(grid, combo)) for combo in itertools.product(*grid.values())]
    print(f'{len(configs)} configs x {len(args.seeds)} seed(s) = {len(configs) * len(args.seeds)} runs')

    grid_dir = os.path.join(HERE, 'runs', f'grid_{time.strftime("%Y%m%d_%H%M%S")}')
    jobs = []
    for ci, cfg in enumerate(configs):
        name = '_'.join(f'{k}{v}' for k, v in cfg.items())
        for seed in args.seeds:
            cmd = [sys.executable, os.path.join(HERE, 'run.py'), '--show-track', 'none',
                   '--seed', str(seed), '--out-dir', os.path.join(grid_dir, f'{name}_s{seed}')]
            for k, v in cfg.items():
                cmd += [flag(k), v]
            jobs.append((ci, cfg, seed, cmd + extra))
    if args.dry_run:
        for _, _, _, cmd in jobs:
            print(' '.join(cmd))
        return

    os.makedirs(grid_dir)
    results_path = os.path.join(grid_dir, 'grid_results.csv')
    with open(results_path, 'w', newline='') as f:
        csv.writer(f).writerow([*grid, 'seed', 'val_score', 'total_gens', 'wall_s', 'run_dir'])

    scores = {ci: [] for ci in range(len(configs))}
    try:
        for k, (ci, cfg, seed, cmd) in enumerate(jobs):
            run_dir = cmd[cmd.index('--out-dir') + 1]
            print(f'[{k + 1}/{len(jobs)}] {cfg} seed {seed}', flush=True)
            start = time.time()
            os.makedirs(run_dir, exist_ok=True)
            with open(os.path.join(run_dir, 'log.txt'), 'w') as log:
                code = subprocess.call(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
            score = final_val_score(run_dir)
            if code != 0:
                print(f'    run.py exited with code {code}; see {run_dir}/log.txt')
            print(f'    val score {score:.1f}  ({time.time() - start:.0f}s)', flush=True)
            scores[ci].append(score)
            with open(results_path, 'a', newline='') as f:
                csv.writer(f).writerow([*cfg.values(), seed, round(score, 3), total_gens(run_dir),
                                        round(time.time() - start, 1), run_dir])
    except KeyboardInterrupt:
        print('\nStopped; ranking the runs finished so far.')

    ranked = sorted((ci for ci in scores if scores[ci] and not np.all(np.isnan(scores[ci]))),
                    key=lambda ci: -np.nanmean(scores[ci]))
    with open(os.path.join(grid_dir, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rank', *grid, 'mean_val_score', 'runs'])
        print('\nrank  ' + '  '.join(f'{k:>14s}' for k in grid) + '   mean val score')
        for rank, ci in enumerate(ranked, 1):
            mean = float(np.nanmean(scores[ci]))
            w.writerow([rank, *configs[ci].values(), round(mean, 3), len(scores[ci])])
            print(f'{rank:4d}  ' + '  '.join(f'{v:>14s}' for v in configs[ci].values()) + f'   {mean:10.1f}')
    if ranked:
        print(f'\nBest: {configs[ranked[0]]}\nResults in {grid_dir}')


if __name__ == '__main__':
    main()
