"""
Race a population of randomly initialised neural nets on one track.

Creates n NNControllers (each with its own random weights), saves every net's
weights and biases, then races all n cars at once on the same track from the
same start pose and records how each did.

The cars are "ghosts" to each other (see make_ghost_env): car-to-car collisions
are off and no car sees the others in its LiDAR, so every car drives exactly as
it would alone. A crashed car is stopped and left where it is.

Outputs, in runs/<track>_<timestamp>/:
    weights.npz  params      (n, P) flat weights+biases per car, as MLP.get_params()
                 layer_sizes layer sizes of the MLP, to rebuild it
                 seeds       weight-init seed per car
    results.csv  car, seed, crashed, death_time_s, sim_time_s, distance_m,
                 median_speed_mps, laps

    death_time_s is the sim time of the crash, empty if the car never crashed
    (it finished 2 laps or hit --max-time).

Rendering is on by default: the camera follows the leading car (orange),
running cars are purple and crashed cars grey. When the race ends the window
stays open until you close it. Closing it mid-race stops the race early and
still saves the results.

    python testing_env.py -n 20
    python testing_env.py -n 50 --map Monza --max-time 30
    python testing_env.py -n 200 --headless                  # no window, much faster
    python testing_env.py --load runs/<run>/weights.npz      # re-race saved nets
"""

import argparse
import csv
import os
import time

import gym
import numpy as np

from f110_gym.envs.base_classes import Integrator
from nn_controller import HERE, NNController, load_tracks

TIMESTEP = 0.01  # s, env physics step

# Car colours in the render window (RGB).
LEADER_COLOUR = (255, 140, 0)
ALIVE_COLOUR = (172, 97, 185)
DEAD_COLOUR = (110, 110, 110)


def is_window_closed(ex):
    # EnvRenderer.on_close raises a plain Exception with this message.
    return 'window was closed' in str(ex)


def add_race_view(env):
    """Render callback: follow the leader, colour cars by state, show race status."""
    def render_callback(e):
        v = env.unwrapped.view
        alive, distance = v['alive'], v['distance']
        leader = int(np.argmax(np.where(alive, distance, -1))) if alive.any() else int(np.argmax(distance))

        for i, car in enumerate(e.cars):
            colour = LEADER_COLOUR if i == leader else ALIVE_COLOUR if alive[i] else DEAD_COLOUR
            car.colors = list(colour) * 4

        x = e.cars[leader].vertices[::2]
        y = e.cars[leader].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.text = (f'Time: {v["time"]:.2f}s  Running: {alive.sum()}/{len(alive)}  '
                              f'Leader: car {leader} ({distance[leader]:.1f} m)')
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left, e.right = left - 800, right + 800
        e.top, e.bottom = top + 800, bottom - 800
    env.add_render_callback(render_callback)


def make_population(n, base_seed):
    """n controllers, each seeded base_seed + i so a run can be reproduced."""
    seeds = [base_seed + i for i in range(n)]
    return [NNController(seed=s) for s in seeds], seeds


def load_population(path):
    data = np.load(path)
    controllers = []
    for params in data['params']:
        c = NNController(hidden=tuple(data['layer_sizes'][1:-1]))
        c.net.set_params(params)
        controllers.append(c)
    return controllers, list(data['seeds'])


def save_weights(path, controllers, seeds):
    first = controllers[0].net
    layer_sizes = [first.weights[0].shape[0]] + [W.shape[1] for W in first.weights]
    np.savez(path,
             params=np.stack([c.net.get_params() for c in controllers]),
             layer_sizes=np.array(layer_sizes),
             seeds=np.array(seeds))


def make_ghost_env(map_path, n):
    """n-car env where the cars cannot collide with or see each other."""
    env = gym.make('f110_gym:f110-v0', map=map_path, map_ext='.png',
                   num_agents=n, timestep=TIMESTEP, integrator=Integrator.RK4)
    sim = env.unwrapped.sim

    # Car-to-car collisions: replace the GJK check with "nobody touching".
    # Wall collisions are checked separately per car and still apply.
    def no_car_collisions():
        sim.collisions = np.zeros(sim.num_agents)
        sim.collision_idx = -1 * np.ones(sim.num_agents)
    sim.check_collision = no_car_collisions

    # LiDAR: skip ray-casting the other cars into each car's scan.
    for agent in sim.agents:
        agent.ray_cast_agents = lambda scan: scan
    return env


def race(env, controllers, start_pose, max_steps, render):
    """Drive all cars at once until every car has crashed or time runs out.

    Returns (results, window_open). window_open is False if the render window
    was closed, which also ends the race early.
    """
    n = len(controllers)
    obs, _, _, _ = env.reset(np.array([start_pose] * n))
    for c in controllers:
        c.reset()

    alive = np.ones(n, dtype=bool)
    death_time = np.full(n, np.nan)
    distance = np.zeros(n)
    speeds = [[] for _ in range(n)]
    xy = np.column_stack([obs['poses_x'], obs['poses_y']])
    # Shared with the render callback; the arrays are updated in place.
    view = env.unwrapped.view = {'alive': alive, 'distance': distance, 'time': 0.0}

    sim_time = 0.0
    steps = 0
    # The env's own done flag only tracks car 0, so it is ignored here.
    while alive.any() and steps < max_steps:
        actions = np.zeros((n, 2))  # crashed cars get [0, 0] and stay put
        for i in np.flatnonzero(alive):
            actions[i] = controllers[i].plan(obs, agent=i)
        obs, step_reward, _, _ = env.step(actions)
        sim_time += step_reward
        steps += 1

        new_xy = np.column_stack([obs['poses_x'], obs['poses_y']])
        distance[alive] += np.linalg.norm(new_xy - xy, axis=1)[alive]
        xy = new_xy
        for i in np.flatnonzero(alive):
            speeds[i].append(abs(obs['linear_vels_x'][i]))
            if obs['collisions'][i]:
                alive[i] = False
                death_time[i] = sim_time
        if render:
            view['time'] = sim_time
            try:
                env.render(mode='human')
            except Exception as ex:
                if not is_window_closed(ex):
                    raise
                print(f'Window closed at {sim_time:.2f}s; stopping the race early.')
                render = False
                break

    results = []
    for i in range(n):
        crashed = not np.isnan(death_time[i])
        results.append({
            'crashed': crashed,
            'death_time_s': round(death_time[i], 3) if crashed else '',
            'sim_time_s': round(death_time[i] if crashed else sim_time, 3),
            'distance_m': round(distance[i], 3),
            'median_speed_mps': round(float(np.median(speeds[i])), 3) if speeds[i] else 0.0,
            'laps': int(obs['lap_counts'][i]),
        })
    return results, render


def main():
    ###################
    # CONFIGURE
    map = 'Budapest'
    n = 10
    render = True
    ###################
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=n, help='number of neural nets')
    parser.add_argument('--map', default=map, help='track name from start_poses.csv')
    parser.add_argument('--seed', type=int, default=0, help='base seed for weight init')
    parser.add_argument('--max-time', type=float, default=60.0, help='sim seconds per car before stopping it')
    parser.add_argument('--load', help='race nets from a saved weights.npz instead of new random ones')
    parser.add_argument('--headless', action='store_true', help='no render window (faster)')
    args = parser.parse_args()
    render = render and not args.headless

    tracks = load_tracks()
    if args.map not in tracks:
        parser.error(f'unknown track {args.map!r}; choose from: {", ".join(tracks)}')
    map_path, start_pose = tracks[args.map]

    if args.load:
        controllers, seeds = load_population(args.load)
    else:
        controllers, seeds = make_population(args.n, args.seed)

    run_dir = os.path.join(HERE, 'runs', f'{args.map.replace(" ", "")}_{time.strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(run_dir)
    save_weights(os.path.join(run_dir, 'weights.npz'), controllers, seeds)

    env = make_ghost_env(map_path, len(controllers))
    if render:
        add_race_view(env)

    max_steps = int(args.max_time / TIMESTEP)
    raced, window_open = race(env, controllers, start_pose, max_steps, render)
    results = [{'car': i, 'seed': seed, **r} for i, (seed, r) in enumerate(zip(seeds, raced))]
    for r in results:
        print(f"car {r['car']:3d}  {'crashed' if r['crashed'] else 'survived':8s}  "
              f"t={r['sim_time_s']:7.2f}s  dist={r['distance_m']:7.2f}m  "
              f"median v={r['median_speed_mps']:5.2f}m/s  laps={r['laps']}")

    with open(os.path.join(run_dir, 'results.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    best = max(results, key=lambda r: r['distance_m'])
    print(f"\nBest: car {best['car']} went {best['distance_m']} m. Saved to {run_dir}")

    # Keep the final frame up until the window is closed.
    if window_open:
        print('Race over. Close the render window to exit.')
        try:
            while True:
                env.render(mode='human')
        except Exception as ex:
            if not is_window_closed(ex):
                raise

if __name__ == '__main__':
    main()
