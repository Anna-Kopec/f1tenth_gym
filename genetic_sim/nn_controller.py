"""
Neural-network controller for the F1TENTH gym (draft).

A small fully connected network, written in plain NumPy, maps the car's
observation straight to an action. The weights are randomly initialised, so
the car will not drive well until they are trained (for example with an
evolutionary method such as CMA-ES, or with RL).

Network:
    inputs  (58) : 54 LiDAR bins (min over each bin, scaled to [0, 1]),
                   forward speed, yaw rate, previous steer, previous speed
    hidden       : 2 x 64, tanh
    outputs (2)  : steering angle -> tanh, scaled to [s_min, s_max]
                   speed          -> sigmoid, scaled to [speed_min, speed_max]

Track maps and start poses come from maps/start_poses/start_poses.csv (built by
maps/start_poses/extract_start_poses.py). Paths are
relative to this file, so it can be run from any folder:
    python nn_controller.py                          # Budapest, rendered
    python nn_controller.py --headless               # no window, prints a summary
    python nn_controller.py --map "Mexico City"      # any track in start_poses.csv
"""

import argparse
import csv
import time
import os
import gym
import numpy as np

from f110_gym.envs.base_classes import Integrator

HERE = os.path.dirname(os.path.abspath(__file__))
MAPS_DIR = os.path.join(HERE, 'maps', 'f1tenth_racetracks')
# Built by extract_start_poses.py: track, map, x, y, theta per track.
START_POSES_CSV = os.path.join(HERE, 'maps', 'start_poses', 'start_poses.csv')


def load_tracks():
    """Return {track: (map path without extension, (x, y, theta))}."""
    tracks = {}
    with open(START_POSES_CSV, newline='') as f:
        for row in csv.DictReader(f):
            pose = (float(row['x']), float(row['y']), float(row['theta']))
            tracks[row['track']] = (os.path.join(MAPS_DIR, row['map']), pose)
    return tracks

# Vehicle limits, taken from the env's default params.
S_MIN, S_MAX = -0.4189, 0.4189  # steering angle (rad)
SPEED_MIN, SPEED_MAX = 0.5, 8.0  # commanded speed (m/s); capped below v_max for a sane draft
LIDAR_MAX_RANGE = 30.0  # m, scanner max range
NUM_BEAMS = 1080
LIDAR_BINS = 54  # 1080 / 54 = 20 beams per bin


class MLP:
    """Fully connected net with tanh hidden layers and a linear output layer."""

    def __init__(self, layer_sizes, rng):
        self.weights = []
        self.biases = []
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            # Xavier/Glorot uniform init, suited to tanh.
            limit = np.sqrt(6.0 / (n_in + n_out))
            self.weights.append(rng.uniform(-limit, limit, size=(n_in, n_out)))
            self.biases.append(np.zeros(n_out))

    def forward(self, x):
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            x = np.tanh(x @ W + b)
        return x @ self.weights[-1] + self.biases[-1]

    # Flat parameter vector access, so an optimiser (e.g. CMA-ES) can tune the net.
    def get_params(self):
        return np.concatenate([p.ravel() for p in self.weights + self.biases])

    def set_params(self, flat):
        i = 0
        for p in self.weights + self.biases:
            p[...] = flat[i:i + p.size].reshape(p.shape)
            i += p.size


class NNController:
    """Turns a gym observation into [steer, speed] using the MLP."""

    def __init__(self, hidden=(64, 64), seed=None):
        rng = np.random.default_rng(seed)
        n_inputs = LIDAR_BINS + 4
        self.net = MLP([n_inputs, *hidden, 2], rng)
        self.prev_steer = 0.0
        self.prev_speed = 0.0

    def reset(self):
        self.prev_steer = 0.0
        self.prev_speed = 0.0

    def features(self, obs, agent=0):
        scan = np.asarray(obs['scans'][agent], dtype=np.float64)
        # Min-pool each bin: the closest obstacle in that direction matters most.
        bins = scan.reshape(LIDAR_BINS, -1).min(axis=1)
        lidar = np.clip(bins / LIDAR_MAX_RANGE, 0.0, 1.0)

        state = np.array([
            obs['linear_vels_x'][agent] / SPEED_MAX,
            obs['ang_vels_z'][agent] / 5.0,
            self.prev_steer / S_MAX,
            self.prev_speed / SPEED_MAX,
        ])
        return np.concatenate([lidar, state])

    def plan(self, obs, agent=0):
        raw_steer, raw_speed = self.net.forward(self.features(obs, agent))

        # Steering head: tanh -> [-1, 1] -> [s_min, s_max].
        steer = S_MIN + (np.tanh(raw_steer) + 1.0) * 0.5 * (S_MAX - S_MIN)
        # Speed head: sigmoid -> [0, 1] -> [speed_min, speed_max].
        speed = SPEED_MIN + (1.0 / (1.0 + np.exp(-raw_speed))) * (SPEED_MAX - SPEED_MIN)

        self.prev_steer, self.prev_speed = steer, speed
        return steer, speed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--headless', action='store_true', help='run without rendering')
    parser.add_argument('--seed', type=int, default=None, help='seed for weight init')
    parser.add_argument('--max-steps', type=int, default=20000)
    parser.add_argument('--map', default='Budapest', help='track name from start_poses.csv')
    parser.add_argument('--start', type=float, nargs=3, metavar=('X', 'Y', 'THETA'),
                        help='override the start pose from start_poses.csv')
    args = parser.parse_args()

    tracks = load_tracks()
    if args.map not in tracks:
        parser.error(f'unknown track {args.map!r}; choose from: {", ".join(tracks)}')
    # The env takes the map path without extension and appends .yaml / map_ext itself.
    map_path, start_pose = tracks[args.map]
    if args.start is not None:
        start_pose = args.start

    controller = NNController(seed=args.seed)
    print('Network parameters:', controller.net.get_params().size)

    env = gym.make('f110_gym:f110-v0', map=map_path, map_ext='.png',
                   num_agents=1, timestep=0.01, integrator=Integrator.RK4)

    if not args.headless:
        def render_callback(e):
            # Follow car 0, like waypoint_follow.py.
            x = e.cars[0].vertices[::2]
            y = e.cars[0].vertices[1::2]
            top, bottom, left, right = max(y), min(y), min(x), max(x)
            e.score_label.x = left
            e.score_label.y = top - 700
            e.left, e.right = left - 800, right + 800
            e.top, e.bottom = top + 800, bottom - 800
        env.add_render_callback(render_callback)

    obs, step_reward, done, info = env.reset(np.array([start_pose]))
    controller.reset()

    laptime = 0.0
    steps = 0
    start = time.time()
    while not done and steps < args.max_steps:
        steer, speed = controller.plan(obs)
        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
        laptime += step_reward
        steps += 1
        if not args.headless:
            env.render(mode='human')

    print(f'Steps: {steps}  Sim time: {laptime:.2f}s  Real time: {time.time() - start:.2f}s  '
          f'Collided: {bool(obs["collisions"][0])}  Laps: {int(obs["lap_counts"][0])}')


if __name__ == '__main__':
    main()
