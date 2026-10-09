# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.



"""!
@file legacy_scan_gen.py
@brief Script that generated legacy_scan.npz: reference LiDAR scans from the old C++-backed gym.

For each of three maps (vegas, berlin, skirk), it resets the old environment at
10 test poses (at the origin, yaw from -1 to 1 rad) and records the ego car's
1080-beam scan. The scans are saved to legacy_scan.npz under the keys
@c vegas, @c berlin and @c skirk, and scan_sim.py's tests compare against them.

@warning This script uses the **old** gym API (init_map(), update_params() with a
         C++ executable directory, reset() with a dict of poses), which the current
         F110Env does not have, and paths to a C++ build and maps that are not in this
         repository. It is kept as a record of how the data was made and will not run as-is.

@author Hongrui Zheng
"""

import numpy as np
import gym
import matplotlib.pyplot as plt

## Beam angles of the legacy 1080-beam, 4.7 rad scan (unused below; kept for plotting).
thetas = np.linspace(-2.35, 2.35, num=1080)

# init: vehicle parameters for the legacy simulator
## Directory of the compiled legacy C++ simulator.
executable_dir = '../../../build/'
mass= 3.74
l_r = 0.17145
I_z = 0.04712
mu = 0.523
h_cg = 0.074
cs_f = 4.718
cs_r = 5.4562

# test poses: rotate in place at the origin, yaw from -1 to 1 rad
## Number of test poses per map.
num_test = 10
test_poses = np.zeros((num_test, 3))
test_poses[:, 2] = np.linspace(-1., 1., num=num_test)

# map 1: vegas
map_path = '../../../maps/vegas.yaml'
map_ext = '.png'
racecar_env = gym.make('f110_gym:f110-v0')
racecar_env.init_map(map_path, map_ext, False, False)
racecar_env.update_params(mu, h_cg, l_r, cs_f, cs_r, I_z, mass, executable_dir, double_finish=True)
vegas_scan = np.empty((num_test, 1080))
for i in range(test_poses.shape[0]):
    # the legacy env always had 2 cars; park the second one far away at (200, 200)
    x = [test_poses[i, 0], 200.]
    y = [test_poses[i, 1], 200.]
    theta = [test_poses[i, 2], 0.]
    obs,_,_,_ = racecar_env.reset({'x': x, 'y': y, 'theta': theta})
    vegas_scan[i,:] = obs['scans'][0]

# map 2: berlin
map_path = '../../../maps/berlin.yaml'
map_ext = '.png'
racecar_env = gym.make('f110_gym:f110-v0')
racecar_env.init_map(map_path, map_ext, False, False)
racecar_env.update_params(mu, h_cg, l_r, cs_f, cs_r, I_z, mass, executable_dir, double_finish=True)
berlin_scan = np.empty((num_test, 1080))
for i in range(test_poses.shape[0]):
    x = [test_poses[i, 0], 200.]
    y = [test_poses[i, 1], 200.]
    theta = [test_poses[i, 2], 0.]
    obs,_,_,_ = racecar_env.reset({'x': x, 'y': y, 'theta': theta})
    berlin_scan[i,:] = obs['scans'][0]

# map 3: skirk
map_path = '../../../maps/skirk.yaml'
map_ext = '.png'
racecar_env = gym.make('f110_gym:f110-v0')
racecar_env.init_map(map_path, map_ext, False, False)
racecar_env.update_params(mu, h_cg, l_r, cs_f, cs_r, I_z, mass, executable_dir, double_finish=True)
skirk_scan = np.empty((num_test, 1080))
for i in range(test_poses.shape[0]):
    x = [test_poses[i, 0], 200.]
    y = [test_poses[i, 1], 200.]
    theta = [test_poses[i, 2], 0.]
    obs,_,_,_ = racecar_env.reset({'x': x, 'y': y, 'theta': theta})
    skirk_scan[i,:] = obs['scans'][0]

# package data
np.savez_compressed('legacy_scan.npz', vegas=vegas_scan, berlin=berlin_scan, skirk=skirk_scan)