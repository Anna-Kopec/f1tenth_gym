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

'''!
@file f110_env.py
@brief OpenAI gym environment for F1TENTH racing (F110Env).

F110Env is the public entry point of the package. It wraps
base_classes.Simulator (which does the physics, LiDAR and collisions) and adds:
- the gym interface: reset() / step() / render();
- lap counting and lap timing, using a "finish zone" around the start pose;
- the episode-end (done) condition;
- rendering through rendering.EnvRenderer.

@section f110_env_usage Typical usage
@code
import gym, numpy as np
env = gym.make('f110_gym:f110-v0', map='vegas', num_agents=1)
obs, reward, done, info = env.reset(np.array([[0., 0., 0.]]))
while not done:
    action = np.array([[steer, speed]])          # one row per agent
    obs, reward, done, info = env.step(action)
    env.render(mode='human')
@endcode

@author Hongrui Zheng
'''

# gym imports
import gym
from gym import error, spaces, utils
from gym.utils import seeding

# base classes
from f110_gym.envs.base_classes import Simulator, Integrator

# others
import numpy as np
import os
import time

# gl
import pyglet
pyglet.options['debug_gl'] = False
from pyglet import gl

# constants

# rendering
## Video width in pixels (currently unused).
VIDEO_W = 600
## Video height in pixels (currently unused).
VIDEO_H = 400
## Initial render window width in pixels.
WINDOW_W = 1000
## Initial render window height in pixels.
WINDOW_H = 800

class F110Env(gym.Env):
    """!
    @brief OpenAI gym environment for F1TENTH.

    Create it with @c gym.make('f110_gym:f110-v0', **kwargs). All keyword
    arguments are optional:

    | kwarg        | Default          | Meaning |
    |--------------|------------------|---------|
    | seed         | 12345            | Seed for the LiDAR noise generators. |
    | map          | (vegas)          | 'berlin', 'skirk' or 'levine' for a bundled map, or a path to a custom map's YAML file *without* the .yaml extension. If omitted, the bundled vegas map is used. |
    | map_ext      | '.png'           | Map image extension, including the dot ('.png', '.pgm'). |
    | params       | F1TENTH defaults | Vehicle parameter dict (see below). |
    | num_agents   | 2                | Number of cars. |
    | timestep     | 0.01             | Physics time step (s). |
    | ego_idx      | 0                | Index of the ego car. |
    | integrator   | Integrator.RK4   | Integrator.RK4 or Integrator.Euler. |
    | lidar_dist   | 0.0              | Forward offset of the LiDAR from the vehicle reference point (m). |

    Vehicle parameters (@c params dict):
    - @c mu: surface friction coefficient
    - @c C_Sf, @c C_Sr: front / rear cornering stiffness coefficients
    - @c lf, @c lr: distance from the centre of gravity to the front / rear axle (m)
    - @c h: height of the centre of gravity (m)
    - @c m: total mass (kg)
    - @c I: yaw moment of inertia (kg m^2)
    - @c s_min, @c s_max: steering angle limits (rad)
    - @c sv_min, @c sv_max: steering velocity limits (rad/s)
    - @c v_switch: speed above which acceleration is power-limited (m/s)
    - @c a_max: maximum longitudinal acceleration (m/s^2)
    - @c v_min, @c v_max: speed limits (m/s)
    - @c width, @c length: body size (m), used for collisions and opponent ray casting

    **Action:** np.ndarray of shape (num_agents, 2). Row i is
    [desired steering angle (rad), desired speed (m/s)] for car i.

    **Observation:** the dict from Simulator.step(), plus @c lap_times and @c lap_counts.

    @note The renderer, latest observation and render callbacks are *class* attributes,
          so they are shared by every F110Env instance in the process.
    """

    ## Supported render modes.
    metadata = {'render.modes': ['human', 'human_fast']}

    # rendering
    ## Shared EnvRenderer window, created on the first render() call.
    renderer = None
    ## Most recent observation dict from step(), shared across instances.
    current_obs = None
    ## Functions called with the renderer on every render(); add with add_render_callback().
    render_callbacks = []

    def __init__(self, **kwargs):
        """!
        @brief Read the kwargs (see the class documentation), create the Simulator and load the map.

        @param kwargs Optional settings; see the table in the class documentation.

        @note Each setting is read inside a bare @c try/except, so a misspelt kwarg
              is silently ignored and the default is used instead.
        """
        # kwargs extraction
        try:
            ## Seed for the LiDAR noise generators.
            self.seed = kwargs['seed']
        except:
            self.seed = 12345
        try:
            ## Map name or path as passed in (only set when the 'map' kwarg is given).
            self.map_name = kwargs['map']
            # different default maps
            if self.map_name == 'berlin':
                ## Full path to the map YAML file.
                self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/berlin.yaml'
            elif self.map_name == 'skirk':
                self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/skirk.yaml'
            elif self.map_name == 'levine':
                self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/levine.yaml'
            else:
                # custom map: the kwarg is a path without the .yaml extension
                self.map_path = self.map_name + '.yaml'
        except:
            self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/vegas.yaml'

        try:
            ## Map image extension, including the dot.
            self.map_ext = kwargs['map_ext']
        except:
            self.map_ext = '.png'

        try:
            ## Vehicle parameter dict shared by every car.
            self.params = kwargs['params']
        except:
            self.params = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}

        # simulation parameters
        try:
            ## Number of cars.
            self.num_agents = kwargs['num_agents']
        except:
            self.num_agents = 2

        try:
            ## Physics time step (s). Also the reward returned by every step().
            self.timestep = kwargs['timestep']
        except:
            self.timestep = 0.01

        # default ego index
        try:
            ## Index of the ego car.
            self.ego_idx = kwargs['ego_idx']
        except:
            self.ego_idx = 0

        # default integrator
        try:
            ## Integration scheme passed to the Simulator.
            self.integrator = kwargs['integrator']
        except:
            self.integrator = Integrator.RK4

        # default LiDAR position
        try:
            ## Forward offset of the LiDAR from the vehicle reference point (m).
            self.lidar_dist = kwargs['lidar_dist']
        except:
            self.lidar_dist = 0.0

        # radius to consider done
        ## Unused leftover; the finish-zone size is hard-coded in _check_done().
        self.start_thresh = 0.5  # 10cm

        # env states
        ## Latest x position of every car.
        self.poses_x = []
        ## Latest y position of every car.
        self.poses_y = []
        ## Latest yaw of every car.
        self.poses_theta = []
        ## Latest collision flag of every car (1.0 = collided).
        self.collisions = np.zeros((self.num_agents, ))
        # TODO: collision_idx not used yet
        # self.collision_idx = -1 * np.ones((self.num_agents, ))

        # loop completion
        ## Unused leftover (replaced by near_starts).
        self.near_start = True
        ## Unused leftover (replaced by toggle_list).
        self.num_toggles = 0

        # race info
        ## Lap time of each car (s): elapsed sim time, frozen once the car completes 2 laps.
        self.lap_times = np.zeros((self.num_agents, ))
        ## Number of completed laps for each car.
        self.lap_counts = np.zeros((self.num_agents, ))
        ## Simulated time since the last reset (s).
        self.current_time = 0.0

        # finish line info
        self.num_toggles = 0
        self.near_start = True
        ## Whether each car is currently inside the finish zone.
        self.near_starts = np.array([True]*self.num_agents)
        ## Number of times each car has entered or left the finish zone.
        #  Leaving and re-entering is one lap, so laps = toggles // 2.
        self.toggle_list = np.zeros((self.num_agents,))
        ## Starting x of each car (from reset()).
        self.start_xs = np.zeros((self.num_agents, ))
        ## Starting y of each car.
        self.start_ys = np.zeros((self.num_agents, ))
        ## Starting yaw of each car.
        self.start_thetas = np.zeros((self.num_agents, ))
        ## 2x2 rotation from the world frame into the ego car's starting frame
        #  (x along the start heading). Used to measure the finish zone.
        self.start_rot = np.eye(2)

        # initiate stuff
        ## The underlying multi-car Simulator.
        self.sim = Simulator(self.params, self.num_agents, self.seed, time_step=self.timestep, integrator=self.integrator, lidar_dist=self.lidar_dist)
        self.sim.set_map(self.map_path, self.map_ext)

        # stateful observations for rendering
        ## Subset of the latest observation passed to the renderer.
        self.render_obs = None

    def __del__(self):
        """!
        @brief Finalizer; does nothing.
        """
        pass

    def _check_done(self):
        """!
        @brief Update lap counts and times, and decide whether the episode is over.

        **Finish zone:** each car's position is expressed relative to its own start
        position, in a frame aligned with the *ego* car's starting heading:
        - forward (along the start heading): dx
        - sideways: dy, but anything within +/-2 m counts as 0, and beyond that
          only the excess over 2 m counts.

        A car is "in the zone" when dx^2 + (excess dy)^2 <= 0.1. That is roughly a
        box about 0.63 m long (+/-0.32 m) and 4 m wide, centred on the start pose.

        **Lap counting:** every time a car enters or leaves the zone, its toggle
        count goes up by 1. Cars start inside the zone, so leaving and coming back
        is 2 toggles, i.e. one lap. A car's lap time keeps updating until it
        reaches 4 toggles (2 laps), then freezes.

        **Done** when the ego car has collided, or every car has completed 2 laps.

        @return Tuple (done, checkpoint_done):
                - @c done (bool): whether the episode is over.
                - @c checkpoint_done (np.ndarray of bool, shape (num_agents,)):
                  whether each car has finished 2 laps.
        """

        # this is assuming 2 agents
        # TODO: switch to maybe s-based
        # half-width of the zone to the left / right of the start line (m)
        left_t = 2
        right_t = 2

        # position relative to each car's own start, rotated into the ego start frame
        poses_x = np.array(self.poses_x)-self.start_xs
        poses_y = np.array(self.poses_y)-self.start_ys
        delta_pt = np.dot(self.start_rot, np.stack((poses_x, poses_y), axis=0))
        # lateral offset: 0 inside the +/-2 m band, otherwise the distance past the band edge
        temp_y = delta_pt[1,:]
        idx1 = temp_y > left_t
        idx2 = temp_y < -right_t
        temp_y[idx1] -= left_t
        temp_y[idx2] = -right_t - temp_y[idx2]
        temp_y[np.invert(np.logical_or(idx1, idx2))] = 0

        dist2 = delta_pt[0, :]**2 + temp_y**2
        closes = dist2 <= 0.1
        for i in range(self.num_agents):
            # count every transition into or out of the zone
            if closes[i] and not self.near_starts[i]:
                self.near_starts[i] = True
                self.toggle_list[i] += 1
            elif not closes[i] and self.near_starts[i]:
                self.near_starts[i] = False
                self.toggle_list[i] += 1
            self.lap_counts[i] = self.toggle_list[i] // 2
            # keep the lap timer running until the car has done 2 laps
            if self.toggle_list[i] < 4:
                self.lap_times[i] = self.current_time

        done = (self.collisions[self.ego_idx]) or np.all(self.toggle_list >= 4)

        return bool(done), self.toggle_list >= 4

    def _update_state(self, obs_dict):
        """!
        @brief Copy poses and collision flags from an observation into the env's own state.

        @param obs_dict Observation dict from Simulator.step().
        """
        self.poses_x = obs_dict['poses_x']
        self.poses_y = obs_dict['poses_y']
        self.poses_theta = obs_dict['poses_theta']
        self.collisions = obs_dict['collisions']

    def step(self, action):
        """!
        @brief Advance the environment by one physics time step.

        @param action np.ndarray of shape (num_agents, 2). Row i is
                      [desired steering angle (rad), desired speed (m/s)] for car i.
        @return Tuple (obs, reward, done, info):
                - @c obs (dict): the observation from Simulator.step() (scans, poses,
                  velocities, collisions, ego_idx), plus @c lap_times and @c lap_counts.
                - @c reward (float): always equal to the time step. Design your own
                  reward for training.
                - @c done (bool): ego collided, or every car has finished 2 laps.
                - @c info (dict): @c checkpoint_done, a per-car bool array of
                  "finished 2 laps".

        @note @c lap_times and @c lap_counts in @c obs are the env's own arrays,
              not copies. They are updated in place, so an @c obs you keep from
              an earlier step will show later values. Copy them if you need a snapshot.
        """

        # call simulation step
        obs = self.sim.step(action)
        obs['lap_times'] = self.lap_times
        obs['lap_counts'] = self.lap_counts

        F110Env.current_obs = obs

        self.render_obs = {
            'ego_idx': obs['ego_idx'],
            'poses_x': obs['poses_x'],
            'poses_y': obs['poses_y'],
            'poses_theta': obs['poses_theta'],
            'lap_times': obs['lap_times'],
            'lap_counts': obs['lap_counts']
            }

        # times
        reward = self.timestep
        self.current_time = self.current_time + self.timestep

        # update data member
        self._update_state(obs)

        # check done
        done, toggle_list = self._check_done()
        info = {'checkpoint_done': toggle_list}

        return obs, reward, done, info

    def reset(self, poses):
        """!
        @brief Start a new episode with every car at the given pose.

        Clears the timer and lap-counting state, records the start poses (the
        finish zone is built around them), resets the Simulator, and then takes
        one step with zero input to produce the first observation.

        @param poses np.ndarray of shape (num_agents, 3). Row i is [x, y, yaw] for car i.
        @return Tuple (obs, reward, done, info), the same as step().

        @note Because reset() calls step() once, the physics has already
              advanced by one time step and current_time equals timestep.
        @note lap_times and lap_counts are not cleared directly; they are
              recalculated by the step() inside reset().
        """
        # reset counters and data members
        self.current_time = 0.0
        self.collisions = np.zeros((self.num_agents, ))
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True]*self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))

        # states after reset
        self.start_xs = poses[:, 0]
        self.start_ys = poses[:, 1]
        self.start_thetas = poses[:, 2]
        # rotation by -(ego start yaw): world frame -> ego start frame
        self.start_rot = np.array([[np.cos(-self.start_thetas[self.ego_idx]), -np.sin(-self.start_thetas[self.ego_idx])], [np.sin(-self.start_thetas[self.ego_idx]), np.cos(-self.start_thetas[self.ego_idx])]])

        # call reset to simulator
        self.sim.reset(poses)

        # get no input observations
        action = np.zeros((self.num_agents, 2))
        obs, reward, done, info = self.step(action)

        self.render_obs = {
            'ego_idx': obs['ego_idx'],
            'poses_x': obs['poses_x'],
            'poses_y': obs['poses_y'],
            'poses_theta': obs['poses_theta'],
            'lap_times': obs['lap_times'],
            'lap_counts': obs['lap_counts']
            }

        return obs, reward, done, info

    def update_map(self, map_path, map_ext):
        """!
        @brief Load a different map into the simulator.

        @param map_path Path to the map YAML file (with the .yaml extension).
        @param map_ext  Map image extension, including the dot.

        @note The renderer is not updated, and an existing render window keeps
              showing the old map.
        """
        self.sim.set_map(map_path, map_ext)

    def update_params(self, params, index=-1):
        """!
        @brief Change vehicle parameters for every car, or for one car.

        @param params Parameter dict (see the class documentation).
        @param index  Car index to update; -1 (the default) updates every car.

        @exception IndexError If @p index >= num_agents.
        """
        self.sim.update_params(params, agent_idx=index)

    def add_render_callback(self, callback_func):
        """!
        @brief Register a function that draws extra things on every render() call.

        The function receives the EnvRenderer, so it can add shapes to
        @c renderer.batch, for example to draw a planned path or waypoints.

        @param callback_func Function taking one argument (the EnvRenderer) and returning None.

        @note Callbacks are stored on the class, so they apply to every F110Env instance.
        """

        F110Env.render_callbacks.append(callback_func)

    def render(self, mode='human'):
        """!
        @brief Draw the environment in a pyglet window.

        Scroll to zoom and click-drag to pan. The window shows the map, the cars,
        the frame rate (bottom-left) and lap information. The window is created on
        the first call.

        @param mode Rendering mode:
                    - 'human': sleep 5 ms after each frame to slow things down for viewing.
                    - 'human_fast': render as fast as possible.

        @exception AssertionError If @p mode is not one of the supported modes.

        @note The first call passes @c self.map_name to the renderer, which
              expects a path without an extension. That only works for a custom
              map given as a path. For the bundled names ('berlin', 'skirk',
              'levine') it looks for e.g. 'berlin.yaml' in the current working
              directory, and if the 'map' kwarg was omitted, @c map_name does
              not exist at all.
        """
        assert mode in ['human', 'human_fast']

        if F110Env.renderer is None:
            # first call, initialize everything
            from f110_gym.envs.rendering import EnvRenderer
            F110Env.renderer = EnvRenderer(WINDOW_W, WINDOW_H)
            F110Env.renderer.update_map(self.map_name, self.map_ext)

        F110Env.renderer.update_obs(self.render_obs)

        for render_callback in F110Env.render_callbacks:
            render_callback(F110Env.renderer)

        # process window events (mouse, resize, close), draw, and swap buffers
        F110Env.renderer.dispatch_events()
        F110Env.renderer.on_draw()
        F110Env.renderer.flip()
        if mode == 'human':
            time.sleep(0.005)
        elif mode == 'human_fast':
            pass
