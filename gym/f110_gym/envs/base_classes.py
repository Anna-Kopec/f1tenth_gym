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
@file base_classes.py
@brief Core simulation classes for the F1TENTH gym: a single vehicle (RaceCar)
       and the multi-vehicle world that steps them together (Simulator).

These are Python replacements for the original C++ RaceCar / Simulator classes.

@section base_classes_state_layout Vehicle state vector
Every RaceCar keeps its full dynamic state in a 7-element numpy array, laid out as:

| Index | Symbol      | Meaning                                         | Units  |
|-------|-------------|-------------------------------------------------|--------|
| 0     | x           | world-frame x of the vehicle reference point    | m      |
| 1     | y           | world-frame y of the vehicle reference point    | m      |
| 2     | delta       | front-wheel steering angle                      | rad    |
| 3     | v           | longitudinal velocity                           | m/s    |
| 4     | psi         | yaw (heading), wrapped to [0, 2*pi]             | rad    |
| 5     | psi_dot     | yaw rate                                        | rad/s  |
| 6     | beta        | slip angle at the centre of gravity             | rad    |

The reference point (x, y) is the centre of gravity when the dynamic
single-track model is active (|v| >= 0.5 m/s). Below that speed the model
switches to kinematic equations written for the rear axle; see
dynamic_models.vehicle_dynamics_st(). The collision rectangle and the LiDAR
offset are both measured from (x, y).

@section base_classes_step_order Order of operations in one simulation step
1. Each car's physics is integrated forward (RaceCar.update_pose), which also
   produces a raw LiDAR scan against the static map only.
2. Car-to-car collisions are checked with GJK on the cars' rectangular
   footprints (Simulator.check_collision).
3. Each car is told where the others now are; its scan is checked for an
   imminent wall collision (iTTC) and then the other cars are ray-cast into
   it (RaceCar.update_scan).
4. An observation dictionary is assembled and returned.

@author Hongrui Zheng
"""
from enum import Enum
import warnings

import numpy as np
from numba import njit

from f110_gym.envs.dynamic_models import vehicle_dynamics_st, pid
from f110_gym.envs.laser_models import ScanSimulator2D, check_ttc_jit, ray_cast
from f110_gym.envs.collision_models import get_vertices, collision_multiple


class Integrator(Enum):
    """!
    @brief Numerical integration scheme used to advance the vehicle dynamics.

    The dynamics model returns the time derivative of the state, and the
    integrator turns that into the next state over one physics time step.
    """

    ## Classic 4th-order Runge-Kutta: four derivative evaluations per step.
    #  More accurate, especially at high speed or with large time steps,
    #  at roughly 4x the cost of Euler.
    RK4 = 1

    ## Forward (explicit) Euler: state += dt * f(state). Cheapest, but less
    #  accurate and can drift or go unstable with large time steps.
    Euler = 2


class RaceCar(object):
    """!
    @brief A single simulated vehicle: physics, LiDAR, and wall-collision detection.

    A RaceCar owns its own state vector and control inputs, but shares a single
    LiDAR scan simulator (and the per-beam lookup tables derived from it) with
    every other RaceCar instance. Those shared objects are stored as class
    attributes and built only once, by whichever car is constructed first.

    A RaceCar only knows about collisions with the static map (walls).
    Car-to-car collisions are detected by Simulator.check_collision().

    @see Simulator for how multiple RaceCars are stepped together.
    """

    # ------------------------------------------------------------------
    # Class-level (shared) attributes.
    # Every RaceCar uses the same LiDAR configuration and map, so these are
    # computed once and shared instead of being duplicated per instance.
    # ------------------------------------------------------------------

    ## Shared ScanSimulator2D instance that ray-casts LiDAR beams against the map.
    #  Created by the first RaceCar constructed; None until then.
    scan_simulator = None

    ## Precomputed cos(angle) for every LiDAR beam, shape (num_beams,).
    #  Used by the iTTC check to project the car's velocity onto each beam.
    cosines = None

    ## Angle of every LiDAR beam relative to the car's heading, shape (num_beams,).
    #  Runs from -fov/2 (rightmost beam) to +fov/2 (leftmost beam).
    scan_angles = None

    ## Distance from the LiDAR origin to the edge of the car's own body along
    #  each beam, shape (num_beams,). Subtracted from the measured range in the
    #  iTTC check so the result is "distance from bumper to wall", not
    #  "distance from sensor to wall".
    side_distances = None

    def __init__(self, params, seed, is_ego=False, time_step=0.01, num_beams=1080, fov=4.7, integrator=Integrator.Euler, lidar_dist=0.0):
        """!
        @brief Construct a vehicle and, if this is the first vehicle, build the shared LiDAR model.

        @param params     Vehicle parameter dictionary. Required keys:
                          - @c mu        tyre-road friction coefficient
                          - @c C_Sf, @c C_Sr  front / rear cornering stiffness
                          - @c lf, @c lr distance from centre of gravity to front / rear axle (m)
                          - @c h         height of centre of gravity (m)
                          - @c m         mass (kg)
                          - @c I         yaw moment of inertia (kg m^2)
                          - @c s_min, @c s_max   steering angle limits (rad)
                          - @c sv_min, @c sv_max steering velocity limits (rad/s)
                          - @c v_switch  speed above which acceleration is power-limited (m/s)
                          - @c a_max     maximum longitudinal acceleration (m/s^2)
                          - @c v_min, @c v_max   speed limits (m/s)
                          - @c length, @c width  body footprint used for collisions (m)
        @param seed       Seed for the LiDAR noise random generator.
        @param is_ego     True if this is the ego (controlled / observed) vehicle.
        @param time_step  Physics time step in seconds.
        @param num_beams  Number of LiDAR beams per scan.
        @param fov        LiDAR field of view in radians (4.7 rad is about 270 degrees).
        @param integrator Integrator used to advance the dynamics (Integrator.Euler or Integrator.RK4).
        @param lidar_dist Forward distance from the vehicle reference point (state x, y) to the LiDAR, along the car's heading (m).

        @note The LiDAR settings (@p num_beams, @p fov) and the body geometry used
              for @ref side_distances are taken from whichever car is built first,
              because those objects are shared by every RaceCar.
        @note @c scan_rng is only created here for the first car built.
              reset() creates it for every car, so call reset() before
              update_pose(). F110Env already does this.
        """

        # initialization
        ## Vehicle parameter dictionary (see the constructor for its keys).
        self.params = params
        ## Seed for this car's LiDAR noise generator; reused on every reset().
        self.seed = seed
        ## True if this is the ego vehicle.
        self.is_ego = is_ego
        ## Physics time step in seconds.
        self.time_step = time_step
        ## Number of LiDAR beams per scan.
        self.num_beams = num_beams
        ## LiDAR field of view in radians.
        self.fov = fov
        ## Integration scheme used in update_pose().
        self.integrator = integrator
        ## Forward offset of the LiDAR from the vehicle reference point (state x, y), in metres.
        self.lidar_dist = lidar_dist
        if self.integrator is Integrator.RK4:
            warnings.warn(f"Chosen integrator is RK4. This is different from previous versions of the gym.")

        ## Full dynamic state:
        #  [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle].
        #  See the "Vehicle state vector" table in the file documentation.
        self.state = np.zeros((7, ))

        ## Poses [x, y, yaw] of every other car, shape (num_agents - 1, 3).
        #  Set each step by update_opp_poses() and used to ray-cast opponents into the scan.
        self.opp_poses = None

        # Control inputs. These hold the low-level commands to the dynamics model
        # (not the desired speed / steering angle given to update_pose()). They
        # are cleared on reset and when the car crashes.
        ## Longitudinal acceleration command (m/s^2).
        self.accel = 0.0
        ## Steering angle velocity command (rad/s).
        self.steer_angle_vel = 0.0

        ## FIFO buffer that delays steering commands to model servo lag.
        #  The newest command is at index 0 and the oldest at index -1.
        self.steer_buffer = np.empty((0, ))
        ## Number of steps of steering delay.
        self.steer_buffer_size = 2

        ## True if the most recent iTTC check found a collision with the environment.
        self.in_collision = False

        ## Instantaneous time-to-collision threshold in seconds. Any beam with an
        #  iTTC in [0, ttc_thresh) counts as a wall collision.
        self.ttc_thresh = 0.005

        # ------------------------------------------------------------------
        # One-time setup of the shared LiDAR model and per-beam lookup tables.
        # Only the first RaceCar constructed runs this block.
        # ------------------------------------------------------------------
        if RaceCar.scan_simulator is None:
            ## Random generator for LiDAR noise. Re-seeded on every reset() so
            #  each episode's noise sequence is reproducible.
            self.scan_rng = np.random.default_rng(seed=self.seed)
            RaceCar.scan_simulator = ScanSimulator2D(num_beams, fov)

            # Angular spacing between adjacent beams.
            scan_ang_incr = RaceCar.scan_simulator.get_increment()

            # angles of each scan beam, distance from lidar to edge of car at each beam, and precomputed cosines of each angle
            RaceCar.cosines = np.zeros((num_beams, ))
            RaceCar.scan_angles = np.zeros((num_beams, ))
            RaceCar.side_distances = np.zeros((num_beams, ))

            # The car body is approximated as a rectangle centred on the LiDAR:
            #   - half-width  (dist_sides) to the left/right sides
            #   - half-wheelbase (dist_fr) to the front/rear faces
            # Note dist_fr uses the wheelbase (lf + lr), not the body length.
            dist_sides = params['width']/2.
            dist_fr = (params['lf']+params['lr'])/2.

            # For each beam, find where it first exits the rectangle. A beam at
            # angle a travels dist_sides / |sin(a)| before reaching a side face
            # and dist_fr / |cos(a)| before reaching the front/rear face. Whichever
            # face it reaches first is where it leaves the body, so take the min.
            # The four branches below are the same formula written for each
            # quadrant so every denominator stays positive.
            for i in range(num_beams):
                angle = -fov/2. + i*scan_ang_incr
                RaceCar.scan_angles[i] = angle
                RaceCar.cosines[i] = np.cos(angle)

                if angle > 0:
                    if angle < np.pi/2:
                        # between 0 and pi/2 (front-left quadrant)
                        to_side = dist_sides / np.sin(angle)
                        to_fr = dist_fr / np.cos(angle)
                        RaceCar.side_distances[i] = min(to_side, to_fr)
                    else:
                        # between pi/2 and pi (rear-left quadrant)
                        to_side = dist_sides / np.cos(angle - np.pi/2.)
                        to_fr = dist_fr / np.sin(angle - np.pi/2.)
                        RaceCar.side_distances[i] = min(to_side, to_fr)
                else:
                    if angle > -np.pi/2:
                        # between 0 and -pi/2 (front-right quadrant)
                        to_side = dist_sides / np.sin(-angle)
                        to_fr = dist_fr / np.cos(-angle)
                        RaceCar.side_distances[i] = min(to_side, to_fr)
                    else:
                        # between -pi/2 and -pi (rear-right quadrant)
                        to_side = dist_sides / np.cos(-angle - np.pi/2)
                        to_fr = dist_fr / np.sin(-angle - np.pi/2)
                        RaceCar.side_distances[i] = min(to_side, to_fr)

    def update_params(self, params):
        """!
        @brief Replace the vehicle's physical parameters.

        The new values take effect on the next call to update_pose(). It is not
        needed at construction time, because the constructor already stores @p params.

        @note This does not rebuild @ref side_distances, which were computed from
              the first car's width and wheelbase at construction.

        @param params New vehicle parameter dictionary (same keys as the constructor).
        """
        self.params = params

    def set_map(self, map_path, map_ext):
        """!
        @brief Load the occupancy-grid map that the LiDAR ray-casts against.

        The scan simulator is shared, so this sets the map for every RaceCar.

        @param map_path Absolute path to the map's YAML metadata file.
        @param map_ext  File extension of the map image (e.g. ".png", ".pgm").
        """
        RaceCar.scan_simulator.set_map(map_path, map_ext)

    def reset(self, pose):
        """!
        @brief Put the vehicle at rest at a given pose and clear all transient state.

        Zeroes the whole state vector (speed, steering, yaw rate, slip), then sets
        position and heading from @p pose. It also clears the control inputs,
        collision flag and steering delay buffer, and re-seeds the LiDAR noise
        generator so each episode is reproducible.

        @param pose np.ndarray of shape (3,): [x, y, yaw] in world coordinates.
        """
        # clear control inputs
        self.accel = 0.0
        self.steer_angle_vel = 0.0
        # clear collision indicator
        self.in_collision = False
        # clear state, then place the car: pose [x, y, yaw] -> state[0], state[1], state[4]
        self.state = np.zeros((7, ))
        self.state[0:2] = pose[0:2]
        self.state[4] = pose[2]
        # empty the steering delay buffer so old commands don't carry over
        self.steer_buffer = np.empty((0, ))
        # reset scan random generator
        self.scan_rng = np.random.default_rng(seed=self.seed)

    def ray_cast_agents(self, scan):
        """!
        @brief Add the other vehicles to a LiDAR scan.

        The scan simulator only knows about the static map. This method treats
        each opponent as a rectangle (using this car's length and width) and,
        for every beam that hits it, shortens the beam's range to the hit distance.

        @param scan np.ndarray of shape (num_beams,): ranges measured against the map.
                    It is modified in place.
        @return np.ndarray of shape (num_beams,): the scan with opponents included.

        @note Rays are cast from the car's reference point (state[0:2]), not the
              LiDAR position offset by @ref lidar_dist that update_pose() uses
              for the map scan.
        """

        # starting from original scan
        new_scan = scan

        # loop over all opponent vehicle poses
        for opp_pose in self.opp_poses:
            # get the four corners of the opponent's body rectangle in world coordinates
            opp_vertices = get_vertices(opp_pose, self.params['length'], self.params['width'])

            # shorten any beam that hits one of the rectangle's edges
            new_scan = ray_cast(np.append(self.state[0:2], self.state[4]), new_scan, self.scan_angles, opp_vertices)

        return new_scan

    def check_ttc(self, current_scan):
        """!
        @brief Detect a collision with the environment using instantaneous time-to-collision (iTTC).

        For every beam i:

            iTTC_i = (range_i - side_distance_i) / (v * cos(angle_i))

        i.e. the gap between the car body and the obstacle, divided by how fast
        the car is closing on it along that beam. If any beam has
        0 <= iTTC < @ref ttc_thresh, the car counts as crashed into a wall.
        When the car is stationary (v == 0), no collision is reported.

        On a collision the car is stopped: control inputs are zeroed and so is
        state[3:].

        @warning state[3:] also includes the yaw angle (index 4), so a crash snaps
                 the car's heading to 0 as well as stopping it.
        @note This does NOT detect collisions with other vehicles; see
              Simulator.check_collision().

        @param current_scan np.ndarray of shape (num_beams,): LiDAR ranges to check.
        @return True if the car is in collision with the environment.
        """

        in_collision = check_ttc_jit(current_scan, self.state[3], self.scan_angles, self.cosines, self.side_distances, self.ttc_thresh)

        # if in collision stop vehicle
        if in_collision:
            self.state[3:] = 0.
            self.accel = 0.0
            self.steer_angle_vel = 0.0

        # update state
        self.in_collision = in_collision

        return in_collision

    def update_pose(self, raw_steer, vel):
        """!
        @brief Advance the vehicle's physics one time step and take a LiDAR scan.

        The steps are:
        1. **Steering delay:** @p raw_steer goes into a FIFO buffer, and the command
           from @ref steer_buffer_size steps ago comes out. Until the buffer is
           full, the applied steering is 0.
        2. **Low-level control:** pid() converts the desired steering angle and
           speed into a steering velocity and an acceleration. Steering is
           bang-bang at +/- sv_max until the target angle is reached. Acceleration
           is proportional to the speed error and capped at a_max.
        3. **Integration:** the single-track (bicycle) model with tyre slip,
           vehicle_dynamics_st(), gives the state derivative, which is integrated
           over one @ref time_step with Euler or RK4.
        4. **Yaw wrap:** heading is wrapped into [0, 2*pi].
        5. **LiDAR scan:** a noisy scan is ray-cast against the static map from
           the LiDAR position (@ref lidar_dist ahead of the reference point).

        @param raw_steer Desired front-wheel steering angle (rad).
        @param vel       Desired longitudinal speed (m/s).
        @return np.ndarray of shape (num_beams,): LiDAR scan against the static map
                only. Other vehicles are added later by update_scan().

        @exception SyntaxError If @ref integrator is not a supported Integrator value.
        """

        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]

        # ---- 1. steering delay ----
        # Push the new command onto the front of the buffer. Once the buffer is
        # full, pop the oldest command from the back and apply it. This delays
        # steering by steer_buffer_size steps.
        steer = 0.
        if self.steer_buffer.shape[0] < self.steer_buffer_size:
            # buffer still filling: apply no steering yet
            steer = 0.
            self.steer_buffer = np.append(raw_steer, self.steer_buffer)
        else:
            steer = self.steer_buffer[-1]
            self.steer_buffer = self.steer_buffer[:-1]
            self.steer_buffer = np.append(raw_steer, self.steer_buffer)


        # ---- 2. low-level control ----
        # Convert desired (speed, steering angle) into the model's actual inputs:
        # acceleration (accl) and steering angle velocity (sv).
        accl, sv = pid(vel, steer, self.state[3], self.state[2], self.params['sv_max'], self.params['a_max'], self.params['v_max'], self.params['v_min'])

        # ---- 3. integrate the dynamics ----
        if self.integrator is Integrator.RK4:
            # RK4 integration
            # Sample the derivative at the start (k1), twice at the midpoint
            # (k2, k3) and at the end (k4) of the step, then take a weighted
            # average. The control input is held constant over the step.
            k1 = vehicle_dynamics_st(
                self.state,
                np.array([sv, accl]),
                self.params['mu'],
                self.params['C_Sf'],
                self.params['C_Sr'],
                self.params['lf'],
                self.params['lr'],
                self.params['h'],
                self.params['m'],
                self.params['I'],
                self.params['s_min'],
                self.params['s_max'],
                self.params['sv_min'],
                self.params['sv_max'],
                self.params['v_switch'],
                self.params['a_max'],
                self.params['v_min'],
                self.params['v_max'])

            # midpoint estimate using the slope at the start
            k2_state = self.state + self.time_step*(k1/2)

            k2 = vehicle_dynamics_st(
                k2_state,
                np.array([sv, accl]),
                self.params['mu'],
                self.params['C_Sf'],
                self.params['C_Sr'],
                self.params['lf'],
                self.params['lr'],
                self.params['h'],
                self.params['m'],
                self.params['I'],
                self.params['s_min'],
                self.params['s_max'],
                self.params['sv_min'],
                self.params['sv_max'],
                self.params['v_switch'],
                self.params['a_max'],
                self.params['v_min'],
                self.params['v_max'])

            # second midpoint estimate using the slope at the first midpoint
            k3_state = self.state + self.time_step*(k2/2)

            k3 = vehicle_dynamics_st(
                k3_state,
                np.array([sv, accl]),
                self.params['mu'],
                self.params['C_Sf'],
                self.params['C_Sr'],
                self.params['lf'],
                self.params['lr'],
                self.params['h'],
                self.params['m'],
                self.params['I'],
                self.params['s_min'],
                self.params['s_max'],
                self.params['sv_min'],
                self.params['sv_max'],
                self.params['v_switch'],
                self.params['a_max'],
                self.params['v_min'],
                self.params['v_max'])

            # end-of-step estimate using the slope at the second midpoint
            k4_state = self.state + self.time_step*k3

            k4 = vehicle_dynamics_st(
                k4_state,
                np.array([sv, accl]),
                self.params['mu'],
                self.params['C_Sf'],
                self.params['C_Sr'],
                self.params['lf'],
                self.params['lr'],
                self.params['h'],
                self.params['m'],
                self.params['I'],
                self.params['s_min'],
                self.params['s_max'],
                self.params['sv_min'],
                self.params['sv_max'],
                self.params['v_switch'],
                self.params['a_max'],
                self.params['v_min'],
                self.params['v_max'])

            # dynamics integration: weighted average of the four slopes
            self.state = self.state + self.time_step*(1/6)*(k1 + 2*k2 + 2*k3 + k4)

        elif self.integrator is Integrator.Euler:
            # Forward Euler: one derivative evaluation at the start of the step
            f = vehicle_dynamics_st(
                self.state,
                np.array([sv, accl]),
                self.params['mu'],
                self.params['C_Sf'],
                self.params['C_Sr'],
                self.params['lf'],
                self.params['lr'],
                self.params['h'],
                self.params['m'],
                self.params['I'],
                self.params['s_min'],
                self.params['s_max'],
                self.params['sv_min'],
                self.params['sv_max'],
                self.params['v_switch'],
                self.params['a_max'],
                self.params['v_min'],
                self.params['v_max'])
            self.state = self.state + self.time_step * f

        else:
            raise SyntaxError(f"Invalid Integrator Specified. Provided {self.integrator.name}. Please choose RK4 or Euler")

        # ---- 4. bound yaw angle to [0, 2*pi] ----
        if self.state[4] > 2*np.pi:
            self.state[4] = self.state[4] - 2*np.pi
        elif self.state[4] < 0:
            self.state[4] = self.state[4] + 2*np.pi

        # ---- 5. update scan ----
        # The LiDAR sits lidar_dist ahead of the reference point along the heading,
        # so shift the scan origin forward before ray-casting.
        scan_x = self.state[0] + self.lidar_dist*np.cos(self.state[4])
        scan_y = self.state[1] + self.lidar_dist*np.sin(self.state[4])
        scan_pose = np.array([scan_x, scan_y, self.state[4]])
        current_scan = RaceCar.scan_simulator.scan(scan_pose, self.scan_rng)
        # current_scan = RaceCar.scan_simulator.scan(np.append(self.state[0:2],  self.state[4]), self.scan_rng)

        return current_scan

    def update_opp_poses(self, opp_poses):
        """!
        @brief Tell this vehicle where every other vehicle is.

        Called by Simulator.step() after all cars have moved, so ray_cast_agents()
        uses this step's poses.

        @param opp_poses np.ndarray of shape (num_agents - 1, 3): [x, y, yaw] of
                         every other vehicle (this car is excluded).
        """
        self.opp_poses = opp_poses


    def update_scan(self, agent_scans, agent_index):
        """!
        @brief Finish this vehicle's scan: check for a wall collision, then add the opponents.

        This is separate from update_pose() because ray-casting opponents needs
        the new poses of every car, so all cars must move before any scan is finished.

        The iTTC check runs on the map-only scan, before opponents are added.
        That way another car close by is not mistaken for a wall collision; car-to-car
        contact is handled separately by Simulator.check_collision().

        @param agent_scans List of every car's scan. Entry @p agent_index is replaced
                           with the finished scan.
        @param agent_index Index of this car in @p agent_scans.
        """

        current_scan = agent_scans[agent_index]

        # check iTTC against the environment (may stop the car)
        self.check_ttc(current_scan)

        # ray cast other agents to modify scan
        new_scan = self.ray_cast_agents(current_scan)

        agent_scans[agent_index] = new_scan


class Simulator(object):
    """!
    @brief Multi-vehicle world: owns every RaceCar and steps them together.

    The Simulator handles what a single RaceCar cannot do alone: stepping all
    cars together, detecting car-to-car collisions, and making cars visible in
    each other's LiDAR scans. F110Env wraps this class and adds lap counting,
    rendering and the gym interface on top.
    """

    def __init__(self, params, num_agents, seed, time_step=0.01, ego_idx=0, integrator=Integrator.RK4, lidar_dist=0.0):
        """!
        @brief Create the world and one RaceCar per agent.

        @param params     Vehicle parameter dictionary, shared by every car
                          (see RaceCar.__init__ for its keys).
        @param num_agents Number of vehicles in the environment.
        @param seed       Seed for the LiDAR noise generators.
        @param time_step  Physics time step in seconds.
        @param ego_idx    Index of the ego vehicle in @ref agents.
        @param integrator Integrator for every car. Note the default here is
                          Integrator.RK4, while RaceCar on its own defaults to Euler.
        @param lidar_dist Forward distance from the vehicle reference point to the LiDAR (m).
        """
        ## Number of vehicles in the environment.
        self.num_agents = num_agents
        ## Seed passed to every car's LiDAR noise generator.
        self.seed = seed
        ## Physics time step in seconds.
        self.time_step = time_step
        ## Index of the ego vehicle in @ref agents.
        self.ego_idx = ego_idx
        ## Vehicle parameter dictionary. Its length and width are used for collision footprints.
        self.params = params
        ## Latest [x, y, yaw] of every car, shape (num_agents, 3). Filled in by step().
        self.agent_poses = np.empty((self.num_agents, 3))
        ## List of RaceCar objects, one per agent.
        self.agents = []
        ## Collision flag per car, shape (num_agents,): 1.0 if the car hit a wall or another car this step.
        self.collisions = np.zeros((self.num_agents, ))
        ## Index of the car each car collided with, shape (num_agents,); -1 if none.
        #  Only car-to-car collisions are recorded here, not wall collisions.
        self.collision_idx = -1 * np.ones((self.num_agents, ))

        # initializing agents
        for i in range(self.num_agents):
            if i == ego_idx:
                ego_car = RaceCar(params, self.seed, is_ego=True, time_step=self.time_step, integrator=integrator, lidar_dist=lidar_dist)
                self.agents.append(ego_car)
            else:
                agent = RaceCar(params, self.seed, is_ego=False, time_step=self.time_step, integrator=integrator, lidar_dist=lidar_dist)
                self.agents.append(agent)

    def set_map(self, map_path, map_ext):
        """!
        @brief Load the map for the environment.

        Every car's set_map() is called, although they all share one scan
        simulator, so the map is effectively loaded once.

        @param map_path Path to the map's YAML metadata file.
        @param map_ext  File extension of the map image (e.g. ".png").
        """
        for agent in self.agents:
            agent.set_map(map_path, map_ext)


    def update_params(self, params, agent_idx=-1):
        """!
        @brief Update vehicle parameters for every car, or for one car.

        @param params    New parameter dictionary (see RaceCar.__init__ for its keys).
        @param agent_idx Index of the car to update. If negative (the default), every car is updated.

        @exception IndexError If @p agent_idx is >= num_agents.

        @note Only the cars' params are updated. Simulator.params, which
              check_collision() uses for footprint size, is not changed.
        """
        if agent_idx < 0:
            # update params for all
            for agent in self.agents:
                agent.update_params(params)
        elif agent_idx >= 0 and agent_idx < self.num_agents:
            # only update one agent's params
            self.agents[agent_idx].update_params(params)
        else:
            # index out of bounds, throw error
            raise IndexError('Index given is out of bounds for list of agents.')

    def check_collision(self):
        """!
        @brief Detect car-to-car collisions with the GJK algorithm.

        Each car is modelled as a length x width rectangle at its current pose.
        Every pair of rectangles is tested for overlap with GJK
        (Gilbert-Johnson-Keerthi), a fast test for intersecting convex shapes.

        Overwrites @ref collisions and @ref collision_idx. Wall collisions are
        added to @ref collisions afterwards in step().
        """
        # get the four corners of every car's body, shape (num_agents, 4, 2)
        all_vertices = np.empty((self.num_agents, 4, 2))
        for i in range(self.num_agents):
            all_vertices[i, :, :] = get_vertices(np.append(self.agents[i].state[0:2],self.agents[i].state[4]), self.params['length'], self.params['width'])
        # pairwise GJK test across all cars
        self.collisions, self.collision_idx = collision_multiple(all_vertices)


    def step(self, control_inputs):
        """!
        @brief Advance the whole world by one physics time step.

        The steps are:
        1. Move every car (RaceCar.update_pose) and record its new pose.
           Each car also takes a scan against the static map.
        2. Check car-to-car collisions on the new poses (check_collision).
        3. For each car: give it the other cars' poses, check for a wall
           collision, and add the other cars to its scan (RaceCar.update_scan).
           A wall collision also sets that car's entry in @ref collisions.
        4. Build and return the observation dictionary.

        @param control_inputs np.ndarray of shape (num_agents, 2). Row i is
                              [desired steering angle (rad), desired speed (m/s)] for car i.
        @return Observation dictionary with keys:
                - @c ego_idx       index of the ego vehicle
                - @c scans         list of LiDAR scans, each np.ndarray of shape (num_beams,)
                - @c poses_x       list of x positions (m)
                - @c poses_y       list of y positions (m)
                - @c poses_theta   list of yaw angles (rad)
                - @c linear_vels_x list of longitudinal speeds (m/s)
                - @c linear_vels_y list of lateral speeds. Always 0; this is not simulated.
                - @c ang_vels_z    list of yaw rates (rad/s)
                - @c collisions    np.ndarray of collision flags (1.0 = collided)
        """


        agent_scans = []

        # ---- 1. move every car ----
        for i, agent in enumerate(self.agents):
            # update each agent's pose
            current_scan = agent.update_pose(control_inputs[i, 0], control_inputs[i, 1])
            agent_scans.append(current_scan)

            # update sim's information of agent poses: [x, y, yaw]
            self.agent_poses[i, :] = np.append(agent.state[0:2], agent.state[4])

        # ---- 2. check collisions between all agents ----
        self.check_collision()

        # ---- 3. finish each car's scan ----
        for i, agent in enumerate(self.agents):
            # every pose except this car's own
            opp_poses = np.concatenate((self.agent_poses[0:i, :], self.agent_poses[i+1:, :]), axis=0)
            agent.update_opp_poses(opp_poses)

            # update each agent's current scan based on other agents
            agent.update_scan(agent_scans, i)

            # update agent collision with environment
            if agent.in_collision:
                self.collisions[i] = 1.

        # ---- 4. fill in observations ----
        # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
        # collision_angles is removed from observations
        observations = {'ego_idx': self.ego_idx,
            'scans': [],
            'poses_x': [],
            'poses_y': [],
            'poses_theta': [],
            'linear_vels_x': [],
            'linear_vels_y': [],
            'ang_vels_z': [],
            'collisions': self.collisions}
        for i, agent in enumerate(self.agents):
            observations['scans'].append(agent_scans[i])
            observations['poses_x'].append(agent.state[0])
            observations['poses_y'].append(agent.state[1])
            observations['poses_theta'].append(agent.state[4])
            observations['linear_vels_x'].append(agent.state[3])
            observations['linear_vels_y'].append(0.)
            observations['ang_vels_z'].append(agent.state[5])

        return observations

    def reset(self, poses):
        """!
        @brief Reset every car to a starting pose.

        @param poses np.ndarray of shape (num_agents, 3). Row i is [x, y, yaw] for car i.

        @exception ValueError If the number of rows in @p poses is not num_agents.

        @note @ref collisions and @ref collision_idx are not cleared here. They
              are overwritten by the next step().
        """

        if poses.shape[0] != self.num_agents:
            raise ValueError('Number of poses for reset does not match number of agents.')

        # loop over poses to reset
        for i in range(self.num_agents):
            self.agents[i].reset(poses[i, :])
