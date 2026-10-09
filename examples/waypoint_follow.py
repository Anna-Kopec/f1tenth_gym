"""!
@file waypoint_follow.py
@brief Example: a pure pursuit controller following a raceline in the F1TENTH gym.

Loads config_example_map.yaml, creates a single-car environment on the example
map, and drives one episode with PurePursuitPlanner, rendering as it goes. The
camera follows the car and the waypoints are drawn on the map.

Run it from the @c examples folder, because the config and waypoint paths are relative:
@code
cd examples
python waypoint_follow.py
@endcode

@section waypoint_follow_pure_pursuit How pure pursuit works
Each step, the controller:
1. finds the point on the raceline nearest the car;
2. walks forward along the raceline to the first point that is the *lookahead
   distance* L away from the car, and aims for it;
3. fits the circular arc that leaves the car along its heading and passes through
   that point. If the point is a lateral (sideways) distance y from the car's
   heading line, the arc has radius R = L^2 / (2y);
4. steers with the bicycle-model angle for that arc: delta = atan(wheelbase / R).
The speed command is the raceline's speed at the nearest point, times a gain.

A longer lookahead gives smoother but wider lines; a shorter one tracks more
tightly but can oscillate.

@section waypoint_follow_config Config file fields used
| Field        | Meaning |
|--------------|---------|
| map_path     | Map path without extension (passed as the env's @c map kwarg). |
| map_ext      | Map image extension, including the dot. |
| sx, sy, stheta | Starting pose [x, y, yaw]. |
| wpt_path     | Raceline CSV file. |
| wpt_delim    | CSV delimiter. |
| wpt_rowskip  | Number of header rows to skip. |
| wpt_xind, wpt_yind | Columns holding x and y (m). |
| wpt_vind     | Column holding the target speed (m/s). |

The other fields in config_example_map.yaml (parameter bounds, popsize,
optim_method, ...) belong to a parameter-tuning setup and are not used here.
"""
import time
from f110_gym.envs.base_classes import Integrator
import yaml
import gym
import numpy as np
from argparse import Namespace

from numba import njit

from pyglet.gl import GL_POINTS

# ----------------------------------------------------------------------------
# Planner Helpers
# ----------------------------------------------------------------------------
@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    """!
    @brief Nearest point to @p point on a piecewise-linear trajectory.

    Projects the point onto every segment (clamped to the segment's ends) and
    returns the closest projection. It is fast: about 0.2 ms for 1000 waypoints.

    @param point      np.ndarray of shape (2,): query point [x, y].
    @param trajectory np.ndarray of shape (N, 2): waypoints [x, y]. Consecutive
                      waypoints must differ, or a zero-length segment causes a division by zero.
    @return Tuple (projection, distance, t, segment_index):
            - @c projection: np.ndarray of shape (2,), the nearest point on the trajectory.
            - @c distance: distance from @p point to it.
            - @c t: position along that segment, from 0 (start) to 1 (end).
            - @c segment_index: index i of the segment, which runs from waypoint i to waypoint i+1.
    """
    diffs = trajectory[1:,:] - trajectory[:-1,:]
    l2s   = diffs[:,0]**2 + diffs[:,1]**2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0]-1, ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    # fraction along each segment of the projection, clamped to the segment
    t = dots / l2s
    t[t<0.0] = 0.0
    t[t>1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1,:] + (t*diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    """!
    @brief First point along the trajectory, searching forward from @p t, that is exactly @p radius from @p point.

    Each segment start + s * V is tested against the circle
    |start + s * V - point| = radius. That is a quadratic in s, and a root with
    s in [0, 1] is a crossing on the segment. On the starting segment, only
    crossings at or after the starting fraction count.

    Algorithm: http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm

    @param point      np.ndarray of shape (2,): circle centre (the car's position).
    @param radius     Circle radius (the lookahead distance) (m).
    @param trajectory np.ndarray of shape (N, 2): waypoints [x, y].
    @param t          Where to start searching: segment index plus fraction along it
                      (e.g. 12.3 = 30% along segment 12), as given by nearest_point_on_trajectory().
    @param wrap       If True and nothing is found before the end, continue from the
                      start of the trajectory, treating it as a closed loop.
    @return Tuple (point, segment_index, fraction), or (None, None, None) if there is no crossing.

    @note The search assumes the starting segment passes within @p radius of @p point.
    @note A tiny 1e-6 is added to each segment's end point to avoid zero-length segments.
    """
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0]-1):
        start = trajectory[i,:]
        end = trajectory[i+1,:]+1e-6
        V = np.ascontiguousarray(end - start)

        # quadratic a*s^2 + b*s + c = 0 for where the segment meets the circle
        a = np.dot(V,V)
        b = 2.0*np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
        discriminant = b*b-4*a*c

        if discriminant < 0:
            # segment's line misses the circle
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0*a)
        t2 = (-b + discriminant) / (2.0*a)
        if i == start_i:
            # on the first segment, only accept crossings at or after the starting fraction
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    # (i = -1 is the closing segment from the last waypoint back to the first)
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0],:]
            end = trajectory[(i+1) % trajectory.shape[0],:]+1e-6
            V = end - start

            a = np.dot(V,V)
            b = 2.0*np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
            discriminant = b*b-4*a*c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0*a)
            t2 = (-b + discriminant) / (2.0*a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t

@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    """!
    @brief Pure pursuit control law: speed and steering angle toward a lookahead point.

    Computes the lookahead point's lateral offset y in the car's frame, the arc
    radius R = L^2 / (2y), and the steering angle atan(wheelbase / R).

    @param pose_theta         Car yaw (rad).
    @param lookahead_point    np.ndarray of shape (3,): [x, y, target speed] of the point to aim for.
    @param position           np.ndarray of shape (2,): car position [x, y].
    @param lookahead_distance Lookahead distance L (m), used as the distance to the point in the formula.
    @param wheelbase          Distance between the axles (m).
    @return Tuple (speed, steering_angle): target speed (m/s) and steering angle (rad, positive = left).
            If the point is straight ahead (|y| < 1e-6), the steering angle is 0.
    """
    # lateral offset of the lookahead point in the car's frame (positive = to the left)
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return speed, steering_angle

class PurePursuitPlanner:
    """!
    @brief Example raceline follower using pure pursuit (see the file overview).
    """
    def __init__(self, conf, wb):
        """!
        @brief Load the raceline from the config and set up the planner.

        @param conf Config namespace (see the "Config file fields used" table).
        @param wb   Wheelbase of the car (m).
        """
        ## Wheelbase (m), used in the steering law.
        self.wheelbase = wb
        ## Config namespace (waypoint file and column indices).
        self.conf = conf
        self.load_waypoints(conf)
        ## If the car is further than this from the raceline (m), give up and use the fallback command.
        self.max_reacquire = 20.

        ## pyglet vertex lists for the drawn waypoints (created on the first render).
        self.drawn_waypoints = []

    def load_waypoints(self, conf):
        """!
        @brief Read the raceline CSV into @c self.waypoints (one row per waypoint, all columns kept).

        @param conf Config namespace with @c wpt_path, @c wpt_delim and @c wpt_rowskip.
        """
        ## Raceline array, shape (N, num_columns). Columns are picked with conf.wpt_xind etc.
        self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)

    def render_waypoints(self, e):
        """!
        @brief Draw the raceline points on the renderer (call from a render callback).

        The points are created on the first call and moved on later calls.
        Coordinates are scaled by 50 to match EnvRenderer's drawing units.

        @param e The EnvRenderer.
        """

        #points = self.waypoints

        points = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T

        scaled_points = 50.*points

        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]

    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
        """!
        @brief Choose the point to aim for.

        - If the car is within @p lookahead_distance of the raceline: the raceline
          point one lookahead distance ahead (see the note).
        - If it is further away but within @c max_reacquire (20 m): the nearest
          raceline segment's start point, to steer back onto the line.
        - Otherwise: None.

        @param waypoints          Full raceline array (all columns), used for the speed column.
        @param lookahead_distance Lookahead distance (m).
        @param position           np.ndarray of shape (2,): car position.
        @param theta              Car yaw (unused).
        @return np.ndarray of shape (3,): [x, y, target speed], or None.

        @note The point returned is the *start* waypoint of the segment where the
              lookahead circle crosses, not the exact crossing point. The speed is
              taken from the waypoint nearest the car.
        """
        wpts = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i+t, wrap=True)
            if i2 == None:
                return None
            current_waypoint = np.empty((3, ))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = waypoints[i, self.conf.wpt_vind]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], waypoints[i, self.conf.wpt_vind])
        else:
            return None

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
        """!
        @brief Compute the control command for the car's current pose.

        @param pose_x, pose_y     Car position (m).
        @param pose_theta         Car yaw (rad).
        @param lookahead_distance Lookahead distance (m).
        @param vgain              Multiplier applied to the raceline's speed.
        @return Tuple (speed, steering_angle). Note the order is the reverse of
                the env's action [steer, speed]. If no target point is found,
                returns (4.0, 0.0): 4 m/s straight ahead.
        """
        position = np.array([pose_x, pose_y])
        lookahead_point = self._get_current_waypoint(self.waypoints, lookahead_distance, position, pose_theta)

        if lookahead_point is None:
            return 4.0, 0.0

        speed, steering_angle = get_actuation(pose_theta, lookahead_point, position, lookahead_distance, self.wheelbase)
        speed = vgain * speed

        return speed, steering_angle


class FlippyPlanner:
    """!
    @brief Test planner that flips its steering left and right to stress the integrator and dynamics.

    For testing only. It was used to expose numerical errors when the dynamic
    (slip) model is active at low speed.

    @note @c counter is never incremented, so the steering flips on *every*
          call regardless of @p flip_every. The default steer of 2 rad is far beyond
          the car's steering limit, so the steering simply swings between its limits.
    """
    def __init__(self, speed=1, flip_every=1, steer=2):
        """!
        @brief Set the constant speed and steering magnitude.

        @param speed      Speed command (m/s).
        @param flip_every Intended number of calls between flips (see the class note).
        @param steer      Steering magnitude (rad).
        """
        ## Constant speed command (m/s).
        self.speed = speed
        ## Intended number of calls between flips.
        self.flip_every = flip_every
        ## Call counter (never incremented).
        self.counter = 0
        ## Current steering command (rad); its sign flips.
        self.steer = steer

    def render_waypoints(self, *args, **kwargs):
        """!
        @brief No-op, so this planner can be swapped in for PurePursuitPlanner.
        """
        pass

    def plan(self, *args, **kwargs):
        """!
        @brief Flip the steering sign and return (speed, steer). All arguments are ignored.

        @return Tuple (speed, steer).
        """
        if self.counter%self.flip_every == 0:
            self.counter = 0
            self.steer *= -1
        return self.speed, self.steer


def main():
    """!
    @brief Run one episode of pure pursuit on the example map, rendering in real time.

    Steps:
    1. Load config_example_map.yaml.
    2. Build the planner with the default F1TENTH wheelbase (lf + lr = 0.15875 + 0.17145 m).
    3. Create a 1-car env with the RK4 integrator and register a render callback.
       The callback makes the camera follow the car and draws the raceline.
    4. Reset at the configured start pose, then step until done: the car crashes
       or completes 2 laps.
    5. Print simulated time against real time.

    The @c work dict holds tuned controller values: @c tlad is the lookahead
    distance (m) and @c vgain the speed gain. @c mass and @c lf are left over
    from the tuning run and are not used.
    """

    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 1.375}#0.90338203837889}

    with open('config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    planner = PurePursuitPlanner(conf, (0.17145+0.15875)) #FlippyPlanner(speed=0.2, flip_every=1, steer=10)

    def render_callback(env_renderer):
        """!
        @brief Render callback: centre the camera on car 0 (16 m margin on each side), move the label, and draw the waypoints.

        Because it resets the view every frame, mouse panning and zooming have no lasting effect.

        @param env_renderer The EnvRenderer.
        """
        # custom extra drawing function

        e = env_renderer

        # update camera to follow car
        # car 0's corner coordinates in drawing units (metres * 50)
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700
        # 800 drawing units = 16 m of margin around the car
        e.left = left - 800
        e.right = right + 800
        e.top = top + 800
        e.bottom = bottom - 800

        planner.render_waypoints(env_renderer)

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1, timestep=0.01, integrator=Integrator.RK4)
    env.add_render_callback(render_callback)

    obs, step_reward, done, info = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
    env.render()

    # sum of rewards = simulated time, since each step's reward is the time step
    laptime = 0.0
    start = time.time()

    while not done:
        speed, steer = planner.plan(obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0], work['tlad'], work['vgain'])
        # env action order is [steer, speed]
        obs, step_reward, done, info = env.step(np.array([[steer, speed]]))
        laptime += step_reward
        env.render(mode='human')

    print('Sim elapsed time:', laptime, 'Real elapsed time:', time.time()-start)

if __name__ == '__main__':
    main()
