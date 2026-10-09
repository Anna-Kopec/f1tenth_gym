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
@file laser_models.py
@brief 2D LiDAR simulation: ray tracing against an occupancy-grid map, adding
       other vehicles to a scan, and time-to-collision checks.

@section laser_models_map_raytrace How map scans are simulated
The map is a black-and-white image (black = obstacle) plus a YAML file giving its
resolution (metres per pixel) and the world pose of its bottom-left corner.

1. **Distance transform (once, when the map loads):** get_dt() computes, for
   every free cell, the distance in metres to the nearest obstacle cell.
2. **Sphere tracing (every beam, every scan):** trace_ray() starts at the sensor
   and looks up the distance to the nearest obstacle, d. No obstacle is closer
   than d in *any* direction, so the ray can safely jump d metres forward. It
   repeats until d is below a small tolerance @c eps (the ray has hit something)
   or the total length passes @c max_range. Rays in open space take a few big
   steps; rays close to walls take many small steps.

To avoid computing sin/cos for every beam, beam directions come from a lookup
table that splits the full circle into @c theta_dis (2000 by default) discrete angles.

@section laser_models_agent_raycast How other vehicles are added
The map scan only contains static walls. ray_cast() then models each opponent as
a 4-sided polygon and intersects every beam that could hit it with its four edges
(get_range()). Any beam that hits an opponent before its current range is shortened.

@section laser_models_ttc Time-to-collision
check_ttc_jit() decides whether the car has hit a wall by computing an
instantaneous time-to-collision (iTTC) along every beam.

@author Hongrui Zheng
"""

import numpy as np
from numba import njit
from scipy.ndimage import distance_transform_edt as edt
from PIL import Image
import os
import yaml

import unittest
import timeit

def get_dt(bitmap, resolution):
    """!
    @brief Distance transform of the map: distance (m) from each cell to the nearest obstacle.

    scipy's Euclidean distance transform gives, for every non-zero (free) cell,
    the distance in cells to the nearest zero (obstacle) cell. Multiplying by the
    resolution converts that to metres. Obstacle cells get 0.

    Uses scipy.ndimage, so it cannot be compiled with numba. It runs only once per map.

    @param bitmap     np.ndarray of shape (rows, cols): binary map, 0 = obstacle and
                      anything > 0 (normally 255) = free space.
    @param resolution Map resolution (metres per cell).
    @return np.ndarray of shape (rows, cols): distance to the nearest obstacle in metres.
    """
    dt = resolution * edt(bitmap)
    return dt

@njit(cache=True)
def xy_2_rc(x, y, orig_x, orig_y, orig_c, orig_s, height, width, resolution):
    """!
    @brief Convert a world (x, y) point into a (row, column) cell index of the map grid.

    The point is shifted by the map origin, rotated into the map frame (which can
    have a yaw), and then divided by the resolution.

    @param x          World x coordinate (m).
    @param y          World y coordinate (m).
    @param orig_x     World x of the map origin (bottom-left cell) (m).
    @param orig_y     World y of the map origin (m).
    @param orig_c     cos(map origin yaw).
    @param orig_s     sin(map origin yaw).
    @param height     Map height in cells (rows).
    @param width      Map width in cells (columns).
    @param resolution Map resolution (metres per cell).
    @return Tuple (r, c): row and column of the cell, or (-1, -1) if the point is outside the map.

    @warning A (-1, -1) result is used directly as an array index by
             distance_transform(). In numpy, index -1 means the last row/column,
             so points outside the map read the distance value of the map's
             top-right cell instead of raising an error.
    """
    # translation
    x_trans = x - orig_x
    y_trans = y - orig_y

    # rotation (world frame -> map frame)
    x_rot = x_trans * orig_c + y_trans * orig_s
    y_rot = -x_trans * orig_s + y_trans * orig_c

    # clip the state to be a cell
    if x_rot < 0 or x_rot >= width * resolution or y_rot < 0 or y_rot >= height * resolution:
        c = -1
        r = -1
    else:
        c = int(x_rot/resolution)
        r = int(y_rot/resolution)

    return r, c

@njit(cache=True)
def distance_transform(x, y, orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt):
    """!
    @brief Distance from a world point to the nearest obstacle, read from the precomputed distance transform.

    @param x, y                   World coordinates of the query point (m).
    @param orig_x, orig_y         World position of the map origin (m).
    @param orig_c, orig_s         cos / sin of the map origin yaw.
    @param height, width          Map size in cells.
    @param resolution             Map resolution (metres per cell).
    @param dt                     Distance transform from get_dt().
    @return Distance to the nearest obstacle (m).

    @see xy_2_rc() for what happens to points outside the map.
    """
    r, c = xy_2_rc(x, y, orig_x, orig_y, orig_c, orig_s, height, width, resolution)
    distance = dt[r, c]
    return distance

@njit(cache=True)
def trace_ray(x, y, theta_index, sines, cosines, eps, orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt, max_range):
    """!
    @brief Length of one LiDAR beam, found by sphere tracing through the distance transform.

    The loop is pure arithmetic, so it is compiled with numba. See the file
    overview for how sphere tracing works.

    @param x, y         World position of the sensor (m).
    @param theta_index  Index of this beam's direction in the @p sines / @p cosines
                        lookup table. It is rounded down to an integer, so beam
                        directions are snapped to the table's discrete angles.
    @param sines        np.ndarray of shape (theta_dis,): lookup table of sin(angle).
    @param cosines      np.ndarray of shape (theta_dis,): lookup table of cos(angle).
    @param eps          Stop when the distance to the nearest obstacle drops below this (m).
    @param orig_x, orig_y, orig_c, orig_s, height, width, resolution
                        Map geometry (see xy_2_rc()).
    @param dt           Distance transform from get_dt().
    @param max_range    Maximum LiDAR range (m). Longer rays are clipped to this.
    @return Distance along the beam to the first obstacle (m), at most @p max_range.
    """

    # int casting, and index precal trigs
    theta_index_ = int(theta_index)
    s = sines[theta_index_]
    c = cosines[theta_index_]

    # distance to nearest initialization
    dist_to_nearest = distance_transform(x, y, orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt)
    total_dist = dist_to_nearest

    # ray tracing iterations
    while dist_to_nearest > eps and total_dist <= max_range:
        # move in the direction of the ray by dist_to_nearest
        # (guaranteed not to pass through an obstacle)
        x += dist_to_nearest * c
        y += dist_to_nearest * s

        # update dist_to_nearest for current point on ray
        # also keeps track of total ray length
        dist_to_nearest = distance_transform(x, y, orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt)
        total_dist += dist_to_nearest

    if total_dist > max_range:
        total_dist = max_range

    return total_dist

@njit(cache=True)
def get_scan(pose, theta_dis, fov, num_beams, theta_index_increment, sines, cosines, eps, orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt, max_range):
    """!
    @brief Simulate a full LiDAR scan by tracing every beam from one pose.

    Beams sweep counter-clockwise from (heading - fov/2) to (heading + fov/2).
    Each beam's direction is tracked as an index into the angle lookup table,
    wrapped into [0, theta_dis).

    @param pose                  np.ndarray of shape (3,): [x, y, yaw] of the sensor in the world.
    @param theta_dis             Number of discrete angles in the lookup table (covering 0 to 2*pi).
    @param fov                   Field of view (rad).
    @param num_beams             Number of beams in the scan.
    @param theta_index_increment Lookup-table index step between neighbouring beams
                                 (beam spacing in radians * theta_dis / 2pi).
    @param sines, cosines        Angle lookup tables of shape (theta_dis,).
    @param eps, orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt, max_range
                                 Passed through to trace_ray().
    @return np.ndarray of shape (num_beams,): range of each beam (m), from the rightmost beam to the leftmost.
    """
    # empty scan array init
    scan = np.empty((num_beams,))

    # convert the first beam's world angle (yaw - fov/2) into a lookup-table index
    theta_index = theta_dis * (pose[2] - fov/2.)/(2. * np.pi)

    # make sure it's wrapped properly into [0, theta_dis)
    theta_index = np.fmod(theta_index, theta_dis)
    while (theta_index < 0):
        theta_index += theta_dis

    # sweep through each beam
    for i in range(0, num_beams):
        # trace the current beam
        scan[i] = trace_ray(pose[0], pose[1], theta_index, sines, cosines, eps, orig_x, orig_y, orig_c, orig_s, height, width, resolution, dt, max_range)

        # increment the beam index
        theta_index += theta_index_increment

        # make sure it stays in the range [0, theta_dis)
        while theta_index >= theta_dis:
            theta_index -= theta_dis

    return scan

@njit(cache=True, error_model='numpy')
def check_ttc_jit(scan, vel, scan_angles, cosines, side_distances, ttc_thresh):
    """!
    @brief Decide whether the vehicle has hit the environment, using instantaneous time-to-collision (iTTC).

    For each beam i:

        closing speed  = vel * cos(angle_i)                (the car's speed along that beam)
        iTTC_i         = (scan_i - side_distance_i) / closing speed

    The numerator is the gap between the car body and the obstacle, not the
    sensor-to-obstacle distance. A beam with 0 <= iTTC < @p ttc_thresh means
    contact is imminent, so the car counts as crashed. Beams pointing away from
    the direction of travel have a negative closing speed and so a negative
    iTTC, which is ignored. A stationary car (vel == 0) never collides.

    @c error_model='numpy' makes division by zero return inf or nan instead of
    raising. This happens for beams at exactly +/-90 degrees, where cos = 0,
    and those beams are then ignored.

    @param scan           np.ndarray of shape (num_beams,): current scan ranges (m).
    @param vel            Current longitudinal velocity (m/s).
    @param scan_angles    np.ndarray of shape (num_beams,): beam angles relative to the heading (unused).
    @param cosines        np.ndarray of shape (num_beams,): precomputed cos(scan_angles).
    @param side_distances np.ndarray of shape (num_beams,): distance from the sensor to the car's
                          own body edge along each beam (m).
    @param ttc_thresh     iTTC threshold below which a collision is declared (s).
    @return True if the vehicle is in collision with the environment.
    """
    in_collision = False
    if vel != 0.0:
        num_beams = scan.shape[0]
        for i in range(num_beams):
            proj_vel = vel*cosines[i]
            ttc = (scan[i] - side_distances[i])/proj_vel
            if (ttc < ttc_thresh) and (ttc >= 0.0):
                in_collision = True
                break
    else:
        in_collision = False

    return in_collision

@njit(cache=True)
def cross(v1, v2):
    """!
    @brief 2D cross product (the z component of the 3D cross product): v1.x * v2.y - v1.y * v2.x.

    Positive if @p v2 is counter-clockwise from @p v1, negative if clockwise, and
    0 if they are parallel.

    @param v1 np.ndarray of shape (2,).
    @param v2 np.ndarray of shape (2,).
    @return Scalar cross product.
    """
    return v1[0]*v2[1]-v1[1]*v2[0]

@njit(cache=True)
def are_collinear(pt_a, pt_b, pt_c):
    """!
    @brief Test whether three 2D points lie on one line (within a tolerance of 1e-8).

    @param pt_a, pt_b, pt_c np.ndarray of shape (2,): the points to test.
    @return True if the points are collinear.
    """
    tol = 1e-8
    ba = pt_b - pt_a
    ca = pt_a - pt_c
    # zero cross product means the two vectors are parallel, so all three points are on one line
    col = np.fabs(cross(ba, ca)) < tol
    return col

@njit(cache=True)
def get_range(pose, beam_theta, va, vb):
    """!
    @brief Distance along one beam to a line segment (one edge of another vehicle).

    Finds where the ray o + d1 * dir meets the segment va + d2 * (vb - va):
    - @c d1 is the distance along the beam. It must be >= 0, so the hit is in front of the sensor.
    - @c d2 is the position along the edge. It must be in [0, 1], so the hit is between the two corners.

    If the beam is parallel to the edge, it only counts as a hit when the beam
    lies exactly along the edge's line, in which case the nearer corner is returned.

    @param pose       np.ndarray of shape (3,): [x, y, yaw] of the scanning sensor. Only x and y are used.
    @param beam_theta World-frame angle of the beam (rad).
    @param va, vb     np.ndarray of shape (2,): the two corners of the edge.
    @return Distance from the sensor to the edge along the beam (m), or inf if the beam misses.
    """
    o = pose[0:2]
    v1 = o - va
    v2 = vb - va
    # beam direction rotated by +90 degrees (a normal to the beam)
    v3 = np.array([np.cos(beam_theta + np.pi/2.), np.sin(beam_theta + np.pi/2.)])

    denom = v2.dot(v3)
    distance = np.inf

    if np.fabs(denom) > 0.0:
        # d1: distance along the beam, d2: fraction along the edge
        d1 = cross(v2, v1) / denom
        d2 = v1.dot(v3) / denom
        if d1 >= 0.0 and d2 >= 0.0 and d2 <= 1.0:
            distance = d1
    elif are_collinear(o, va, vb):
        # beam runs along the edge: report the nearer corner
        da = np.linalg.norm(va - o)
        db = np.linalg.norm(vb - o)
        distance = min(da, db)

    return distance

@njit(cache=True)
def get_blocked_view_indices(pose, vertices, scan_angles):
    """!
    @brief Range of beam indices that could hit another vehicle.

    Computes the bearing of each of the opponent's four corners relative to the
    scanner's heading, finds the beam nearest each bearing, and returns the
    smallest and largest of those beam indices. Only beams in this range need to
    be tested against the opponent, which saves a lot of work.

    @param pose        np.ndarray of shape (3,): [x, y, yaw] of the scanning vehicle.
    @param vertices    np.ndarray of shape (4, 2): the opponent's corners in the world frame.
    @param scan_angles np.ndarray of shape (num_beams,): beam angles relative to the heading.
    @return Tuple (min_ind, max_ind): first and last beam index (inclusive) to test.

    @note Corners outside the field of view snap to the nearest end beam. If an
          opponent sits directly behind the scanner, where bearings wrap around
          +/-pi, the min/max range may cover the wrong span. That region is
          outside a 270-degree field of view anyway.
    """
    # find four vectors formed by pose and 4 vertices:
    vecs = vertices - pose[:2]
    vec_sq = np.square(vecs)
    norms = np.sqrt(vec_sq[:, 0] + vec_sq[:, 1])
    unit_vecs = vecs / norms.reshape(norms.shape[0], 1)

    # find angles between all four and pose vector
    ego_x_vec = np.array([[np.cos(pose[2])], [np.sin(pose[2])]])

    angles_with_x = np.empty((4, ))
    for i in range(4):
        # heading angle minus corner bearing, wrapped to [-pi, pi]
        angle = np.arctan2(ego_x_vec[1], ego_x_vec[0]) - np.arctan2(unit_vecs[i, 1], unit_vecs[i, 0])
        if angle > np.pi:
            angle = angle - 2*np.pi
        elif angle < -np.pi:
            angle = angle + 2*np.pi
        # negate to get the corner's bearing relative to the heading (counter-clockwise positive)
        angles_with_x[i] = -angle[0]

    # nearest beam to each corner's bearing
    ind1 = int(np.argmin(np.abs(scan_angles - angles_with_x[0])))
    ind2 = int(np.argmin(np.abs(scan_angles - angles_with_x[1])))
    ind3 = int(np.argmin(np.abs(scan_angles - angles_with_x[2])))
    ind4 = int(np.argmin(np.abs(scan_angles - angles_with_x[3])))
    inds = [ind1, ind2, ind3, ind4]
    return min(inds), max(inds)


@njit(cache=True)
def ray_cast(pose, scan, scan_angles, vertices):
    """!
    @brief Add one other vehicle to a scan by intersecting beams with its four edges.

    For every beam that could hit the opponent (see get_blocked_view_indices()),
    the beam is tested against all four edges. If any edge is closer than the
    beam's current range, the range is shortened to that distance.

    @param pose        np.ndarray of shape (3,): [x, y, yaw] of the scanning vehicle.
    @param scan        np.ndarray of shape (num_beams,): scan to modify. Modified in place.
    @param scan_angles np.ndarray of shape (num_beams,): beam angles relative to the heading.
    @param vertices    np.ndarray of shape (4, 2): the opponent's corners, in order around
                       the rectangle (as returned by collision_models.get_vertices()).
    @return np.ndarray of shape (num_beams,): the modified scan (the same object as @p scan).
    """
    # pad vertices so loops around: edge j runs from looped_vertices[j] to looped_vertices[j+1]
    looped_vertices = np.empty((5, 2))
    looped_vertices[0:4, :] = vertices
    looped_vertices[4, :] = vertices[0, :]

    min_ind, max_ind = get_blocked_view_indices(pose, vertices, scan_angles)
    # looping over beams
    for i in range(min_ind, max_ind + 1):
        # looping over vertices
        for j in range(4):
            # check if original scan is longer than ray casted distance
            scan_range = get_range(pose, pose[2]+scan_angles[i], looped_vertices[j,:], looped_vertices[j+1,:])
            if scan_range < scan[i]:
                scan[i] = scan_range
    return scan

class ScanSimulator2D(object):
    """!
    @brief 2D LiDAR simulator: holds a map and its distance transform, and produces noisy scans.

    Usage:
    @code
    sim = ScanSimulator2D(num_beams=1080, fov=4.7)
    sim.set_map('maps/vegas.yaml', '.png')
    ranges = sim.scan(np.array([x, y, yaw]), np.random.default_rng(0))
    @endcode
    """

    def __init__(self, num_beams, fov, eps=0.0001, theta_dis=2000, max_range=30.0):
        """!
        @brief Configure the LiDAR and build the angle lookup tables. No map is loaded yet; call set_map().

        @param num_beams Number of beams per scan.
        @param fov       Field of view (rad). Beams are evenly spaced from -fov/2 to +fov/2.
        @param eps       Sphere-tracing stop tolerance (m): a ray counts as hitting
                         something once it is this close to an obstacle.
        @param theta_dis Number of discrete angles in the 0 to 2*pi lookup table.
                         A larger value gives more precise beam directions.
        @param max_range Maximum LiDAR range (m).

        @note The table is built with np.linspace(0, 2*pi, theta_dis), which includes
              both endpoints, so its spacing is 2*pi/(theta_dis-1). Indices are
              computed as if the spacing were 2*pi/theta_dis. The resulting angle
              error is at most one table step (about 0.18 degrees at the default).
        """
        # initialization
        ## Number of beams per scan.
        self.num_beams = num_beams
        ## Field of view (rad).
        self.fov = fov
        ## Sphere-tracing stop tolerance (m).
        self.eps = eps
        ## Number of discrete angles in the lookup table.
        self.theta_dis = theta_dis
        ## Maximum range (m).
        self.max_range = max_range
        ## Angle between neighbouring beams (rad).
        self.angle_increment = self.fov / (self.num_beams - 1)
        ## Angle between neighbouring beams, in lookup-table index units.
        self.theta_index_increment = theta_dis * self.angle_increment / (2. * np.pi)
        ## cos(map origin yaw). Set by set_map().
        self.orig_c = None
        ## sin(map origin yaw). Set by set_map().
        self.orig_s = None
        ## World x of the map origin (m). Set by set_map().
        self.orig_x = None
        ## World y of the map origin (m). Set by set_map().
        self.orig_y = None
        ## Map height in cells. None until a map is loaded.
        self.map_height = None
        ## Map width in cells.
        self.map_width = None
        ## Map resolution (metres per cell).
        self.map_resolution = None
        ## Distance transform of the map (metres to the nearest obstacle per cell).
        self.dt = None

        # precomputing corresponding cosines and sines of the angle array
        theta_arr = np.linspace(0.0, 2*np.pi, num=theta_dis)
        ## Lookup table of sin(angle) over 0 to 2*pi.
        self.sines = np.sin(theta_arr)
        ## Lookup table of cos(angle) over 0 to 2*pi.
        self.cosines = np.cos(theta_arr)

    def set_map(self, map_path, map_ext):
        """!
        @brief Load a map (YAML + image), binarise it, and compute its distance transform.

        The image path is the YAML path with its extension replaced by @p map_ext.
        The YAML's own @c image field is ignored, so the image must sit next to the
        YAML with the same base name. The image is:
        - flipped vertically, so row 0 is the bottom of the map, matching the world y axis;
        - thresholded at 128: pixels <= 128 become obstacles (0) and brighter pixels free space (255).

        From the YAML only @c resolution and @c origin ([x, y, yaw]) are read.

        @param map_path Path to the map YAML file.
        @param map_ext  Image extension *including the dot*, e.g. ".png" or ".pgm".
        @return Always True.

        @note If the YAML fails to parse, the error is printed and the method
              carries on, failing later with an AttributeError on @c self.origin.
        """
        # TODO: do we open the option to flip the images, and turn rgb into grayscale? or specify the exact requirements in documentation.
        # TODO: throw error if image specification isn't met

        # load map image (flip so row 0 is the bottom edge, matching world +y up)
        map_img_path = os.path.splitext(map_path)[0] + map_ext
        ## Binarised map image (0 = obstacle, 255 = free), flipped so row 0 is the bottom.
        self.map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM))
        self.map_img = self.map_img.astype(np.float64)

        # grayscale -> binary
        self.map_img[self.map_img <= 128.] = 0.
        self.map_img[self.map_img > 128.] = 255.

        self.map_height = self.map_img.shape[0]
        self.map_width = self.map_img.shape[1]

        # load map yaml
        with open(map_path, 'r') as yaml_stream:
            try:
                map_metadata = yaml.safe_load(yaml_stream)
                self.map_resolution = map_metadata['resolution']
                ## Map origin [x, y, yaw] from the YAML: world pose of the bottom-left cell.
                self.origin = map_metadata['origin']
            except yaml.YAMLError as ex:
                print(ex)

        # calculate map parameters
        self.orig_x = self.origin[0]
        self.orig_y = self.origin[1]
        self.orig_s = np.sin(self.origin[2])
        self.orig_c = np.cos(self.origin[2])

        # get the distance transform
        self.dt = get_dt(self.map_img, self.map_resolution)

        return True

    def scan(self, pose, rng, std_dev=0.01):
        """!
        @brief Simulate one LiDAR scan from @p pose, with optional Gaussian noise.

        @param pose    np.ndarray of shape (3,): [x, y, yaw] of the sensor in the world.
        @param rng     np.random.Generator used for noise, or None for a noise-free scan.
        @param std_dev Standard deviation of the zero-mean Gaussian noise added to every beam (m).
        @return np.ndarray of shape (num_beams,): beam ranges (m), from the rightmost beam to the leftmost.
                Noise is added after clipping, so values can be slightly above
                max_range or slightly below 0.

        @exception ValueError If no map has been loaded with set_map().
        """

        if self.map_height is None:
            raise ValueError('Map is not set for scan simulator.')

        scan = get_scan(pose, self.theta_dis, self.fov, self.num_beams, self.theta_index_increment, self.sines, self.cosines, self.eps, self.orig_x, self.orig_y, self.orig_c, self.orig_s, self.map_height, self.map_width, self.map_resolution, self.dt, self.max_range)

        if rng is not None:
            noise = rng.normal(0., std_dev, size=self.num_beams)
            scan += noise

        return scan

    def get_increment(self):
        """!
        @brief Angle between neighbouring beams.

        @return Beam spacing (rad), equal to fov / (num_beams - 1).
        """
        return self.angle_increment


# ----------------------------------------------------------------------------
# Unit tests for the 2D scan simulator class
# Author: Hongrui Zheng
#
# Test cases:
#   - Map comparison against the legacy C++ simulator (berlin, skirk), using MSE
#     as the metric. Currently commented out because the legacy data file is not included.
#   - FPS test: scans per second should be greater than 500.
#   - RNG test: re-seeding the generator reproduces the same noise.
# ----------------------------------------------------------------------------


class ScanTests(unittest.TestCase):
    """!
    @brief Unit tests for ScanSimulator2D.

    @note Map paths are relative ('../envs/maps/berlin.yaml'), so run these from a
          directory where that path resolves, e.g. f110_gym/envs or a sibling folder.
    """

    def setUp(self):
        """!
        @brief Set LiDAR settings and a set of test poses rotating in place at the origin.
        """
        # test params
        self.num_beams = 1080
        self.fov = 4.7

        self.num_test = 10
        self.test_poses = np.zeros((self.num_test, 3))
        self.test_poses[:, 2] = np.linspace(-1., 1., num=self.num_test)

        # # legacy gym data
        # sample_scan = np.load('legacy_scan.npz')
        # self.berlin_scan = sample_scan['berlin']
        # self.skirk_scan = sample_scan['skirk']

    # def test_map_berlin(self):
    #     scan_rng = np.random.default_rng(seed=12345)
    #     scan_sim = ScanSimulator2D(self.num_beams, self.fov)
    #     new_berlin = np.empty((self.num_test, self.num_beams))
    #     map_path = '../../../maps/berlin.yaml'
    #     map_ext = '.png'
    #     scan_sim.set_map(map_path, map_ext)
    #     # scan gen loop
    #     for i in range(self.num_test):
    #         test_pose = self.test_poses[i]
    #         new_berlin[i,:] = scan_sim.scan(test_pose, scan_rng)
    #     diff = self.berlin_scan - new_berlin
    #     mse = np.mean(diff**2)
    #     # print('Levine distance test, norm: ' + str(norm))

    #     # plotting
    #     import matplotlib.pyplot as plt
    #     theta = np.linspace(-self.fov/2., self.fov/2., num=self.num_beams)
    #     plt.polar(theta, new_berlin[1,:], '.', lw=0)
    #     plt.polar(theta, self.berlin_scan[1,:], '.', lw=0)
    #     plt.show()

    #     self.assertLess(mse, 2.)

    # def test_map_skirk(self):
    #     scan_rng = np.random.default_rng(seed=12345)
    #     scan_sim = ScanSimulator2D(self.num_beams, self.fov)
    #     new_skirk = np.empty((self.num_test, self.num_beams))
    #     map_path = '../../../maps/skirk.yaml'
    #     map_ext = '.png'
    #     scan_sim.set_map(map_path, map_ext)
    #     print('map set')
    #     # scan gen loop
    #     for i in range(self.num_test):
    #         test_pose = self.test_poses[i]
    #         new_skirk[i,:] = scan_sim.scan(test_pose, scan_rng)
    #     diff = self.skirk_scan - new_skirk
    #     mse = np.mean(diff**2)
    #     print('skirk distance test, mse: ' + str(mse))

    #     # plotting
    #     import matplotlib.pyplot as plt
    #     theta = np.linspace(-self.fov/2., self.fov/2., num=self.num_beams)
    #     plt.polar(theta, new_skirk[1,:], '.', lw=0)
    #     plt.polar(theta, self.skirk_scan[1,:], '.', lw=0)
    #     plt.show()

    #     self.assertLess(mse, 2.)

    def test_fps(self):
        """!
        @brief Scanning the berlin map runs faster than 500 scans per second.
        """
        # scan fps should be greater than 500

        scan_rng = np.random.default_rng(seed=12345)
        scan_sim = ScanSimulator2D(self.num_beams, self.fov)
        map_path = '../envs/maps/berlin.yaml'
        map_ext = '.png'
        scan_sim.set_map(map_path, map_ext)

        import time
        start = time.time()
        for i in range(10000):
            x_test = i/10000
            scan = scan_sim.scan(np.array([x_test, 0., 0.]), scan_rng)
        end = time.time()
        fps = 10000/(end-start)
        # print('FPS test')
        # print('Elapsed time: ' + str(end-start) + ' , FPS: ' + str(1/fps))
        self.assertGreater(fps, 500.)

    def test_rng(self):
        """!
        @brief Noise is reproducible: re-seeding the generator reproduces the same
               scans, and successive scans from one generator differ.
        """
        num_beams = 1080
        fov = 4.7
        map_path = '../envs/maps/berlin.yaml'
        map_ext = '.png'
        it = 100

        scan_rng = np.random.default_rng(seed=12345)
        scan_sim = ScanSimulator2D(num_beams, fov)
        scan_sim.set_map(map_path, map_ext)
        scan1 = scan_sim.scan(np.array([0., 0., 0.]), scan_rng)
        scan2 = scan_sim.scan(np.array([0., 0., 0.]), scan_rng)
        for i in range(it):
            scan3 = scan_sim.scan(np.array([0., 0., 0.]), scan_rng)
        scan4 = scan_sim.scan(np.array([0., 0., 0.]), scan_rng)

        scan_rng = np.random.default_rng(seed=12345)
        scan5 = scan_sim.scan(np.array([0., 0., 0.]), scan_rng)
        scan2 = scan_sim.scan(np.array([0., 0., 0.]), scan_rng)
        for i in range(it):
            _ = scan_sim.scan(np.array([0., 0., 0.]), scan_rng)
        scan6 = scan_sim.scan(np.array([0., 0., 0.]), scan_rng)

        self.assertTrue(np.allclose(scan1, scan5))
        self.assertFalse(np.allclose(scan1, scan2))
        self.assertFalse(np.allclose(scan1, scan3))
        self.assertTrue(np.allclose(scan4, scan6))


def main():
    """!
    @brief Manual demo: time 10000 scans on the example map, then animate a scan rotating in place on a polar plot.

    Not run by default; uncomment @c main() in the @c __main__ block to use it.
    """
    num_beams = 1080
    fov = 4.7
    # map_path = '../envs/maps/berlin.yaml'
    map_path = '../../../examples/example_map.yaml'
    map_ext = '.png'
    scan_rng = np.random.default_rng(seed=12345)
    scan_sim = ScanSimulator2D(num_beams, fov)
    scan_sim.set_map(map_path, map_ext)
    scan = scan_sim.scan(np.array([0., 0., 0.]), scan_rng)

    # fps test
    import time
    start = time.time()
    for i in range(10000):
        x_test = i/10000
        scan = scan_sim.scan(np.array([x_test, 0., 0.]), scan_rng)
    end = time.time()
    fps = (end-start)/10000
    print('FPS test')
    print('Elapsed time: ' + str(end-start) + ' , FPS: ' + str(1/fps))

    # visualization
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    num_iter = 100
    theta = np.linspace(-fov/2., fov/2., num=num_beams)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    ax.set_ylim(0, 31)
    line, = ax.plot([], [], '.', lw=0)
    def update(i):
        # x_ani = i * 3. / num_iter
        theta_ani = -i * 2 * np.pi / num_iter
        x_ani = 0.
        current_scan = scan_sim.scan(np.array([x_ani, 0., theta_ani]), scan_rng)
        print(np.max(current_scan))
        line.set_data(theta, current_scan)
        return line,
    ani = FuncAnimation(fig, update, frames=num_iter, blit=True)
    plt.show()

if __name__ == '__main__':
    unittest.main()
    #main()

    # import time
    # pt_a = np.array([1., 1.])
    # pt_b = np.array([1., 2.])
    # pt_c = np.array([1., 3.])
    # col = are_collinear(pt_a, pt_b, pt_c)
    # print(col)

    # pose = np.array([0., 0., -1.])
    # beam_theta = 0.
    # start = time.time()
    # dist = get_range(pose, beam_theta, pt_a, pt_b)
    # print(dist, time.time()-start)

    # num_beams = 1080
    # scan = 100.*np.ones((num_beams, ))
    # scan_angles = np.linspace(-2.35, 2.35, num=num_beams)
    # assert scan.shape[0] == scan_angles.shape[0]
    # vertices = np.asarray([[4,11.],[5,5],[9,9],[10,10]])
    # start = time.time()
    # new_scan = ray_cast(pose, scan, scan_angles, vertices)
    # print(time.time()-start)
