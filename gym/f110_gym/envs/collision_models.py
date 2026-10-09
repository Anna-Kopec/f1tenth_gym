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
@file collision_models.py
@brief Car-to-car collision detection (2D GJK) and helpers for vehicle body geometry.

@section collision_models_gjk_overview How the GJK collision test works
GJK (Gilbert-Johnson-Keerthi) decides whether two convex shapes overlap without
testing their edges against each other directly. It relies on one fact:

> Two convex shapes A and B overlap if and only if their Minkowski difference
> A - B = { a - b : a in A, b in B } contains the origin.

The Minkowski difference is never built explicitly. Instead, a *support function*
returns its furthest point in any requested direction. The algorithm grows a
simplex (a point, then a line segment, then a triangle in 2D) from support points,
each time searching in the direction of the origin:
- If a new support point fails to get past the origin in the search direction,
  the origin cannot be inside A - B, so the shapes are separated.
- If the triangle ever encloses the origin, the shapes overlap.

The implementation is a port of https://github.com/kroitor/gjk.c. Every
function is compiled with numba (@c njit) for speed.

@section collision_models_vertex_helpers Vehicle geometry helpers
get_trmtx() and get_vertices() turn a pose [x, y, yaw] plus a body length and
width into the four world-frame corners of the car's rectangle. Both the
Simulator (for collisions) and the renderer (for drawing) use them.

@author Hongrui Zheng
"""

import numpy as np
from numba import njit

@njit(cache=True)
def perpendicular(pt):
    """!
    @brief Rotate a 2D vector by -90 degrees: (x, y) -> (y, -x).

    GJK uses this as a fallback search direction when the origin lies exactly
    on the current simplex edge, where the triple product is zero.

    @warning @p pt is modified in place, and the same array is returned.

    @param pt np.ndarray of shape (2,): input vector.
    @return np.ndarray of shape (2,): the perpendicular vector (the same object as @p pt).
    """
    temp = pt[0]
    pt[0] = pt[1]
    pt[1] = -1*temp
    return pt


@njit(cache=True)
def tripleProduct(a, b, c):
    """!
    @brief 2D vector triple product (a x b) x c, expanded as b(a.c) - a(b.c).

    GJK uses this to find a direction perpendicular to a simplex edge that
    points toward the origin. For example, tripleProduct(ab, ao, ab) is
    perpendicular to edge ab and on the same side as the origin.

    @param a np.ndarray of shape (2,).
    @param b np.ndarray of shape (2,).
    @param c np.ndarray of shape (2,).
    @return np.ndarray of shape (2,): b * (a.c) - a * (b.c).
    """
    ac = a.dot(c)
    bc = b.dot(c)
    return b*ac - a*bc


@njit(cache=True)
def avgPoint(vertices):
    """!
    @brief Mean of a set of vertices (the centroid of the corner points).

    GJK uses it to pick a sensible first search direction (from one body's
    centre toward the other's).

    @param vertices np.ndarray of shape (n, 2).
    @return np.ndarray of shape (2,): mean of the rows of @p vertices.
    """
    return np.sum(vertices, axis=0)/vertices.shape[0]


@njit(cache=True)
def indexOfFurthestPoint(vertices, d):
    """!
    @brief Index of the vertex that lies furthest along direction @p d.

    "Furthest along d" means the vertex with the largest dot product with d.
    For a convex polygon this is its support point in direction d.

    @param vertices np.ndarray of shape (n, 2): polygon vertices.
    @param d        np.ndarray of shape (2,): search direction (need not be unit length).
    @return int: row index into @p vertices.
    """
    return np.argmax(vertices.dot(d))


@njit(cache=True)
def support(vertices1, vertices2, d):
    """!
    @brief Support function of the Minkowski difference (body1 - body2) in direction @p d.

    The furthest point of A - B along d is the furthest point of A along d
    minus the furthest point of B along -d.

    @param vertices1 np.ndarray of shape (n, 2): vertices of the first body.
    @param vertices2 np.ndarray of shape (m, 2): vertices of the second body.
    @param d         np.ndarray of shape (2,): search direction.
    @return np.ndarray of shape (2,): a single point on the boundary of the Minkowski difference.
    """
    i = indexOfFurthestPoint(vertices1, d)
    j = indexOfFurthestPoint(vertices2, -d)
    return vertices1[i] - vertices2[j]


@njit(cache=True)
def collision(vertices1, vertices2):
    """!
    @brief GJK test: do two convex polygons overlap?

    See the file overview for the idea behind the algorithm. Here,
    @c a is always the newest simplex point and @c ao = -a points from it to the origin.

    @param vertices1 np.ndarray of shape (n, 2): vertices of the first convex body.
    @param vertices2 np.ndarray of shape (m, 2): vertices of the second convex body.
    @return True if the bodies overlap (touching counts as overlapping), False otherwise.

    @note The loop gives up after 1000 iterations and returns False (no collision).
          Polygons this small normally finish in a few iterations.
    """
    index = 0
    # simplex holds up to 3 points; index is the slot of the newest one
    simplex = np.empty((3, 2))

    position1 = avgPoint(vertices1)
    position2 = avgPoint(vertices2)

    # first search direction: from body 2's centre toward body 1's centre
    d = position1 - position2

    # centres coincide: any direction will do
    if d[0] == 0 and d[1] == 0:
        d[0] = 1.0

    a = support(vertices1, vertices2, d)
    simplex[index, :] = a

    # the furthest point along d doesn't get past the origin -> separated
    if d.dot(a) <= 0:
        return False

    # next, search from that point back toward the origin
    d = -a

    iter_count = 0
    while iter_count < 1e3:
        a = support(vertices1, vertices2, d)
        index += 1
        simplex[index, :] = a
        # the new point didn't pass the origin -> origin is outside A - B -> no overlap
        if d.dot(a) <= 0:
            return False

        ao = -a

        # simplex is a line segment (2 points): search perpendicular to it, toward the origin
        if index < 2:
            b = simplex[0, :]
            ab = b-a
            d = tripleProduct(ab, ao, ab)
            # origin lies on the line through ab: pick either perpendicular
            if np.linalg.norm(d) < 1e-10:
                d = perpendicular(ab)
            continue

        # simplex is a triangle (a newest, then b, c): find which edge region holds the origin
        b = simplex[1, :]
        c = simplex[0, :]
        ab = b-a
        ac = c-a

        # outward normal of edge ac (pointing away from b)
        acperp = tripleProduct(ab, ac, ac)

        if acperp.dot(ao) >= 0:
            # origin is outside edge ac: drop b, keep (c, a), search along acperp
            d = acperp
        else:
            # outward normal of edge ab (pointing away from c)
            abperp = tripleProduct(ac, ab, ab)
            if abperp.dot(ao) < 0:
                # origin is inside both edges adjacent to a -> inside the triangle -> overlap
                return True
            # origin is outside edge ab: drop c, keep (b, a), search along abperp
            simplex[0, :] = simplex[1, :]
            d = abperp

        # shift the newest point down so the simplex is back to 2 points
        simplex[1, :] = simplex[2, :]
        index -= 1

        iter_count += 1
    return False

@njit(cache=True)
def collision_multiple(vertices):
    """!
    @brief Run the GJK test on every pair of bodies.

    Every unordered pair (i, j) with i < j is tested once, which is O(n^2) in
    the number of bodies.

    @param vertices np.ndarray of shape (num_bodies, 4, 2): the four corners of each body.
    @return Tuple (collisions, collision_idx):
            - @c collisions np.ndarray of shape (num_bodies,): 1.0 if that body
              overlaps any other body, else 0.0.
            - @c collision_idx np.ndarray of shape (num_bodies,): index of the body
              it overlaps, or -1 if none.

    @note If a body overlaps several others, @c collision_idx keeps only the
          partner from the last colliding pair tested. That is the highest-numbered
          partner above it if there is one, otherwise the highest-numbered one below it.
    """
    collisions = np.zeros((vertices.shape[0], ))
    collision_idx = -1 * np.ones((vertices.shape[0], ))
    # looping over all pairs
    for i in range(vertices.shape[0]-1):
        for j in range(i+1, vertices.shape[0]):
            # check collision (GJK needs contiguous arrays for the numba dot products)
            vi = np.ascontiguousarray(vertices[i, :, :])
            vj = np.ascontiguousarray(vertices[j, :, :])
            ij_collision = collision(vi, vj)
            # fill in results for both bodies
            if ij_collision:
                collisions[i] = 1.
                collisions[j] = 1.
                collision_idx[i] = j
                collision_idx[j] = i

    return collisions, collision_idx

# ----------------------------------------------------------------------------
# Utility functions for getting vertices by pose and shape
# ----------------------------------------------------------------------------

@njit(cache=True)
def get_trmtx(pose):
    """!
    @brief Homogeneous transform from the vehicle frame to the world frame.

    The result is a 4x4 matrix: a rotation by the yaw about z, plus a translation
    by (x, y). Multiplying a vehicle-frame point [px, py, 0, 1] by H gives that
    point in world coordinates.

    @param pose np.ndarray of shape (3,): [x, y, yaw] of the vehicle in the world frame.
    @return np.ndarray of shape (4, 4): homogeneous transform H.
    """
    x = pose[0]
    y = pose[1]
    th = pose[2]
    cos = np.cos(th)
    sin = np.sin(th)
    H = np.array([[cos, -sin, 0., x], [sin, cos, 0., y], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    return H

@njit(cache=True)
def get_vertices(pose, length, width):
    """!
    @brief World-frame corners of a rectangular car body.

    The rectangle is centred on (x, y) with its long side along the heading.
    Each corner is defined in the vehicle frame (+/- length/2 forward,
    +/- width/2 to the left) and then transformed to the world frame with get_trmtx().

    @param pose   np.ndarray of shape (3,): [x, y, yaw] of the body centre in the world frame.
    @param length Body length along the heading (m).
    @param width  Body width (m).
    @return np.ndarray of shape (4, 2): corners in the order
            rear-left, rear-right, front-right, front-left.
            Consecutive rows (wrapping from the last back to the first) form the edges.
    """
    H = get_trmtx(pose)
    # corners in the vehicle frame as homogeneous column vectors, transformed to world
    rl = H.dot(np.asarray([[-length/2],[width/2],[0.], [1.]])).flatten()
    rr = H.dot(np.asarray([[-length/2],[-width/2],[0.], [1.]])).flatten()
    fl = H.dot(np.asarray([[length/2],[width/2],[0.], [1.]])).flatten()
    fr = H.dot(np.asarray([[length/2],[-width/2],[0.], [1.]])).flatten()
    # normalise by the homogeneous coordinate (always 1 for a rigid transform)
    rl = rl/rl[3]
    rr = rr/rr[3]
    fl = fl/fl[3]
    fr = fr/fr[3]
    vertices = np.asarray([[rl[0], rl[1]], [rr[0], rr[1]], [fr[0], fr[1]], [fl[0], fl[1]]])
    return vertices


# ----------------------------------------------------------------------------
# Unit tests for GJK collision checks
# Author: Hongrui Zheng
# ----------------------------------------------------------------------------

import time
import unittest

class CollisionTests(unittest.TestCase):
    """!
    @brief Unit tests for get_vertices(), collision() and collision_multiple().

    Run with: @c python collision_models.py
    """

    def setUp(self):
        """!
        @brief Fix the random seed and define a test polygon and car size.
        """
        # test params
        np.random.seed(1234)

        # Collision check body
        self.vertices1 = np.asarray([[4,11.],[5,5],[9,9],[10,10]])

        # car size
        self.length = 0.32
        self.width = 0.22

    def test_get_vert(self):
        """!
        @brief get_vertices() returns a (4, 2) array, and plots the rectangle for visual inspection.

        @note This opens a matplotlib window and blocks until it is closed.
        """
        test_pose = np.array([2.3, 6.7, 0.8])
        vertices = get_vertices(test_pose, self.length, self.width)
        rect = np.vstack((vertices, vertices[0,:]))
        import matplotlib.pyplot as plt
        plt.scatter(test_pose[0], test_pose[1], c='red')
        plt.plot(rect[:, 0], rect[:, 1])
        plt.xlim([1, 4])
        plt.ylim([5, 8])
        plt.axes().set_aspect('equal')
        plt.show()
        self.assertTrue(vertices.shape == (4, 2))

    def test_get_vert_fps(self):
        """!
        @brief get_vertices() runs faster than 500 calls per second.
        """
        test_pose = np.array([2.3, 6.7, 0.8])
        start = time.time()
        for _ in range(1000):
            vertices = get_vertices(test_pose, self.length, self.width)
        elapsed = time.time() - start
        fps = 1000/elapsed
        print('get vertices fps:', fps)
        self.assertTrue(fps>500)

    def test_random_collision(self):
        """!
        @brief Two slightly perturbed copies of the same polygon always collide.
        """
        # perturb the body by a small amount and make sure it all collides with the original body
        for _ in range(1000):
            a = self.vertices1 + np.random.normal(size=(self.vertices1.shape))/100.
            b = self.vertices1 + np.random.normal(size=(self.vertices1.shape))/100.
            self.assertTrue(collision(a,b))

    def test_multiple_collisions(self):
        """!
        @brief collision_multiple() flags six overlapping bodies and leaves one distant body (g) clear.
        """
        a = self.vertices1 + np.random.normal(size=(self.vertices1.shape))/100.
        b = self.vertices1 + np.random.normal(size=(self.vertices1.shape))/100.
        c = self.vertices1 + np.random.normal(size=(self.vertices1.shape))/100.
        d = self.vertices1 + np.random.normal(size=(self.vertices1.shape))/100.
        e = self.vertices1 + np.random.normal(size=(self.vertices1.shape))/100.
        f = self.vertices1 + np.random.normal(size=(self.vertices1.shape))/100.
        g = self.vertices1 + 10.
        allv = np.stack((a,b,c,d,e,f,g))
        collisions, collision_idx = collision_multiple(allv)
        self.assertTrue(np.all(collisions == np.array([1., 1., 1., 1., 1., 1., 0.])))
        self.assertTrue(np.all(collision_idx == np.array([5., 5., 5., 5., 5., 4., -1.])))

    def test_fps(self):
        """!
        @brief collision() runs faster than 500 calls per second.
        """
        # also perturb the body but mainly want to test GJK speed
        start = time.time()
        for _ in range(1000):
            a = self.vertices1 + np.random.normal(size=(self.vertices1.shape))/100.
            b = self.vertices1 + np.random.normal(size=(self.vertices1.shape))/100.
            collision(a, b)
        elapsed = time.time() - start
        fps = 1000/elapsed
        print('gjk fps:', fps)
        self.assertTrue(fps>500)

if __name__ == '__main__':
    unittest.main()
