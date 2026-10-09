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
@file collision_checks.py
@brief Older standalone prototype of the GJK collision code. Superseded by collision_models.py.

This file is kept for reference only; nothing in the gym imports it. Its functions
are identical to the ones in envs/collision_models.py, except that it has
**no collision_multiple()** (the pairwise check across many cars that the Simulator
uses) and no test for it.

For how GJK works and detailed per-function documentation, see collision_models.py.

Run the tests with: @c python collision_checks.py
(test_get_vert opens a matplotlib window and blocks until it is closed).

@author Hongrui Zheng
"""

import numpy as np
from numba import njit

@njit(cache=True)
def perpendicular(pt):
    """!
    @brief Rotate a 2D vector by -90 degrees: (x, y) -> (y, -x). Modifies @p pt in place.

    @param pt np.ndarray of shape (2,).
    @return The same array, now perpendicular to its original direction.
    @see collision_models.py: perpendicular()
    """
    temp = pt[0]
    pt[0] = pt[1]
    pt[1] = -1*temp
    return pt


@njit(cache=True)
def tripleProduct(a, b, c):
    """!
    @brief 2D vector triple product (a x b) x c = b(a.c) - a(b.c).

    @param a, b, c np.ndarray of shape (2,).
    @return np.ndarray of shape (2,).
    @see collision_models.py: tripleProduct()
    """
    ac = a.dot(c)
    bc = b.dot(c)
    return b*ac - a*bc


@njit(cache=True)
def avgPoint(vertices):
    """!
    @brief Mean of a set of vertices.

    @param vertices np.ndarray of shape (n, 2).
    @return np.ndarray of shape (2,).
    @see collision_models.py: avgPoint()
    """
    return np.sum(vertices, axis=0)/vertices.shape[0]


@njit(cache=True)
def indexOfFurthestPoint(vertices, d):
    """!
    @brief Index of the vertex with the largest dot product with direction @p d.

    @param vertices np.ndarray of shape (n, 2).
    @param d        np.ndarray of shape (2,): search direction.
    @return int row index into @p vertices.
    @see collision_models.py: indexOfFurthestPoint()
    """
    return np.argmax(vertices.dot(d))


@njit(cache=True)
def support(vertices1, vertices2, d):
    """!
    @brief Support point of the Minkowski difference (body1 - body2) in direction @p d.

    @param vertices1 np.ndarray of shape (n, 2): first body.
    @param vertices2 np.ndarray of shape (m, 2): second body.
    @param d         np.ndarray of shape (2,): search direction.
    @return np.ndarray of shape (2,).
    @see collision_models.py: support()
    """
    i = indexOfFurthestPoint(vertices1, d)
    j = indexOfFurthestPoint(vertices2, -d)
    return vertices1[i] - vertices2[j]


@njit(cache=True)
def collision(vertices1, vertices2):
    """!
    @brief GJK test: do two convex polygons overlap?

    @param vertices1 np.ndarray of shape (n, 2): first convex body.
    @param vertices2 np.ndarray of shape (m, 2): second convex body.
    @return True if they overlap (touching counts), False otherwise.
    @see collision_models.py: collision() for a step-by-step explanation.
    """
    index = 0
    simplex = np.empty((3, 2))

    position1 = avgPoint(vertices1)
    position2 = avgPoint(vertices2)

    d = position1 - position2

    if d[0] == 0 and d[1] == 0:
        d[0] = 1.0

    a = support(vertices1, vertices2, d)
    simplex[index, :] = a

    if d.dot(a) <= 0:
        return False

    d = -a

    iter_count = 0
    while iter_count < 1e3:
        a = support(vertices1, vertices2, d)
        index += 1
        simplex[index, :] = a
        if d.dot(a) <= 0:
            return False

        ao = -a

        if index < 2:
            b = simplex[0, :]
            ab = b-a
            d = tripleProduct(ab, ao, ab)
            if np.linalg.norm(d) < 1e-10:
                d = perpendicular(ab)
            continue

        b = simplex[1, :]
        c = simplex[0, :]
        ab = b-a
        ac = c-a

        acperp = tripleProduct(ab, ac, ac)

        if acperp.dot(ao) >= 0:
            d = acperp
        else:
            abperp = tripleProduct(ac, ab, ab)
            if abperp.dot(ao) < 0:
                return True
            simplex[0, :] = simplex[1, :]
            d = abperp

        simplex[1, :] = simplex[2, :]
        index -= 1

        iter_count += 1
    return False

# ----------------------------------------------------------------------------
# Utility functions for getting vertices by pose and shape
# ----------------------------------------------------------------------------

@njit(cache=True)
def get_trmtx(pose):
    """!
    @brief 4x4 homogeneous transform from the vehicle frame to the world frame.

    @param pose np.ndarray of shape (3,): [x, y, yaw].
    @return np.ndarray of shape (4, 4).
    @see collision_models.py: get_trmtx()
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
    @brief World-frame corners of a rectangle centred on @p pose.

    @param pose   np.ndarray of shape (3,): [x, y, yaw] of the rectangle centre.
    @param length Rectangle length along the heading (m).
    @param width  Rectangle width (m).
    @return np.ndarray of shape (4, 2): rear-left, rear-right, front-right, front-left.
    @see collision_models.py: get_vertices()
    """
    H = get_trmtx(pose)
    rl = H.dot(np.asarray([[-length/2],[width/2],[0.], [1.]])).flatten()
    rr = H.dot(np.asarray([[-length/2],[-width/2],[0.], [1.]])).flatten()
    fl = H.dot(np.asarray([[length/2],[width/2],[0.], [1.]])).flatten()
    fr = H.dot(np.asarray([[length/2],[-width/2],[0.], [1.]])).flatten()
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
    @brief Unit tests for get_vertices() and collision().
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
        @brief get_vertices() returns a (4, 2) array; also plots the rectangle (blocks until the window is closed).
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