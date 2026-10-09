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
@file rendering.py
@brief pyglet / OpenGL renderer for the F1TENTH gym (EnvRenderer).

@section rendering_coords Drawing coordinates
Everything is drawn in a "pixel world" frame, which is world metres multiplied
by 50. The camera is an orthographic projection whose bounds (left, right,
bottom, top) change as you pan and zoom. Zooming changes how many of these
units fit in the window. The text label sits at a fixed position in this
frame, so it moves with the map rather than staying fixed on screen.

@section rendering_drawn What is drawn
- **Map:** one grey point for every obstacle pixel in the map image.
- **Cars:** a filled rectangle per car. The ego car is drawn in a brighter purple.
- **Label:** lap time and ego lap count.
- **FPS counter:** in the bottom-left corner.

Everything except the FPS counter is held in one pyglet Batch, so it is drawn in a single call.
Extra shapes can be added to @c batch from F110Env render callbacks.

@author Hongrui Zheng
"""

# opengl stuff
import pyglet
from pyglet.gl import *

# other
import numpy as np
from PIL import Image
import yaml

# helpers
from f110_gym.envs.collision_models import get_vertices

# zooming constants
## Scale applied per scroll step when zooming. Despite the name, scrolling up
#  (dy > 0) multiplies the zoom level by this, which shows more of the map (zooms out).
ZOOM_IN_FACTOR = 1.2
## Scale applied per scroll step in the other direction.
ZOOM_OUT_FACTOR = 1/ZOOM_IN_FACTOR

# vehicle shape constants
## Length of the drawn car rectangle (m). Fixed here; does not follow the env's params.
CAR_LENGTH = 0.58
## Width of the drawn car rectangle (m). Fixed here; does not follow the env's params.
CAR_WIDTH = 0.31

class EnvRenderer(pyglet.window.Window):
    """!
    @brief pyglet window that draws the environment and handles panning, zooming and resizing.

    F110Env.render() creates and drives it: update_obs(), then any render
    callbacks, then dispatch_events(), on_draw() and flip().
    """
    def __init__(self, width, height, *args, **kwargs):
        """!
        @brief Open the window (4x anti-aliasing, double-buffered, resizable) and set up the camera, batch and labels.

        @param width  Initial window width in pixels.
        @param height Initial window height in pixels.
        @param args, kwargs Passed on to pyglet.window.Window.
        """
        conf = Config(sample_buffers=1,
                      samples=4,
                      depth_size=16,
                      double_buffer=True)
        super().__init__(width, height, config=conf, resizable=True, vsync=False, *args, **kwargs)

        # gl init: dark blue background
        glClearColor(9/255, 32/255, 87/255, 1.)

        # initialize camera values
        ## Left edge of the visible area, in drawing units.
        self.left = -width/2
        ## Right edge of the visible area.
        self.right = width/2
        ## Bottom edge of the visible area.
        self.bottom = -height/2
        ## Top edge of the visible area.
        self.top = height/2
        ## Drawing units per screen pixel. Larger values show more of the map.
        self.zoom_level = 1.2
        ## Visible width in drawing units.
        self.zoomed_width = width
        ## Visible height in drawing units.
        self.zoomed_height = height

        # current batch that keeps track of all graphics
        ## pyglet Batch holding the map points, car shapes and label. Render callbacks can add to it.
        self.batch = pyglet.graphics.Batch()

        # current env map
        ## Obstacle points in drawing units, shape (n, 3). None until update_map() is called.
        self.map_points = None

        # current env agent poses, (num_agents, 3), columns are (x, y, theta)
        ## Latest car poses, shape (num_agents, 3): [x, y, yaw] in metres. None until update_obs() is called.
        self.poses = None

        # current env agent vertices, (num_agents, 4, 2), 2nd and 3rd dimensions are the 4 corners in 2D
        ## Reserved for car corners; currently unused.
        self.vertices = None

        # current score label
        ## Text label showing lap time and ego lap count. Placed at (0, -800) in drawing units.
        self.score_label = pyglet.text.Label(
                'Lap Time: {laptime:.2f}, Ego Lap Count: {count:.0f}'.format(
                    laptime=0.0, count=0.0),
                font_size=36,
                x=0,
                y=-800,
                anchor_x='center',
                anchor_y='center',
                # width=0.01,
                # height=0.01,
                color=(255, 255, 255, 255),
                batch=self.batch)

        ## Frame-rate counter, drawn in the bottom-left corner.
        self.fps_display = pyglet.window.FPSDisplay(self)

    def update_map(self, map_path, map_ext):
        """!
        @brief Load a map and add one point per obstacle pixel to the batch.

        Each obstacle pixel (value exactly 0) is converted to world metres using
        the YAML's resolution and origin, then scaled by 50 into drawing units.

        @param map_path Path to the map *without* an extension; '.yaml' and
                        @p map_ext are appended to it.
        @param map_ext  Image extension, including the dot.

        @note The origin yaw in the YAML is ignored here (the scan simulator does use it).
        @note Only pixels that are exactly 0 are drawn. The scan simulator treats
              everything <= 128 as an obstacle, so grey pixels count as walls but are not drawn.
        @note Calling this again adds the new points without removing the old ones.
        """

        # load map metadata
        with open(map_path + '.yaml', 'r') as yaml_stream:
            try:
                map_metadata = yaml.safe_load(yaml_stream)
                map_resolution = map_metadata['resolution']
                origin = map_metadata['origin']
                origin_x = origin[0]
                origin_y = origin[1]
            except yaml.YAMLError as ex:
                print(ex)

        # load map image (flip so row 0 is the bottom edge)
        map_img = np.array(Image.open(map_path + map_ext).transpose(Image.FLIP_TOP_BOTTOM)).astype(np.float64)
        map_height = map_img.shape[0]
        map_width = map_img.shape[1]

        # convert map pixels to coordinates
        range_x = np.arange(map_width)
        range_y = np.arange(map_height)
        map_x, map_y = np.meshgrid(range_x, range_y)
        map_x = (map_x * map_resolution + origin_x).flatten()
        map_y = (map_y * map_resolution + origin_y).flatten()
        map_z = np.zeros(map_y.shape)
        map_coords = np.vstack((map_x, map_y, map_z))

        # mask and only leave the obstacle points, scaled from metres to drawing units
        map_mask = map_img == 0.0
        map_mask_flat = map_mask.flatten()
        map_points = 50. * map_coords[:, map_mask_flat].T
        for i in range(map_points.shape[0]):
            self.batch.add(1, GL_POINTS, None, ('v3f/stream', [map_points[i, 0], map_points[i, 1], map_points[i, 2]]), ('c3B/stream', [183, 193, 222]))
        self.map_points = map_points

    def on_resize(self, width, height):
        """!
        @brief pyglet callback when the window is resized: recentre the camera at the current zoom level.

        Overrides pyglet.window.Window.on_resize.

        @param width  New window width in pixels.
        @param height New window height in pixels.

        @note Any panning is lost on resize; the view recentres on the origin.
        """

        # call overrided function
        super().on_resize(width, height)

        # update camera value
        (width, height) = self.get_size()
        self.left = -self.zoom_level * width/2
        self.right = self.zoom_level * width/2
        self.bottom = -self.zoom_level * height/2
        self.top = self.zoom_level * height/2
        self.zoomed_width = self.zoom_level * width
        self.zoomed_height = self.zoom_level * height

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """!
        @brief pyglet callback for mouse drag: pan the camera so the map follows the cursor.

        Overrides pyglet.window.Window.on_mouse_drag.

        @param x         Cursor distance in pixels from the left edge of the window.
        @param y         Cursor distance in pixels from the bottom edge of the window.
        @param dx        Horizontal movement since the last event (pixels).
        @param dy        Vertical movement since the last event (pixels).
        @param buttons   Bitwise combination of the mouse buttons currently pressed.
        @param modifiers Bitwise combination of the keyboard modifiers currently active.
        """

        # pan camera (pixels -> drawing units via zoom_level)
        self.left -= dx * self.zoom_level
        self.right -= dx * self.zoom_level
        self.bottom -= dy * self.zoom_level
        self.top -= dy * self.zoom_level

    def on_mouse_scroll(self, x, y, dx, dy):
        """!
        @brief pyglet callback for the scroll wheel: zoom about the cursor position.

        The point under the cursor stays fixed while the visible area grows or
        shrinks. The zoom level is kept between 0.01 and 10.

        Overrides pyglet.window.Window.on_mouse_scroll.

        @param x  Cursor distance in pixels from the left edge of the window.
        @param y  Cursor distance in pixels from the bottom edge of the window.
        @param dx Horizontal scroll amount.
        @param dy Vertical scroll amount (positive = scroll up).
        """

        # Get scale factor
        f = ZOOM_IN_FACTOR if dy > 0 else ZOOM_OUT_FACTOR if dy < 0 else 1

        # If zoom_level is in the proper range
        if .01 < self.zoom_level * f < 10:

            self.zoom_level *= f

            (width, height) = self.get_size()

            # cursor position as a fraction of the window
            mouse_x = x/width
            mouse_y = y/height

            # world point currently under the cursor
            mouse_x_in_world = self.left + mouse_x*self.zoomed_width
            mouse_y_in_world = self.bottom + mouse_y*self.zoomed_height

            self.zoomed_width *= f
            self.zoomed_height *= f

            # rebuild the view so that point stays under the cursor
            self.left = mouse_x_in_world - mouse_x * self.zoomed_width
            self.right = mouse_x_in_world + (1 - mouse_x) * self.zoomed_width
            self.bottom = mouse_y_in_world - mouse_y * self.zoomed_height
            self.top = mouse_y_in_world + (1 - mouse_y) * self.zoomed_height

    def on_close(self):
        """!
        @brief pyglet callback when the window's close button is clicked: close the window, then raise.

        Raising stops a simulation loop that would otherwise keep running with
        no window. Overrides pyglet.window.Window.on_close.

        @exception Exception Always raised, with the message 'Rendering window was closed.'
        """

        super().on_close()
        raise Exception('Rendering window was closed.')

    def on_draw(self):
        """!
        @brief Draw one frame: set the orthographic camera to the current view, then draw the batch and FPS counter.

        @exception Exception If update_map() or update_obs() has not been called yet.
        """

        # if map and poses doesn't exist, raise exception
        if self.map_points is None:
            raise Exception('Map not set for renderer.')
        if self.poses is None:
            raise Exception('Agent poses not updated for renderer.')

        # Initialize Projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        # Initialize Modelview matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # Save the default modelview matrix
        glPushMatrix()

        # Clear window with ClearColor
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Set orthographic projection matrix (camera bounds in drawing units)
        glOrtho(self.left, self.right, self.bottom, self.top, 1, -1)

        # Draw all batches
        self.batch.draw()
        self.fps_display.draw()
        # Remove default modelview matrix
        glPopMatrix()

    def update_obs(self, obs):
        """!
        @brief Move the car shapes to the latest poses and refresh the label text.

        The car shapes are created on the first call, which fixes the number of
        cars and which one is the ego. Later calls only move them.

        @param obs Dict with keys @c ego_idx, @c poses_x, @c poses_y, @c poses_theta,
                   @c lap_times and @c lap_counts (F110Env.render_obs).

        @note The lap time shown is car 0's, not necessarily the ego car's.
        """

        ## Index of the ego car, from the latest observation.
        self.ego_idx = obs['ego_idx']
        poses_x = obs['poses_x']
        poses_y = obs['poses_y']
        poses_theta = obs['poses_theta']

        num_agents = len(poses_x)
        if self.poses is None:
            # first call: create one coloured rectangle per car
            ## Pyglet vertex lists for each car's rectangle.
            self.cars = []
            for i in range(num_agents):
                if i == self.ego_idx:
                    vertices_np = get_vertices(np.array([0., 0., 0.]), CAR_LENGTH, CAR_WIDTH)
                    vertices = list(vertices_np.flatten())
                    car = self.batch.add(4, GL_QUADS, None, ('v2f', vertices), ('c3B', [172, 97, 185, 172, 97, 185, 172, 97, 185, 172, 97, 185]))
                    self.cars.append(car)
                else:
                    vertices_np = get_vertices(np.array([0., 0., 0.]), CAR_LENGTH, CAR_WIDTH)
                    vertices = list(vertices_np.flatten())
                    car = self.batch.add(4, GL_QUADS, None, ('v2f', vertices), ('c3B', [99, 52, 94, 99, 52, 94, 99, 52, 94, 99, 52, 94]))
                    self.cars.append(car)

        # move each car's rectangle to its current pose (metres -> drawing units)
        poses = np.stack((poses_x, poses_y, poses_theta)).T
        for j in range(poses.shape[0]):
            vertices_np = 50. * get_vertices(poses[j, :], CAR_LENGTH, CAR_WIDTH)
            vertices = list(vertices_np.flatten())
            self.cars[j].vertices = vertices
        self.poses = poses

        self.score_label.text = 'Lap Time: {laptime:.2f}, Ego Lap Count: {count:.0f}'.format(laptime=obs['lap_times'][0], count=obs['lap_counts'][obs['ego_idx']])
