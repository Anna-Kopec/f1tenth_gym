"""!
@file pyglet_test.py
@brief Standalone experiment: draw a map's obstacle pixels as points in a pyglet window.

Loads a map (YAML + image) the same way rendering.py: EnvRenderer.update_map()
does, converts every pure-black pixel to a world-coordinate point, and draws them
with a 3D perspective projection. It also contains two small 2D camera classes
(Camera, CenteredCamera).

Usage:
@code
python pyglet_test.py --map_path ../envs/maps/vegas --map_ext .png
@endcode

@note The Camera classes use @c window.view, which only exists in pyglet 2.x,
      while the drawing code uses the pyglet 1.x batch.add() format. setup.py pins
      pyglet < 1.5, so the script is unlikely to run unchanged with either version.
"""
import numpy as np
from PIL import Image
import yaml

from pyglet.gl import *
import pyglet
from pyglet import font, graphics, window

import argparse

class Camera:
    """!
    @brief A simple 2D camera: a scroll offset and a zoom factor, applied to the window's view matrix.

    Use it around drawing code, either with begin() / end() or as a context manager:
    @code
    with cam:
        batch.draw()
    @endcode
    """

    def __init__(self, window: pyglet.window.Window, scroll_speed=1, min_zoom=1, max_zoom=4):
        """!
        @brief Create a camera for @p window, at offset (0, 0) and zoom 1 (clamped to the zoom limits).

        @param window       The pyglet window whose view matrix is changed.
        @param scroll_speed Distance moved per unit of move() input.
        @param min_zoom     Smallest allowed zoom.
        @param max_zoom     Largest allowed zoom.

        @exception AssertionError If @p min_zoom > @p max_zoom.
        """
        assert min_zoom <= max_zoom, "Minimum zoom must not be greater than maximum zoom"
        self._window = window
        ## Distance moved per unit of move() input.
        self.scroll_speed = scroll_speed
        ## Largest allowed zoom.
        self.max_zoom = max_zoom
        ## Smallest allowed zoom.
        self.min_zoom = min_zoom
        ## Horizontal scroll offset.
        self.offset_x = 0
        ## Vertical scroll offset.
        self.offset_y = 0
        self._zoom = max(min(1, self.max_zoom), self.min_zoom)

    @property
    def zoom(self):
        """!
        @brief Current zoom factor.
        """
        return self._zoom

    @zoom.setter
    def zoom(self, value):
        """!
        @brief Set the zoom factor, clamped to [min_zoom, max_zoom].
        """
        self._zoom = max(min(value, self.max_zoom), self.min_zoom)

    @property
    def position(self):
        """!
        @brief Current scroll offset as a tuple (offset_x, offset_y).
        """
        return self.offset_x, self.offset_y

    @position.setter
    def position(self, value):
        """!
        @brief Set the scroll offset directly from a tuple (x, y).
        """
        self.offset_x, self.offset_y = value

    def move(self, axis_x, axis_y):
        """!
        @brief Move the offset by scroll_speed in the given direction, e.g. move(-1, 0) moves left.

        @param axis_x Horizontal direction (-1, 0 or 1, or any scale factor).
        @param axis_y Vertical direction.
        """
        self.offset_x += self.scroll_speed * axis_x
        self.offset_y += self.scroll_speed * axis_y

    def begin(self):
        """!
        @brief Apply the camera before drawing: translate by -offset * zoom, then scale by zoom.
        """
        # Set the current camera offset so you can draw your scene.

        # Translate using the offset.
        view_matrix = self._window.view.translate(-self.offset_x * self._zoom, -self.offset_y * self._zoom, 0)
        # Scale by zoom level.
        view_matrix = view_matrix.scale(self._zoom, self._zoom, 1)

        self._window.view = view_matrix

    def end(self):
        """!
        @brief Undo begin() after drawing (reverse scale, then reverse translate).

        Without this, the transforms would build up on every frame.
        """
        # Since this is a matrix, you will need to reverse the translate after rendering otherwise
        # it will multiply the current offset every draw update pushing it further and further away.

        # Reverse scale, since that was the last transform.
        view_matrix = self._window.view.scale(1 / self._zoom, 1 / self._zoom, 1)
        # Reverse translate.
        view_matrix = view_matrix.translate(self.offset_x * self._zoom, self.offset_y * self._zoom, 0)

        self._window.view = view_matrix

    def __enter__(self):
        """!
        @brief Context-manager entry: calls begin().
        """
        self.begin()

    def __exit__(self, exception_type, exception_value, traceback):
        """!
        @brief Context-manager exit: calls end().
        """
        self.end()


class CenteredCamera(Camera):
    """!
    @brief Camera variant where (0, 0) is the centre of the window instead of the bottom-left corner.
    """

    def begin(self):
        """!
        @brief Apply the camera, shifted by half the window size so the offset point is centred.
        """
        x = -self._window.width // 2 / self._zoom + self.offset_x
        y = -self._window.height // 2 / self._zoom + self.offset_y

        view_matrix = self._window.view.translate(-x * self._zoom, -y * self._zoom, 0)
        view_matrix = view_matrix.scale(self._zoom, self._zoom, 1)
        self._window.view = view_matrix

    def end(self):
        """!
        @brief Undo begin().
        """
        x = -self._window.width // 2 / self._zoom + self.offset_x
        y = -self._window.height // 2 / self._zoom + self.offset_y

        view_matrix = self._window.view.scale(1 / self._zoom, 1 / self._zoom, 1)
        view_matrix = view_matrix.translate(x * self._zoom, y * self._zoom, 0)
        self._window.view = view_matrix



# ---- command line ----
parser = argparse.ArgumentParser()
parser.add_argument('--map_path', type=str, required=True, help='Path to the map without extensions')
parser.add_argument('--map_ext', type=str, required=True, help='Extension of the map image file')
args = parser.parse_args()

# load map yaml
with open(args.map_path + '.yaml', 'r') as yaml_stream:
    try:
        map_metada = yaml.safe_load(yaml_stream)
        map_resolution = map_metada['resolution']
        origin = map_metada['origin']
        origin_x = origin[0]
        origin_y = origin[1]
    except yaml.YAMLError as ex:
        print(ex)

# load map image (flip so row 0 is the bottom edge)
map_img = np.array(Image.open(args.map_path + args.map_ext).transpose(Image.FLIP_TOP_BOTTOM)).astype(np.float64)
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

# mask and only leave the obstacle points (world metres, unlike EnvRenderer's 50x scale)
map_mask = map_img == 0.0
map_mask_flat = map_mask.flatten()
## Obstacle points in world metres, shape (n, 3).
map_points = map_coords[:, map_mask_flat].T

# prep opengl
try:
    # Try and create a window with multisampling (antialiasing)
    config = Config(sample_buffers=1, samples=4,
                    depth_size=16, double_buffer=True, )
    window = window.Window(resizable=True, config=config)
except window.NoSuchConfigException:
    # Fall back to no multisampling for old hardware
    window = window.Window(resizable=True)

glClearColor(18/255, 4/255, 88/255, 1.)
glEnable(GL_DEPTH_TEST)
# move the scene in front of the perspective camera
glTranslatef(25, -5, -60)

## 2D camera wrapped around the draw call.
cam = Camera(window)

@window.event
def on_resize(width, height):
    """!
    @brief Window resize handler: set up a 3D perspective projection (60 degree field of view).
    """
    # Override the default on_resize handler to create a 3D projection
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60., width / float(height), .1, 1000.)
    glMatrixMode(GL_MODELVIEW)
    return pyglet.event.EVENT_HANDLED

## pyglet Batch holding one point per obstacle pixel.
batch = graphics.Batch()

## Vertex lists for each point, kept so they could be updated later.
points = []
for i in range(map_points.shape[0]):
    particle = batch.add(1, GL_POINTS, None, ('v3f/stream', [map_points[i, 0], map_points[i, 1], map_points[i, 2]]))
    points.append(particle)

def loop(dt):
    """!
    @brief Scheduled every frame: print the current frame rate.

    @param dt Time since the last call (s), supplied by pyglet.clock.
    """
    print(pyglet.clock.get_fps())
    pass

@window.event
def on_draw():
    """!
    @brief Draw handler: clear the window and draw the map points in pink, inside the camera.
    """
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glColor3f(254/255, 117/255, 254/255)
    cam.begin()
    batch.draw()
    cam.end()

pyglet.clock.schedule(loop)
pyglet.app.run()
