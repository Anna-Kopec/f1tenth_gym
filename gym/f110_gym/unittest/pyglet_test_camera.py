"""!
@file pyglet_test_camera.py
@brief Standalone experiment for the pan / zoom camera later used by rendering.py: EnvRenderer.

Opens an 800x800 window with one coloured square. Drag with the mouse to pan
and scroll to zoom about the cursor. The camera logic (left/right/bottom/top
bounds plus a glOrtho projection) is the same approach EnvRenderer uses.

Run with: @c python pyglet_test_camera.py
(the window opens as soon as the module is run or imported).
"""
import pyglet
from pyglet.gl import *

# Zooming constants
## Zoom multiplier per scroll step up. Scrolling up multiplies zoom_level by this,
#  which shows more of the scene (zooms out).
ZOOM_IN_FACTOR = 1.2
## Zoom multiplier per scroll step down.
ZOOM_OUT_FACTOR = 1/ZOOM_IN_FACTOR

class App(pyglet.window.Window):
    """!
    @brief Test window with a draggable, zoomable orthographic camera over one coloured square.
    """

    def __init__(self, width, height, *args, **kwargs):
        """!
        @brief Open the window with 4x anti-aliasing and a camera that shows (0, 0) to (width, height).

        @param width, height Window size in pixels.
        @param args, kwargs  Passed on to pyglet.window.Window (e.g. resizable=True).
        """
        conf = Config(sample_buffers=1,
                      samples=4,
                      depth_size=16,
                      double_buffer=True)
        super().__init__(width, height, config=conf, *args, **kwargs)

        #Initialize camera values
        ## Left edge of the visible area (scene units).
        self.left   = 0
        ## Right edge of the visible area.
        self.right  = width
        ## Bottom edge of the visible area.
        self.bottom = 0
        ## Top edge of the visible area.
        self.top    = height
        ## Scene units per screen pixel.
        self.zoom_level = 1
        ## Visible width in scene units.
        self.zoomed_width  = width
        ## Visible height in scene units.
        self.zoomed_height = height

    def init_gl(self, width, height):
        """!
        @brief Set the clear colour, line/polygon smoothing, alpha blending and viewport.

        @note Never called; the matching call in on_resize() is commented out.

        @param width, height Viewport size in pixels.
        """
        # Set clear color
        glClearColor(0/255, 0/255, 0/255, 0/255)

        # Set antialiasing
        glEnable( GL_LINE_SMOOTH )
        glEnable( GL_POLYGON_SMOOTH )
        glHint( GL_LINE_SMOOTH_HINT, GL_NICEST )

        # Set alpha blending
        glEnable( GL_BLEND )
        glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA )

        # Set viewport
        glViewport( 0, 0, width, height )

    def on_resize(self, width, height):
        """!
        @brief pyglet callback on resize: reset the camera to show (0, 0) to the new window size.

        @note This also resets the zoomed width and height to 1:1 without resetting
              zoom_level, so panning and zooming after a resize are off by the old zoom factor.

        @param width, height New window size in pixels.
        """
        super().on_resize(width, height)
        size = self.get_size()
        self.left = 0
        self.right = size[0]
        self.bottom = 0
        self.top = size[1]
        self.zoomed_width = size[0]
        self.zoomed_height = size[1]

        # # Set window values
        # self.width  = width
        # self.height = height
        # # Initialize OpenGL context
        # self.init_gl(width, height)
        # self.width = width
        # self.height = height
        # pass

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """!
        @brief pyglet callback on mouse drag: pan the camera so the scene follows the cursor.

        @param x, y      Cursor position in pixels.
        @param dx, dy    Movement since the last event in pixels.
        @param buttons   Mouse buttons held.
        @param modifiers Keyboard modifiers held.
        """
        # Move camera
        self.left   -= dx*self.zoom_level
        self.right  -= dx*self.zoom_level
        self.bottom -= dy*self.zoom_level
        self.top    -= dy*self.zoom_level

    def on_mouse_scroll(self, x, y, dx, dy):
        """!
        @brief pyglet callback on scroll: zoom about the cursor, keeping zoom_level between 0.2 and 5.

        @param x, y   Cursor position in pixels.
        @param dx, dy Scroll amounts (dy > 0 = scroll up = zoom out).
        """
        # Get scale factor
        f = ZOOM_IN_FACTOR if dy > 0 else ZOOM_OUT_FACTOR if dy < 0 else 1
        # If zoom_level is in the proper range
        if .2 < self.zoom_level*f < 5:

            self.zoom_level *= f

            size = self.get_size()

            # cursor position as a fraction of the window
            mouse_x = x/size[0]
            mouse_y = y/size[1]

            # scene point currently under the cursor
            mouse_x_in_world = self.left   + mouse_x*self.zoomed_width
            mouse_y_in_world = self.bottom + mouse_y*self.zoomed_height

            self.zoomed_width  *= f
            self.zoomed_height *= f

            # rebuild the view so that point stays under the cursor
            self.left   = mouse_x_in_world - mouse_x*self.zoomed_width
            self.right  = mouse_x_in_world + (1 - mouse_x)*self.zoomed_width
            self.bottom = mouse_y_in_world - mouse_y*self.zoomed_height
            self.top    = mouse_y_in_world + (1 - mouse_y)*self.zoomed_height

    def on_draw(self):
        """!
        @brief Draw one frame: set the orthographic camera, then draw a 100x100 square with a colour per corner.
        """
        # Initialize Projection matrix
        glMatrixMode( GL_PROJECTION )
        glLoadIdentity()

        # Initialize Modelview matrix
        glMatrixMode( GL_MODELVIEW )
        glLoadIdentity()
        # Save the default modelview matrix
        glPushMatrix()

        # Clear window with ClearColor
        glClear( GL_COLOR_BUFFER_BIT )

        # Set orthographic projection matrix
        glOrtho( self.left, self.right, self.bottom, self.top, 1, -1 )

        # Draw quad (immediate mode), corners red, yellow, green, blue
        glBegin( GL_QUADS )
        glColor3ub( 0xFF, 0, 0 )
        glVertex2i( 10, 10 )

        glColor3ub( 0xFF, 0xFF, 0 )
        glVertex2i( 110, 10 )

        glColor3ub( 0, 0xFF, 0 )
        glVertex2i( 110, 110 )

        glColor3ub( 0, 0, 0xFF )
        glVertex2i( 10, 110 )
        glEnd()

        # Remove default modelview matrix
        glPopMatrix()

    def run(self):
        """!
        @brief Start the pyglet event loop (blocks until the window is closed).
        """
        pyglet.app.run()


App(800, 800, resizable=True).run()
