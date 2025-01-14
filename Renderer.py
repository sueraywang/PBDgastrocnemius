import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
import pyrr

# Vertex and fragment shaders remain the same
vertex_shader = """
#version 330 core
layout (location = 0) in vec3 position;

out vec3 FragPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    FragPos = vec3(model * vec4(position, 1.0));
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
"""

fragment_shader = """
#version 330 core
out vec4 FragColor;

in vec3 FragPos;

uniform vec3 objectColor;

void main()
{
    vec3 result = objectColor;
    FragColor = vec4(result, 1.0);
}
"""

class Renderer:
    def __init__(self, vertices, faces, width=800, height=600):
        # Initialize GLFW
        if not glfw.init():
            raise Exception("GLFW initialization failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

        # Create window
        self.window = glfw.create_window(width, height, "XPBD Cylinder Simulation", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Window creation failed")

        # Set context
        glfw.make_context_current(self.window)

        # Set callbacks
        glfw.set_framebuffer_size_callback(self.window, self.framebuffer_size_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)
        glfw.set_cursor_pos_callback(self.window, self.mouse_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)

        # OpenGL configuration
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        
        # Wireframe mode flag
        self.wireframe_mode = True
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        # Orbital camera parameters
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.camera_distance = 5.0  # Distance from target
        self.theta = np.pi / 2  # Horizontal angle (azimuth), starts on Y axis
        self.phi = np.pi / 2    # Vertical angle (polar), starts in XY plane
        self.camera_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.convert_to_cartesian()

        # Mouse control parameters
        self.last_x = width / 2
        self.last_y = height / 2
        self.first_mouse = True
        self.fov = 45.0
        self.mouse_sensitivity = 0.005
        self.pan_sensitivity = 0.01
        self.right_mouse_pressed = False
        self.middle_mouse_pressed = False

        # Compile shaders
        self.shader = self.compile_shader_program()
        self.model_loc = glGetUniformLocation(self.shader, "model")
        self.view_loc = glGetUniformLocation(self.shader, "view")
        self.projection_loc = glGetUniformLocation(self.shader, "projection")
        self.object_color_loc = glGetUniformLocation(self.shader, "objectColor")

        # Load mesh data
        self.setup_mesh(vertices, faces)

    def convert_to_cartesian(self):
        x = self.camera_distance * np.sin(self.phi) * np.cos(self.theta)
        y = self.camera_distance * np.sin(self.phi) * np.sin(self.theta)
        z = self.camera_distance * np.cos(self.phi)
        self.camera_pos = self.target + np.array([x, y, z], dtype=np.float32)
        self.camera_front = (self.target - self.camera_pos) / np.linalg.norm(self.target - self.camera_pos)

    def mouse_button_callback(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_RIGHT:
            self.right_mouse_pressed = action == glfw.PRESS
            if action == glfw.PRESS:
                self.first_mouse = True
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self.middle_mouse_pressed = action == glfw.PRESS
            if action == glfw.PRESS:
                self.first_mouse = True

    def mouse_callback(self, window, xpos, ypos):
        if self.first_mouse:
            self.last_x = xpos
            self.last_y = ypos
            self.first_mouse = False
            return

        xoffset = xpos - self.last_x
        yoffset = ypos - self.last_y

        # Store current position for next frame
        self.last_x = xpos
        self.last_y = ypos

        if self.right_mouse_pressed:
            self.theta -= xoffset * self.mouse_sensitivity
            self.phi -= yoffset * self.mouse_sensitivity
            
            # Clamp phi to avoid gimbal lock
            self.phi = np.clip(self.phi, 0.1, np.pi - 0.1)
            
            # Update camera position based on new angles
            self.convert_to_cartesian()
            
        elif self.middle_mouse_pressed:  # Pan
            # Calculate right and up vectors in world space
            right = np.cross(self.camera_front, self.camera_up)
            right = right / np.linalg.norm(right)
            up = self.camera_up/np.linalg.norm(self.camera_up)
            
            # Update target position (panning)
            pan_x = -xoffset * self.pan_sensitivity
            pan_y = +yoffset * self.pan_sensitivity
            
            self.target += right * pan_x + up * pan_y
            self.convert_to_cartesian()

    def scroll_callback(self, window, xoffset, yoffset):
        # Update camera distance (zoom)
        self.camera_distance -= yoffset * 0.5
        self.camera_distance = max(2.0, min(50.0, self.camera_distance))
        self.convert_to_cartesian()

    def framebuffer_size_callback(self, window, width, height):
        glViewport(0, 0, width, height)

    def compile_shader_program(self):
        vert_shader = shaders.compileShader(vertex_shader, GL_VERTEX_SHADER)
        frag_shader = shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER)
        shader_program = shaders.compileProgram(vert_shader, frag_shader)
        return shader_program

    def setup_mesh(self, vertices, faces):
        """Initial setup of mesh data and OpenGL buffers"""
        vertex_data = np.array(vertices, dtype=np.float32)
        indices = faces.flatten().astype(np.uint32)
        self.num_indices = len(indices)

        # Create and bind VAO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # Create and bind VBO
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_DYNAMIC_DRAW)  # Note: GL_DYNAMIC_DRAW

        # Create and bind EBO
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        # Set vertex attributes
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

    def update_vertex_positions(self, vertices):
        """Update vertex positions in the VBO"""
        vertex_data = np.array(vertices, dtype=np.float32)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_DYNAMIC_DRAW)
    
    def render(self):

        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(self.shader)

        model = pyrr.matrix44.create_identity()
        glUniformMatrix4fv(self.model_loc, 1, GL_FALSE, model)

        view = pyrr.matrix44.create_look_at(
            self.camera_pos,
            self.camera_pos + self.camera_front,
            self.camera_up
        )
        glUniformMatrix4fv(self.view_loc, 1, GL_FALSE, view)

        width, height = glfw.get_window_size(self.window)
        projection = pyrr.matrix44.create_perspective_projection_matrix(
            self.fov, width/height, 0.1, 100.0
        )
        glUniformMatrix4fv(self.projection_loc, 1, GL_FALSE, projection)

        object_color = np.array([1.0, 1.0, 1.0])
        glUniform3fv(self.object_color_loc, 1, object_color)

        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.num_indices, GL_UNSIGNED_INT, None)

        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def should_close(self):
        return glfw.window_should_close(self.window)

    def cleanup(self):
        glfw.terminate()