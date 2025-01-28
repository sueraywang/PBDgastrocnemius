import glfw
from OpenGL.GL import shaders
import pyrr
from Mesh import *

# Main shader code remains the same
vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

out vec3 FragPos;
out vec3 Normal;

uniform mat4 view;
uniform mat4 projection;

void main()
{
    FragPos = aPos;
    Normal = aNormal;  
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
"""

fragment_shader = """
#version 330 core
out vec4 FragColor;

in vec3 Normal;  
in vec3 FragPos;  
  
uniform vec3 lightPos; 
uniform vec3 viewPos; 
uniform vec3 lightColor;
uniform vec3 objectColor;

void main()
{
    // ambient
    float ambientStrength = 0.3;
    vec3 ambient = ambientStrength * lightColor;
  	
    // diffuse 
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    // specular
    float specularStrength = 0.1;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;  
        
    vec3 result = (ambient + diffuse + specular) * objectColor;
    FragColor = vec4(result, 1.0);
} 
"""

# Simplified coordinate axes shaders
axes_vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

out vec3 vertexColor;

uniform mat4 view;
uniform mat4 projection;

void main()
{
    gl_Position = projection * view * vec4(aPos, 1.0);
    vertexColor = aColor;
}
"""

axes_fragment_shader = """
#version 330 core
in vec3 vertexColor;
out vec4 FragColor;

void main()
{
    FragColor = vec4(vertexColor, 1.0);
}
"""

class Renderer:
    def __init__(self, width=800, height=600, cameraRadius = 5.0, lookAtPosition = np.array([0.0, 0.0, 0.0]), 
                 h_angle=-np.pi/2, v_angle=np.pi/2, dtype=np.float32):
        # Initialize GLFW
        if not glfw.init():
            raise Exception("GLFW initialization failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

        # Ground plane settings
        self.ground_size = 10.0  # Size of the ground plane
        self.ground_color = np.array([0.8, 0.8, 0.8], dtype=np.float32)  # Light gray

        # Create window
        self.window = glfw.create_window(width, height, "Multi-Mesh Renderer", None, None)
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
        glEnable(GL_LINE_SMOOTH)

        # Camera setup
        self.target = lookAtPosition
        self.camera_distance = cameraRadius
        self.theta = h_angle # Horizontal
        self.phi = v_angle
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
        self.shader = self.compile_shader_program(vertex_shader, fragment_shader)
        self.axes_shader = self.compile_shader_program(axes_vertex_shader, axes_fragment_shader)

        # Initialize mesh collection
        self.meshes = []
        
        # Setup ground plane
        self.setup_ground_plane()
        
        # Setup coordinate axes
        self.setup_coordinate_axes()

    def setup_ground_plane(self):
        """Setup the ground plane VAO and buffers with a checkered pattern"""
        s = self.ground_size
        tiles = 20  # Number of tiles in each direction
        tile_size = (2 * s) / tiles
        vertices = []
        indices = []
        vertex_count = 0

        # Create vertices and indices for each tile
        for i in range(tiles):
            for j in range(tiles):
                # Calculate tile corners
                x1 = -s + i * tile_size
                x2 = x1 + tile_size
                y1 = -s + j * tile_size
                y2 = y1 + tile_size
                
                # Add vertices for this tile (position and normal)
                vertices.extend([
                    x1, y1, 0.0,  0.0, 0.0, 1.0,  # Bottom-left
                    x2, y1, 0.0,  0.0, 0.0, 1.0,  # Bottom-right
                    x2, y2, 0.0,  0.0, 0.0, 1.0,  # Top-right
                    x1, y2, 0.0,  0.0, 0.0, 1.0,  # Top-left
                ])

                # Add indices for this tile
                indices.extend([
                    vertex_count, vertex_count + 1, vertex_count + 2,  # First triangle
                    vertex_count + 2, vertex_count + 3, vertex_count   # Second triangle
                ])
                vertex_count += 4

        vertices = np.array(vertices, dtype=np.float32)
        indices = np.array(indices, dtype=np.uint32)

        # Store the number of tiles for rendering
        self.ground_tiles = tiles
        self.ground_indices_count = len(indices)

        # Create and bind VAO
        self.ground_vao = glGenVertexArrays(1)
        glBindVertexArray(self.ground_vao)

        # Create and bind VBO
        self.ground_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.ground_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

        # Create and bind EBO
        self.ground_ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ground_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        # Set vertex attributes
        # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # Normal attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)

        # Unbind VAO
        glBindVertexArray(0)

    def add_mesh(self, mesh):
        self.meshes.append(mesh)

    def update_meshes(self):
        for mesh in self.meshes:
            mesh.update_mesh()

    def render(self):
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Main scene rendering with regular shader
        glUseProgram(self.shader)

        # Set view and projection matrices
        view = pyrr.matrix44.create_look_at(
            self.camera_pos,
            self.camera_pos + self.camera_front,
            self.camera_up
        )
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 1, GL_FALSE, view)
        width, height = glfw.get_window_size(self.window)
        projection = pyrr.matrix44.create_perspective_projection_matrix(
            self.fov, width / height, 0.1, 100.0
        )
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "projection"), 1, GL_FALSE, projection)

        # Set lighting uniforms
        glUniform3fv(glGetUniformLocation(self.shader, "lightPos"), 1, np.array([0.0, 0.0, 5.0]))
        glUniform3fv(glGetUniformLocation(self.shader, "viewPos"), 1, self.camera_pos)
        glUniform3fv(glGetUniformLocation(self.shader, "lightColor"), 1, np.array([1.0, 1.0, 1.0]))

        # Render the ground plane
        glBindVertexArray(self.ground_vao)
        # Draw ground faces with alternating colors
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        for i in range(self.ground_tiles):
            for j in range(self.ground_tiles):
                # Calculate the index for this tile
                tile_index = (i * self.ground_tiles + j) * 6  # 6 indices per tile
                
                # Set color based on checkerboard pattern
                if (i + j) % 2 == 0:
                    color = np.array([0.7, 0.7, 0.7], dtype=np.float32)  # White
                else:
                    color = np.array([0.3, 0.3, 0.3], dtype=np.float32)  # Dark grey
                
                glUniform3fv(glGetUniformLocation(self.shader, "objectColor"), 1, color)
                glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, ctypes.c_void_p(tile_index * 4))

        # Render meshes
        for mesh in self.meshes:
            glUniform3fv(glGetUniformLocation(self.shader, "objectColor"), 1, mesh.color)
            glBindVertexArray(mesh.vao)
            
            # Draw faces
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glDrawElements(GL_TRIANGLES, mesh.num_indices, GL_UNSIGNED_INT, None)

            # Draw edges
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glUniform3fv(glGetUniformLocation(self.shader, "objectColor"), 1, np.array([0.0, 0.0, 0.0]))
            glDrawElements(GL_TRIANGLES, mesh.num_indices, GL_UNSIGNED_INT, None)

        # Render coordinate axes
        glUseProgram(self.axes_shader)
        
        # Set view and projection matrices for axes
        glUniformMatrix4fv(glGetUniformLocation(self.axes_shader, "view"), 1, GL_FALSE, view)
        glUniformMatrix4fv(glGetUniformLocation(self.axes_shader, "projection"), 1, GL_FALSE, projection)
        
        # Draw axes
        glBindVertexArray(self.axes_vao)
        glDrawArrays(GL_LINES, 0, 6)  # Draw 3 lines (2 vertices each)

        glfw.swap_buffers(self.window)
        glfw.poll_events()

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
        self.last_x = xpos
        self.last_y = ypos

        if self.right_mouse_pressed:
            self.theta -= xoffset * self.mouse_sensitivity
            self.phi -= yoffset * self.mouse_sensitivity
            self.phi = np.clip(self.phi, 0.1, np.pi - 0.1)
            self.convert_to_cartesian()
            
        elif self.middle_mouse_pressed:
            right = np.cross(self.camera_front, self.camera_up)
            right = right / np.linalg.norm(right)
            up = self.camera_up/np.linalg.norm(self.camera_up)
            
            pan_x = -xoffset * self.pan_sensitivity
            pan_y = +yoffset * self.pan_sensitivity
            
            self.target += right * pan_x + up * pan_y
            self.convert_to_cartesian()

    def scroll_callback(self, window, xoffset, yoffset):
        self.camera_distance -= yoffset * 0.5
        self.camera_distance = max(0.5, min(50.0, self.camera_distance))
        self.convert_to_cartesian()

    def framebuffer_size_callback(self, window, width, height):
        glViewport(0, 0, width, height)

    def compile_shader_program(self, vertex_source, fragment_source):
        vert_shader = shaders.compileShader(vertex_source, GL_VERTEX_SHADER)
        frag_shader = shaders.compileShader(fragment_source, GL_FRAGMENT_SHADER)
        program = shaders.compileProgram(vert_shader, frag_shader)
        return program

    def should_close(self):
        return glfw.window_should_close(self.window)

    def cleanup(self):
        # Clean up ground plane resources
        glDeleteVertexArrays(1, [self.ground_vao])
        glDeleteBuffers(1, [self.ground_vbo])
        glDeleteBuffers(1, [self.ground_ebo])
        
        # Delete axes resources
        glDeleteVertexArrays(1, [self.axes_vao])
        for vbo in self.axes_vbos:
            glDeleteBuffers(1, [vbo])
        glDeleteProgram(self.axes_shader)
        
        # Clean up all mesh resources
        for mesh in self.meshes:
            glDeleteVertexArrays(1, [mesh.vao])
            glDeleteBuffers(1, [mesh.vbo])
            glDeleteBuffers(1, [mesh.ebo])
        
        # Delete shader program
        glDeleteProgram(self.shader)
        
        # Terminate GLFW
        glfw.terminate()

    def setup_coordinate_axes(self):
        """Setup the coordinate axes VAO and buffers"""
        # Create vertices for three orthogonal lines
        vertices = np.array([
            # X axis line (red)
            0.0, 10.0, 0.0,
            1.0, 10.0, 0.0,
            # Y axis line (green)
            0.0, 10.0, 0.0,
            0.0, 11.0, 0.0,
            # Z axis line (blue)
            0.0, 10.0, 0.0,
            0.0, 10.0, 1.0
        ], dtype=np.float32)
        
        colors = np.array([
            # X axis color (red)
            1.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            # Y axis color (green)
            0.0, 1.0, 0.0,
            0.0, 1.0, 0.0,
            # Z axis color (blue)
            0.0, 0.0, 1.0,
            0.0, 0.0, 1.0
        ], dtype=np.float32)

        # Create and bind VAO
        self.axes_vao = glGenVertexArrays(1)
        glBindVertexArray(self.axes_vao)

        # Create and bind VBO for vertices
        vbo_vertices = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        # Create and bind VBO for colors
        vbo_colors = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_colors)
        glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

        # Unbind VAO
        glBindVertexArray(0)

        # Store VBOs for cleanup
        self.axes_vbos = [vbo_vertices, vbo_colors]