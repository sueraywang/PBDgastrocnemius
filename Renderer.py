import glfw
from OpenGL.GL import shaders
import pyrr
from Mesh import *

class ShaderPrograms:
    VERTEX_SHADER = """
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;
    layout (location = 2) in float aCollisionActive;

    out vec3 FragPos;
    out vec3 vertexNormal;
    flat out float CollisionActive;

    uniform mat4 view;
    uniform mat4 projection;

    void main()
    {
        FragPos = aPos;
        vertexNormal = aNormal;
        CollisionActive = aCollisionActive;
        gl_Position = projection * view * vec4(FragPos, 1.0);
    }
    """

    FRAGMENT_SHADER = """
    #version 330 core
    out vec4 FragColor;

    in vec3 FragPos;
    in vec3 vertexNormal;    
    flat in float CollisionActive;
        
    uniform vec3 lightPos; 
    uniform vec3 viewPos; 
    uniform vec3 lightColor;
    uniform bool useNormalColor;
    uniform bool visualizeCollision;
    uniform vec3 objectColor;
    uniform bool isPointPass;

    void main()
    {
        if (isPointPass) {
            // For point rendering pass, only output red for colliding vertices
            if (CollisionActive > 0.5) {
                FragColor = vec4(1.0, 0.0, 0.0, 1.0);
            } else {
                discard; // Don't render non-colliding points
            }
            return;
        }

        // Regular rendering for triangles
        float ambientStrength = 0.5;
        vec3 ambient = ambientStrength * lightColor;
        
        vec3 norm = normalize(vertexNormal);
        vec3 lightDir = normalize(lightPos - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * lightColor;
        
        vec3 baseColor;
        if (useNormalColor) {
            baseColor = (normalize(vertexNormal) + 1.0) * 0.5;
        } else {
            baseColor = objectColor;
        }
        
        float specularStrength = 0.1;
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 reflectDir = reflect(-lightDir, norm);  
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
        vec3 specular = specularStrength * spec * lightColor;
        
        vec3 result = (ambient + diffuse + specular) * baseColor;
        FragColor = vec4(result, 1.0);
    } 
    """

    AXES_VERTEX_SHADER = """
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

    AXES_FRAGMENT_SHADER = """
    #version 330 core
    in vec3 vertexColor;
    out vec4 FragColor;

    void main()
    {
        FragColor = vec4(vertexColor, 1.0);
    }
    """

class CameraController:
    def __init__(self, initial_radius=1.0, initial_target=np.array([0.0, 0.0, 0.0]),
                 initial_h_angle=-np.pi/2, initial_v_angle=np.pi/2):
        self.target = initial_target
        self.camera_distance = initial_radius
        self.theta = initial_h_angle  # Horizontal angle
        self.phi = initial_v_angle    # Vertical angle
        self.camera_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.camera_pos = None
        self.camera_front = None
        self.spherical_to_cartesian()

    def spherical_to_cartesian(self):
        """Convert spherical coordinates to Cartesian"""
        x = self.camera_distance * np.sin(self.phi) * np.cos(self.theta)
        y = self.camera_distance * np.sin(self.phi) * np.sin(self.theta)
        z = self.camera_distance * np.cos(self.phi)
        self.camera_pos = self.target + np.array([x, y, z], dtype=np.float32)
        self.camera_front = (self.target - self.camera_pos) / np.linalg.norm(self.target - self.camera_pos)

    def get_view_matrix(self):
        """Returns the view matrix for the current camera position"""
        return pyrr.matrix44.create_look_at(
            self.camera_pos,
            self.camera_pos + self.camera_front,
            self.camera_up
        )

class Renderer:
    def __init__(self, width=1000, height=800, title="Neo-Hookean Cylinders", cameraRadius=1.0, lookAtPosition=np.array([0.0, 0.0, 0.0]),
                 h_angle=-np.pi/2, v_angle=np.pi/2):
        """Initialize renderer"""
        self._init_glfw(width, height, title)
        self._init_opengl()
        self._init_controls()
        self._init_shaders()
        
        """Initialize scene elements"""
        self.meshes = []
        self.ground_size = 10.0
        self.ground_color = np.array([0.8, 0.8, 0.8], dtype=np.float32)
        self.setup_ground_plane()
        self.setup_coordinate_axes()
        self.camera = CameraController(initial_radius=cameraRadius, initial_target=lookAtPosition,
                 initial_h_angle=h_angle, initial_v_angle=v_angle)

    def _init_glfw(self, width, height, title):
        """Initialize GLFW and create window"""
        if not glfw.init():
            raise Exception("GLFW initialization failed")
            
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("Window creation failed")

        glfw.make_context_current(self.window)
        self._setup_callbacks()
    
    def _init_opengl(self):
        """Configure OpenGL settings"""
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glEnable(GL_LINE_SMOOTH)
    
    def _setup_callbacks(self):
        """Set up GLFW callback functions"""
        glfw.set_framebuffer_size_callback(self.window, self.framebuffer_size_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)
        glfw.set_cursor_pos_callback(self.window, self.mouse_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_key_callback(self.window, self.key_callback)

    def _init_controls(self):
        """Initialize control-related variables"""
        width, height = glfw.get_window_size(self.window)
        self.last_x = width / 2
        self.last_y = height / 2
        self.first_mouse = True
        self.fov = 45.0
        self.mouse_sensitivity = 0.005
        self.pan_sensitivity = 0.001
        self.right_mouse_pressed = False
        self.middle_mouse_pressed = False
        self.use_normal_coloring = False
        self.render_edges = True
        self.visualize_collisions = True

    def _init_shaders(self):
        """Compile and initialize shader programs"""
        self.shader = self._compile_shader_program(ShaderPrograms.VERTEX_SHADER, 
                                                 ShaderPrograms.FRAGMENT_SHADER)
        self.axes_shader = self._compile_shader_program(ShaderPrograms.AXES_VERTEX_SHADER, 
                                                      ShaderPrograms.AXES_FRAGMENT_SHADER)
        self.use_normal_color_loc = glGetUniformLocation(self.shader, "useNormalColor")
        self.visualize_collisions_loc = glGetUniformLocation(self.shader, "visualizeCollision")
        self.is_point_pass_loc = glGetUniformLocation(self.shader, "isPointPass")

    def _compile_shader_program(self, vertex_source, fragment_source):
        """Compile shader program from vertex and fragment shader sources"""
        vert_shader = shaders.compileShader(vertex_source, GL_VERTEX_SHADER)
        frag_shader = shaders.compileShader(fragment_source, GL_FRAGMENT_SHADER)
        program = shaders.compileProgram(vert_shader, frag_shader)
        return program

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

    def render(self):
        """Main render function"""
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        self._setup_matrices_and_lighting()
        glUseProgram(self.shader)
        self._render_ground()
        self._render_meshes()
        self._render_axes()
        
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def _setup_matrices_and_lighting(self):
        """Setup view, projection matrices and lighting uniforms"""
        view = self.camera.get_view_matrix()
        width, height = glfw.get_window_size(self.window)
        projection = pyrr.matrix44.create_perspective_projection_matrix(
            self.fov, width / height, 0.1, 100.0
        )
        
        # Set matrices and lighting for main shader
        glUseProgram(self.shader)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "view"), 1, GL_FALSE, view)
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "projection"), 1, GL_FALSE, projection)
        
        # Set lighting uniforms
        glUniform3fv(glGetUniformLocation(self.shader, "lightPos"), 1, np.array([5.0, 5.0, 5.0], dtype=np.float32))
        glUniform3fv(glGetUniformLocation(self.shader, "viewPos"), 1, self.camera.camera_pos)
        glUniform3fv(glGetUniformLocation(self.shader, "lightColor"), 1, np.array([1.0, 1.0, 1.0], dtype=np.float32))
        
        # Set matrices for axes shader
        glUseProgram(self.axes_shader)
        glUniformMatrix4fv(glGetUniformLocation(self.axes_shader, "view"), 1, GL_FALSE, view)
        glUniformMatrix4fv(glGetUniformLocation(self.axes_shader, "projection"), 1, GL_FALSE, projection)

    def _render_ground(self):
        """Render the ground plane"""
        glUniform1i(self.use_normal_color_loc, False)
        glUniform1i(self.visualize_collisions_loc, False)
        glBindVertexArray(self.ground_vao)
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

    def _render_meshes(self):
        """Render all meshes in the scene"""
        for mesh in self.meshes:
            glBindVertexArray(mesh.vao)
            
            # First pass: Regular triangle rendering
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glUniform1i(self.is_point_pass_loc, 0)
            
            if self.use_normal_coloring:
                glUniform1i(self.use_normal_color_loc, 1)
                glDrawElements(GL_TRIANGLES, mesh.num_indices, GL_UNSIGNED_INT, None)
                glUniform1i(self.use_normal_color_loc, 0)
            else:
                glUniform3fv(glGetUniformLocation(self.shader, "objectColor"), 1, mesh.color)
                glDrawElements(GL_TRIANGLES, mesh.num_indices, GL_UNSIGNED_INT, None)
            
            # Second pass: Edge rendering
            if self.render_edges:    
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                glUniform3fv(glGetUniformLocation(self.shader, "objectColor"), 1, np.array([0.0, 0.0, 0.0]))
                glDrawElements(GL_TRIANGLES, mesh.num_indices, GL_UNSIGNED_INT, None)
            
            # Third pass: Collision point rendering
            if self.visualize_collisions:
                glPointSize(10.0)  # Set point size
                glPolygonMode(GL_FRONT_AND_BACK, GL_POINT)
                glEnable(GL_PROGRAM_POINT_SIZE)
                glUniform1i(self.is_point_pass_loc, 1)
                
                # Draw all vertices - shader will discard non-colliding ones
                glDrawArrays(GL_POINTS, 0, len(mesh.vertices))
                
                glDisable(GL_PROGRAM_POINT_SIZE)
                glUniform1i(self.is_point_pass_loc, 0)
            
            # Reset states
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    def _render_axes(self):
        """Render coordinate axes"""
        glUseProgram(self.axes_shader)
        glBindVertexArray(self.axes_vao)
        glDrawArrays(GL_LINES, 0, 6)

    # Event callbacks
    def key_callback(self, window, key, scancode, action, mods):
        if key == glfw.KEY_N and action == glfw.PRESS:
            self.use_normal_coloring = not self.use_normal_coloring
        if key == glfw.KEY_C and action == glfw.PRESS:
            self.visualize_collisions = not self.visualize_collisions
        if key == glfw.KEY_E and action == glfw.PRESS:
            self.render_edges = not self.render_edges

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
            self.camera.theta -= xoffset * self.mouse_sensitivity
            self.camera.phi -= yoffset * self.mouse_sensitivity
            self.camera.phi = np.clip(self.camera.phi, 0.1, np.pi - 0.1)
            self.camera.spherical_to_cartesian()
            
        elif self.middle_mouse_pressed:
            right = np.cross(self.camera.camera_front, self.camera.camera_up)
            right = right / np.linalg.norm(right)
            up = self.camera.camera_up/np.linalg.norm(self.camera.camera_up)
            
            pan_x = -xoffset * self.pan_sensitivity
            pan_y = +yoffset * self.pan_sensitivity
            
            self.camera.target += right * pan_x + up * pan_y
            self.camera.spherical_to_cartesian()

    def scroll_callback(self, window, xoffset, yoffset):
        self.camera.camera_distance -= yoffset * 0.05
        self.camera.camera_distance = max(0.1, min(10.0, self.camera.camera_distance))
        self.camera.spherical_to_cartesian()

    def framebuffer_size_callback(self, window, width, height):
        glViewport(0, 0, width, height)

    # Mesh management methods
    def add_mesh(self, mesh):
        self.meshes.append(mesh)

    def update_meshes(self):
        for mesh in self.meshes:
            mesh.update_mesh()

    def cleanup(self):
        """Clean up OpenGL resources"""
        # Clean up ground plane resources
        glDeleteVertexArrays(1, [self.ground_vao])
        glDeleteBuffers(1, [self.ground_vbo])
        glDeleteBuffers(1, [self.ground_ebo])
        
        # Clean up all mesh resources
        for mesh in self.meshes:
            glDeleteVertexArrays(1, [mesh.vao])
            glDeleteBuffers(1, [mesh.vbo])
            glDeleteBuffers(1, [mesh.ebo])
        
        # Clean up axes resources
        glDeleteVertexArrays(1, [self.axes_vao])
        for vbo in self.axes_vbos:
            glDeleteBuffers(1, [vbo])
        
        # Delete shader program
        glDeleteProgram(self.shader)
        glDeleteProgram(self.axes_shader)
        
        # Terminate GLFW
        glfw.terminate()

    def should_close(self):
        """Check if window should close"""
        return glfw.window_should_close(self.window)