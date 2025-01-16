import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
import pyrr

# Updated vertex shader with normal support
vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

out vec3 FragPos;
out vec3 Normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = mat3(transpose(inverse(model))) * aNormal;  
    
    gl_Position = projection * view * vec4(FragPos, 1.0);
}
"""

# Updated fragment shader with basic lighting
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
        glDisable(GL_CULL_FACE)  # Enable face culling
        glEnable(GL_LINE_SMOOTH)

        # Camera setup (same as before)
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.camera_distance = 5.0
        self.theta = np.pi / 2
        self.phi = np.pi / 3
        self.camera_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.convert_to_cartesian()

        # Mouse control parameters (same as before)
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

        # Calculate normals and setup mesh
        self.calculate_normals(vertices, faces)
        self.setup_mesh(vertices, faces)

    def calculate_normals(self, vertices, surface_faces):
        face_normals = np.zeros((len(surface_faces), 3))
        vertex_normals = np.zeros((len(vertices), 3))
        vertex_counts = np.zeros(len(vertices))
        
        for i, face in enumerate(surface_faces):
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            
            length = np.linalg.norm(normal)
            if length > 0:
                normal = normal / length
                
            # Make normal point outward
            face_center = (v0 + v1 + v2) / 3
            to_center = np.zeros(3) - face_center
            if np.dot(normal, to_center) > 0:
                normal = -normal
                
            face_normals[i] = normal
            
            # Accumulate normals at vertices
            for vertex_idx in face:
                vertex_normals[vertex_idx] += normal
                vertex_counts[vertex_idx] += 1
        
        # Average and normalize vertex normals
        for i in range(len(vertex_normals)):
            if vertex_counts[i] > 0:
                vertex_normals[i] = vertex_normals[i] / vertex_counts[i]
                length = np.linalg.norm(vertex_normals[i])
                if length > 0:
                    vertex_normals[i] = vertex_normals[i] / length
                    
        self.normals = vertex_normals

    def setup_mesh(self, vertices, faces):
        """Setup mesh with vertices and normals"""
        # Combine vertex positions and normals
        vertex_data = np.zeros((len(vertices), 6), dtype=np.float32)
        vertex_data[:, 0:3] = vertices
        vertex_data[:, 3:6] = self.normals
        
        indices = faces.flatten().astype(np.uint32)
        self.num_indices = len(indices)

        # Create and bind VAO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # Create and bind VBO
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_DYNAMIC_DRAW)

        # Create and bind EBO
        self.ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

        # Set vertex attributes
        # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        # Normal attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)

    def update_vertex_positions(self, vertices, faces):
        # Recalculate normals for the new vertex positions
        self.calculate_normals(vertices, faces)
        
        # Combine new positions with updated normals
        vertex_data = np.zeros((len(vertices), 6), dtype=np.float32)
        vertex_data[:, 0:3] = vertices
        vertex_data[:, 3:6] = self.normals
        
        # Update buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_DYNAMIC_DRAW)

    def render(self):
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(self.shader)

        # Set uniforms

        model = pyrr.matrix44.create_identity()
        glUniformMatrix4fv(glGetUniformLocation(self.shader, "model"), 1, GL_FALSE, model)

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
        glUniform3fv(glGetUniformLocation(self.shader, "lightPos"), 1, np.array([0.0, 5.0, 5.0]))
        glUniform3fv(glGetUniformLocation(self.shader, "viewPos"), 1, self.camera_pos)
        glUniform3fv(glGetUniformLocation(self.shader, "objectColor"), 1, np.array([0.75, 0.5, 0.1]))
        glUniform3fv(glGetUniformLocation(self.shader, "lightColor"), 1, np.array([1.0, 1.0, 1.0]))

        # Draw faces
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, self.num_indices, GL_UNSIGNED_INT, None)

        # Draw edges
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glUniform3fv(glGetUniformLocation(self.shader, "objectColor"), 1, np.array([0.0, 0.0, 0.0]))
        glDrawElements(GL_TRIANGLES, self.num_indices, GL_UNSIGNED_INT, None)

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

    def should_close(self):
        return glfw.window_should_close(self.window)

    def cleanup(self):
        glfw.terminate()