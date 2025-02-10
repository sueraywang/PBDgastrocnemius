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
    flat out vec3 FaceNormal;
    out float CollisionActive;

    uniform mat4 view;
    uniform mat4 projection;

    void main()
    {
        FragPos = aPos;
        FaceNormal = aNormal;
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
    flat in vec3 FaceNormal;
    in float CollisionActive;
        
    uniform vec3 lightPos; 
    uniform vec3 viewPos; 
    uniform vec3 lightColor;
    uniform bool useNormalColor;
    uniform bool visualizeCollision;
    uniform vec3 objectColor;

    void main()
    {
        float ambientStrength = 0.5;
        vec3 ambient = ambientStrength * lightColor;
        
        vec3 norm = normalize(vertexNormal);
        vec3 lightDir = normalize(lightPos - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * lightColor;
        
        if (visualizeCollision && CollisionActive > 0.5) {
            // Use red color for collisions
            vec3 result = (ambient + diffuse) * objectColor;
            FragColor = vec4(result, 1.0);
        } else if (useNormalColor) {
            vec3 normalColor = (normalize(FaceNormal) + 1.0) * 0.5;
            float ambientStrength = 0.3;
            vec3 ambient = ambientStrength * lightColor;
            
            vec3 norm = normalize(FaceNormal);
            vec3 lightDir = normalize(lightPos - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor;
            
            vec3 result = (ambient + diffuse) * normalColor;
            FragColor = vec4(result, 1.0);
        } else {
            float ambientStrength = 0.5;
            vec3 ambient = ambientStrength * lightColor;
            
            vec3 norm = normalize(vertexNormal);
            vec3 lightDir = normalize(lightPos - FragPos);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * lightColor;
            
            float specularStrength = 0.1;
            vec3 viewDir = normalize(viewPos - FragPos);
            vec3 reflectDir = reflect(-lightDir, norm);  
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
            vec3 specular = specularStrength * spec * lightColor;  
                
            vec3 result = (ambient + diffuse + specular) * objectColor;
            FragColor = vec4(result, 1.0);
        }
    } 
    """

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
    
    def _init_opengl(self):
        """Configure OpenGL settings"""
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glEnable(GL_LINE_SMOOTH)

    def _init_controls(self):
        """Initialize control-related variables"""
        width, height = glfw.get_window_size(self.window)
        self.fov = 45.0
        self.use_normal_coloring = False
        self.render_edges = True
        self.visualize_collisions = True

    def _init_shaders(self):
        """Compile and initialize shader programs"""
        self.shader = self._compile_shader_program(ShaderPrograms.VERTEX_SHADER, 
                                                 ShaderPrograms.FRAGMENT_SHADER)
        self.use_normal_color_loc = glGetUniformLocation(self.shader, "useNormalColor")
        self.visualize_collisions_loc = glGetUniformLocation(self.shader, "visualizeCollision")

    def _compile_shader_program(self, vertex_source, fragment_source):
        """Compile shader program from vertex and fragment shader sources"""
        vert_shader = shaders.compileShader(vertex_source, GL_VERTEX_SHADER)
        frag_shader = shaders.compileShader(fragment_source, GL_FRAGMENT_SHADER)
        program = shaders.compileProgram(vert_shader, frag_shader)
        return program

    def render(self):
        """Main render function"""
        glClearColor(0.2, 0.3, 0.3, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        self._setup_matrices_and_lighting()
        glUseProgram(self.shader)
        self._render_meshes()
        
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

    def _render_meshes(self):
        """Render all meshes in the scene"""
        for mesh in self.meshes:
            glBindVertexArray(mesh.vao)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            if self.use_normal_coloring:
                # Normal visualization mode
                glUniform1i(self.use_normal_color_loc, True)
                glDrawElements(GL_TRIANGLES, mesh.num_indices, GL_UNSIGNED_INT, None)
                # Disable normal coloring for future rendering
                glUniform1i(self.use_normal_color_loc, False)

            else:
                # Original rendering mode
                glUniform3fv(glGetUniformLocation(self.shader, "objectColor"), 1, mesh.color)
                glDrawElements(GL_TRIANGLES, mesh.num_indices, GL_UNSIGNED_INT, None)
            
            if self.visualize_collisions:
                glUniform1i(self.visualize_collisions_loc, True)
            
            if self.render_edges:    
                # Draw edges in black
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                glUniform3fv(glGetUniformLocation(self.shader, "objectColor"), 1, np.array([0.0, 0.0, 0.0]))
                glDrawElements(GL_TRIANGLES, mesh.num_indices, GL_UNSIGNED_INT, None)

    # Mesh management methods
    def add_mesh(self, mesh):
        self.meshes.append(mesh)

    def update_meshes(self):
        for mesh in self.meshes:
            mesh.update_mesh()

    def cleanup(self):
        """Clean up OpenGL resources"""
        
        # Clean up all mesh resources
        for mesh in self.meshes:
            glDeleteVertexArrays(1, [mesh.vao])
            glDeleteBuffers(1, [mesh.vbo])
            glDeleteBuffers(1, [mesh.ebo])
        
        # Delete shader program
        glDeleteProgram(self.shader)
        
        # Terminate GLFW
        glfw.terminate()

    def should_close(self):
        """Check if window should close"""
        return glfw.window_should_close(self.window)