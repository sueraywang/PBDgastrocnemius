import numpy as np
from OpenGL.GL import *

class Mesh:
    def __init__(self, center, vertices, faces, tets, color=np.array([0.75, 0.5, 0.1])):
        self.vertices = vertices + center
        self.faces = faces
        self.tets = tets
        self.color = color
        self.center = center # geometric center of mesh in world space
        self.face_normals = self.calculate_face_normals()
        self.vertex_normals = self.calculate_vertex_normals()
        
        self.collision_active = np.zeros(len(vertices), dtype=np.float32)

        self.setup_mesh()
        self.setup_collision_visualization()
        #self.setup_face_normals_visualization()

    def calculate_face_normals(self):  
        face_normals = []
        for face in self.faces:
            v0, v1, v2 = self.vertices[face[0]], self.vertices[face[1]], self.vertices[face[2]]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            length = np.linalg.norm(normal)
            if length > 0:
                normal = normal / length
                
            # Make normal point outward
            face_center = (v0 + v1 + v2) / 3
            face_center_local = face_center - self.center
            if np.dot(normal, face_center_local) < 0:
                normal = -normal
            face_normals.append(normal)
        
        return np.array(face_normals)
    
    def calculate_vertex_normals(self):
        vertex_normals = np.zeros((len(self.vertices), 3))
        vertex_counts = np.zeros(len(self.vertices))
        
        for i, face in enumerate(self.faces):
            normal = self.face_normals[i]
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
                    
        return vertex_normals
    
    def setup_collision_visualization(self):
        self.collision_active_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.collision_active_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.collision_active.nbytes, self.collision_active, GL_DYNAMIC_DRAW)
        
        glBindVertexArray(self.vao)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(2)

    def setup_mesh(self):
        # Combine vertex positions and normals
        vertex_data = np.zeros((len(self.vertices), 6), dtype=np.float32)
        vertex_data[:, 0:3] = self.vertices
        vertex_data[:, 3:6] = self.vertex_normals
        
        indices = self.faces.flatten().astype(np.uint32)
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
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)

    def update_collision_visualization(self, colliding_vertices, colliding_faces, vertex_offset, face_offset):
        self.collision_active.fill(0.0)
        
        # Filter vertices that belong to this mesh
        mesh_colliding_vertices = []
        for v_idx in colliding_vertices:
            if v_idx >= vertex_offset and v_idx < vertex_offset + len(self.vertices):
                mesh_colliding_vertices.append(v_idx - vertex_offset)
        
        # Filter faces that belong to this mesh
        mesh_colliding_faces = []
        if len(mesh_colliding_faces) > 0:
            for f_idx in colliding_faces:
                if f_idx >= face_offset and f_idx < face_offset + len(self.faces):
                    mesh_colliding_faces.append(f_idx - face_offset)

        # Set active flag for colliding vertices and faces
        if mesh_colliding_vertices:
            self.collision_active[mesh_colliding_vertices] = 1.0
            
        #for face_idx in mesh_colliding_faces:
            #face_vertices = self.faces[face_idx]
            #self.collision_active[face_vertices] = 1.0
        
        # Update GPU buffer
        glBindBuffer(GL_ARRAY_BUFFER, self.collision_active_vbo)
        glBufferData(GL_ARRAY_BUFFER, self.collision_active.nbytes, self.collision_active, GL_DYNAMIC_DRAW)

    def update_mesh(self):
        self.center = np.mean(self.vertices, axis=0)
        
        vertex_data = np.zeros((len(self.vertices), 6), dtype=np.float32)
        vertex_data[:, 0:3] = self.vertices
        vertex_data[:, 3:6] = self.vertex_normals
        
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_DYNAMIC_DRAW)

        #self.setup_face_normals_visualization()
        
    def setup_face_normals_visualization(self):
        normal_lines = []
        for i, face in enumerate(self.faces):
            v0, v1, v2 = self.vertices[face[0]], self.vertices[face[1]], self.vertices[face[2]]
            center = (v0 + v1 + v2) / 3.0
            normal = self.face_normals[i]
            line_start = center
            line_end = center + normal * 0.1  # Scale normal for visibility
            normal_lines.append(line_start)
            normal_lines.append(line_end)

        self.normal_lines = np.array(normal_lines, dtype=np.float32)

        # Update or create buffer for normals
        if hasattr(self, "normals_vbo"):
            glBindBuffer(GL_ARRAY_BUFFER, self.normals_vbo)
            glBufferData(GL_ARRAY_BUFFER, self.normal_lines.nbytes, self.normal_lines, GL_DYNAMIC_DRAW)
        else:
            self.normals_vao = glGenVertexArrays(1)
            glBindVertexArray(self.normals_vao)

            self.normals_vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.normals_vbo)
            glBufferData(GL_ARRAY_BUFFER, self.normal_lines.nbytes, self.normal_lines, GL_STATIC_DRAW)

            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
            glEnableVertexAttribArray(0)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)