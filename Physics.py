# Physics.py
import numpy as np
from test import *

# Physical constants
DENSITY = 1000.0  # kg/m**3
GRAVITY = np.array([0.0, 0.0, -9.8])  # m/s**2
DT = 0.01  # second

# XPBD constants
SUB_STEPS = 5
DEVIATORIC_COMPLIANCE = 1.0/100000.0
VOLUME_COMPLIANCE = 0.0
COLLISION_COMPLIANCE = 1e-3  # Stiffness for collision response

class MeshState:
    """Class to store the state and parameters of a single mesh"""
    def __init__(self, mesh, density=DENSITY):
        # Store reference to the mesh
        self.mesh = mesh
        
        # Vertices
        self.numVertices = len(mesh.vertices)
        self.positions = mesh.vertices #(#vers, 3) array
        self.prev_positions = self.positions.copy()
        self.velocities = np.zeros_like(mesh.vertices)
        self.fixed_vertices = set()
        
        # Tetrahedrons
        self.numTets = len(mesh.tets)
        self.tetIds = mesh.tets #(#tets, 4) array
        self.density = density
        self.invRestVolume = np.zeros(self.numTets) # (#tets, 1) array
        self.invRestPoses = np.zeros((self.numTets, 3, 3)) # (#tets, (3, 3) matrix) array
        self.invMasses = np.zeros(self.numVertices)
        
        # Per-mesh temporary matrices for parallel computation
        self.P = np.zeros((3, 3))
        self.F = np.zeros((3, 3))
        self.dF = np.zeros((3, 3))
        self.grads = np.zeros((4, 3))

class CollisionConstraint:
    def __init__(self, vertex_idx, vertex_mesh_idx, trig_idx, triangle_mesh_idx):
        self.vertex_idx = vertex_idx
        self.vertex_mesh_idx = vertex_mesh_idx
        self.trig_idx = trig_idx
        self.triangle_mesh_idx = triangle_mesh_idx

class Simulator:
    def __init__(self, devCompliance=DEVIATORIC_COMPLIANCE, volCompliance=VOLUME_COMPLIANCE):
        self.meshes = [] # It stores MeshState objs, not the Mesh obj
        self.devCompliance = devCompliance
        self.volCompliance = volCompliance
        self.collision_constraints = []  # Store active collision constraints

    def add_mesh(self, mesh, density=DENSITY):
        mesh_state = MeshState(mesh, density)
        self.meshes.append(mesh_state)
        self.init_mesh_physics(mesh_state)
    
    def init_mesh_physics(self, mesh_state):
        # Gather positions of the four vertices for all tetrahedra
        v0 = mesh_state.positions[mesh_state.tetIds[:, 0]]
        v1 = mesh_state.positions[mesh_state.tetIds[:, 1]]
        v2 = mesh_state.positions[mesh_state.tetIds[:, 2]]
        v3 = mesh_state.positions[mesh_state.tetIds[:, 3]]
        
        # Compute edge vectors for all tetrahedra
        edge1 = v1 - v0
        edge2 = v2 - v0
        edge3 = v3 - v0

        # Compute the volume for all tetrahedra
        volumes = np.einsum('ij,ij->i', np.cross(edge1, edge2), edge3) / 6.0
        mesh_state.invRestVolume = 1.0 / volumes
        mesh_state.invRestPoses = np.linalg.inv(np.stack((edge1, edge2, edge3), axis=1))

        # Add mass from all tetrahedra
        vMass = np.maximum(volumes, 0) * mesh_state.density / 4.0
        np.add.at(mesh_state.invMasses, mesh_state.tetIds.flatten(), np.repeat(vMass, 4))
        
        # Convert masses to inverse masses, change fixed vertices mass to 0
        mesh_state.invMasses = np.where(
            (mesh_state.invMasses > 0) & (~np.isin(np.arange(mesh_state.numVertices), list(mesh_state.fixed_vertices))),
            1.0 / mesh_state.invMasses,
            0.0
        )

    def step(self):
        dt = DT / SUB_STEPS
        for _ in range(SUB_STEPS):
            for mesh_state in self.meshes:
                mesh_state.prev_positions = mesh_state.positions.copy()
                mask = ~np.isin(np.arange(mesh_state.numVertices), list(mesh_state.fixed_vertices))
                mesh_state.velocities[mask] += GRAVITY * dt
                mesh_state.positions[mask] += mesh_state.velocities[mask] * dt
                self.detect_collisions(mesh_state)
            
             # Solve internal constraints
            self.solve_all_constraints(dt)
        
            # Final velocity update
            for mesh_state in self.meshes:
                mesh_state.velocities = (mesh_state.positions - mesh_state.prev_positions) / dt

    def solve_all_constraints(self, dt):
        for mesh_state in self.meshes:
            for tetNr in range(mesh_state.numTets):
                self.solve_hydro_constraint(mesh_state, tetNr, dt)
                self.solve_dev_constraint(mesh_state, tetNr, dt)
        self.solve_collision_constraints(dt)

    def solve_dev_constraint(self, mesh_state, tetNr, dt):
        vIds = mesh_state.tetIds[tetNr]
        edge1 = mesh_state.positions[vIds[1]] - mesh_state.positions[vIds[0]]
        edge2 = mesh_state.positions[vIds[2]] - mesh_state.positions[vIds[0]]
        edge3 = mesh_state.positions[vIds[3]] - mesh_state.positions[vIds[0]]
        mesh_state.P = np.array([edge1, edge2, edge3])
        mesh_state.F = mesh_state.P @ mesh_state.invRestPoses[tetNr]
        r_s = np.sqrt(np.dot(mesh_state.F[0], mesh_state.F[0]) + 
                     np.dot(mesh_state.F[1], mesh_state.F[1]) + 
                     np.dot(mesh_state.F[2], mesh_state.F[2]))
        r_s_inv = 1.0 / r_s
        
        C = r_s
        mesh_state.grads = np.zeros((4, 3))
        product = r_s_inv * mesh_state.F @ np.transpose(mesh_state.invRestPoses[tetNr])
        mesh_state.grads[1] = product[0]
        mesh_state.grads[2] = product[1]
        mesh_state.grads[3] = product[2]
        self.correct_position(mesh_state, tetNr, C, self.devCompliance, dt)

    def solve_hydro_constraint(self, mesh_state, tetNr, dt):
        vIds = mesh_state.tetIds[tetNr]
        edge1 = mesh_state.positions[vIds[1]] - mesh_state.positions[vIds[0]]
        edge2 = mesh_state.positions[vIds[2]] - mesh_state.positions[vIds[0]]
        edge3 = mesh_state.positions[vIds[3]] - mesh_state.positions[vIds[0]]
        mesh_state.P = np.array([edge1, edge2, edge3])
        mesh_state.F = mesh_state.P @ mesh_state.invRestPoses[tetNr]
        mesh_state.dF[0] = np.cross(mesh_state.F[1], mesh_state.F[2])
        mesh_state.dF[1] = np.cross(mesh_state.F[2], mesh_state.F[0])
        mesh_state.dF[2] = np.cross(mesh_state.F[0], mesh_state.F[1])
        vol = np.linalg.det(mesh_state.F)
        
        C = vol - 1.0 - self.volCompliance / self.devCompliance
        mesh_state.grads = np.zeros((4, 3))
        product = mesh_state.dF @ np.transpose(mesh_state.invRestPoses[tetNr])
        mesh_state.grads[1] = product[0]
        mesh_state.grads[2] = product[1]
        mesh_state.grads[3] = product[2]
        self.correct_position(mesh_state, tetNr, C, self.volCompliance, dt)

    def correct_position(self, mesh_state, tetNr, C, compliance, dt):
        if C == 0.0:
            return
        
        mesh_state.grads[0] = -mesh_state.grads[1]
        mesh_state.grads[0] -= mesh_state.grads[2]
        mesh_state.grads[0] -= mesh_state.grads[3]
        
        w = 0.0
        for i in range(4):
            id = mesh_state.tetIds[tetNr][i]
            w += np.dot(mesh_state.grads[i], mesh_state.grads[i]) * mesh_state.invMasses[id]
        
        if w == 0.0:
            return
        
        alpha = compliance * mesh_state.invRestVolume[tetNr] / dt / dt
        dlambda = -C / (w + alpha)
        
        for i in range(4):
            id = mesh_state.tetIds[tetNr][i]
            mesh_state.positions[id] += mesh_state.invMasses[id] * dlambda * mesh_state.grads[i]
    
    def detect_collisions(self, mesh_state):
        ground_level = 0.0
        for i in range(mesh_state.numVertices):
            if mesh_state.positions[i, 2] < ground_level:
                mesh_state.positions[i, 2] = ground_level 

        current_mesh_idx = self.meshes.index(mesh_state)
        self.collision_constraints = []

        """
        spatial_hash = SpatialHash(cell_size=0.05)
        # Do a broadphase collision detection
        for other_idx, other_mesh_state in enumerate(self.meshes):
            if other_idx == current_mesh_idx:
                continue
            collisions = spatial_hash.find_vertex_triangle_collisions(
                        mesh_state.positions,
                        other_mesh_state.positions,
                        other_mesh_state.mesh.faces,
                        threshold=0.0
                    )
                    
            for c in collisions:
                vId = c[0]
                trigId = c[1]
                
                # Check if vertex is "within" trig
                normal = other_mesh_state.mesh.face_normals[trigId]
                dot_product = np.dot(normal, mesh_state.positions[vId] - other_mesh_state.positions[trigId[0]])
                if dot_product > 0.0:
                    continue
                        
                # Calculate intersection
                t = np.dot(normal, mesh_state.positions[vId] - mesh_state.prev_positions[vId]) / dot_product
                    
                intersection = mesh_state.prev_positions[vId] + t * (mesh_state.positions[vId] - mesh_state.prev_positions[vId])
                if self.point_in_triangle(intersection,
                                        other_mesh_state.positions[trigId[0]],
                                        other_mesh_state.positions[trigId[1]],
                                        other_mesh_state.positions[trigId[2]]):
                    self.collision_constraints.append(CollisionConstraint(vId, current_mesh_idx, trigId, other_idx))

        """
        # Mesh-to-mesh collision detection
        current_min = np.min(mesh_state.positions, axis=0)
        current_max = np.max(mesh_state.positions, axis=0)
        for other_idx, other_mesh_state in enumerate(self.meshes):
            if other_idx == current_mesh_idx:
                continue
            
            # AABB test (unchanged)
            other_min = np.min(other_mesh_state.positions, axis=0)
            other_max = np.max(other_mesh_state.positions, axis=0)
            if (current_max[0] < other_min[0] or current_min[0] > other_max[0] or
                current_max[1] < other_min[1] or current_min[1] > other_max[1] or
                current_max[2] < other_min[2] or current_min[2] > other_max[2]):
                continue
            
            # For each vertex in mesh1, check against each triangle in mesh2
            for vertex_idx, vertex in enumerate(mesh_state.positions):
                prev_pos = mesh_state.prev_positions[vertex_idx]
                curr_pos = vertex
                direction = curr_pos - prev_pos
                direction_length = np.linalg.norm(direction)
                
                if direction_length < 1e-6:  # Skip if barely moving
                    continue
                    
                # Normalize direction
                direction_normalized = direction / direction_length
                
                for face_idx, face in enumerate(other_mesh_state.mesh.faces):
                    normal = other_mesh_state.mesh.face_normals[face_idx]
                    v0 = other_mesh_state.positions[face[0]]
                    
                    # Check if ray and plane are nearly parallel
                    dot_product = np.dot(normal, direction_normalized)
                    if abs(dot_product) < 1e-6:
                        continue
                        
                    # Calculate intersection
                    t = np.dot(normal, v0 - prev_pos) / dot_product
                    
                    # Scale t back to original range
                    t = t * direction_length
                    
                    if 0.0 <= t <= 1.0:
                        intersection = prev_pos + t * direction
                        if self.point_in_triangle(intersection,
                                                other_mesh_state.positions[face[0]],
                                                other_mesh_state.positions[face[1]],
                                                other_mesh_state.positions[face[2]]):
                            self.collision_constraints.append(CollisionConstraint(vertex_idx, current_mesh_idx, face_idx, other_idx))

    def point_in_triangle(self, p, a, b, c):
        """Check if point p is inside triangle abc using barycentric coordinates"""
        # Compute vectors
        v0 = c - a
        v1 = b - a
        v2 = p - a
        
        # Compute dot products
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)
        
        # Compute barycentric coordinates
        denom = dot00 * dot11 - dot01 * dot01
        if abs(denom) < 1e-10:  # Degenerate triangle
            return False
            
        u = (dot11 * dot02 - dot01 * dot12) / denom
        v = (dot00 * dot12 - dot01 * dot02) / denom
        
        # Check if point is in triangle
        return (u >= 0) and (v >= 0) and (u + v <= 1)
    
    def solve_collision_constraints(self, dt):
        for constraint in self.collision_constraints:
            self.solve_single_collision(constraint, dt)

    def solve_single_collision(self, constraint, dt):
        vertex_mesh = self.meshes[constraint.vertex_mesh_idx]
        v = vertex_mesh.positions[constraint.vertex_idx]
        trig_mesh = self.meshes[constraint.triangle_mesh_idx]
        trig = trig_mesh.mesh.faces[constraint.trig_idx]
        normal = trig_mesh.mesh.face_normals[constraint.trig_idx]
        C = np.dot((v - trig_mesh.positions[trig[0]]), normal)

        if C >= 0:
            return
        
        # Calculate weights
        w_vertex = vertex_mesh.invMasses[constraint.vertex_idx]
        w_triangle = sum(trig_mesh.invMasses[trig[i]] for i in range(3))
        
        total_weight = w_vertex + w_triangle
        if total_weight < 1e-6:
            return
            
        alpha = COLLISION_COMPLIANCE / dt / dt
        dlambda = C / (total_weight + alpha)
        
        # Apply to vertex
        vertex_mesh.positions[constraint.vertex_idx] += w_vertex * dlambda * normal
        
        # Apply to triangle vertices - NO additional division by 3 needed!
        for i in range(3):
            trig_mesh.positions[trig[i]] -= w_triangle/3.0 * dlambda * normal