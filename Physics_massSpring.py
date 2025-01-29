# Physics.py
import numpy as np

# Physical constants
DENSITY = 1000 # kg/m**3
GRAVITY = np.array([0.0, 0.0, -9.8])  # m/s**2
DT = 0.01  # second

# XPBD constants
SUB_STEPS = 20
EDGE_COMPLIANCE = 1e-5
VOLUME_COMPLIANCE = 0.0
COLLISION_COMPLIANCE = 0.0

class SoftBody:
    """Class to store the state and parameters of a single body"""
    def __init__(self, mesh, density=DENSITY, edgeCompliance=EDGE_COMPLIANCE, volCompliance=VOLUME_COMPLIANCE):
        # Store reference to the mesh
        self.mesh = mesh
        
        # Vertices
        self.numVertices = len(mesh.vertices)
        self.positions = mesh.vertices #(#vers, 3) array
        self.prev_positions = self.positions.copy()
        self.velocities = np.zeros_like(mesh.vertices)
        self.invMasses = np.zeros(self.numVertices)
        self.fixed_vertices = set()
        
        # Tetrahedrons
        self.numTets = len(mesh.tets)
        self.tetIds = mesh.tets #(#tets, 4) array
        self.density = density
        self.restVolume = np.zeros(self.numTets) # (#tets, 1) array
        self.invRestPoses = np.zeros((self.numTets, 3, 3)) # (#tets, (3, 3) matrix) array

        # Edges
        self.numEdges = len(mesh.edges)
        self.edgeIds = mesh.edges #(#eges, 2) array
        self.edgeLengths = np.zeros(self.numEdges)
        
        # Per-body temporary matrices and properties
        self.temp = np.zeros((4, 3))
        self.grads = np.zeros((4, 3))
        self.edgeCompliance = edgeCompliance
        self.volCompliance = volCompliance
        self.collision_constraints = [] # Store active collision constraints of certain mesh

class CollisionConstraint:
    def __init__(self, vertex_idx, trig_idx, triangle_mesh_idx):
        self.vertex_idx = vertex_idx
        self.trig_idx = trig_idx
        self.triangle_mesh_idx = triangle_mesh_idx

class Simulator:
    def __init__(self, collisionCompliance = COLLISION_COMPLIANCE):
        self.bodies = [] # It stores all soft bodies
        self.volIdOrder = [[1,3,2], [0,2,3], [0,3,1], [0,1,2]]
        self.collisionCompliance = collisionCompliance

    def add_body(self, mesh):
        body = SoftBody(mesh)
        self.bodies.append(body)
        self.init_mesh_physics(body)
    
    def init_mesh_physics(self, body):
        # Gather positions of the four vertices for all tetrahedra
        v0 = body.positions[body.tetIds[:, 0]]
        v1 = body.positions[body.tetIds[:, 1]]
        v2 = body.positions[body.tetIds[:, 2]]
        v3 = body.positions[body.tetIds[:, 3]]
        
        # Compute edge vectors for all tetrahedra
        edge1 = v1 - v0
        edge2 = v2 - v0
        edge3 = v3 - v0

        # Compute the rest volume for all tetrahedra
        volumes = np.einsum('ij,ij->i', np.cross(edge1, edge2), edge3) / 6.0
        body.restVolume = volumes

        # Add mass from all tetrahedra
        vMass = np.maximum(volumes, 0) * body.density / 4.0
        np.add.at(body.invMasses, body.tetIds.flatten(), np.repeat(vMass, 4))
        # Convert masses to inverse masses, change fixed vertices mass to 0
        body.invMasses = np.where(
            (body.invMasses > 0) & (~np.isin(np.arange(body.numVertices), list(body.fixed_vertices))),
            1.0 / body.invMasses,
            0.0
        )

        # Compute all the edge lengths
        edges = body.positions[body.edgeIds[:, 0]] - body.positions[body.edgeIds[:, 1]]
        body.edgeLengths = np.linalg.norm(edges, axis=1)

    def step(self):
        for body in self.bodies:
            body.collision_constraints = []
            # Generate mesh-mesh collision constraints
            self.detect_collisions(body)
        dt = DT / SUB_STEPS
        for _ in range(SUB_STEPS):
            for body in self.bodies:
                body.prev_positions = body.positions.copy()
                mask = ~np.isin(np.arange(body.numVertices), list(body.fixed_vertices))
                body.velocities[mask] += GRAVITY * dt
                body.positions[mask] += body.velocities[mask] * dt
                # Simple ground collisions
                for i in range(len(body.positions)):
                    if (body.positions[i, 2] < 0.0):
                        body.positions[i] = body.prev_positions[i].copy()
                        body.positions[i, 2] = 0.0

            for body in self.bodies: # It makes sure that all bodies' current position is predicted before we solve constraints
                self.solve_volume_constraint(body, dt)
                self.solve_edge_constraint(body, dt)
                self.solve_collision_constraints(body, dt)
            
            for body in self.bodies: # It makes sure that all bodies' constrains are solved before we update velocity
                body.velocities = (body.positions - body.prev_positions) / dt

    def solve_edge_constraint(self, body, dt):
        for edgeNr in range(body.numEdges):
            id0 = body.edgeIds[edgeNr, 0]
            id1 = body.edgeIds[edgeNr, 1]
            w0 = body.invMasses[id0]
            w1 = body.invMasses[id1]
            w = w0 + w1
            if (w == 0.0):
                continue

            grad = body.positions[id0] - body.positions[id1]
            len = np.linalg.norm(grad)
            if (len == 0.0):
                continue
            grad = grad / len
            restLen = body.edgeLengths[edgeNr]
            C = len - restLen
            alpha = body.edgeCompliance / dt / dt
            dlambda = -C / (w + alpha)
            
            body.positions[id0] += body.invMasses[id0] * dlambda * grad
            body.positions[id1] -= body.invMasses[id1] * dlambda * grad

    def solve_volume_constraint(self, body, dt):
        for tetNr in range(body.numTets):
            body.grads = np.zeros((4, 3))
            body.temp = np.zeros((4, 3))
            w = 0.0	
            for j in range(4):
                id0 = body.tetIds[tetNr, self.volIdOrder[j][0]]
                id1 = body.tetIds[tetNr, self.volIdOrder[j][1]]
                id2 = body.tetIds[tetNr, self.volIdOrder[j][2]]

                body.temp[0] = body.positions[id1] - body.positions[id0]
                body.temp[1] = body.positions[id2] - body.positions[id0]
                body.grads[j] = np.cross(body.temp[0], body.temp[1]) / 6.0

                w += body.invMasses[body.tetIds[tetNr, j]] * np.dot(body.grads[j], body.grads[j])
            if (w == 0.0):
                return

            vol = self.getTetVolume(body, tetNr)
            C = vol - body.restVolume[tetNr]
            alpha = body.volCompliance / dt / dt
            dlambda = -C / (w + alpha)
            
            for i in range(4):
                id = body.tetIds[tetNr][i]
                body.positions[id] += body.invMasses[id] * dlambda * body.grads[i]
    
    def getTetVolume(self, body, tetNr):
        body.temp = np.zeros((4, 3))
        id0 = body.tetIds[tetNr, 0]
        id1 = body.tetIds[tetNr, 1]
        id2 = body.tetIds[tetNr, 2]
        id3 = body.tetIds[tetNr, 3]
        body.temp[0] = body.positions[id1] - body.positions[id0]
        body.temp[1] = body.positions[id2] - body.positions[id0]
        body.temp[2] = body.positions[id3] - body.positions[id0]
        body.temp[3] = np.cross(body.temp[0], body.temp[1])
        return np.dot(body.temp[3], body.temp[2]) / 6.0
    
    def detect_collisions(self, body):
        current_min = np.min(body.positions, axis=0)
        current_max = np.max(body.positions, axis=0)
        for other_body_idx, other_body in enumerate(self.bodies):
            if other_body_idx == self.bodies.index(body):
                continue
            
            # Expand AABB to include both previous and current positions
            motion_min = np.minimum(np.min(body.prev_positions, axis=0), current_min)
            motion_max = np.maximum(np.max(body.prev_positions, axis=0), current_max)
            other_min = np.min(other_body.positions, axis=0)
            other_max = np.max(other_body.positions, axis=0)
            
            # AABB test with expanded bounds
            if (motion_max[0] < other_min[0] or motion_min[0] > other_max[0] or
                motion_max[1] < other_min[1] or motion_min[1] > other_max[1] or
                motion_max[2] < other_min[2] or motion_min[2] > other_max[2]):
                continue
            
            predicted_positions = body.positions + (body.velocities + GRAVITY * DT) * DT
            # For each vertex in mesh1, check against each triangle in mesh2
            for vertex_idx in range(len(body.positions)):
                for face_idx, face in enumerate(other_body.mesh.faces):
                    normal = other_body.mesh.face_normals[face_idx]
                    # Proximity test for vertices near or inside triangle
                    face_to_vertex = body.positions[vertex_idx] - other_body.positions[face[0]]
                    face_to_predicted_vertex = predicted_positions[vertex_idx] - other_body.positions[face[0]]
                    signed_distance = np.dot(face_to_vertex, normal)
                    signed_oredicted_distance = np.dot(face_to_predicted_vertex, normal)
                    
                    if (signed_distance < 0.0 and abs(signed_distance) < 0.01) or (signed_oredicted_distance < 0.0 and abs(signed_oredicted_distance) < 0.01):  # Threshold distance for proximity
                        # Project vertex onto triangle plane
                        projected_point = body.positions[vertex_idx] + signed_distance * normal
                        if self.point_in_triangle(projected_point,
                                                other_body.positions[face[0]],
                                                other_body.positions[face[1]],
                                                other_body.positions[face[2]]):
                            body.collision_constraints.append(CollisionConstraint(vertex_idx, face_idx, other_body_idx))

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
    
    def solve_collision_constraints(self, body, dt):
        for constraint in body.collision_constraints:
            vertex = body.positions[constraint.vertex_idx]
            trig_body = self.bodies[constraint.triangle_mesh_idx]
            trig = trig_body.mesh.faces[constraint.trig_idx]
            normal = trig_body.mesh.face_normals[constraint.trig_idx]
            
            C = np.dot((vertex - trig_body.positions[trig[0]]), normal)
            
            # Calculate weights
            w_vertex = body.invMasses[constraint.vertex_idx]
            w_triangle = sum(trig_body.invMasses[trig[i]] for i in range(3))
            total_weight = w_vertex + w_triangle
            if total_weight < 1e-6:
                continue
                
            alpha = self.collisionCompliance / dt / dt
            d_lambda = -C / (total_weight + alpha)
            
            # Apply corrections
            body.positions[constraint.vertex_idx] += w_vertex * d_lambda * normal
            for i in range(3):
                trig_body.positions[trig[i]] -= (trig_body.invMasses[trig[i]] * d_lambda * normal)