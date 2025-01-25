# Physics.py
import numpy as np
import warp as wp

# Physical constants
DENSITY = 1000 # kg/m**3
GRAVITY = wp.vec3(0.0, 0.0, -10.0)  # m/s**2
DT = 0.01  # second

# XPBD constants
SUB_STEPS = 100
EDGE_COMPLIANCE = 1e-4
VOLUME_COMPLIANCE = 0.0
COLLISION_COMPLIANCE = 1e-7
COLLISION_THRESHOLD = 0.01

# Parallelization parameters
JACOBISCALE = 0.2

wp.init()

class SoftBody:
    """Class to store the state and parameters of a single body"""
    def __init__(self, mesh, density=DENSITY, edgeCompliance=EDGE_COMPLIANCE, volCompliance=VOLUME_COMPLIANCE):
        device = "cpu"
        self.mesh = mesh
        
        # Vertices
        self.numVertices = len(mesh.vertices)
        self.positions = wp.array(mesh.vertices.astype(np.float32), dtype = wp.vec3, device = "cuda")
        self.prev_positions = wp.array(mesh.vertices.astype(np.float32), dtype = wp.vec3, device = "cuda")
        self.corrections = wp.array(mesh.vertices.astype(np.float32), dtype = wp.vec3, device = "cuda")
        self.velocities = wp.array(mesh.vertices.astype(np.float32), dtype = wp.vec3, device = "cuda")
        self.invMasses = wp.array(np.zeros(self.numVertices).astype(np.float32), dtype = float, device = "cuda")
        
        # Tetrahedrons
        self.numTets = len(mesh.tets)
        self.tetIds = wp.array(mesh.tets.astype(np.int32), dtype=wp.int32, device = "cuda")
        self.density = density
        self.restVolumes = wp.array(np.zeros(self.numTets).astype(np.float32), dtype = float, device = "cuda")

        # Edges
        self.numEdges = len(mesh.edges)
        self.edgeIds = wp.array(mesh.edges.astype(np.int32), dtype=wp.int32, device = "cuda")
        self.edgeLengths = wp.array(np.zeros(self.numEdges).astype(np.float32), dtype = float, device = "cuda")
        
        # Face normals
        self.faceIds = wp.array(mesh.faces.astype(np.int32), dtype=wp.int32, device = "cuda")
        self.faceNormals = wp.array(mesh.face_normals.astype(np.float32), dtype = wp.vec3, device = "cuda")
        self.vertexNormals = wp.array(np.zeros((self.numVertices, 3)).astype(np.float32), dtype = wp.vec3, device = "cuda")
        self.vertexNormalCounts = wp.array(np.zeros(self.numVertices).astype(np.int32), dtype=wp.int32, device = "cuda")
        
        # Per-body temporary matrices and properties
        self.edgeCompliance = edgeCompliance
        self.volCompliance = volCompliance
        self.collisionConstraints = []

class Simulator:
    def __init__(self, collisionCompliance=COLLISION_COMPLIANCE):
        device = "cpu"
        self.bodies = []
        self.collisionCompliance = collisionCompliance

    def add_body(self, mesh):
        body = SoftBody(mesh)
        self.bodies.append(body)
        self.init_mesh_physics(body)
    
    def init_mesh_physics(self, body):
        # Compute tet rest volumes
        wp.launch(kernel=self.compute_tet_volumes, dim=body.numTets,
            inputs=[body.positions, body.tetIds, body.restVolumes])
        # Compute masses
        wp.launch(kernel=self.compute_vertex_masses, dim=body.numTets,
            inputs=[body.tetIds, body.restVolumes, body.density, body.invMasses])
        # Compute edge lengths
        wp.launch(kernel=self.compute_edge_lengths, dim=body.numEdges,
            inputs=[body.positions, body.edgeIds, body.edgeLengths])
        # Convert masses to inverse masses
        wp.launch(kernel=self.invert_array,dim=body.numVertices,
            inputs=[body.invMasses])

    def detect_collisions(self, body, dt):
        for other_body_idx, other_body in enumerate(self.bodies):
            if other_body_idx == self.bodies.index(body):
                continue
            collision_results = wp.zeros((body.numVertices, len(other_body.faceIds)), dtype=wp.int32)
            wp.launch(
                kernel=self.check_vertex_triangle_collision,
                inputs=[body.positions, body.velocities, other_body.faceIds, other_body.faceNormals, other_body.positions, 
                        GRAVITY, dt, COLLISION_THRESHOLD, collision_results],
                dim=(body.numVertices, len(other_body.faceIds)), device = "cuda")
            
            collisions = np.argwhere(collision_results.numpy() == 1)
            for collision in collisions:
                body.collisionConstraints.append([collision[0], collision[1], other_body_idx])
    
    def solve_collision_constraints(self, body, dt):
        for other_body_idx, other_body in enumerate(self.bodies):
            if other_body_idx == self.bodies.index(body):
                continue
            # Filter constraint data
            n_constraints = len(body.collisionConstraints)
            constraints = np.array([row for row in body.collisionConstraints if row[2] == 1])
            if len(constraints) <= 0:
                continue
            vertex_indices = wp.array(constraints[:,0].astype(np.int32), dtype=wp.int32, device="cuda")
            trig_indices = wp.array(constraints[:,1].astype(np.int32), dtype=wp.int32, device="cuda")
            
            # Launch kernel
            wp.launch(
                kernel=self.solve_collision_constraints_kernel,
                dim=n_constraints,
                inputs=[
                    body.positions, 
                    body.invMasses,
                    other_body.positions,
                    other_body.invMasses,
                    other_body.faceIds,
                    other_body.faceNormals,
                    vertex_indices,
                    trig_indices,
                    self.collisionCompliance/(dt**2)
                ]
            )

    def step(self):
        for body in self.bodies:
            # Update the normals and copy results back to CPU
            meshGeoCenter = wp.vec3(*body.mesh.center.astype(np.float32))
            wp.launch(kernel=self.calculate_faceNormals,
                    inputs=[body.positions, body.faceIds, body.faceNormals, body.vertexNormals, body.vertexNormalCounts, meshGeoCenter],
                    dim=len(body.faceIds), device = "cuda")
            wp.launch(kernel=self.calculate_vertex_normals,
                inputs=[body.vertexNormals, body.vertexNormalCounts], dim=body.numVertices, device = "cuda")
            body.mesh.faceNormals = body.faceNormals.numpy()
            body.mesh.vertex_normals = body.vertexNormals.numpy()
            
            # Generate mesh-mesh collision constraints [vertexId, trigId, trigMeshId]
            #body.collisionConstraints = []
            #self.detect_collisions(body, DT)

        dt = DT / SUB_STEPS
        for _ in range(SUB_STEPS):
            for body in self.bodies:
                wp.launch(kernel = self.integrate, 
                    inputs = [body.positions, body.prev_positions, body.invMasses, body.velocities, GRAVITY, dt], 
                    dim = body.numVertices, device = "cuda")

            for body in self.bodies: # It makes sure that all bodies' current position is predicted before we solve constraints
                # Solve edges with multiple Jacobi passes
                self.iterative_jacobi_solve(device="cuda", kenerl=self.solve_edge_constraints, 
                                       inputs=[body.positions, body.invMasses, body.edgeIds, body.edgeLengths, body.edgeCompliance/(dt**2), body.corrections], dim=body.numEdges)
                self.iterative_jacobi_solve(device="cuda", kenerl=self.solve_volume_constraints, 
                                       inputs=[body.positions, body.invMasses, body.tetIds, body.restVolumes, body.volCompliance/(dt**2), body.corrections], dim=body.numTets)
                #if len(body.collisionConstraints) > 0:
                    #self.solve_collision_constraints(body, dt)
                # Copy results back to CPU
                body.mesh.vertices = body.positions.numpy()
            
            for body in self.bodies: # It makes sure that all bodies' constrains are solved before we update velocity
                wp.launch(kernel = self.update_velocity, 
                    inputs = [body.positions, body.prev_positions, body.velocities, dt], dim = body.numVertices, device = "cuda")
    
    @wp.kernel
    def invert_array(invMasses: wp.array(dtype=float)):
        mNr = wp.tid()
        invMasses[mNr] = 1.0 / invMasses[mNr]

    @wp.kernel
    def compute_tet_volumes(vertices: wp.array(dtype=wp.vec3),
                            tetIds: wp.array2d(dtype=wp.int32),
                            tetVolumes: wp.array(dtype=wp.float32)):
        
        tetNr = wp.tid()
        v0 = vertices[tetIds[tetNr, 0]]
        v1 = vertices[tetIds[tetNr, 1]]
        v2 = vertices[tetIds[tetNr, 2]]
        v3 = vertices[tetIds[tetNr, 3]]
        # Compute volume using cross product
        volume = wp.dot(wp.cross(v1 - v0, v2 - v0), v3 - v0) / 6.0
        tetVolumes[tetNr] = volume
            
    @wp.kernel
    def compute_vertex_masses(tetIds: wp.array2d(dtype=wp.int32),
                            volumes: wp.array(dtype=wp.float32),
                            density: float,
                            vertexMasses: wp.array(dtype=wp.float32)):
        
        tetNr = wp.tid()
        avg_mass = wp.max(volumes[tetNr] * density / 4.0, 0.0)
        # Accumulate the masses
        for i in range(4):
            wp.atomic_add(vertexMasses, tetIds[tetNr, i], avg_mass)

    # Kernel for computing edge lengths
    @wp.kernel
    def compute_edge_lengths(vertices: wp.array(dtype=wp.vec3),
                            edgeIds: wp.array2d(dtype=wp.int32),
                            edgeLengths: wp.array(dtype=wp.float32)):
        
        edgeNr = wp.tid()
        v0 = vertices[edgeIds[edgeNr, 0]]
        v1 = vertices[edgeIds[edgeNr, 1]]
        edgeLengths[edgeNr] = wp.length(v1 - v0)
                        
    @wp.kernel
    def solve_edge_constraints(vertices: wp.array(dtype=wp.vec3),
                            invMasses: wp.array(dtype=wp.float32),
                            edgeIds: wp.array2d(dtype=wp.int32),
                            edgeLengths: wp.array(dtype=wp.float32),
                            alpha: wp.float32,
                            corrections: wp.array(dtype = wp.vec3)):
                        
        eNr = wp.tid()
        id0 = edgeIds[eNr, 0]
        id1 = edgeIds[eNr, 1]
        w0 = invMasses[id0]
        w1 = invMasses[id1]
        w = w0 + w1
        if w > 0.0:
            grad = vertices[id0] - vertices[id1]
            len = wp.length(grad)
            if len > 0.0:
                grad = grad / len
                rest_len = edgeLengths[eNr]
                C = len - rest_len
                dlambda = -C / (w + alpha)
                
                wp.atomic_add(corrections, id0, w0 * dlambda * grad)
                wp.atomic_add(corrections, id1, -w1 * dlambda * grad)

    @wp.kernel
    def solve_volume_constraints(vertices: wp.array(dtype=wp.vec3),
                                invMasses: wp.array(dtype=wp.float32),
                                tetIds: wp.array2d(dtype=wp.int32),
                                restVolumes: wp.array(dtype=wp.float32),
                                alpha: wp.float32,
                                corrections: wp.array(dtype = wp.vec3)):
        
        tetNr = wp.tid()
        i0 = tetIds[tetNr, 0]
        i1 = tetIds[tetNr, 1]
        i2 = tetIds[tetNr, 2]
        i3 = tetIds[tetNr, 3]
        # Get vertices of tetrahedron
        p0 = vertices[i0]
        p1 = vertices[i1]
        p2 = vertices[i2]
        p3 = vertices[i3]
        # Calculate volume
        e1 = p1 - p0
        e2 = p2 - p0
        e3 = p3 - p0
        vol = wp.dot(wp.cross(e1, e2), e3) / 6.0
        # Calculate gradients
        grad0 = wp.cross(p3 - p1, p2 - p1) / 6.0
        grad1 = wp.cross(p2 - p0, p3 - p0) / 6.0
        grad2 = wp.cross(p3 - p0, p1 - p0) / 6.0
        grad3 = wp.cross(p1 - p0, p2 - p0) / 6.0
        # Calculate weights
        w = 0.0
        w += invMasses[i0] * wp.dot(grad0, grad0)
        w += invMasses[i1] * wp.dot(grad1, grad1)
        w += invMasses[i2] * wp.dot(grad2, grad2)
        w += invMasses[i3] * wp.dot(grad3, grad3)
        if w > 0.0:
            C = vol - restVolumes[tetNr]
            dlambda = -C / (w + alpha)
            # Apply corrections using atomic operations
            wp.atomic_add(corrections, i0, invMasses[i0] * dlambda * grad0)
            wp.atomic_add(corrections, i1, invMasses[i1] * dlambda * grad1)
            wp.atomic_add(corrections, i2, invMasses[i2] * dlambda * grad2)
            wp.atomic_add(corrections, i3, invMasses[i3] * dlambda * grad3)

    @wp.kernel
    def add_corrections(vertices: wp.array(dtype=wp.vec3),
                        corrections: wp.array(dtype = wp.vec3),
                        scale: float):
        vNr = wp.tid()
        vertices[vNr] = vertices[vNr] + corrections[vNr] * scale
        
    def iterative_jacobi_solve(self, device, kenerl, inputs, dim, numIterations=5, jacobiScale = JACOBISCALE):
        for _ in range(numIterations):
            # Zero out corrections from previous iteration
            inputs[-1].zero_()
            # Launch kernel to compute all partial corrections
            wp.launch(kernel=kenerl, dim=dim, inputs=inputs, device=device)
            # Launch kernel to add corrections
            wp.launch(kernel=self.add_corrections,dim=inputs[0].shape[0],
                inputs=[inputs[0], inputs[-1], jacobiScale],
                device=device)

    @wp.kernel
    def integrate(positions: wp.array(dtype = wp.vec3),
                prevPositions: wp.array(dtype = wp.vec3),
                invMass: wp.array(dtype = float),
                velocities: wp.array(dtype = wp.vec3),
                acceleration: wp.vec3,
                dt: float):
        vNr = wp.tid()
        # Copy current positions
        prevPositions[vNr] = positions[vNr]
        # Integrate
        if invMass[vNr] > 0.0:
            velocities[vNr] = velocities[vNr] + acceleration * dt
            positions[vNr] = positions[vNr] + velocities[vNr] * dt
        # Simple ground collisions
        if (positions[vNr][2] < 0.0):
            positions[vNr] = prevPositions[vNr]
            positions[vNr][2] = 0.0
            
    @wp.kernel
    def update_velocity(positions: wp.array(dtype = wp.vec3),
                        prevPositions: wp.array(dtype = wp.vec3),
                        velocities: wp.array(dtype = wp.vec3),
                        dt: float):
        vNr = wp.tid()
        velocities[vNr] = (positions[vNr] - prevPositions[vNr]) / dt
    
    @wp.kernel   
    def calculate_faceNormals(positions: wp.array(dtype = wp.vec3),
                            faceIds: wp.array2d(dtype=wp.int32),
                            faceNormals: wp.array(dtype = wp.vec3),
                            vertexNormals: wp.array(dtype = wp.vec3),
                            vertexNormalCounts: wp.array(dtype = wp.int32),
                            meshGeoCenter: wp.vec3):  
        fNr = wp.tid()
        v0, v1, v2 = positions[faceIds[fNr, 0]], positions[faceIds[fNr, 1]], positions[faceIds[fNr, 2]]
        normal = wp.cross(v1 - v0, v2 - v0)
        length = wp.length(normal)
        if length > 0:
            normal = normal / length
        # Make normal point outward
        face_center = (v0 + v1 + v2) / 3.0
        face_center_local = face_center - meshGeoCenter
        if wp.dot(normal, face_center_local) < 0:
            normal = -normal
        faceNormals[fNr] = normal
        # For each vertex in the face, add the face normal to vertex normal
        for j in range(3):
            vertex_idx = faceIds[fNr, j]
            wp.atomic_add(vertexNormals, faceIds[fNr, j], normal)
            wp.atomic_add(vertexNormalCounts, vertex_idx, 1)
    
    @wp.kernel 
    def calculate_vertex_normals(vertexNormals: wp.array(dtype = wp.vec3),
                                vertexNormalCounts: wp.array(dtype = wp.int32)):
        vNr = wp.tid()
        if vertexNormalCounts[vNr] > 0:
            length = wp.length(vertexNormals[vNr])
            if length > 0:
                vertexNormals[vNr] = vertexNormals[vNr] / length
                
    @wp.kernel
    def check_vertex_triangle_collision(vertices: wp.array(dtype=wp.vec3), 
                                        velocities: wp.array(dtype = wp.vec3),
                                        faceIds: wp.array2d(dtype=wp.int32),
                                        faceNormals: wp.array(dtype=wp.vec3), 
                                        trigMeshVertices: wp.array(dtype=wp.vec3),
                                        acceleration: wp.vec3,
                                        dt: float,
                                        threshold: float,
                                        collisionConstraints: wp.array2d(dtype=wp.int32)):
        vNr, trigNr = wp.tid()

        # Get vertex positions
        vertex = vertices[vNr]
        predicted_position = vertex + (velocities[vNr] + acceleration * dt) * dt
        # Get triangle vertices and normal
        idx0, idx1, idx2 = faceIds[trigNr, 0], faceIds[trigNr, 1], faceIds[trigNr, 2]
        a, b, c = trigMeshVertices[idx0], trigMeshVertices[idx1], trigMeshVertices[idx2]
        normal = faceNormals[trigNr]

        # Compute signed distances
        face_to_vertex = vertex - a
        face_to_predicted = predicted_position - a
        signed_distance = wp.dot(face_to_vertex, normal)
        signed_predicted_distance = wp.dot(face_to_predicted, normal)
        flag = False
        if (signed_distance < 0.0 and abs(signed_distance) < threshold):
            projected_point = vertex + signed_distance * normal
            flag = True
        if (signed_predicted_distance < 0.0 and abs(signed_predicted_distance) < threshold):
            projected_point = predicted_position + signed_predicted_distance * normal
            flag = True
        # Point in triangle test    
        if flag: 
            v0 = c - a
            v1 = b - a
            v2 = projected_point - a
            dot00 = np.dot(v0, v0)
            dot01 = np.dot(v0, v1)
            dot02 = np.dot(v0, v2)
            dot11 = np.dot(v1, v1)
            dot12 = np.dot(v1, v2)
            # Compute barycentric coordinates
            denom = dot00 * dot11 - dot01 * dot01
            if abs(denom) > 0.0:  # Degenerate triangle
                u = (dot11 * dot02 - dot01 * dot12) / denom
                v = (dot00 * dot12 - dot01 * dot02) / denom
                if u >= 0.0 and v >= 0.0 and u + v <= 1.0:
                    collisionConstraints[vNr, trigNr] = 1
    
    @wp.kernel
    def solve_collision_constraints_kernel(
        vertices: wp.array(dtype=wp.vec3),
        vInvMasses: wp.array(dtype=float),
        trigMeshVertices: wp.array(dtype=wp.vec3),
        trigInvMasses: wp.array(dtype=float),
        faceIds: wp.array2d(dtype=wp.int32),
        faceNormals: wp.array(dtype=wp.vec3),
        constraint_vertex_indices: wp.array(dtype=int),
        constraint_trig_indices: wp.array(dtype=int),
        alpha: float):
        
        cNr = wp.tid()
        vertex_idx = constraint_vertex_indices[cNr]
        trig_idx = constraint_trig_indices[cNr]
        # Get vertex position and triangle data
        vertex = vertices[vertex_idx]
        normal = faceNormals[trig_idx]
        
        C = wp.dot((vertex - trigMeshVertices[faceIds[trig_idx, 0]]), normal)
        w_vertex = vInvMasses[vertex_idx]
        w_triangle = trigInvMasses[faceIds[trig_idx, 0]] + trigInvMasses[faceIds[trig_idx, 1]] + trigInvMasses[faceIds[trig_idx, 2]]
        total_weight = w_vertex + w_triangle
        if total_weight < 1e-6:
            return
        d_lambda = -C / (total_weight + alpha)
        
        # Apply corrections
        delta = normal * d_lambda
        wp.atomic_add(vertices, vertex_idx, delta * w_vertex)
        
        for i in range(3):
            wp.atomic_add(trigMeshVertices, faceIds[trig_idx, i], -delta * trigInvMasses[faceIds[trig_idx, i]])