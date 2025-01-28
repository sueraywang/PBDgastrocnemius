# Physics.py
import numpy as np
import warp as wp

# Physical constants
DENSITY = 1e3 # kg/m**3
GRAVITY = wp.vec3(0.0, 0.0, -10.0)  # m/s**2
DT = 5e-3  # second

# XPBD constants
SUB_STEPS = 30
EDGE_COMPLIANCE = 1e-4
VOLUME_COMPLIANCE = 0.0
COLLISION_COMPLIANCE = 0.0
COLLISION_THRESHOLD = 0.01

# Constants for triangle contact features
TRI_CONTACT_FEATURE_VERTEX_A = 0
TRI_CONTACT_FEATURE_VERTEX_B = 1
TRI_CONTACT_FEATURE_VERTEX_C = 2
TRI_CONTACT_FEATURE_EDGE_AB = 3
TRI_CONTACT_FEATURE_EDGE_AC = 4
TRI_CONTACT_FEATURE_EDGE_BC = 5
TRI_CONTACT_FEATURE_FACE_INTERIOR = 6

# Parallelization parameters
JACOBISCALE = 0.2

wp.init()

@wp.struct
class CollisionInfo:
    """Stores collision data between meshes"""
    # Vertex to triangle collisions
    vertex_colliding_triangles: wp.array(dtype=wp.int32)        # Array storing triangle indices for each vertex collision
    vertex_collision_offsets: wp.array(dtype=wp.int32)          # Start index in the collision array for each vertex
    vertex_collision_counts: wp.array(dtype=wp.int32)           # Number of collisions for each vertex
    vertex_collision_distances: wp.array(dtype=wp.float32)      # Distance to each colliding triangle
    
    # Triangle to vertex collisions (inverse mapping)
    triangle_colliding_vertices: wp.array(dtype=wp.int32)       # Array storing vertex indices for each triangle collision
    triangle_collision_offsets: wp.array(dtype=wp.int32)        # Start index in the collision array for each triangle
    triangle_collision_counts: wp.array(dtype=wp.int32)         # Number of collisions for each triangle
    triangle_collision_distances: wp.array(dtype=wp.float32)    # Distance to each colliding vertex

class Simulator:
    def __init__(self, meshes=None, density=DENSITY, edgeCompliance=EDGE_COMPLIANCE, volCompliance=VOLUME_COMPLIANCE, collisionCompliance=COLLISION_COMPLIANCE, collisionThreshold = COLLISION_THRESHOLD):
        self.meshes = meshes
        self.density = density
        self.edgeCompliance = edgeCompliance
        self.volCompliance = volCompliance
        self.collisionCompliance = collisionCompliance
        
        # Collision detection parameters
        self.collision_buffer_size = 8  # Each vertex/triangle can store up to 8 collisions
        self.collisionThreshold = collisionThreshold  # Minimum vertex-to-vertex distance
        
        if self.meshes is not None:
            self.init_simulator()
        
    def add_bodies(self, meshes):
        self.meshes = meshes
        self.init_simulator()

    def init_simulator(self):
        # Prepare data
        vertex_offsets = [0]
        tet_offsets = [0]
        edge_offsets = [0]
        face_offsets = [0]
        mesh_centers = []
        all_vetices = np.empty((0, 3))
        all_tets = np.empty((0, 4))
        all_edges = np.empty((0, 2))
        all_faces = np.empty((0, 3))
        
        for mesh in self.meshes:
            vertex_offsets.append(vertex_offsets[-1] + len(mesh.vertices))
            tet_offsets.append(tet_offsets[-1] + len(mesh.tets))
            edge_offsets.append(edge_offsets[-1] + len(mesh.edges))
            face_offsets.append(face_offsets[-1] + len(mesh.faces))
            mesh_centers.append(mesh.center)
            
            all_tets = np.concatenate((all_tets, mesh.tets + len(all_vetices)), axis=0)
            all_edges = np.concatenate((all_edges, mesh.edges + len(all_vetices)), axis=0)
            all_faces = np.concatenate((all_faces, mesh.faces + len(all_vetices)), axis=0)
            all_vetices = np.concatenate((all_vetices, mesh.vertices), axis=0)
            
        self.vertexOffsets = wp.array(np.array(vertex_offsets).astype(np.int32), dtype=wp.int32, device = "cuda")
        self.tetOffsets = wp.array(np.array(tet_offsets).astype(np.int32), dtype=wp.int32, device = "cuda")
        self.edgeOffsets = wp.array(np.array(edge_offsets).astype(np.int32), dtype=wp.int32, device = "cuda")
        self.faceOffsets = wp.array(np.array(face_offsets).astype(np.int32), dtype=wp.int32, device = "cuda")
        
        # Vertices
        self.numVertices = len(all_vetices)
        self.positions = wp.array(all_vetices.astype(np.float32), dtype = wp.vec3, device = "cuda")
        self.prev_positions = wp.array(all_vetices.astype(np.float32), dtype = wp.vec3, device = "cuda")
        self.corrections = wp.zeros(self.numVertices, dtype = wp.vec3, device = "cuda")
        self.velocities = wp.zeros(self.numVertices, dtype=wp.vec3, device="cuda")
        self.invMasses = wp.zeros(self.numVertices, dtype=wp.float32, device="cuda")
        self.vertexNormals = wp.zeros(self.numVertices, dtype = wp.vec3, device = "cuda")
        self.vertexNormalCounts = wp.zeros(self.numVertices, dtype=wp.int32, device = "cuda")
        
        # Tetrahedrons
        self.numTets = len(all_tets)
        self.tetIds = wp.array(all_tets.astype(np.int32), dtype=wp.int32, device = "cuda")
        self.restVolumes = wp.zeros(self.numTets, dtype = wp.float32, device = "cuda")

        # Edges
        self.numEdges = len(all_edges)
        self.edgeIds = wp.array(all_edges.astype(np.int32), dtype=wp.int32, device = "cuda")
        self.edgeLengths = wp.zeros(self.numEdges, dtype = wp.float32, device = "cuda")
        
        # Faces
        self.numFaces = len(all_faces)
        self.faceIds = wp.array(all_faces.astype(np.int32), dtype=wp.int32, device = "cuda")
        self.faceNormals = wp.zeros(self.numFaces, dtype = wp.vec3, device = "cuda")
        self.meshCenters = wp.array(np.array(mesh_centers).astype(np.float32), dtype=wp.vec3, device="cuda")
        
        # Create BVH using all vertices and faces
        self.BVH = wp.Mesh(points=self.positions, indices=wp.array(all_faces.flatten().astype(np.int32), dtype=wp.int32))
        
        # Compute tet rest volumes
        wp.launch(kernel=compute_tet_volumes, dim=self.numTets, device = "cuda",
            inputs=[self.positions, self.tetIds, self.restVolumes])
        # Compute masses
        wp.launch(kernel=compute_vertex_masses, dim=self.numTets, device = "cuda",
            inputs=[self.tetIds, self.restVolumes, self.density, self.invMasses])
        # Compute edge lengths
        wp.launch(kernel=compute_edge_lengths, dim=self.numEdges, device = "cuda",
            inputs=[self.positions, self.edgeIds, self.edgeLengths])
        # Convert masses to inverse masses
        wp.launch(kernel=invert_array,dim=self.numVertices, device = "cuda",
            inputs=[self.invMasses])
        
        # Initialize collision detection buffers
        self.init_collision_buffers()
        
    def update_bvh(self):
        # Update BVH mesh vertices
        wp.copy(self.BVH.points, self.positions)
        # Refit BVH after updating vertices
        self.BVH.refit()

    def init_collision_buffers(self):
        """Initialize arrays for storing collision information"""
        # Pre-allocate collision arrays with fixed buffer size per vertex/triangle
        vertex_collision_buffer_size = self.numVertices * self.collision_buffer_size
        triangle_collision_buffer_size = self.numFaces * self.collision_buffer_size
        
        # Initialize vertex collision arrays
        self.vertex_colliding_triangles = wp.zeros(vertex_collision_buffer_size, dtype=wp.int32, device="cuda")
        self.vertex_collision_counts = wp.zeros(self.numVertices, dtype=wp.int32, device="cuda")
        self.vertex_collision_distances = wp.zeros(vertex_collision_buffer_size, dtype=wp.float32, device="cuda")
        
        # Initialize triangle collision arrays
        self.triangle_colliding_vertices = wp.zeros(triangle_collision_buffer_size, dtype=wp.int32, device="cuda")
        self.triangle_collision_counts = wp.zeros(self.numFaces, dtype=wp.int32, device="cuda")
        self.triangle_collision_distances = wp.zeros(triangle_collision_buffer_size, dtype=wp.float32, device="cuda")
        
        # Compute offset arrays
        self.vertex_collision_offsets = self.compute_collision_offsets(self.numVertices, self.collision_buffer_size)
        self.triangle_collision_offsets = self.compute_collision_offsets(self.numFaces, self.collision_buffer_size)
        
        # Create CollisionInfo struct instance
        self.collision_info = CollisionInfo()
        self.collision_info.vertex_colliding_triangles = self.vertex_colliding_triangles
        self.collision_info.vertex_collision_offsets = self.vertex_collision_offsets
        self.collision_info.vertex_collision_counts = self.vertex_collision_counts
        self.collision_info.vertex_collision_distances = self.vertex_collision_distances
        
        self.collision_info.triangle_colliding_vertices = self.triangle_colliding_vertices
        self.collision_info.triangle_collision_offsets = self.triangle_collision_offsets
        self.collision_info.triangle_collision_counts = self.triangle_collision_counts
        self.collision_info.triangle_collision_distances = self.triangle_collision_distances

    def compute_collision_offsets(self, count: int, buffer_size: int) -> wp.array:
        """Compute offset array for collision buffer access"""
        offsets = np.arange(count + 1) * buffer_size
        return wp.array(offsets.astype(np.int32), dtype=wp.int32, device="cuda")

    def step(self):
        # Calculate the normals
        wp.launch(kernel=calculate_faceNormals, dim=self.numFaces, device="cuda",
                    inputs=[self.positions, self.faceIds, self.faceOffsets, self.meshCenters,
                            self.faceNormals, self.vertexNormals, self.vertexNormalCounts])
        wp.launch(kernel=calculate_vertex_normals, dim=self.numVertices, device="cuda",
                    inputs=[self.vertexNormals, self.vertexNormalCounts])
        
        # Update BVH and detect collisions
        self.update_bvh()
        self.detect_collisions()
        
        dt = DT / SUB_STEPS
        for _ in range(SUB_STEPS):
            wp.launch(kernel = integrate, dim=self.numVertices, device = "cuda",
                        inputs = [self.positions, self.prev_positions, self.invMasses, self.velocities, GRAVITY, dt])
            
            # Solve constraints with multiple Jacobi passes
            self.iterative_jacobi_solve(device="cuda", kernel=solve_edge_constraints, dim=self.numEdges,
                                    inputs=[self.positions, self.invMasses, self.edgeIds, self.edgeLengths, self.edgeCompliance/(dt**2), self.corrections])
            self.iterative_jacobi_solve(device="cuda", kernel=solve_volume_constraints, dim=self.numTets,
                                    inputs=[self.positions, self.invMasses, self.tetIds, self.restVolumes, self.volCompliance/(dt**2), self.corrections])
            # Solve constraints with multiple Jacobi passes
            self.iterative_jacobi_solve(device="cuda", kernel=solve_collision_constraints, dim=self.numVertices,
                                    inputs=[self.positions, self.invMasses, self.faceIds,
                                           self.vertex_colliding_triangles, self.vertex_collision_offsets,
                                           self.vertex_collision_counts, self.vertex_collision_distances,
                                           self.collisionCompliance/(dt**2), self.corrections])
            
            wp.launch(kernel = update_velocity, dim=self.numVertices, device = "cuda",
                        inputs = [self.positions, self.prev_positions, self.velocities, dt])
        
        # Transfer data between CPU and GPU
        meshCenters = []
        vertices = self.positions.numpy()
        fNormals = self.faceNormals.numpy()
        vNormals = self.vertexNormals.numpy()
        vOff = self.vertexOffsets.numpy()
        fOff = self.faceOffsets.numpy()
        for idx, mesh in enumerate(self.meshes):
            mesh.vertices = vertices[vOff[idx] : vOff[idx+1]]
            mesh.faceNormals = fNormals[fOff[idx] : fOff[idx+1]]
            mesh.vertex_normals = vNormals[vOff[idx] : vOff[idx+1]]
            meshCenters.append(mesh.center)
        self.meshCenters = wp.array(np.array(meshCenters).astype(np.float32), dtype=wp.vec3, device="cuda")

    def iterative_jacobi_solve(self, device, kernel, inputs, dim, numIterations=5, jacobiScale = JACOBISCALE):
        for _ in range(numIterations):
            # Zero out corrections from previous iteration
            inputs[-1].zero_()
            wp.launch(kernel=kernel, dim=dim, inputs=inputs, device=device)
            wp.launch(kernel=add_corrections,dim=inputs[0].shape[0],
                inputs=[inputs[0], inputs[-1], jacobiScale],
                device=device)

    def detect_collisions(self):
        # Initialize collision data
        wp.launch(
            kernel=init_collision_data_kernel,
            dim=self.numFaces,
            inputs=[
                self.collisionThreshold,
                self.triangle_collision_counts,
                self.triangle_collision_distances
            ],
            device="cuda"
        )
        
        # Clear vertex collision counts
        wp.launch(
            kernel=clear_collision_counts,
            dim=self.numVertices,
            inputs=[self.vertex_collision_counts],
            device="cuda"
        )
        
        # Detect vertex-triangle collisions
        wp.launch(
            kernel=detect_vertex_triangle_collisions,
            dim=self.numVertices,
            inputs=[
                self.positions,
                self.velocities,
                self.BVH.id,
                self.faceIds,
                self.collisionThreshold,  # This is now the base threshold
                DT,                       # Pass timestep
                self.vertex_colliding_triangles,
                self.vertex_collision_offsets,
                self.vertex_collision_counts,
                self.vertex_collision_distances,
                self.triangle_colliding_vertices,
                self.triangle_collision_offsets,
                self.triangle_collision_counts,
                self.triangle_collision_distances
            ],
            device="cuda"
        )
        
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
                        faceOffsets: wp.array(dtype=wp.int32),
                        meshCenters: wp.array(dtype=wp.vec3),
                        faceNormals: wp.array(dtype = wp.vec3),
                        vertexNormals: wp.array(dtype = wp.vec3),
                        vertexNormalCounts: wp.array(dtype = wp.int32)):
    fNr = wp.tid()
    for i in range(faceOffsets.shape[0] - 1):
        if faceOffsets[i] <= fNr and fNr < faceOffsets[i + 1]:
            mesh_id = i
    
    v0, v1, v2 = positions[faceIds[fNr, 0]], positions[faceIds[fNr, 1]], positions[faceIds[fNr, 2]]
    normal = wp.cross(v1 - v0, v2 - v0)
    length = wp.length(normal)
    if length > 0:
        normal = normal / length
        
    face_center = (v0 + v1 + v2) / 3.0
    if wp.dot(normal, face_center - meshCenters[mesh_id]) < 0:
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
def clear_collision_counts(counts: wp.array(dtype=wp.int32)):
    idx = wp.tid()
    counts[idx] = 0            

@wp.func
def triangle_closest_point_barycentric(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3):
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return wp.vec3(1.0, 0.0, 0.0)
    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return wp.vec3(0.0, 1.0, 0.0)
    vc = d1 * d4 - d3 * d2
    v = d1 / (d1 - d3)
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        return wp.vec3(1.0 - v, v, 0.0)
    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return wp.vec3(0.0, 0.0, 1.0)
    vb = d5 * d2 - d1 * d6
    w = d2 / (d2 - d6)
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        return wp.vec3(1.0 - w, 0.0, w)
    va = d3 * d6 - d5 * d4
    w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        return wp.vec3(0.0, 1.0 - w, w)
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return wp.vec3(1.0 - v - w, v, w)

@wp.func
def triangle_closest_point(a: wp.vec3, b: wp.vec3, c: wp.vec3, p: wp.vec3):
    """Compute closest point on triangle to point p"""
    ab = b - a
    ac = c - a
    ap = p - a
    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        feature_type = TRI_CONTACT_FEATURE_VERTEX_A
        bary = wp.vec3(1.0, 0.0, 0.0)
        return a, bary, feature_type
    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        feature_type = TRI_CONTACT_FEATURE_VERTEX_B
        bary = wp.vec3(0.0, 1.0, 0.0)
        return b, bary, feature_type
    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        feature_type = TRI_CONTACT_FEATURE_VERTEX_C
        bary = wp.vec3(0.0, 0.0, 1.0)
        return c, bary, feature_type
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        feature_type = TRI_CONTACT_FEATURE_EDGE_AB
        bary = wp.vec3(1.0 - v, v, 0.0)
        return a + v * ab, bary, feature_type
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        v = d2 / (d2 - d6)
        feature_type = TRI_CONTACT_FEATURE_EDGE_AC
        bary = wp.vec3(1.0 - v, 0.0, v)
        return a + v * ac, bary, feature_type
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        v = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        feature_type = TRI_CONTACT_FEATURE_EDGE_BC
        bary = wp.vec3(0.0, 1.0 - v, v)
        return b + v * (c - b), bary, feature_type
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    feature_type = TRI_CONTACT_FEATURE_FACE_INTERIOR
    bary = wp.vec3(1.0 - v - w, v, w)
    return a + v * ab + w * ac, bary, feature_type

@wp.kernel
def init_collision_data_kernel(
    query_radius: float,
    triangle_collision_counts: wp.array(dtype=wp.int32),
    triangle_collision_distances: wp.array(dtype=wp.float32)
):
    tri_index = wp.tid()
    triangle_collision_counts[tri_index] = 0
    triangle_collision_distances[tri_index] = query_radius

@wp.kernel
def detect_vertex_triangle_collisions(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    face_ids: wp.array2d(dtype=wp.int32),
    threshold: float,
    dt: float,
    vertex_colliding_triangles: wp.array(dtype=wp.int32),
    vertex_collision_offsets: wp.array(dtype=wp.int32),
    vertex_collision_counts: wp.array(dtype=wp.int32),
    vertex_collision_distances: wp.array(dtype=wp.float32),
    triangle_colliding_vertices: wp.array(dtype=wp.int32),
    triangle_collision_offsets: wp.array(dtype=wp.int32),
    triangle_collision_counts: wp.array(dtype=wp.int32),
    triangle_collision_distances: wp.array(dtype=wp.float32)
):
    """Detect collisions between vertices and triangles with velocity-expanded thresholds"""
    vNr = wp.tid()
    
    vertex = positions[vNr]
    predicted_position = vertex + velocities[vNr] * dt
    
    # Calculate AABB for the vertex trajectory
    lower = wp.vec3(
        wp.min(vertex[0], predicted_position[0]),
        wp.min(vertex[1], predicted_position[1]),
        wp.min(vertex[2], predicted_position[2])
    )
    upper = wp.vec3(
        wp.max(vertex[0], predicted_position[0]),
        wp.max(vertex[1], predicted_position[1]),
        wp.max(vertex[2], predicted_position[2])
    )
    
    # Query BVH for potential collisions
    query = wp.mesh_query_aabb(mesh_id, lower, upper)
    
    # Track collisions for this vertex
    vertex_num_collisions = wp.int32(0)
    min_dist_to_tris = threshold
    
    # Check each potential triangle collision
    tri_index = wp.int32(0)
    while wp.mesh_query_aabb_next(query, tri_index):
        # Get triangle vertices
        t1 = face_ids[tri_index, 0]
        t2 = face_ids[tri_index, 1]
        t3 = face_ids[tri_index, 2]
        
        # Skip if vertex belongs to this triangle
        if vNr == t1 or vNr == t2 or vNr == t3:
            continue
            
        # Get triangle vertex positions
        v0 = positions[t1]
        v1 = positions[t2]
        v2 = positions[t3]
        
        # Compute triangle normal
        edge1 = v1 - v0
        edge2 = v2 - v0
        tri_normal = wp.normalize(wp.cross(edge1, edge2))
        
        # Compute signed distances
        signed_distance = wp.dot(vertex - v0, tri_normal)
        signed_predicted_distance = wp.dot(predicted_position - v0, tri_normal)
        
        if (signed_distance > 0.0 or abs(signed_distance) >= threshold) and \
           (signed_predicted_distance > 0.0 or abs(signed_predicted_distance) >= threshold):
            continue
            
        # Project point onto triangle plane
        projected_point = vertex
        if signed_distance < 0.0 and abs(signed_distance) < threshold:
            projected_point = vertex - tri_normal * signed_distance
        elif signed_predicted_distance < 0.0 and abs(signed_predicted_distance) < threshold:
            projected_point = predicted_position - tri_normal * signed_predicted_distance
            
        # Point in triangle test
        v0_to_v2 = v2 - v0
        v0_to_v1 = v1 - v0
        v0_to_p = projected_point - v0
        
        dot00 = wp.dot(v0_to_v2, v0_to_v2)
        dot01 = wp.dot(v0_to_v2, v0_to_v1)
        dot02 = wp.dot(v0_to_v2, v0_to_p)
        dot11 = wp.dot(v0_to_v1, v0_to_v1)
        dot12 = wp.dot(v0_to_v1, v0_to_p)
        
        denom = dot00 * dot11 - dot01 * dot01
        if abs(denom) > 0.0:
            u = (dot11 * dot02 - dot01 * dot12) / denom
            v = (dot00 * dot12 - dot01 * dot02) / denom
            if u >= 0.0 and v >= 0.0 and u + v <= 1.0:
                # Store vertex->triangle collision if buffer space available
                vertex_buffer_offset = vertex_collision_offsets[vNr]
                vertex_buffer_size = vertex_collision_offsets[vNr + 1] - vertex_buffer_offset
                
                min_dist_to_tris = wp.min(min_dist_to_tris, signed_distance)
                if vertex_num_collisions < vertex_buffer_size:
                    collision_idx = vertex_buffer_offset + vertex_num_collisions
                    vertex_colliding_triangles[collision_idx] = tri_index
                    vertex_collision_distances[collision_idx] = signed_distance
                
                vertex_num_collisions = vertex_num_collisions + 1
                
                # Store triangle->vertex collision
                wp.atomic_min(triangle_collision_distances, tri_index, signed_distance)
                tri_buffer_size = wp.int32(32)  # Fixed buffer size for now
                tri_num_collisions = wp.atomic_add(triangle_collision_counts, tri_index, 1)
                
                if tri_num_collisions < tri_buffer_size:
                    tri_buffer_offset = triangle_collision_offsets[tri_index]
                    triangle_colliding_vertices[tri_buffer_offset + tri_num_collisions] = vNr
    
    vertex_collision_counts[vNr] = vertex_num_collisions
    
@wp.kernel
def solve_collision_constraints(
    positions: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    face_ids: wp.array2d(dtype=wp.int32),
    vertex_colliding_triangles: wp.array(dtype=wp.int32),
    vertex_collision_offsets: wp.array(dtype=wp.int32),
    vertex_collision_counts: wp.array(dtype=wp.int32),
    vertex_collision_distances: wp.array(dtype=wp.float32),
    alpha: float,
    corrections: wp.array(dtype=wp.vec3)
):
    """Solve collision constraints for vertex-triangle face collisions"""
    vertex_idx = wp.tid()
    
    # Get collision count for this vertex
    num_collisions = vertex_collision_counts[vertex_idx]
    if num_collisions == 0:
        return
        
    vertex_pos = positions[vertex_idx]
    vertex_w = inv_masses[vertex_idx]
    
    # Process each collision
    offset = vertex_collision_offsets[vertex_idx]
    for i in range(num_collisions):
        tri_idx = vertex_colliding_triangles[offset + i]
        
        # Get triangle vertices
        t1_idx = face_ids[tri_idx, 0]
        t2_idx = face_ids[tri_idx, 1]
        t3_idx = face_ids[tri_idx, 2]
            
        # Get current positions for triangle vertices
        t1_pos = positions[t1_idx]
        t2_pos = positions[t2_idx]
        t3_pos = positions[t3_idx]
        
        # Compute triangle normal
        edge1 = t2_pos - t1_pos
        edge2 = t3_pos - t1_pos
        normal = wp.normalize(wp.cross(edge1, edge2))
        
        # Get current vertex position and compute new closest point
        vertex_pos = positions[vertex_idx]
        closest_p, bary, feature_type = triangle_closest_point(t1_pos, t2_pos, t3_pos, vertex_pos)
        
        # For now, only handle face feature
        #if feature_type != TRI_CONTACT_FEATURE_FACE_INTERIOR:
            #continue
            
        # Compute current signed distance
        to_vertex = vertex_pos - closest_p
        dist = wp.length(to_vertex)
        signed_dist = dist * wp.sign(wp.dot(to_vertex, normal))
        
        # Only process if penetrating (negative distance)
        if signed_dist >= 0.0:
            continue
            
        # Set up gradients
        grad_v = normal  # gradient for vertex
        grad_t1 = -bary[0] * normal  # gradient for triangle vertex 1
        grad_t2 = -bary[1] * normal  # gradient for triangle vertex 2
        grad_t3 = -bary[2] * normal  # gradient for triangle vertex 3
        
        # Compute denominator for XPBD
        t1_w = inv_masses[t1_idx]
        t2_w = inv_masses[t2_idx]
        t3_w = inv_masses[t3_idx]
        
        w_sum = vertex_w * wp.dot(grad_v, grad_v) + \
                t1_w * wp.dot(grad_t1, grad_t1) + \
                t2_w * wp.dot(grad_t2, grad_t2) + \
                t3_w * wp.dot(grad_t3, grad_t3)
                
        # Compute lambda for XPBD
        dlambda = (-signed_dist) / (w_sum + alpha)
        
        # Apply position corrections
        wp.atomic_add(corrections, vertex_idx, vertex_w * dlambda * grad_v)
        wp.atomic_add(corrections, t1_idx, t1_w * dlambda * grad_t1)
        wp.atomic_add(corrections, t2_idx, t2_w * dlambda * grad_t2)
        wp.atomic_add(corrections, t3_idx, t3_w * dlambda * grad_t3)