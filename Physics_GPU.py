# Physics.py
import numpy as np
import warp as wp

# Physical constants
DENSITY = 1e3  # kg/m**3
GRAVITY = np.array([0.0, 0.0, 0.0])  # m/s**2
DT = 0.01  # second

# XPBD constants
SUB_STEPS = 20
POSSION_RATIO = 0.4
DEVIATORIC_COMPLIANCE = 2e-4
COLLISION_COMPLIANCE = 0.0
COLLISION_THRESHOLD = 0.01
DAMPING_CONSTANT = 0.1
QUERY_RADIUS = 0.01

ENABLE_DAMPING = False

mu = 1/DEVIATORIC_COMPLIANCE
YOUNG_MODULUS = 2 * mu * (1 + POSSION_RATIO)
lamb = YOUNG_MODULUS * POSSION_RATIO / ((1 + POSSION_RATIO) * (1 - 2 * POSSION_RATIO))
VOLUME_COMPLIANCE = 1/lamb

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
    def __init__(self, meshes=None, device="cuda", density=DENSITY, 
                 devCompliance=DEVIATORIC_COMPLIANCE, 
                 volCompliance=VOLUME_COMPLIANCE, 
                 collisionCompliance=COLLISION_COMPLIANCE, 
                 collisionThreshold=COLLISION_THRESHOLD,
                 queryRadius=QUERY_RADIUS):
        self.frameCount = 0.0
        self.meshes = meshes
        self.density = density
        self.devCompliance = devCompliance
        self.volCompliance = volCompliance
        self.collisionCompliance = collisionCompliance
        self.device = device
        
        # Collision detection parameters
        self.collision_buffer_size = 8
        self.queryRadius = queryRadius
        self.collisionThreshold = collisionThreshold
        
        if self.meshes is not None:
            self.init_simulator()
            
    def add_bodies(self, meshes):
        self.meshes = meshes
        self.init_simulator()

    def init_simulator(self):
        # Prepare data
        vertex_offsets = [0]
        tet_offsets = [0]
        face_offsets = [0]
        mesh_centers = []
        all_vetices = np.empty((0, 3))
        all_tets = np.empty((0, 4))
        all_faces = np.empty((0, 3))
        
        for mesh in self.meshes:
            vertex_offsets.append(vertex_offsets[-1] + len(mesh.vertices))
            tet_offsets.append(tet_offsets[-1] + len(mesh.tets))
            face_offsets.append(face_offsets[-1] + len(mesh.faces))
            mesh_centers.append(mesh.center)
            
            all_tets = np.concatenate((all_tets, mesh.tets + len(all_vetices)), axis=0)
            all_faces = np.concatenate((all_faces, mesh.faces + len(all_vetices)), axis=0)
            all_vetices = np.concatenate((all_vetices, mesh.vertices), axis=0)
        
        self.vertexOffsets = wp.array(np.array(vertex_offsets).astype(np.int32), dtype=wp.int32, device=self.device)
        self.tetOffsets = wp.array(np.array(tet_offsets).astype(np.int32), dtype=wp.int32, device=self.device)
        self.faceOffsets = wp.array(np.array(face_offsets).astype(np.int32), dtype=wp.int32, device=self.device)
        
        # Vertices
        self.numVertices = len(all_vetices)
        self.positions = wp.array(all_vetices.astype(np.float32), dtype = wp.vec3, device=self.device)
        self.prev_positions = wp.array(all_vetices.astype(np.float32), dtype = wp.vec3, device=self.device)
        self.corrections = wp.zeros(self.numVertices, dtype = wp.vec3, device=self.device)
        self.velocities = wp.zeros(self.numVertices, dtype=wp.vec3, device=self.device)
        self.invMasses = wp.zeros(self.numVertices, dtype=wp.float32, device=self.device)
        self.vertexNormals = wp.zeros(self.numVertices, dtype = wp.vec3, device=self.device)
        self.vertexNormalCounts = wp.zeros(self.numVertices, dtype=wp.int32, device=self.device)
        
        # Tetrahedrons
        self.numTets = len(all_tets)
        self.tetIds = wp.array(all_tets.astype(np.int32), dtype=wp.int32, device=self.device)
        self.invRestVolumes = wp.zeros(self.numTets, dtype = wp.float32, device=self.device)
        self.invRestPoses = wp.zeros(self.numTets, dtype = wp.mat33, device=self.device)
        
        # Faces
        self.numFaces = len(all_faces)
        self.faceIds = wp.array(all_faces.astype(np.int32), dtype=wp.int32, device=self.device)
        self.faceNormals = wp.zeros(self.numFaces, dtype = wp.vec3, device=self.device)
        self.meshCenters = wp.array(np.array(mesh_centers).astype(np.float32), dtype=wp.vec3, device=self.device)
        
        # Create BVH
        self.Mesh_BVH = wp.Mesh(points=self.positions, indices=wp.array(all_faces.flatten().astype(np.int32), dtype=wp.int32))
        
        # Compute tet rest states
        wp.launch(kernel=compute_tet_states, dim=self.numTets, device=self.device,
            inputs=[self.positions, self.tetIds, self.density, self.invRestVolumes, self.invRestPoses, self.invMasses])
        # Convert masses to inverse masses
        wp.launch(kernel=invert_array,dim=self.numVertices, device=self.device,
            inputs=[self.invMasses])
        wp.launch(kernel=fix_vertices,dim=self.numVertices, device=self.device,
            inputs=[self.positions, self.invMasses])
        
        masses = self.invMasses.numpy()
        zero_indices = np.flatnonzero(masses == 0)
        self.fixedVertices = wp.array(zero_indices.astype(np.int32), dtype=wp.int32, device=self.device)
        #self.moving_vertices = wp.empty(len(self.fixedVertices)//2, dtype=wp.int32, device=self.device)
        #wp.copy(dest=self.moving_vertices, src=self.fixedVertices, count=len(self.fixedVertices)//2)
        
        # Initialize collision detection buffers
        self.init_collision_buffers()
        
    def step(self):
        if (self.frameCount <= 120.0):
            wp.launch(kernel=rotate_fixed_vertices, dim=len(self.fixedVertices), device = "cuda",
                            inputs = [self.positions, self.fixedVertices, np.radians(0.5)])
            self.frameCount += 0.5
        """
        movement = wp.vec3(0.0, 0.0, 0.0)
        if (self.frameCount < 200.0):
            movement = wp.vec3(-0.001, 0.0, 0.0)
        wp.launch(kernel=move_cylinder, dim=len(self.moving_vertices), device=self.device,
                        inputs = [self.positions, self.moving_vertices, movement])
        self.frameCount += 1
        """
        
        # Calculate the normals
        wp.launch(kernel=calculate_faceNormals, dim=self.numFaces, device=self.device,
                    inputs=[self.positions, self.faceIds, self.faceOffsets, self.meshCenters,
                            self.faceNormals, self.vertexNormals, self.vertexNormalCounts])
        wp.launch(kernel=calculate_vertex_normals, dim=self.numVertices, device=self.device,
                    inputs=[self.vertexNormals, self.vertexNormalCounts])
        
        self.update_bvh()
        self.detect_collisions()
        
        # Get collision information from GPU
        vertex_counts = self.vertex_collision_counts.numpy()
        vertex_colliding_triangles = self.vertex_colliding_triangles.numpy()
        vertex_collision_offsets = self.vertex_collision_offsets.numpy()
        vertex_offsets = self.vertexOffsets.numpy()
        face_offsets = self.faceOffsets.numpy()
        
        # Collect colliding vertices and faces
        colliding_vertices = []
        colliding_faces = []
        
        for i in range(len(vertex_counts)):
            if vertex_counts[i] > 0:
                colliding_vertices.append(i)
                offset = vertex_collision_offsets[i]
                #for j in range(vertex_counts[i]):
                    #colliding_faces.append(vertex_colliding_triangles[offset + j])
        
        # Update collision visualization for each mesh
        for idx, mesh in enumerate(self.meshes):
            mesh.update_collision_visualization(
                colliding_vertices,
                colliding_faces,
                vertex_offsets[idx],
                face_offsets[idx]
            )
        
        dt = DT / SUB_STEPS
        for _ in range(SUB_STEPS):
            wp.launch(kernel=integrate, dim=self.numVertices, device=self.device,
                        inputs = [self.positions, self.prev_positions, self.invMasses, self.velocities, GRAVITY, dt])
            
            
            
            self.iterative_jacobi_solve(kernel=solve_material_constraints, dim=self.numTets, numIterations=5,
                            inputs=[self.positions, self.invMasses, self.tetIds, self.invRestPoses,
                                    self.invRestVolumes, self.volCompliance, self.devCompliance, 
                                    dt, 0, self.corrections])
            
            """
            self.iterative_jacobi_solve(kernel=solve_collision_constraints, dim=self.numVertices, numIterations=5, jacobiScale=0.2,
                                    inputs=[self.positions, self.invMasses, self.faceIds,
                                           self.vertex_colliding_triangles, self.vertex_collision_offsets,
                                           self.vertex_collision_counts, self.vertex_collision_distances,
                                           self.collisionCompliance/(dt**2), self.corrections])
            """
            
            collision_counts = self.vertex_collision_counts.numpy()
            colliding_vertices = np.nonzero(collision_counts)[0]
            
            # Process only vertices that have collisions
            for vertex_idx in colliding_vertices:
                wp.launch(
                    kernel=solve_collision_constraints_sequential_direct,
                    dim=1,
                    device=self.device,
                    inputs=[
                        self.positions, self.prev_positions, self.invMasses, self.faceIds,
                        self.vertex_colliding_triangles, self.vertex_collision_offsets,
                        self.vertex_collision_counts, self.meshCenters, self.faceOffsets,
                        dt, vertex_idx
                    ]
                )
            
            
            wp.launch(kernel=update_velocity, dim=self.numVertices, device=self.device,
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
        self.meshCenters = wp.array(np.array(meshCenters).astype(np.float32), dtype=wp.vec3, device=self.device)

    def update_bvh(self):
        wp.copy(self.Mesh_BVH.points, self.positions)
        # Refit BVH after updating vertices
        self.Mesh_BVH.refit()
    
    def init_collision_buffers(self):
        """Initialize arrays for storing collision information"""
        # Initialize inside vertices array
        self.inside_vertices = wp.zeros(self.numVertices, dtype=wp.int32, device=self.device)
        self.num_inside_vertices = wp.zeros(1, dtype=wp.int32, device=self.device)
        
        # Pre-allocate collision arrays with fixed buffer size per vertex/triangle
        vertex_collision_buffer_size = self.numVertices * self.collision_buffer_size
        triangle_collision_buffer_size = self.numFaces * self.collision_buffer_size
        
        # Initialize vertex collision arrays
        self.vertex_colliding_triangles = wp.zeros(vertex_collision_buffer_size, dtype=wp.int32, device=self.device)
        self.vertex_collision_counts = wp.zeros(self.numVertices, dtype=wp.int32, device=self.device)
        self.vertex_collision_distances = wp.zeros(vertex_collision_buffer_size, dtype=wp.float32, device=self.device)
        
        # Initialize triangle collision arrays
        self.triangle_colliding_vertices = wp.zeros(triangle_collision_buffer_size, dtype=wp.int32, device=self.device)
        self.triangle_collision_counts = wp.zeros(self.numFaces, dtype=wp.int32, device=self.device)
        self.triangle_collision_distances = wp.zeros(triangle_collision_buffer_size, dtype=wp.float32, device=self.device)
        
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
        return wp.array(offsets.astype(np.int32), dtype=wp.int32, device=self.device)
    
    def iterative_jacobi_solve(self, kernel, inputs, dim, numIterations=5, jacobiScale=JACOBISCALE):
        for _ in range(numIterations):
            inputs[-1].zero_()
            wp.launch(
                kernel=kernel,
                dim=dim,
                inputs=inputs,
                device=self.device
            )
            wp.launch(
                kernel=add_corrections,
                dim=inputs[0].shape[0],
                inputs=[inputs[0], inputs[-1], jacobiScale],
                device=self.device
            )

    #"""      
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
                device=self.device
            )
            
            # Clear vertex collision counts
            wp.launch(
                kernel=clear_collision_counts,
                dim=self.numVertices,
                inputs=[self.vertex_collision_counts],
                device=self.device
            )
            
            # Detect vertex-triangle collisions
            wp.launch(
                kernel=detect_vertex_triangle_collisions,
                dim=self.numVertices,
                inputs=[
                    self.positions,
                    self.velocities,
                    self.Mesh_BVH.id,
                    self.faceIds,
                    self.faceOffsets,
                    self.vertexOffsets,
                    self.queryRadius,
                    self.faceNormals,
                    self.vertex_colliding_triangles,
                    self.vertex_collision_offsets,
                    self.vertex_collision_counts,
                    self.vertex_collision_distances,
                    self.triangle_colliding_vertices,
                    self.triangle_collision_offsets,
                    self.triangle_collision_counts,
                    self.triangle_collision_distances
                ],
                device=self.device
            )
    """

    def detect_collisions(self):
        # Clear collision counts and inside counter
        wp.launch(
            kernel=clear_collision_counts,
            dim=self.numVertices,
            inputs=[self.vertex_collision_counts],
            device=self.device
        )
        self.num_inside_vertices.zero_()
        
        # Detect inside vertices and compact them
        wp.launch(
            kernel=detect_and_compact_inside_vertices,
            dim=self.numVertices,
            inputs=[
                self.positions,
                self.velocities,
                self.faceIds,
                self.vertexOffsets,
                self.faceOffsets,
                self.inside_vertices,
                self.num_inside_vertices,
                self.BVH.id
            ],
            device=self.device
        )
        
        # Get number of inside vertices
        num_inside = self.num_inside_vertices.numpy()[0]
        
        if num_inside > 0:
            # Find collision triangles for inside vertices only
            wp.launch(
                kernel=find_collision_triangles,
                dim=int(num_inside),
                inputs=[
                    self.positions,
                    self.velocities,
                    self.faceIds,
                    self.faceNormals,
                    self.inside_vertices,
                    self.vertexOffsets,
                    self.faceOffsets,
                    self.vertex_colliding_triangles,
                    self.vertex_collision_offsets,
                    self.vertex_collision_counts,
                    self.queryRadius,
                    self.Mesh_BVH.id
                ],
                device=self.device
            )
    """
@wp.kernel
def invert_array(invMasses: wp.array(dtype=float)):
    mNr = wp.tid()
    invMasses[mNr] = 1.0 / invMasses[mNr]

@wp.kernel
def fix_vertices(vertices: wp.array(dtype=wp.vec3),
                 invMasses: wp.array(dtype=wp.float32)):
    vNr = wp.tid()
    if (wp.abs(vertices[vNr][2] - 1.1) < 1e-5) or \
        (wp.abs(vertices[vNr][2] - 0.9) < 1e-5) or \
        (wp.abs(vertices[vNr][1] - 0.1) < 1e-5) or \
        (wp.abs(vertices[vNr][1] + 0.1) < 1e-5):
        invMasses[vNr] = 0.0
        
@wp.kernel
def rotate_fixed_vertices(vertices: wp.array(dtype=wp.vec3),
                          fixedId: wp.array(dtype=wp.int32),
                          angle: wp.float32):
    vNr = fixedId[wp.tid()]
    # Get current radius and angle of each vertex
    relative_pos = vertices[vNr] - wp.vec3(0.0, 0.0, vertices[vNr][2])
    radius = wp.length(relative_pos)
    if radius > 1e-5:
        current_angle = wp.atan2(vertices[vNr][1], vertices[vNr][0])

    if vertices[vNr][2] > 1.0:
        # New position maintains same radius but at new angle
        new_pos = wp.vec3(
            radius * wp.cos(current_angle + angle),
            radius * wp.sin(current_angle + angle),
            vertices[vNr][2]  # Keep Z coordinate unchanged
        )
    else: 
        new_pos = wp.vec3(
            radius * wp.cos(current_angle - angle),
            radius * wp.sin(current_angle - angle),
            vertices[vNr][2]  # Keep Z coordinate unchanged
        )

    movement = new_pos - vertices[vNr]
    vertices[vNr] += movement
    
@wp.kernel
def move_cylinder(vertices: wp.array(dtype=wp.vec3),
                  fixedId: wp.array(dtype=wp.int32),
                  movement: wp.vec3):
    vNr = fixedId[wp.tid()]
    vertices[vNr] += movement

@wp.kernel
def compute_tet_states(vertices: wp.array(dtype=wp.vec3),
                        tetIds: wp.array2d(dtype=wp.int32),
                        density: float,
                        invTetVolumes: wp.array(dtype=wp.float32),
                        invTetPoses: wp.array(dtype=wp.mat33),
                        vertexMasses: wp.array(dtype=wp.float32)):
    
    tetNr = wp.tid()
    v0 = vertices[tetIds[tetNr, 0]]
    v1 = vertices[tetIds[tetNr, 1]]
    v2 = vertices[tetIds[tetNr, 2]]
    v3 = vertices[tetIds[tetNr, 3]]
    # Calculate edges
    e1 = v1 - v0
    e2 = v2 - v0
    e3 = v3 - v0
    inv_rest_pose = wp.inverse(wp.mat33(e1, e2, e3))
    # Compute volume using cross product
    volume = wp.abs(wp.dot(wp.cross(v1 - v0, v2 - v0), v3 - v0)) / 6.0
    invTetVolumes[tetNr] = 1.0 / volume
    invTetPoses[tetNr] = inv_rest_pose
    # Accumulate the masses
    avg_mass = wp.max(density * volume / 4.0, 0.0)
    for i in range(4):
        wp.atomic_add(vertexMasses, tetIds[tetNr, i], avg_mass)
                    
@wp.kernel
def solve_material_constraints(vertices: wp.array(dtype=wp.vec3),
                        invMasses: wp.array(dtype=wp.float32),
                        tetIds: wp.array2d(dtype=wp.int32),
                        invRestPoses: wp.array(dtype=wp.mat33),
                        invRestVolumes: wp.array(dtype=wp.float32),
                        volCompliance: wp.float32,
                        devCompliance: wp.float32,
                        dt: wp.float32,
                        firstTetId: wp.int32,
                        corrections: wp.array(dtype = wp.vec3)):
                    
    tetNr = firstTetId + wp.tid()
    i0 = tetIds[tetNr, 0]
    i1 = tetIds[tetNr, 1]
    i2 = tetIds[tetNr, 2]
    i3 = tetIds[tetNr, 3]
    # Get weights
    w0 = invMasses[i0]
    w1 = invMasses[i1]
    w2 = invMasses[i2]
    w3 = invMasses[i3]
    # Calculate current state
    e1 = vertices[i1] - vertices[i0]
    e2 = vertices[i2] - vertices[i0]
    e3 = vertices[i3] - vertices[i0]
    current_pose = wp.mat33(e1, e2, e3)
    F = current_pose @ invRestPoses[tetNr]

    f1 = wp.vec3(F[0, 0], F[1, 0], F[2, 0])
    f2 = wp.vec3(F[0, 1], F[1, 1], F[2, 1])
    f3 = wp.vec3(F[0, 2], F[1, 2], F[2, 2])
    
    C = float(0.0)
    dC = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    compliance = float(0.0)
    
    for term in range(0, 2):
        if term == 0:
            C = wp.determinant(F) - 1.0 - volCompliance / devCompliance
            dC = wp.mat33(wp.cross(f2, f3), wp.cross(f3, f1), wp.cross(f1, f2))
            compliance = volCompliance
        elif term == 1:
            C = wp.sqrt(wp.dot(f1, f1) + wp.dot(f2, f2) + wp.dot(f3, f3))
            dC = 1.0/C * wp.mat33(f1, f2, f3)
            compliance = devCompliance
    
        if C != 0.0:
            dP = dC * wp.transpose(invRestPoses[tetNr])
            grad1 = wp.vec3(dP[0][0], dP[1][0], dP[2][0])
            grad2 = wp.vec3(dP[0][1], dP[1][1], dP[2][1])
            grad3 = wp.vec3(dP[0][2], dP[1][2], dP[2][2])
            grad0 = -(grad1 + grad2 + grad3)  # More explicit negation
                
            w = (
                wp.dot(grad0, grad0) * w0
                + wp.dot(grad1, grad1) * w1
                + wp.dot(grad2, grad2) * w2
                + wp.dot(grad3, grad3) * w3
            )

            if w > 0.0:
                alpha = compliance * invRestVolumes[tetNr] / dt / dt
                dlambda = -C / (w + alpha)
                
                wp.atomic_add(corrections, i0, w0 * dlambda * grad0)
                wp.atomic_add(corrections, i1, w1 * dlambda * grad1)
                wp.atomic_add(corrections, i2, w2 * dlambda * grad2)
                wp.atomic_add(corrections, i3, w3 * dlambda * grad3)

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
    prevPositions[vNr] = wp.copy(positions[vNr])
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
    face_offsets: wp.array(dtype=wp.int32),   # Added face offsets
    vertex_offsets: wp.array(dtype=wp.int32),  # Added vertex offsets
    queryRadius: float,
    faceNormals: wp.array(dtype = wp.vec3),
    vertex_colliding_triangles: wp.array(dtype=wp.int32),
    vertex_collision_offsets: wp.array(dtype=wp.int32),
    vertex_collision_counts: wp.array(dtype=wp.int32),
    vertex_collision_distances: wp.array(dtype=wp.float32),
    triangle_colliding_vertices: wp.array(dtype=wp.int32),
    triangle_collision_offsets: wp.array(dtype=wp.int32),
    triangle_collision_counts: wp.array(dtype=wp.int32),
    triangle_collision_distances: wp.array(dtype=wp.float32)
):
    vNr = wp.tid()
    vertex_mesh_id = get_mesh_id(vNr, vertex_offsets)
    
    vertex = positions[vNr]
    
    # Calculate AABB for the vertex trajectory
    lower = wp.vec3(vertex[0] - queryRadius, vertex[1] - queryRadius, vertex[2] - queryRadius)
    upper = wp.vec3(vertex[0] + queryRadius, vertex[1] + queryRadius, vertex[2] + queryRadius)
    
    query = wp.mesh_query_aabb(mesh_id, lower, upper)
    vertex_num_collisions = wp.int32(0)
    tri_index = wp.int32(0)
    while wp.mesh_query_aabb_next(query, tri_index):
        # Get triangle's mesh ID
        triangle_mesh_id = get_mesh_id(tri_index, face_offsets)
        
        # Skip if same mesh (self-collision)
        if vertex_mesh_id == triangle_mesh_id:
            continue
            
        # Get triangle vertices
        t1 = face_ids[tri_index, 0]
        t2 = face_ids[tri_index, 1]
        t3 = face_ids[tri_index, 2]
        t1_pos = positions[t1]
        t2_pos = positions[t2]
        t3_pos = positions[t3]
        
        closest_p, bary, feature_type = triangle_closest_point(t1_pos, t2_pos, t3_pos, vertex)
        
        if (feature_type != TRI_CONTACT_FEATURE_FACE_INTERIOR):
            continue
        
        to_vertex = vertex - closest_p
        normal = faceNormals[tri_index]
        signed_dist = wp.dot(to_vertex, normal)
        vel = velocities[vNr]
        
        if signed_dist < 0.0 and wp.dot(vel, normal) < 0:
            vertex_buffer_offset = vertex_collision_offsets[vNr]
            vertex_buffer_size = vertex_collision_offsets[vNr + 1] - vertex_buffer_offset

            # record v-f collision to vertex
            if vertex_num_collisions < vertex_buffer_size:
                vertex_colliding_triangles[vertex_buffer_offset + vertex_num_collisions] = tri_index

            vertex_num_collisions = vertex_num_collisions + 1
            """
            wp.atomic_min(triangle_collision_distances, tri_index, signed_dist)
            tri_buffer_offset = triangle_collision_offsets[vNr]
            tri_buffer_size = triangle_collision_offsets[vNr + 1] - tri_buffer_offset
            tri_num_collisions = wp.atomic_add(triangle_collision_counts, tri_index, 1)

            if tri_num_collisions < tri_buffer_size:
                tri_buffer_offset = triangle_collision_offsets[tri_index]
                # record v-f collision to triangle
                triangle_colliding_vertices[tri_buffer_offset + tri_num_collisions] = vNr
                """
    vertex_collision_counts[vNr] = vertex_num_collisions
    
@wp.func
def get_mesh_id(index: int, offsets: wp.array(dtype=wp.int32)) -> int:
    """
    Binary search through offsets to find mesh ID
    index: vertex or face index to look up
    offsets: array of offsets marking start of each mesh's data
    returns: mesh ID (index into offsets array - 1)
    """
    left = int(0)
    right = int(offsets.shape[0] - 1)
    
    while left < right:
        mid = int((left + right) // 2)
        if offsets[mid] <= index:
            if offsets[mid + 1] > index:
                return mid
            left = mid + 1
        else:
            right = mid
            
    return left - 1

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

@wp.kernel
def solve_collision_constraints_sequential_direct(
    positions: wp.array(dtype=wp.vec3),
    prev_pos: wp.array(dtype=wp.vec3),
    inv_masses: wp.array(dtype=wp.float32),
    face_ids: wp.array2d(dtype=wp.int32),
    vertex_colliding_triangles: wp.array(dtype=wp.int32),
    vertex_collision_offsets: wp.array(dtype=wp.int32),
    vertex_collision_counts: wp.array(dtype=wp.int32),
    meshCenters: wp.array(dtype=wp.vec3),
    face_offsets: wp.array(dtype=wp.int32),
    dt: float,
    index: int  # Added index parameter for sequential execution
):
    vertex_idx = index
    
    # Get collision count for this vertex
    num_collisions = vertex_collision_counts[vertex_idx]
    if num_collisions == 0:
        return
        
    vertex_w = inv_masses[vertex_idx]
    
    # Process each collision
    offset = vertex_collision_offsets[vertex_idx]
    for i in range(num_collisions):
        tri_idx = vertex_colliding_triangles[offset + i]
        mesh_id = get_mesh_id(tri_idx, face_offsets)
        
        # Get triangle vertices
        t1_idx = face_ids[tri_idx, 0]
        t2_idx = face_ids[tri_idx, 1]
        t3_idx = face_ids[tri_idx, 2]
            
        # Get current positions
        vertex_pos = positions[vertex_idx]
        t1_pos = positions[t1_idx]
        t2_pos = positions[t2_idx]
        t3_pos = positions[t3_idx]
        vertex_pos_prev = prev_pos[vertex_idx]
        t1_pos_prev = prev_pos[t1_idx]
        t2_pos_prev = prev_pos[t2_idx]
        t3_pos_prev = prev_pos[t3_idx]
        
        edge1 = t2_pos - t1_pos
        edge2 = t3_pos - t1_pos
        normal = wp.normalize(wp.cross(edge1, edge2))
        
        length = wp.length(normal)
        if length > 0:
            normal = normal / length
            
        face_center = (t1_pos + t2_pos + t3_pos) / 3.0
        if wp.dot(normal, face_center - meshCenters[mesh_id]) < 0:
            normal = -normal
        
        # Compute closest point
        closest_p, bary, feature_type = triangle_closest_point(t1_pos, t2_pos, t3_pos, vertex_pos)
        
        # Compute signed distance
        to_vertex = vertex_pos - closest_p
        signed_dist = wp.dot(to_vertex, normal)
        
        # Only process if penetrating
        if signed_dist >= 0.0:
            continue
            
        # Set up gradients
        grad_v = normal
        grad_t1 = -bary[0] * normal
        grad_t2 = -bary[1] * normal
        grad_t3 = -bary[2] * normal
        
        # Get inverse masses
        t1_w = inv_masses[t1_idx]
        t2_w = inv_masses[t2_idx]
        t3_w = inv_masses[t3_idx]
        
        # Compute denominator
        w_sum = vertex_w * wp.dot(grad_v, grad_v) + \
                t1_w * wp.dot(grad_t1, grad_t1) + \
                t2_w * wp.dot(grad_t2, grad_t2) + \
                t3_w * wp.dot(grad_t3, grad_t3)
        
        alpha = wp.float32(COLLISION_COMPLIANCE / (dt*dt))
        if (ENABLE_DAMPING):   
            # Compute lambda
            beta = wp.float32(DAMPING_CONSTANT * dt*dt)
            gamma = wp.float32((alpha * beta)/dt)
            dlambdav = -(signed_dist + gamma * wp.dot(grad_v, vertex_pos - vertex_pos_prev)) / ((1.0 + gamma) * w_sum + alpha)
            dlambdat1 = -(signed_dist + gamma * wp.dot(grad_t1, t1_pos - t1_pos_prev)) / ((1.0 + gamma) * w_sum + alpha)
            dlambdat2 = -(signed_dist + gamma * wp.dot(grad_t2, t2_pos - t2_pos_prev)) / ((1.0 + gamma) * w_sum + alpha)
            dlambdat3 = -(signed_dist + gamma * wp.dot(grad_t3, t3_pos - t3_pos_prev)) / ((1.0 + gamma) * w_sum + alpha)
            
            positions[vertex_idx] += vertex_w * dlambdav * grad_v
            positions[t1_idx] += t1_w * dlambdat1 * grad_t1
            positions[t2_idx] += t2_w * dlambdat2 * grad_t2
            positions[t3_idx] += t3_w * dlambdat3 * grad_t3
        else:
            dlambda = (-signed_dist) / (w_sum + alpha)
        
            positions[vertex_idx] += vertex_w * dlambda * grad_v
            positions[t1_idx] += t1_w * dlambda * grad_t1
            positions[t2_idx] += t2_w * dlambda * grad_t2
            positions[t3_idx] += t3_w * dlambda * grad_t3