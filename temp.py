@wp.kernel
def compute_tet_volumes(
    positions: wp.array(dtype=wp.vec3),
    tetIds: wp.array2d(dtype=wp.int32),
    volumes: wp.array(dtype=wp.float32)):
    
    tetNr = wp.tid()
    
    # Get vertex positions
    v0 = positions[tetIds[tetNr, 0]]
    v1 = positions[tetIds[tetNr, 1]]
    v2 = positions[tetIds[tetNr, 2]]
    v3 = positions[tetIds[tetNr, 3]]
    
    # Compute edges
    edge1 = v1 - v0
    edge2 = v2 - v0
    edge3 = v3 - v0
    
    # Compute volume using cross product
    volume = wp.dot(wp.cross(edge1, edge2), edge3) / 6.0
    volumes[tetNr] = volume
        
@wp.kernel
def compute_masses(
    tetIds: wp.array2d(dtype=wp.int32),
    density: float,
    volumes: wp.array(dtype=wp.float32),
    inv_masses: wp.array(dtype=wp.float32)):
    
    tetNr = wp.tid()
    # Compute vertex mass contribution
    vertex_mass = wp.max(volumes[tetNr] * density / 4.0, 0.0)
    for i in range(4):
        wp.atomic_add(inv_masses, tetIds[tetNr, i], vertex_mass)

# Kernel for computing edge lengths
@wp.kernel
def compute_edge_lengths(
    positions: wp.array(dtype=wp.vec3),
    edgeIds: wp.array2d(dtype=wp.int32),
    edgeLengths: wp.array(dtype=wp.float32)):
    edgeNr = wp.tid()
    id0 = edgeIds[edgeNr, 0]
    id1 = edgeIds[edgeNr, 1]
    v0 = positions[id0]
    v1 = positions[id1]
    edgeLengths[edgeNr] = wp.length(v1 - v0)
                    
@wp.kernel
def solve_edge_constraints(
        positions: wp.array(dtype=wp.vec3),
        inv_masses: wp.array(dtype=wp.float32),
        edge_ids: wp.array2d(dtype=wp.int32),
        edge_lengths: wp.array(dtype=wp.float32),
        edge_compliance: wp.float32,
        dt: wp.float32,
        corrections: wp.array(dtype = wp.vec3)):
    tid = wp.tid()
    
    id0 = edge_ids[tid, 0]
    id1 = edge_ids[tid, 1]
    w0 = inv_masses[id0]
    w1 = inv_masses[id1]
    w = w0 + w1
    
    if w > 0.0:
        p0 = positions[id0]
        p1 = positions[id1]
        grad = p0 - p1
        len = wp.length(grad)
        
        if len > 0.0:
            grad = grad / len
            rest_len = edge_lengths[tid]
            C = len - rest_len
            alpha = edge_compliance / (dt * dt)
            dlambda = -C / (w + alpha)
            
            wp.atomic_add(corrections, id0, w0 * dlambda * grad)
            wp.atomic_add(corrections, id1, -w1 * dlambda * grad)

@wp.kernel
def solve_volume_constraints(
        positions: wp.array(dtype=wp.vec3),
        inv_masses: wp.array(dtype=wp.float32),
        tet_ids: wp.array2d(dtype=wp.int32),
        rest_volumes: wp.array(dtype=wp.float32),
        vol_compliance: wp.float32,
        dt: wp.float32,
        corrections: wp.array(dtype = wp.vec3)):
    tid = wp.tid()  # Tetrahedron ID
    
    # Get vertices of tetrahedron
    i0 = tet_ids[tid, 0]
    i1 = tet_ids[tid, 1]
    i2 = tet_ids[tid, 2]
    i3 = tet_ids[tid, 3]
    
    p0 = positions[i0]
    p1 = positions[i1]
    p2 = positions[i2]
    p3 = positions[i3]
    
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
    w += inv_masses[i0] * wp.dot(grad0, grad0)
    w += inv_masses[i1] * wp.dot(grad1, grad1)
    w += inv_masses[i2] * wp.dot(grad2, grad2)
    w += inv_masses[i3] * wp.dot(grad3, grad3)
    
    if w > 0.0:
        C = vol - rest_volumes[tid]
        alpha = vol_compliance / dt / dt
        dlambda = -C / (w + alpha)
        
        # Apply corrections using atomic operations
        wp.atomic_add(corrections, i0, inv_masses[i0] * dlambda * grad0)
        wp.atomic_add(corrections, i1, inv_masses[i1] * dlambda * grad1)
        wp.atomic_add(corrections, i2, inv_masses[i2] * dlambda * grad2)
        wp.atomic_add(corrections, i3, inv_masses[i3] * dlambda * grad3)

@wp.kernel
def add_corrections(
        positions: wp.array(dtype=wp.vec3),
        corrections: wp.array(dtype = wp.vec3),
        scale: float):
    vNr = wp.tid()
    positions[vNr] = positions[vNr] + corrections[vNr] * scale
    
def jacobi_solve_volumes(
    device,
    positions_wp,        # (V,) wp.vec3
    inv_masses_wp,       # (V,) wp.float32
    tet_ids_wp,          # (T,4) wp.int32
    rest_volumes_wp,     # (T,) wp.float32
    vol_compliance,     # float
    dt,                  # float
    num_iterations=10     # how many Jacobi passes
):
    # Create a corrections buffer
    V = positions_wp.shape[0]
    corrections_wp = wp.zeros(V, dtype=wp.vec3, device=device)

    for _ in range(num_iterations):
        # Zero out corrections from previous iteration
        corrections_wp.zero_()

        # Launch kernel to compute all partial corrections
        wp.launch(
            kernel=solve_volume_constraints,
            dim=tet_ids_wp.shape[0],
            inputs=[
                positions_wp,
                inv_masses_wp,
                tet_ids_wp,
                rest_volumes_wp,
                vol_compliance,
                dt,
                corrections_wp
            ],
            device=device
        )

        # Launch kernel to add corrections
        wp.launch(
            kernel=add_corrections,
            dim=V,
            inputs=[positions_wp, corrections_wp, JACOBISCALE],
            device=device
        )
        
def jacobi_solve_edges(
    device,
    positions_wp,        # (V,) wp.vec3
    inv_masses_wp,       # (V,) wp.float32
    edge_ids_wp,         # (E,2) wp.int32
    edge_lengths_wp,     # (E,) wp.float32
    edge_compliance,     # float
    dt,                  # float
    num_iterations=10     # how many Jacobi passes
):
    # Create a corrections buffer
    V = positions_wp.shape[0]
    corrections_wp = wp.zeros(V, dtype=wp.vec3, device=device)

    for _ in range(num_iterations):
        # Zero out corrections from previous iteration
        corrections_wp.zero_()

        # Launch kernel to compute all partial corrections
        wp.launch(
            kernel=solve_edge_constraints,
            dim=edge_ids_wp.shape[0],
            inputs=[
                positions_wp,
                inv_masses_wp,
                edge_ids_wp,
                edge_lengths_wp,
                edge_compliance,
                dt,
                corrections_wp
            ],
            device=device
        )

        # Launch kernel to add corrections
        wp.launch(
            kernel=add_corrections,
            dim=V,
            inputs=[positions_wp, corrections_wp, JACOBISCALE],
            device=device
        )

@wp.kernel
def integrate(
        invMass: wp.array(dtype = float),
        prevPos: wp.array(dtype = wp.vec3),
        pos: wp.array(dtype = wp.vec3),
        vel: wp.array(dtype = wp.vec3),
        acceleration: wp.vec3,
        dt: float):
    vNr = wp.tid()

    prevPos[vNr] = pos[vNr]
    if invMass[vNr] > 0.0:
        vel[vNr] = vel[vNr] + acceleration * dt
        pos[vNr] = pos[vNr] + vel[vNr] * dt
    
    # Simple ground collisions
    if (pos[vNr][2] < 0.0):
        pos[vNr] = prevPos[vNr]
        pos[vNr][2] = 0.0
        
@wp.kernel
def update_velocity(
        prevPos: wp.array(dtype = wp.vec3),
        pos: wp.array(dtype = wp.vec3),
        vel: wp.array(dtype = wp.vec3),
        dt: float):
    vNr = wp.tid()
    vel[vNr] = (pos[vNr] - prevPos[vNr]) / dt
 
@wp.kernel   
def calculate_face_normals(pos: wp.array(dtype = wp.vec3),
                           face_normals: wp.array(dtype = wp.vec3)):  
    fNr = wp.tid()
    v0, v1, v2 = pos[face[0]], self.vertices[face[1]], self.vertices[face[2]]
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
            length = np.linalg.norm(vertex_normals[i])
            if length > 0:
                vertex_normals[i] = vertex_normals[i] / length
                
    return vertex_normals