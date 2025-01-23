# Physics.py
import numpy as np
import warp as wp

# Physical constants
DENSITY = 1000 # kg/m**3
GRAVITY = wp.vec3(0.0, 0.0, -10.0)  # m/s**2
DT = 0.01  # second

# XPBD constants
SUB_STEPS = 5
EDGE_COMPLIANCE = 1e-4
VOLUME_COMPLIANCE = 0.0
COLLISION_COMPLIANCE = 0.0
STATIC_COF = 0.2
DYNAMIC_COF = 0.1

# Parallelization parameters
JACOBISCALE = 0.25

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
        print(self.tetIds.shape)
        self.edgeLengths = wp.array(np.zeros(self.numEdges).astype(np.float32), dtype = float, device = "cuda")
        
        # Per-body temporary matrices and properties
        self.edgeCompliance = edgeCompliance
        self.volCompliance = volCompliance
        self.collision_constraints = [] # Store active collision constraints of certain mesh

class CollisionConstraint:
    def __init__(self, vertex_idx, trig_idx, triangle_mesh_idx):
        self.vertex_idx = vertex_idx
        self.trig_idx = trig_idx
        self.triangle_mesh_idx = triangle_mesh_idx

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
        wp.launch(kernel=compute_tet_volumes, dim=body.numTets,
            inputs=[body.positions, body.tetIds, body.restVolumes])
        # Compute masses
        wp.launch(kernel=compute_masses, dim=body.numTets,
            inputs=[body.tetIds, body.density, body.restVolumes, body.invMasses])
        # Compute edge lengths
        wp.launch(kernel=compute_edge_lengths, dim=body.numEdges,
            inputs=[body.positions, body.edgeIds, body.edgeLengths])
        
        # Convert masses to inverse masses
        body.invMasses = wp.array(1.0 / body.invMasses.numpy(), dtype=float)

    def step(self):
        dt = DT / SUB_STEPS
        for _ in range(SUB_STEPS):
            for body in self.bodies:
                wp.launch(kernel = integrate, 
                    inputs = [body.invMasses, body.prev_positions, body.positions, body.velocities, GRAVITY, dt], 
                    dim = body.numVertices, device = "cuda")

            for body in self.bodies: # It makes sure that all bodies' current position is predicted before we solve constraints
                # Solve edges with multiple Jacobi passes
                jacobi_solve_volumes(
                    device="cuda",
                    positions_wp=body.positions,
                    inv_masses_wp=body.invMasses,
                    tet_ids_wp=body.tetIds,
                    rest_volumes_wp=body.restVolumes,
                    vol_compliance=body.volCompliance,
                    dt=dt,
                    num_iterations=5  # tweak as desired
                )
                jacobi_solve_edges(
                    device="cuda",
                    positions_wp=body.positions,
                    inv_masses_wp=body.invMasses,
                    edge_ids_wp=body.edgeIds,
                    edge_lengths_wp=body.edgeLengths,
                    edge_compliance=body.edgeCompliance,
                    dt=dt,
                    num_iterations=5  # tweak as desired
                )
                # Copy results back to CPU
                body.mesh.vertices = body.positions.numpy()
            
            for body in self.bodies: # It makes sure that all bodies' constrains are solved before we update velocity
                wp.launch(kernel = update_velocity, 
                inputs = [body.prev_positions, body.positions, body.velocities, dt], dim = body.numVertices, device = "cuda")
    
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
    
    def solve_collision_constraints(self, body, staticCOF, dt):
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
            correction = w_vertex * d_lambda * normal

            # Calculate tangential frictions
            d_p = body.positions[constraint.vertex_idx] - body.prev_positions[constraint.vertex_idx]
            d_tangent = d_p - np.dot(d_p, normal) * normal
            d_lambda_T = np.linalg.norm(d_tangent) / (total_weight + alpha)
            if (d_lambda_T < staticCOF * d_lambda):
                correction -= d_tangent
            
            # Apply corrections
            body.positions[constraint.vertex_idx] += correction
            for i in range(3):
                trig_body.positions[trig[i]] -= (trig_body.invMasses[trig[i]] * d_lambda * normal)

            constraint.d_lambda = d_lambda
            constraint.weight = total_weight

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