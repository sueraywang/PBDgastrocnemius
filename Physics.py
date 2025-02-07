# Physics.py
import numpy as np
import time
from collections import defaultdict

class PerformanceMetrics:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.constraint_times = defaultdict(list)
        self.step_times = []
        self.current_step_start = None
    
    def start_step(self):
        self.current_step_start = time.perf_counter()
    
    def end_step(self):
        if self.current_step_start is not None:
            self.step_times.append(time.perf_counter() - self.current_step_start)
            self.current_step_start = None
    
    def add_constraint_time(self, constraint_type, duration):
        self.constraint_times[constraint_type].append(duration)
    
    def get_statistics(self):
        stats = {
            'step_time': {
                'mean': np.mean(self.step_times) if self.step_times else 0,
                'std': np.std(self.step_times) if self.step_times else 0,
                'min': np.min(self.step_times) if self.step_times else 0,
                'max': np.max(self.step_times) if self.step_times else 0
            },
            'constraints': {}
        }
        
        for constraint_type, times in self.constraint_times.items():
            if times:
                stats['constraints'][constraint_type] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'total_calls': len(times)
                }
        
        return stats

# Physical constants
DENSITY = 1e3  # kg/m**3
GRAVITY = np.array([0.0, 0.0, 0.0])  # m/s**2
DT = 0.01  # second

# XPBD constants
SUB_STEPS = 10
POSSION_RATIO = 0.4
DEVIATORIC_COMPLIANCE = 2e-4
COLLISION_COMPLIANCE = 0.0

mu = 1/DEVIATORIC_COMPLIANCE
YOUNG_MODULUS = 2 * mu * (1 + POSSION_RATIO)
lamb = YOUNG_MODULUS * POSSION_RATIO / ((1 + POSSION_RATIO) * (1 - 2 * POSSION_RATIO))
VOLUME_COMPLIANCE = 1/lamb

class Simulator:
    def __init__(self, meshes=None, density=DENSITY, devCompliance=DEVIATORIC_COMPLIANCE, volCompliance=VOLUME_COMPLIANCE):
        self.meshes = meshes
        self.density = density
        self.devCompliance = devCompliance
        self.volCompliance = volCompliance
        self.frameCount = 0.0
        self.grads = np.zeros((4, 3))
        self.metrics = PerformanceMetrics()
        
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
        
        self.vertexOffsets = np.array(vertex_offsets).astype(np.int32)
        self.tetOffsets = np.array(tet_offsets).astype(np.int32)
        self.faceOffsets = np.array(face_offsets).astype(np.int32)
        
        # Vertices
        self.numVertices = len(all_vetices)
        self.positions = all_vetices.astype(np.float32)
        self.prev_positions = all_vetices.astype(np.float32)
        self.velocities = np.zeros((self.numVertices,3)).astype(np.float32)
        self.invMasses = np.zeros(self.numVertices).astype(np.float32)
        
        # Tetrahedrons
        self.numTets = len(all_tets)
        self.tetIds = all_tets.astype(np.int32)
        self.invRestVolume = np.zeros(self.numTets).astype(np.float32)
        self.invRestPoses = np.zeros((self.numTets, 3, 3)).astype(np.float32)
        
        self.init_mesh_physics()
    
    def init_mesh_physics(self):
        # Gather positions of the four vertices for all tetrahedra
        v0 = self.positions[self.tetIds[:, 0]]
        v1 = self.positions[self.tetIds[:, 1]]
        v2 = self.positions[self.tetIds[:, 2]]
        v3 = self.positions[self.tetIds[:, 3]]
        
        # Compute edge vectors for all tetrahedra
        edge1 = v1 - v0
        edge2 = v2 - v0
        edge3 = v3 - v0

        # Compute the rest state for all tetrahedra
        volumes = np.einsum('ij,ij->i', np.cross(edge1, edge2), edge3) / 6.0
        self.invRestVolume = 1.0 / volumes
        self.invRestPoses = np.linalg.inv(np.stack((edge1, edge2, edge3), axis=1))
        
        # Find fixed vertices
        mask_fixed = (np.abs(self.positions[:, 2] - 1.1) < 1e-5) | (np.abs(self.positions[:, 2] - 0.9) < 1e-5) | \
            (np.abs(self.positions[:, 1] + 0.1) < 1e-5) | (np.abs(self.positions[:, 1] - 0.1) < 1e-5)
        self.fixed_vertices = np.where(mask_fixed)[0]
        mask_moving = (np.abs(self.positions[:, 1] + 0.1) < 1e-5) | (np.abs(self.positions[:, 1] - 0.1) < 1e-5)
        self.moving_vertices = np.where(mask_moving)[0]
        
        # Add mass from all tetrahedra
        vMass = np.maximum(volumes, 0) * self.density / 4.0
        np.add.at(self.invMasses, self.tetIds.flatten(), np.repeat(vMass, 4))
        # Convert masses to inverse masses, change fixed vertices mass to 0
        self.invMasses = np.where(
            (self.invMasses > 0) & (~np.isin(np.arange(self.numVertices), list(self.fixed_vertices))),
            1.0 / self.invMasses,
            0.0
        )
        
    def step(self):
        self.metrics.start_step()
        
        fixed_mask = (np.isin(np.arange(self.numVertices), list(self.fixed_vertices)))
        moving_mask = (np.isin(np.arange(self.numVertices), list(self.moving_vertices)))
        non_fixed_mask = ~fixed_mask
        movement = np.array([0.0, 0.0, 0.0])
        if (self.frameCount < 200.0):
            movement = np.array([-0.0005, 0.0, 0.0])
        """
        elif ((self.frameCount % 300) < 150):
            movement = np.array([0.0, -0.0005, 0.0])
        else:
            movement = np.array([0.0, 0.0005, 0.0])
        """
        self.positions[moving_mask] += movement
        self.frameCount += 1
        
        # Integrate the free vertices
        dt = DT / SUB_STEPS
        for _ in range(SUB_STEPS):
            self.prev_positions = self.positions.copy()
            self.velocities[non_fixed_mask] += GRAVITY * dt
            self.positions[non_fixed_mask] += self.velocities[non_fixed_mask] * dt
            # Simple ground collisions
            for i in range(len(self.positions)):
                if (self.positions[i, 2] < 0.0):
                    self.positions[i] = self.prev_positions[i].copy()
                    self.positions[i, 2] = 0.0
            
            # Solve the constraints
            for i in range(len(self.meshes)):
                for tetNr in range(self.tetOffsets[i], self.tetOffsets[i+1]):
                    self.solve_material_constraint(tetNr, dt)
            
            self.velocities[non_fixed_mask] = (self.positions[non_fixed_mask] - self.prev_positions[non_fixed_mask]) / dt
        
        # Transfer data for rendering
        for idx, mesh in enumerate(self.meshes):
            mesh.vertices = self.positions[self.vertexOffsets[idx] : self.vertexOffsets[idx+1]]
        
        self.metrics.end_step()
        
    def solve_dev_constraint(self, tetNr, dt):
        vIds = self.tetIds[tetNr]
        # Compute deformation gradient
        F = self.invRestPoses[tetNr] @ (self.positions[vIds[1:]] - self.positions[vIds[0]])
        
        r_s = np.sqrt(np.dot(F[0], F[0]) + 
                     np.dot(F[1], F[1]) + 
                     np.dot(F[2], F[2]))
        r_s_inv = 1.0 / r_s
        
        C = r_s
        self.grads.fill(0)
        self.grads[1:] = r_s_inv * np.transpose(self.invRestPoses[tetNr]) @ F
        self.correct_position(tetNr, C, self.devCompliance, dt)

    def solve_hydro_constraint(self, tetNr, dt):
        vIds = self.tetIds[tetNr]
        # Compute deformation gradient
        F = self.invRestPoses[tetNr] @ (self.positions[vIds[1:]] - self.positions[vIds[0]])

        # Compute cross products and store in a matrix
        dF = np.vstack([
            np.cross(F[1], F[2]),
            np.cross(F[2], F[0]),
            np.cross(F[0], F[1])
        ])

        # Compute volume constraint
        vol = np.linalg.det(F)
        C = vol - 1.0 - self.volCompliance / self.devCompliance

        # Compute gradients
        self.grads.fill(0)
        self.grads[1:] = self.invRestPoses[tetNr].T @ dF  # Direct matrix multiplication

        self.correct_position(tetNr, C, self.volCompliance, dt)

    def correct_position(self, tetNr, C, compliance, dt):
        if C == 0.0:
            return

        vIds = self.tetIds[tetNr]

        # Compute self.grads[0] using NumPy sum for efficiency
        self.grads[0] = -np.sum(self.grads[1:], axis=0)

        # Compute w using np.einsum for efficient dot product computation
        inv_masses = self.invMasses[vIds]
        w = np.einsum('ij,ij->i', self.grads, self.grads) @ inv_masses

        if w == 0.0:
            return

        # Compute λ (correction factor)
        alpha = compliance * self.invRestVolume[tetNr] / (dt * dt)
        dlambda = -C / (w + alpha)

        # Update positions efficiently using NumPy broadcasting
        self.positions[vIds] += (inv_masses[:, None] * dlambda * self.grads)
        
    def solve_material_constraint(self, tetNr, dt):
        start_time = time.perf_counter()
        vIds = self.tetIds[tetNr]
        
        # Compute deformation gradient
        F = self.invRestPoses[tetNr] @ (self.positions[vIds[1:]] - self.positions[vIds[0]])
        
        # **Hydrostatic Constraint**
        dF = np.cross(F[[1, 2, 0]], F[[2, 0, 1]])  # Avoid explicit vstack (faster)
        C_hydro = np.linalg.det(F) - 1.0 - self.volCompliance / self.devCompliance
        dC_hydro = np.zeros((4, 3))
        dC_hydro[1:] = self.invRestPoses[tetNr].T @ dF

        # **Deviatoric Constraint**
        r_s = np.linalg.norm(F, ord='fro')  # Frobenius norm
        r_s_inv = 1.0 / r_s if r_s > 1e-8 else 0.0  # Avoid division by zero
        C_dev = r_s
        dC_dev = np.zeros((4, 3))
        dC_dev[1:] = r_s_inv * self.invRestPoses[tetNr].T @ F

        # **Compute combined correction**
        dC_hydro[0] = -np.sum(dC_hydro[1:], axis=0)
        dC_dev[0] = -np.sum(dC_dev[1:], axis=0)
        
        # **Compute weights using fused einsum**
        inv_masses = self.invMasses[vIds]
        w_hydro, w_dev = np.einsum('ij,ij->i', dC_hydro, dC_hydro) @ inv_masses, \
                        np.einsum('ij,ij->i', dC_dev, dC_dev) @ inv_masses

        if w_hydro == 0.0 and w_dev == 0.0:
            return

        # **Compute lambda correction factors**
        alpha_hydro = self.volCompliance * self.invRestVolume[tetNr] / (dt * dt)
        alpha_dev = self.devCompliance * self.invRestVolume[tetNr] / (dt * dt)
        
        dlambda_hydro = -C_hydro / (w_hydro + alpha_hydro) if w_hydro != 0.0 else 0.0
        dlambda_dev = -C_dev / (w_dev + alpha_dev) if w_dev != 0.0 else 0.0

        # **Apply both corrections in one step (avoids extra array allocations)**
        self.positions[vIds] += inv_masses[:, None] * (dlambda_hydro * dC_hydro + dlambda_dev * dC_dev)
        
        self.metrics.add_constraint_time('material', time.perf_counter() - start_time)
    
    def solve_coupled_constraint(self, tetNr, dt):
        start_time = time.perf_counter()
        vIds = self.tetIds[tetNr]
        
        # Compute deformation gradient
        F = self.invRestPoses[tetNr] @ (self.positions[vIds[1:]] - self.positions[vIds[0]])
        
        # **Hydrostatic Constraint**
        dF = np.cross(F[[1, 2, 0]], F[[2, 0, 1]])
        C_hydro = np.linalg.det(F) - 1.0 - self.volCompliance / self.devCompliance
        dC_hydro = np.zeros((4, 3))
        dC_hydro[1:] = self.invRestPoses[tetNr].T @ dF

        # **Deviatoric Constraint**
        r_s = np.linalg.norm(F, ord='fro')
        r_s_inv = 1.0 / r_s if r_s > 1e-8 else 0.0
        C_dev = r_s
        dC_dev = np.zeros((4, 3))
        dC_dev[1:] = r_s_inv * self.invRestPoses[tetNr].T @ F

        # **Compute combined correction**
        dC_hydro[0] = -np.sum(dC_hydro[1:], axis=0)
        dC_dev[0] = -np.sum(dC_dev[1:], axis=0)

        # Couple the constraints
        C_couple = np.array([C_hydro, C_dev])
        
        # Properly stack dC_couple
        dC_couple = np.vstack((dC_hydro.reshape(1, -1), 
                            dC_dev.reshape(1, -1)))

        # Compliance matrix (2x2) with proper scaling
        alpha_couple = np.array([[self.volCompliance, 0], 
                            [0, self.devCompliance]]) * self.invRestVolume[tetNr] / (dt * dt)

        # Construct mass matrix
        M_12_diag = np.repeat(self.invMasses[vIds], 3)
        M_12_inv = np.diag(M_12_diag)
        
        # Compute system matrix
        weight = dC_couple @ M_12_inv @ dC_couple.T
        
        # Add compliance terms
        W = weight + alpha_couple

        try:
            # Solve for Δλ
            delta_lambda = np.linalg.solve(W, -C_couple)
            
            # Apply Correction
            correction = (delta_lambda[0] * dC_hydro + delta_lambda[1] * dC_dev)
            self.positions[vIds] += self.invMasses[vIds][:, None] * correction
        except np.linalg.LinAlgError:
            # Handle singular matrix case
            pass
        
        self.metrics.add_constraint_time('coupled', time.perf_counter() - start_time)
    
    def get_performance_stats(self):
        return self.metrics.get_statistics()
    
    def reset_metrics(self):
        self.metrics.reset()

def rotate_fixed_vertices(vertices, angle, mask):
    # Get current radius and angle of each vertex
    relative_pos = vertices[mask] - np.array([0.0, 0.0, 1.0]) * vertices[mask, 2][:, np.newaxis]
    
    # Compute radius correctly (element-wise)
    radius = np.linalg.norm(relative_pos, axis=1)  # Shape (N,)

    # Compute angle safely (set zero for small radius values)
    current_angle = np.where(radius > 1e-5, np.atan2(vertices[mask, 1], vertices[mask, 0]), 0.0)

    # Compute new positions for masked vertices
    new_pos = np.zeros_like(vertices[mask])  # Shape (N,3)

    # Apply rotation: for z > 1.0
    z_mask = vertices[mask, 2] > 1.0
    new_pos[z_mask, 0] = radius[z_mask] * np.cos(current_angle[z_mask] + angle)
    new_pos[z_mask, 1] = radius[z_mask] * np.sin(current_angle[z_mask] + angle)
    new_pos[z_mask, 2] = vertices[mask, 2][z_mask]  # Keep Z unchanged

    # Apply rotation: for z <= 1.0
    new_pos[~z_mask, 0] = radius[~z_mask] * np.cos(current_angle[~z_mask] - angle)
    new_pos[~z_mask, 1] = radius[~z_mask] * np.sin(current_angle[~z_mask] - angle)
    new_pos[~z_mask, 2] = vertices[mask, 2][~z_mask]  # Keep Z unchanged

    # Update vertices in place
    vertices[mask] = new_pos

