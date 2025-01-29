# Physics.py
import numpy as np

# Physical constants
DENSITY = 1e3  # kg/m**3
GRAVITY = np.array([0.0, 0.0, -10.0])  # m/s**2
DT = 0.01  # second

# XPBD constants
SUB_STEPS = 5
DEVIATORIC_COMPLIANCE = 1e-5
VOLUME_COMPLIANCE = 0.0
COLLISION_COMPLIANCE = 0.0  # Stiffness for collision response

class Simulator:
    def __init__(self, meshes=None, density=DENSITY, devCompliance=DEVIATORIC_COMPLIANCE, volCompliance=VOLUME_COMPLIANCE):
        self.meshes = meshes
        self.density = density
        self.devCompliance = devCompliance
        self.volCompliance = volCompliance
        
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
        self.fixed_vertices = set()
        
        # Tetrahedrons
        self.numTets = len(all_tets)
        self.tetIds = all_tets.astype(np.int32)
        self.invRestVolume = np.zeros(self.numTets).astype(np.float32)
        self.invRestPoses = np.zeros((self.numTets, 3, 3)).astype(np.float32)
        
        # Per-mesh temporary matrices and properties
        self.P = np.zeros((3, 3)).astype(np.float32)
        self.F = np.zeros((3, 3)).astype(np.float32)
        self.dF = np.zeros((3, 3)).astype(np.float32)
        self.grads = np.zeros((4, 3)).astype(np.float32)
        
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
        dt = DT / SUB_STEPS
        for _ in range(SUB_STEPS):
            self.prev_positions = self.positions.copy()
            #mask = ~np.isin(np.arange(self.numVertices), list(self.fixed_vertices))
            self.velocities += GRAVITY * dt
            self.positions += self.velocities * dt
            # Simple ground collisions
            for i in range(len(self.positions)):
                if (self.positions[i, 2] < 0.0):
                    self.positions[i] = self.prev_positions[i].copy()
                    self.positions[i, 2] = 0.0
            
            for i in range(len(self.meshes)):
                for tetNr in range(self.tetOffsets[i], self.tetOffsets[i+1]):
                    self.solve_hydro_constraint(tetNr, dt)
                    self.solve_dev_constraint(tetNr, dt)
            
            self.velocities = (self.positions - self.prev_positions) / dt
        
        # Transfer data for rendering
        for idx, mesh in enumerate(self.meshes):
            mesh.vertices = self.positions[self.vertexOffsets[idx] : self.vertexOffsets[idx+1]]

    def solve_dev_constraint(self, tetNr, dt):
        vIds = self.tetIds[tetNr]
        edge1 = self.positions[vIds[1]] - self.positions[vIds[0]]
        edge2 = self.positions[vIds[2]] - self.positions[vIds[0]]
        edge3 = self.positions[vIds[3]] - self.positions[vIds[0]]
        self.P = np.array([edge1, edge2, edge3])
        self.F = self.invRestPoses[tetNr] @ self.P
        r_s = np.sqrt(np.dot(self.F[0], self.F[0]) + 
                     np.dot(self.F[1], self.F[1]) + 
                     np.dot(self.F[2], self.F[2]))
        r_s_inv = 1.0 / r_s
        
        C = r_s
        self.grads = np.zeros((4, 3))
        product = r_s_inv * np.transpose(self.invRestPoses[tetNr]) @ self.F
        self.grads[1] = product[0]
        self.grads[2] = product[1]
        self.grads[3] = product[2]
        self.correct_position(tetNr, C, self.devCompliance, dt)

    def solve_hydro_constraint(self, tetNr, dt):
        vIds = self.tetIds[tetNr]
        edge1 = self.positions[vIds[1]] - self.positions[vIds[0]]
        edge2 = self.positions[vIds[2]] - self.positions[vIds[0]]
        edge3 = self.positions[vIds[3]] - self.positions[vIds[0]]
        self.P = np.array([edge1, edge2, edge3])
        self.F = self.invRestPoses[tetNr] @ self.P
        self.dF[0] = np.cross(self.F[1], self.F[2])
        self.dF[1] = np.cross(self.F[2], self.F[0])
        self.dF[2] = np.cross(self.F[0], self.F[1])
        vol = np.linalg.det(self.F)
        
        C = vol - 1.0 - self.volCompliance / self.devCompliance
        self.grads = np.zeros((4, 3))
        product = np.transpose(self.invRestPoses[tetNr]) @ self.dF
        self.grads[1] = product[0]
        self.grads[2] = product[1]
        self.grads[3] = product[2]
        self.correct_position(tetNr, C, self.volCompliance, dt)

    def correct_position(self, tetNr, C, compliance, dt):
        if C == 0.0:
            return
        
        self.grads[0] = -self.grads[1] - self.grads[2] - self.grads[3]
        
        w = 0.0
        for i in range(4):
            id = self.tetIds[tetNr][i]
            w += np.dot(self.grads[i], self.grads[i]) * self.invMasses[id]
        
        if w == 0.0:
            return
        
        alpha = compliance * self.invRestVolume[tetNr] / dt / dt
        dlambda = -C / (w + alpha)
        
        for i in range(4):
            id = self.tetIds[tetNr][i]
            self.positions[id] += self.invMasses[id] * dlambda * self.grads[i]