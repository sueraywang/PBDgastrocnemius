# Physics.py
# Objects/Constrains are created here
import numpy as np

# Physical constants
DENSITY = 1000.0
GRAVITY = np.array([0.0, 0.0, -90.8])
DT = 1/60

# XPBD constants
SUB_STEPS = 1
DEVIATORIC_COMPLIANCE = 1.0/100000.0
VOLUME_COMPLIANCE = 0.0

class Mesh:
    def __init__(self, vertices, tets, edges, density=DENSITY, devCompliance = DEVIATORIC_COMPLIANCE, volCompliance = VOLUME_COMPLIANCE):
        # Vertices
        self.numVertices = len(vertices)
        self.positions = vertices #(#vers, 3) array
        self.prev_positions = self.positions.copy()
        self.velocities = np.zeros_like(vertices) #(#vers, 3) array
        # Tetrahedrons
        self.numTets = len(tets)
        self.tetIds = tets #(#tets, 4) array
        self.density = density # float
        self.invRestVolume = np.zeros(self.numTets) # (#tets, 1) array
        self.invRestPoses = np.zeros((self.numTets, 3, 3)) # (#tets, column major (3, 3) matrix) array
        self.invMasses = np.zeros(self.numVertices) # (#vers, 1) array
        # Edges
        self.edgeIds = edges #(#edges, 2) array
        # Constants
        self.devCompliance = devCompliance # float
        self.volCompliance = volCompliance # float

        # Matrices
        self.P = np.zeros((3,3)) #column major (3, 3) matrix
        self.F = np.zeros((3,3)) #column major (3, 3) matrix
        self.dF = np.zeros((3,3)) #column major (3, 3) matrix
        self.grads = np.zeros((4,3)) #column major (4, 3) matrix
        
        self.initPhysics()
    
    def initPhysics(self):
        # Gather positions of the four vertices for all tetrahedra
        v0 = self.positions[self.tetIds[:, 0]]
        v1 = self.positions[self.tetIds[:, 1]]
        v2 = self.positions[self.tetIds[:, 2]]
        v3 = self.positions[self.tetIds[:, 3]]
        # Compute edge vectors for all tetrahedra
        edge1 = v1 - v0
        edge2 = v2 - v0
        edge3 = v3 - v0

        # Compute the volume for all tetrahedra
        volumes = np.einsum('ij,ij->i', np.cross(edge1, edge2), edge3) / 6.0
        self.invRestVolume = 1.0 / volumes
        
        self.invRestPoses = np.linalg.inv(np.stack((edge1, edge2, edge3), axis=1))

        # Add mass from all tetrahedra
        vMass = np.maximum(volumes, 0) * self.density / 4.0  # Compute per-tetrahedron mass
        np.add.at(self.invMasses, self.tetIds.flatten(), np.repeat(vMass, 4))
        # Convert masses to inverse masses
        self.invMasses = np.where(self.invMasses > 0, 1.0 / self.invMasses, 0.0)

        """
        for tetNr in range(self.numTets):
            vIds = self.tetIds[tetNr]
            edge1 = self.positions[vIds[1]] - self.positions[vIds[0]]
            edge2 = self.positions[vIds[2]] - self.positions[vIds[0]]
            edge3 = self.positions[vIds[3]] - self.positions[vIds[0]]
            volume = np.dot(np.cross(edge1, edge2), edge3) / 6.0
            self.invRestVolume[tetNr] = 1.0 / volume
            # Initialize invRestPoses (column major)
            self.invRestPoses[tetNr] = np.linalg.inv(np.transpose(np.array([edge1, edge2, edge3])))
            # Initialize masses
            for i in range(4):
                self.invMasses[vIds[i]] += volume * self.density / 4.0 if volume > 0 else 0.0
        for mass in self.invMasses:
            # convert masses to invMasses
            mass = 1.0 / mass
        """
        
    def step(self):
        for _ in range(SUB_STEPS):
            self.substep()

    def substep(self):
        self.preSolve(DT/SUB_STEPS, GRAVITY)
        self.solve(DT/SUB_STEPS)
          
    def preSolve(self, dt, acc):
        for idx in range(len(self.positions)):
            if (self.positions[idx][2] == 2.0):
                continue
            self.prev_positions[idx] = self.positions[idx].copy()
            self.velocities[idx] += acc * dt
            self.positions[idx] += self.velocities[idx] * dt

    def solve(self, dt):
        self.solveHydroConstraint(self.volCompliance, dt)
        self.solveDevConstraint(self.devCompliance, dt)
        # Update velocity
        for idx in range(len(self.positions)):
            if (self.positions[idx][2] == 2.0):
                continue
            self.velocities[idx] = (self.prev_positions[idx] - self.positions[idx]) / dt

    def solveDevConstraint(self, compliance, dt):
        for tetNr in range(self.numTets):
            vIds = self.tetIds[tetNr]
            edge1 = self.positions[vIds[1]] - self.positions[vIds[0]]
            edge2 = self.positions[vIds[2]] - self.positions[vIds[0]]
            edge3 = self.positions[vIds[3]] - self.positions[vIds[0]]
            self.P = np.array([edge1, edge2, edge3])
            self.F = self.P @ self.invRestPoses[tetNr]
            r_s = np.sqrt(np.dot(self.F[0], self.F[0]) + np.dot(self.F[1], self.F[1]) + np.dot(self.F[2], self.F[2]))
            r_s_inv = 1.0 / r_s
            
            C = r_s
            # Compute C_grad
            self.grads = np.zeros((4,3))
            product = r_s_inv * self.F @ np.transpose(self.invRestPoses[tetNr])
            self.grads[1] = product[0]
            self.grads[2] = product[1]
            self.grads[3] = product[2]
            self.correctPosition(tetNr, C, compliance, dt)

    def solveHydroConstraint(self, compliance, dt):
        for tetNr in range(self.numTets):
            vIds = self.tetIds[tetNr]
            edge1 = self.positions[vIds[1]] - self.positions[vIds[0]]
            edge2 = self.positions[vIds[2]] - self.positions[vIds[0]]
            edge3 = self.positions[vIds[3]] - self.positions[vIds[0]]
            self.P = np.array([edge1, edge2, edge3])
            self.F = self.P @ self.invRestPoses[tetNr]
            self.dF[0] = np.cross(self.F[1], self.F[2])
            self.dF[1] = np.cross(self.F[2], self.F[0])
            self.dF[2] = np.cross(self.F[0], self.F[1])           
            vol = np.linalg.det(self.F)
            
            C = vol - 1.0 - self.volCompliance / self.devCompliance
             # Compute C_grad
            self.grads = np.zeros((4,3))
            product = self.dF @ np.transpose(self.invRestPoses[tetNr])
            self.grads[1] = product[0]
            self.grads[2] = product[1]
            self.grads[3] = product[2]
            self.correctPosition(tetNr, C, compliance, dt)
    
    def correctPosition(self, tetNr, C, compliance, dt):
        if C == 0.0:
            return
        
        self.grads[0] = -self.grads[1]
        self.grads[0] -= self.grads[2]
        self.grads[0] -= self.grads[3]
        
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
            if (self.positions[id][2] == 2.0):
                continue
            self.positions[id] += self.invMasses[id] * dlambda * self.grads[i]