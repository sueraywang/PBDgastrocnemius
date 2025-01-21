# Physics.py
import numpy as np
from test import *

# Physical constants
DENSITY = 1000  # kg/m**3
GRAVITY = np.array([0.0, 0.0, -9.8])  # m/s**2
DT = 0.01  # second

# XPBD constants
SUB_STEPS = 5
DEVIATORIC_COMPLIANCE = 1.0/100000.0
VOLUME_COMPLIANCE = 0.0
COLLISION_COMPLIANCE = 1e-3  # Stiffness for collision response

class SoftBody:
    """Class to store the state and parameters of a single mesh"""
    def __init__(self, mesh, density=DENSITY, devCompliance=DEVIATORIC_COMPLIANCE, volCompliance=VOLUME_COMPLIANCE):
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
        self.invRestVolume = np.zeros(self.numTets) # (#tets, 1) array
        self.invRestPoses = np.zeros((self.numTets, 3, 3)) # (#tets, (3, 3) matrix) array
        self.invMasses = np.zeros(self.numVertices)
        
        # Per-mesh temporary matrices and properties
        self.P = np.zeros((3, 3))
        self.F = np.zeros((3, 3))
        self.dF = np.zeros((3, 3))
        self.grads = np.zeros((4, 3))
        self.devCompliance = devCompliance
        self.volCompliance = volCompliance
        self.collision_constraints = [] # Store active collision constraints of certain mesh

class CollisionConstraint:
    def __init__(self, vertex_idx, triangle_mesh_idx, trig_idx):
        self.vertex_idx = vertex_idx
        self.triangle_mesh_idx = triangle_mesh_idx
        self.trig_idx = trig_idx

class Simulator:
    def __init__(self):
        self.bodies = [] # It stores all soft bodies

    def add_mesh(self, mesh, density=DENSITY, devCompliance=DEVIATORIC_COMPLIANCE, volCompliance=VOLUME_COMPLIANCE):
        body = SoftBody(mesh, density, devCompliance, volCompliance)
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

        # Compute the rest state for all tetrahedra
        volumes = np.einsum('ij,ij->i', np.cross(edge1, edge2), edge3) / 6.0
        body.invRestVolume = 1.0 / volumes
        body.invRestPoses = np.linalg.inv(np.stack((edge1, edge2, edge3), axis=1))

        # Add mass from all tetrahedra
        vMass = np.maximum(volumes, 0) * body.density / 4.0
        np.add.at(body.invMasses, body.tetIds.flatten(), np.repeat(vMass, 4))
        # Convert masses to inverse masses, change fixed vertices mass to 0
        body.invMasses = np.where(
            (body.invMasses > 0) & (~np.isin(np.arange(body.numVertices), list(body.fixed_vertices))),
            1.0 / body.invMasses,
            0.0
        )

    def step(self):
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
                for tetNr in range(body.numTets):
                    self.solve_hydro_constraint(body, tetNr, dt)
                    self.solve_dev_constraint(body, tetNr, dt)
        
            for body in self.bodies: # It makes sure that all bodies' constrains are solved before we update velocity
                body.velocities = (body.positions - body.prev_positions) / dt

    def solve_dev_constraint(self, body, tetNr, dt):
        vIds = body.tetIds[tetNr]
        edge1 = body.positions[vIds[1]] - body.positions[vIds[0]]
        edge2 = body.positions[vIds[2]] - body.positions[vIds[0]]
        edge3 = body.positions[vIds[3]] - body.positions[vIds[0]]
        body.P = np.array([edge1, edge2, edge3])
        body.F = body.P @ body.invRestPoses[tetNr]
        r_s = np.sqrt(np.dot(body.F[0], body.F[0]) + 
                     np.dot(body.F[1], body.F[1]) + 
                     np.dot(body.F[2], body.F[2]))
        r_s_inv = 1.0 / r_s
        
        C = r_s
        body.grads = np.zeros((4, 3))
        product = r_s_inv * body.F @ np.transpose(body.invRestPoses[tetNr])
        body.grads[1] = product[0]
        body.grads[2] = product[1]
        body.grads[3] = product[2]
        self.correct_position(body, tetNr, C, body.devCompliance, dt)

    def solve_hydro_constraint(self, body, tetNr, dt):
        vIds = body.tetIds[tetNr]
        edge1 = body.positions[vIds[1]] - body.positions[vIds[0]]
        edge2 = body.positions[vIds[2]] - body.positions[vIds[0]]
        edge3 = body.positions[vIds[3]] - body.positions[vIds[0]]
        body.P = np.array([edge1, edge2, edge3])
        body.F = body.P @ body.invRestPoses[tetNr]
        body.dF[0] = np.cross(body.F[1], body.F[2])
        body.dF[1] = np.cross(body.F[2], body.F[0])
        body.dF[2] = np.cross(body.F[0], body.F[1])
        vol = np.linalg.det(body.F)
        
        C = vol - 1.0 - body.volCompliance / body.devCompliance
        body.grads = np.zeros((4, 3))
        product = body.dF @ np.transpose(body.invRestPoses[tetNr])
        body.grads[1] = product[0]
        body.grads[2] = product[1]
        body.grads[3] = product[2]
        self.correct_position(body, tetNr, C, body.volCompliance, dt)

    def correct_position(self, body, tetNr, C, compliance, dt):
        if C == 0.0:
            return
        
        body.grads[0] = -body.grads[1]
        body.grads[0] -= body.grads[2]
        body.grads[0] -= body.grads[3]
        
        w = 0.0
        for i in range(4):
            id = body.tetIds[tetNr][i]
            w += np.dot(body.grads[i], body.grads[i]) * body.invMasses[id]
        
        if w == 0.0:
            return
        
        alpha = compliance * body.invRestVolume[tetNr] / dt / dt
        dlambda = -C / (w + alpha)
        
        for i in range(4):
            id = body.tetIds[tetNr][i]
            body.positions[id] += body.invMasses[id] * dlambda * body.grads[i]