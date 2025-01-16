# Physics.py
import numpy as np
import pybullet as p
import pybullet_data

# Physical constants
DENSITY = 1000.0  # kg/m**3
GRAVITY = np.array([0.0, 0.0, -9.8])  # m/s**2
DT = 0.01  # second

# XPBD constants
SUB_STEPS = 5
DEVIATORIC_COMPLIANCE = 1.0/100000.0
VOLUME_COMPLIANCE = 0.0

class MeshState:
    """Class to store the state and parameters of a single mesh"""
    def __init__(self, mesh, density=DENSITY):
        # Store reference to the mesh
        self.mesh = mesh
        
        # Vertices
        self.numVertices = len(mesh.vertices)
        self.positions = mesh.vertices #(#vers, 3) array
        self.prev_positions = self.positions.copy()
        self.velocities = np.zeros_like(mesh.vertices)
        self.fixed_vertices = set()
        
        # Tetrahedrons
        self.numTets = len(mesh.tets)
        self.tetIds = mesh.tets #(#tets, 4) array
        self.density = density
        self.invRestVolume = np.zeros(self.numTets) # (#tets, 1) array
        self.invRestPoses = np.zeros((self.numTets, 3, 3)) # (#tets, (3, 3) matrix) array
        self.invMasses = np.zeros(self.numVertices)
        
        # Per-mesh temporary matrices for parallel computation
        self.P = np.zeros((3, 3))
        self.F = np.zeros((3, 3))
        self.dF = np.zeros((3, 3))
        self.grads = np.zeros((4, 3))

class Simulator:
    def __del__(self):
        p.disconnect(physicsClientId=self.physicsClientId)
    
    def init_mesh_physics(self, mesh_state):
        # Gather positions of the four vertices for all tetrahedra
        v0 = mesh_state.positions[mesh_state.tetIds[:, 0]]
        v1 = mesh_state.positions[mesh_state.tetIds[:, 1]]
        v2 = mesh_state.positions[mesh_state.tetIds[:, 2]]
        v3 = mesh_state.positions[mesh_state.tetIds[:, 3]]
        
        # Compute edge vectors for all tetrahedra
        edge1 = v1 - v0
        edge2 = v2 - v0
        edge3 = v3 - v0

        # Compute the volume for all tetrahedra
        volumes = np.einsum('ij,ij->i', np.cross(edge1, edge2), edge3) / 6.0
        mesh_state.invRestVolume = 1.0 / volumes
        mesh_state.invRestPoses = np.linalg.inv(np.stack((edge1, edge2, edge3), axis=1))

        # Add mass from all tetrahedra
        vMass = np.maximum(volumes, 0) * mesh_state.density / 4.0
        np.add.at(mesh_state.invMasses, mesh_state.tetIds.flatten(), np.repeat(vMass, 4))
        
        # Convert masses to inverse masses, change fixed vertices mass to 0
        mesh_state.invMasses = np.where(
            (mesh_state.invMasses > 0) & (~np.isin(np.arange(mesh_state.numVertices), list(mesh_state.fixed_vertices))),
            1.0 / mesh_state.invMasses,
            0.0
        )

    def step(self):
        dt = DT / SUB_STEPS
        for mesh_state in self.meshes:
            mesh_state.prev_positions = mesh_state.positions.copy()
        
        for _ in range(SUB_STEPS):
            # Update velocities and positions for all meshes
            for mesh_state in self.meshes:
                mask = ~np.isin(np.arange(mesh_state.numVertices), list(mesh_state.fixed_vertices))
                mesh_state.velocities[mask] += GRAVITY * dt
                mesh_state.positions[mask] += mesh_state.velocities[mask] * dt
            
            # Solve all constraints for all meshes
            self.solve_all_constraints(dt)
            
            # Update velocities for all meshes
            for mesh_state in self.meshes:
                mesh_state.velocities = (mesh_state.positions - mesh_state.prev_positions) / dt
                mesh_state.prev_positions = mesh_state.positions.copy()

    def solve_all_constraints(self, dt):
        # Solve hydrostatic constraints for all meshes
        for mesh_state in self.meshes:
            for tetNr in range(mesh_state.numTets):
                self.solve_hydro_constraint(mesh_state, tetNr, dt)
        
        # Solve deviatoric constraints for all meshes
        for mesh_state in self.meshes:
            for tetNr in range(mesh_state.numTets):
                self.solve_dev_constraint(mesh_state, tetNr, dt)
        
        # Handle collisions for all meshes
        for mesh_state in self.meshes:
            self.solve_collisions(mesh_state)

    def solve_dev_constraint(self, mesh_state, tetNr, dt):
        vIds = mesh_state.tetIds[tetNr]
        edge1 = mesh_state.positions[vIds[1]] - mesh_state.positions[vIds[0]]
        edge2 = mesh_state.positions[vIds[2]] - mesh_state.positions[vIds[0]]
        edge3 = mesh_state.positions[vIds[3]] - mesh_state.positions[vIds[0]]
        mesh_state.P = np.array([edge1, edge2, edge3])
        mesh_state.F = mesh_state.P @ mesh_state.invRestPoses[tetNr]
        r_s = np.sqrt(np.dot(mesh_state.F[0], mesh_state.F[0]) + 
                     np.dot(mesh_state.F[1], mesh_state.F[1]) + 
                     np.dot(mesh_state.F[2], mesh_state.F[2]))
        r_s_inv = 1.0 / r_s
        
        C = r_s
        mesh_state.grads = np.zeros((4, 3))
        product = r_s_inv * mesh_state.F @ np.transpose(mesh_state.invRestPoses[tetNr])
        mesh_state.grads[1] = product[0]
        mesh_state.grads[2] = product[1]
        mesh_state.grads[3] = product[2]
        self.correct_position(mesh_state, tetNr, C, self.devCompliance, dt)

    def solve_hydro_constraint(self, mesh_state, tetNr, dt):
        vIds = mesh_state.tetIds[tetNr]
        edge1 = mesh_state.positions[vIds[1]] - mesh_state.positions[vIds[0]]
        edge2 = mesh_state.positions[vIds[2]] - mesh_state.positions[vIds[0]]
        edge3 = mesh_state.positions[vIds[3]] - mesh_state.positions[vIds[0]]
        mesh_state.P = np.array([edge1, edge2, edge3])
        mesh_state.F = mesh_state.P @ mesh_state.invRestPoses[tetNr]
        mesh_state.dF[0] = np.cross(mesh_state.F[1], mesh_state.F[2])
        mesh_state.dF[1] = np.cross(mesh_state.F[2], mesh_state.F[0])
        mesh_state.dF[2] = np.cross(mesh_state.F[0], mesh_state.F[1])
        vol = np.linalg.det(mesh_state.F)
        
        C = vol - 1.0 - self.volCompliance / self.devCompliance
        mesh_state.grads = np.zeros((4, 3))
        product = mesh_state.dF @ np.transpose(mesh_state.invRestPoses[tetNr])
        mesh_state.grads[1] = product[0]
        mesh_state.grads[2] = product[1]
        mesh_state.grads[3] = product[2]
        self.correct_position(mesh_state, tetNr, C, self.volCompliance, dt)

    def correct_position(self, mesh_state, tetNr, C, compliance, dt):
        if C == 0.0:
            return
        
        mesh_state.grads[0] = -mesh_state.grads[1]
        mesh_state.grads[0] -= mesh_state.grads[2]
        mesh_state.grads[0] -= mesh_state.grads[3]
        
        w = 0.0
        for i in range(4):
            id = mesh_state.tetIds[tetNr][i]
            w += np.dot(mesh_state.grads[i], mesh_state.grads[i]) * mesh_state.invMasses[id]
        
        if w == 0.0:
            return
        
        alpha = compliance * mesh_state.invRestVolume[tetNr] / dt / dt
        dlambda = -C / (w + alpha)
        
        for i in range(4):
            id = mesh_state.tetIds[tetNr][i]
            mesh_state.positions[id] += mesh_state.invMasses[id] * dlambda * mesh_state.grads[i]

    def add_mesh(self, mesh, density=DENSITY):
        mesh_state = MeshState(mesh, density)
        self.meshes.append(mesh_state)
        self.init_mesh_physics(mesh_state)
        
        try:
            # Convert to convex hull for simpler collision detection
            vertices = mesh_state.positions.astype(np.float32)
            
            # Create a convex hull collision shape
            collision_shape = p.createCollisionShape(
                shapeType=p.GEOM_MESH,
                vertices=vertices.tolist(),
                physicsClientId=self.physicsClientId,
                flags=p.GEOM_FORCE_CONCAVE_TRIMESH
            )
            
            # Create multibody
            mesh_state.bullet_body = p.createMultiBody(
                baseMass=1.0,  # Non-zero mass for dynamic body
                baseCollisionShapeIndex=collision_shape,
                basePosition=[0, 0, -1],  # Start slightly above ground
                baseOrientation=[0, 0, 0, 1],
                physicsClientId=self.physicsClientId
            )
            
            # Set physical properties
            p.changeDynamics(
                mesh_state.bullet_body,
                -1,  # -1 for base link
                restitution=0.3,
                lateralFriction=0.5,
                physicsClientId=self.physicsClientId
            )
            
        except Exception as e:
            print(f"Error in mesh creation: {str(e)}")
            raise
        
        return mesh_state

    def solve_collisions(self, mesh_state):
        """Handle collisions using PyBullet with debug output"""
        try:
            # Print current mesh state
            min_z = np.min(mesh_state.positions[:, 2])
            max_z = np.max(mesh_state.positions[:, 2])
            print(f"Mesh Z range: min={min_z:.3f}, max={max_z:.3f}")

            # Remove old collision object
            if hasattr(mesh_state, 'bullet_body'):
                p.removeBody(mesh_state.bullet_body, physicsClientId=self.physicsClientId)
            
            # Create new collision shape with updated vertices
            vertices = mesh_state.positions.astype(np.float32)
            collision_shape = p.createCollisionShape(
                shapeType=p.GEOM_MESH,
                vertices=vertices.tolist(),
                physicsClientId=self.physicsClientId,
                flags=p.GEOM_FORCE_CONCAVE_TRIMESH
            )
            
            # Create new multibody with updated collision shape
            mesh_state.bullet_body = p.createMultiBody(
                baseMass=1.0,
                baseCollisionShapeIndex=collision_shape,
                basePosition=[0, 0, 0],  # Changed to match actual mesh position
                baseOrientation=[0, 0, 0, 1],
                physicsClientId=self.physicsClientId
            )
            
            # Step simulation to detect collisions
            p.stepSimulation(physicsClientId=self.physicsClientId)
            
            # Get contact points
            contacts = p.getContactPoints(
                bodyA=self.planeId,
                bodyB=mesh_state.bullet_body,
                physicsClientId=self.physicsClientId
            )
            
            print(f"Number of contacts detected: {len(contacts)}")
            
            for i, contact in enumerate(contacts):
                point_on_b = contact[6]  # Contact point on our mesh
                normal = contact[7]      # Normal from A to B
                depth = contact[8]       # Penetration depth
                print(f"Contact {i}: point={point_on_b}, normal={normal}, depth={depth}")
                
                if depth > 0:
                    # Find closest vertex
                    point = np.array(point_on_b)
                    distances = np.linalg.norm(mesh_state.positions - point, axis=1)
                    vertex_id = np.argmin(distances)
                    
                    print(f"Found vertex {vertex_id} at position {mesh_state.positions[vertex_id]}")
                    
                    if vertex_id not in mesh_state.fixed_vertices:
                        # Project vertex out of collision
                        correction = (depth + self.collision_margin) * np.array(normal)
                        mesh_state.positions[vertex_id] += correction
                        
                        # Update velocity for bounce
                        velocity = mesh_state.velocities[vertex_id]
                        normal_velocity = np.dot(velocity, normal)
                        
                        if normal_velocity < 0:
                            reflection = -normal_velocity * (1 + self.restitution)
                            mesh_state.velocities[vertex_id] += reflection * np.array(normal)
                            print(f"Applied bounce: old_vel={velocity}, new_vel={mesh_state.velocities[vertex_id]}")
            
        except Exception as e:
            print(f"Error in collision handling: {str(e)}")
            return False
        
        return True

    def __init__(self, devCompliance=DEVIATORIC_COMPLIANCE, volCompliance=VOLUME_COMPLIANCE):
        self.meshes = []
        self.devCompliance = devCompliance
        self.volCompliance = volCompliance
        
        # Initialize PyBullet in DIRECT mode
        self.physicsClientId = p.connect(p.DIRECT)
        print("PyBullet initialized in DIRECT mode")
        
        p.setGravity(0, 0, -9.81, physicsClientId=self.physicsClientId)
        print("Gravity set to [0, 0, -9.81]")
        
        # Create ground plane at z=0
        self.planeId = p.createCollisionShape(
            p.GEOM_PLANE,
            planeNormal=[0, 0, 1],
            physicsClientId=self.physicsClientId
        )
        self.ground_body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=self.planeId,
            basePosition=[0, 0, -1],
            physicsClientId=self.physicsClientId
        )
        print(f"Ground plane created: shape_id={self.planeId}, body_id={self.ground_body}")
        
        # Collision parameters
        self.collision_margin = 0.01
        self.restitution = 0.3
