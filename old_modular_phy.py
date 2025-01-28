import warp as wp
import warp.sim
import numpy as np

wp.init()

# Physical constants
DENSITY = 1e3 # kg/m**3
GRAVITY = wp.vec3(0.0, 0.0, -10.0)  # m/s**2
DT = 0.01  # second

# XPBD constants
SUB_STEPS = 100
EDGE_COMPLIANCE = 1e-4
VOLUME_COMPLIANCE = 0.0
COLLISION_COMPLIANCE = 0.0
COLLISION_THRESHOLD = 0.01

# Parallelization parameters
JACOBISCALE = 0.2

class SoftBody:
    def __init__(self, meshes, density=1e3):
        self.sim_substeps = 100
        self.frame_dt = 0.01
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.meshes = meshes

        builder = wp.sim.ModelBuilder()
        
        for meshidx, mesh in enumerate(meshes):
            builder.add_soft_mesh(
                pos=mesh.center,
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                scale=1.0,
                vertices=mesh.vertices.astype(np.float32),
                indices=mesh.tets.astype(np.int32).flatten(),
                density=density,
                k_mu=10000.0,
                k_lambda=50000.0,
                k_damp=0.0
                )
            meshobj = wp.sim.Mesh(mesh.vertices.astype(np.float32), mesh.tets.astype(np.int32).flatten())
            b = builder.add_body(origin=wp.vec3(mesh.center.astype(np.float32)))
            builder.add_shape_mesh(
                body=b,
                mesh=meshobj,
                pos=wp.vec3(mesh.center.astype(np.float32)),
                rot=wp.quat_identity(),
                has_shape_collision=True
            )

        self.model = builder.finalize("cuda")
        print(self.model.shape_count)
        self.model.ground = True
        
        self.model.soft_contact_restitution = 0.8

        self.integrator = wp.sim.SemiImplicitIntegrator()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

    def simulate(self):
        # Update the normals and copy results back to CPU
        for mesh in self.meshes:
            meshGeoCenter = wp.vec3(mesh.center.astype(np.float32))
            numVertices = len(mesh.vertices)
            positions = wp.array(mesh.vertices.astype(np.float32), dtype = wp.vec3, device = "cuda")
            faceIds = wp.array(mesh.faces.astype(np.int32), dtype=wp.int32, device = "cuda")
            faceNormals = wp.array(mesh.face_normals.astype(np.float32), dtype = wp.vec3, device = "cuda")
            vertexNormals = wp.array(np.zeros((numVertices, 3)).astype(np.float32), dtype = wp.vec3, device = "cuda")
            vertexNormalCounts = wp.array(np.zeros(numVertices).astype(np.int32), dtype=wp.int32, device = "cuda")
            wp.launch(kernel=calculate_faceNormals,
                    inputs=[positions, faceIds, faceNormals, vertexNormals, vertexNormalCounts, meshGeoCenter],
                    dim=len(faceIds), device = "cuda")
            wp.launch(kernel=calculate_vertex_normals,
                inputs=[vertexNormals, vertexNormalCounts], dim=numVertices, device = "cuda")
            mesh.face_normals = faceNormals.numpy()
            mesh.vertex_normals = vertexNormals.numpy()

        for _ in range(self.sim_substeps):
            wp.sim.collide(self.model, self.state_0)
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
            for meshidx, mesh in enumerate(self.meshes):
                mesh.vertices = self.state_1.particle_q[len(mesh.vertices)*meshidx:len(mesh.vertices)*(meshidx+1)].numpy()

            # swap states
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        
class Simulator:
    def __init__(self):
        device = "cpu"

    def add_bodies(self, meshes):
        self.sim = SoftBody(meshes)

    def step(self):
        self.sim.step()
    
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