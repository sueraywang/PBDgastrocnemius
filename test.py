import warp as wp
import tetgen
from colorGraph import *
from Mesh import *

# Generate Mesh (real muscle statistics: diam = 0.04, h = 0.3 (in meters), about 300 resolution)
cylinder = pv.Cylinder(radius=0.02, height=0.2, center=(0, 0, 0), direction=(0, 0, 1.0), resolution=32).triangulate()
tet = tetgen.TetGen(cylinder)
vertices, tets = tet.tetrahedralize()
surface_faces = generate_surface_faces(tets)

# Calculate face bounding boxes
num_faces = len(surface_faces)
lowers = np.zeros((num_faces, 3))
uppers = np.zeros((num_faces, 3))

for i, face in enumerate(surface_faces):
    v1, v2, v3 = vertices[face]
    lowers[i] = np.minimum(np.minimum(v1, v2), v3)
    uppers[i] = np.maximum(np.maximum(v1, v2), v3)

BVH = wp.Bvh(
    wp.array(lowers.astype(np.float32), dtype=wp.vec3, device="cpu"),
    wp.array(uppers.astype(np.float32), dtype=wp.vec3, device="cpu")
)

@wp.kernel
def ray_query_kernel(bvh: wp.uint64):
    print("start query...")
    start = wp.vec3(0.0, 0.0, 0.0)
    direction = wp.vec3(1.0, 0.0, 0.0)
    bounds_nr = int(0)
    for i in range(2):
        print(i)
        query_2 = wp.bvh_query_ray(bvh, start, direction)
        while wp.bvh_query_next(query_2, bounds_nr):
            print(bounds_nr)

wp.launch(ray_query_kernel, inputs=[BVH.id], dim=1)
        
