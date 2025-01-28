import time
import tetgen
from colorGraph import *
from Renderer import *
from warp_simulator import *

def main():    
    # Initialize renderer, simulator, and performance monitor
    renderer = Renderer(cameraRadius=3.0, lookAtPosition=np.array([0.0, 0.0, 0.5]), h_angle=-np.pi/3, v_angle=np.pi/2)
    simulator = Simulator()

    # Position bottom cylinder on ground, top cylinder above it
    top = np.array([0.0, 0.0, 0.8])  # Lift slightly to account for radius
    bottom = np.array([0.0, 0.0, 0.25])     # Start higher to fall onto bottom cylinder
    # Generate Mesh (real muscle statistics: r = 0.04, h = 0.1 (in meters), about 300 resolution)
    cylinder = pv.Cylinder(radius=0.25, height=1.0, center=(0, 0, 0), direction=(0, 0, 1), resolution=100).triangulate()
    tet = tetgen.TetGen(cylinder)
    vertices, tets = tet.tetrahedralize()
    #print(vertices.shape)
    surface_faces = generate_surface_faces(tets)
    edges = generate_edges(tets)
    
    #mesh_size = compute_min_vertex_distance(vertices)
    #print(f"Minimum vertex-to-vertex distance (mesh size): {mesh_size}")

    R1 = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    R2 = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
    ])
    R3 = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])

    bodies = [
        Mesh(top, vertices @ R1.T, edges, surface_faces, tets),
        Mesh(bottom, vertices @ R2.T, edges, surface_faces, tets)
    ]
    
    try:
        # Initialize bodies
        for i in range(len(bodies)):
            renderer.add_mesh(bodies[i])
        simulator.add_bodies(bodies)
        
        while not renderer.should_close():
            
            glfw.poll_events()
            
            for i in range(len(bodies)):
                renderer.update_meshes()
                renderer.render()
                
            simulator.step()
            time.sleep(DT)
    
    finally:
        renderer.cleanup()
        
def compute_min_vertex_distance(vertices):
    # Compute pairwise Euclidean distances
    distances = np.linalg.norm(vertices[:, np.newaxis, :] - vertices[np.newaxis, :, :], axis=-1)

    # Ignore zero distances (self-distances)
    np.fill_diagonal(distances, np.inf)

    # Find the smallest distance
    min_distance = np.min(distances)
    return min_distance

if __name__ == "__main__":
    main()