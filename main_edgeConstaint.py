import time
import tetgen
from colorGraph import *
from Renderer import *
from warp_simulator import *

def main():    
    # Initialize renderer, simulator, and performance monitor
    renderer = Renderer()
    simulator = Simulator()

    # Position bottom cylinder on ground, top cylinder above it
    top = np.array([0.3, 0.3, 0.8])  # Lift slightly to account for radius
    bottom = np.array([0.0, 0.0, 0.25])     # Start higher to fall onto bottom cylinder
    # Generate Mesh (real muscle statistics: r = 0.04, h = 0.1 (in meters), about 300 resolution)
    cylinder = pv.Cylinder(radius=0.25, height=1.0, center=(0, 0, 0), direction=(0, 1, 0), resolution=300).triangulate()
    tet = tetgen.TetGen(cylinder)
    vertices, tets = tet.tetrahedralize()
    print(vertices.shape)
    surface_faces = generate_surface_faces(tets)
    edges = generate_edges(tets)

    axis = 'z'
    if axis == 'x':
        R = np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ])
    elif axis == 'y':
        R = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]
        ])
    elif axis == 'z':
        R = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Invalid axis! Choose 'x', 'y', or 'z'.")

    # Rotate the vertices
    rotated_vertices = vertices @ R.T

    bodies = [
        Mesh(top, vertices, edges, surface_faces, tets),
        Mesh(bottom, rotated_vertices, edges, surface_faces, tets)
    ]
    
    try:
        # Initialize bodies
        for i in range(len(bodies)):
            renderer.add_mesh(bodies[i])
            simulator.add_body(bodies[i])
        
        while not renderer.should_close():
            
            glfw.poll_events()
            for i in range(len(bodies)):
                simulator.step()
                renderer.update_meshes()
                renderer.render()
            
            time.sleep(DT)
    
    finally:
        renderer.cleanup()

if __name__ == "__main__":
    main()