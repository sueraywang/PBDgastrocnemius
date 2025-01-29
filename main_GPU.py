import time
import tetgen
from colorGraph import *
from Renderer import *
from Physics_GPU import *

def main():    
    # Initialize renderer and simulator
    renderer = Renderer(cameraRadius=3.0, lookAtPosition=np.array([0.0, 0.0, 0.5]), h_angle=-np.pi/3, v_angle=np.pi/2)
    simulator = Simulator()

    # Position bottom cylinder on ground, top cylinder above it
    top = np.array([0.0, 0.0, 1.0])  # Lift slightly to account for radius
    bottom = np.array([0.0, 0.0, 0.25])     # Start higher to fall onto bottom cylinder
    
    #"""
    # Generate Mesh (real muscle statistics: r = 0.04, h = 0.1 (in meters), about 300 resolution)
    cylinder = pv.Cylinder(radius=0.25, height=1.0, center=(0, 0, 0), direction=(0, 0, 1.0), resolution=300).triangulate()
    tet = tetgen.TetGen(cylinder)
    vertices, tets = tet.tetrahedralize()
    surface_faces = generate_surface_faces(tets)
    """
    # Get the vertices (points)
    tetrahedron = pv.Tetrahedron()
    tet = tetgen.TetGen(tetrahedron)
    vertices, tets = tet.tetrahedralize()
    print(vertices, tets)
    surface_faces = tetrahedron.faces.reshape(-1, 4)[:, 1:]  # Ignore the size indicator
    #"""
    
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
        Mesh(top, vertices @ R1.T, surface_faces, tets),
        Mesh(bottom, vertices @ R2.T, surface_faces, tets)
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

if __name__ == "__main__":
    main()