import time
import tetgen
from colorGraph import *
from Renderer import *
from Physics_GPU import *

def main():    
    # Initialize renderer and simulator
    renderer = Renderer(cameraRadius=0.5, lookAtPosition=np.array([0.0, 0.0, 1.0]), h_angle=-np.pi/2, v_angle=np.pi/2)
    simulator = Simulator()

    # Position bottom cylinder on ground, top cylinder above it
    top = np.array([0.03, 0.0, 1.0])  # Lift slightly to account for radius
    bottom = np.array([-0.03, 0.0, 1.0])     # Start higher to fall onto bottom cylinder
    
    # Generate Mesh (real muscle statistics: diam = 0.04, h = 0.3 (in meters), about 300 resolution)
    cylinder = pv.Cylinder(radius=0.02, height=0.2, center=(0, 0, 0), direction=(0, 0, 1.0), resolution=32).triangulate()
    tet = tetgen.TetGen(cylinder)
    vertices, tets = tet.tetrahedralize()
    surface_faces = generate_surface_faces(tets)
    """
    print(f"Num of vertices: {vertices.shape}")
    print(f"Num of tets: {tets.shape}")
    print(f"Young's modulus: {YOUNG_MODULUS}")
    print(f"Volume compliance: {VOLUME_COMPLIANCE}")
    """
    Rx = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    Ry = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0]
    ])
    Rz = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])

    bodies = [
        Mesh(top, vertices, surface_faces, tets),
        Mesh(bottom, vertices, surface_faces, tets)
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