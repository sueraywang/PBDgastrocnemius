import time
import tetgen
from colorGraph import *
from Renderer import *
from Physics import *

def main():    
    # Initialize renderer and simulator
    renderer = Renderer(cameraRadius=6.0, lookAtPosition=np.array([0.0, 0.0, 1.0]), h_angle=-np.pi/2, v_angle=np.pi/2)
    simulator = Simulator()

    # Position bottom cylinder on ground, top cylinder above it
    top = np.array([0.0, 0.0, 2.0])  # Lift slightly to account for radius
    
    #"""
    # Generate Mesh (real muscle statistics: r = 0.04, h = 0.1 (in meters), about 300 resolution)
    cylinder = pv.Cylinder(radius=0.25, height=1.0, center=(0, 0, 0), direction=(0, 0.5, 0.5), resolution=16).triangulate()
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
    

    bodies = [
        Mesh(top, vertices, surface_faces, tets)
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