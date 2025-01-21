import time
from Renderer import *
from Physics_copy import *
from FileReader import *
import pyvista as pv
import tetgen

def main():    
    # Initialize renderer and simulator
    renderer = Renderer()
    simulator = Simulator()

    #"""
    # Position bottom cylinder on ground, top cylinder above it
    left = np.array([-0.3, 0.0, 1.0])  # Lift slightly to account for radius
    right = np.array([0.3, 0.0, 1.0])     # Start higher to fall onto bottom cylinder
    # Generate Mesh (real muscle statistics: r = 0.04, h = 0.1 (in meters))
    cylinder = pv.Cylinder(radius=0.25, height=1.0, center=(0, 0, 0), direction=(1, 0, 0.5), resolution=16).triangulate()
    tet = tetgen.TetGen(cylinder)
    vertices, tets = tet.tetrahedralize()
    surface_faces = generate_surface_faces(tets)
    
    """

    # Position bottom cylinder on ground, top cylinder above it
    top = np.array([0.3, 0.3, 0.8])  # Lift slightly to account for radius
    bottom = np.array([0.0, 0.0, 0.25])     # Start higher to fall onto bottom cylinder
    # Generate Mesh (real muscle statistics: r = 0.04, h = 0.1 (in meters))
    cylinder = pv.Cylinder(radius=0.25, height=1.0, center=(0, 0, 0), direction=(0, 1, 0), resolution=8).triangulate()
    tet = tetgen.TetGen(cylinder)
    vertices, tets = tet.tetrahedralize()
    surface_faces = generate_surface_faces(tets)
    cylinder2 = pv.Cylinder(radius=0.25, height=1.0, center=(0, 0, 0), direction=(1, 0, 0), resolution=8).triangulate()
    tet2 = tetgen.TetGen(cylinder2)
    vertices2, tets2 = tet2.tetrahedralize()
    surface_faces2 = generate_surface_faces(tets2)
    """
    
    meshes = [
        #Mesh(top, vertices2, surface_faces2, tets2),
        #Mesh(left, vertices, surface_faces, tets),
        Mesh(right, vertices, surface_faces, tets)
    ]
    
    try:
        for i in range(len(meshes)):
            renderer.add_mesh(meshes[i])
            simulator.add_mesh(meshes[i])
            #fix_surfaces(simulator, i, 1.5)
            #fix_surfaces(simulator, i, 0.5)

        while not renderer.should_close():
            glfw.poll_events()
            for i in range(len(meshes)):
                #rotate_fixed_vertex(simulator, i, 1.5, np.radians(1.0))
                #rotate_fixed_vertex(simulator, i, 0.5, np.radians(-1.0))
                simulator.step()
                renderer.update_meshes()
                renderer.render()

            time.sleep(DT)
    
    finally:
        renderer.cleanup()

def fix_surfaces(simulator, mesh_idx, z, tolerance=1e-5):
    for i in range(simulator.meshes[mesh_idx].numVertices):
        if abs(simulator.meshes[mesh_idx].positions[i, 2] - z) <= tolerance:
            simulator.meshes[mesh_idx].fixed_vertices.add(i)
            simulator.meshes[mesh_idx].invMasses[i] = 0.0

def rotate_fixed_vertex(simulator, mesh_idx, z, angle):
    for i in simulator.meshes[mesh_idx].fixed_vertices:
        if abs(simulator.meshes[mesh_idx].positions[i, 2] - z) < 1e-5:
            local_coord = simulator.meshes[mesh_idx].positions[i] - np.array([0, 0, z])
            # Get current radius and angle of each vertex
            radius = np.linalg.norm(local_coord)
            if radius < 1e-5:
                continue
                    
            current_angle = np.arctan2(local_coord[1], local_coord[0])
            
            # New position maintains same radius but at new angle
            new_pos = np.array([
                radius * np.cos(current_angle + angle),
                radius * np.sin(current_angle + angle),
                local_coord[2]  # Keep Z coordinate unchanged
            ])
            
            movement = new_pos - local_coord
            simulator.meshes[mesh_idx].positions[i] += movement
            simulator.meshes[mesh_idx].prev_positions[i] += movement

if __name__ == "__main__":
    main()