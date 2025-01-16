import time
import glfw
from Renderer import *
from Physics import *
from FileReader import *
import pyvista as pv
import tetgen

def main():
    # Generate Mesh (real muscle statistics: r = 0.04, h = 0.1 (in meters))
    cylinder = pv.Cylinder(radius=0.5, height=0.5, center=(0, 0, 0), direction=(0, 0, 1), resolution=16).triangulate()
    tet = tetgen.TetGen(cylinder)
    vertices, tets = tet.tetrahedralize()
    surface_faces = generate_surface_faces(tets)
    
    # Initialize renderer and simulator
    renderer = Renderer()
    centers = [np.array([-1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
    meshes = [Mesh(vertices + centers[0], surface_faces, tets), Mesh(vertices + centers[1], surface_faces, tets)]
    simulator = Simulator()
    for i in range(len(meshes)):
        renderer.add_mesh(meshes[i])
        simulator.add_mesh(meshes[i])
        
    # Manipulate the mesh
    #for i in range(len(meshes)):
        #fix_surfaces(simulator, i, 1.0, tolerance=1e-5)
        #fix_surfaces(simulator, i, -1.0, tolerance=1e-5)

    while not renderer.should_close():
        glfw.poll_events()
        for i in range(len(meshes)):
            #rotate_fixed_vertex(simulator, i, centers[i] + np.array([0.0, 0.0, 1.0]), 1.0, np.radians(0.5))
            #rotate_fixed_vertex(simulator, i, centers[i] + np.array([0.0, 0.0, -1.0]), -1.0, np.radians(-0.5))
            simulator.step()
            renderer.update_mesh_positions(i)
            renderer.render()
        time.sleep(DT)
    
    for i in range(len(renderer)):
        renderer.cleanup()
    
def fix_surfaces(simulator, mesh_idx, z, tolerance=1e-5):
    for i in range(simulator.meshes[mesh_idx].numVertices):
        if abs(simulator.meshes[mesh_idx].positions[i, 2] - z) <= tolerance:
            simulator.meshes[mesh_idx].fixed_vertices.add(i)
            simulator.meshes[mesh_idx].invMasses[i] = 0.0

def rotate_fixed_vertex(simulator, mesh_idx, center, z, angle):
    for i in simulator.meshes[mesh_idx].fixed_vertices:
        if abs(simulator.meshes[mesh_idx].positions[i, 2] - z) < 1e-5:
            local_coord = simulator.meshes[mesh_idx].positions[i] - center
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