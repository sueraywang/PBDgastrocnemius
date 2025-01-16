import time
import glfw
from Renderer import Renderer
from Physics import *
from FileReader import *
import pyvista as pv
import tetgen

def main():
    # Generate Mesh (real muscle statistics: r = 0.04, h = 0.1 (in meters))
    cylinder = pv.Cylinder(radius=0.5, height=2.0, center=(0, 0, 0), direction=(0, 0, 1), resolution=32).triangulate()
    tet = tetgen.TetGen(cylinder)
    vertices, tets = tet.tetrahedralize()
    #vertices, tets = read_obj("PBDMuscles/cuboid.obj")
    surface_faces = generate_surface_faces(tets)

    mesh = Mesh(vertices, tets)
    fix_surfaces(mesh, 1.0)
    fix_surfaces(mesh, -1.0)
    renderer = Renderer(vertices, surface_faces)

    while not renderer.should_close():
        glfw.poll_events()
        rotate_fixed_vertex(mesh, np.radians(0.5))
        mesh.step()
        renderer.update_vertex_positions(mesh.positions, surface_faces)
        renderer.render()
        time.sleep(DT)

    renderer.cleanup()
    
def fix_surfaces(mesh, pos, tolerance=1e-5):
    for i in range(mesh.numVertices):
        if abs(mesh.positions[i, 2] - pos) <= tolerance:
            mesh.fixed_vertices.add(i)
            mesh.invMasses[i] = 0.0

def move_fixed_vertex(mesh, movement):
    for i in mesh.fixed_vertices:
        mesh.positions[i] += movement
        mesh.prev_positions[i] += movement  # Update previous position to prevent velocity

def rotate_fixed_vertex(mesh, angle):
    for i in mesh.fixed_vertices:
        # Get current radius and angle of each vertex
        radius = np.linalg.norm(mesh.positions[i, :2])
        if radius < 1e-5:
            continue
                
        current_angle = np.arctan2(mesh.positions[i, 1], mesh.positions[i, 0])
        
        if mesh.positions[i, 2] > 1e-5:
            # New position maintains same radius but at new angle
            new_pos = np.array([
                radius * np.cos(current_angle + angle),
                radius * np.sin(current_angle + angle),
                mesh.positions[i, 2]  # Keep Z coordinate unchanged
            ])
        else: 
            new_pos = np.array([
                radius * np.cos(current_angle - angle),
                radius * np.sin(current_angle - angle),
                mesh.positions[i, 2]  # Keep Z coordinate unchanged
            ])
        
        movement = new_pos - mesh.positions[i]
        mesh.positions[i] += movement
        mesh.prev_positions[i] += movement
        
if __name__ == "__main__":
    main()