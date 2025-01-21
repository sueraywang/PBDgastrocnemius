import time
from Renderer import *
from Physics_edgeConstaint import *
from FileReader import *
import pyvista as pv
import tetgen

def main():    
    # Initialize renderer, simulator, and performance monitor
    renderer = Renderer()
    simulator = Simulator()

    # Position bottom cylinder on ground, top cylinder above it
    top = np.array([0.3, 0.3, 0.9])  # Lift slightly to account for radius
    bottom = np.array([0.0, 0.0, 0.25])     # Start higher to fall onto bottom cylinder
    left = np.array([-0.3, 0.0, 1.0])  # Lift slightly to account for radius
    right = np.array([0.3, 0.0, 1.0])     # Start higher to fall onto bottom cylinder
    # Generate Mesh (real muscle statistics: r = 0.04, h = 0.1 (in meters))
    cylinder = pv.Cylinder(radius=0.25, height=1.0, center=(0, 0, 0), direction=(0, 0, 1), resolution=16).triangulate()
    tet = tetgen.TetGen(cylinder)
    vertices, tets = tet.tetrahedralize()
    surface_faces = generate_surface_faces(tets)
    edges = np.array([
        [tets[:, 0], tets[:, 1]],
        [tets[:, 0], tets[:, 2]],
        [tets[:, 0], tets[:, 3]],
        [tets[:, 1], tets[:, 2]],
        [tets[:, 1], tets[:, 3]],
        [tets[:, 2], tets[:, 3]],
    ]).reshape(-1, 2)
    # Sort the vertex indices of each edge so that (a, b) == (b, a)
    edges = np.sort(edges, axis=1)
    # Remove duplicate edges
    edges = np.unique(edges, axis=0)

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
        #Mesh(top, vertices, edges, surface_faces, tets),
        Mesh(left, vertices, edges, surface_faces, tets),
        #Mesh(bottom, rotated_vertices, edges, surface_faces, tets)
        Mesh(right, vertices, edges, surface_faces, tets)
    ]
    
    try:
        # Initialize bodies
        for i in range(len(bodies)):
            renderer.add_mesh(bodies[i])
            simulator.add_body(bodies[i])
            fix_surfaces(simulator, i, 1.5)
            fix_surfaces(simulator, i, 0.5)
        
        while not renderer.should_close():
            
            glfw.poll_events()
            for i in range(len(bodies)):
                rotate_fixed_vertex(simulator, i, 1.5, np.radians(1.0))
                rotate_fixed_vertex(simulator, i, 0.5, np.radians(-1.0))
                
                simulator.step()
                renderer.update_meshes()
                renderer.render()
            
            time.sleep(DT)
    
    finally:
        renderer.cleanup()

def fix_surfaces(simulator, mesh_idx, z, tolerance=1e-5):
    for i in range(simulator.bodies[mesh_idx].numVertices):
        if abs(simulator.bodies[mesh_idx].positions[i, 2] - z) <= tolerance:
            simulator.bodies[mesh_idx].fixed_vertices.add(i)
            simulator.bodies[mesh_idx].invMasses[i] = 0.0

def rotate_fixed_vertex(simulator, mesh_idx, z, angle):
    for i in simulator.bodies[mesh_idx].fixed_vertices:
        if abs(simulator.bodies[mesh_idx].positions[i, 2] - z) < 1e-5:
            local_coord = simulator.bodies[mesh_idx].positions[i] - np.array([0, 0, z])
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
            simulator.bodies[mesh_idx].positions[i] += movement
            simulator.bodies[mesh_idx].prev_positions[i] += movement

if __name__ == "__main__":
    main()