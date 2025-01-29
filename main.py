import time
import tetgen
from colorGraph import *
from Renderer import *
from Physics import *

fixed_index = 3

def main():    
    # Initialize renderer and simulator
    renderer = Renderer(cameraRadius=6.0, lookAtPosition=np.array([0.0, 0.0, 1.0]), h_angle=-np.pi/2, v_angle=np.pi/2)
    simulator = Simulator()

    # Position bottom cylinder on ground, top cylinder above it
    top = np.array([0.0, 0.0, 2.0])  # Lift slightly to account for radius
    bottom = np.array([0.0, 0.0, 0.25])     # Start higher to fall onto bottom cylinder
    
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
    #vertices = tetrahedron.points
    #tets = np.array([[0, 1, 2, 3]])
    print(vertices, tets)
    surface_faces = tetrahedron.faces.reshape(-1, 4)[:, 1:]  # Ignore the size indicator
    """
    #mesh_size = compute_min_vertex_distance(vertices)
    #print(f"Minimum vertex-to-vertex distance (mesh size): {mesh_size}")
    #print(vertices.shape)
    
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

    meshes = [
        Mesh(top, vertices, surface_faces, tets)
        #Mesh(bottom, vertices @ R2.T, surface_faces, tets)
    ]
    
    try:
        for i in range(len(meshes)):
            renderer.add_mesh(meshes[i])
            simulator.add_mesh(meshes[i])
            #fix_surfaces(simulator, i, 1.5)
            #fix_surfaces(simulator, i, 0.5)

            #simulator.bodies[i].fixed_vertices.add(fixed_index)
            #simulator.bodies[i].invMasses[fixed_index] = 0.0

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