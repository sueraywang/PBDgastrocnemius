import numpy as np
import pyvista as pv
import tetgen

def generate_surface_faces(tetrahedra):
    face_count = {}

    for tet in tetrahedra:
        # Extract all 4 faces of the tetrahedron
        faces = [
            tuple(sorted([tet[0], tet[1], tet[2]])),
            tuple(sorted([tet[0], tet[1], tet[3]])),
            tuple(sorted([tet[0], tet[2], tet[3]])),
            tuple(sorted([tet[1], tet[2], tet[3]]))
        ]

        # Count each face occurrence
        for face in faces:
            if face in face_count:
                face_count[face] += 1
            else:
                face_count[face] = 1

    # Keep only faces that appear once (surface faces)
    surface_faces = [face for face, count in face_count.items() if count == 1]
    return np.array(surface_faces)

def calculate_surface_normals(vertices, surface_faces):
    face_normals = np.zeros((len(surface_faces), 3))
    vertex_normals = np.zeros((len(vertices), 3))
    vertex_counts = np.zeros(len(vertices))
    
    for i, face in enumerate(surface_faces):
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        
        length = np.linalg.norm(normal)
        if length > 0:
            normal = normal / length
            
        # Make normal point outward
        face_center = (v0 + v1 + v2) / 3
        to_center = np.zeros(3) - face_center
        if np.dot(normal, to_center) > 0:
            normal = -normal
            
        face_normals[i] = normal
        
        # Accumulate normals at vertices
        for vertex_idx in face:
            vertex_normals[vertex_idx] += normal
            vertex_counts[vertex_idx] += 1
    
    # Average and normalize vertex normals
    for i in range(len(vertex_normals)):
        if vertex_counts[i] > 0:
            vertex_normals[i] = vertex_normals[i] / vertex_counts[i]
            length = np.linalg.norm(vertex_normals[i])
            if length > 0:
                vertex_normals[i] = vertex_normals[i] / length
                
    return face_normals, vertex_normals

# Create cylinder and tetrahedralize
cylinder = pv.Cylinder(radius=0.5, height=2.0, center=(0, 0, 0), 
                      direction=(0, 0, 1), resolution=16).triangulate()
tet = tetgen.TetGen(cylinder)
vertices, tets = tet.tetrahedralize()

# Get surface faces
surface_faces = generate_surface_faces(tets)

# Calculate normals
face_normals, vertex_normals = calculate_surface_normals(vertices, surface_faces)

# Create surface mesh
surf_mesh = pv.PolyData(vertices, 
                       np.hstack([np.full((len(surface_faces), 1), 3), 
                                surface_faces]).astype(np.int32))
surf_mesh.point_data["normals"] = vertex_normals

# Visualization
plotter = pv.Plotter()

# Add the surface mesh
plotter.add_mesh(surf_mesh, color='lightblue', show_edges=True)

# Add normal vectors as arrows
plotter.add_arrows(vertices, vertex_normals, mag=0.2, color='red')

# Set camera position
plotter.camera_position = [(2, 2, 2),  # Camera position
                          (0, 0, 0),    # Focus point
                          (0, 0, 1)]    # Up vector

# Add axes for reference
plotter.add_axes()

# Show the plot
plotter.show()