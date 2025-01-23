import numpy as np
import pyvista as pv
from collections import defaultdict

def generate_edges(tets):
    # Initialize list to store all edges
    all_edges = []
    
    # For each tetrahedron
    for tet in tets:
        # Add all 6 edges of the tetrahedron
        all_edges.extend([
            [tet[0], tet[1]],
            [tet[0], tet[2]],
            [tet[0], tet[3]],
            [tet[1], tet[2]],
            [tet[1], tet[3]],
            [tet[2], tet[3]]
        ])
    
    # Convert to numpy array
    edges = np.array(all_edges)
    
    # Sort vertex indices within each edge
    edges = np.sort(edges, axis=1)
    
    # Remove duplicate edges
    edges = np.unique(edges, axis=0)
    
    return edges

# Generate the surface faces from tetrahedral elements
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

        # Count each face occurance
        for face in faces:
            if face in face_count:
                face_count[face] += 1
            else:
                face_count[face] = 1

    # Keep only faces that appear once
    surface_faces = [face for face, count in face_count.items() if count == 1]

    return np.array(surface_faces)

def create_adjacency_list(edges):
    # Create vertex to edge mapping
    vertex_to_edges = defaultdict(list)
    for edge_idx, (v1, v2) in enumerate(edges):
        vertex_to_edges[v1].append(edge_idx)
        vertex_to_edges[v2].append(edge_idx)
    
    # Create edge adjacency list
    edge_adjacency = defaultdict(set)
    for vertex, edge_list in vertex_to_edges.items():
        # Add all pairs of edges that share this vertex
        for i in range(len(edge_list)):
            for j in range(i + 1, len(edge_list)):
                edge_adjacency[edge_list[i]].add(edge_list[j])
                edge_adjacency[edge_list[j]].add(edge_list[i])
    
    return edge_adjacency

def create_tet_adjacency_list(tets):
    # Create vertex to tetrahedra mapping
    vertex_to_tets = defaultdict(list)
    for tet_idx, tet in enumerate(tets):
        for vertex in tet:
            vertex_to_tets[vertex].append(tet_idx)
    
    # Create tetrahedra adjacency list
    tet_adjacency = defaultdict(set)
    for vertex, tet_list in vertex_to_tets.items():
        # Add all pairs of tetrahedra that share this vertex
        for i in range(len(tet_list)):
            for j in range(i + 1, len(tet_list)):
                tet_adjacency[tet_list[i]].add(tet_list[j])
                tet_adjacency[tet_list[j]].add(tet_list[i])
    
    return tet_adjacency

def color_tetrahedra(tets):
    tet_adjacency = create_tet_adjacency_list(tets)
    num_tets = len(tets)
    colors = [-1] * num_tets  # -1 represents uncolored
    
    # For each tetrahedron
    for tet_idx in range(num_tets):
        # Get colors used by adjacent tetrahedra
        used_colors = set()
        for adj_tet in tet_adjacency[tet_idx]:
            if colors[adj_tet] != -1:
                used_colors.add(colors[adj_tet])
        
        # Find the smallest unused color
        color = 0
        while color in used_colors:
            color += 1
        
        colors[tet_idx] = color
    
    return np.array(colors)

def color_edges(edges):
    edge_adjacency = create_adjacency_list(edges)
    num_edges = len(edges)
    colors = [-1] * num_edges  # -1 represents uncolored
    
    # For each edge
    for edge_idx in range(num_edges):
        # Get colors used by adjacent edges
        used_colors = set()
        for adj_edge in edge_adjacency[edge_idx]:
            if colors[adj_edge] != -1:
                used_colors.add(colors[adj_edge])
        
        # Find the smallest unused color
        color = 0
        while color in used_colors:
            color += 1
        
        colors[edge_idx] = color
    
    return np.array(colors)

def visualize_colored_edges(vertices, edges, colors):
    # Create line segments for visualization
    points = vertices
    lines = np.empty((len(edges), 3), dtype=int)
    lines[:, 0] = 2  # Each line has 2 points
    lines[:, 1:] = edges
    
    # Create the PolyData object with lines
    mesh = pv.PolyData(points, lines=lines)
    mesh.cell_data["colors"] = colors
    
    # Create plotter with a white background
    plotter = pv.Plotter()
    plotter.background_color = 'white'
    
    # Add the mesh with colored edges
    plotter.add_mesh(mesh, scalars="colors", cmap="tab20", line_width=3, render_lines_as_tubes=True)
    
    # Set a better camera position and zoom
    plotter.camera_position = 'xy'
    plotter.show()

def visualize_surface(vertices, faces):
    # Create triangular surface mesh
    surf_mesh = pv.PolyData(vertices, faces=np.c_[np.full(len(faces), 3), faces])
    
    # Create plotter with a white background
    plotter = pv.Plotter()
    plotter.background_color = 'white'
    
    # Add the surface mesh
    plotter.add_mesh(surf_mesh, style='wireframe', line_width=2, color='black')
    plotter.add_mesh(surf_mesh, opacity=0.5, color='lightgray')
    
    # Set a better camera position and zoom
    plotter.camera_position = 'xy'
    plotter.show()
    
def visualize_colored_tetrahedra(vertices, tets, colors):
    # Create an unstructured grid from the tetrahedral mesh
    cells = np.c_[np.full(len(tets), 4), tets]  # 4 points per tetrahedron
    grid = pv.UnstructuredGrid(cells, np.ones(len(tets), dtype=np.int32) * pv.CellType.TETRA, vertices)
    
    # Add colors to the grid
    grid.cell_data["colors"] = colors
    
    # Create plotter with a white background
    plotter = pv.Plotter()
    plotter.background_color = 'white'
    
    # Add the colored tetrahedral mesh with enhanced visualization settings
    plotter.add_mesh(
        grid, 
        scalars="colors",
        cmap="tab20",
        opacity=0.2,  # More transparent to see internal structure
        show_edges=True,
        edge_color='black',
        line_width=1,
        show_scalar_bar=True,
        scalar_bar_args={"title": "Tetrahedra Colors"}
    )
    
    # Enable depth peeling for proper transparency rendering
    plotter.enable_depth_peeling(10)
    
    # Set multiple camera views for better understanding
    plotter.camera_position = 'iso'
    plotter.camera.zoom(1.5)
    
    # Add axes for orientation
    plotter.add_axes()
    
    plotter.show()