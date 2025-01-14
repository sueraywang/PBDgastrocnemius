import numpy as np

# tetGen files reader
def read_node_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        # Parse the header
        header = lines[0].strip().split()
        num_nodes = int(header[0])  # Number of nodes
        dim = int(header[1])  # Dimension (should be 3)
        num_attrs = int(header[2])  # Number of attributes
        has_markers = int(header[3])  # Boundary markers

        # Initialize list for node data
        nodes = []

        # Process the lines with node data
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) > 0 and parts[0].isdigit():  # Ensure the line contains valid data
                coords = list(map(float, parts[1:4]))  # Extract x, y, z
                nodes.append(coords)
        
        return np.array(nodes)

def read_ele_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        # Parse the header
        header = lines[0].strip().split()
        num_elements = int(header[0])  # Number of elements
        nodes_per_element = int(header[1])  # Number of nodes per element (should be 4)
        num_attrs = int(header[2])  # Number of attributes

        # Initialize list for element data
        elements = []

        # Process the lines with element data
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) > 0 and parts[0].isdigit():  # Ensure the line contains valid data
                node_indices = list(map(int, parts[1:1 + nodes_per_element]))  # Extract node indices
                elements.append(node_indices)
        
        return np.array(elements) - 1 # Subtract 1 for zero-based indexing

def read_edge_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        # Parse the header
        header = lines[0].strip().split()
        num_edges = int(header[0])  # Number of edges
        has_markers = int(header[1])  # Boundary markers

        # Initialize list for edge data
        edges = []

        # Process the lines with edge data
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) > 0 and parts[0].isdigit():  # Ensure the line contains valid data
                node_indices = list(map(int, parts[1:3]))  # Extract node indices
                edges.append((node_indices))
        
        return np.array(edges) - 1 # Subtract 1 for zero-based indexing

def read_face_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        
        # Parse the header
        header = lines[0].strip().split()
        num_faces = int(header[0])  # Number of faces
        has_markers = int(header[1])  # Boundary markers

        # Initialize list for face data
        faces = []

        # Process the lines with face data
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) > 0 and parts[0].isdigit():  # Ensure the line contains valid data
                node_indices = list(map(int, parts[1:4]))  # Extract node indices
                faces.append((node_indices))
        
        return np.array(faces) - 1 # Subtract 1 for zero-based indexing

# obj files reader
def read_obj(filename):
    vertices = []
    tetrahedra = []

    with open(filename, 'r') as file:
        for line in file:
            parts = line.split()
            if not parts:
                continue
            # Parse vertices
            if parts[0] == 'v':
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            # Parse tetrahedra
            elif parts[0] == 'f':
                # Convert 1-based OBJ indices to 0-based Python indices
                tetrahedra.append([int(idx) - 1 for idx in parts[1:]])

    # Convert lists to numpy arrays
    vertices = np.array(vertices)
    tetrahedra = np.array(tetrahedra, dtype=int)

    return vertices, tetrahedra

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
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create a collection of triangular faces
    face_vertices = vertices[surface_faces]
    collection = Poly3DCollection(face_vertices, alpha=0.5, edgecolor='k')
    ax.add_collection3d(collection)

    # Plot vertices
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='r')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

