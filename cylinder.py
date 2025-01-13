import numpy as np

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
