import pyvista as pv

# Create a tetrahedron mesh
tetrahedron = pv.Tetrahedron()

# Get the vertices (points)
vertices = tetrahedron.points

# Tetrahedron `PolyData` does not directly store tetrahedral connectivity. 
# However, its surface triangles are stored in the `faces` attribute.

# Extract surface faces (triangle connectivity)
faces = tetrahedron.faces.reshape(-1, 4)  # Each face has 1 size indicator + 3 vertex indices
triangle_faces = faces[:, 1:]  # Ignore the size indicator

# Note: Since this is `PolyData`, the tetrahedral connectivity is not directly available.

# Display the results
print("Vertices (Points):")
print(vertices)

print("\nSurface Faces (Triangle Connectivity):")
print(triangle_faces)
