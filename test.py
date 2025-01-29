import numpy as np

# Define your three vectors
v1 = np.array([[1, 1, 1], [11, 11, 11]])  # First vector
v2 = np.array([[2, 2, 2], [22, 22, 22]])  # Second vector
v3 = np.array([[3, 3, 3], [33, 33, 33]])  # Third vector

# Stack using transpose
matrix = np.stack((v1, v2, v3), axis=1)

# Print the result
print("Matrix with vectors as columns:")
print(matrix)
