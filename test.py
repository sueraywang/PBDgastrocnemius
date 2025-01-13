import numpy as np
from cylinder import *

a = np.array([0,0,0,0])
indices = [0, 1, 0, 2, 3, 3, 3, 3]
values = np.repeat([1,2,3], 4)

#np.add.at(a, indices, values)

print(values) 
print(a) 

node_file = "PBDMuscles/cylinder.1.node"
tet_file = "PBDMuscles/cylinder.1.ele"
edge_file = "PBDMuscles/cylinder.1.edge"
face_file = "PBDMuscles/cylinder.1.face"
    
vertices = read_node_file(node_file)
tets = read_ele_file(tet_file)
edges = read_edge_file(edge_file)
faces = read_face_file(face_file)

print(tets.flatten().shape)