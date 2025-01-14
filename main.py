import time
import glfw
from Renderer import Renderer
from Physics import *
from FileReader import *

def main():
    """
    node_file = "PBDMuscles/cylinder.1.node"
    tet_file = "PBDMuscles/cylinder.1.ele"
    edge_file = "PBDMuscles/cylinder.1.edge"
    face_file = "PBDMuscles/cylinder.1.face"
    vertices = read_node_file(node_file)
    tets = read_ele_file(tet_file)
    edges = read_edge_file(edge_file)
    faces = read_face_file(face_file)
    """
    
    vertices, tets = read_obj('PBDMuscles/mesh.obj')
    surface_faces = generate_surface_faces(tets)

    mesh = Mesh(vertices, tets)
    renderer = Renderer(vertices, surface_faces)

    while not renderer.should_close():
        glfw.poll_events()
        mesh.step()
        renderer.update_vertex_positions(mesh.positions)
        renderer.render()
        time.sleep(DT)

    renderer.cleanup()

if __name__ == "__main__":
    main()