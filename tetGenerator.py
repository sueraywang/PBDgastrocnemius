import pyvista as pv
import tetgen
import numpy as np

height = 2.0
radius = 0.5
cylinder = pv.Cylinder(radius=radius, height=height, center=(0, 0, 0), direction=(0,0,1), resolution=16).triangulate()
tet = tetgen.TetGen(cylinder)
vertices, tets = tet.tetrahedralize(switches="pq1.414a0.1T")

"""
# Create a plotter with 3 viewports
plotter = pv.Plotter(shape=(1, 2))
# 1. Plot vertices (points)
plotter.subplot(0, 0)
point_cloud = pv.PolyData(vertices)
plotter.add_mesh(point_cloud, color='red', point_size=10, render_points_as_spheres=True)
plotter.add_title("Vertices")
# 2. Plot tetrahedra
plotter.subplot(0, 1)
tetrahedral_mesh = pv.UnstructuredGrid({10: tet.grid.cells.reshape(-1, 5)[:, 1:]}, vertices)
plotter.add_mesh(tetrahedral_mesh, color='lightblue', opacity=0.3, 
                show_edges=True, edge_color='black')
plotter.add_title("Tetrahedra")

# Link all views
plotter.link_views()
plotter.show()
"""