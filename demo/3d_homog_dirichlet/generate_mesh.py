import pygmsh as pg
import meshio

lc=0.5
geom = pg.opencascade.Geometry()

big_box = geom.add_box([-0.5, -0.5, -0.5], [1., 1., 1.], char_length = lc)
corner_box = geom.add_box([-0.5, -0.5, -0.5], [0.5, 0.5, 1.], char_length = lc)

geom.boolean_difference([big_box], [corner_box])

mesh = pg.generate_mesh(geom)

meshio.write("mesh.xdmf", meshio.Mesh(
	points = mesh.points,
	cells={"tetra": mesh.cells["tetra"]}))
