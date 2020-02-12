import pygmsh as pg
import meshio

lc=0.1
geom = pg.opencascade.Geometry()

geom.add_box([0., 0., 0.], [1., 1., 1.], char_length = lc)

mesh = pg.generate_mesh(geom)

meshio.write("mesh.xdmf", meshio.Mesh(
	points = mesh.points,
	cells={"tetra": mesh.cells["tetra"]}))
