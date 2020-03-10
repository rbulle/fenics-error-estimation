## Copyright 2019-2020, Jack S. Hale, Raphaël Bulle
## SPDX-License-Identifier: LGPL-3.0-or-later
import pygmsh as pg
import meshio

lc = 0.05
geom = pg.opencascade.Geometry()

big_box = geom.add_box([-0.5, -0.5, 0.], [1., 1., 0.5], char_length= lc)
corner_box = geom.add_box([-0.5, -0.5, 0.], [0.5, 0.5, 0.5], char_length=lc)

geom.boolean_difference([big_box], [corner_box])

mesh = pg.generate_mesh(geom)

meshio.write("./mesh.xdmf", meshio.Mesh(
    points=mesh.points,
    cells={"tetra": mesh.cells["tetra"]}))

