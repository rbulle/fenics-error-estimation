## Copyright 2019-2020, Jack S. Hale, Raphaël Bulle
## SPDX-License-Identifier: LGPL-3.0-or-later
import pygmsh as pg
import meshio

lc = 0.5

geom = pg.built_in.Geometry()

# Points
p1 = geom.add_point( [0.5, 0.5, 0], lcar = lc)
p2 = geom.add_point( [-0.5, 0.5, 0], lcar = lc)
p3 = geom.add_point( [-0.5, -0.5, 0], lcar = lc)
p4 = geom.add_point( [0.5, -0.5, 0], lcar = lc)

# Lines 
l1 = geom.add_line(p1, p2)
l2 = geom.add_line(p2, p3)
l3 = geom.add_line(p3, p4)
l4 = geom.add_line(p4, p1)

# Surface
loop = geom.add_line_loop( [l1, l2, l3, l4])
surf = geom.add_plane_surface(loop)

mesh = pg.generate_mesh(geom)

mesh.points = mesh.points[:, :2] # Used to convert 3D mesh into 2D

meshio.write("./mesh.xdmf", meshio.Mesh(
    points=mesh.points,
    cells={'triangle': mesh.cells["triangle"]}))

