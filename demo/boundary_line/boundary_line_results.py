from dolfin import *
import numpy as np
import ufl
import sys
sys.path.append('../')
from marking import collective_marking,\
                    estimator_marking,\
                    exact_error_marking
from estimation import BW_estimation,\
                       ZZ_estimation,\
                       RES_estimation
from adaptive_loop import adaptive_refinement
from bank_weiser.bw_estimation import bw_setup,\
                                      verfurth_setup
from oscillations import force_oscillations
from results import convergence_single_plot,\
                    convergence_multiple_plots,\
                    efficiency
from solve_method import SOLVE
from update import update_params
import pandas as pd
from analytical_expressions import cartesian2polar_ufl
import pygmsh as pg
import meshio

import os
import shutil

paths = ['./mesh/', './outputs/']
for p in paths:
    if not os.path.exists(p):
        os.makedirs(p)
    else:
        shutil.rmtree(p)
        os.makedirs(p)


def boundary_line_res(params):
    lc = params.get('initial length')
    '''
    ==== Mesh generation ====
    '''
    geom = pg.built_in.Geometry()

    # Points
    p1 = geom.add_point( [0., 0., 0.], lcar = lc)
    p2 = geom.add_point( [1., 0., 0.], lcar = lc)
    p3 = geom.add_point( [1., 1., 0.], lcar = lc)
    p4 = geom.add_point( [0., 1., 0.], lcar = lc)

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

    meshio.write("./mesh/unit_square.xdmf", meshio.Mesh(
        points=mesh.points,
        cells={'triangle': mesh.cells["triangle"]}))

    # Mesh
    mesh = Mesh()
    with XDMFFile(MPI.comm_world, './mesh/unit_square.xdmf') as f:
        f.read(mesh)

    # Degree FE
    k = params.get('degree')

    # Exact solution
    alpha = 0.7
    x = ufl.SpatialCoordinate(mesh)
    u_exact = x[0]**alpha
    params['exact solution'] = u_exact

    # Data
    f = -ufl.div(ufl.grad(u_exact))
    params['force data'] = f

    # Parameters of the bilinear form
    params['B'] = ufl.as_matrix([[1., 0.], [0., 1.]])
    params['b'] = ufl.as_vector([0., 0.])
    params['c'] = Constant(0.)

    # Boundary data
    class Boundary0(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    boundary0 = Boundary0()
    boundary_parts = [boundary0]
    u_exact = params.get('exact solution')
    bc_data = {0: {'Dirichlet': u_exact}}

    params['boundary parts'] = boundary_parts
    params['boundary data'] = bc_data

    # Norm type
    params['norm'] = 'energy'

    # Marking parameter
    params['marking parameter'] = 0.3

    if k==1:
        names = ['Err']#, 'Res', 'BW', 'Ver', 'ZZ']
        est_methods = [None]#, RES_estimation, BW_estimation, BW_estimation, ZZ_estimation]
        mark_methods = [exact_error_marking]#, estimator_marking, estimator_marking,
                       # estimator_marking, estimator_marking]
        degs = [None]#, None, [k, k+1], None, None]
        osc_methods = [None]#, None, None, None, None]

        results = []
        for name, est_method, mark_method, deg, osc_method in zip(names, est_methods,\
                                                      mark_methods, degs,\
                                                      osc_methods):
            params['mesh'] = mesh
            params['FE space'] = FunctionSpace(mesh, 'CG', k)
            params = update_params(params)
            params['name'] = name
            params['estimation method'] = est_method
            params['marking method'] = mark_method
            # params['oscillations method'] = osc_method

            if 'BW' in name:
                params['degrees'] = deg

            results.append(adaptive_refinement(params))
    else:
        names = ['Err', 'Res', 'BW', 'Ver']#, 'ZZ']
        est_methods = [None, RES_estimation, BW_estimation, BW_estimation]#, ZZ_estimation]
        mark_methods = [exact_error_marking, estimator_marking, estimator_marking,
                        estimator_marking]#, estimator_marking]
        degs = [None, None, [k, k+1], None]#, None]
        osc_methods = [None, None, None, None]#, None]

        results = []
        for name, est_method, mark_method, deg, osc_method in zip(names, est_methods,\
                                                      mark_methods, degs,\
                                                      osc_methods):
            params['mesh'] = mesh
            params['FE space'] = FunctionSpace(mesh, 'CG', k)
            params = update_params(params)
            params['name'] = name
            params['estimation method'] = est_method
            params['marking method'] = mark_method
            # params['oscillations method'] = osc_method

            if 'BW' in name:
                params['degrees'] = deg

            results.append(adaptive_refinement(params))
    return results
