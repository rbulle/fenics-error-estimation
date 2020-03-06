'''
Boundary line singularity test case from Mitchell 2013.
The solution u_exact belongs to H^(alpha + 1/2 - epsilon) for all epsilon > 0.
'''

import sys
sys.path.append('../')
from dolfin import *
import ufl
import pygmsh as pg
import meshio
from bank_weiser import bw_estimation
import verfurth_setup
from estimation import BW_estimation, ZZ_estimation, residual_estimation
from uniform_refinement import refinement
from adaptive_refinement import adaptive
from results import *
import sympy as sp
import analytical_expressions as al

import os
import shutil

paths = ['./mesh/', './outputs/']
for p in paths:
    if not os.path.exists(p):
        os.makedirs(p)
    else:
        shutil.rmtree(p)
        os.makedirs(p)

K = 10
mesh = UnitSquareMesh(K, K)
# mesh.coordinates()[:] += eps

# Max iterations
iteration_max = 30

# Tolerance (check the value of the estimator or the exact error if provided)
tolerance = 8.e-2

# Degree FE
k = 1

# Initial FE space for primal problem
init_V_h = FunctionSpace(mesh, 'CG', k)

# Exact solution
alpha = 0.7
x = ufl.SpatialCoordinate(ufl.Mesh(mesh.ufl_coordinate_element()))

u_exact = x[0]**alpha

reg = alpha + 1./2. # Exact solution regularity 

# Data
f = ufl.div(ufl.grad(u_exact))

# Parameters of the bilinear form
B = None
b = None
c = 0.

# Lambda function defining the linear form
lin = lambda v, dx: inner(f, v)*dx

# Boundary data
class Boundary0(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

boundary0 = Boundary0()

boundary_parts = [boundary0]

bc_data = {0: {'Dirichlet': u_exact}}

bw_info = []
results = []
primal_sols = []
estimators = []

names = ['BW1']# , 'ZZ', 'Residual', 'Verfurth']
methods = [BW_estimation]# , ZZ_estimation, residual_estimation, BW_estimation]
dofs_lists = [None]# , None, None, None]
degrees = [[k, k+1]]# , None, None, None]
V_f = None
N = None
for (name, method, dofs_list, degree) in zip(names, methods, dofs_lists, degrees):
    python = False # Decide if we use python or C++ solver for BW 
    result = []
    if 'BW' in name:
        V_f, N = bw_estimation.bw_setup(mesh, degree, dofs_list=dofs_list)
        info = {}
        info['Name'] = name
        info['Dofs list'] = dofs_list
        info['Degrees'] = degrees
        bw_info.append(info)
        python = False
    if 'Verfurth' in name:
        V_f, N = verfurth_setup.setup(mesh)
        info = {}
        info['Name'] = name
        info['Dofs list'] = dofs_list
        info['Degrees'] = degrees
        bw_info.append(info)
        python = True

    result, u_h, est_W = adaptive(name, method,\
                                  init_V_h, lin,\
                                  boundary_parts,\
                                  bc_data, tolerance,\
                                  iteration_max, B_init=B, b_init=b, c_init=c,\
                                  V_f = V_f, N = N,\
                                  u_exact = u_exact,\
                                  f = f, osc_ref=True, python=python)
    results.append(result)
    primal_sols.append(u_h)
    estimators.append(est_W)

# ==== CONVERGENCE PLOTS ====
gdim = mesh.geometry().dim()
convergence_single_plot(names, k, 'Boundary line case (effect of inital mesh)', results, bw_info=None,\
                        rate= - (reg-1.)/gdim, monochrome=True)
convergence_multiple_plots(names, k, 'Boundary line case (effect of initial mesh)', results,\
                           rate = - (reg-1.)/gdim, monochrome=True)


# ==== EFFICIENCY PLOT ====
efficiency(names, k, 'Boundary line case', results, monochrome=True)

# ==== Data frame ====
for r, result in enumerate(results):
    df = pd.DataFrame(results[r])
    df.to_csv('outputs/{}_data_frame.csv'.format(names[r]))
