from dolfin import *
import ufl
import sys
sys.path.append('../')
from marking import collective_marking, estimator_marking, exact_error_marking
from estimation import BW_estimation, RES_estimation
from adaptive_loop import adaptive_refinement
from bank_weiser.bw_estimation import bw_setup
from oscillations import force_oscillations
from results import convergence_single_plot
from solve_method import SOLVE
import pandas as pd

params = {}

# Mesh
K = 15
mesh = UnitSquareMesh(K, K)
params['mesh'] = mesh

# FE space
k = 1
params['FE space'] = FunctionSpace(mesh, 'CG', k)

# Equation parameters 
# Exact solution
alpha = 0.7
x = ufl.SpatialCoordinate(mesh)
u_exact = x[0]**alpha

'''
ref_mesh = mesh
for i in range(5):
    ref_mesh = refine(mesh)
ref_x = ufl.SpatialCoordinate(ref_mesh)
u_f = ref_x[0]**alpha
V_f = FunctionSpace(ref_mesh, 'CG', 1)
u_f_p = project(u_f, V_f)
with XDMFFile('./outputs/exact_solution.xdmf') as xdmf:
    xdmf.write_checkpoint(u_f_p, 'exact solution')
'''

params['exact solution'] = u_exact

# Force data
f = -ufl.div(ufl.grad(u_exact))
params['force data'] = f

# Bilinear form coefficients
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

# Estimation method
# params['estimation method'] = RES_estimation
# params['degrees'] = [1, 2]
params['name'] = 'ERR'

# params = bw_setup(params)

# Oscillations method
# params['oscillations method'] = force_oscillations

# Marking strategy
params['marking method'] = exact_error_marking
params['marking parameters'] = [.95, .05, .25]

# Tolerance
params['tolerance'] = 1.e-2

# Iteration max
params['iteration max'] = 40

results = adaptive_refinement(params)

results = [results]
names = [params.get('name')]
# Results frame
for r, result in enumerate(results):
    df = pd.DataFrame(results[r])
    df.to_csv('outputs/{}_data_frame.csv'.format(names[r]))

# Plots
case = 'Boundary line test case'
convergence_single_plot(names, k, case, results, bw_info=None, rate=-0.5,\
                        monochrome=False)
