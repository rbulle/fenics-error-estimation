import numpy as np

from dolfin import *
import fenics_error_estimation

mesh = UnitCubeMesh(96, 96, 96)

k = 1
V = FunctionSpace(mesh, 'CG', k)

u = TrialFunction(V)
v = TestFunction(V)

f = Expression("12. * pow(pi, 2) * sin(2. * pi * x[0]) * sin(2. * pi * x[1]) * sin(2. * pi * x[2])", degree=k + 3)

a = inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx

class Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

bcs = DirichletBC(V, Constant(0.0), 'on_boundary')

u_h = Function(V)
A, b = assemble_system(a, L, bcs=bcs)

PETScOptions.set("ksp_type", "cg")
PETScOptions.set("ksp_rtol", 1E-10)
PETScOptions.set("ksp_monitor_true_residual")
PETScOptions.set("pc_type", "hypre")
PETScOptions.set("pc_hypre_type", "boomeramg")
PETScOptions.set("pc_hypre_boomeramg_strong_threshold", 0.5)
PETScOptions.set("pc_hypre_boomeramg_coarsen_type", "HMIS")
PETScOptions.set("pc_hypre_boomeramg_agg_nl", 4)
PETScOptions.set("pc_hypre_boomeramg_agg_num_paths", 2)
PETScOptions.set("pc_hypre_boomeramg_interp_type", "ext+i")
PETScOptions.set("pc_hypre_boomeramg_truncfactor", 0.35)
PETScOptions.set("ksp_view")
solver = PETScKrylovSolver()
solver.set_from_options()
solver.solve(A, u_h.vector(), b)

element_f = FiniteElement("DG", tetrahedron, k + 1)
element_g = FiniteElement("DG", tetrahedron, k)
V_f = FunctionSpace(mesh, element_f)

N = fenics_error_estimation.create_interpolation(element_f, element_g)

e = TrialFunction(V_f)
v = TestFunction(V_f)

f = Expression("12. * pow(pi, 2) * sin(2. * pi * x[0]) * sin(2. * pi * x[1]) * sin(2. * pi * x[2])", degree=k + 3)

bcs = DirichletBC(V_f, Constant(0.0), "on_boundary", "geometric")

n = FacetNormal(mesh)
a_e = inner(grad(e), grad(v)) * dx
L_e = inner(f + div(grad(u_h)), v) * dx + \
    inner(jump(grad(u_h), -n), avg(v)) * dS

e_h = fenics_error_estimation.estimate(a_e, L_e, N, bcs)
# error = norm(e_h, "H10")

# Computation of local error indicator
V_e = FunctionSpace(mesh, "DG", 0)
v = TestFunction(V_e)

eta_h = Function(V_e)
eta = assemble(inner(inner(grad(e_h), grad(e_h)), v) * dx)
eta_h.vector()[:] = eta

print('BW estimator =', np.sqrt(sum(eta_h.vector()[:])))

with XDMFFile("output/mesh.xdmf") as of:
    of.write(mesh)

with XDMFFile("output/u.xdmf") as of:
    of.write(u_h)

with XDMFFile("output/eta.xdmf") as of:
    of.write(eta_h)
