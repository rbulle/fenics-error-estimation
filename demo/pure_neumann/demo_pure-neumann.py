## Copyright 2019-2020, Jack S. Hale, Raphaël Bulle
## SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np

from dolfin import *
import fenics_error_estimation

mesh = UnitSquareMesh(128, 128)

k = 1
element_V = FiniteElement("CG", triangle, k)
element_R = FiniteElement("Real", triangle, 0)
element = MixedElement([element_V, element_R])
V = FunctionSpace(mesh, element)

u, c = TrialFunctions(V)
v, d = TestFunctions(V)

f = Expression(
    "(2*pow(2*pi,2)+1)*sin(2*pi*x[0]-0.5*pi)*sin(2*pi*x[1]-0.5*pi)", degree=k + 3)

a = inner(grad(u), grad(v))*dx + c*v*dx + u*d*dx
L = inner(f, v)*dx

u_h = Function(V)
A, b = assemble_system(a, L)

solver = PETScLUSolver()
solver.solve(A, u_h.vector(), b)

u_h, c_h = u_h.split()

element_f = FiniteElement("DG", triangle, k + 1)
element_g = FiniteElement("DG", triangle, k)

N = fenics_error_estimation.create_interpolation(element_f, element_g)

V_f = FunctionSpace(mesh, element_f)
e = TrialFunction(V_f)
v = TestFunction(V_f)

n = FacetNormal(mesh)
a_e = inner(grad(e), grad(v))*dx
L_e = inner(f + div(grad(u_h)), v)*dx + \
      inner(jump(grad(u_h), -n), avg(v))*dS

e_h = fenics_error_estimation.estimate(a_e, L_e, N)
error = norm(e_h, "H10")

# Computation of local error indicator
V_e = FunctionSpace(mesh, "DG", 0)
v = TestFunction(V_e)

eta_h = Function(V_e)
eta = assemble(inner(inner(grad(e_h), grad(e_h)), v)*dx)
eta_h.vector()[:] = eta

u_exact = Expression(
    "sin(2*pi*x[0]-0.5*pi)*sin(2*pi*x[1]-0.5*pi)", degree=k + 3)

error_bw = np.sqrt(eta_h.vector().sum())
error_exact = errornorm(u_exact, u_h, "H10")

print("Exact error: {}".format(error_exact))
print("Bank-Weiser error from estimator: {}".format(error_bw))


def test():
    pass
