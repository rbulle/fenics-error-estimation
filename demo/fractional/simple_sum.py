from dolfin import *
import dolfin as do
import numpy as np
import sympy
from mpmath import *
import matplotlib.pyplot as plt
import ufl
import sys
import fenics_error_estimation
from fenics_error_estimation.estimate import estimate_python
import pandas as pd

parameters["ghost_mode"] = 'shared_facet'

p = 1

def main():
    f = Constant(1.)

    for i, a in enumerate(a_s):
        results = []
        mesh_f = UnitIntervalMesh(1000)
        W = FunctionSpace(mesh_f, 'CG', 4)
        u_exact, _ = sum_solve(W, f, func)
        for nk in range(10, 300, 10):
            result = {}
            mesh = UnitIntervalMesh(nk)
            V = FunctionSpace(mesh, 'CG', p)
            V_f = FunctionSpace(mesh, 'DG', p + 1)

            u = Function(V)
            eta = Function(V_f)

            uh, etah = sum_solve(V, f, func)

            u.vector()[:] = np.sum([uhi.vector()[:] for uhi in uh])
            eta.vector()[:] = np.sum([etahi.vector()[:] for etahi in etah])
            result['num dofs'] = V.dim()
            u_W = project(u, W)
            result['error'] = errornorm(u_exact, u_W, norm_type="L2")
            result["error bw"] = do.norm(eta, norm_type= 'L2')

            results.append(result)

        df = pd.DataFrame(results)
        df.to_pickle('results_simple_sum_{}.pkl'.format(i))
        print(df)
    return

def sum_solve(V, f, func):
    mesh = V.mesh()
    u_h = []
    eta = []
    for i, k in enumerate([1, 2]):
        print('[Direct sum] Solve num. {}'.format(i))
        u_pde = solve_pde(V, f, k)
        eta_pde = bw_estimate(u_pde, f, k)

        u_h.apppend(u_pde)
        eta.append(eta_pde)
    return u_h, eta

def solve_pde(V, f, k):
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v)) * dx + (Constant(k) * inner(u, v)) * dx

    L = inner(f, v) * dx

    bcs = DirichletBC(V, Constant(0.), 'on_boundary')

    A, b = assemble_system(a, L, bcs=bcs)

    u_h = Function(V)

    PETScOptions.set("ksp_type", "cg")
    PETScOptions.set("ksp_rtol", 1E-10)
    PETScOptions.set("ksp_monitor_true_residual")
    PETScOptions.set("pc_type", "hypre")
    PETScOptions.set("pc_hypre_type", "boomeramg")
    solver = PETScKrylovSolver()
    solver.set_from_options()
    solver.solve(A, u_h.vector(), b)
    return u_h

def bw_estimate(u_h, f, k, df=p+1, dg=p, verf=False, dof_list=None):
    mesh = u_h.function_space().mesh()

    if verf:
        element_f = FiniteElement('Bubble', UnitIntervalMesh, 3)\
                    + FiniteElement('DG', UnitIntervalMesh, 2)
        element_g = FiniteElement('DG', UnitIntervalMesh, 1)
    else:
        element_f = FiniteElement("DG", interval, df)
        element_g = FiniteElement("DG", interval, dg)

    N = fenics_error_estimation.create_interpolation(element_f, element_g, dof_list)

    V_f = FunctionSpace(mesh, element_f)

    e = TrialFunction(V_f)
    v = TestFunction(V_f)

    bcs = DirichletBC(V_f, Constant(0.0), "on_boundary", "geometric")

    n = FacetNormal(mesh)
    a_e = Constant(k)*inner(e,v)*dx + inner(grad(e), grad(v))*dx
    L_e = inner(f - Constant(k)*u_h + div(grad(u_h)), v)*dx + \
        inner(jump(grad(u_h), -n), avg(v))*dS

    e_h = estimate_python(a_e, L_e, N, bcs)

    '''
    error = norm(e_h, "H10")

    # Computation of local error indicator
    V_e = FunctionSpace(mesh, "DG", 0)
    v = TestFunction(V_e)

    eta_h = Function(V_e, name="eta_h")
    eta = assemble(inner(inner(grad(e_h), grad(e_h)), v)*dx)
    eta_h.vector()[:] = eta
    '''
    return e_h

if __name__ == "__main__":
    main()


