## Copyright 2019-2020, Jack S. Hale, Raphaël Bulle
## SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
import pandas as pd

from dolfin import *
import ufl

import fenics_error_estimation

parameters['ghost_mode'] = 'shared_facet'

k = 2

def main():
    K = 15
    init_mesh = UnitSquareMesh(K, K)

    results = {}
    if k == 1:
        names = ['bw', 'ver', 'zz', 'res', 'exact']
        estimators = [bw_estimate, ver_estimate, zz_estimate, res_estimate, exact_estimate]
    else:
        names = ['bw', 'ver', 'res', 'exact']
        estimators = [bw_estimate, ver_estimate, res_estimate, exact_estimate]

    for name, estimator in zip(names, estimators):
        print('{} estimation...'.format(name))
        result, dofs, error = refinement_loop(init_mesh, name, estimator, results)
        results['dofs_{}'.format(name)] = dofs
        results['{}'.format(name)] = result
        results['{}_exact_error'.format(name)] = error
    
    if (MPI.comm_world.rank == 0):
        df = pd.DataFrame(results)
        df.to_pickle("output/results.pkl")
        print(df)

def refinement_loop(mesh, name, estimator, results):
    result = []
    error = []
    dofs = []
    for i in range(0, 16):
        u_exact, f = pbm_data(mesh)
        V = FunctionSpace(mesh, "CG", k)
        u_h = solve(V, u_exact, f)
        with XDMFFile("output/{}/u_h_{}.xdmf".format(name, str(i).zfill(4))) as xdmf:
            xdmf.write(u_h)

        eta = estimator(u_h, f, u_exact=u_exact)
        with XDMFFile("output/{}/eta_{}.xdmf".format(name, str(i).zfill(4))) as xdmf:
            xdmf.write_checkpoint(eta, "eta_{}".format(name))

        result.append(np.sqrt(eta.vector().sum()))
        dofs.append(V.dim())

        exact_err = exact_estimate(u_h, f, u_exact=u_exact)

        error.append(np.sqrt(exact_err.vector().sum()))

        markers = fenics_error_estimation.dorfler(eta, 0.5)
        mesh = refine(mesh, markers, redistribute=True)

        with XDMFFile("output/{}/mesh_{}.xdmf".format(name, str(i).zfill(4))) as xdmf:
            xdmf.write(mesh)

    return result, dofs, error

def bw_estimate(u_h, f, u_exact= None, df=k+1, dg=k, verf=False, dof_list=None):
    mesh = u_h.function_space().mesh()

    if verf:
        element_f = FiniteElement('Bubble', triangle, 3)\
                    + FiniteElement('DG', triangle, 2)
        element_g = FiniteElement('DG', triangle, 1)
    else:
        element_f = FiniteElement("DG", triangle, df)
        element_g = FiniteElement("DG", triangle, dg)

    N = fenics_error_estimation.create_interpolation(element_f, element_g, dof_list)

    V_f = FunctionSpace(mesh, element_f)

    e = TrialFunction(V_f)
    v = TestFunction(V_f)

    bcs = DirichletBC(V_f, Constant(0.0), "on_boundary", "geometric")

    n = FacetNormal(mesh)
    a_e = inner(grad(e), grad(v))*dx
    L_e = inner(f + div(grad(u_h)), v)*dx + \
        inner(jump(grad(u_h), -n), avg(v))*dS

    e_h = fenics_error_estimation.estimate(a_e, L_e, N, bcs)
    error = norm(e_h, "H10")

    # Computation of local error indicator
    V_e = FunctionSpace(mesh, "DG", 0)
    v = TestFunction(V_e)

    eta_h = Function(V_e, name="eta_h")
    eta = assemble(inner(inner(grad(e_h), grad(e_h)), v)*dx)
    eta_h.vector()[:] = eta
    return eta_h

def ver_estimate(u_h, f, u_exact=None):
    eta_h = bw_estimate(u_h, f, verf=True)
    return eta_h

def zz_estimate(u_h, f, u_exact=None):
    mesh = u_h.function_space().mesh()
    k = u_h.ufl_element().degree()

    try:
        assert k == 1
    except AssertionError:
        print('Finite element degree must be 1 to use ZZ estimator.')

    W = VectorFunctionSpace(mesh, 'CG', 1, 2)

    # Global grad recovery
    w_h = TrialFunction(W)
    v_h = TestFunction(W)

    A = assemble(inner(w_h, v_h) * dx, form_compiler_parameters={'quadrature_rule': 'vertex', 'representation': 'quadrature'})
    b = assemble(inner(grad(u_h), v_h) *dx)

    G_h = Function(W)

    PETScOptions.set("ksp_type", "cg")
    PETScOptions.set("pc_type", "hypre")
    PETScOptions.set("pc_hypre_type", "boomeramg")
    PETScOptions.set("ksp_rtol", 1.0e-9)
    solver = PETScKrylovSolver()
    solver.set_operator(A)
    solver.set_from_options()

    solver.solve(A, G_h.vector(), b)

    disc_zz = grad(u_h) - G_h

    # Computation of local error indicator
    V_e = FunctionSpace(mesh, "DG", 0)
    v = TestFunction(V_e)

    eta_h = Function(V_e, name="eta_h")
    eta = assemble(inner(inner(disc_zz, disc_zz), v)*dx)
    eta_h.vector()[:] = eta
    return eta_h

def res_estimate(u_h, f, u_exact=None):
    mesh = u_h.function_space().mesh()

    n = FacetNormal(mesh)
    h_T = CellDiameter(mesh)
    h_E = FacetArea(mesh)

    r = f + div(grad(u_h))
    J_h = jump(grad(u_h), -n)

    V_e = FunctionSpace(mesh, 'DG', 0)
    v_e = TestFunction(V_e)

    R = h_T**2*inner(inner(r,r),v_e)*dx + avg(h_E)*inner(inner(J_h, J_h), avg(v_e))*dS

    # Computation of local error indicator
    V_e = FunctionSpace(mesh, "DG", 0)

    eta_h = Function(V_e)
    eta = assemble(R)[:]
    eta_h.vector()[:] = eta
    return eta_h

def exact_estimate(u_h, f, u_exact=None):
    mesh = u_h.function_space().mesh()
    V_e = FunctionSpace(mesh, "DG", 0)
    eta_exact = Function(V_e, name="eta_exact")
    v = TestFunction(V_e)
    eta_exact.vector()[:] = assemble(inner(inner(grad(u_h - u_exact), grad(u_h - u_exact)), v)*dx(mesh))
    return eta_exact

def solve(V, u_exact, f):
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx
    L = inner(f, v)*dx

    def all_boundary(x, on_boundary):
        return on_boundary

    bcs = DirichletBC(V, u_exact ,all_boundary)

    A, b = assemble_system(a, L, bcs=bcs)

    u_h = Function(V, name='u_h')
    solver = PETScLUSolver('mumps')
    solver.solve(A, u_h.vector(), b)

    return u_h

def pbm_data(mesh):
    # Exact solution
    x = ufl.SpatialCoordinate(mesh)

    alpha = 0.7    # Radius of internal layer
    u_exact = x[0]**alpha

    # Data
    f = -ufl.div(ufl.grad(u_exact))
    return u_exact, f

if __name__ == "__main__":
    main()


def test():
    main()
