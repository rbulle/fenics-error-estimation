## Copyright 2019-2020, Jack S. Hale, Raphaël Bulle
## SPDX-License-Identifier: LGPL-3.0-or-later

# Mixed robust incompressible linear elasticity error estimator from Khan,
# Powell and Silvester (2019) https://doi.org/10.1002/nme.6040. We solve
# the problem from Carstensen and Gedicke https://doi.org/10.1016/j.cma.2015.10.001.
#
# We implement the Poisson problem local estimator detailed in Section 3.5.
# Somewhat remarkably, despite the complexity of the mixed formulation, an
# highly efficient implicit estimator is derived involving the solution of two
# Poisson problems on a special local finite element space. An additional
# explicit estimator computing related to the pressure can be computed. No
# local inf-sup condition must be satisfied by the estimation problem.

# Differences with the presentation in Khan et al.:
# We do not split Equation 50a into two Poisson sub-problems. Instead, we solve
# it as a single monolithic system. In practice we found the performance
# benefits of the splitting negligible, especially given the additional
# complexity.  Note also that the original paper uses quadrilateral finite
# elements. We use the same estimation strategy on triangular finite elements
# without any issues (see Page 28).

import pandas as pd
import numpy as np

from dolfin import *
import ufl

import fenics_error_estimation

parameters["ghost_mode"] = "shared_facet"
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True


# k_f\k_g for bw definition
k_f = 3
k_g = 1

path = 'bw_P{}_P{}/'.format(k_f, k_g)

def main():
    K = 4
    mesh = UnitSquareMesh(K, K)
    mesh.coordinates()[:] -= 0.5
    mesh.coordinates()[:] *= 2.
    
    X_el = VectorElement('CG', triangle, 2)
    M_el = FiniteElement('CG', triangle, 1)

    V_el = MixedElement([X_el, M_el])

    results = []
    for i in range(0, 5):
        V = FunctionSpace(mesh, V_el)
        
        result = {}
        result['num_cells'] = V.mesh().num_cells()

        w_h, err = solve(V)
        print('Exact error = {}'.format(err))
        result['exact_error'] = err

        print('Estimating...')
        eta_h = estimate(w_h)
        result['error_bw'] = np.sqrt(eta_h.vector().sum())
        print('BW = {}'.format(np.sqrt(eta_h.vector().sum())))
        result['hmin'] = mesh.hmin()
        result['hmax'] = mesh.hmax()
        result['num_dofs'] = V.dim()

        print('Estimating (res)...')
        eta_res = residual_estimate(w_h)
        result['error_res'] = np.sqrt(eta_res.vector().sum())
        print('Res = {}'.format(np.sqrt(eta_res.vector().sum())))
        '''
        print('Marking...')
        markers = fenics_error_estimation.dorfler(eta_h, 0.5)
        print('Refining...')
        mesh = refine(mesh, markers, redistribute=True)
        '''

        mesh = refine(mesh)

        with XDMFFile('output/{}bank-weiser/mesh_{}.xdmf'.format(path, str(i).zfill(4))) as f:
            f.write(mesh)

        with XDMFFile('output/{}bank-weiser/disp_{}.xdmf'.format(path, str(i).zfill(4))) as f:
            f.write_checkpoint(w_h.sub(0), 'u_{}'.format(str(i).zfill(4)))

        with XDMFFile('output/{}bank-weiser/pres_{}.xdmf'.format(path, str(i).zfill(4))) as f:
            f.write_checkpoint(w_h.sub(1), 'p_{}'.format(str(i).zfill(4)))

        with XDMFFile('output/{}bank-weiser/eta_{}.xdmf'.format(path, str(i).zfill(4))) as f:
            f.write_checkpoint(eta_h, 'eta_{}'.format(str(i).zfill(4)))

        results.append(result)

    if (MPI.comm_world.rank == 0):
        df = pd.DataFrame(results)
        df.to_pickle('output/{}results.pkl'.format(path))
        print(df)


def solve(V):
    """Solve Stokes problem for viscous incompressible flow using a P2-P1 mixed finite
    element method. This is completely standard."""
    mesh = V.mesh()

    f = Constant((0., 0.))

    w_exact = Expression(('20.*x[0]*pow(x[1], 3)', '5.*pow(x[0], 4)-5.*pow(x[1], 4)', '60.*pow(x[0], 2)*x[1]- 20.*pow(x[1], 3)'), degree = 4)

    u_exact = Expression(('20.*x[0]*pow(x[1], 3)', '5.*pow(x[0], 4)-5.*pow(x[1], 4)'), degree = 4)

    (u, p) = TrialFunctions(V)
    (v, q) = TestFunctions(V)

    a = inner(grad(u), grad(v))*dx
    b = - inner(p, div(v))*dx
    c = - inner(div(u), q)*dx

    B = a + b + c
    L = inner(f, v)*dx

    bcs = DirichletBC(V, w_exact, 'on_boundary')

    A, b = assemble_system(B, L, bcs=bcs)

    w_h = Function(V)

    PETScOptions.set('pc_type', 'lu')
    PETScOptions.set('pc_factor_mat_solver_type', 'mumps')
    solver = PETScKrylovSolver()
    solver.set_from_options()
    solver.solve(A, w_h.vector(), b)

    u_h = w_h.sub(0)
    p_h = w_h.sub(1)

    with XDMFFile('output/displacement.xdmf') as f:
        f.write_checkpoint(u_h, 'u_h')
    with XDMFFile('output/pressure.xdmf') as f:
        f.write_checkpoint(p_h, 'p_h')

    X_el_f = VectorElement('CG', triangle, 3)
    M_el_f = FiniteElement('CG', triangle, 2)

    V_el_f = MixedElement([X_el_f, M_el_f])

    V_f = FunctionSpace(mesh, V_el_f)

    w_h_f = project(w_h, V_f)
    w_f = project(w_exact, V_f)

    with XDMFFile('output/exact_disp.xdmf') as f:
        f.write_checkpoint(w_f.sub(0), 'exact_disp')

    with XDMFFile('output/exact_p.xdmf') as f:
        f.write_checkpoint(w_f.sub(1), 'exact_p')

    w_diff = Function(V_f)
    w_diff.vector()[:] = w_h_f.vector()[:] - w_f.vector()[:]
    local_exact_err_2 = energy_norm(w_diff)
    exact_err = sqrt(sum(local_exact_err_2[:]))
    return w_h, exact_err


def estimate(w_h):
    """Estimator described in Section 3.3 of Liao and Silvester"""
    mesh = w_h.function_space().mesh()

    u_h = w_h.sub(0)
    p_h = w_h.sub(1)

    X_element_f = VectorElement('DG', triangle, k_f)
    X_element_g = VectorElement('DG', triangle, k_g)

    N_X = fenics_error_estimation.create_interpolation(
        X_element_f, X_element_g)

    X_f = FunctionSpace(mesh, X_element_f)

    f = Constant((0., 0.))

    e_X = TrialFunction(X_f)
    v_X = TestFunction(X_f)

    bcs = DirichletBC(X_f, Constant((0., 0.)), 'on_boundary', 'geometric')

    n = FacetNormal(mesh)
    R_T = f + div(grad(u_h)) - grad(p_h)
    I = Identity(2)

    R_E = (1./2.)*jump(-p_h*I + grad(u_h), -n)

    a_X_e = inner(grad(e_X), grad(v_X))*dx
    L_X_e = inner(R_T, v_X)*dx - inner(R_E, 2.*avg(v_X))*dS

    e_h = fenics_error_estimation.estimate(a_X_e, L_X_e, N_X, bcs)

    r_K = div(u_h)

    V_e = FunctionSpace(mesh, 'DG', 0)
    v = TestFunction(V_e)

    eta_h = Function(V_e)
    eta = assemble(inner(inner(grad(e_h), grad(e_h)), v)*dx + inner(inner(r_K, r_K), v)*dx)
    eta_h.vector()[:] = eta

    return eta_h


def residual_estimate(w_h):
    """Residual estimator described in Section 3.1 of Liao and Silvester"""
    mesh = w_h.function_space().mesh()

    f = Constant((0., 0.))

    u_h = w_h.sub(0)
    p_h = w_h.sub(1)

    n = FacetNormal(mesh)
    R_T = f + div(grad(u_h)) - grad(p_h)
    r_T = div(u_h)
    I = Identity(2)
    R_E = (1./2.)*jump(-p_h*I + grad(u_h), -n)

    V = FunctionSpace(mesh, "DG", 0)
    v = TestFunction(V)
    h = CellDiameter(mesh)

    eta_h = Function(V)

    eta = assemble(h**2*R_T**2*v*dx + r_T**
                   2*v*dx + avg(h)*R_E**2*avg(v)*dS)
    eta_h.vector()[:] = eta

    return eta_h


def energy_norm(x):
    mesh = x.function_space().mesh()
    u = x.sub(0)
    p = x.sub(1)

    W = FunctionSpace(mesh, 'DG', 0)
    v = TestFunction(W)

    form = inner(inner(grad(u), grad(u)), v)*dx + inner(inner(p, p), v)*dx
    norm_2 = assemble(form)
    return norm_2


if __name__ == "__main__":
    main()
