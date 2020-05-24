from dolfin import *
import dolfin as do
import numpy as np
import sympy
import quadpy
from mpmath import *
import matplotlib.pyplot as plt
import ufl
import sys
import fenics_error_estimation
from fenics_error_estimation.estimate import estimate_python
import pandas as pd
parameters['ghost_mode'] = 'shared_facet'
import sys
sys.setrecursionlimit(100000)

p = 1           # FE polynomial order

def main():
    alphas = [1.5, 1., 0.75, 0.5]
    for i, alpha in enumerate(alphas):
        results = []
        mesh = UnitIntervalMesh(10000)
        mesh.coordinates()[:] *= 2. * np.pi
        x = ufl.SpatialCoordinate(mesh)
        u_exact_init = np.power(2., -alpha) * ufl.sin(2. * x[0])
        #u_exact = ufl.cos(x[0]/2.)
        #f_vect = [Constant((1. / np.pi) * (2. * i ** (alpha - 1.)) / (4. * i ** 2. - 1.)) * ufl.sin(i * x[0]) for i in range(1, 5000)]
        #f_sum = sum(f_vect)
        #W = FunctionSpace(mesh, 'CG', p+3)
        #f_fine = customproj(f_sum, W)
        for nk in range(100, 1500, 100):
            result = {}
            mesh = UnitIntervalMesh(nk)
            mesh.coordinates()[:] *= 2.*np.pi
            x = ufl.SpatialCoordinate(mesh)
            g = Expression('-(1./pi)*x[0] + 1.', degree = 1)    # Dirichlet boundary datum
            V = FunctionSpace(mesh, 'CG', p)
            W = FunctionSpace(mesh, 'CG', p+3)
            #f = project(f_fine, W)
            f = ufl.sin(2.*x[0])
            u_exact_vect = [u_exact_init(x) + g(x) for x in W.tabulate_dof_coordinates()]
            '''
            plt.figure()
            plt.plot(np.ndarray.flatten(W.tabulate_dof_coordinates()), u_exact_vect, label='u_exact')
            '''
            u_exact = Function(W)
            u_exact.vector()[:] = u_exact_vect

            u_exp, eta_exp, num_solve_exp = exponential_quadrature(V, f, alpha)
            u_boundary = boundary_problem(V, g)

            u = Function(V)
            u.vector()[:] = u_exp.vector()[:] + u_boundary.vector()[:]
            '''
            u_exp_vect = [u_exp(Point(x)) for x in W.tabulate_dof_coordinates()]
            u_bound_vect = [u_boundary(Point(x)) for x in W.tabulate_dof_coordinates()]
            u_vect = [u(Point(x)) for x in W.tabulate_dof_coordinates()]
            plt.plot(np.ndarray.flatten(W.tabulate_dof_coordinates()), u_vect, label='u_h')
            plt.plot(np.ndarray.flatten(W.tabulate_dof_coordinates()), u_exp_vect, label='u_exp')
            plt.plot(np.ndarray.flatten(W.tabulate_dof_coordinates()), u_bound_vect, label='u_boundary')
            plt.legend()
            plt.savefig("./heterogeneous_solution.pdf")
            '''

            result['num dofs'] = V.dim()
            result['error'] = errornorm(u_exact, u, norm_type="L2")
            result['error bw'] = do.norm(eta_exp, norm_type='L2')
            results.append(result)

        df = pd.DataFrame(results)
        df.to_pickle('results_1d_heterogeneous_d_{}.pkl'.format(i))
        print(df)
    return

def exponential_quadrature(V, f, alpha):
    mesh = V.mesh()
    num_solve = 0
    k = 0.4  # Size factor for the quadrature subdivision

    M = np.ceil(np.pi ** 2 / (2. * alpha * k ** 2))  # Lower bound quadrature sum
    N = np.ceil(np.pi ** 2 / (4. * (1. - alpha / 2.) * k ** 2))  # Upper bound quadrature sum

    V_f = FunctionSpace(mesh, 'DG', p + 1)
    u_h = Function(V)
    eta_bw = Function(V_f)
    for i, l in enumerate(np.arange(-M, N + 1, 1)):
        coefs = [Constant(np.exp(2. * l * k)), Constant(1.)]
        print('alpha = {}, Solve num. {}'.format(alpha, i))
        u_pde = solve_pde(V, f, coefs)
        eta_pde = bw_estimate(u_pde, f, coefs)
        num_solve += 1
        with XDMFFile('output/u_pde_{}.xdmf'.format(str(i).zfill(4))) as xdmf:
            xdmf.write_checkpoint(u_pde, 'u_pde_{}'.format(str(i).zfill(4)))

        u_h.vector()[:] += np.exp(alpha * l * k) * u_pde.vector()[:]
        eta_bw.vector()[:] += np.exp(alpha * l * k) * eta_pde.vector()[:]
    u_h.vector()[:] *= (2. * k * np.sin(np.pi * alpha / 2.)) / np.pi
    eta_bw.vector()[:] *= (2. * k * np.sin(np.pi * alpha / 2.)) / np.pi
    return u_h, eta_bw, num_solve

def boundary_problem(V,g):
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx
    L = inner(Constant(0.),v)*dx(domain=V.mesh())
    g_h = project(g, V)
    bc = DirichletBC(V, g_h, 'on_boundary')

    A, b = assemble_system(a, L, bcs=bc)

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

def solve_pde(V, f, coefs):
    u = TrialFunction(V)
    v = TestFunction(V)

    a = (coefs[0] * inner(grad(u), grad(v))) * dx + coefs[1] * inner(u, v) * dx

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


def bw_estimate(u_h, f, coefs, df=p + 1, dg=p, verf=False, dof_list=None):
    mesh = u_h.function_space().mesh()

    if verf:
        element_f = FiniteElement('Bubble', UnitIntervalMesh, 3) \
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
    a_e = coefs[1] * inner(e, v) * dx + coefs[0] * inner(grad(e), grad(v)) * dx
    L_e = inner(f - coefs[1] * u_h + coefs[0] * div(grad(u_h)), v) * dx + \
          inner(coefs[0] * jump(grad(u_h), -n), avg(v)) * dS

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

def customproj(f, V):
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(u, v)*dx
    L = inner(f, v)*dx
    A, b = assemble_system(a, L)

    f_h = Function(V)

    PETScOptions.set("ksp_type", "cg")
    PETScOptions.set("ksp_rtol", 1E-10)
    PETScOptions.set("ksp_monitor_true_residual")
    PETScOptions.set("pc_type", "hypre")
    PETScOptions.set("pc_hypre_type", "boomeramg")
    solver = PETScKrylovSolver()
    solver.set_from_options()
    solver.solve(A, f_h.vector(), b)
    return f_h

if __name__ == "__main__":
    main()