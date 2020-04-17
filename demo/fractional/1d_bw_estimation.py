# Solve a fractional Poisson problem on the unit square using a method from Bonito & Pasciak 2013

from dolfin import *
import dolfin as df
import numpy as np
import scipy.linalg as sp
import scipy.integrate
import sympy
import quadpy
from mpmath import *
import matplotlib.pyplot as plt
import ufl
import sys
import fenics_error_estimation
from fenics_error_estimation.estimate import estimate_python

parameters['ghost_mode'] = 'shared_facet'

p = 1           # FE polynomial order
K = 100

alpha = 0.5  # in (0, 2) beta = alpha/2. is the fractional power 

def main():
    num_dofs = []
    errors_exp = []
    errors_bw = []
    for nk in range(100, 1000, 100):
        mesh = UnitIntervalMesh(nk)
        mesh.coordinates()[:] *= 2.*np.pi
        V = FunctionSpace(mesh, 'CG', p)
        num_dofs.append(V.dim())
        xs = mesh.coordinates()[:]

        f, u_exact = analytical_data(100, mesh)
        ys_exact = [u_exact(x) for x in xs]
        u_exact_vect = [u_exact(x) for x in V.tabulate_dof_coordinates()]
        u_exact = Function(V)
        u_exact.vector()[:] = u_exact_vect

        plt.figure()
        plt.plot(xs, ys_exact, label='u_exact')

        print('Exponential quadrature...') 
        u_exp, eta_exp, num_solve_exp = exponential_quadrature(V, f, alpha, mesh)
        ys_exp = [u_exp(x) for x in xs]
        error_exp = errornorm(u_exact, u_exp, norm_type='L2')
        errors_exp.append(error_exp)
        error_bw = df.norm(eta_exp, norm_type='L2')
        errors_bw.append(error_bw)
        plt.plot(xs, ys_exp, ':', label='u_exp')

        plt.legend()
        plt.savefig('output/1d_results.pdf')
        print('Num. solve exp =', num_solve_exp)
        print('Error exp =', error_exp)
        print('Error bw =', error_bw)
        plots(num_dofs, errors_exp, errors_bw)
    return

def analytical_data(k, mesh):
    x = ufl.SpatialCoordinate(mesh)
    f_exp = [Constant((1./(sqrt(i)*np.log(i+1.))))*ufl.sin(i*x[0]) for i in range(1,k+1)] 
    f = sum(f_exp)
    u_exp = [Constant((i**(-alpha)/(sqrt(i)*np.log(i+1.))))*ufl.sin(i*x[0]) for i in range(1, k+1)]
    u_exact = sum(u_exp)
    return f, u_exact

def exponential_quadrature(V, f, alpha, mesh):
    num_solve = 0
    k = 0.2      # Size factor for the quadrature subdivision 
    
    M = np.ceil(np.pi**2/(2.*alpha*k**2))        # Lower bound quadrature sum 
    N = np.ceil(np.pi**2/(4.*(1.-alpha/2.)*k**2))   # Upper bound quadrature sum
   
    V_f = FunctionSpace(mesh, 'DG', p+1)
    u_h = Function(V)
    eta_bw = Function(V_f)
    eta_mean = Function(V_f)
    for i, l in enumerate(np.arange(-M, N+1, 1)):
        coefs = [Constant(np.exp(l*k)), Constant(1.)]
        print('Solve num. {}'.format(i))
        u_pde = solve_pde(V, f, coefs)
        eta_pde = bw_estimate(u_h, f, coefs)
        num_solve += 1
        with XDMFFile('output/u_pde_{}.xdmf'.format(str(i).zfill(4))) as xdmf:
            xdmf.write_checkpoint(u_pde, 'u_pde_{}'.format(str(i).zfill(4)))

        u_h.vector()[:] += np.exp(alpha*l*k)*u_pde.vector()[:]
        eta_bw.vector()[:] += np.exp(alpha*l*k)*eta_pde.vector()[:] 
    u_h.vector()[:] *= (2.*k*np.sin(np.pi*alpha/2.))/np.pi
    eta_bw.vector()[:] *= (2.*k*np.sin(np.pi*alpha/2.))/np.pi
    return u_h, eta_bw, num_solve

def solve_pde(V, f, coefs):
    u = TrialFunction(V)
    v = TestFunction(V)
    
    a = (coefs[0]**2*inner(grad(u), grad(v)))*dx + coefs[1]**2*inner(u,v)*dx

    L = inner(f, v)*dx

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

def bw_estimate(u_h, f, coefs, df=p+1, dg=p, verf=False, dof_list=None):
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
    a_e = coefs[1]**2*inner(e,v)*dx + coefs[0]**2*inner(grad(e), grad(v))*dx
    L_e = inner(f - coefs[1]**2*u_h + coefs[0]**2*div(grad(u_h)), v)*dx + \
        inner(coefs[0]**2*jump(grad(u_h), -n), avg(v))*dS

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

# Create custom Gaussian quadrature rule using quadpy
def custom_quadrature(l_bound, u_bound, r, weight):
    moments = quadpy.tools.integrate(lambda x: [weight(x)*x**k for k in range(2*r)], l_bound, u_bound)
    alpha, beta = quadpy.tools.chebyshev(moments)
    points, weights = quadpy.tools.scheme_from_rc(
                        np.array([sympy.N(a) for a in alpha], dtype=float),
                        np.array([sympy.N(b) for b in beta], dtype=float),
                        mode='numpy')
    return points, weights

def plots(num_dofs, errors_exp, errors_bw):
    plt.figure()
    plt.loglog(num_dofs, errors_exp, label='Error exp')
    plt.loglog(num_dofs, errors_bw, label='Error bw')
    plt.legend()
    plt.savefig('./conv.pdf')


if __name__ == "__main__":
    main()

