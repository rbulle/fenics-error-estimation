# Mixed-formulation implementation of linear elasticity from Khan-Powell-Silvester 2018
import pandas as pd
import numpy as np

from dolfin import *
import ufl

parameters["ghost_mode"] = "shared_facet"
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True

def main():
    K = 10
    mesh = UnitSquareMesh(K,K)

    X_el = VectorElement('CG', triangle, 2)
    M_el = FiniteElement('CG', triangle, 1)

    V_el = MixedElement([X_el, M_el])

    V = FunctionSpace(mesh, V_el)

    mu = 100. # Lamé coef
    nu = .4   # Poisson ratio
    lmbda = 2.*mu*nu/(1.-2.*nu) # Lamé coef

    f = Expression(('-2.*mu*pow(pi,2)*cos(pi*x[1])*sin(pi*x[1])*(2.*cos(2.*pi*x[0]) - 1.)', '2.*mu*pow(pi,3)*cos(pi*x[0])*sin(pi*x[0])*(2.*cos(2.*pi*x[1]) -1.)'),
                   mu=mu, degree = 4)

    u_exact = Expression(('pi*cos(pi*x[1])*pow(sin(pi*x[0]), 2)*sin(pi*x[1])', '-pi*cos(pi*x[0])*pow(sin(pi*x[1]), 2)*sin(pi*x[0])'),
                         mu = mu, degree = 4)

    p_exact = Constant(0.)

    eps = lambda u: .5*sym(grad(u))
    
    (u, p) = TrialFunctions(V)
    (v, q) = TestFunctions(V)

    a = 2.*mu*inner(eps(u),eps(v))*dx
    b_1 = - inner(p, div(v))*dx
    b_2 = - inner(q, div(u))*dx
    c = (1./lmbda)*inner(p,q)*dx

    B = a + b_1 + b_2 - c
    L = inner(f, v)*dx

    bcs = DirichletBC(V.sub(0), Constant((0., 0.)), 'on_boundary')

    A, b = assemble_system(B, L, bcs=bcs)

    w_h = Function(V)

    '''
    PETScOptions.set("ksp_type", "cg")
    PETScOptions.set("ksp_rtol", 1E-10)
    PETScOptions.set("ksp_monitor_true_residual")
    PETScOptions.set("pc_type", "hypre")
    PETScOptions.set("pc_hypre_type", "boomeramg")
    solver = PETScKrylovSolver()
    solver.set_from_options()
    solver.solve(A, w_h.vector(), b)
    '''
    solve(A, w_h.vector(), b)

    u_h = w_h.sub(0)
    p_h = w_h.sub(1)
    return

     



if __name__ == "__main__":
    main()

