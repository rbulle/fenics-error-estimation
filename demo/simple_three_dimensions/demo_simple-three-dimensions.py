import pandas as pd
import numpy as np

from dolfin import *
import ufl

import bank_weiser

k = 1
parameters["ghost_mode"] = "shared_facet"
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters['form_compiler']['representation'] = 'uflacs'

def main():
    ns = 8 
    mesh = UnitCubeMesh(ns, ns, ns)

    results = []
    for i in range(0, 4):
        result = {}
        V = FunctionSpace(mesh, "CG", k)
        u_h = solve(V)
        result["hmin"] = mesh.hmin()
        result["hmax"] = mesh.hmax()
        result["num_dofs"] = V.dim()
        g = Expression("sin(pi*x[0])*sin(pi*x[1])*sin(pi*x[2])",
                       degree=6)
        result["error"] = errornorm(g, u_h, norm_type='h10', degree_rise=3)

        eta_h_bw = estimate(u_h)
        eta_h_res = residual_estimate(u_h)
        result["error_bw"] = np.sqrt(eta_h_bw.vector().sum())
        result["error_res"] = np.sqrt(eta_h_res.vector().sum())

        mesh = refine(mesh)

        with XDMFFile("output/mesh_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write(mesh)

        with XDMFFile("output/u_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write(u_h)

        results.append(result)

    if (MPI.comm_world.rank == 0):
        df = pd.DataFrame(results)
        df.to_pickle("output/results.pkl")
        print(df)

def solve(V):
    u = TrialFunction(V)
    v = TestFunction(V)

    g = Expression('sin(pi*x[0])*sin(pi*x[1])*sin(pi*x[2])',
                   degree=3)
    f = Expression('3*pi*pi*sin(pi*x[0])*sin(pi*x[1])*sin(pi*x[2])',
                   degree=3)

    a = inner(grad(u), grad(v))*dx
    L = inner(f, v)*dx

    bcs = DirichletBC(V, g, "on_boundary")

    A, b = assemble_system(a, L, bcs=bcs)

    u_h = Function(V)

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

    return u_h


def estimate(u_h):
    mesh = u_h.function_space().mesh()

    element_f = FiniteElement("DG", tetrahedron, k + 1)
    element_g = FiniteElement("DG", tetrahedron, k)
    V_f = FunctionSpace(mesh, element_f)

    N = bank_weiser.create_interpolation(element_f, element_g)

    e = TrialFunction(V_f)
    v = TestFunction(V_f)

    g = Expression('sin(pi*x[0])*sin(pi*x[1])*sin(pi*x[2])',
                   element=V_f.ufl_element())
    f = Expression('3*pi*pi*sin(pi*x[0])*sin(pi*x[1])*sin(pi*x[2])',
                   element=V_f.ufl_element())

    bcs = DirichletBC(V_f, g, "on_boundary", "geometric")

    n = FacetNormal(mesh)
    a_e = inner(grad(e), grad(v))*dx
    L_e = inner(f + div(grad(u_h)), v)*dx + \
          inner(jump(grad(u_h), -n), avg(v))*dS

    e_h = bank_weiser.estimate(a_e, L_e, N, bcs)
    error = norm(e_h, "H10")

    # Computation of local error indicator
    V_e = FunctionSpace(mesh, "DG", 0)
    v = TestFunction(V_e)

    eta_h = Function(V_e)
    eta = assemble(inner(inner(grad(e_h), grad(e_h)), v)*dx)
    eta_h.vector()[:] = eta

    return eta_h

def residual_estimate(u_h):
    mesh = u_h.function_space().mesh()

    f = Expression('3*pi*pi*sin(pi*x[0])*sin(pi*x[1])*sin(pi*x[2])', degree=3)
    
    n = FacetNormal(mesh)
    r = f + div(grad(u_h))
    J_h = jump(grad(u_h), -n)

    V = FunctionSpace(mesh, "DG", 0)
    v = TestFunction(V)
    h = CellDiameter(mesh)

    eta_h = Function(V)
    eta = assemble(h**2*r**2*v*dx + avg(h)*J_h**2*avg(v)*dS)
    eta_h.vector()[:] = eta

    return eta_h

if __name__ == "__main__":
    main()

def test():
    pass