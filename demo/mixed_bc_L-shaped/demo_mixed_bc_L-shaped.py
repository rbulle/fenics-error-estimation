## Copyright 2019-2020, Jack S. Hale, Raphaël Bulle
## SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
import pandas as pd

from dolfin import *
import ufl

import fenics_error_estimation

parameters['ghost_mode'] = 'shared_facet'

k = 1


def main():
    mesh = Mesh()
    try:
        with XDMFFile(MPI.comm_world, './mesh/mesh.xdmf') as f:
            f.read(mesh)
    except:
        print(
            'Generate the mesh using `python3 generate_mesh.py` before running this script.')
        exit()

    results = []
    for i in range(0, 16):
        result = {}

        u_exact, f, g = pbm_data(mesh)
        V = FunctionSpace(mesh, "CG", k)

        class BoundaryN(SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and (x[0] < 0.) and (near(x[1], 0., DOLFIN_EPS))

        boundaryN = BoundaryN()

        boundary_marker = MeshFunction('size_t', mesh, 1)
        boundaryN.mark(boundary_marker, 1)

        dN = Measure('ds', domain=mesh, subdomain_data=boundary_marker)

        u_h = solve(V, u_exact, f, g, dN)
        with XDMFFile("output/u_h_{}.xdmf".format(str(i).zfill(4))) as xdmf:
            xdmf.write(u_h)

        u_exact_V = project(u_exact, u_h.function_space())
        u_exact_V.rename("u_exact_V", "u_exact_V")
        with XDMFFile("output/u_exact_{}.xdmf".format(str(i).zfill(4))) as xdmf:
            xdmf.write(u_exact_V)

        eta_bw = bw_estimate(u_h, f, g, dN)
        with XDMFFile("output/eta_bw_{}.xdmf".format(str(i).zfill(4))) as xdmf:
            xdmf.write_checkpoint(eta_bw, "eta_bw")

        result["error_bw"] = np.sqrt(eta_bw.vector().sum())

        eta_ver = bw_estimate(u_h, f, g, dN, verf=True)
        with XDMFFile("output/eta_ver_{}.xdmf".format(str(i).zfill(4))) as xdmf:
            xdmf.write_checkpoint(eta_ver, "eta_ver")

        result["error_ver"] = np.sqrt(eta_ver.vector().sum())

        if k == 1:
            eta_zz = zz_estimate(u_h, f)
            with XDMFFile("output/eta_zz_{}.xdmf".format(str(i).zfill(4))) as xdmf:
                xdmf.write_checkpoint(eta_zz, "eta_zz")

            result["error_zz"] = np.sqrt(eta_zz.vector().sum())

        eta_res = residual_estimate(u_h, f, g, dN)
        with XDMFFile("output/eta_res_{}.xdmf".format(str(i).zfill(4))) as xdmf:
            xdmf.write_checkpoint(eta_res, "eta_res")

        result["error_res"] = np.sqrt(eta_res.vector().sum())

        V_e = eta_bw.function_space()
        eta_exact = Function(V_e, name="eta_exact")
        v = TestFunction(V_e)
        eta_exact.vector()[:] = assemble(inner(inner(grad(u_h - u_exact), grad(u_h - u_exact)), v) * dx(mesh))
        result["exact_error"] = np.sqrt(eta_exact.vector().sum())
        with XDMFFile("output/eta_exact_{}.xdmf".format(str(i).zfill(4))) as xdmf:
            xdmf.write(eta_exact)

        result["hmin"] = mesh.hmin()
        result["hmax"] = mesh.hmax()
        result["num_dofs"] = V.dim()

        markers = fenics_error_estimation.dorfler(eta_res, 0.5)
        mesh = refine(mesh, markers, redistribute=True)

        with XDMFFile("output/mesh_{}.xdmf".format(str(i).zfill(4))) as xdmf:
            xdmf.write(mesh)

        results.append(result)

    if (MPI.comm_world.rank == 0):
        df = pd.DataFrame(results)
        df.to_pickle("output/results.pkl")
        print(df)


def solve(V, u_exact, f, g, dN):
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v)) * dx

    class BoundaryD(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and not ((x[0] < 0.) and (near(x[1], 0., DOLFIN_EPS)))

    boundaryD = BoundaryD()

    L = inner(f, v) * dx + inner(g, v) * dN(1)

    bcs = DirichletBC(V, u_exact, boundaryD)

    A, b = assemble_system(a, L, bcs=bcs)

    u_h = Function(V, name='u_h')
    PETScOptions.set("ksp_type", "cg")
    PETScOptions.set("ksp_rtol", 1E-10)
    PETScOptions.set("ksp_monitor_true_residual")
    PETScOptions.set("pc_type", "hypre")
    PETScOptions.set("pc_hypre_type", "boomeramg")
    solver = PETScKrylovSolver()
    solver.set_from_options()
    solver.solve(A, u_h.vector(), b)

    return u_h

def bw_estimate(u_h, f, g, dN, df=k + 1, dg=k, verf=False, dof_list=None):
    mesh = u_h.function_space().mesh()

    if verf:
        element_f = FiniteElement('Bubble', triangle, 3) \
                    + FiniteElement('DG', triangle, 2)
        element_g = FiniteElement('DG', triangle, 1)
    else:
        element_f = FiniteElement("DG", triangle, df)
        element_g = FiniteElement("DG", triangle, dg)

    N = fenics_error_estimation.create_interpolation(element_f, element_g, dof_list)

    V_f = FunctionSpace(mesh, element_f)

    e = TrialFunction(V_f)
    v = TestFunction(V_f)

    class BoundaryD(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and not ((x[0] < 0.) and (near(x[1], 0., DOLFIN_EPS)))

    boundaryD = BoundaryD()

    bcs = DirichletBC(V_f, Constant(0.0), boundaryD, "geometric")

    n = FacetNormal(mesh)
    a_e = inner(grad(e), grad(v)) * dx
    L_e = inner(f + div(grad(u_h)), v) * dx\
          + inner(jump(grad(u_h), -n), avg(v)) * dS\
          + inner(g - inner(grad(u_h), n), v) * dN(1)

    e_h = fenics_error_estimation.estimate(a_e, L_e, N, bcs=bcs)

    # Computation of local error indicator
    V_e = FunctionSpace(mesh, "DG", 0)
    v = TestFunction(V_e)

    eta_h = Function(V_e, name="eta_h")
    eta = assemble(inner(inner(grad(e_h), grad(e_h)), v) * dx)
    eta_h.vector()[:] = eta
    return eta_h


def zz_estimate(u_h, f):
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

    A = assemble(inner(w_h, v_h) * dx,
                 form_compiler_parameters={'quadrature_rule': 'vertex', 'representation': 'quadrature'})
    b = assemble(inner(grad(u_h), v_h) * dx)

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
    eta = assemble(inner(inner(disc_zz, disc_zz), v) * dx)
    eta_h.vector()[:] = eta
    return eta_h


def residual_estimate(u_h, f, g, dN):
    mesh = u_h.function_space().mesh()

    n = FacetNormal(mesh)
    h_T = CellDiameter(mesh)
    h_E = FacetArea(mesh)

    r = f + div(grad(u_h))
    J_I = jump(grad(u_h), -n)
    J_N = g - inner(grad(u_h), n)

    V_e = FunctionSpace(mesh, 'DG', 0)
    v_e = TestFunction(V_e)

    R = h_T ** 2 * inner(inner(r, r), v_e) * dx\
        + avg(h_E) * inner(inner(J_I, J_I), avg(v_e)) * dS\
        + h_E * inner(inner(J_N, J_N), v_e) * dN(1)

    # Computation of local error indicator
    V_e = FunctionSpace(mesh, "DG", 0)

    eta_h = Function(V_e)
    eta = assemble(R)[:]
    eta_h.vector()[:] = eta
    return eta_h


def pbm_data(mesh):
    # Exact solution
    x = ufl.SpatialCoordinate(mesh)

    r = ufl.sqrt(x[0] ** 2 + x[1] ** 2)
    theta = ufl.mathfunctions.Atan2(x[1], x[0])

    # Exact solution
    u_exact = r ** (1. / 3.) * ufl.sin((1. / 3.) * (theta + ufl.pi / 2.))

    # Data
    f = Constant(0.)
    g = Constant(0.)
    return u_exact, f, g

if __name__ == "__main__":
    main()

def test():
    main()