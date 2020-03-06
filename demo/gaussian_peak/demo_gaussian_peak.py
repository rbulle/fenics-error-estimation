## Copyright 2019-2020, Jack S. Hale, Raphaël Bulle
## SPDX-License-Identifier: LGPL-3.0-or-later
import os

import numpy as np
import pandas as pd

from dolfin import *
import ufl

import mpi4py.MPI

import fenics_error_estimation

parameters['ghost_mode'] = 'shared_facet'

k = 1

comm = MPI.comm_world
mesh = Mesh(comm)
try:
    with XDMFFile(comm, os.path.join(current_dir, 'mesh.xdmf')) as f:
        f.read(mesh)
except:
    print(
        'Generate the mesh using `python3 generate_mesh.py` before running this script.')
    exit()

# Exact solution
x = ufl.SpatialCoordinate(mesh)

# Exact solution
alpha = 1000
m = [0.25, 0.5]
u_exact = ufl.exp(-alpha*((x[0]-m[0])**2+(x[1]-m[1])**2))
params['exact solution'] = u_exact

# Data
f = - ufl.div(ufl.grad(u_exact))
params['force data'] = f

def main():
    results = []
    for i in range(0, 15):
        result = {}

        V = FunctionSpace(mesh, "CG", k)
        u_h = solve(V)
        with XDMFFile("output/u_h_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write(u_h)
        result["error"] = errornorm(u_exact, u_h, "H10")

        u_exact_V = interpolate(u_exact, u_h.function_space())
        u_exact_V.rename("u_exact_V", "u_exact_V")
        with XDMFFile("output/u_exact_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write(u_exact_V)

        eta_bw = bw_estimate(u_h)
        with XDMFFile("output/eta_bw_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write_checkpoint(eta_bw, "eta_bw")

        result["error_bw"] = np.sqrt(eta_bw.vector().sum())

        eta_zz = zz_estimate(u_h)
        with XDMFFile("output/eta_zz_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write_checkpoint(eta_zz, "eta_zz")

        result["error_zz"] = np.sqrt(eta_zz.vector().sum())
        
        eta_res = residual_estimate(u_h)
        with XDMFFile("output/eta_res_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write_checkpoint(eta_res, "eta_res")

        result["error_res"] = np.sqrt(eta_res.vector().sum())

        V_e = eta_bw.function_space()
        eta_exact = Function(V_e, name="eta_exact")
        v = TestFunction(V_e)
        eta_exact.vector()[:] = assemble(inner(inner(grad(u_h - u_exact_V), grad(u_h - u_exact_V)), v)*dx(mesh))
        result["error_exact"] = np.sqrt(eta_exact.vector().sum())
        with XDMFFile("output/eta_exact_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write(eta_exact)

        result["hmin"] = comm.reduce(mesh.hmin(), op=mpi4py.MPI.MIN, root=0)
        result["hmax"] = comm.reduce(mesh.hmax(), op=mpi4py.MPI.MAX, root=0)
        result["num_dofs"] = V.dim()

        markers = fenics_error_estimation.dorfler(eta_bw, 0.5)
        mesh = refine(mesh, markers, redistribute=True)

        with XDMFFile("output/mesh_{}.xdmf".format(str(i).zfill(4))) as f:
            f.write(mesh)

        results.append(result)

    if (MPI.comm_world.rank == 0):
        df = pd.DataFrame(results)
        df.to_pickle("output/results.pkl")
        print(df)

def solve(V):
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx
    L = inner(f, v)*dx

    def all_boundary(x, on_boundary):
        return on_boundary

    bcs = DirichletBC(V, u_exact ,all_boundary)

    A, b = assemble_system(a, L, bcs=bcs)

    u_h = Funtion(V, name='u_h')
    solve = PETScLUSolver('mumps')
    solver.solve(A, u_h.vector(), b)

    return u_h

def bw_estimate(u_h):
    mesh = u_h.function_space().mesh()

    element_f = FiniteElement("DG", triangle, k + 2)
    element_g = FiniteElement("DG", triangle, k)

    N = fenics_error_estimation.create_interpolation(element_f, element_g)

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

def ZZ_estimate(u_h):
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

def residual_estimate(u_h):
    mesh = u_h.function_space().mesh()

    n = FacetNormal(mesh)
    h_T = CellDiameter(mesh)
    h_E = FacetArea(mesh)

    r = f + div(grad(u_h))
    J_h = jump(grad(u_h), -n)

    V_e = FunctionSpace(mesh, 'DG', 0)
    v_e = TestFunction(V_e)

    R = h_T**2*inner(inner(r,r),v_e)*dx + avg(h_E)*inner(inner(J_h, J_h), avg(v_e))*dS

    eta_h = assemble(R)[:]
    return eta_h

if __name__ == "__main__":
    main()


def test():
    main()