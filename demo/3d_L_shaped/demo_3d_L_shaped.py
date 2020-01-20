import pandas as pd
import numpy as np

from dolfin import *
import ufl

import bank_weiser

k = 1
parameters["ghost_mode"] = "shared_facet"
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True

def main():
    mesh = Mesh()
    try:
        with XDMFFile(MPI.comm_world, 'mesh.xdmf') as f:
            f.read(mesh)
    except:
        print(
            "Generate the mesh using `python3 generate_mesh.py` before running this script.")
        exit()

    results = []
    for i in range(0, 16):
        result = {}
        V = FunctionSpace(mesh, 'CG', k)
        print('V dim = {}'.format(V.dim()))
        u_h, err = solve(V)
        print('Exact error = {}'.format(err))
        result['exact_error'] = err

        print('Estimating (BW)...')
        eta_h = estimate(u_h)
        result['error_bw'] = np.sqrt(eta_h.vector().sum())
        print('BW = {}'.format(np.sqrt(eta_h.vector().sum())))
        result['hmin'] = mesh.hmin()
        result['hmax'] = mesh.hmax()
        result['num_dofs'] = V.dim()

        print('Estimating (res)...')
        eta_res = residual_estimate(u_h)
        result['error_res'] = np.sqrt(eta_res.vector().sum())
        print('Res = {}'.format(np.sqrt(eta_res.vector().sum())))


        print('Marking...')
        markers = bank_weiser.maximum(eta_h, 0.2)
        print('Refining...')
        mesh = refine(mesh, markers, redistribute=True)

        with XDMFFile('output/mesh_{}.xdmf'.format(str(i).zfill(4))) as f:
            f.write(mesh)

        with XDMFFile('output/u_{}.xdmf'.format(str(i).zfill(4))) as f:
            f.write(u_h)

        with XDMFFile('output/eta_{}.xdmf'.format(str(i).zfill(4))) as f:
            f.write(eta_h)

        results.append(result)

    if (MPI.comm_world.rank == 0):
        df = pd.DataFrame(results)
        df.to_pickle('output/results.pkl')
        print(df)

def solve(V):
    mesh = V.mesh()
    u = TrialFunction(V)
    v = TestFunction(V)

    f = Constant(0.)

    a = inner(grad(u), grad(v))*dx
    L = inner(f, v)*dx

    # Exact solution
    x = ufl.SpatialCoordinate(mesh)

    r, theta = cartesian2polar(x)

    u_2d = r**(2./3.)*ufl.sin((2./3.)*(theta+ufl.pi/2.))

    u_exact = u_2d

    bcs = DirichletBC(V, u_2d, 'on_boundary')

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

    # Compute the exact error
    err = u_exact - u_h
    V_e = FunctionSpace(mesh, 'DG', 0)
    v_T = CellVolume(mesh)
    err = sqrt(assemble(inner(grad(err), grad(err))*dx))

    return u_h, err

def estimate(u_h):
    mesh = u_h.function_space().mesh()

    element_f = FiniteElement('DG', tetrahedron, k + 1)
    element_g = FiniteElement('DG', tetrahedron, k)
    V_f = FunctionSpace(mesh, element_f)

    N = bank_weiser.create_interpolation(element_f, element_g)

    e = TrialFunction(V_f)
    v = TestFunction(V_f)

    f = Constant(0.)

    bcs = DirichletBC(V_f, Constant(0.), 'on_boundary', 'geometric')

    n = FacetNormal(mesh)
    a_e = inner(grad(e), grad(v))*dx
    L_e = inner(f + div(grad(u_h)), v)*dx + \
          inner(jump(grad(u_h), -n), avg(v))*dS

    e_h = bank_weiser.estimate(a_e, L_e, N, bcs)

    V_e = FunctionSpace(mesh, 'DG', 0)
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

def cartesian2polar(x):
    r = ufl.sqrt(x[0]**2 + x[1]**2)
    theta = ufl.mathfunctions.Atan2(x[1], x[0])
    return r, theta

if __name__ == "__main__":
    main()

def test():
    pass
