import pandas as pd
import numpy as np

from dolfin import *
import ufl

import fenics_error_estimation

k = 1
if k==1:
    path = 'linear'
if k==2:
    path = 'quadratic'

parameters["ghost_mode"] = "shared_facet"
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True

def main():
    K = 10
    mesh = UnitCubeMesh(10, 10, 10)

    mesh_init = mesh
    results_bw = []
    print('BW AFEM')

    for i in range(0, 8):
        result = {}
        V = FunctionSpace(mesh, 'CG', k)

        # Exact solution
        x = ufl.SpatialCoordinate(mesh)

        u_exact = ufl.sin(2.*np.pi*x[0])*ufl.sin(2.*np.pi*x[1])*ufl.sin(2.*np.pi*x[2]) 
        f = 3.*(2.*np.pi)**2*u_exact

        print('V dim = {}'.format(V.dim()))
        u_h, err = solve(V, f, u_exact)
        print('Exact error = {}'.format(err))
        result['exact_error'] = err

        print('Estimating...')
        eta_h = estimate(u_h, f)
        result['error_bw'] = np.sqrt(eta_h.vector().sum())
        print('BW = {}'.format(np.sqrt(eta_h.vector().sum())))
        result['hmin'] = mesh.hmin()
        result['hmax'] = mesh.hmax()
        result['num_dofs'] = V.dim()

        print('Estimating (res)...')
        eta_res = residual_estimate(u_h, f)
        result['error_res'] = np.sqrt(eta_res.vector().sum())
        print('Res = {}'.format(np.sqrt(eta_res.vector().sum())))

        print('Marking...')
        markers = fenics_error_estimation.dorfler_parallel(eta_h, 0.3)
        print('Refining...')
        mesh = refine(mesh, markers, redistribute=True)

        with XDMFFile('output/{}/bank-weiser/mesh_{}.xdmf'.format(path, str(i).zfill(4))) as f:
            f.write(mesh)

        with XDMFFile('output/{}/bank-weiser/u_{}.xdmf'.format(path, str(i).zfill(4))) as f:
            f.write(u_h)

        with XDMFFile('output/{}/bank-weiser/eta_{}.xdmf'.format(path, str(i).zfill(4))) as f:
            f.write(eta_h)

        results_bw.append(result)

    if (MPI.comm_world.rank == 0):
        df = pd.DataFrame(results_bw)
        df.to_pickle('output/{}/bank-weiser/results.pkl'.format(path))
        print(df)

    mesh = mesh_init
    results_res = []
    print('Residual AFEM')
    for i in range(0,8):
        result = {}
        V = FunctionSpace(mesh, 'CG', k)
        
        # Exact solution
        x = ufl.SpatialCoordinate(mesh)

        u_exact = ufl.sin(2.*np.pi*x[0])*ufl.sin(2.*np.pi*x[1])*ufl.sin(2.*np.pi*x[2]) 
        f = 3.*(2.*np.pi)**2*u_exact

        print('V dim = {}'.format(V.dim()))
        u_h, err = solve(V, f, u_exact)
        print('Exact error = {}'.format(err))
        result['exact_error'] = err

        print('Estimating...')
        eta_h = residual_estimate(u_h, f)
        result['error_res'] = np.sqrt(eta_h.vector().sum())
        print('res = {}'.format(np.sqrt(eta_h.vector().sum())))
        result['hmin'] = mesh.hmin()
        result['hmax'] = mesh.hmax()
        result['num_dofs'] = V.dim()

        print('Marking...')
        markers = fenics_error_estimation.dorfler_parallel(eta_h, 0.3)
        print('Refining...')
        mesh = refine(mesh, markers, redistribute=True)

        with XDMFFile('output/{}/residual/mesh_{}.xdmf'.format(path, str(i).zfill(4))) as f:
            f.write(mesh)

        with XDMFFile('output/{}/residual/u_{}.xdmf'.format(path, str(i).zfill(4))) as f:
            f.write(u_h)

        with XDMFFile('output/{}/residual/eta_{}.xdmf'.format(path, str(i).zfill(4))) as f:
            f.write(eta_h)

        results_res.append(result)

    if (MPI.comm_world.rank == 0):
        df = pd.DataFrame(results_res)
        df.to_pickle('output/{}/residual/results.pkl'.format(path))
        print(df)
        

def solve(V, f, u_exact):
    mesh = V.mesh()
    u = TrialFunction(V)
    v = TestFunction(V)

    a = inner(grad(u), grad(v))*dx

    bcs = DirichletBC(V, Constant(0.), 'on_boundary')

    L = inner(f, v)*dx

    A, b = assemble_system(a, L, bcs=bcs)

    u_h = Function(V)
    print('Test')
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
    err = sqrt(assemble(inner(ufl.grad(err), ufl.grad(err))*dx))

    return u_h, err

def estimate(u_h, f):
    mesh = u_h.function_space().mesh()

    element_f = FiniteElement('DG', tetrahedron, k + 1)
    element_g = FiniteElement('DG', tetrahedron, k)
    V_f = FunctionSpace(mesh, element_f)

    N = fenics_error_estimation.create_interpolation(element_f, element_g)

    e = TrialFunction(V_f)
    v = TestFunction(V_f)

    bcs = DirichletBC(V_f, Constant(0.), 'on_boundary', 'geometric')

    n = FacetNormal(mesh)
    a_e = inner(grad(e), grad(v))*dx
    L_e = inner(f + div(grad(u_h)), v)*dx + \
          inner(jump(grad(u_h), -n), avg(v))*dS

    e_h = fenics_error_estimation.estimate(a_e, L_e, N, bcs)

    V_e = FunctionSpace(mesh, 'DG', 0)
    v = TestFunction(V_e)

    eta_h = Function(V_e)
    eta = assemble(inner(inner(grad(e_h), grad(e_h)), v)*dx)
    eta_h.vector()[:] = eta

    return eta_h

def residual_estimate(u_h, f):
    mesh = u_h.function_space().mesh()

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
