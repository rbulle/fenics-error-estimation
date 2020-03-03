import numpy as np
import scipy.linalg as sp

from dolfin import *

from fenics_error_estimation import create_interpolation, estimate
from fenics_error_estimation.estimate import estimate_python

from dolfin.fem.assembling import _create_dolfin_form
import fenics_error_estimation.cpp as cpp

def main():
    mesh = UnitTriangleMesh.create()

    E_vec_f = VectorElement('DG', triangle, 2)
    E_vec_g = VectorElement('DG', triangle, 1)
    E_scal_f = FiniteElement('DG', triangle, 2)
    E_scal_g = FiniteElement('DG', triangle, 1)

    N_vec = create_interpolation(
        E_vec_f, E_vec_g)
    N_scal = create_interpolation(
        E_scal_f, E_scal_g)

    V_f = FunctionSpace(mesh, E_vec_f)
    V = FunctionSpace(mesh, E_scal_f)

    u = TrialFunction(V_f)
    v = TestFunction(V_f)

    a = inner(grad(u), grad(v))*dx

    E_W = VectorElement("DG", triangle, 4)
    W = FunctionSpace(mesh, E_W)
    for i in range(W.dim()):
        f = Function(W)
        f.vector()[i] = 1.
        f_0 = f.sub(0)
        f_1 = f.sub(1)
        L = inner(f, v)*dx

        e = estimate_python(a, L, N_vec)

        dofs_0 = V_f.sub(0).dofmap().dofs()
        dofs_1 = V_f.sub(1).dofmap().dofs()

        u_0 = TrialFunction(V)
        v_0 = TestFunction(V)

        a_0 = inner(grad(u_0), grad(v_0))*dx
        L_0 = inner(f_0, v_0)*dx

        e_0 = estimate(a_0, L_0, N_scal)

        u_1 = TrialFunction(V)
        v_1 = TestFunction(V)

        a_1 = inner(grad(u_1), grad(v_1))*dx
        L_1 = inner(f_1, v_1)*dx

        e_1 = estimate(a_1, L_1, N_scal)

        e_vec = Function(V_f)
        assign(e_vec.sub(0), e_0)
        assign(e_vec.sub(1), e_1)
        with XDMFFile('output/e_{}.xdmf'.format(i)) as xdmf:
            xdmf.write_checkpoint(e, 'e')
        with XDMFFile('output/e_vec_{}.xdmf'.format(i)) as xdmf:
            xdmf.write_checkpoint(e_vec, 'e_vec')

        '''
        for i in range(V.dim()):
            v = Function(V_f)
            v.vector()[dofs_0[i]] = 1.
            v_scal = Function(V)
            v_scal.vector()[i] = 1.
            assert np.isclose(assemble(inner(v.sub(0)-v_scal, v.sub(0)-v_scal)*dx), 0.)
        '''
        '''
        for j in range(V_f.dim()):
            v = Function(V_f)
            v.vector()[j] = 1.
            with XDMFFile('output/v_{}.xdmf'.format(j)) as xdmf:
                xdmf.write_checkpoint(v, 'v_{}'.format(j))
        '''
        print('int grad(e_vec -e) =', assemble(inner(grad(e_vec-e), grad(e_vec-e))*dx))
        '''
        assert np.isclose(assemble(inner(grad(e_vec-e), grad(e_vec-e))*dx), 0., rtol=0., atol=1.e-3)
        '''
        print(i)
    return

if __name__ == "__main__":
    main()
