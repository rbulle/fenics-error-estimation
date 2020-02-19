import itertools

import numpy as np
import scipy.linalg as sp
import pytest

from dolfin import *

from fenics_error_estimation import create_interpolation

def non_reduced_operator(V_f, V_g):
    V_f_dim = V_f.dim()
    V_g_dim = V_g.dim()

    assert(V_f_dim > V_g_dim)

    w = Function(V_f)

    # Get interpolation matrices from fine space to coarse one and conversely
    G_1 = PETScDMCollection.create_transfer_matrix(V_f, V_g).array()
    G_2 = PETScDMCollection.create_transfer_matrix(V_g, V_f).array()
    # Using "Function" prior to create_transfer_matrix, initialises PETSc for
    # unknown reason...

    # Create a square matrix for interpolation from fine space to coarse one
    # with coarse space seen as a subspace of the fine one
    G = G_2@G_1
    G[np.isclose(G, 0.0)] = 0.0

    # Create square matrix of the interpolation on supplementary space of
    # coarse space into fine space
    N = np.eye(V_f.dim()) - G

    # Change of basis to reduce N as a diagonal with only ones and zeros
    eigs, P = np.linalg.eig(N)
    eigs = np.real(eigs)
    P = np.real(P)
    return N, P

@pytest.mark.parametrize('k,cell_type', itertools.product([1, 2, 3, 4], [interval, triangle, tetrahedron]))
def test_vectorial_interpolation_operator(k, cell_type):
    E_f_vect = VectorElement('DG', cell_type, k+1)
    E_g_vect = VectorElement('DG', cell_type, k)
    
    E_f = FiniteElement('DG', cell_type, k+1)
    E_g = FiniteElement('DG', cell_type, k)

    if cell_type == interval:
        mesh = UnitIntervalMesh(MPI.comm_self, 1)
    elif cell_type == triangle:
        mesh = Mesh(MPI.comm_self)
        editor = MeshEditor()
        editor.open(mesh, "triangle", 2, 2)

        editor.init_vertices(3)
        editor.init_cells(1)

        editor.add_vertex(0, np.array([0.0, 0.0]))
        editor.add_vertex(1, np.array([1.0, 0.0]))
        editor.add_vertex(2, np.array([0.0, 1.0]))
        editor.add_cell(0, np.array([0, 1, 2], dtype=np.uintp))

        editor.close()
    elif cell_type == tetrahedron:
        mesh = Mesh(MPI.comm_self)
        editor = MeshEditor()
        editor.open(mesh, "tetrahedron", 3, 3)

        editor.init_vertices(4)
        editor.init_cells(1)

        editor.add_vertex(0, np.array([0.0, 0.0, 0.0]))
        editor.add_vertex(1, np.array([1.0, 0.0, 0.0]))
        editor.add_vertex(2, np.array([0.0, 1.0, 0.0]))
        editor.add_vertex(3, np.array([0.0, 0.0, 1.0]))
        editor.add_cell(0, np.array([0, 1, 2, 3], dtype=np.uintp))

        editor.close()

    V_f_vect = FunctionSpace(mesh, E_f_vect)
    V_g_vect = FunctionSpace(mesh, E_g_vect)

    N_vect, P_vect = non_reduced_operator(V_f_vect, V_g_vect)
    P_vect = np.asarray(P_vect)

    V_f = FunctionSpace(mesh, E_f)
    V_g = FunctionSpace(mesh, E_g)

    N, P = non_reduced_operator(V_f, V_g)
    P = np.asarray(P)

    if cell_type == interval:
        N_bloc = N
    elif cell_type == triangle:
        N_bloc = sp.block_diag(N, N)
    elif cell_type == tetrahedron:
        N_bloc = sp.block_diag(N, N, N)

    eigs_vect, _ = np.linalg.eig(N_vect)
    eigs_bloc, _ = np.linalg.eig(N_bloc)

    assert  np.array_equal(np.sort(np.round(np.real(eigs_vect), 10)), np.sort(np.round(np.real(eigs_bloc), 10)))
