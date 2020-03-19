import itertools

import numpy as np
import pytest

from dolfin import *

from fenics_error_estimation import create_interpolation

@pytest.mark.parametrize('k,cell_type', itertools.product([1, 2, 3], [interval, triangle, tetrahedron]))
def test_interpolation_operator(k, cell_type):
    V_f = FiniteElement("DG", cell_type, k + 1)
    V_g = FiniteElement("DG", cell_type, k + 1)

    gdim = V_f.cell().geometric_dimension()

    if gdim == 1:
        df = k + 2
    elif gdim == 2:
        df = sum(range(k+2))
    elif gdim == 3:
        df = sum([sum(range(r+1)) for r in range(k+3)])

    combinations = []
    for r in range(1,df):
        combinations.append(itertools.combinations(range(df), r))

    for comb in combinations:
        for l in comb:
            # Various assertions build into function
            N = create_interpolation(V_f, V_g, dof_list=l) 


