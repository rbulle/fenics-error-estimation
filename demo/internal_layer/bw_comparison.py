import sys
sys.path.append('../')
from dolfin import *
import pygmsh as pg
import meshio
from bank_weiser import estimate, bw_estimation
from bw_adaptive_refinement import bw_adaptive
from residual_adaptive_refinement import res_adaptive
from zz_adaptive_refinement import zz_adaptive
from uniform_refinement import refinement
from results import *
import matplotlib.pyplot as plt
from mpltools import annotation
from marker_pos import loglog_marker_pos
import scipy.special as sp

def ref_triangle(degrees, num, path, dofs=None):
    dofs = np.asarray(dofs)
    W_g = FunctionSpace(UnitTriangleMesh.create(), 'CG', degrees[0])

    W_g_dofs = W_g.tabulate_dof_coordinates()

    W_f = FunctionSpace(UnitTriangleMesh.create(), 'CG', degrees[1])

    W_f_dofs = W_f.tabulate_dof_coordinates()

    plt.figure()
    # plt.title('Dofs position on reference cell for BW estimator {}.'.format(num))
    plt.axis('off')
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.gca().set_aspect('equal', adjustable='box')
    for i in range(len(W_f_dofs[:,0])):
        plt.scatter(W_f_dofs[i,0], W_f_dofs[i,1], color='r', marker='x', \
                    zorder=100, s=100)

    for i in range(len(W_g_dofs[:,0])):
        if i in dofs:
            plt.scatter(W_g_dofs[i,0], W_g_dofs[i, 1], color='b', marker='o', \
                        zorder=200, s=100)
        else:
            plt.scatter(W_g_dofs[i,0], W_g_dofs[i, 1], color='b', marker='o', \
                        zorder=200, s=100, facecolor='none')
        x = W_g_dofs[i, 0]
        y = W_g_dofs[i, 1]
        plt.text(x-.05, y-.05, i, fontsize=9)
    plt.plot([0, 1, 0, 0], [0, 0, 1, 0], color='black', zorder=3)

    legend_elements = [plt.Line2D([0], [0], marker='x', color='r', lw=4,
                                  markersize = 8, linestyle='none', \
                              label=r'$V_h^{{{}}}(\widetilde{{T}})$ dofs'.format(degrees[1])), \
                   plt.Line2D([0], [0], marker='o', color='b', markerfacecolor='none', \
                          label=r'$V_h^{{{}}}(\widetilde{{T}})$ dofs'.format(degrees[0]), \
                          markersize=8, linestyle='none'), \
                   plt.Line2D([0], [0], marker='o', color='b', \
                          label=r'$V_h^{{{}}}(\widetilde{{T}})$ dofs "off"'.format(degrees[0]), \
                          markersize=8, linestyle='none')]

    plt.legend(handles=legend_elements, loc=1)

    plt.savefig('{}ref_triangle_{}.pdf'.format(path, num))
    plt.savefig('{}ref_triangle_{}.png'.format(path, num))
    return

'''
==== Mesh generation ====
'''
lc = 0.1

geom = pg.built_in.Geometry()

# Points
p1 = geom.add_point( [0.5, 0.5, 0], lcar = lc)
p2 = geom.add_point( [-0.5, 0.5, 0], lcar = lc)
p3 = geom.add_point( [-0.5, -0.5, 0], lcar = lc)
p4 = geom.add_point( [0.5, -0.5, 0], lcar = lc)

# Lines 
l1 = geom.add_line(p1, p2)
l2 = geom.add_line(p2, p3)
l3 = geom.add_line(p3, p4)
l4 = geom.add_line(p4, p1)

# Surface
loop = geom.add_line_loop( [l1, l2, l3, l4])
surf = geom.add_plane_surface(loop)

mesh = pg.generate_mesh(geom)

mesh.points = mesh.points[:, :2] # Used to convert 3D mesh into 2D

meshio.write("./mesh/centered_square.xdmf", meshio.Mesh(
    points=mesh.points,
    cells={'triangle': mesh.cells["triangle"]}))

# FE degree
k = 1

# Mesh creation
mesh = Mesh()
with XDMFFile(MPI.comm_world, './mesh/centered_square.xdmf') as f: f.read(mesh)

# FE space
V_init = FunctionSpace(mesh, 'CG', k)

'''
PRIMAL PROBLEM
'''
# True solution
r = 0.25    # Radius of internal layer
u_exact = Expression("atan(60.0*(pow(x[0],2)+pow(x[1],2) - r))", r=r, degree = k + 3)

# Data
f = Expression('-240./(1.+pow(60.*(pow(x[0], 2) + pow(x[1], 2) - r), 2)) \
               + pow(120., 3)*pow(x[0], 2)*(pow(x[0], 2) + pow(x[1], 2) - r)/pow(1. + pow(60.*(pow(x[0], 2) + pow(x[1], 2) - r), 2), 2) \
               + pow(120., 3)*pow(x[1], 2)*(pow(x[0], 2) + pow(x[1], 2) - r)/pow(1. + pow(60.*(pow(x[0], 2) + pow(x[1], 2) - r), 2), 2)', r=r, degree = k+3)

# ---- Boundary conditions ----
class Boundary0(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

boundary0 = Boundary0()

boundary_parts = [boundary0]

bc_data = {0: {'Dirichlet': u_exact}}  # Python dictionary describing boundary datas,
                            # only need to specify type of bc (between
                            # Dirichlet and Neumann) and the associated data in
                            # Neumann case

# Lambda function defining the bilinear form
bili = lambda u,v: inner(grad(u), grad(v))*dx

'''
COMPUTATION OF THE ASYMPTOTIC MESH
'''
tolerance = 0.5
iteration_max = 25
degrees = [k, k+1]

V_f, N = bw_estimation.bw_setup(mesh, degrees)

result, u_h, details = bw_adaptive('BW classic', V_init, f, V_f, bili, \
                                   boundary_parts, bc_data, tolerance, \
                                   iteration_max, N, u_exact = u_exact)

V_h = u_h.function_space()
mesh = V_h.mesh()

# ---- Boundary conditions ----
class Boundary0(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

boundary0 = Boundary0()

boundary_parts = [boundary0]

bc_data = {0: {'Dirichlet': u_exact}}  # Python dictionary describing boundary datas,
                            # only need to specify type of bc (between
                            # Dirichlet and Neumann) and the associated data in
                            # Neumann case

max_bound = len(boundary_parts)
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(max_bound)

'''
BW COMPARISONS
'''
def dofs_comparison():
    degrees = [k+1, k+2]

    bw_errors = []
    dims = []

    d = sum(np.arange(1, degrees[0]+2))

    if degrees[0] == degrees[1]:
        dofs = [0]
        ran = np.arange(1, d)
    else:
        dofs = []
        ran = np.arange(d+1)

    for i in ran:
        print('dofs =', dofs)
        # Bank-Weiser setup
        V_f, N = bw_estimation.bw_setup(mesh, degrees, dofs_list=dofs)
        print('BW space dim =', np.shape(N)[1])
        path = './outputs/dofs_comp_ref_triangles/'
        ref_triangle(degrees, np.shape(N)[1], path, dofs=dofs)
        dims.append(np.shape(N)[1])
        e_V_e, e_V_f = bw_estimation.estimation(V_h, f, V_f, u_h, bili, boundaries, bc_data, N)
        bw2 = e_V_e.vector()[:]

        # ---- True error ----
        W = FunctionSpace(mesh, 'CG', k+2)
        u_exact_W = interpolate(u_exact, W)
        u_h_W = interpolate(u_h, W)

        # Cell volume (used in projection on V_e to simplify cell volume factor)
        V_e = FunctionSpace(mesh, 'DG', 0)
        v_e = TestFunction(V_e)
        vol = Function(V_e)
        vol.vector()[:] = assemble(v_e*dx)

        err = u_exact_W - u_h_W
        err_e = project(inner(inner(grad(err), grad(err)), vol), V_e)

        '''
        with XDMFFile("output/e_V_e.xdmf") as f:
            f.write_checkpoint(e_V_e, "Element-wise bw estimator")

        with XDMFFile("output/err_e.xdmf") as f:
            f.write_checkpoint(err_e, "Element-wise true error")
        '''

        bw_errors.append(sqrt(np.sum(bw2))) # Global estimator
        error_exact = errornorm(u_exact, u_h, "H10") # Global true error

        print("Bank-Weiser {} value: {}".format(i, bw_errors[-1]))
        if degrees[0] == degrees[1]:
            if not (i == 0) :
                dofs.append(i)
        else:
            dofs.append(i)

    print("Exact error = {}".format(error_exact))

    bw_errors = np.asarray(bw_errors)
    error_exacts = np.ones(len(bw_errors))*error_exact
    efficiencies = np.divide(bw_errors, error_exacts)
    abs_relative_err = np.abs(efficiencies - np.ones(len(efficiencies)))

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title('Efficiencies of BW estimators against dimension of BW space for [{}, {}] FE degrees'.format(degrees[0], degrees[1]), fontsize='small')
    plt.plot(dims, efficiencies, color='blue', marker='x')
    plt.plot(dims, np.ones(len(dims)), color='red')
    plt.xlabel('Dimension of $V_h^{BW}$', fontsize='small')
    plt.ylabel(r'eff $= \frac{\eta_{BW}}{||\nabla e||}$', fontsize='small')
    plt.tick_params(axis='both', which='both',  labelsize=6)
    plt.subplot(2, 1, 2)
    plt.title('Absolute relative error of BW estimators against dimension of BW space for [{}, {}] FE degrees'.format(degrees[0], degrees[1]), fontsize='small')
    plt.plot(dims, abs_relative_err, color='blue', marker='x')
    plt.xlabel('Dimension of $V_h^{BW}$', fontsize='small')
    plt.ylabel('|eff - 1|', fontsize='small')
    plt.tick_params(axis='both', which='both',  labelsize=6)
    plt.tight_layout()
    plt.savefig('outputs/dofs_comparison_{}_{}.pdf'.format(degrees[0], degrees[1]))
    return

def dof_dim_comparison():
    degrees = [k+1, k+1]

    bw_errors = []
    dims = []

    d = sum(np.arange(1, degrees[0]+2))

    ran = np.arange(d)

    comb = np.array(np.meshgrid(ran, ran, ran)).T.reshape(-1, 3)
    mask = [len(np.unique(b)) == len(b) for b in comb]
    comb = comb[mask]
    comb = np.unique(np.sort(comb), axis=0)

    for i, dofs in enumerate(comb):
        print('dofs =', dofs)
        # Bank-Weiser setup
        V_f, N = bw_estimation.bw_setup(mesh, degrees, dofs_list=dofs)
        
        path = './outputs/dofs_dim_comp_ref_triangles/'
        ref_triangle(degrees, i, path, dofs=dofs)

        e_V_e, e_V_f = bw_estimation.estimation(V_h, f, V_f, u_h, bili, boundaries, bc_data, N)
        bw2 = e_V_e.vector()[:]

        # ---- True error ----
        W = FunctionSpace(mesh, 'CG', k+2)
        u_exact_W = interpolate(u_exact, W)
        u_h_W = interpolate(u_h, W)

        # Cell volume (used in projection on V_e to simplify cell volume factor)
        V_e = FunctionSpace(mesh, 'DG', 0)
        v_e = TestFunction(V_e)
        vol = Function(V_e)
        vol.vector()[:] = assemble(v_e*dx)

        err = u_exact_W - u_h_W
        err_e = project(inner(inner(grad(err), grad(err)), vol), V_e)

        with XDMFFile("outputs/residual_solutions/e_V_f_{}.xdmf".format(i)) as xdmf:
            xdmf.write_checkpoint(e_V_f, "Residual solution")
        
        '''
        with XDMFFile("output/err_e.xdmf") as f:
            f.write_checkpoint(err_e, "Element-wise true error")
        '''

        bw_errors.append(sqrt(np.sum(bw2))) # Global estimator
        error_exact = errornorm(u_exact, u_h, "H10") # Global true error

        print("Bank-Weiser {} value: {}".format(i, bw_errors[-1]))

    print("Exact error = {}".format(error_exact))

    bw_errors = np.asarray(bw_errors)
    error_exacts = np.ones(len(bw_errors))*error_exact
    efficiencies = np.divide(bw_errors, error_exacts)
    abs_relative_err = np.abs(efficiencies - np.ones(len(efficiencies)))
    
    xs = np.arange(len(comb))
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.title('Efficiencies of BW estimators for {} dof(s) and [{}, {}] FE degrees'.format(len(dofs), degrees[0], degrees[1]), fontsize='small')
    plt.plot(xs, efficiencies, color='blue', marker='x')
    plt.plot(xs, np.ones(len(xs)), color='red')
    plt.xlabel('Estimator number', fontsize='small')
    plt.ylabel(r'eff $= \frac{\eta_{BW}}{||\nabla e||}$', fontsize='small')
    plt.tick_params(axis='both', which='both',  labelsize=6)
    plt.subplot(2, 1, 2)
    plt.title('Absolute relative error of BW estimators for {} dof(s) and [{}, {}] FE degrees'.format(len(dofs), degrees[0], degrees[1]), fontsize='small')
    plt.plot(xs, abs_relative_err, color='blue', marker='x')
    plt.xlabel('Estimator number', fontsize='small')
    plt.ylabel('|eff - 1|', fontsize='small')
    plt.tick_params(axis='both', which='both',  labelsize=6)
    plt.tight_layout()
    plt.savefig('outputs/dofs_{}_dim_{}_{}.pdf'.format(len(dofs), degrees[0], degrees[1]))
    return

#dofs_comparison()
#dof_dim_comparison()
