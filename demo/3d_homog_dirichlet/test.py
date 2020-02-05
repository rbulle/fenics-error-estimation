from dolfin import *
import ufl

def cartesian2polar(x):
    r = ufl.sqrt(x[0]**2 + x[1]**2)
    theta = ufl.mathfunctions.Atan2(x[1], x[0])
    return r, theta

mesh = Mesh()
with XDMFFile(MPI.comm_world, 'mesh.xdmf') as f:
    f.read(mesh)

V = FunctionSpace(mesh, 'CG', 1)

x = ufl.SpatialCoordinate(mesh)

r, theta = cartesian2polar(x)

cut_off = (1.-x[0]**2)**2*(1.-x[1]**2)**2*(1.-x[2]**2)**2
u_exact = cut_off*(r**(2./3.)*ufl.sin((2./3.)*(theta+ufl.pi/2.)))

sV = FunctionSpace(mesh, 'CG', 1)

su = TrialFunction(sV)
sv = TestFunction(sV)

su_exact = Function(sV)

a = inner(su, sv)*dx
L = inner(u_exact, sv)*dx

A, b = assemble_system(a, L)

solver = PETScLUSolver()
solver.solve(A, su_exact.vector(), b)

with XDMFFile('output/su.xdmf') as f:
    f.write_checkpoint(su_exact, 'su')

