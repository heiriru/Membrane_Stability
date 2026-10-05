
import numpy as np
from mpi4py import MPI
from dolfin import *
from petsc4py import PETSc
from slepc4py import SLEPc

def get_smallest_eigenvalue_interval(mesh, alpha):
    # FE-space
    V = FunctionSpace(mesh, "Lagrange", 2)
    u = TrialFunction(V)
    v = TestFunction(V)

    # variational forms
    a_form = -alpha * dot(grad(u), grad(v)) * dx
    m_form = u * v * dx

    # assemble matrices
    A_full = PETScMatrix()
    M_full = PETScMatrix()
    assemble(a_form, tensor=A_full)
    assemble(m_form, tensor=M_full)

    # 2) Determine BCS DOF
    bcs = DirichletBC(V, Constant(0.0), "on_boundary")
    bdofs = np.array(list(bcs.get_boundary_values().keys()), dtype=np.int32)

    # 3) Construct PETSc index set für Interior-DOFs
    all_dofs = np.arange(V.dim(), dtype=np.int32)
    interior = np.setdiff1d(all_dofs, bdofs, assume_unique=True)
    is_interior = PETSc.IS().createGeneral(interior, comm=PETSc.COMM_WORLD)

    # 4) extract intertior matrices to avoid spurious eigenvalues
    A_int = A_full.mat().createSubMatrix(is_interior, is_interior)
    M_int = M_full.mat().createSubMatrix(is_interior, is_interior)

    # 5) solve SLEPc 
    eps = SLEPc.EPS().create(comm=PETSc.COMM_WORLD)
    eps.setOperators(A_int, M_int)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps.setDimensions(nev=1)
    eps.setTolerances(tol=1e-10, max_it=1000)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
    eps.getST().getKSP().setType("preonly")
    eps.getST().getKSP().getPC().setType("lu")
    eps.getST().getKSP().getPC().setFactorSolverType("mumps")
    eps.solve()
    if eps.getConverged() < 1:
        raise RuntimeError("SLEPc: no eigenvalue converged")

    # 6) Extract eigenvalue and eigenvector
    vr_int, vi_int = A_int.createVecs()
    lam = eps.getEigenpair(0, vr_int, vi_int).real

    # 7) interpolate
    full_vr = PETSc.Vec().createWithArray(np.zeros(V.dim()), comm=PETSc.COMM_WORLD)
    full_vr.setValues(interior, vr_int.getArray())
    full_vr.assemble()

    # 8) In dolfin Function
    u_h = Function(V)
    u_h.vector().set_local(full_vr.getArray())
    u_h.vector().apply("insert")

    return lam, u_h

# use on 1d mesh
def main():
    N = 100
    mesh = UnitIntervalMesh(N)
    h = 1/(N+1)
    alpha = 3
    lam, u = get_smallest_eigenvalue_interval(mesh, alpha)
    D = (-2*np.eye(N) + np.eye(N, k=1) + np.eye(N, k=-1)) / h**2
    eigvals = np.linalg.eigvalsh(D)
    max_real_disc = np.max(eigvals)
    true_val = max_real_disc * alpha
    print(f"Computed: {lam:.6f}, Expected: {true_val:.6f}")
    #assert np.isclose(lam, true_val, rtol=1e-3)

if __name__ == "__main__":
    main()
