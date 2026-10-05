from mpi4py import MPI
from dolfin import *
import importlib
import switch_problem   as swi
from slepc4py import SLEPc
from petsc4py import PETSc

import numpy as np
import function_spaces_steady as fsp
from dolfin import Mesh, XDMFFile, Measure
from dolfin import MeshValueCollection


def compute_laplacian_eigen(rmsh, degree=2, nev=1, tol=1e-10):

    # 1) FE-Space Pdegree
    #V = fsp.Q_p
    V = FunctionSpace(rmsh.lmsh.mesh, "Lagrange", degree)
    u = TrialFunction(V)
    v = TestFunction(V)

    # 2) Variationsformen ohne Minus
    a_form = dot(grad(u), grad(v)) * rmsh.dx
    m_form = u * v * rmsh.dx

    # 3) Assembly
    A = PETScMatrix(); M = PETScMatrix()
    assemble(a_form, tensor=A)
    assemble(m_form, tensor=M)

    # 4) Dirichlet-BC
    bc_u_in   = DirichletBC(V, Constant(0.0), rmsh.boundary_l)
    bc_u_w    = DirichletBC(V, Constant(0.0), rmsh.boundary_tb)
    bc_u_cyl  = DirichletBC(V, Constant(0.0), rmsh.boundary_circle)
    bc_p_out  = DirichletBC(V, Constant(0.0), rmsh.boundary_r)
    bc = [bc_u_in, bc_u_w, bc_u_cyl, bc_p_out]
    #bc = DirichletBC(V, Constant(0.0), rmsh.boundary_l + rmsh.boundary_tb + rmsh.boundary_circle + rmsh.boundary_r)
    all_bdofs = np.hstack([list(b.get_boundary_values().keys()) for b in bc])
    bdofs = np.unique(all_bdofs).astype(np.int32)
    all_dofs = np.arange(V.dim(), dtype=np.int32)
    #interior = np.setdiff1d(all_dofs, bdofs)
    interior = np.setdiff1d(np.arange(V.dim(), dtype=np.int32), bdofs, assume_unique=True)

    # 5) Index-Sets 
    is_int = PETSc.IS().createGeneral(interior, comm=PETSc.COMM_WORLD)
    A_int = A.mat().createSubMatrix(is_int, is_int)
    M_int = M.mat().createSubMatrix(is_int, is_int)

    # 6) SLEPc-Setup
    eps = SLEPc.EPS().create(comm=PETSc.COMM_WORLD)
    eps.setOperators(A_int, M_int)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps.setDimensions(nev=nev)
    eps.setTolerances(tol=tol, max_it=1000)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)

    # LU-Preconditioner (MUMPS)
    st = eps.getST()
    ksp = st.getKSP()
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    print("Matrix size:", A_int.getSize(), "  #interior DOFs:", len(interior))
    print("Min/Max interior index:", interior.min(), interior.max())
    eps.solve()
    if eps.getConverged() < 1:
        raise RuntimeError("Keine Eigenzahl konvergiert")

    # 7) Eigenvalue and vector
    vr, _ = A_int.createVecs()
    lam = eps.getEigenpair(0, vr, None).real

    # 8) Reconstruct full vector with the BC
    full = PETSc.Vec().createWithArray(np.zeros(V.dim()), comm=PETSc.COMM_WORLD)
    full.setValues(interior, vr.getArray())
    full.assemble()
    u_h = Function(V)
    u_h.vector().set_local(full.getArray())
    u_h.vector().apply("insert")

    return lam, u_h

def main():
    rmsh = importlib.import_module(swi.rmsh)
    # Jetzt die Laplace-Eigen­zahl
    lambda1, u1 = compute_laplacian_eigen(rmsh, degree=2)
    print(f"Kleinste Laplace-Eigenzahl: {lambda1:.6f}")

    # Optional: Visualisierung
    import matplotlib.pyplot as plt
    plot(u1)
    plt.title("First Eigenfunction of Laplacian on the grid")
    plt.savefig("eigenfunc_on_custom_mesh.png")
    plt.show()

if __name__ == "__main__":
    main()
