import numpy as np
from mpi4py import MPI
from dolfin import *
from petsc4py import PETSc
from slepc4py import SLEPc

def get_smallest_eigenvalue(mesh, alpha):
    # 1) FE-Raum P2
    V = FunctionSpace(mesh, "Lagrange", 2)
    u = TrialFunction(V)
    v = TestFunction(V)

    # 2) Variationsformen
    a_form = -alpha * dot(grad(u), grad(v)) * dx
    m_form = u * v * dx

    # 3) Assembly
    A_full = PETScMatrix()
    M_full = PETScMatrix()
    assemble(a_form, tensor=A_full)
    assemble(m_form, tensor=M_full)

    # 4) Dirichlet-Randbedingungen
    bcs = DirichletBC(V, Constant(0.0), "on_boundary")
    bdofs = np.array(list(bcs.get_boundary_values().keys()), dtype=np.int32)

    # 5) Index-Set für Innen-DOFs
    all_dofs = np.arange(V.dim(), dtype=np.int32)
    interior = np.setdiff1d(all_dofs, bdofs, assume_unique=True)
    is_interior = PETSc.IS().createGeneral(interior, comm=PETSc.COMM_WORLD)

    # 6) Extrahiere Innen-Matrizen
    A_int = A_full.mat().createSubMatrix(is_interior, is_interior)
    M_int = M_full.mat().createSubMatrix(is_interior, is_interior)

    # 7) SLEPc-Einstellungen
    eps = SLEPc.EPS().create(comm=PETSc.COMM_WORLD)
    eps.setOperators(A_int, M_int)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps.setDimensions(nev=1)
    eps.setTolerances(tol=1e-10, max_it=1000)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
    # Direktlösung mit MUMPS
    st = eps.getST()
    ksp = st.getKSP()
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")

    eps.solve()
    if eps.getConverged() < 1:
        raise RuntimeError("SLEPc: keine Eigenzahl konvergiert")

    # 8) Eigenwert und -vektor
    vr_int, vi_int = A_int.createVecs()
    lam = eps.getEigenpair(0, vr_int, vi_int).real

    # 9) Voll-Vektor rekonstruieren
    full_vr = PETSc.Vec().createWithArray(np.zeros(V.dim()), comm=PETSc.COMM_WORLD)
    full_vr.setValues(interior, vr_int.getArray())
    full_vr.assemble()

    # 10) In Dolfin-Funktion
    u_h = Function(V)
    u_h.vector().set_local(full_vr.getArray())
    u_h.vector().apply("insert")

    return lam, u_h

def main():
    # Feinheit pro Richtung
    N = 50
    mesh = UnitSquareMesh(N, N)
    alpha = 3.0

    lam, u = get_smallest_eigenvalue(mesh, alpha)

    # Analytisch: λ1 = -2 α π²
    true_val = -2 * alpha * np.pi**2

    print(f"Computed: {lam:.6f}")
    print(f"Expected (from continous case): {true_val:.6f}")

    # Optional: Visualisierung der Eigenfunktion
    import matplotlib.pyplot as plt
    plot(u)
    plt.title("First Eigenfunction of the Laplacian")
    plt.savefig("eigenfunction_square_laplacian.png")
    plt.show()

if __name__ == "__main__":
    main()
