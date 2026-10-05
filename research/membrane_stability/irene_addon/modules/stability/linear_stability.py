'''
Linear stability of a steady finite-element solution, built directly from the FE residual of the steady problem.

Consider a time-dependent problem written in the weak form

    m(d w / dt, w_test) + F(w; w_test) = 0            for all test functions w_test,

where F is the residual of the steady problem (the same form that is given to the steady Newton solver) and m is the
mass form, which only contains the fields that have a time derivative (e.g. the velocity for Navier-Stokes, not the
pressure, which acts as a Lagrange multiplier). Perturbing a steady state w* with w = w* + eps * w' exp(lambda t) and
keeping the terms of order eps gives the generalized eigenvalue problem

    A w' = lambda B w',     A = - dF/dw (w*)  (the Jacobian that Newton's method already uses),     B = m.

w* is linearly stable if all eigenvalues satisfy Re(lambda) < 0, and unstable as soon as one eigenvalue has
Re(lambda) > 0. For Navier-Stokes, the pressure rows of B vanish: this imposes div u' = 0 on the eigenvectors, which is
equivalent to applying the Leray (Helmholtz-Hodge) projector P to the linearized operator, i.e. the eigenvalues are
those of P L on the space of divergence-free fields (see checks/check_projector_equivalence.py).

Nothing here is specific to Navier-Stokes: any steady problem written as (F, w, bcs, mass) can be analysed, e.g. the
membrane problems.

Dirichlet conditions of the perturbations are imposed by replacing their rows by identity rows in A and zero rows in B:
these rows then correspond to infinite eigenvalues, which the shift-and-invert transformation maps to 0 and never
returns.
'''
import numpy as np
from fenics import PETScMatrix, TrialFunction, TestFunction, Function, assemble, derivative, as_backend_type
from petsc4py import PETSc
from slepc4py import SLEPc

# MUMPS workspace relaxation (percent): the default is too small for some mixed problems (PETSc error 76, INFO(1) = -9)
for _prefix in ("", "st_"):  # "st_": the KSP inside SLEPc's spectral transformation
    PETSc.Options().setValue(_prefix + "mat_mumps_icntl_14", 200)


def assemble_eigenproblem(F, w, bcs_hom, mass):
    '''
    Return the PETSc matrices (A, B) of the generalized eigenvalue problem A x = lambda B x.
    F:       residual form of the steady problem, F(w; w_test)
    w:       Function which contains the steady state w*
    bcs_hom: homogeneous Dirichlet BCs of the perturbation
    mass:    function (trial, test) -> mass form
    '''
    W = w.function_space()
    trial, test = TrialFunction(W), TestFunction(W)
    A = PETScMatrix()
    B = PETScMatrix()
    assemble(-derivative(F, w, trial), tensor=A)
    assemble(mass(trial, test), tensor=B)
    for bc in bcs_hom:
        bc.apply(A)  # identity row
        bc.zero(B)  # zero row
    return A, B


class Eigenpair:
    def __init__(self, eigenvalue, x_real, x_imag, residual):
        self.eigenvalue = eigenvalue
        self.x_real = x_real  # numpy arrays with the real and imaginary part of the eigenvector
        self.x_imag = x_imag
        self.residual = residual  # relative residual |A x - lambda B x| / |lambda x|

    @property
    def growth_rate(self):
        return self.eigenvalue.real

    @property
    def frequency(self):
        return self.eigenvalue.imag


def solve_eigenproblem(A, B, target=0.0, nev=30, ncv=None, tol=1e-10, max_it=500):
    '''
    Compute the nev eigenvalues of A x = lambda B x closest to 'target' with SLEPc (Krylov-Schur, shift-and-invert
    with a MUMPS LU factorization of A - target B). The eigenpairs are returned sorted by decreasing real part.
    PETSc is compiled with real scalars: complex eigenvalues are returned in conjugate pairs, the real and imaginary
    parts of the eigenvector are returned separately.
    '''
    A = as_backend_type(A).mat() if not isinstance(A, PETSc.Mat) else A
    B = as_backend_type(B).mat() if not isinstance(B, PETSc.Mat) else B
    eps = SLEPc.EPS().create(comm=PETSc.COMM_WORLD)
    eps.setOperators(A, B)
    eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eps.setDimensions(nev=nev, ncv=ncv if ncv is not None else max(2 * nev + 10, 40))
    eps.setTarget(target)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    eps.setTolerances(tol=tol, max_it=max_it)
    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    st.setShift(target)
    ksp = st.getKSP()
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    ksp.getPC().setFromOptions()
    eps.solve()

    pairs = []
    vr, vi = A.createVecs()
    for i in range(eps.getConverged()):
        lam = eps.getEigenpair(i, vr, vi)
        residual = eps.computeError(i, SLEPc.EPS.ErrorType.RELATIVE)
        pairs.append(Eigenpair(complex(lam), vr.getArray().copy(), vi.getArray().copy(), residual))
    eps.destroy()
    pairs.sort(key=lambda pair: -pair.eigenvalue.real)
    return pairs


def petsc_to_scipy(M):
    import scipy.sparse as sp
    M = as_backend_type(M).mat() if not isinstance(M, PETSc.Mat) else M
    indptr, indices, data = M.getValuesCSR()
    return sp.csr_matrix((data, indices, indptr), shape=M.getSize())


class ComplexShiftedSolver:
    '''
    Solve (A - s B) z = y for a complex shift s = a + i b in real arithmetic, with the equivalent real system
        [[A - a B,   b B    ],   [z_r]   [y_r]
         [ -b B,     A - a B]] . [z_i] = [y_i],
    factorized once with MUMPS.
    '''

    def __init__(self, A_s, B_s, shift):
        import scipy.sparse as sp
        a, b = shift.real, shift.imag
        K = sp.bmat([[A_s - a * B_s, b * B_s], [-b * B_s, A_s - a * B_s]], format="csr")
        K.sort_indices()
        self.n = A_s.shape[0]
        self.K = PETSc.Mat().createAIJ(size=K.shape, csr=(K.indptr.astype(PETSc.IntType),
                                                          K.indices.astype(PETSc.IntType), K.data))
        self.K.assemble()
        self.ksp = PETSc.KSP().create()
        self.ksp.setOperators(self.K)
        self.ksp.setType("preonly")
        self.ksp.getPC().setType("lu")
        self.ksp.getPC().setFactorSolverType("mumps")
        self.ksp.setUp()
        self.rhs, self.sol = self.K.createVecs()

    def solve(self, y):
        self.rhs.setArray(np.concatenate([y.real, y.imag]))
        self.ksp.solve(self.rhs, self.sol)
        z = self.sol.getArray()
        return z[:self.n] + 1j * z[self.n:]

    def destroy(self):
        self.ksp.destroy()
        self.K.destroy()


def solve_eigenproblem_complex_shift(A, B, shift, nev=10, tol=1e-10, max_iterations=3000):
    '''
    Eigenvalues of A x = lambda B x closest to the complex 'shift'. PETSc/SLEPc are built with real scalars, so the
    complex shift-and-invert operator (A - shift B)^{-1} B is applied through its real 2n x 2n form (factorized with
    MUMPS) and its largest eigenvalues mu = 1 / (lambda - shift) are computed with ARPACK (scipy).
    '''
    import scipy.sparse.linalg as spla
    A_s, B_s = petsc_to_scipy(A), petsc_to_scipy(B)
    n = A_s.shape[0]
    solver = ComplexShiftedSolver(A_s, B_s, shift)
    operator = spla.LinearOperator((n, n), matvec=lambda x: solver.solve(B_s @ x), dtype=complex)
    try:
        mu, X = spla.eigs(operator, k=nev, which="LM", tol=tol, ncv=max(3 * nev, 30), maxiter=max_iterations)
    except spla.ArpackNoConvergence as error:  # keep the eigenpairs that did converge
        mu, X = error.eigenvalues, error.eigenvectors
    solver.destroy()
    pairs = []
    for k in range(len(mu)):
        lam = shift + 1.0 / mu[k]
        x = X[:, k]
        residual = np.linalg.norm(A_s @ x - lam * (B_s @ x)) / (abs(lam) * np.linalg.norm(B_s @ x))
        pairs.append(Eigenpair(complex(lam), x.real.copy(), x.imag.copy(), residual))
    pairs.sort(key=lambda pair: -pair.eigenvalue.real)
    return pairs


DEFAULT_SHIFTS = tuple(0.1 + 0.3j * k for k in range(6))


def leading_eigenvalues(A, B, shifts=DEFAULT_SHIFTS, nev=8, tol=1e-10):
    '''
    Scan the upper half of the complex plane with several shifts (the eigenvalues come in complex-conjugate pairs,
    so Im(lambda) >= 0 is enough), merge the eigenvalues found around each shift and sort them by decreasing real
    part: pairs[0] is the leading (least stable) eigenvalue.
    The shifts are placed slightly to the right of the imaginary axis (Re = 0.1), so that the eigenvalues closest to
    them are the least stable ones and not the dense cluster of strongly damped modes (Re(lambda) ~ -0.1) that the
    discretized wake produces; with a spacing of 0.3 on the imaginary axis, any eigenvalue with Re(lambda) > -0.05
    in the range 0 <= Im(lambda) <= 1.65 is closer to a shift than all eigenvalues of that cluster.
    '''
    pairs = []
    for shift in shifts:
        for pair in solve_eigenproblem_complex_shift(A, B, complex(shift), nev=nev, tol=tol):
            if pair.eigenvalue.imag < -1e-8:
                continue
            if all(abs(pair.eigenvalue - other.eigenvalue) > 1e-6 * max(1.0, abs(pair.eigenvalue)) for other in pairs):
                pairs.append(pair)
    pairs.sort(key=lambda pair: -pair.eigenvalue.real)
    return pairs


def eigenvector_to_functions(pair, W):
    '''real and imaginary part of the eigenvector as FEniCS Functions in the (mixed) space W'''
    functions = []
    for x in (pair.x_real, pair.x_imag):
        f = Function(W)
        f.vector().set_local(x)
        f.vector().apply("insert")
        functions.append(f)
    return functions


def leading_eigenpair(pairs):
    return pairs[0]


def tie_dofs(A, B, groups):
    '''
    Constrain groups of degrees of freedom to move together (e.g. the height of a rigid inclusion: z' = h' on its whole
    rim, with h' one new unknown). With the prolongation P (n x n_reduced: identity on the untied dofs, one column per
    group with ones on its dofs), the Galerkin-reduced problem is  P^T A P y = lambda P^T B P y,  x = P y.  The row of a
    group is the sum of the rows of its dofs: the total generalized force on the group (e.g. the net vertical force on
    the inclusion), which the reduced problem sets to zero (force control / free height).
    A and B must be assembled WITHOUT Dirichlet conditions on the tied dofs. Returns (A_r, B_r, P) (PETSc, PETSc, scipy).
    '''
    import scipy.sparse as sp
    A_s, B_s = petsc_to_scipy(A), petsc_to_scipy(B)
    n = A_s.shape[0]
    tied = np.zeros(n, dtype=bool)
    for g in groups:
        tied[np.asarray(g)] = True
    free = np.flatnonzero(~tied)
    rows = list(free)
    cols = list(range(len(free)))
    for k, g in enumerate(groups):
        rows += list(g)
        cols += [len(free) + k] * len(g)
    P = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, len(free) + len(groups)))

    def to_petsc(M):
        M = sp.csr_matrix(M)
        M.sort_indices()
        out = PETSc.Mat().createAIJ(size=M.shape, csr=(M.indptr.astype(PETSc.IntType), M.indices.astype(PETSc.IntType),
                                                       M.data))
        out.assemble()
        return out

    return to_petsc(P.T @ A_s @ P), to_petsc(P.T @ B_s @ P), P
