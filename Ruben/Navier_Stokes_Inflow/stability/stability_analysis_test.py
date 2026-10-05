import sys, importlib
from fenics import NonlinearVariationalProblem, NonlinearVariationalSolver, parameters, DirichletBC, Constant
import colorama as col

import runtime_arguments as rarg
import switch_problem   as swi
import function_spaces_steady  as fsp
import print_out_solution as pr_sol  
import stability_operators as ops
import numpy as np
from mpi4py import MPI
from dolfin import *
from petsc4py import PETSc
from slepc4py import SLEPc
from fenics import Constant as Cnst


def solve_steady():
    # Load mesh and variational problem
    rmsh = importlib.import_module(swi.rmsh)
    vp   = importlib.import_module(f"variational_problem_bc_{rarg.args.problem}_steady")

    # Pull mixed-space and problem data
    W    = vp.W
    up   = vp.up
    bcs  = vp.bcs
    F    = vp.F
    J    = vp.J

    # Setup and solve nonlinear variational problem
    problem = NonlinearVariationalProblem(F, up, bcs, J)
    solver  = NonlinearVariationalSolver(problem)
    solver.parameters["newton_solver"]["absolute_tolerance"]  = 1e-8
    solver.parameters["newton_solver"]["relative_tolerance"]  = 1e-6
    solver.parameters["newton_solver"]["maximum_iterations"]  = 100
    solver.solve()

    # Extract velocity and pressure
    u_star, p_star = up.split()

    # Output solutions
    pr_sol.print_solution_steady(u_star, p_star)
    print('Solved Steady-State Problem')

    # Return mesh module, velocity-space, velocity solution, and BCs
    return rmsh, fsp.Q_v, u_star

def get_largest_eigenvalue_PL(V,Q, u_star, rmsh, R):

    # Parameter 
    #Re = Constant(0.0)
    Re = R
    # Trial-/Testfunctions
    V = fsp.Q_v.collapse()  # collapse to velocity subspace
    Q = fsp.Q_p.collapse()  # collapse to pressure subspace
    du, v = TrialFunction(V), TestFunction(V)
    p,  q = TrialFunction(Q), TestFunction(Q)
  

    # 1) Bilinear Form L
    a_L = (
        - Re * inner(dot(u_star, nabla_grad(du)), v)
        - Re * inner(dot(du, nabla_grad(u_star)), v)
        - inner(grad(du), grad(v))
    ) * rmsh.dx

    # 2) Mass‐Matrix 
    m_L = inner(du, v) * rmsh.dx

    # (2) D and DG for projector
    #Q = FunctionSpace(mesh, "Lagrange", 1)
    a_DG = -inner(grad(p), grad(q)) * rmsh.dx
    b_D  = - inner(q, div(du))  * rmsh.dx

    # Assemble matrices without BC
    A_L    = PETScMatrix()
    M_m    = PETScMatrix()
    DG_mat = PETScMatrix()
    D_mat  = PETScMatrix()
    assemble(a_L,   tensor=A_L)
    assemble(m_L, tensor=M_m)
    assemble(a_DG,  tensor=DG_mat)
    assemble(b_D,   tensor=D_mat)

    # ---------------------------------------------------------------
    # (3) BCs

    bc_u_in  = DirichletBC(V, Cnst((0.0,0.0)), rmsh.boundary_l)
    bc_u_w   = DirichletBC(V, Cnst((0.0,0.0)), rmsh.boundary_tb)
    bc_u_cyl = DirichletBC(V, Cnst((0.0,0.0)), rmsh.boundary_circle)
    bc_p_out = DirichletBC(Q, Cnst(0.0),      rmsh.boundary_r)
    bcs_u = [bc_u_in, bc_u_w, bc_u_cyl]
    bdofs_u = np.unique(np.hstack([list(bc.get_boundary_values().keys()) for bc in bcs_u])).astype(np.int32)
    bcs_p = [bc_p_out]
    bdofs_p = np.unique(np.hstack([list(bc.get_boundary_values().keys()) for bc in bcs_p])).astype(np.int32)
    # 3c) inner DOFs, eliminated Dirichlet-BCS
    all_u = np.arange(V.dim(), dtype=np.int32)
    all_p = np.arange(Q.dim(), dtype=np.int32)
    int_u = np.setdiff1d(all_u, bdofs_u, assume_unique=True)
    int_p = np.setdiff1d(all_p, bdofs_p, assume_unique=True)

    # 3d) PETSc Index Sets
    is_u = PETSc.IS().createGeneral(int_u, comm=PETSc.COMM_WORLD)
    is_p = PETSc.IS().createGeneral(int_p, comm=PETSc.COMM_WORLD)
    # ---------------------------------------------------------------

    # (4) Submatrices on Interior-DOFs
    A_L_int = A_L.mat().createSubMatrix(is_u, is_u) #L goes from V->V hence the BCS
    M_m_int = M_m.mat().createSubMatrix(is_u, is_u) # M goes from V->V hence the BCS
    DG_int  = DG_mat.mat().createSubMatrix(is_p, is_p) # DG goes from Q->Q hence the BCS
    D_int   = D_mat.mat().createSubMatrix(is_p, is_u) # D goes from V->Q hence the BCS
    G_full = D_mat.mat().transpose()
    G_int   = G_full.createSubMatrix(is_u, is_p)  # Gradient = Transpose from D_int

    # (5) Inverter for DG_int
    ksp_DG = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
    ksp_DG.setOperators(DG_int)
    ksp_DG.setType("preonly")
    ksp_DG.getPC().setType("lu")
    ksp_DG.getPC().setFactorSolverType("mumps")
    ksp_DG.setUp()
    n_p = DG_int.getSize()[0]
    invDG = PETSc.Mat().createAIJ(size=[n_p, n_p], comm=PETSc.COMM_WORLD)
    invDG.setUp()
    for j in range(n_p):
        e = DG_int.createVecRight()
        e.set(0)
        e.setValue(j, 1.0)
        e.assemble()
        sol = DG_int.createVecLeft()
        ksp_DG.solve(e, sol)
        rows = np.arange(n_p, dtype=np.int32).tolist()
        invDG.setValues(rows, [j], sol.getArray())
    invDG.assemble()
    #Solution vector for DG_int
    P   = PETSc.Mat().createAIJ(A_L_int.getSize(), comm=PETSc.COMM_WORLD)
    diag = PETSc.Vec().createWithArray(np.ones(len(int_u)), comm=PETSc.COMM_WORLD)
    P.setDiagonal(diag)
    P.assemble()
    tmp = G_int.matMult(invDG).matMult(D_int)
    P.axpy(-1.0, tmp)
    PL = P.matMult(A_L_int)
    # (7) M_int = P_int * A_L_int

    # (8) solve EV problem
    eps = SLEPc.EPS().create(comm=PETSc.COMM_WORLD)
    eps.setFromOptions()
    eps.setOperators(PL, M_m_int)
    eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP) # General Non Hermitian Eigenvalue Problem
    eps.setDimensions(nev=1)
    #eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)

    eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
    eps.setTolerances(tol=1e-8, max_it=1000)
    st = eps.getST()
    st.setFromOptions() 
    st.setType(SLEPc.ST.Type.SINVERT)
    st.getKSP().setType("preonly")
    st.getKSP().getPC().setType("lu")
    st.getKSP().getPC().setFactorSolverType("mumps")
    #eps.getST().getKSP().setType("preonly")
    #eps.getST().getKSP().getPC().setType("lu")
    #eps.getST().getKSP().getPC().setFactorSolverType("mumps")
    #PetscOptions().setValue("-log_summary","")
    eps.view()
    st = eps.getST()
    st.view()    
    eps.solve()
    #if eps.getConverged() < 1:
    #    raise RuntimeError("SLEPc: kein Eigenwert konvergiert")

    # (9) extract eigenvalue and eigenvector
    #lam = eps.getEigenpair(0,
    #                       PETSc.Vec().createSeq(M_int.getSize()[0]),
    #                       PETSc.Vec().createSeq(M_int.getSize()[0]))[0].real
    #return lam
def main():
    Rval = 0.000000001
    rarg.args.Re = Rval
    rmsh, Q_v, u_star = solve_steady()
    Q_p = fsp.Q_p
    lam_max = get_largest_eigenvalue_PL(Q_v, Q_p,u_star, rmsh, Rval)
    #print(f"Größter Eigenwert von P L: {lam_max:.6e}")

if __name__ == "__main__":
    main()



