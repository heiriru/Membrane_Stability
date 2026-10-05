from dolfin import *
import numpy as np
from petsc4py import PETSc
from slepc4py import SLEPc
import importlib
import runtime_arguments as rarg
import switch_problem as swi
import function_spaces_steady as fsp
import print_out_solution as pr_sol

# Initialize PETSc and SLEPc before creating any solver objects
PETSc.Initialize()
SLEPc.Initialize()

# Ensure PETSc and SLEPc are available
if not has_linear_algebra_backend("PETSc"):
    print("DOLFIN has not been configured with PETSc. Exiting.")
    exit()
if not has_slepc():
    print("DOLFIN has not been configured with SLEPc. Exiting.")
    exit()


def solve_steady():
    """
    Load mesh and variational problem; solve steady-state Navier–Stokes.
    Returns:
      rmsh     -- mesh module
      V        -- velocity function space
      u_star   -- computed steady velocity solution
    """
    rmsh = importlib.import_module(swi.rmsh)
    vp   = importlib.import_module(f"variational_problem_bc_{rarg.args.problem}_steady")
    W, up, bcs, F, J = vp.W, vp.up, vp.bcs, vp.F, vp.J

    problem = NonlinearVariationalProblem(F, up, bcs, J)
    solver  = NonlinearVariationalSolver(problem)
    solver.parameters["newton_solver"].update({
        "absolute_tolerance": 1e-8,
        "relative_tolerance": 1e-6,
        "maximum_iterations": 100
    })
    solver.solve()

    u_star, _ = up.split()
    pr_sol.print_solution_steady(u_star)
    print('Solved Steady-State Problem')

    return rmsh, fsp.Q_v, u_star


def eigenvalues_stability(V, Q, u_star, rmsh, Re, neigs=1):
    """
    Compute leading eigenvalues of linearized Navier–Stokes around u_star.
    Demonstrates:
      - Matrix assembly (MatAssembly)
      - Spectral Transform configuration
      - (Optional) Shift-and-invert
    """
    du, v = TrialFunction(V), TestFunction(V)
    a = (
        -Re*inner(dot(u_star, nabla_grad(du)), v)
        -Re*inner(dot(du, nabla_grad(u_star)), v)
        +inner(grad(du), grad(v))
    ) * rmsh.dx
    m = inner(du, v) * rmsh.dx

    # Assemble into PETSc matrices via assemble_system
    dummy = v[0]*rmsh.dx
    A = PETScMatrix(); assemble_system(a, dummy, [], A_tensor=A)
    M = PETScMatrix(); assemble_system(m, dummy, [], A_tensor=M)

    # Explicit MatAssembly to avoid unassembled error (error 73)
    A.mat().assemble()
    M.mat().assemble()

    # Eliminate constrained DOFs in M via BC.zero()
    from fenics import Constant as Cnst
    bc_u = [DirichletBC(V, Cnst((0,0)), bc)
            for bc in (rmsh.boundary_l, rmsh.boundary_tb, rmsh.boundary_circle)]
    for bc in bc_u:
        bc.zero(M)

    # Create SLEPc eigen solver
    solver = SLEPcEigenSolver(A, M)
    solver.parameters["solver"]       = "krylov-schur"
    solver.parameters["problem_type"] = "gen_non_hermitian"
    solver.parameters["which"]        = "LR"    # largest real
    # No spectral transform (direct solve)
    solver.parameters["spectral_transform"] = "none"

    # Configure ST's internal KSP and PC to avoid wrong-state
    eps = solver._eps  # underlying SLEPc EPS
    st  = eps.getST()
    ksp = st.getKSP()
    pc  = ksp.getPC()
    ksp.setUp()
    pc.setUp()

    # Optional: Use shift-and-invert to improve convergence (_instead of 'none')
    # solver.parameters["spectral_transform"] = "shift-and-invert"
    # solver.parameters["spectral_shift"]     = 1e-3  # example shift

    solver.solve(neigs)

    # Extract eigenvalues
    computed = []
    for i in range(min(neigs, solver.get_number_converged())):
        r, _ = solver.get_eigenvalue(i)
        computed.append(r)
    return np.array(computed)


def print_stability(rmsh, V, u_star, Re):
    vals = eigenvalues_stability(V, fsp.Q_p, u_star, rmsh, Re)
    print(f"Stability eigenvalues (largest real): {vals}")


def main():
    Re = float(getattr(rarg.args, 'Re', 1e-9))
    rarg.args.Re = Re
    rmsh, V, u_star = solve_steady()
    print_stability(rmsh, V, u_star, Re)

if __name__ == '__main__':
    main()

# Finalize SLEPc if desired
# SLEPc.Finalize()
