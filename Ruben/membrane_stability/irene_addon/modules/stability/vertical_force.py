'''
Vertical force on a protein inclusion (PI) from the conserved vertical stress flux of the membrane (Monge gauge),
evaluated as a domain integral (no derivatives on the boundary).

Translations in z are a symmetry (nothing depends on z itself), so the vertical component of the membrane force balance
is a conservation law  d_i J^i = 0  in the base plane, with the flux (force per unit base length across a line with
base normal e_i)
    J^i = J_bend^i + sigma omega_i / sqrt(g) + 2 eta sqrt(g) d^{ij} omega_j,
    J_bend^i = dL/dz_i - d_j (dL/dz_ij),   L = 2 kappa H^2 sqrt(g)   (Noether current of the Helfrich energy),
    dL/dz_ij = 2 kappa H P_ij,  P_ij = delta_ij - z_i z_j / g,
    dL/dz_i  = 2 kappa H^2 z_i / sqrt(g) + 4 kappa H sqrt(g) dH/dz_i,
    dH/dz_i  = 1/2 [ -z_kk z_i g^{-3/2} - 2 z_l z_il g^{-3/2} + 3 z_i z_k z_l z_kl g^{-5/2} ]   (at fixed z_kl),
with H = mu (IRENE's mean curvature, H = 1/2 g^{ij} b_ij, b_ij = z_ij / sqrt(g)), z_i = omega_i, z_ij = d_j omega_i.
The tension and viscous parts are the vertical components of the in-plane stress T^{ij} e_j (e_j . e_z = z_j).
With a cutoff phi (1 near the PI, 0 beyond a radius b, and inside the domain) the force of the membrane on the PI is
    F = - int_Omega J^i d_i phi dx     (sign fixed against the virtual-work force -dE/dh in the ring problem),
independent of phi at equilibrium (d_i J^i = 0); its spread over several cutoffs measures the discretization error.
'''
from fenics import Expression, as_tensor, as_vector, assemble, grad, sqrt
import ufl


def flux(omega, mu, sigma, kappa, d_contra=None, eta=0.0):
    i, j, k, l = ufl.indices(4)
    g = 1.0 + omega[0] ** 2 + omega[1] ** 2
    sg = sqrt(g)
    Z = grad(omega)                               # Z[a, b] = d_b omega_a = z_ab (symmetric up to discretization)
    Zs = 0.5 * (Z + Z.T)
    H = mu
    trZ = Zs[0, 0] + Zs[1, 1]
    zZz = omega[k] * Zs[k, l] * omega[l]
    dH = as_vector([0.5 * (-trZ * omega[a] / g ** 1.5 - 2.0 * (Zs[a, 0] * omega[0] + Zs[a, 1] * omega[1]) / g ** 1.5
                           + 3.0 * omega[a] * zZz / g ** 2.5) for a in range(2)])
    dLdzi = as_vector([2 * kappa * H ** 2 * omega[a] / sg + 4 * kappa * H * sg * dH[a] for a in range(2)])
    P = as_tensor([[(1.0 if a == b else 0.0) - omega[a] * omega[b] / g for b in range(2)] for a in range(2)])
    M = 2 * kappa * H * P                          # dL/dz_ij
    divM = as_vector([M[a, 0].dx(0) + M[a, 1].dx(1) for a in range(2)])
    J = dLdzi - divM + sigma * omega / sg
    if d_contra is not None and eta:
        J = J + 2 * eta * sg * as_vector([d_contra[a, 0] * omega[0] + d_contra[a, 1] * omega[1] for a in range(2)])
    return J


def cutoff(center, a, b, mesh, degree=4):
    '''smooth radial cutoff: 1 for r < a, cos^2 taper to 0 at r = b'''
    rr = "sqrt((x[0]-cx)*(x[0]-cx) + (x[1]-cy)*(x[1]-cy))"
    return Expression(f"{rr} < a ? 1.0 : ({rr} > b ? 0.0 : pow(cos(0.5*pi*({rr} - a)/(b - a)), 2))",
        cx=center[0], cy=center[1], a=a, b=b, degree=degree, domain=mesh)


def vertical_force(omega, mu, sigma, kappa, dx, center, radii=((1.5, 4.0), (2.0, 6.0), (3.0, 8.0)), d_contra=None,
                   eta=0.0):
    '''force of the membrane on the PI for several cutoffs (list)'''
    J = flux(omega, mu, sigma, kappa, d_contra, eta)
    mesh = omega.function_space().mesh()
    return [-assemble(ufl.dot(J, grad(cutoff(center, a, b, mesh))) * dx) for a, b in radii]
