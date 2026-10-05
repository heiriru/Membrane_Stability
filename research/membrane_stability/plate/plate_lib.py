'''
Standalone flat-membrane model of the flow-driven buckling (no IRENE): 2D membrane Stokes flow around one or several
protein inclusions (PIs), and the linear stability of the flat state as a plate under the flow-induced stress.

Reduction (exact for a flat base state, checked against IRENE's full equations to 0.1-0.4 %, see README):
    in-plane:  div T + f = 0,  div v = 0,  T = sigma I + 2 eta d(v)   (sigma: tension, f: tangential body force)
    normal:    zeta dz/dt = -kappa Delta^2 z + T : grad grad z
                          = -kappa Delta^2 z + div(T grad z) + f . grad z
(the tangential force has no normal component on the flat state; with f != 0 the operator is not self-adjoint:
follower load, possible flutter). The body force is f = f0 e_x + b (V e_x - v): a uniform tangential force density
(shear stress of the surrounding fluid) and/or friction with a surrounding fluid moving at V (Brinkman, screening
length ell_b = sqrt(eta / b)).

Geometries
    box:      [0, Lx] x [0, Ly], inflow v = v_in e_x on the left, slip walls (v_y = 0, zero tangential traction),
              traction T n = sigma0 n at the outflow, z = dz/dn = 0 on the outer boundary (IRENE's square_b).
    periodic: unit cell [0, Lx] x [0, Ly] of a lattice of PIs (periodic v, sigma; mean tension sigma0), driven by f0
              and/or the friction b (V - v); perturbations are Bloch waves z = exp(i q.x) p(x) with p periodic.
PIs: no slip (v = 0); vertically 'rigid' (free height, zero net vertical force: z' = h' on the rim, one unknown per PI)
or 'clamped' (z' = 0); zero slope on the rim (dz/dn = 0, natural in the mixed formulation m = Delta z).

All fields are linear in the drive D (inflow v_in, or the drive of the periodic cell): T = sigma0 I + D T1 with T1 the
stress at unit drive, f = D f1. Units: lengths r0 (PI radius 1), kappa = 1, eta = 1, zeta = 1.
'''
import os

import gmsh
import meshio
import numpy as np
from dolfin import (Constant, DirichletBC, FacetNormal, FiniteElement, Function, FunctionSpace, Measure, Mesh,
                    MeshValueCollection, MixedElement, PETScMatrix, SubDomain, TestFunctions, TrialFunctions,
                    VectorElement, XDMFFile, as_backend_type, as_vector, assemble, cpp, div, dot, grad, inner, near,
                    project, solve, sym, parameters, vertex_to_dof_map)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "irene_addon", "modules"))
from stability import linear_stability as ls  # noqa: E402

parameters["form_compiler"]["quadrature_degree"] = 4
LEFT, RIGHT, TOP, BOTTOM, HOLE0 = 2, 3, 4, 5, 10


# ---------------------------------------------------------------------------------------------------------------------
# meshes
def make_mesh(out, Lx, Ly, holes, r=1.0, h_min=0.15, h_max=3.0, grading=None, periodic=False):
    '''triangle mesh of the rectangle minus disks of radius r at 'holes' (list of centres), graded from h_min at the
    holes to h_max; facet tags LEFT, RIGHT, TOP, BOTTOM, HOLE0 + k. For periodic=True the boundary nodes of opposite
    sides match (gmsh periodic constraint). Writes mesh.xdmf and facets.xdmf in 'out'.'''
    os.makedirs(out, exist_ok=True)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("cell")
    occ = gmsh.model.occ
    rect = occ.addRectangle(0, 0, 0, Lx, Ly)
    disks = [occ.addDisk(cx, cy, 0, r, r) for cx, cy in holes]
    if disks:
        occ.cut([(2, rect)], [(2, d) for d in disks])
    occ.synchronize()
    eps = 1e-6

    def curves_in(xmin, ymin, xmax, ymax):
        return [c[1] for c in gmsh.model.getEntitiesInBoundingBox(xmin - eps, ymin - eps, -eps, xmax + eps,
                                                                   ymax + eps, eps, 1)]
    left, right = curves_in(0, 0, 0, Ly), curves_in(Lx, 0, Lx, Ly)
    bottom, top = curves_in(0, 0, Lx, 0), curves_in(0, Ly, Lx, Ly)
    hole_curves = [curves_in(cx - r, cy - r, cx + r, cy + r) for cx, cy in holes]
    surf = [s[1] for s in gmsh.model.getEntities(2)]
    gmsh.model.addPhysicalGroup(2, surf, 1)
    for tag, cs in ((LEFT, left), (RIGHT, right), (TOP, top), (BOTTOM, bottom)):
        gmsh.model.addPhysicalGroup(1, cs, tag)
    for k, cs in enumerate(hole_curves):
        gmsh.model.addPhysicalGroup(1, cs, HOLE0 + k)
    if periodic:
        gmsh.model.mesh.setPeriodic(1, right, left, [1, 0, 0, Lx, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        gmsh.model.mesh.setPeriodic(1, top, bottom, [1, 0, 0, 0, 0, 1, 0, Ly, 0, 0, 1, 0, 0, 0, 0, 1])
    all_holes = [c for cs in hole_curves for c in cs]
    if all_holes:
        fd = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(fd, "CurvesList", all_holes)
        gmsh.model.mesh.field.setNumber(fd, "Sampling", 200)
        ft = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(ft, "InField", fd)
        gmsh.model.mesh.field.setNumber(ft, "SizeMin", h_min)
        gmsh.model.mesh.field.setNumber(ft, "SizeMax", h_max)
        gmsh.model.mesh.field.setNumber(ft, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(ft, "DistMax", grading if grading else 0.25 * max(Lx, Ly))
        gmsh.model.mesh.field.setAsBackgroundMesh(ft)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    else:
        gmsh.option.setNumber("Mesh.MeshSizeMax", h_max)
    gmsh.model.mesh.generate(2)
    msh_file = os.path.join(out, "mesh.msh")
    gmsh.write(msh_file)
    gmsh.finalize()
    m = meshio.read(msh_file)
    pts = m.points[:, :2]
    tri = m.get_cells_type("triangle")
    lines = m.get_cells_type("line")
    line_tags = m.get_cell_data("gmsh:physical", "line")
    meshio.write(os.path.join(out, "mesh.xdmf"), meshio.Mesh(pts, [("triangle", tri)]))
    meshio.write(os.path.join(out, "facets.xdmf"),
                 meshio.Mesh(pts, [("line", lines)], cell_data={"tag": [line_tags.astype(np.int32)]}))
    np.savez(os.path.join(out, "geometry.npz"), Lx=Lx, Ly=Ly, holes=np.array(holes, dtype=float).reshape(-1, 2), r=r,
             periodic=periodic, h_min=h_min, h_max=h_max)
    return out


class Periodic(SubDomain):
    def __init__(self, Lx, Ly):
        super().__init__()
        self.Lx, self.Ly = Lx, Ly

    def inside(self, x, on_boundary):
        # left and bottom sides, without the two corners that are images of others
        return bool((near(x[0], 0) or near(x[1], 0)) and not (near(x[0], self.Lx) or near(x[1], self.Ly))
                    and on_boundary)

    def map(self, x, y):
        if near(x[0], self.Lx) and near(x[1], self.Ly):
            y[0], y[1] = x[0] - self.Lx, x[1] - self.Ly
        elif near(x[0], self.Lx):
            y[0], y[1] = x[0] - self.Lx, x[1]
        elif near(x[1], self.Ly):
            y[0], y[1] = x[0], x[1] - self.Ly
        else:
            y[0], y[1] = -1000.0, -1000.0


class Geometry:
    def __init__(self, path):
        g = np.load(os.path.join(path, "geometry.npz"))
        self.Lx, self.Ly, self.r = float(g["Lx"]), float(g["Ly"]), float(g["r"])
        self.holes = [tuple(h) for h in g["holes"]]
        self.periodic = bool(g["periodic"])
        self.mesh = Mesh()
        with XDMFFile(os.path.join(path, "mesh.xdmf")) as f:
            f.read(self.mesh)
        mvc = MeshValueCollection("size_t", self.mesh, 1)
        with XDMFFile(os.path.join(path, "facets.xdmf")) as f:
            f.read(mvc, "tag")
        self.facets = cpp.mesh.MeshFunctionSizet(self.mesh, mvc)
        self.ds = Measure("ds", domain=self.mesh, subdomain_data=self.facets)
        self.dx = Measure("dx", domain=self.mesh)
        self.pbc = Periodic(self.Lx, self.Ly) if self.periodic else None
        self.area = assemble(Constant(1.0) * self.dx)
        self.phi = len(self.holes) * np.pi * self.r ** 2 / (self.Lx * self.Ly)   # area fraction of PIs


# ---------------------------------------------------------------------------------------------------------------------
# base flow
class BaseFlow:
    '''Stokes flow at unit drive. box: inflow 1 (drive = v_in). periodic: drive = 1 multiplies the body force
    f0_unit e_x + b (V_unit - v) (give f0_unit, V_unit, b; e.g. f0_unit = 1, b = 0: uniform force; f0_unit = 0,
    V_unit = 1: friction with a fluid moving at V). sigma: deviation from sigma0 (mean zero for periodic).'''

    def __init__(self, geo, eta=1.0, b=0.0, f0_unit=0.0, V_unit=0.0):
        self.geo, self.eta, self.b, self.f0_unit, self.V_unit = geo, eta, b, f0_unit, V_unit
        mesh = geo.mesh
        el = MixedElement([VectorElement("P", mesh.ufl_cell(), 2), FiniteElement("P", mesh.ufl_cell(), 1)])
        W = FunctionSpace(mesh, el, constrained_domain=geo.pbc)
        (v, s), (nu, q) = TrialFunctions(W), TestFunctions(W)
        dx, ds = geo.dx, geo.ds
        d = lambda u: sym(grad(u))  # noqa: E731
        a = (2 * eta * inner(d(v), d(nu)) + s * div(nu) + q * div(v) + b * dot(v, nu)) * dx
        L = dot(Constant((f0_unit + b * V_unit, 0.0)), nu) * dx
        bcs = [DirichletBC(W.sub(0), Constant((0.0, 0.0)), geo.facets, HOLE0 + k) for k in range(len(geo.holes))]
        if not geo.periodic:
            bcs += [DirichletBC(W.sub(0), Constant((1.0, 0.0)), geo.facets, LEFT),
                    DirichletBC(W.sub(0).sub(1), Constant(0.0), geo.facets, TOP),
                    DirichletBC(W.sub(0).sub(1), Constant(0.0), geo.facets, BOTTOM)]
            # outflow: T n = sigma0 n; at unit drive the deviation from sigma0 has zero traction (natural)
        else:
            x0 = mesh.coordinates()[np.argmin(np.sum((mesh.coordinates() - [0.0, 0.0]) ** 2, axis=1))]
            bcs.append(DirichletBC(W.sub(1), Constant(0.0), f"near(x[0], {x0[0]}) && near(x[1], {x0[1]})",
                                   method="pointwise"))
        w = Function(W)
        solve(a == L, w, bcs, solver_parameters={"linear_solver": "mumps"})
        self.v, self.s = w.split(deepcopy=True)
        if geo.periodic:
            self.s.vector()[:] -= assemble(self.s * dx) / geo.area
        self.T1 = self.s * as_matrix_id() + 2 * eta * d(self.v)
        # body force per unit drive and its total (= drag on the PIs)
        self.f1 = as_vector((f0_unit + b * (V_unit - self.v[0]), -b * self.v[1]))
        self.U = assemble(self.v[0] * dx) / geo.area                      # mean membrane velocity at unit drive
        self.F_drag = assemble(self.f1[0] * dx) if geo.periodic else None  # total drag on the PIs at unit drive
        if not geo.periodic:
            n = FacetNormal(mesh)
            hole = sum((geo.ds(HOLE0 + k) for k in range(1, len(geo.holes))), geo.ds(HOLE0)) if geo.holes else None
            # force of the membrane on the PIs: -int T n (n outward of the membrane domain)
            self.F_drag = -assemble(dot(self.T1, n)[0] * hole) if hole else 0.0
        self.max_speed = float(np.abs(self.v.vector().get_local()).max())


def as_matrix_id():
    from dolfin import Identity
    return Identity(2)


# ---------------------------------------------------------------------------------------------------------------------
# plate operator
class Plate:
    '''Mixed (z, m = Delta z) P2 x P2 plate model on the base flow; Bloch wavevector q for periodic cells (real
    formulation of the complex Bloch problem: fields (z_r, m_r, z_i, m_i)).'''

    def __init__(self, base, pi_bc="rigid", bloch=False, kappa=1.0, zeta=1.0):
        geo = base.geo
        self.base, self.geo, self.pi_bc, self.bloch, self.kappa, self.zeta = base, geo, pi_bc, bloch, kappa, zeta
        mesh = geo.mesh
        P2 = FiniteElement("P", mesh.ufl_cell(), 2)
        n_fields = 4 if bloch else 2
        self.V = FunctionSpace(mesh, MixedElement([P2] * n_fields), constrained_domain=geo.pbc)
        self.q = Constant((0.0, 0.0))
        U, Phi = TrialFunctions(self.V), TestFunctions(self.V)
        if bloch:
            zr, mr, zi, mi = U
            pr, cr, pi_, ci = Phi
            q = self.q

            def D(a_r, a_i):
                return grad(a_r) - q * a_i, grad(a_i) + q * a_r

            def dd(A, B):
                return dot(A[0], B[0]) + dot(A[1], B[1])
            Dz, Dm, Dp, Dc = D(zr, zi), D(mr, mi), D(pr, pi_), D(cr, ci)
            T1, f1 = base.T1, base.f1
            self.a_m = (mr * cr + mi * ci + dd(Dz, Dc)) * geo.dx
            self.a_bend = -kappa * dd(Dm, Dp) * geo.dx
            self.a_tens = dd(Dz, Dp) * geo.dx
            # weak form of -(div(T grad z) + f . grad z): int (T Dz) . conj(D phi) - (f . Dz) conj(phi)
            self.a_flow = (dot(dot(T1, Dz[0]), Dp[0]) + dot(dot(T1, Dz[1]), Dp[1])
                           - dot(f1, Dz[0]) * pr - dot(f1, Dz[1]) * pi_) * geo.dx
            self.mass = zeta * (zr * pr + zi * pi_) * geo.dx
            z_subs = [0, 2]
        else:
            z, m = U
            p, c = Phi
            self.a_m = (m * c + dot(grad(z), grad(c))) * geo.dx
            self.a_bend = -kappa * dot(grad(m), grad(p)) * geo.dx
            self.a_tens = dot(grad(z), grad(p)) * geo.dx
            self.a_flow = (dot(dot(base.T1, grad(z)), grad(p)) - dot(base.f1, grad(z)) * p) * geo.dx
            self.mass = zeta * z * p * geo.dx
            z_subs = [0]
        self.bcs = []
        if not geo.periodic:
            for sub in z_subs:
                for tag in (LEFT, RIGHT, TOP, BOTTOM):
                    self.bcs.append(DirichletBC(self.V.sub(sub), Constant(0.0), geo.facets, tag))
        # rigid PIs: one (complex, for Bloch) unknown height h per PI; the rim values are z = h, i.e. for a Bloch wave
        # p = h exp(-i q.(x - c)) on the rim (c: PI centre): weighted prolongation, built in _eig for the current q
        self.groups = []
        coords = self.V.tabulate_dof_coordinates()
        for k, c in enumerate(geo.holes):
            rim = []
            for sub in z_subs:
                bc = DirichletBC(self.V.sub(sub), Constant(0.0), geo.facets, HOLE0 + k)
                if pi_bc == "clamped":
                    self.bcs.append(bc)
                else:
                    rim.append(np.array(sorted(bc.get_boundary_values().keys())))
            if rim:
                self.groups.append(dict(dofs=rim, xy=[coords[d] - np.array(c) for d in rim]))
        self.self_adjoint = base.b == 0 and base.f0_unit == 0

    def set_q(self, qx, qy):
        self.q.assign(Constant((qx, qy)))

    def _assemble(self, form, identity):
        M = PETScMatrix()
        assemble(form, tensor=M, keep_diagonal=True)
        for b in self.bcs:
            (b.apply if identity else b.zero)(M)
        return M

    def prolongation(self, n):
        '''P (n x n_reduced): identity on the free dofs, one column per rigid-PI unknown (h, or Re h and Im h for
        Bloch waves with the phase factors exp(-i q.(x - c)) on the rim)'''
        import scipy.sparse as sp
        tied = np.zeros(n, dtype=bool)
        for g in self.groups:
            for d in g["dofs"]:
                tied[d] = True
        free = np.flatnonzero(~tied)
        rows, cols, vals = list(free), list(range(len(free))), [1.0] * len(free)
        col = len(free)
        qv = np.array([float(self.q.values()[0]), float(self.q.values()[1])])
        for g in self.groups:
            if not self.bloch:
                rows += list(g["dofs"][0])
                cols += [col] * len(g["dofs"][0])
                vals += [1.0] * len(g["dofs"][0])
                col += 1
                continue
            (dr, di), (xr, xi) = g["dofs"], g["xy"]
            th_r, th_i = xr @ qv, xi @ qv
            # column Re h: p_r = cos, p_i = -sin;  column Im h: p_r = sin, p_i = cos
            rows += list(dr) + list(di)
            cols += [col] * (len(dr) + len(di))
            vals += list(np.cos(th_r)) + list(-np.sin(th_i))
            rows += list(dr) + list(di)
            cols += [col + 1] * (len(dr) + len(di))
            vals += list(np.sin(th_r)) + list(np.cos(th_i))
            col += 2
        return sp.csr_matrix((vals, (rows, cols)), shape=(n, col))

    def _eig(self, Kf, Mf, target, nev):
        K, M = self._assemble(Kf, True), self._assemble(Mf, False)
        if not self.groups:
            return ls.solve_eigenproblem(K, M, target=target, nev=nev)
        from petsc4py import PETSc
        A_s, B_s = ls.petsc_to_scipy(K), ls.petsc_to_scipy(M)
        P = self.prolongation(A_s.shape[0])

        def to_petsc(Ms):
            Ms = Ms.tocsr()
            Ms.sort_indices()
            out = PETSc.Mat().createAIJ(size=Ms.shape, csr=(Ms.indptr.astype(PETSc.IntType),
                                                            Ms.indices.astype(PETSc.IntType), Ms.data))
            out.assemble()
            return out
        Ar, Br = to_petsc(P.T @ A_s @ P), to_petsc(P.T @ B_s @ P)
        pairs = ls.solve_eigenproblem(Ar, Br, target=target, nev=nev)
        Ar.destroy()                       # explicit: otherwise every solve leaks ~10 MB (petsc4py / dolfin objects)
        Br.destroy()
        as_backend_type(K).mat().destroy()
        as_backend_type(M).mat().destroy()
        for p in pairs:
            p.x_real, p.x_imag = P @ p.x_real, P @ p.x_imag
        return pairs

    def rates(self, drive, sigma0, target=None, nev=8):
        '''temporal eigenpairs: zeta lambda z = -kappa Delta^2 z + T : grad grad z at the given drive (sorted by
        decreasing real part). In a periodic cell with free PIs the uniform translation z = const is neutral
        (lambda = 0 exactly, also for Bloch q = 0): the shift is then placed slightly below 0 and that mode dropped.'''
        scale = self.kappa * (2 * np.pi / max(self.geo.Lx, self.geo.Ly)) ** 4 / self.zeta
        if target is None:
            target = -1e-3 * scale if self.geo.periodic else 0.0
        pairs = self._eig(self.a_m - (self.a_bend + sigma0 * self.a_tens + drive * self.a_flow), self.mass, target,
                          nev + (2 if self.geo.periodic else 0))
        if self.geo.periodic:
            pairs = [p for p in pairs if abs(p.eigenvalue) > 1e-9 * scale]
        return pairs

    def drive_threshold(self, sigma0, nev=6):
        '''divergence threshold as a generalized eigenvalue problem in the drive (valid for the self-adjoint case):
        smallest positive real drive D with (a_m + bending + sigma0 tension) x = D (-a_flow) x'''
        pairs = self._eig(self.a_m + self.a_bend + sigma0 * self.a_tens, -self.a_flow, 0.0, nev)
        pos = [p for p in pairs if p.eigenvalue.real > 0 and abs(p.eigenvalue.imag) < 1e-8 * abs(p.eigenvalue)]
        return min(pos, key=lambda p: p.eigenvalue.real) if pos else None

    def leading(self, drive, sigma0, nev=8, target=None):
        pairs = self.rates(drive, sigma0, target=target, nev=nev)
        return pairs[0], pairs

    def temporal_threshold(self, sigma0, d_lo, d_hi, rtol=2e-3, nev=8):
        '''drive at which the leading real part crosses zero (bisection; d_lo stable, d_hi unstable). Returns
        (drive_c, leading eigenvalue just above threshold)'''
        lam_hi = self.leading(d_hi, sigma0, nev)[0].eigenvalue
        while d_hi - d_lo > rtol * d_hi:
            mid = 0.5 * (d_lo + d_hi)
            lam = self.leading(mid, sigma0, nev)[0].eigenvalue
            if lam.real > 0:
                d_hi, lam_hi = mid, lam
            else:
                d_lo = mid
        return 0.5 * (d_lo + d_hi), lam_hi

    def z_field(self, x, part=0):
        '''z (P1 vertex values) of a solution vector x of the mixed space (part 0: real, 1: imaginary for Bloch)'''
        f = Function(self.V)
        f.vector()[:] = x
        zs = f.split(deepcopy=True)
        z = zs[2 * part]
        P1 = FunctionSpace(self.geo.mesh, "P", 1)
        return project(z, P1).vector().get_local(), P1


def p1_triangulation(geo):
    P1 = FunctionSpace(geo.mesh, "P", 1)
    c = P1.tabulate_dof_coordinates()
    return c[:, 0], c[:, 1], vertex_to_dof_map(P1)[geo.mesh.cells()]
