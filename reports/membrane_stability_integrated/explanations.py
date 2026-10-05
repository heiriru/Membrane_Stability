"""Explanatory text integrated with the archived equations and results.
These passages explain existing calculations; they introduce no new simulations.
"""
section_guides = {
'sec:intro': r'''
\subsection{The three questions answered by the calculations}
A computed membrane shape answers only the first of three questions. \emph{Existence}: can the forces and geometric constraints balance at this shape? \emph{Stability}: if the shape or flow is disturbed slightly, does it return or depart? \emph{Nonlinear fate}: if it departs, does it settle into another shape, oscillate, or continue to deform? A stationary solver addresses existence; an eigenvalue calculation addresses local stability; continuation and time integration address different parts of the nonlinear fate. The project needed all three because a stationary solver can converge to a shape that would never persist in an experiment.

The simplest analogy is an overdamped displacement $a$ obeying $\zeta\dot a=-ka$. The state $a=0$ solves the stationary equation for either sign of $k$. For $k>0$, a small displacement decays as $\exp(-kt/\zeta)$; for $k<0$, it grows. The membrane has many coupled displacement and flow degrees of freedom, so its restoring coefficient is a matrix rather than one number. Stability analysis identifies the combinations of those degrees of freedom that behave like independent growing or decaying disturbances.

The reference physical system is a membrane patch with a hole occupied by a rigid protein. The protein obstructs membrane flow, and the outer boundary supplies the flow and holds the patch. The protein's radius, the patch size, the bending stiffness, and the imposed tension all matter. Mobile curvature proteins introduced later are a separate field transported over this membrane; they should not be confused with the rigid anchored inclusion. Periodic arrays, introduced last, replace one inclusion and a distant boundary by a repeating arrangement of anchored inclusions.

\subsection{How to read the evidence}
Each part below follows a physical question, a mathematical problem, a numerical calculation, and its interpretation. An eigenvalue curve describes small disturbances of a particular base state. A continuation curve describes stationary solutions, including unstable ones. A time trace describes one evolving initial condition. A snapshot shows geometry or a field at one time. These are complementary observations: a striking final snapshot cannot establish an onset threshold, and a precise onset cannot establish the final morphology. The older diagnostic figures remain because they explain how the corrected problem was identified; their boundary conditions are stated where they are discussed.
''',
'sec:operator': r'''
\subsection{From a stationary state to a measurable growth rate}
Let $\Psi_*$ denote the complete stationary state: not just the height, but also the velocities, tension, curvature variables, and any additional coupled fields. It satisfies the discrete force and constraint equations $\mathcal R(\Psi_*)=0$. We ask what happens to a small disturbance $\delta\Psi$, while keeping the imposed boundary drive and the physical control conditions unchanged. Linearization keeps terms proportional to $\delta\Psi$ and discards terms quadratic or higher in its size. It is a local approximation about this one stationary state.

A normal mode separates spatial structure from time dependence:
\begin{equation}
\delta\Psi(t)=\varepsilon\,\Real\{x\exp(\lambda t)\},\qquad
\lambda=\gamma+i\omega.
\label{eq:reading-mode}
\end{equation}
The vector $x$ specifies the disturbance of every field at every mesh degree of freedom. Its height component shows where the membrane bends; its velocity and tension components show the accompanying response. The real part $\gamma$ is the growth rate, and $\omega$ is the angular frequency. A negative growth rate means decay, a positive one means instability, and a zero one means neutral stability at linear order. If $\omega\ne0$, the real disturbance oscillates as it grows or decays, with period $2\pi/|\omega|$. With a complex eigenvector, the real and imaginary spatial patterns are a quarter-period apart; plotting only one part does not show the whole cycle.

The \emph{leading} physical mode is the one with the largest growth rate among the admissible finite eigenvalues. A static buckling onset occurs when a real eigenvalue passes through zero. A Hopf onset occurs when a complex-conjugate pair crosses the imaginary axis at a nonzero frequency. A rigid translation allowed by the boundary conditions can instead produce a neutral mode for every drive; that mode must be recognized before interpreting zero as a buckling threshold. Eigenvector amplitude is arbitrary in a linear calculation, so a plotted mode is a shape of a disturbance, not a prediction of its eventual depth.

\subsection{What the finite-element residual represents}
The surface is divided into triangles, and each unknown field is represented by basis functions with unknown coefficients. A $P_1$ field is linear within a triangle; a $P_2$ field is quadratic and has additional degrees of freedom. The governing equations are imposed in weak form: multiply each equation by a test function, integrate over the domain, and integrate derivatives by parts where needed. Doing this for all basis test functions gives a vector $\mathcal R$ of residuals. At a converged stationary solution, every component is approximately zero.

Weak form is especially useful for bending, which would otherwise require fourth derivatives of height. Independent gradient and curvature fields replace that high-order scalar equation by coupled lower-order equations. Their values are not new physical freedoms: separate residual rows constrain them to equal the derivatives of the surface. Likewise, tension acts as a Lagrange multiplier enforcing membrane-area incompressibility. A perturbation must satisfy these constraints as well as force balance. Automatic differentiation of the assembled weak expressions includes geometry, boundary terms, and all these couplings; deleting one of them changes the stability problem.
''',
'sec:validation': r'''
\subsection{Why several different verification problems are needed}
A small algebraic residual says that the software solved its discrete equations accurately; it does not say that those equations describe the intended physical problem. The checks here therefore separate four possible errors. Exact spectra test the sign, time normalization, and spatial discretization. Mesh refinement tests whether results change as the discrete space is enlarged. A second formulation tests whether the same physical limit is recovered without reusing the whole operator. Time integration tests whether a predicted mode actually grows or decays at the predicted rate. Agreement across these checks is much more informative than tightening one eigensolver tolerance.

The annulus is chosen because a flat membrane has a known bending--tension relaxation law and separable angular modes. The cylinder is chosen because its leading instability is oscillatory and its velocity is constrained by pressure. It exercises aspects of the generic method that a purely real membrane spectrum cannot test. The detailed cylinder confirmation below uses the successful newer implementation; its growth-rate comparison is made in the early linear interval, before nonlinear vortex shedding changes the frequency.
''',
'sec:force': r'''
\subsection{What it means for the protein to be force-free}
A rigid protein has one vertical position $h$, shared by the whole rim. There are two distinct choices. Prescribing $h$ allows the external apparatus to exert whatever vertical reaction is needed. Leaving $h$ free requires a new equation stating that the \emph{total} vertical force on the protein is zero. Merely freeing each rim height node does neither: it can allow the rim to deform and leave the global rigid-body force unbalanced.

The distinction is important even when the base membrane is flat. A missing condition may be invisible in that stationary geometry but become active in a perturbed shape. If the rim can do unaccounted work as it moves, the spectrum measures the stability of a problem with an unintended force or compliance. The corrected formulation ties the rim to one rigid height and supplies its corresponding force equation. The force comparisons below establish both how that reaction is evaluated and why a boundary line integral alone was not adequate in the inspected discretization.
''',
'sec:ring': r'''
\subsection{The numerical experiment and its two controls}
For the static experiment, the outer ring stays fixed and no tangential flow is applied. At each selected protein height, the stationary solver relaxes the remaining membrane shape at the prescribed contact slope. The resulting family supplies profiles, bending energy, membrane force, and a fixed-height spectrum. Sweeping height is \emph{continuation}: the previous solution provides an initial guess for the next, allowing a whole branch to be followed rather than solving unrelated initial-value problems.

A height-controlled experiment perturbs the membrane while preventing the protein from moving vertically. A force-controlled experiment prescribes an external pulling force and allows the protein to move. The latter admits an extra disturbance and can therefore become unstable even when every tested fixed-height shape is stable. The force--height curve tells us when that additional degree of freedom loses its restoring force; the relaxed-energy calculation below makes the statement precise.
''',
'sec:tubes': r'''
\subsection{Why a peak force and a tube force are different}
To create a tether, the membrane first has to form a highly curved neck out of the initially confined shape. Once a long cylindrical tube exists, adding length mainly adds a repeated cylindrical segment. These operations cost different amounts of work. The maximum holding force along the formation branch measures the difficult nucleation stage; the long-tube plateau measures the incremental cost of extending an established tether. Consequently, a large force barrier in a small ring is compatible with a much smaller steady tether force.

A height graph $z(r)$ cannot continue through a vertical tangent because its derivative becomes infinite. Arclength coordinates follow the curve itself and stay meaningful when the tangent rotates beyond vertical. This is why a new geometric formulation is needed here, rather than just a finer version of the same height mesh. It also explains why evidence from this axisymmetric no-flow calculation must be distinguished from the later asymmetric flowing-membrane fold.
''',
'sec:flowhistory': r'''
\subsection{Separating a numerical check from a physical correction}
The first flow calculations used the original rim implementation. They supplied a stationary base state, a spectrum, and a time evolution that could be compared with one another. Their agreement was useful: it showed that the stability tool was predicting the dynamics of those implemented equations. It could not detect an incorrect physical boundary condition merely by agreeing with a time integrator using the same condition.

The force audit supplied that independent physical test. After changing the protein condition, the stationary and spectral calculations had to be repeated for the corrected problem. A threshold near sixteen and a threshold near twenty-seven are thus not two discretizations of the same model. The ledger below identifies the boundary condition attached to each result so that a successful early solver check is not mistaken for the final physical prediction.
''',
'sec:buckling': r'''
\subsection{How a tangential flow produces a normal instability}
There is no requirement that the imposed flow push directly upward or downward. The anchored inclusion resists tangential motion, so the base flow redistributes in-plane stress. The membrane can maintain a flat geometry under this stress. However, a slightly bent patch exposes that stress to a normal displacement. Tensile stress tends to straighten the displacement, whereas compressive stress can amplify it. Flow therefore changes the \emph{restoring force of a disturbance}, even when it does not deform the perfect base state.

The calculation is deliberately split into two steps. First solve the flat in-plane flow and its tension/stress fields. Then insert those fields into the normal stability operator and find when its least stable mode grows. The independent plate model repeats the second step with a simpler set of unknowns. Its agreement with the full mixed surface calculation is significant because it identifies stress-induced buckling rather than an unexplained mode of the larger algebraic system.
''',
'sec:geometry': r'''
\subsection{What is held fixed in a domain study}
Changing the outer box changes where velocity, height, slope, and tension constraints act. It changes both the base stress field and the admissible buckling shapes. A domain study must therefore specify the physical parameters held fixed. Keeping a dimensionless tension involving box size fixed generally changes the dimensional tension when the box is enlarged; keeping the dimensional tension fixed is a different experiment.

The adjoint calculation asks a related but more targeted question: if a small force is added at a given location, how much does the threshold change? The direct mode shows the disturbance that becomes unstable; the adjoint mode tells how strongly a perturbation of the equations excites or changes that mode. Together with the response of the base flow, it yields a sensitivity map without performing a complete new threshold search at every possible force location.
''',
'sec:multi': r'''
\subsection{Why velocity and drag give different comparisons}
Two arrangements exposed to the same far-field speed need not experience the same force. Proteins shield one another, and their wakes and stressed regions overlap. Critical velocity can therefore vary considerably with arrangement. Drag integrates the tangential stress transmitted to the inclusions and provides a more direct measure of the loading responsible for compression.

The computation first finds each arrangement's base flow and onset, then evaluates its in-plane drag at that onset. Plotting those forces against the bending scale tests whether the arrangements fail under comparable integrated loading. For a finite group, the relevant recorded comparison uses the sum of the forces on all inclusions. For an infinite periodic lattice, the reported dilute comparison is the force per cell or per inclusion. These quantities must remain distinct even when both are numerically close to $\kappa/r_0$.
''',
'sec:snap': r'''
\subsection{What linear onset leaves undecided}
At a real zero eigenvalue, one shape disturbance stops relaxing. Linear theory cannot determine whether that disturbance saturates at a small amplitude. To answer this, the calculation follows nearby stationary deformed states and expands the equations in the amplitude of the neutral mode. In a perfectly up--down symmetric problem, changing the sign of height must reverse the amplitude equation, so its leading nonlinear term is cubic.

Write its local structure as $\dot A=\alpha\epsilon A+\beta A^3+f$, where $\epsilon$ measures drive above the perfect onset and $f$ represents a symmetry-breaking imperfection. With the stated dynamic sign convention and $\alpha>0$, a negative cubic can arrest growth above onset; a positive cubic reinforces growth and creates a small unstable branch below onset. This latter, subcritical case explains why a slowly varied drive can cause a finite departure instead of a continuous small deformation. The cubic theory describes the local branch and its barrier; it does not supply the remote stable branch needed to prove a complete hysteresis loop.

The coefficients here are calculated from residual derivatives and direct/adjoint modes, not obtained by fitting the continuation curve. Comparing the resulting prediction with independently continued states is therefore a quantitative test of the theory. Normalizing an eigenvector differently changes the numerical amplitude and coefficient values, so meaningful comparisons use the same normalization or normalization-independent fold predictions.
''',
'sec:collapse': r'''
\subsection{What is evolved after the snap}
The dynamical calculation begins close to the steady branch, changes the drive or applies a small disturbance, and advances the coupled membrane equations in time. Unlike continuation, it does not constrain the system to remain stationary. It can therefore show how a disturbance passes through the slow region near a lost equilibrium and enters a strongly nonlinear collapse.

For a force-free inclusion, the protein height is not a prescribed trajectory. At each step it is adjusted until the consistent vertical reaction vanishes, while the membrane shape and flow respond to that adjustment. The rising protein and deep upstream pit are thus simultaneous outputs of the coupled solve. They are not produced by pulling the protein upward externally. Late-time pictures are assessed together with force-balance and mesh-quality diagnostics because large slopes can make an apparently dramatic state less numerically trustworthy.
''',
'sec:fold': r'''
\subsection{Replacing the height graph by an embedded surface}
In Monge coordinates, each horizontal point $(x,y)$ has exactly one height. This representation is efficient before folding but rules out an overhang by construction. A parametric surface instead stores a position $\bm X(\xi^1,\xi^2)$ in three-dimensional space for each mesh coordinate. The mesh coordinates label points of the numerical surface; they need not be horizontal positions. Tangents, metric, curvature, and the normal are then computed from $\bm X$.

Physical membrane motion and mesh redistribution are different operations. Tangential mesh motion can change which coordinates describe a shape without changing that shape. This coordinate freedom is called a mesh gauge. A useful gauge should keep elements well shaped while preserving the physical evolution. In the tested extension, the surface can turn through vertical, but element stretching and anisotropy eventually make the calculation unreliable. Remeshing must also transfer curvature, constraints, and forces consistently; interpolating positions alone is not enough.
''',
'sec:flutter': r'''
\subsection{Why oscillations can appear without appreciable inertia}
A purely dissipative membrane with a conservative restoring operator relaxes through real growth rates. Tangential drag against a surrounding medium changes the in-plane force balance and makes the normal response non-conservative. Different spatial disturbances can then feed into each other with a phase lag, producing complex eigenvalues even in the overdamped shape limit. Flutter here is consequently not an ordinary mass--spring resonance.

The computation searches for both real crossings and complex-pair crossings as friction and tension are varied. After identifying an oscillatory onset, time integration tests whether growth ends in sustained waves or in collapse. A below-threshold restart then tests whether the observed oscillation persists when the drive is reduced. This sequence distinguishes a linear oscillatory instability, a saturated dynamic state, and the tested absence of hysteresis; none follows automatically from the other.
''',
'sec:proteins': r'''
\subsection{The extra feedback loop introduced by mobile proteins}
A curvature-coupled composition field changes the membrane's preferred curvature. Bending the membrane also changes the chemical potential of that field. Shape can therefore recruit a composition deviation, which in turn favors further bending. Diffusion tends to relax chemical-potential differences, while imposed flow transports the field through the patch. The competition adds a second relaxation process to the shape dynamics.

The field $\varphi$ is a signed deviation from a reference composition. It is not an absolute number density constrained to be positive. The rigid anchored obstacle remains part of the geometry; $\varphi$ describes mobile material on the surrounding membrane. The symbol $m$ denotes its chemical potential, whereas the mixed geometric variable $\mu$ denotes mean curvature. Keeping these distinctions explicit is necessary to interpret the later coupled eigenmodes and pit compositions.
''',
'sec:transport': r'''
\subsection{Separating phase separation from residence time}
At rest, a perturbation has time to grow into domains if the coupled free energy favors demixing. In an open flowing patch, the same perturbation may leave through the outflow before it grows appreciably. A disappearance of domains in the observed patch can therefore reflect transport as well as a change of local stability. The calculations compare spectra, spatial patterns, and boundary treatments to distinguish these effects.

The P\'eclet number expresses the competition between transport and diffusive relaxation on a specified length scale. Small mobility corresponds to slow chemical relaxation relative to advection; larger mobility allows composition to respond to curvature while the material is still in the patch. Because the chemical potential contains gradient and curvature contributions, its relaxation rate depends on wavelength. One mobility parameter is thus not a wavelength-independent molecular diffusion coefficient.
''',
'sec:merger': r'''
\subsection{Following modes rather than naming them from one picture}
A mode dominated by height in one parameter regime can acquire a strong composition component in another. Conversely, a demixing-like mode can become strongly geometric. The calculation follows eigenvalues and eigenvectors as parameters vary rather than assigning permanent labels from one snapshot. Two real branches approaching one another and becoming a complex pair indicate coupling within a common mode family.

Near a candidate double-zero point, both restoring and relaxation terms of a reduced two-variable dynamics can become small. Eliminating one variable can produce a second-order-in-time equation for the other, even though the underlying model has no added mechanical inertia. This explains the relevance of a Bogdanov--Takens-type description. Establishing its exact mathematical classification would require a resolved double-zero eigenvalue, its generalized eigenvector structure, and reliable nonlinear coefficients. The archived evidence motivates that description but does not complete all of those tests.
''',
'sec:wake': r'''
\subsection{How a source changes the perfect buckling experiment}
The symmetric reference problem has no preferred sign of the critical amplitude. A source at the protein rim creates a stationary advected composition wake and its associated curvature bias. One must first solve this new base state, then classify its disturbances. The source does not simply change an eigenvalue while leaving the old flat base intact.

Projecting the wake-induced forcing onto the critical mode gives an imperfection term in the same amplitude equation used for a nonzero contact slope. This supplies a specific prediction: a small source can move the fold before the perfect linear threshold. Comparing the projected prediction with source-dependent continuation tests whether the wake acts as this kind of imperfection. It also separates earlier onset from weaker nonlinear feedback; the source can advance the snap while making the backward branch steeper.
''',
'sec:lattice': r'''
\subsection{Why one periodic cell can describe collective disturbances}
An infinite repeating array has no arbitrarily distant outer edge. The stationary problem is solved on a cell containing one protein, with fields matched across opposite cell boundaries. A collective disturbance need not repeat identically from cell to cell: it can acquire a phase $\exp(i\bm q\cdot\bm a)$ after translation by a lattice vector $\bm a$. Bloch boundary conditions encode that phase, so a single-cell calculation can represent disturbances whose wavelength spans many cells.

At $\bm q=0$, a uniform vertical translation of a freely translating membrane/inclusion array changes no curvature. It is therefore a neutral mode independently of drive. The physical question is whether a \emph{slowly varying} height displacement grows. Following the branch from this translation mode at small nonzero $\bm q$ and expanding its growth rate is the appropriate long-wave test. Reporting the smallest absolute eigenvalue at $\bm q=0$ would always report zero and would miss that distinction.
''',
'sec:terraces1d': r'''
\subsection{From a microscopic cell to a slowly varying slope}
The cell spectrum gives onset but not the response of a finite tilt. For the nonlinear closure, the cell is inclined by a prescribed mean slope and its membrane/flow fields are solved subject to periodicity in the tilted geometry. The cell-averaged stress response is sampled over several small slopes and represented by a polynomial. Those coefficients enter a coarse-scale equation for a slope field varying over many cells.

This is homogenization: the microscopic protein-scale fields are replaced by constitutive coefficients for a much longer-wave shape. It assumes scale separation and the tested tilted-cell closure. A terrace is a region of approximately constant slope joined to another slope by a transition layer; integrating the slope gives piecewise ramp-like membrane height. The predicted saturation depends on the nonlinear stress coefficients, so changing the fit window or using an uncontrolled large slope can change the inferred criticality.
''',
'sec:terraces2d': r'''
\subsection{Three distinct tests of a persistent pattern}
A one-dimensional terrace could break into transverse structure when allowed to vary in the second direction. Two-dimensional coefficients and simulations test this possibility by allowing perturbations both along and across the flow. Persistence in a periodic domain still does not imply persistence in a finite protein patch: the wave or terrace may drift out of the patch.

A localized disturbance is \emph{convectively} unstable when it amplifies while moving, but eventually leaves a fixed observation point. It is \emph{absolutely} unstable when it grows at that fixed point. Thus a periodic-array linear threshold and a finite-patch sustained-pattern threshold answer different questions. The patch calculation compares spatial growth with drift to find when a source can remain visible, using the reduced long-wave model rather than a new fully resolved patch of microscopic cells.
''',
'sec:discussion': r'''
\subsection{The logical chain of the result}
The workflow begins with a trustworthy relation between residuals and disturbance growth. The force audit then determines which boundary-value problem is physically intended. The independent flat-plate reduction explains its onset as loss of restoring stiffness through upstream compression. Continuation and residual projections establish a subcritical local branch and predict how imperfections move its fold. Time integration follows the departure beyond that local theory, while parametric geometry identifies the start of folding and its numerical limit.

Friction, mobile composition, and periodic arrays are three extensions of that chain, each adding a distinct ingredient: non-conservative stress, a transported conserved field, or collective spatial organization. They do not erase the earlier requirements of consistent force balance, specified boundaries, and adequate numerical resolution. The synthesis below therefore separates firmly supported conclusions from unresolved endpoints and robustness checks rather than assigning the same confidence to every panel.
'''
}

subsection_guides = {
'Mixed fields and geometric constraints': r'''
Here $z$ is height, $\bm v$ is tangential velocity, and $w$ is normal velocity. The normal and tangents convert force components between the surface and laboratory directions. The metric $g_{ij}$ measures distances and areas on the curved surface; $\mathcal H$ measures local bending and $\mathcal K$ the product of principal curvatures. These quantities depend on height and its derivatives, so perturbing height also perturbs the geometry appearing in the flow equations. The surface-incompressibility condition includes $-2\mathcal H w$ because normal motion changes local area on a curved membrane.
''',
'Dynamic residual, mass matrix, and sign convention': r'''
The matrix called ``mass'' collects coefficients multiplying time derivatives; it need not represent material inertia. A shape kinematic equation and a conserved composition equation also contribute to this matrix. To derive the spectrum, substitute $\Psi=\Psi_*+\delta\Psi$ into the dynamic residual. Since $\dot\Psi_*=0$, the variation of $\mathsf B(\Psi)$ multiplies a zero base-state derivative and does not contribute at first order. Thus
\begin{equation}
\mathsf B_*\delta\dot\Psi+\mathsf J\delta\Psi=0,
\qquad \mathsf J=D\mathcal R(\Psi_*).
\end{equation}
Inserting $\delta\Psi=x\exp(\lambda t)$ gives $-\mathsf Jx=\lambda\mathsf B_*x$. The minus sign is physical: a positive restoring residual produces a negative relaxation rate. Newton's method uses the same $\mathsf J$ in $\mathsf J\Delta\Psi=-\mathcal R$ to find a stationary state; that iterative convergence is not a time evolution and cannot determine physical stability.

Rows for pressure/tension and geometric constraints have no independent time derivative. Their zero mass rows require the perturbation to satisfy the corresponding linearized constraint. Giving those rows an artificial mass would introduce artificial relaxation modes. The singular generalized problem is therefore intentional, not a numerical defect to be removed by filling its diagonal.
''',
'Boundary reduction, shift-invert, and spectral coverage': r'''
A perturbation changes the solution, not the imposed experimental setting. A velocity fixed to a nonzero value in the base problem has a \emph{zero} perturbation on that boundary. A rigid free protein permits a shared height perturbation, while a clamped protein permits none. The prolongation matrix maps these admissible reduced degrees of freedom into the full mesh vector; reducing both matrices with the same map keeps the constraints consistent.

The matrices are too large to compute every eigenvalue. Shift-invert searches near a chosen complex number $s$ by applying $(\mathsf A-s\mathsf B)^{-1}\mathsf B$. A finite eigenvalue $\lambda$ becomes $1/(\lambda-s)$, making modes near $s$ easier to extract. This requires repeated sparse linear solves, not an explicit dense inverse. A mode nearest one shift need not have the largest real part in the whole spectrum. Multiple shifts and branch tracking are therefore essential, especially when a growing oscillatory mode lies far from the real axis.

An essential boundary condition fixes an allowed field value, while a natural condition supplies the force or moment arising from boundary work in the weak equations. Slope constraints may be enforced weakly through penalty/Nitsche terms, whose differentiated contributions must also remain in the perturbation problem. The penalty coefficient is a numerical enforcement parameter, not a membrane material property.

The practical sequence is: converge a base state; assemble its differentiated residual and time form; impose homogeneous perturbation conditions; reduce rigid-body constraints; search overlapping spectral neighborhoods; check the returned residuals and mode identities; then continue the relevant growth rate through zero as the drive changes. A small eigenpair residual checks the algebraic solve. Agreement under mesh refinement and independent verification is still needed to check its physical accuracy.
''',
'Flat annulus: exact spectrum and mesh refinement': r'''
For a flat membrane, bending opposes short-wave deformation with a fourth spatial derivative, and positive tension supplies a second-derivative restoring term. On an infinite plane, a Fourier wave would decay at $\lambda=-(\kappa k^4+\sigma_0k^2)/\zeta$. The annular boundaries instead select radial wave numbers through four endpoint conditions: height and slope at both circles. Angular dependence separates into integer $m$ values; cosine and sine modes with $m\ge1$ have the same rate because a circular annulus has no preferred orientation. The radial solution combines ordinary Bessel functions $J_m,Y_m$ and modified Bessel functions $I_m,K_m$, the two types associated with the factorized fourth-order radial equation. Their coefficients are fixed by the boundary conditions, not fitted to the finite-element eigenvalues. Recovering both the rates and angular degeneracy checks more than reproducing a single fitted decay time.
''',
'Cylinder base state and spatial resolution': r'''
The stationary solver finds a symmetric wake even where that wake is unstable to time-dependent disturbances. This is exactly the state about which the eigenproblem must be built. If one instead linearized around an already shedding instantaneous flow, the simple exponential normal-mode ansatz would no longer classify the stationary wake. Drag and recirculation length check the accuracy of the base state before using its Jacobian; refinement of the onset then checks the disturbance calculation.
''',
'Complex spectra, competing branches, and onset': r'''
For each Reynolds number, a spectrum contains many modes: most decay, and several branches may change order as parameters vary. The wake pair is identified by its spatial pattern and continuous trajectory, not just by taking whichever eigenvalue happens to be returned first. Its growth rate crossing zero is the shedding onset; its imaginary part there predicts the small-amplitude oscillation frequency. The eigenvalue trajectories below should therefore be read horizontally against the zero-growth line, with the frequencies used to distinguish competing branches.
''',
'Independent confirmation by time stepping and vortex shedding': r'''
The independent test gives the base flow a weak kick and watches what happens after the kick is removed. A stable wake should forget it; an unstable wake should amplify some component of it. The disturbance need not initially equal one eigenmode: several modes contribute during the early transient. Fitting only after that transient, and before saturation, isolates the predicted linear behavior. The nonlinear vortex street is the later outcome and can have a different frequency.
''',
'Virtual work and the missing boundary moment': r'''
Virtual work asks how the relaxed membrane energy changes when the rigid protein is displaced by a small amount. If $E_*(h)$ is the minimized energy at each fixed height, the membrane force is $F_m=-\dd E_*/\dd h$. Numerically, this can be compared with energy differences or with the weak residual tested by a displacement representing rigid vertical motion. Relaxation of the other fields contributes no first-order energy change at equilibrium, because their stationary variations vanish.

Bending transmits both forces and moments across a boundary. A vertical displacement is not a spatially constant \emph{normal} displacement on a tilted surface: its normal component is multiplied by $n_z$. Consequently, the normal virtual displacement has a derivative along the membrane, and a boundary moment can perform work. Omitting that work gives an incomplete force diagnostic. Even after restoring it, poorly resolved boundary curvature derivatives can prevent a line integral from converging. The tests below distinguish those two failures.
''',
'A conserved vertical stress and consistent reactions': r'''
Instead of sampling high derivatives exactly on the protein rim, the conserved-stress method measures the force transmitted across an enclosing annulus. The smooth cutoff decreases from one near the protein to zero outside that annulus, so its gradient distributes the flux measurement over a resolved area. At a force-balanced stationary state, moving the annulus should not change the answer: no net external vertical load has been inserted between the two contours. A spread between cutoff choices is therefore a useful consistency diagnostic, particularly once the shape becomes steep.
''',
'Relaxed energy and the force-controlled zero mode': r'''
The variables $q$ below collect all deformable shape degrees of freedom other than the protein height $h$. At fixed $h$, they relax to $q_*(h)$. Allowing $h$ to vary gives those same variables an opportunity to readjust and soften the response. The term subtracted in the Schur complement is precisely this relaxation-induced softening. An external dead load $P$ has potential $E-P h$, so stability in the newly allowed height direction requires the relaxed energy curvature $E_*''=\dd P/\dd h$ to be positive. At a force maximum it vanishes; beyond it, a small upward motion reduces the required holding force, leaving the imposed load in excess and driving further motion.
''',
'Arclength formulation and endpoint conditions': r'''
The unknown tangent angle $\psi$ gives $\dd r/\dd s_a=\cos\psi$ and $\dd z/\dd s_a=\sin\psi$. A vertical wall corresponds to $\psi=\pi/2$, where these derivatives remain finite. The remaining equations impose bending equilibrium and tension through curvature and force multipliers. Both endpoint positions and tangent angles are prescribed, but the total arclength is solved for; the additional condition below closes that free-length problem. Adaptive branch continuation changes the control gradually so the boundary-value solver can stay on the same shape family through a steep neck.
''',
'Exact plate operator about the flat base state': r'''
The stress tensor $T_{ij}$ contains both tension and viscous in-plane stress; local scalar tension alone is not the complete loading. About a flat surface, normal bending couples to that tensor through $T_{ij}z_{,ij}$. For uniform isotropic stress $T_{ij}=\tau\delta_{ij}$, a Fourier disturbance gives
\begin{equation}
\lambda(k)=\frac{-\kappa k^4-\tau k^2}{\zeta}.
\end{equation}
Positive $\tau$ damps every wavelength; negative $\tau$ makes sufficiently long waves grow, while bending still suppresses very short waves. Around a protein the stress is nonuniform and anisotropic, and boundaries restrict which disturbances fit. The eigenproblem determines the actual mode rather than substituting a uniform local compression estimate.
''',
'Independent mixed discretization and energetic stiffness': r'''
The quadratic functional $\mathcal Q$ is the second-order energetic cost of a normal shape disturbance when the no-body-force base stress is divergence-free and boundary work is treated consistently. Its bending and positive-tension parts resist deformation. The flow-induced stress part can be negative for an upstream mode. At threshold those contributions cancel on the critical admissible shape. Introducing a mixed Laplacian variable lets the independent solver evaluate this fourth-order problem with ordinary finite elements; agreement does not rely on differentiating the full IRENE geometry again.
''',
'Direct and adjoint forcing sensitivity': r'''
Differentiate the eigenproblem with respect to a parameter. Multiplication by the left eigenvector removes the unknown derivative of the right eigenvector, leaving the projection formula below. The normalization $y^\dagger\mathsf Bx=1$ fixes the scale of that projection. For a spatial body force, the parameter changes the base flow before it changes the normal operator, so that intermediate response must be included. The final sensitivity is read as the sign and magnitude of the threshold shift per small applied force, not as the amplitude of the buckling mode itself.
''',
'Residual derivatives and slaved second-order fields': r'''
The amplitude $A$ measures displacement along the neutral mode $x_1$. Most other modes still decay at onset, so they adjust to this displacement rather than becoming independent slow variables. Their leading adjustment is $A^2x_2$, found by solving a linear system forced by the quadratic residual derivative. Substituting that adjustment back into the cubic order accounts for feedback from changes in flow, tension, and geometry. A bare third derivative along $x_1$ would miss this feedback.

At cubic order, the singular linear operator cannot be inverted along its neutral direction. Projection onto the left nullvector supplies the solvability condition, which becomes the amplitude equation. Fixing the complementary-space solution and mode normalization prevents arbitrary pieces of $x_1$ from contaminating $x_2$. The procedure is a local reduction of the full constrained residual, not a separately postulated phenomenological cubic.
''',
'Fold construction and Koiter sensitivity': r'''
A fold is a point where two stationary solutions meet and the restoring derivative with respect to amplitude vanishes. For the local equation $g(A)=\alpha\epsilon A+\beta A^3+f=0$, impose both $g=0$ and $\partial_Ag=0$. Eliminating $\epsilon$ gives $A_f^3=f/(2\beta)$ and $\epsilon_f=-3\beta A_f^2/\alpha$. Hence the advance of the fold scales as $|f|^{2/3}$, while its deflection scales as $|f|^{1/3}$. This is the origin of Koiter's two-thirds law here: a small symmetry-breaking bias can move failure much more than a linear-in-bias estimate would suggest. Its prefactor is determined by the calculated projection coefficients and by how the physical contact slope enters $f$.
''',
'BDF2 stepping and a force-free height solve': r'''
BDF2 replaces a time derivative by $(3\Psi^{n+1}-4\Psi^n+\Psi^{n-1})/(2\Delta t)$ after a first-order startup. The new state is then obtained from the coupled discrete equations, retaining the algebraic constraints. An implicit step helps handle rapid bending relaxation, but convergence of a nonlinear solve at each step still has to be monitored. The protein-height iteration adds a scalar force-balance solve to this procedure rather than assigning independent inertial motion to the protein. Time-step agreement and consistent cutoff forces test different aspects of the trajectory; neither alone guarantees mesh adequacy near a fold.
''',
'Follower forcing and the divergence--flutter map': r'''
Without body force, $\nabla\cdot\mathsf T=0$, so $\mathsf T:\nabla\nabla z$ can be written as $\nabla\cdot(\mathsf T\nabla z)$. With $\nabla\cdot\mathsf T+\bm f_0=0$, the identity instead reads
\begin{equation}
\mathsf T:\nabla\nabla z=\nabla\cdot(\mathsf T\nabla z)+\bm f_0\cdot\nabla z.
\end{equation}
The extra first-derivative term cannot be dropped. It is part of the non-conservative response that permits flutter. Normal friction $\zeta$ controls shape relaxation, whereas tangential Brinkman friction changes the base flow and this stress identity; varying one is not equivalent to varying the other. The map compares which real or complex crossing occurs first, not merely how fast the same buckling mode evolves.
''',
'Energy, density convention, and nonlinear virtual power': r'''
The chemical potential is the variational derivative of the free energy with respect to composition. A conserved diffusive flux is directed down its gradient, $\bm j=-M\nabla m$, so local conservation contains $M\Delta m$ as well as advection and any source. If the free energy is locally nonconvex, this transport can amplify composition differences instead of simply smoothing concentration. Gradient energy suppresses arbitrarily short domains, and curvature coupling transfers this feedback to shape. Deriving both mechanical forces and chemical transport from the same functional is needed for an energy-consistent no-flow model.
''',
'Analytical dispersion and relative advection': r'''
In a uniform plane, Fourier transformation reduces the coupled height/composition equations at each wavevector to a small matrix. Its diagonal entries represent their separate relaxation and advection; its off-diagonal entries represent curvature--composition feedback. The eigenvalues tell whether the combined disturbance decays, demixes, bends, or travels. A common advection speed adds an imaginary frequency without changing growth in an infinite uniform medium. Relative advection and the open inclusion geometry can change the coupling itself, which is why this analytical check does not replace the finite-domain calculation.
''',
'A candidate degenerate double-zero organizing centre': r'''
A pair becoming complex is not by itself proof of a Bogdanov--Takens point. The special organizing centre requires both eigenvalues to approach zero with the appropriate degeneracy. Moreover, coefficients computed extremely close to it can be small differences of larger projected terms, making their signs sensitive to numerical resolution. The discussion below reports the candidate and the limitations of its unfolding rather than using a generic normal form to certify every nearby sector.
''',
'Bloch perturbations and density-controlled thresholds': r'''
Writing the periodic part of a perturbation separately replaces derivatives by $\nabla+i\bm q$. Solving at several small $q$ values then estimates the coefficients in a long-wave expansion. The $q^2$ coefficient determines whether very long disturbances grow; a stabilizing higher-order term limits short wavelengths within that expansion. An imaginary term describes drift. The area fraction $\Phi$ of rigid discs sets cell spacing; it is distinct from the signed mobile composition $\varphi$ used in the preceding sections.
''',
'A convected slope equation and density-dependent criticality': r'''
The slope equation combines a destabilizing long-wave response, regularization from bending, nonlinear stress from tilted cells, and drift. Its structure resembles conserved phase-separation dynamics for slope, but the ``phases'' are different inclinations of the membrane, not different protein concentrations. Dense-cell coefficients can favor finite stable slopes; dilute-cell coefficients can reinforce growth instead. Simulating this equation tests consequences of the closure and permits large-scale terrace dynamics that would be costly to resolve cell by cell. It does not establish the validity of that closure at arbitrary amplitude or wavelength.
''',
'Convective versus absolute instability': r'''
The spatial calculation asks whether amplification can overcome removal by drift. A wavepacket can grow strongly in its moving frame and still decay at a fixed point because its centre has left. This explains why the finite-patch onset can require substantially more drive than the periodic long-wave threshold. If the selected transition-layer width is comparable to or smaller than a microscopic cell, the assumed scale separation breaks down. The reported factor-of-two result is consequently a reduced-model prediction to be tested against a resolved finite array.
''',
'Dimensional force, velocity, and time scales': r'''
The computational scales are converted only after the dimensionless calculation. Bending divided by inclusion radius gives a force scale; dividing by membrane viscosity and a length gives a velocity scale. Choosing the protein radius or the outer patch size changes that velocity unit and the associated dimensionless drive. Frequencies additionally depend on the specified normal relaxation coefficient, so an angular frequency in numerical time units is not yet a biological frequency without calibrating that coefficient. The examples below state the chosen parameters explicitly.
'''
}

subsection_readouts = {
'Flat annulus: exact spectrum and mesh refinement': r'''
The coarse/fine comparison therefore tests the entire normal-mode construction on the annulus: geometry, bending stiffness, tension, time normalization, and clamped endpoint conditions. The smaller discrepancy on refinement supports convergence of these sampled modes. It does not automatically certify a different boundary condition or a surface with a nearly vertical wall.
''',
'Compression and critical-mode geometry': r'''
The mechanism is read from three outputs together: the base-flow stress identifies the compressed upstream region; the critical height eigenvector localizes the impending deformation; and the energy decomposition shows that compressive work cancels bending and tension at onset. Their spatial and quantitative agreement is what supports the column-buckling analogy. It is not an inference from a pit snapshot alone.
''',
'Total finite-patch drag versus dilute per-protein drag': r'''
The scale $\kappa/r_0$ follows dimensionally because $\kappa$ is an energy and $r_0$ is a length. It suggests the force needed to compete with bending near an inclusion-sized structure, but dimensional analysis cannot supply its numerical prefactor. The sampled configurations provide that empirical prefactor and its range of validity. Dense arrays and strong tension introduce additional scales, so the observation should not be promoted to an unrestricted universal force threshold.
''',
'Saturation and the below-threshold restart': r'''
A sustained trace means that the tested trajectory has reached bounded oscillations instead of the monotonic deepening seen in the unscreened case. Decay on restarting below onset supports the absence of hysteresis in that tested interval. It does not amount to a Floquet stability analysis of the periodic orbit against every perturbation, nor to a complete survey of other attractors.
''',
'Final mobility regimes with absorbing outflow': r'''
The regime labels describe the leading mode of each specified force-free base problem. ``Travelling'' means a nonzero imaginary part and an evolving spatial phase; it does not guarantee a persistent nonlinear travelling wave. The strongly reduced threshold in the fast-relaxation regime comes from cooperative shape/composition feedback. Subsequent nonlinear merger and wake calculations use their stated clamped conditions, so their precise thresholds should not be spliced into this force-free sweep.
''',
'Tilted cells, rigid-rim geometry, and frozen-flow closure': r'''
The fit is an additional approximation beyond the finite-element solve: a polynomial compresses several numerical cell responses into a constitutive law. Its linear coefficient should recover the appropriate small-slope limit, while its nonlinear coefficients are meaningful only over a supported slope range. Differences between frozen-flow and fully coupled cell responses are therefore model differences, not simply eigensolver residual errors.
''',
'Established conclusions and remaining scientific questions': r'''
No new numerical sweeps were performed to prepare this explanatory revision. Numerical values and figures refer to the archived computations reviewed for the report. The expanded account explains their construction and interpretation; the outstanding checks remain necessary before a final manuscript claims broader robustness or a resolved late-time endpoint.
'''
}

subsection_guides.update({
'Geometry, material parameters, and control variables': r'''
Bending rigidity penalizes curvature, tension penalizes added area, and surface viscosity resists tangential strain rate. Normal friction supplies the dissipation needed to turn a normal restoring force into a shape-relaxation rate. These parameters have different roles: increasing normal friction slows an otherwise unchanged overdamped instability, whereas changing tension alters the restoring stiffness and can move the onset. The dimensionless drive $\SL$ compares flow-induced loading with bending on the stated outer length $L$; $\Gamma$ compares tension with bending on that same length. They are control parameters, not eigenvalues.

Throughout, a prime on a perturbation means a small change from the base state; a prime on a one-variable energy or force means a derivative with respect to that variable. Bold $\bm\omega$ below is a geometric gradient field, while unbold $\omega$ in a complex eigenvalue is a temporal angular frequency. The growth rate is the real part of $\lambda$; the local tension field is $\sigma$. In the cylinder subsection only, $\sigma_{\rm DNS}$ and $\sigma_{\rm eig}$ label fitted and predicted growth rates, and $\mathrm{Re}$ denotes Reynolds number rather than the real-part operator $\Real$.
''',
'Elementary exact tests': r'''
The generic checks prescribe a simple operator with a spectrum known independently, assemble it through the same interface, and compare the computed eigenpairs with that spectrum. Recovering a negative diffusive rate checks the dynamic sign; recovering a complex pair checks the real/imaginary representation and complex-shift machinery. For an incompressible problem, pressure removes the velocity component that would violate the constraint. Comparing with a properly mass-weighted projection checks that the singular pressure rows retain precisely the allowed velocity dynamics rather than introducing extra modes.
''',
'Corrected conditions and the quantitative verification ledger': r'''
The two tables answer different questions. The first records independent checks of parts of the workflow, with their own measures of discrepancy. The second compares thresholds after solving three specific protein-boundary problems on successive meshes. Convergence within one row supports that row's prediction; a persistent difference between rows shows the physical effect of the boundary condition. The close full/plate comparison supplies an additional formulation check for the corrected flat-state problem.

Slope conditions imposed by penalty or Nitsche terms are enforced through the weak equations rather than only by fixing nodal values. The penalty coefficient controls the strength of that enforcement. A sensible check varies it while refining the mesh and verifies that the physical answer is insensitive in the resolved range. A stable answer under one original-rim penalty scan does not establish this for every corrected nonlinear calculation.
''',
'Anisotropic coefficients and transverse stability': r'''
The two-dimensional constitutive law has separate longitudinal and transverse stresses because the imposed flow singles out the $x$ direction. Tilts at several angles identify cross-couplings that cannot be recovered from tilting only along $x$. To test a straight terrace, perturb its slope in $y$ and linearize the transverse stress about its finite longitudinal slope. The resulting coefficient $D_{yy}$ measures transverse restoring stiffness. Positive values resist broad transverse modulations in the tested closure; simulations additionally check whether perturbed ridges return to straight ones.
''',
'Implementation and evidence map': r'''
The source map identifies which files implement each stage of the reasoning. The generic eigensolver accepts a residual, state, boundary conditions, and time form; problem-specific interfaces supply those objects. Separate scripts sweep parameters, continue branches, evaluate forces, or integrate dynamics. Plotting scripts convert stored outputs into the figures. This separation matters for reproduction: rebuilding a figure from its archived data checks the presentation, whereas rerunning its solver with the same mesh and parameters checks the numerical calculation.
'''
})

subsection_readouts['Dynamic residual, mass matrix, and sign convention'] = r'''
In the time form, the hatted fields represent a trial rate and $\nu$ the collection of test functions; subscripts select their corresponding equations. The density $\rho$ weights velocity inertia. Setting it to zero removes inertial evolution while leaving the force and incompressibility equations to determine velocity at each instant. The height still evolves through kinematics and normal friction. Thus an overdamped membrane is not a stationary membrane: its shape can grow, decay, or oscillate while velocities remain instantaneously force-balanced.
'''
subsection_readouts['Continuation and the amplitude equation'] = r'''
For the perfect case $s=0$ and $N_3>0$, the nonzero stationary solutions satisfy $A^2=-\lambda_v(v_0-v_c)/N_3$ and exist below onset. The derivative of the amplitude dynamics on those solutions is $-2\lambda_v(v_0-v_c)>0$, so they are unstable. Below onset, the flat state attracts sufficiently small disturbances, but these unstable solutions mark a local amplitude barrier. A disturbance beyond that barrier departs from the nearby flat basin. This is the precise sense in which a backward branch can coexist with a linearly stable flat membrane.

To continue through a fold, varying drive alone is insufficient because two different shapes may have the same drive. The archived branch calculation instead adds an amplitude constraint and solves for the drive as an unknown in a bordered Newton system. This allows stationary unstable segments to be traced as well as stable ones. Stability along that branch is then assessed independently rather than inferred from whether Newton converges.
'''
