"""Rebuild the figure atlas from the audited 2026 manifest; no numerical runs."""
import hashlib,json,pathlib
HERE=pathlib.Path(__file__).resolve().parent
rows=json.loads((HERE/'figure_manifest.json').read_text())
main={
'membrane_flow2_threshold_vs_tension.png':1,'membrane_ring_force_physical_units.png':2,
'membrane_flow2_plate_model.png':3,'membrane_r3_multi.png':4,'membrane_r3_landau.png':5,
'membrane_r5_fold_parametric.png':6,'membrane_r5_flutter.png':7,'membrane_r5_peclet.png':8,
'membrane_r6_slope2d.png':9}
caps={
'base_flow_and_eigenmode':'Steady cylinder base flows and leading perturbations below, near, and above the Hopf threshold. The unstable symmetric steady solution is a valid base state even when nonlinear dynamics sheds vortices.',
'base_flow_validation':'Cylinder drag and recirculation length compared with reference values. This tests the base solver independently of the eigensolver.',
'convergence_and_residuals':'Cylinder onset, frequency, and eigenpair residuals under mesh refinement. The residual checks numerical solution of the discrete problem; mesh changes test its approximation to the continuum.',
'dns_lift':'Lift time series at Reynolds numbers 40, 60, and 100. The low-Reynolds-number perturbation decays; the unstable cases grow and saturate in vortex shedding.',
'dns_vorticity':'Nonlinear cylinder-wake snapshots for the benchmark runs. The vortex street above onset provides an independent dynamical check of the computed Hopf instability.',
'dns_vs_eigenvalues':'Perturbation amplitude in cylinder DNS compared with exponential eigenvalue predictions. Growth is fitted in the linear interval before nonlinear saturation.',
'eigenvalue_trajectories':'Cylinder eigenvalue trajectories as Reynolds number varies. The relevant oscillatory pair can be missed by searching only near the origin.',
'eigenvalues_vs_dns':'Cylinder growth rates and frequencies from eigenanalysis and nonlinear time integration. Agreement is strongest for frequency and within a few percent for the fitted growth rates.',
'growth_rate_vs_Re':'Leading cylinder growth rate and frequency versus Reynolds number on three meshes. The first instability occurs near Reynolds number 46.3.',
'mesh_coarse':'Coarse cylinder mesh, including a close view of the obstacle. This is the first of the three benchmark resolutions.',
'mesh_fine':'Fine cylinder mesh and obstacle refinement. It gives the reported benchmark onset near Reynolds number 46.32.',
'mesh_medium':'Intermediate cylinder mesh and obstacle refinement. It supplies the middle point of the convergence sequence.',
'mode_branches_vs_Re':'Competing cylinder growth-rate branches and the leading Hopf Strouhal number. Selecting the least damped returned eigenpair requires covering the relevant complex-frequency range.',
'spectra':'Cylinder spectra near the sampled complex shifts. The Hopf pair and dense damped cluster illustrate why a real shift alone is insufficient.',
'stability_map':'Cylinder stable/unstable classification from eigenvalues and independent DNS outcomes. Threshold differences across meshes are small but remain measurable.',
'summary':'Overview of the cylinder benchmark: leading growth rate, threshold, and contrasting nonlinear outcomes below and above onset.',
'check_eigensolver':'Generic eigensolver checks on a heat equation and a rotation-coupled system with complex eigenpairs. Both real and complex spectral calculations are tested against exact values.',
'check_projector_equivalence':'Mixed incompressible eigenproblem compared with the correct discrete Leray projector. A Euclidean projector is a different operator and produces different spectra.',
'check_force_methods':'Ring force diagnostic under refinement. Virtual work converges; the original boundary line integral does not. Adding the moment term improves a coarse result but does not remove the boundary-derivative discretization error.',
'contact_angle_R10':'Contact-slope dependence of ring force and leading relaxation rate at outer radius ten. Force extrema shift, while the sampled height-controlled branches remain stable.',
'eigenvalues_vs_h':'Ring relaxation spectra versus protein height, resolved by azimuthal number. No growing fixed-height mode is found in these sampled branches; spectra soften as the neck becomes steep.',
'flow2_bifurcation':'Corrected clamped-protein continuation. The perfect branch is subcritical and a nonzero contact slope produces a fold; the fitted imperfection exponent is close to two thirds. The later projected theory supplies an independently computed prefactor.',
'flow2_domain':'Corrected physical thresholds versus domain length, protein radius, width, and outer-height constraints. Fixed-tension and fixed-Gamma size sweeps are different experiments.',
'flow2_friction':'Early friction comparison showing divergence thresholds and full-equation flutter crossings. The real plate threshold curve alone is not the first onset when a complex pair crosses earlier; the later map in the archive resolves this.',
'flow2_sensitivity':'Normal direct/adjoint shape, wavemaker, and tangential-force direction that increases the reference threshold. The strong response is located upstream of the inclusion.',
'flow_irene_example_collision':'Inertial bending-wave collision in the slip-inclusion example. This is an illustrative high-inertia mechanism, not the overdamped lipid-membrane threshold.',
'flow_irene_example_convergence':'Slip-penalty example: weak oscillatory growth shrinks with mesh size, while the apparent threshold depends strongly on penalty strength. These results diagnose numerical and boundary-condition sensitivity and are excluded from quantitative core claims.',
'flow_pre_SLc_vs_tension':'Superseded original-rim calculation. Its approximate linear threshold-versus-tension fit belongs to the implementation without vertical protein force balance and must not be used for the corrected physical model.',
'flow_pre_eigenvalue_vs_SL':'Superseded original-rim flow spectra and large imperfect deformation. The roughly sixteen threshold is a diagnostic of the original boundary-value problem; corrected thresholds are near twenty-seven in the reference geometry.',
'flow_pre_imperfect_shapes':'Superseded original-rim shapes and leading modes at nonzero contact slope. Their large smooth deformation is not evidence for a smooth transition with a physically force-free protein.',
'flow_pre_mechanism':'Superseded original-rim stress and eigenmode fields. They reveal upstream compression but place the mode nearer the protein than the corrected force-balanced problem.',
'flow_pre_tension_centreline':'Original-rim flat base-flow tension along the centreline. The upstream tension decrease is mechanistic evidence; the onset value attached to this figure is superseded by the physical protein condition.',
'flow_time_check':'BDF2 versus eigenvalue growth and decay for the original-rim implementation. This checks that the numerical linearization predicts that implementation\'s dynamics, not that its rim condition is physically force-free.',
'force_displacement':'Ring forces and fixed-height eigenvalues for several radii. Dashed original line-force curves are retained as diagnostics; extrema and force-control conclusions use virtual work.',
'profiles_ring_R10_tan0.5':'Ring profiles for outer radius ten and contact slope one half. Increasing imposed height produces a steep neck while the corresponding fixed-height spectrum remains stable in the sampled range.',
'profiles_ring_R10_tan0':'Symmetric zero-contact-slope ring profiles at outer radius ten. Upward and downward branches are related by reflection in the reference plane.',
'profiles_ring_R10_tan1':'Ring profiles at outer radius ten and unit contact slope. The imposed slope biases the spontaneous height and the two force-control barriers.',
'profiles_ring_R30_tan0.5':'Profiles for outer radius thirty and contact slope one half. The displayed Monge window does not reach the long-tube force plateau.',
'profiles_ring_R5_tan0.5':'Strongly confined ring profiles at outer radius five. Large end slopes limit a height representation and distinguish the confined neck from a freely developed wide tube.',
'r3_flutter_map':'Divergence/flutter onset map versus screening length and tension, with full-equation checks. The local absolute-instability panel is a guide to the global wave source; near onset the local wavelength is comparable to the compressed region.',
'r3_forcefree':'Nonlinear force-free branches for nonzero contact slopes, contrasted with clamped and original-rim conditions. The corrected free protein stays near its rest deformation before an imperfection-sensitive fold.',
'r3_lattice':'Periodic array spectra and patterns. Early finite-wavevector scans are interpreted with the dedicated later long-wave coefficients: a neutral uniform translation exists at zero wavevector, and the longitudinal stiffness controls onset.',
'r3_mobile':'Uniform-plane curvature-protein dispersion under relative drift. Transverse wavevectors remain least affected, giving stripes parallel to the flow; travelling longitudinal bands require suppressing those transverse modes.',
'r3_snap':'Clamped-protein time stepping after the fold: bottleneck, deepening upstream pit, and rapid slope growth. The final steep shapes are at the Monge limit, not resolved steady attractors.',
'r3_tube':'No-flow axisymmetric arclength continuation. Large rings develop tubes tending to the tether-force plateau; small rings constrain a neck and show a large force overshoot. This is distinct from flow-driven tube growth, which remains unfinished.',
'r3_wrinkles':'Plate-model modes well above onset and in sliding-membrane examples. Increasing compression permits multiple lobes and wrinkle trains; these modes do not establish a stable nonlinear pattern around a single protein.',
'r4_lattice_nonlinear':'Nonlinear tilted-cell stress and geometric softening. Frozen flat-state flow is used; the small-slope cubic changes sign between intermediate and dense arrays, while the driven linear coefficient differs modestly from the dynamic Bloch value.',
'r4_post_snap_forcefree':'Force-free protein lift during upstream collapse. The protein rises onto a downstream crest and the wall steepens; cutoff-force consistency deteriorates at the last slopes.',
'r4_proteins':'Earlier mobile-protein thresholds with no-flux outflow. The high-flow demixing mode is an outflow-layer artifact. The final absorbing-outflow mobility study replaces those high-flow thresholds.',
'r5_proteins_dyn':'Consistent nonlinear mobile-protein dynamics: no-flow energy relaxation, amplitude saturation, flow-elongated domains, and a source-driven plume. Open reservoirs/sinks mean global protein conservation must include boundary exchange.',
'r5_slope_equation':'One-dimensional homogenized lattice dynamics: continuous or hysteretic slope states, coarsening terraces, and finite-patch drift. The absolute-threshold estimate uses a long-wave model near the edge of its spatial validity.',
'r5_tricritical':'Small-slope nonlinear-cell fits locating the cubic sign change near anchored-disc area fraction 7.5 percent. This onset diagnostic uses a narrower fitting window than the later two-dimensional finite-amplitude polynomial.',
'r6_codim2':'Clamped-protein, intermediate-mobility eigenvalue sectors and nonlinear runaway. The nearly vanishing eigenvalues motivate a degenerate Bogdanov-Takens-type interpretation; an exact double-zero location and quintic unfolding remain unresolved.',
'r6_wake':'Source-induced folds and the Koiter prediction. Recruitment is an imperfection that advances rather than removes the snap. The largest-source case is outside the smallest-amplitude cubic regime.',
'ring_stability_diagram':'Height/contact-slope ring stability diagram over the computed windows. The force-control boundaries are virtual-work force extrema; height-controlled branches remain stable. Grey endpoints indicate limits of the sampled branch.',
'stability_summary':'Ring control-ensemble summary. Force-controlled softening segments are unstable even when the fixed-height shape spectrum remains stable.',
'flat_ring_R5_coarse':'Coarse-mesh flat-annulus eigenvalues by azimuthal number compared with the exact Bessel spectrum. This is a separate asset from the fine-mesh check with the same original basename.',
'flat_ring_R5_fine':'Fine-mesh flat-annulus spectrum. Refinement reduces the maximum discrepancy to about 2.3 times ten to the minus four across the tested twenty eigenvalues.'}
anim={
'dns_Re40_vorticity':'Cylinder benchmark below onset: an initial perturbation decays to a steady symmetric wake.',
'dns_Re60_vorticity':'Cylinder benchmark above onset: growing perturbations develop into a vortex street.',
'dns_Re100_vorticity':'Cylinder benchmark at Reynolds number one hundred: nonlinear saturated vortex shedding.',
'gif1_flow_buckling_onset':'Critical-mode visualization of flow-driven buckling and upstream compression. This onset sequence belongs to the corrected study\'s visualization workflow; the static plots and force condition supply the quantitative threshold.',
'gif2_subcritical_snap':'Branch visualization of the subcritical snap and its imperfection-sensitive fold. A continuation animation is a sequence of equilibria, not a physical-time simulation.',
'gif3_divergence_vs_flutter':'Real divergence mode versus oscillatory flutter mode. The travelling phase of the complex mode is distinct from a nonlinear saturated cycle.',
'gif4_ring_pulling':'Ring-pulling profile sequence. The imposed height varies along a branch; force-control stability is determined separately by the virtual-work stiffness.',
'gif5_post_snap_collapse':'Clamped-protein post-fold time integration: an upstream pit grows and the wall approaches vertical.',
'gif6_lattice_wave':'Collective Bloch-wave pattern of a periodic protein array. This visualizes the long-wave instability and upstream phase drift, not a nonlinear terrace simulation.',
'gif7_tube_extrusion':'No-flow axisymmetric tube/neck continuation beyond the Monge range. The force plateau for a large ring agrees with tether theory.',
'gif8_proteins_rest_3d':'Consistent nonlinear curvature-protein demixing at rest. A modulated labyrinth forms while the free energy decreases.',
'gif9_proteins_flow_3d':'Mobile proteins in open flow. Downstream domains elongate along the flow and the membrane follows their curvature modulation.',
'gif10_proteins_source_3d':'Rim-source protein plume carried downstream with a shallow associated trough. The final run uses absorbing outflow.',
'gif11_flutter_3d':'Nonlinear friction-driven flutter: protein oscillation and a sustained source of upstream membrane waves.',
'gif12_fold_parametric_3d':'Parametric continuation of the force-free post-snap wall through vertical into an overhang underneath the lifted disc. The final state is distortion-limited.',
'gif13_codim2_static_runaway_3d':'Intermediate-mobility clamped-protein static-sector runaway. Shape and signed density deviation grow into a deep pit; no terminal attractor is resolved.',
'gif14_codim2_oscillatory_3d':'Intermediate-mobility oscillatory-sector runaway. A growing travelling mode evolves into collapse rather than a saturated wave source.'}

def stem(r):
 name=pathlib.Path(r['asset']).stem
 return name.split('_',1)[1]

def caption(r):
 s=stem(r)
 if s not in caps: raise ValueError('Missing caption: '+s)
 return caps[s]+r' \textit{Source:} \path{'+r['source']+r'}. First added '+r['added']+'.'
static=[r for r in rows if r['frames']==1 and pathlib.Path(r['asset']).name not in main]
# One large figure per page; pair only sufficiently wide plots to retain readable labels.
lines=[];sn=0;i=0
while i<len(static):
 group=[static[i]]
 if i+1<len(static):
  a,b=static[i],static[i+1]
  if 172*a['height']/a['width']+172*b['height']/b['width']+70 < 221:
   group.append(b)
 for r in group:
  sn+=1;r['placement']=f'Supplement Figure S{sn}'
  lines.append(r'\begin{center}\includegraphics[width=\linewidth,height='+('86mm' if len(group)==2 else '182mm')+r',keepaspectratio]{'+r['asset']+r'}\captionof{figure}[Archived figure]{'+caption(r)+r'}\end{center}')
 lines.append(r'\clearpage')
 i+=len(group)
(HERE/'atlas_static.tex').write_text('\n'.join(lines)+'\n')
lines=[]
for r in rows:
 fn=pathlib.Path(r['asset']).name
 r['sha256']=hashlib.sha256((HERE/r['asset']).read_bytes()).hexdigest()
 if fn in main:r['placement']=f'Main Figure {main[fn]}; Supplement Figure S{75+main[fn]}'
 if r['frames']>1:
  s=stem(r)
  if s not in anim:raise ValueError('Missing animation caption: '+s)
  sn+=1;r['placement']=f'Supplement Figure S{sn} (storyboard and bundled GIF)'
  cap=anim[s]+r' Four ordered frames are selected by frame index; spacing is not asserted to be uniform in physical time. \textit{Original GIF:} \path{'+r['source']+r'}. First added '+r['added']+'.'
  lines.append(r'\begin{center}\includegraphics[width=\linewidth,height=170mm,keepaspectratio]{'+r['stills']+r'}\captionof{figure}[Archived figure]{'+cap+r'}\end{center}')
  lines.append(r'\noindent\textit{Playback file in this source package:} \path{'+r['asset']+r'}.\clearpage')
(HERE/'atlas_animations.tex').write_text('\n'.join(lines)+'\n')
(HERE/'figure_manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
print(f'Indexed {len(rows)} assets: {len(main)} main figures, {len(static)} static supplement figures, 17 GIF storyboards.')

lines=[]
for r in sorted((r for r in rows if pathlib.Path(r['asset']).name in main),key=lambda r:main[pathlib.Path(r['asset']).name]):
 n=main[pathlib.Path(r['asset']).name]
 lines.append(r'\begin{center}\includegraphics[width=\linewidth,height=175mm,keepaspectratio]{'+r['asset']+r'}\captionof{figure}[Main-text figure, larger copy]{Larger copy of main-text Figure '+str(n)+r'. Consult its main-text caption for interpretation. \textit{Source:} \path{'+r['source']+r'}. First added '+r['added']+r'.}\end{center}\clearpage')
(HERE/'atlas_main.tex').write_text('\n'.join(lines)+'\n')
