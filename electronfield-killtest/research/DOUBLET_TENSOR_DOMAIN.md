# Physical doublet domain: implementation checks passed

17 November 2026. This extends a missing physical domain of the research
state. It is **not an accepted molecular accuracy result**. Its independent
CUDA implementation test passed at a verified idle GPU boundary after
the user resumed the goal. Speed work remains disabled.

## Physical construction

The accepted larger-basis two-site implementation requires a total
singlet. Earlier odd-electron tensor primitives do not by themselves
validate this solver for open-shell chemistry. `gpu_doublet_embedding.py`
prepares an odd-electron physical doublet with one inert auxiliary spin:

\[
 |\Xi\rangle =
 (|\Psi_{1/2}\rangle|\downarrow\rangle
  -|\Psi_{-1/2}\rangle|\uparrow\rangle)/\sqrt2.
\]

Tracing out that auxiliary gives the positive physical ensemble

\[
 \Gamma_{\rm phys}=\tfrac12\sum_{M=\pm1/2}
 |\Psi_M\rangle\langle\Psi_M|.
\]

It has the physical odd electron count and S=1/2. For a spin-independent
Hamiltonian its energy equals the pure doublet energy. The auxiliary
has zero one-/two-electron Hamiltonian terms, exactly one electron, and
is excluded from physical density and electron-count reports. Coupling
and purification are established mathematics, not our claimed novelty.

The physical-prefix bond is fixed to (N_phys,2S=1); the auxiliary site's
local occupancy is fixed to one. These are exact physical-sector
constraints. The two-site builder otherwise permits every allowed
intermediate N,S and checks that the literal Hamiltonian does not leak
into excluded physical/auxiliary sectors. Omitting these constraints
would allow artificial charge transfer to the zero-energy auxiliary.
The method must not count its extra electron as larger physical-system
coverage or compare an untraced enlarged-system spin with a doublet.

## Independent acceptance test

`doublet_embedding_checks_protocol_20261117.json` freezes one new
three-orbital/three-physical-electron implementation check. The singlet
purification has four total electrons in four sites, but its physical
claim remains **three electrons**, with the auxiliary removed.

After the frozen forward search and common-state audit, an independently
enumerated wavefunction is explicitly traced over the auxiliary. The
test checks positivity, normalization, physical charge support, physical
spin Casimir variance, the auxiliary's equally mixed spin, literal
physical energy, and the independent complete doublet-sector minimum.
Neither an enlarged-system singlet check nor a correct total charge can
substitute for these physical-subsystem checks.

`gpu_three_particle.py` supplies an independent CUDA reference action in
an antisymmetric alpha-alpha-beta tensor, together with a sparse exact
doublet projector. The same new test compares both with independently
built matrices. At M=1/2, S^2=S_+S_- -1/4 and
P_(S=1/2)=(15/4-S^2)/3 for three electrons. No CPU reference or fallback
is used. This is a reference primitive, not an XC model or speed work.

The authoritative [CUDA report](doublet_embedding_checks_gpu_20261117.json)
passed. Independent physical-reference energy error is 2.22e-15 Hartree;
partial-trace energy error is 4.44e-16 Hartree. Physical number error,
spin error, wrong-number probability, auxiliary charge variance and physical
spin-Casimir variance are zero at reported precision. Positivity,
normalization and maximally mixed auxiliary-spin errors are below 2.3e-16.
The independent three-particle operator error is 1.33e-15 Hartree and the
common-context gap is 4.00e-15 Hartree. Do not repeat this settled check,
change the frozen BH protocol, or expose the locked confirmation panel.

Before its first numerical run, this new doublet protocol now selects
one fixed generic physical Krylov component. A retained-only vector can
have zero residual in an excited invariant sector. The independent
[spectral-start check](RITZ_START_GUARD.md) must pass before the doublet
test. Both tests have now passed; neither changes the original BH job.

Even a passed new domain test would leave independent molecular
open-shell/balanced-energy transfer, larger physical electron counts,
realistic-basis accuracy, universal off-shell XC derivatives and WTMAD-2
unproved. A new molecular protocol must be frozen before using this
extension for accuracy claims.
