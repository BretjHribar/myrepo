# Physical tensor sector completion

17 November 2026. Accuracy work, with speed optimization disabled. The
24-orbital water state has no independent correlated reference score;
neither chemical accuracy nor the challenge is established.

## Why the fixed seed family is closed

The initial all-site physical search used exact elimination of the
auxiliary coordinate: minimizing x in the statewise regularized energy
recovers the literal physical Hamiltonian for every state. Its numerical
implementation passed independent four-orbital singlet/doublet tests,
including interrupted-update recovery. Their energy errors were
1.33e-15 and 1.28e-10 Hartree. These test the new update implementation,
not chemistry or an independent universal functional.

On the saved water state, four updates completed before the local metric
guard failed at center 4: 5.10e-8 versus the unchanged 1e-8 threshold.
No failed update was installed. More importantly, the seed's charge labels
at the cut after five initially occupied orbitals permit prefix N>=8.
For ten total electrons this excludes every component with three or more
electrons in virtual orbitals. One-site sweeps cannot create missing bond
charge/spin sectors, regardless of their convergence.

The fixed-seed protocol is therefore **closed**. Do not resume it with
more sweeps or change seeds, ranks or tolerances to rescue that family.
Its four valid updates remain available as a warm state, with its failed
report preserved in `tensor_water_forward_gpu_20261117.json`.

## Conditioning repair, with both outcomes retained

An initial gauge recovery used inverse boundary singular values. It
preserved the state but failed the canonical overlap guard at 2.31e-8.
That report remains `tensor_gauge_recovery_gpu_20261117.json`.

`gpu_tensor_gauge.py` now reconstructs canonical cores directly from
reduced-tensor SVD factors and absorbs the singular factors into the
neighbor. Full magnetic partners share each channel transformation.
Right-canonical factors include the spin-degeneracy ratio
sqrt((2*S_right+1)/(2*S_middle+1)). No inverse small singular value is
applied. This is a numerical coordinate repair, not a different state.

The direct-factor recovery preserves energy to 7.11e-14 Hartree and
density to 2.91e-15. The canonical overlap error is 2.96e-14 and the
local physical overlap error is 1.47e-12. No optimization update was
repeated and no tolerance was relaxed. The accepted record is
`tensor_gauge_direct_recovery_gpu_20261117.json`.

## Changed physical family

`gpu_two_site_tensor.py` enumerates every allowed intermediate N,S of two
local CG transitions, independently of the old middle-bond labels. Outer
physical supports are canonical; the exact literal two-site MPO supplies
the physical Hamiltonian residual. No global Fock vector is constructed.

The residual is decomposed into physical Schmidt blocks. Its projection
on the existing left support is removed before selecting new directions.
The existing support is copied unchanged; added directions are complete
spin multiplets, and degenerate singular blocks stay together. The old
right coefficients are copied with zeros on new channels. Thus the old
physical state is exactly retained before an enlarged local variational
update. This avoids replacing the state by an uncontrolled truncated
two-site optimum.

Independent new implementation checks pass:

- The complete two-site action agrees with an independently built full
  matrix to 1.78e-15 Hartree; its overlap and split-state errors are below
  2e-15. The local Ritz minimum agrees to 4.44e-16 Hartree.
- The residual-sector test retains the original wavefunction exactly,
  admits missing N,S sectors, and its subsequent variational update
  agrees with the independent full-section minimum to 1.44e-15 Hartree.
  Number, total spin, occupations and literal energy checks pass.

Sources: `two_site_tensor_checks_gpu_20261117.json` and
`sector_enrichment_checks_gpu_20261117.json`. These verify new numerical
components; they are not repeats of frontier methods or a novelty claim.

## Larger-basis experiment and remaining gate

`tensor_water_sector_protocol_20261117.json` freezes one residual-selected
test at sites 4 and 5. It keeps the same 64-state bond limit, adding at
most eight magnetic states to the existing 56. Reference energies/densities
are excluded. The complete two-site section has dimension 1574. The added
support consists of two N=6 singlets and three N=7 doublets. This captures
75.00% of the squared residual component outside the existing left support,
not 75% of the full many-body residual. Before optimization, energy is
unchanged and the density difference is 2.25e-18.

The authoritative terminal result is `tensor_water_sector_gpu_20261117.json`.
Only a passed terminal record establishes this physical sector test. It
does not establish common tensor stationarity, a complete residual,
context-independent off-shell XC derivatives or a correlated reference
error. The remaining work must connect the changed family to the
auxiliary universal search, obtain common stationarity and independent
realistic-basis balanced-energy evidence. The locked confirmation panel
remains withheld until a real candidate and protocol are frozen.

The molecular experiment is now terminal and **passed**. The enlarged
local section has dimension 768. Its update lowers the physical energy
from -76.2129045296 to -76.2151646370 Hartree (0.00226010744 Hartree).
The probability of three or more virtual electrons changes from zero to
1.10133e-5. Particle-number error is 8.88e-15, spin-squared error 2.39e-18,
and spatial occupations are 0.00015538 to 1.99992736. Local/global energy
agreement is 9.51e-12 Hartree. Peak CUDA allocation was 0.821 GB; complete
project storage was 21.179 GB of the unchanged 40 GB. No correlated
reference or accuracy score was produced. The local physical improvement
does not demonstrate the sub-kcal goal.

## Changing sectors throughout a molecule

`physical_two_site_forward.py` now performs bounded physical two-site
updates with all intermediate N,S sectors. Each local eigenproblem has
an explicit residual/capacity criterion. A spin-complete Schmidt split
enforces the predeclared bond bound; candidates that increase the literal
energy beyond its roundoff guard are rejected and retained in the report.
The final audit changes no physical tensor: it checks all pair minima on
the same final state. This is coordinate stationarity, not a full-Hilbert
residual or a proof of the global minimum.

The new bounded-split check independently verifies its nonzero discarded
norm (0.11758), retained norm and fidelity against explicit wavefunctions;
identity errors are below 1.3e-15 and spin/number checks pass. A separate
six-orbital/six-electron forward check reaches the independent complete
singlet energy to reported floating precision, with a 3.20e-14 Hartree
common-pair gap. Six sweeps, 30 distinct updates, five frozen audit pairs
and intentional checkpoint recovery pass without duplicate updates.
Sources: `bounded_spin_split_checks_gpu_20261117.json` and
`two_site_forward_checks_gpu_20261117.json`. These validate new code;
the known small-space physics is not promoted as chemical transfer.

The first larger-basis test is frozen in
`tensor_bh_accuracy_protocol_20261117.json`: new BH geometries at 1.20 and
1.80 Angstrom, all 19 cc-pVDZ orbitals, six electrons, a predeclared
128-state spin-complete cap and unchanged physical/coordinate guards.
Both predictions must finish before either independent reference is
generated. If either prediction fails its original gate, reference work
is deferred for this failed protocol. No reference, rank, seed or
iteration sweep is allowed to select the prediction.

Before any reference was generated, a correctness amendment replaced the
planned determinant-only reference start. Such a start can be restricted
to an additional spatial symmetry. The new single fixed start adds an
equal-norm generic full-space singlet component, retaining the determinant
as well; its seed, capacity and residual thresholds are prespecified in
`tensor_bh_reference_policy_20261117.json`. Predictions and acceptance
thresholds did not change. The calculation resumed its saved forward
cursor, with cached CUDA integrals/RHF and all completed updates retained.

This new molecular test is running; its authoritative status and any
terminal accuracy decision are `tensor_bh_accuracy_gpu_20261117.json`.
No result may be inferred from partial energy lowering. The independent
reference uses a different six-particle action, sharing the input
integrals. Its full fixed-Sz dimension is 938,961. No CPU reference is
permitted. Even a passed two-geometry stretch contrast would remain
finite-basis development evidence, short of broad reactions, complete-basis
accuracy, global arbitrary-density XC derivatives and WTMAD-2.

CG coupling, canonical gauges, tensor/MPO algebra, SVD and residual
selection are established mathematics. Their implementation alone is
not a transferable learned model or an AlphaFold-scale advance. The
research hypothesis remains a physical density-response state with a
reliable scalar XC energy and potential. WTMAD-2, complete-basis accuracy
and broad out-of-sample generalization remain unproved.
