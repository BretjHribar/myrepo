# Own-research execution record

Updated 17 November 2026. Goal remains active: a useful XC functional inside the organizer's sub-kcal / roughly 10 CPU-second target box. **That goal has not been reached.**

**Latest user priority and result:** full public GMTKN55 first; make all
tests work and establish an initial measured assessment before pursuing
sub-1 accuracy. All 2,462 upstream coordinate pairs pass CUDA geometry/unit
checks. The new data-only parser and CUDA scorer pass all 55 sets / 1,505
reactions, requiring 2,442 structure IDs. No reaction is filtered; 20
unreferenced upstream structure records are listed explicitly. Corrected
v1 normalization is 57.8167 with pinned subset means. Full contraction,
weighting and missing/invalid-prediction guards pass independent tests.
Those tests use synthetic energies and are not model accuracy evidence.

The first full target-basis model-domain inventory finds 2,421 structures
blocked by static memory/domain checks, 21 still uncomputed, and zero
accepted predictions. The memory estimates are preflight estimates, not
2,421 attempted molecular calculations. Overlapping blockers include
2,359 dense CUDA input-array lower bounds above 6.5 GB, 2,421 host word-list
estimates above budget, 60 triplet/quartet domains and 81 ECP requirements.
The largest basis has 3,420 spatial functions. Only four reaction definitions
lack a state preflight blocker. A full end-to-end coverage score correctly
returns 0/1,505 measured reactions and no WTMAD-2 or full-suite estimate.
This is a serious implementation/scope failure, not evidence close to the
accuracy target. See [the full pipeline record](GMTKN55_PIPELINE.md).

At the resumed idle GPU boundary, the fixed generic Ritz-start safeguard
and independent physical-doublet implementation checks both passed. These
are correctness/domain evidence only. The original BH job remains
checkpointed and interrupted with unchanged scientific settings; no new
reference or accuracy result exists. Project storage is about 21.26 GB.
Speed work remains disabled; no frontier method was replicated.

**Subsequent actual public measurement:** the exact N=1 physical component
now predicts all 11 required one-electron structures in full def2-QZVP,
using CUDA integrals/eigensolves and no reference values in the predictor.
All pass physical/eigen-residual gates. Five public reactions can be scored:
G21IP/001 error -1.155744 kcal/mol; SIE4x4/001-004 errors -0.019127,
-0.013515, -0.030466 and -0.177301. All other reactions remain visible.
Current counts are 11 accepted structure predictions, 2,417 unsupported
and 14 uncomputed. Full WTMAD-2 and a full-suite estimate remain unavailable;
this selected exact domain cannot estimate broad chemistry. The tiny
measured contribution to the full denominator is only a lower bound.

The new density-square physical-operator path passes independent all-sector
matrix, correlated tensor-energy and local-action checks. Its first test
comparison reversed the reference bit order; that unaccepted interrupted
report is retained and only the test ordering was corrected. A supplied
40-electron product-energy check also passes without a quartic word list
or four-index ERI. It is not a coupled 40-electron accuracy prediction or
a new general model. Large-basis integral factors and complete spin/ECP
domains remain outstanding. See [the operator record](DENSITY_SQUARE_OPERATOR.md).
The original BH checkpoint and settings remain unchanged. Project storage
after these measurements is about 21.26 GB of 40 GB.

Latest work: the [conditional tensor functional](CONDITIONAL_TENSOR_FUNCTIONAL.md)
passes its fixed-context larger-basis gates. A subsequent all-site search
exposed a seed-sector restriction excluding triple and higher excitations,
and is closed. [Physical sector completion](TENSOR_SECTOR_COMPLETION.md)
now tests the changed family after independent operator/update checks and
an accepted state-preserving gauge repair. No broad chemical-accuracy
result is established. Detailed results and retained failures are below.
Current complete storage accounting is about 21.2 GB of 40 GB; older lower
counts below are historical and omitted external research crash dumps.
Speed work remains disabled; the locked confirmation panel is withheld.

## Binding instructions

No frontier replication, installation, reproduction, retraining, or sanity checks. This supersedes the earlier Skala allowance. One logical CPU core, full local GPU, and 40 GB total project disk. Initial working ceiling 32 GB. No frontier method was installed or run.

**Latest steering:** numerical work is now **GPU only**. The single CPU core is limited to orchestration, metadata/basis bookkeeping, decompression and file I/O. No CPU SCF, CPU integral/reference calculations, or silent CPU numerical fallback. Legacy numerical entry points stop explicitly. **Accuracy below 1 kcal/mol comes before speed optimization.** The final target rectangle remains unchanged; CPU timings are deferred under the current instruction.

## Implemented and established — do not repeat as routine work

- Dedicated `XCResearch` WSL distribution in `.research/wsl`, with a 20 GB virtual-disk capacity limit. Linux Python environment is `/opt/xc-env` inside that disk.
- PyTorch CUDA works. Compute commands use `taskset -c 0`, library thread counts of one, and a host resource guard. Project disk use after the first experiments: **10.10 GB decimal**, including the Linux environment, caches and data.
- The optional GPU4PySCF import has an unresolved shared-library search-path issue. Defer it; our model runs with PyTorch CUDA and reference calculations run quickly on one CPU core. Do not spend this research phase repairing an optional integration.
- Generated **24 3D anchor geometries × 9 electronic states = 216 finite-basis FCI references**, in about 32 seconds end to end. Data: `.research/data/pilot_3d_ccpvdz/`. FCI is exact in cc-pVDZ, not CBS. Its specification/manifest fixes family assignments before calculation. All reference and input-density calculations converged.
- Built a differentiable scalar-energy model with coarse spatial interactions, one-step density/response training, and a PySCF SCF hook for our own functional.
- Built a structurally different second candidate with analytic exchange, Pauli-protected regional electron-pair populations, and a bounded two-state correlation expression. Its current exact AO-exchange implementation does not establish large-system speed.

## First experiments and decisions

All numbers below are tiny-system finite-basis diagnostics. They are **not GMTKN55, reaction-benchmark, CBS, or hidden-suite scores**. Inspecting this pilot's results makes it exploratory development data for subsequent decisions; reserve new families for later confirmation.

| Own candidate | Evidence | Decision |
|---|---|---|
| Collective density network, energy only | Frozen-input training energy MAE 0.79 kcal/mol; held-out MAE 18.98 versus input PBE's 7.04. 250 epochs took about 14 seconds of training. | Reject this fitted version as a transferable functional. A good training fit did not transfer. |
| Same collective network, density + response | Frozen-input held-out MAE 13.10, still worse than 7.04. 250 epochs took about 94 seconds. | Response loss alone did not solve the failure. Do not expand seeds or repeatedly tune its weight. |
| Pauli-protected pair-state model | Single-electron H₂⁺ at three unseen bond lengths converged and recovered finite-basis reference energies to numerical precision. This follows from its analytic self-interaction cancellation. | Keep the physical construction, not as a breakthrough claim. No need to rerun this established limit. |
| Pair-state model on new two-electron geometries | Fresh SCF on three triangular H₃⁺ geometries converged, but MAE was 24.87 kcal/mol. | Independent pair lowerings overcorrelate unfamiliar multicentre systems. This fitted correlation expression fails transfer. Do not scale it unchanged. |

A single fresh-SCF integration calculation for the first model also converged; this verifies that its energy/potential hook operates, not that it generalizes.

Artifacts:

- `pilot_energy_20261116.json` / `.pt`
- `pilot_response_20261116.json` / `.pt`
- `pilot_energy_scf_integration.json`
- `pilot_pair_state_20261116.json` / `.pt`
- `pilot_pair_state_h2plus_scf.json`
- `pilot_pair_state_h3plus_scf.json`

## Subsequent tiny-system experiments

The shared pair Hamiltonian plus flat-spin prior gave held-out MAE 41.27 kcal/mol. A conserved regional-hole flow improved validation to 3.038 but failed aggregate held-out transfer, 9.114 versus PBE 7.04. Classical canonical-ensemble charge allocation gave 18.39 versus 7.04. These versions are rejected; do not sweep seeds, weights or temperatures to rescue them. The useful unresolved issue is quantum delocalization and shared orbital occupation.

## Broader chemistry: development evidence

Prepared def2-TZVP, grid-level-1, density-fit PBE input states for **191 molecules plus 15 atoms**, in about 578 seconds. One of 192 selected molecular inputs failed convergence and remains recorded. Whole-composition assignment gives 151 training, 22 validation and 18 test molecules. A selection-order defect made this inherited panel **Al/B-rich**; it is not an unbiased broad-chemistry sample. Future selection ordering is fixed. All three panels are now development data because successive research decisions used their results.

Public molecular energy labels are pinned to MSR-ACC/TAE25 revision `f61c01d2103c230f103099314e86752032548836`, with source/license/checksums in `.research/data/msr-acc-tae25/source.json`. No frontier model code or weights are used. Scores below are unweighted atomization errors, not reaction, barrier, WTMAD-2, or hidden ALD/CVD scores.

| Own candidate | Frozen-input MAE: training / validation / test, kcal/mol | Finding |
|---|---|---|
| Diffusive shared screening state | 4.026 / 3.780 / 3.167 | Input PBE values are 5.942 / 6.499 / 5.213. This gain remains far from the required advance. |
| Coherent operator plus density-derived ensemble constraints | 4.838 / 5.686 / 5.022 | The identity prior conflicted with chemical fitting in this representation. |
| Coherent state plus regional charge and context readout | 3.701 / 4.225 / 3.748 | Better fit, still above chemical accuracy. Another MLP fit is insufficient evidence to scale. |
| Orbital coherence plus consistent ensemble energy/direction loss | 5.970 / 6.946 / 6.084 | Adding descriptors did not resolve the competing losses. Reject this fitted version. |
| Structural ensemble cancellation plus learned screening | 1.868 / 2.157 / 2.257 | Better frozen-input fit, still above chemical accuracy. Fresh CPU calculations miss the speed target. |

One fresh-SCF screen of the diffusive candidate on four development molecules gave **4.766 kcal/mol atomization MAE**, with all required atoms and molecules converged. Median molecular time was **7.906 seconds on the GPU path**, in TZVP. This is not organizer CPU timing or target-basis evidence. Do not repeatedly evaluate that unchanged model.

Artifacts: `screened_state_broad_20261116.{json,pt}`, `screened_state_broad_scf_20261116.json`, `coherent_exact_constraint_20261116.{json,pt}`, and `contextual_coherent_20261116.{json,pt}`.

## Current changed hypothesis and next scientific action

Local density and Pauli descriptors do not identify which spatial regions share an occupied orbital. The next model uses compact blocks of the orthogonal AO density matrix `P = sqrt(S) D sqrt(S)`: regional charge `m_A = Tr(P_AA)`, shared-orbital weight `C_AB = ||P_AB||_F^2`, occupation moment `sum_B C_AB / m_A`, and regional purity `C_AA / m_A^2`. Coherence gates the Coulomb-range interaction in a positive-definite latent screening operator.

These are differentiable, occupied-orbital-unitary invariant electronic descriptors. They remain **AO-basis dependent**; basis transfer and organizer eligibility of this generalized KS construction are unproven. The latent state is not a many-electron variational upper bound.

Completed once: reconstructed each cached KS operator from saved density fields and diagonalized it, without an SCF convergence rerun. `.research/data/quantum_tzvp_191/` contains consistent fields, density matrices and orbital descriptors for all 191 molecules and 15 atoms, with reconstruction changes recorded. Dense AO grids are not persisted.

Generated **62 normalized trial orbitals** using fitting compositions and atoms only. Exact N<=1 charge/spin ensemble XC energies and three orbital-direction derivatives per shape use analytic RI Hartree integrals. Fields and density-matrix descriptors come from the same orbital. Data: `.research/data/orbital_one_electron/`; generation took about 18 seconds including startup. These identities train the new model rather than repeat a settled model benchmark.

Completed `quantum_coherent_20261116.{json,pt}`: the 600-step fit failed chemistry transfer and is rejected. Do not repeat it with different seeds or weights.

The next construction, **coherence-resolved ensemble cancellation**, enforces the one-orbital identity directly. Regional Coulomb projections come from the analytic RI integrals in the orthogonal AO basis. Coherence gates which regional Hartree terms are cancelled; a smooth occupation/rank filter excludes many-orbital cores. A matching share of baseline PBE XC is removed. Learned local/context energies and screening sources vanish in the exact one-orbital ensemble limit, so chemical fitting cannot break that identity. This does not claim exact exchange for general determinants.

The changed scalar-energy derivative, including analytic Coulomb and baseline-XC removal, agreed with an orbital-rotation finite difference to 4.6e-11 Hartree per unit rotation before tightening the local rank filter. That one-time diagnostic is `ensemble_new_derivative_20261116.json`; it is not a transfer score and is not a recurring check.

Completed `ensemble_screened_20261116.{json,pt}`, 600 bounded steps, using chemical energy labels without the competing identity loss. Regional Coulomb and baseline-XC fields are cached in `.research/data/ensemble_tzvp_191/`; dense integral tensors are temporary. Measured project storage before this fit: **11.03 GB decimal**.

Fresh **CPU** SCF on four development molecules: all seven required atoms converged; two molecules converged in 25.29 and 25.69 seconds, with atomization errors -3.89 and -3.49 kcal/mol. Two molecules exceeded their 30-second bounded attempts. No complete four-molecule MAE is reported. Artifact: `ensemble_screened_cpu_scf_20261116.json`. A follow-up kinetic-coefficient diagnosis found no negative coefficients on their input fields (`ensemble_kinetic_diagnosis_20261116.json`); do not change that descriptor to fix an unobserved wrong-sign problem. The timeouts do not establish mathematical divergence. They do fail our practical time budget.

The balanced reaction screen is complete: **4.220 kcal/mol frozen-input MAE over 14 reactions**, versus the inputs' **2.962**. All 22 needed molecular corrections returned; two constituent molecules overlap fitting data. Errors include 20.69 kcal/mol for the carbon-dioxide hydrogenation contrast and 7.07 for ammonia formation. This fit is rejected as a transferable candidate. Do not rerun it on more grids, seeds, or an expanded unchanged evaluation. Artifact: `ensemble_reactions_frozen_20261116.json`.

Reaction preparation selected 23 structures and supplied 14 of 18 proposed formula-balanced contrasts; Al2H6, AlH3O and B2H6 were absent from the source. All selected inputs converged. Structures, reference energies, missing formulas, and fitting overlap are retained in `.research/data/reaction_panel_tzvp/manifest.json`. Consistent orbital/ensemble fields are cached in `quantum_reactions_tzvp/` and `ensemble_reactions_tzvp/`. These are development contrasts, not transition pathways or a blind benchmark.

Latest measured total storage is **11.10 GB decimal**, within 40 GB. No compute job remains running at the end of this experiment. A useful next construction must control shared many-electron charge fluctuations, not merely add more local descriptors or identity-loss weights. One candidate is a common regional probability state with explicit empty, spin-up, spin-down and paired populations; loss of coherence must have a kinetic cost shared across all neighbours. This hypothesis is not yet implemented or established. The existing Al/B-heavy sampling also requires a deliberately diverse training design before claiming general chemical transfer. Preserve an untouched confirmation panel; do not use these now-inspected reactions as a future blind test.

## GPU-only shared-pair experiment

Implemented a changed construction in `joint_transport.py`. A **single nonnegative pair-probability table** has row/column marginals fixed by the two electronic densities. All spatial regions share that table. The predicted state pays an explicit positive Fisher localization/coherence cost and its Coulomb expectation; correlation is their difference from the uncorrelated product state. There is no additive learned energy head. A neural kernel chooses the constrained state and a positive kinetic scale. This is an approximate coarse-state energy, **not** a real-wavefunction variational upper bound.

Two radial density modes per atom give the state room to represent intra-atomic avoidance as well as interatomic charge rearrangement. Coulomb integrals of the projected modes are generated on CUDA. Their product-state expectation is normalized to the actual input opposite-spin Coulomb integral, using no correlated labels. AO projection errors and operator consistency residuals are retained.

Prepared **18 singlet two-electron anchors, nine cached field states each**, with no preprocessing failures. Six other anchors are explicitly outside this prototype's domain; this does not establish general-electron coverage. Existing FCI labels and SCF inputs were reused without recomputation. CUDA generated the new four-centre integrals while CPU integral entry points were blocked. The native CUDA XC wrapper segfaulted, so it was bypassed with an unpolarized PyTorch CUDA PBE expression from primary LibXC equations. The new operators passed a saved-Fock trace consistency condition below 2e-5 Hartree. No CPU XC evaluation was used.

Data and operator cache: `.research/data/gpu_pair_state/`. Preparation took about 32 seconds including startup. Storage after preparation: **11.109 GB decimal**. The existing CUDA library needed an unversioned `libcusolver.so` link; no package download or frontier installation occurred. Core dumps are disabled for future GPU jobs.

The first 500-step run, `joint_transport_gpu_20261116.{json,pt}`, exposed density-marginal drift up to 0.00892. Its scores are marked numerically invalid. The necessary repair scales rows/columns downward and fills their nonnegative deficits with a rank-one term, giving a feasible probability table to floating-point precision.

The single corrected run, `joint_transport_exact_marginals_gpu_20261116.{json,pt}`, finished with training / validation / excluded-family MAEs **9.535 / 7.501 / 5.927 kcal/mol**. Maximum marginal error was 1.67e-16; kinetic costs stayed positive. This valid fit fails the sub-kcal objective and is closed. Do not vary its seeds or loss weights. The matched excluded two-electron inputs have PBE MAE 8.207; these are not the full earlier pilot's error numbers.

One exact two-site capacity calculation on He, with kinetic cost removed, allowed correlation as negative as -0.04482 Hartree versus the required -0.03245. It therefore **does not prove** a representational impossibility. Artifact: `joint_transport_radial_capacity_gpu_20261116.json`. It is not a reason for an indefinite optimizer sweep.

The next changed construction is implemented in `gpu_geminal.py`: radial and explicit dipolar/angular orbital directions are assembled from the saved occupied density, projected into its AO basis and Pauli-orthogonalized. Rotation-scalar channels combine these directions across all regions at several physical length scales. A small variational eigenproblem chooses their **shared pair wavefunction**. Its energy uses the actual one-/two-body Hamiltonian, replacing the approximate kinetic penalty and regional Coulomb calibration. All numerical work runs on CUDA using already cached GPU integrals; no FCI solver or reference regeneration is called.

Current run: `collective_geminal_gpu_20261116.json`. At most 22 proposed collective channels are used independent of atom count, followed by linear-dependence removal. Report retained dimension against the complete AO singlet space and check the variational sign against references. This remains a **two-electron, finite-basis mechanism experiment**, not a general-electron DFA or challenge entry. No correlation energies are fitted; cached references are used for reporting only. A sub-kcal result here would still require genuinely new family/element/electron-count and reaction evidence. World novelty has not been established by a literature assessment.

The first radial/angular trial space failed, with excluded-family MAE 7.826 and stretched H2 error 41.79 kcal/mol. It lacked a separately variable charge-transfer orbital. Adding explicit charge directions changed the physical trial space rather than a fitted weight: `charge_angular_geminal_gpu_20261116.json` gave **0.265 kcal/mol MAE on six multicentre development cases, maximum 0.750**. All 18 two-electron anchors were below 1; their retained state counts were 4-24 versus complete singlet dimensions 15-210. No reference energies were fitted. This is the first sub-kcal **mechanism** finding, not a broad chemistry result.

The first three-electron extension, an antisymmetrized pair plus a freely moving unpaired electron, failed: **24.379 kcal/mol MAE** over two H3 neutral geometries (`odd_collective_geminal_gpu_20261116.json`). Its initial direct four-index transform created an unnecessarily large GPU intermediate; future transforms are sequential and subsequent runs used about 67 MB peak CUDA allocation. No CPU numerical fallback was used.

Local connected charge gates improved the three-electron result to 17.727, still a failure (`connected_flow_gpu_20261116.json`). Simultaneously transforming eight physical symmetry-sector states improved it to 7.271, still a failure (`sector_flow_gpu_20261116.json`). Those fits are closed. The physical states are not random training seeds. No energy-reference loss was used; state parameters minimize their Hamiltonian expectation.

Current changed experiment: `gpu_spin_flow.py`, 400 fixed optimization steps per case. It adds isotropic spin recoupling to the same number-conserving charge operators, with exact local spin-gate exponentiation in three-dimensional permutation blocks. Charge-only gates cannot change spin coupling within a fixed singly occupied pattern. The new gate family checks antisymmetry and the doublet spin identity; its final energy is recomputed as the actual Rayleigh quotient. **Broad accuracy and out-of-sample reliability remain unproven.** Do not extend depth/weights/seeds merely to improve these two development cases.

Locked an untouched **16-composition / 32-structure confirmation panel** in `.research/data/locked_composition_confirmation.json`. Selection read geometry/composition metadata only, not reference-energy columns. All previously inspected compositions are excluded. Each pair defines a balanced same-composition energy contrast. Freeze a general-electron candidate before unblinding; record missing labels or failures without replacing cases. This neutral closed-shell panel does not test barriers, ions, radicals or hidden precursor chemistry.

The actual challenge requests an XC functional. Current wavefunction studies are representation experiments; a universal density-constrained kinetic-plus-Coulomb functional and consistent XC potential have **not** yet been constructed. Keep that bridge explicit rather than substituting a small wavefunction calculation for the full objective. Latest measured project disk use before the spin run: **11.113 GB decimal**. GPU numerical work only; speed work remains deferred until a valid transferable model is supported.

The spin/charge flow preserved antisymmetry and the doublet identity but still
failed accuracy: **3.950 kcal/mol MAE**, maximum 4.381. Adding density-directed
orbital rotations gave **2.598**, maximum 2.875. An additional validity audit
found that the odd-electron pair basis did not guarantee coverage of the input
determinant. A reference-complete construction explicitly retains its
spin-projected input; it passed the non-degradation condition but still gave
**2.353 kcal/mol MAE**, maximum 3.026. These versions are not generalization
successes. Artifacts: `spin_charge_flow_gpu_20261116.json`,
`orbital_spin_charge_flow_gpu_20261116.json`, and
`reference_complete_flow_gpu_20261116.json`.

Current changed operator: `gpu_angular_flow.py`. It adds the rotation-scalar
sum of dipole fluctuation products, acting on every electron pair, to the
best reference-complete physical state. This changes the available angular
correlations rather than increasing a failed fit's seeds or depth. Its
exponential action has a controlled polynomial approximation; actual
Rayleigh energies, fermionic antisymmetry and the doublet spin identity are
checked. Numerical work is CUDA-only. Run handle at launch: exec session
`2712`, bounded at 1200 seconds; inspect this handle or authoritative process
state before restarting any work. Output:
`angular_connected_flow_gpu_20261116.json`.

That run is complete: the dipole operator reduced three-electron MAE to
**1.215 kcal/mol**, maximum 1.561. Completing the local angular family through
quadrupole rank gave **1.197**, maximum 1.548. A targeted parameterization audit
found no active small-angle cap: effective angles stayed below 0.035 versus
the 0.1 bound, with zero parameters above 95% of the cap. Do not expand that
bound or rerun iterations to fix a nonexistent numerical limitation.

The last changed construction added an irreducible connected third density
moment, whose definition vanishes for fewer than three particles. It gave
**1.053 kcal/mol MAE**, maximum **1.382**, with no failed cases or reference-energy
fit. This still fails the sub-kcal criterion and does not establish broad
out-of-sample behavior. The local-operator prototype is closed; do not keep
adding operators to these same two development cases.

Artifacts: `angular_connected_flow_gpu_20261116.json`,
`quadrupole_connected_flow_gpu_20261116.json`,
`angular_gate_capacity_audit_gpu_20261116.json`, and
`connected_triple_flow_gpu_20261116.json`. All recorded numerical work ran on
CUDA. No compute job remains running at the end of this phase.

## Next research direction after the local-operator prototype

The current dense tensor implementation is limited to two/three electrons.
Replace that representation before claiming a model for new molecules. A
particle-number-resolved fermionic tensor or determinant representation must
support the same connected charge/spin/angular operators at general electron
counts, retain its physical mean-field reference, and make truncation/total-spin
errors explicit. This is a model-scope requirement, not speed tuning. Do not
install or reproduce an existing frontier method to supply it.

The fixed one-Bohr region/radial scales are also an untested chemical-transfer
assumption. A consolidated candidate should use density/geometry-derived local
scales with controlled coordinate scaling and fragment limits, rather than
fit those constants to the two remaining H3 cases. Keep universal-functional
density constraints and the XC-potential bridge explicit. Do not relabel an
unconstrained wavefunction energy as the challenge's XC functional.

The locked 16-composition confirmation protocol is now reviewable in
`confirmation_manifest_20261116.json`; its reference contrasts remain withheld.
Do not unblind until a general-electron candidate and protocol are frozen.
Latest full storage accounting at the last run: **11.140 GB decimal**, below
both the 32 GB working ceiling and 40 GB hard cap. The goal remains active and
unachieved. Accuracy, physical validity and out-of-sample evidence come first.

The transfer and functional-definition requirements are explicit in
`MODEL_REQUIREMENTS.md`. No physically broad, validated out-of-sample model has
yet been established. The source file was previously parsed by public-data
utilities, but the new confirmation selection read no energy columns and those
contrasts remain withheld from model evaluation. Do not call that procedural
holdout a guarantee that the energy bytes were never loaded historically.

Full self-consistency and balanced relative energies remain necessary before scaling to precursor chemistry. The current model and constraints are in `screened_state.py`, `quantum_features.py`, `train_broad.py`, `exact_constraints.py`, `prepare_quantum_inputs.py`, and `prepare_orbital_constraints.py`.

## General-electron and constrained-functional work, 16 November evening

The general state engine is implemented in `gpu_spin_chain.py` and
`gpu_manybody_ops.py`. Reduced tensor coefficients have exact particle-number
and SU(2) total-spin labels. Fermionic occupation/Jordan-Wigner operators act
on CUDA. This uses established tensor/linear-algebra primitives; no new tensor
network method or frontier reproduction is claimed.

`general_spin_state_4e_gpu_20261116.json` changed the filling of a cached H4
Hamiltonian to N=4 and froze six orbitals before optimization. A new CUDA
active-space singlet reference was generated for that changed sector. The
400-step state was **0.05223 kcal/mol** above it, with N=4, S^2 about 8e-17
and independent energy agreement at 2e-15 Hartree. This is a six-orbital
projection, not a full-basis chemical result.

The first T+W moment-constrained penalty search ended with moment residual
**0.00682**. Its apparent -0.930 kcal/mol error is **invalid as functional
accuracy**; the density constraints were not met. The changed feasibility
restoration/SQP construction maintained those moments at **1.1e-14** and gave
**+0.2955 kcal/mol** versus the finite-moment oracle. Its potential was still
not converged (difference norm 0.1374). Artifacts:
`density_moment_functional_gpu_20261116.json` and
`feasible_density_functional_gpu_20261116.{json,pt}`. Do not increase penalty
iterations or promote the feasible energy as a validated XC potential.

Residual completion replaced fixed state capacity with directions from T+W
and spatial-density response. The N=4 finite-moment result had density
residual 4.4e-13, energy difference 3.3e-8 kcal/mol and potential difference
norm 3.6e-7. It used **96 directions out of a 105-dimensional singlet space**
and still hit its stated stationarity capacity limit (9.9e-6). This is not
compression or a learned generalization breakthrough. Artifact:
`density_residual_completion_gpu_20261116.json`. Do not rerun the old H4
case merely to make its convergence flag pass.

The frozen four-case chemistry protocol first tested the unchanged two-channel
state at LiH/BeH2 neutral geometries in their complete STO-3G bases. LiH errors
were 0.0712/0.0551 kcal/mol; BeH2 errors **1.5765/4.1846**. The BeH2 stretch
contrast missed by 2.6081. This fixed-capacity construction fails transfer and
is closed; no seed/rank/iteration sweep is authorized by these results.
Artifact: `general_chemistry_transfer_gpu_20261116.json`.

The changed residual-completed state used a frozen protocol on new CH, NH
and OH cases with **7, 8 and 9 electrons**, respectively. Number and total spin
were preserved; residuals were below 1e-7 without reference-energy fitting.
NH/OH matched their minimal-basis reference energies to numerical precision.
CH doublet was 0.0946 kcal/mol above the lowest fixed-Sz state, which is a
**quartet in this minimal basis**, so that is a conservative energy-error
upper bound for the targeted doublet, not a spin-matched exact error. The raw
three-case MAE against those lower references is 0.03154. This is new
small-basis development evidence from per-molecule physical solves, not a
trained predictor, broad chemistry result or CBS claim. Artifact:
`residual_spin_transfer_gpu_20261116.json`; protocol hashes are in the report.

Added `compose_singlets` as an explicit product operation in the same state
family. Two ordered neutral LiH singlets embed into N=8/12 orbitals with
normalized squared product overlap 1.0000000000000009, additivity error
3.6e-15 Hartree, N=8 and S^2 about -4.1e-18. No old reference was regenerated.
This assumes supplied independent components/orbital ordering; **it does not
prove automatic size consistency of a finite global residual space**.
Artifact: `fragment_composition_gpu_20261116.json`.

### Coulomb-complete moments and kinetic completion

The changed scalar hypothesis and its limits are derived in
`COULOMB_MOMENT_FUNCTIONAL.md`. It uses
**E_xc[D]=F_M(q[D])-Ts_M(q[D])-E_H[D]**, keeping Tr(TD) in the total energy.
Coulomb probes at each nucleus span the external potential; their universal
T+W search is not weighted by nuclear charge. This adds an important missing
condition to arbitrary coarse moments and retains kinetic cost outside them.
Its frozen-frame potential is **(-lambda+u_s).Q-J[D]** when both searches are
stationary. Forces require additional frame/basis derivatives.

For a new water geometry the probe/external-span errors were about 4e-14 and
6e-14 Hartree. The first operator setup stopped on an integer/float charge
dtype mismatch; it was corrected before model evaluation. The first actual
pure-state/gapped-determinant constrained attempt then failed density
representability. Its reported -386.9 kcal/mol diagnostic is **invalid**.
Artifact: `coulomb_functional_water_gpu_20261116.json`. Do not treat that as
an accuracy improvement or rerun the same pure-state construction.

The necessary changed construction allows ensembles: canonical mixtures in
the fixed-N,S interacting sector and fractional one-body densities constrained
to 0<=D<=2, Tr D=N. Entropy continuation has an explicit 1e-5 Hartree energy
tolerance; no energy labels set it. On development water the interacting
moment residual reached 1.6e-13, the kinetic number/density residual 6.9e-11,
state stationarity 6.2e-9, N=10 and S^2 about -1.6e-19. Both constrained solves
are now valid. **The one-shot energy still fails badly: +305.65 kcal/mol.**
The 0.48684 Hartree kinetic mismatch at the HF input dominates. A feasible
scalar/potential is not enough to claim accuracy at that input density.
Artifact: `ensemble_coulomb_water_gpu_20261116.{json,pt}`.

The next actual job is the direct dual density fixed point: lambda=v_ext,
followed by noninteracting inversion to the matching moment density. This is
an expensive implicit physical solve, not a conventional KS timing benchmark
or learned inference. A systematic-sampling lift explicitly expresses its
fractional 1RDM as fixed-N singlet determinants. Water is development; the
protocol fixes a new H2S composition, sulfur element and N=18 before its new
reference is generated. Maximum subspace, tolerances and all failures are
fixed in `dual_fixedpoint_protocol_20261116.json`. Launch handle was exec
session 97238, bounded at 1200 seconds; it is now terminal with exit code 0.

All new numerical work used CUDA; only basis metadata/orchestration/I/O used
the host, pinned to one core. Storage before that launch was **11.1614 GB
decimal**, with a bounded 2 GB reserve. The 40 GB cap is unchanged. No frontier
method was installed/run/reproduced. No speed work was performed. The locked
32-structure confirmation contrasts remain withheld; the challenge goal is
active and unachieved.

### Terminal fixed-point, canonical-lift and derivative results

The direct dual density fixed point converged for development water and the
new H2S application. Complete STO-3G results against the same-basis exact
labels were **+0.001650 / +0.000991 kcal/mol**, respectively. Their fixed-N
interacting state residuals were 8.1e-8/4.8e-8 Hartree, moment matching
6.0e-11/1.3e-9, and KS commutators 5.2e-14/1.8e-13. H2S has **18 electrons**,
11 spatial orbitals and a 3025-dimensional fixed-Sz sector; its new reference
was generated only after prediction. No reference-energy loss or error-based
stopping was used. The direct solve used 37/108 residual directions.

An explicit canonical-lift certificate initially failed despite the valid
fractional 1RDM: its systematic-sampling breakpoints had the wrong sign.
Corrected them from fractional(cumulative occupation) to
fractional(-cumulative occupation), using saved states/densities only. **No
energy/reference solve was repeated.** Three fixed-number singlet determinants
now reproduce the water/H2S moments at 1.0e-10/1.4e-9 and natural occupations
at about 1e-10, with S^2 near zero. Old failed certificate residuals remain in
the corrected report for audit. Artifact:
`dual_fixedpoint_transfer_gpu_20261116.json`; both predictions pass the full
stated validity gate after correction.

The callable finite-entropy functional is
`xc3d/coulomb_moment_model.py::evaluate_xc`. It accepts a valid even-N
spin-unpolarized ensemble density, constructs its fixed-N canonical lift,
solves the interacting and kinetic moment problems, and returns one scalar
XC energy and its frozen-frame derivative. It rejects failed density
constraints/stationarity instead of exporting a potential. The interacting
entropy allowance and kinetic regularization have a stated energy tolerance;
the direct zero-temperature-state fixed-point scores above should not be
silently relabelled as scores of the different finite-entropy scalar.

A necessary check for this **new scalar/potential** used a water orbital
rotation that preserves 0<=D<=2 and Tr D=N. Finite-difference XC derivative
**2.0343036180 Hartree** matched the potential prediction **2.0343036141**,
difference **3.95e-9 Hartree**. All three constrained moment residuals stayed
below 3.3e-13, state stationarity below 2.1e-8 and kinetic number/density
residuals below 4.9e-10. No reference label was used or regenerated. Artifact:
`coulomb_functional_derivative_gpu_20261116.json`. Its exec session 42326 is
terminal, exit code 0; no current job remains running.

The current result is a physically consistent **small-basis implicit physical
functional prototype**, not a learned generalization breakthrough. Dense
occupation action, full-density/basis convergence, open-shell functional
definitions, automatic fragment limits, forces and balanced reaction transfer
remain unresolved. Do not repeat these successful small-basis checks or
optimize their speed. The next representation must support realistic bases
while retaining the scalar/ensemble contracts; this is required accuracy
scope. See `CURRENT_MODEL.md` for the consolidated state and next hypothesis.

The probe frame currently uses supplied nuclear positions. Fixed-Q universality
of T+W does **not** establish a single universal density-only XC functional
when Q changes between geometries. A density-selected/variational potential
frame and its derivative are a substantive next hypothesis, alongside the
larger-basis representation. Challenge eligibility is not established by
this fixed-frame prototype.

Latest full project accounting after these jobs is **11.4549 GB decimal** of
40 GB. The goal remains active and unachieved; confirmation references remain
withheld. All numerical work stayed on CUDA, no frontier replication occurred
and speed work remains deferred.

## Commands

## Larger-basis work, temporary speed authorization, and restored accuracy gate

The user resumed work with speed alongside larger tests, then clarified that
the model must be very accurate and within the goal **before further speed
work**. The latest instruction is recorded in `AGENTS.md`,
`resource_budget.json` and `ACCURACY_GATE.md`. Historical speed entry points
now stop when the policy disallows them. No speed job remains active.

### Completed work during the brief speed authorization

`gpu_four_particle.py` implements a matrix-free antisymmetric two-up/two-down
tensor operator and particle-hole conjugation. It supports four particles or
four holes, not arbitrary electron sectors. Five contractions replace ten
using exact pair antisymmetry. Cached water/N=18 H2S actions agreed with the
old operators to 1.3e-13/1.3e-12 Hartree; their references were not regenerated.
Dense multiplication is still faster on those tiny cached cases. This is a
memory/capability improvement, not an across-the-board dense-kernel speed win.

New full cc-pVDZ LiH geometries (19 orbitals, sector 29241) and Be (14 orbitals,
sector 8281) converged to 1e-8 physical state residuals. LiH Hamiltonian data
use about 1.05 MB versus an estimated 6.84 GB dense matrix. Cached-image Ritz
solves took 0.91/0.97/0.15 seconds, excluding preparation/startup; these are
own-reference numerical solves, not model transfer errors. Artifact:
`speed_larger_gpu_20261116.json`.

Static CUDA graphs plus FP32 searches/FP64 polishing retained reference
energies to about 2e-12 kcal/mol, but **were not a consistent end-to-end speed
win**: cold/reused setup costs and additional polishing outweighed kernel
gains in two cases. Do not adopt this experiment as the default or infer
CPU/A100 performance. Artifact: `mixed_speed_larger_gpu_20261116.json`.

The changed coherent state uses Coulomb-response orbital directions, paired
Slater states and a physical generalized Rayleigh quotient. Polynomial
cofactors avoid the incorrect zero-overlap couplings that an inverse-based
regularization would lose. Determinants/cofactors/Ritz mathematics are
established primitives, not a claimed new method. No frontier reproduction
was run. The first frozen 400-step full cc-pVDZ state errors were
**0.06574 / 0.07715 / 0.17643 kcal/mol** for the LiH pair/Be.

An exact occupied-subspace coordinate change and a Cholesky fast path retained
the saved physical spans to about 7e-15 Hartree, then gave matched new 400-step
errors **0.04914 / 0.08804 / 0.12364**. Total measured times were about
20.9/12.8/12.6 seconds, with accuracy changes disclosed; these are not broad
generalization or final-runtime evidence. Artifacts:
`coherent_response_larger_gpu_20261116.json` and
`fast_coherent_larger_gpu_20261116.json`. The last speed handle 11971 is
terminal, exit code 0. Do not repeat these fits or benchmark them again under
the restored accuracy-first instruction.

### Accuracy audit: fixed global coherent rank is not fragment closed

The saved LiH state has alpha/beta Schmidt rank 13 at relative 1e-10. Two
independent copies require rank 169, while a 13-term global paired Slater sum
has rank at most 13. This is a structural representational obstruction,
independent of orbital optimization. An explicit local tensor product has
exact additive energy, but a fixed global determinant budget cannot contain
it. At eight copies the maximum squared overlap at global rank 13 is about
0.9261. These fidelity bounds are **not energy-error bounds**.

No optimizer or reference was rerun for this audit. Artifact:
`coherent_composition_accuracy_gpu_20261116.json`. The fixed-global-rank
family is rejected as a final composition rule; retain its local-state kernels.

### Accuracy audit: coupled neutral singlet blocks fail strong coupling

Implemented a tensor product of complete two-electron local singlet spaces,
with exact projected direct/exchange density couplings. Local number and
metric identities hold to about 4e-16/1e-16. This includes quantum links and
independent products but excludes local triplets and charge-transfer sectors.

The frozen cc-pVDZ panel gave errors **0.04252 kcal/mol (He2), 0.96020
(distant parallel H4), and 209.96167 (close square H4)**. The balanced H4
geometry contrast missed by **209.00147**. New full-space singlet references
were generated after prediction with residuals below 1e-8. The panel took
about **21 seconds including startup**.

Before interpreting the large error, checked a coherent sum of product states
through both the block projection and independent full-coordinate Hamiltonian.
All three discrepancies were below 9e-16 Hartree; no reference was repeated.
The failure is a restriction of the tested state space, not an unexplained
projection bug. Artifact: `linked_block_accuracy_gpu_20261116.json`.
The neutral-singlet-only hypothesis is closed. Adding optimizer effort or
speed tuning is not a response to omitted charge/spin sectors.

### Accuracy audit: new 6/8/10-electron local states

The frozen five-case local-state panel used the same response seed and 400
physical-energy steps, with no reference-energy loss. It reused old BeH2
labels and generated only the new needed BH3/NH3 labels after prediction.
Model number/spin structure and independent fermionic matrix elements passed;
kernel discrepancies were below 4e-12 Hartree and S^2 stayed near zero.

| Case | Electrons | Same minimal-basis error, kcal/mol |
|---|---:|---:|
| BeH2 near | 6 | 0.47237 |
| BeH2 stretched | 6 | **2.47486** |
| BH3 planar | 8 | 0.98664 |
| NH3 pyramidal | 10 | **1.28745** |
| NH3 planar | 10 | **1.10121** |

The BeH2 stretch-contrast error is **2.00249 kcal/mol**. The supplied NH3
geometry contrast error is -0.18624; it is not an optimized barrier claim.
The rejection panel fails even in this minimal basis. It took **58.6 seconds
including startup and new references**, with about 1.37 GB peak CUDA
allocation. Artifact: `local_sector_accuracy_gpu_20261116.json`.

This is why the good four-electron results cannot justify optimization or
promotion. Do not rescue the global state by new ranks/seeds/iterations.
The next state architecture must explicitly carry local particle-number/spin
multiplets, fermionic transfer links and exact independent products. It must
then retain a scalar density-only XC definition and derivative; these state
experiments alone do not establish that functional.

All current compute handles (1235, 73121, 41349, 72823) are terminal. The
challenge accuracy goal remains unachieved. The locked confirmation panel
has not been unblinded. Latest full project accounting is about **11.854 GB
decimal**, within the unchanged 40 GB cap. All numerical work used CUDA and
host work was pinned to one core. No frontier method was reproduced.

## Commands (historical examples; current policy still applies)

Run from repository root. The guard sets host affinity and checks disk use; the Linux launcher pins the actual compute process as well. Commands are examples for a changed experiment, not instructions to repeat the completed runs.

```powershell
python electronfield-killtest/tools/resource_guard.py --reserve-gb 1 --timeout 600 -- wsl.exe -d XCResearch -u root --exec bash /mnt/c/Users/breth/myrepo/electronfield-killtest/tools/linux_gpu.sh -m xc3d.train_joint_transport --output research/CHANGED_EXPERIMENT.json
```

Implementation entry points: `xc3d/model.py`, `xc3d/pair_state.py`, `xc3d/make_references.py`, `xc3d/train.py`, `xc3d/train_pair.py`, and `xc3d/evaluate_scf.py`.

## 17 November: charge/spin state and spatial density accuracy work

The active user goal remains get below 1 kcal/mol before speed work. No
frontier method was installed or run. All new numerical work used CUDA,
with one host core and the unchanged 40 GB total storage cap.

The new state carries local electron-number/spin multiplets, all magnetic
partners, the input determinant's full Schmidt support and exact fermionic
inter-block interactions. A tensor product replaces the failed fixed global
coherent budget; local charge/spin sectors replace neutral-only blocks.
Hamiltonian residuals select response directions without reference-error
objectives. Generic tensor, spin and linear-algebra primitives are not
claimed as new mathematics.

The initial four-update nine-case panel retained a 4.146 kcal/mol close-H4
error and a large unfinished physical residual. Preserve that failure.
Continuation selected the largest residual, recovered the saved state and
reached the original 1e-5 Hartree criterion, with error about 9.45e-9
kcal/mol in cc-pVDZ. The five other unfinished panel states were completed
uniformly; completed states and old labels were not recalculated. All nine
now meet the physical criterion, with same-basis numerical errors of order
1e-8 kcal/mol or less. This is not broad chemical accuracy.

A prespecified new H2F+ ion and 18-electron peroxide tested new compositions.
The peroxide fixed-budget prediction gave a 1.57e-8 kcal/mol error but a
1.59e-5 Hartree residual at 32 updates. A subsequent exact-checkpoint
continuation added two response updates, reached 8.35e-6 Hartree and gave
3.39e-9 kcal/mol error. That continuation is post-label development; the
initial fixed-budget test is retained. It uses 37154 of 48400 fixed-Sz
product dimensions, so it is not strong compression.

Saved H4/peroxide/ionic states were exported into charge/spin Schmidt tensors
without refits or new labels. State reconstruction errors were below 3e-12
and energy changes below 2e-14 Hartree. The two-block export is a foundation,
not a general many-block Hamiltonian solver.

Two new BeH2 geometries at 1.65/2.75 Angstrom in 6-31G (13 orbitals, six
electrons) reached their original physical residual target. New references
were generated only after saving each prediction. Errors were below 7e-9
kcal/mol and the balanced stretch contrast error was -2.82e-10 kcal/mol.
Selected dimensions were 11019/10865 of 81796. The experiment took about
157 seconds including setup and needed references, with 2.934 GB peak CUDA
allocation. This split-valence basis has no polarization and does not
establish complete-basis chemistry, full GMTKN55 or the challenge goal.

A density-frame construction based on Coulomb-Gram eigenvalues failed:
external-span error reached 1.71e-5 Hartree and an orbital-coordinate
projector discrepancy reached 0.0122. The failure is retained, without a
threshold sweep. Direct SVD of normalized spatial orbital products is the
changed representation. Independent point errors are below 6e-14,
orbital-coordinate projector errors below 5e-13 and subsequently assessed
external potential span errors below 8e-13 Hartree.

Reused interacting states and a matching noninteracting ensemble search
provide on-shell finite-basis XC certificates. Energy reconstructions differ
by about 0.00035/0.00144 kcal/mol within the declared entropy allowance.
Separate fixed plus/minus potential paths give scalar/potential derivative
differences of 1.36e-7 and 5.74e-8 Hartree; all four new path state residuals
are below 9e-10 Hartree. No central fit or correlated reference was repeated.
This does not implement arbitrary-density search or prove global functional
coverage. The molecular-h-selected state projector cannot be treated as a
universal interacting functional without the missing constrained search.

A bounded universal-interface diagnostic is specified separately: provide
only T, W, Q, occupation metadata and an oracle ensemble input density,
withhold h and labels from the solver, and reject unconverged scalar/potential
outputs. Its authoritative result is universal_search_gpu_20261117.json;
do not infer convergence from a running log.

Sources: charge_spin_accuracy_gpu_20261117.json,
charge_spin_continuation_gpu_20261117.json,
stationary_panel_completion_gpu_20261117.json,
peroxide_accuracy_gpu_20261117.json, peroxide_completion_gpu_20261117.json,
spin_tensor_state_checks_gpu_20261117.json,
larger_charge_spin_accuracy_gpu_20261117.json,
intrinsic_density_certificate_gpu_20261117.json,
spatial_density_certificate_gpu_20261117.json,
spatial_derivative_gpu_20261117.json and accuracy_status_gpu_20261117.json.

Remaining requirements are a general constrained-search functional and
consistent derivatives, a many-block state/action without global Fock
storage, open-shell coverage, realistic-basis reaction-family transfer and
full WTMAD-2 evaluation. The locked 32-structure panel remains withheld.
No further speed experiment is authorized. Latest accounting is about
12.712 GB including the WSL environment/VHD; no new-drive capacity is assumed.

### Final functional-interface decision, 17 November

The universal-interface test is terminal and failed. The first cached-image
attempt was interrupted after 518.6 seconds without returning its first
solve. Reusing the existing preallocated implementation finished the inner
search but failed its residual. Both are retained separately. A singlet seed
alone was not an adequate enforcement strategy for arbitrary trial potentials,
so a polynomial in S squared now projects all trial vectors and images.

The strict singlet variant completed three ground solves, while density
residuals remained 3.78/3.63/3.52. A later trial potential hit its declared
capacity with state residual 2.52e-5 rather than 1e-8 Hartree. Its S-squared
state norm was 7.21e-16. The interface accepted no XC scalar or potential.
This fails the numerical functional implementation gate; it does not refute
constrained search or prove that this density is unrepresentable.

Independent CG states with total spin 0,1,2,3, random-vector idempotence and
Hermiticity, and a physical-Hamiltonian commutator check passed for the
singlet projector. The latter norm is 4.78e-15. Artifact:
singlet_projection_checks_gpu_20261117.json. No reference or ground solve
was repeated for these checks.

The cold pure-ground-state dual protocol is marked closed_failed, and its
entry point refuses an unchanged rerun. Do not increase Ritz size, iterations,
seeds or tolerances, or supply the known molecular potential, to rescue it.
A feasible fixed-N ensemble search and general tensor state/action are the
next substantive hypotheses; broad goal-level accuracy remains unachieved.
All compute handles from this work are terminal. Speed remains unauthorized.

### Feasible ensembles and auxiliary-density completion, 17 November

Two changed hard-density ensemble searches were run once. The thermal dual
failed its reduced density convergence, while the positive primal variant
failed its feasibility/gap checks. Both retained the original feasible
canonical ensembles and returned no accepted XC scalar/potential. Their
universal energies remained 63.431/190.155 kcal/mol above the saved oracle
values. These numerical constructions are closed; no penalty, rank, seed
or iteration sweeps are authorized. Records:
`ensemble_spatial_search_gpu_20261117.json`,
`positive_ensemble_spatial_gpu_20261117.json`. Independent sparse density
link and positive-ensemble primitive checks passed before these failures.

The changed mathematical formulation is described in
`AUXILIARY_DENSITY_STATE.md`: an established Moreau envelope on the finite
spatial-density coordinates, with distinct physical and auxiliary densities,
positive fixed-N singlet states, a matching kinetic envelope and an explicit
counterterm recovering the physical energy. The exact recovery identity,
simplex/Pauli projection and analytic proximal/derivative tests pass in
`density_resolvent_checks_gpu_20261117.json`. These are mathematical/numerical
foundations, not a claimed new theorem or replication of a frontier model.

The first forward run was interrupted when the dedicated WSL environment
stopped responding during checkpoint I/O. Memory pressure is suspected,
not proved. The only registered distribution was the research environment.
It was restarted with a 6 GiB RAM cap instead of 4 GiB and the same 1 GiB
swap, now explicitly under `.research`. Numerical work remains CUDA-only,
taskset core 0, one numerical-library thread. Original checkpoints and
interrupted report remain preserved. See
`auxiliary_environment_recovery_20261117.json`.

Two independent numerical errors were diagnosed rather than fit away:

- A response-rank cutoff below floating-point resolution admitted 46
  directions for an exactly rank-one source. The repaired relative
  machine-precision test gives rank one and retains a genuine threefold
  degeneracy. Before/after records are `response_rank_*_gpu_20261117.json`.
- The first completion used P H P without ensuring that every response
  basis vector was singlet. A saved candidate's reported energy differed
  from its literal physical-Hamiltonian expectation by 28.484 Hartree;
  its S-squared expectation was 5.545. It never passed the physical gate.
  The job was stopped and the protocol is `closed_invalid_spin`.
  `auxiliary_spin_failure_gpu_20261117.json` preserves the diagnosis.
  The corrected code uses the literal Hamiltonian, explicitly purifies
  every extension and checks spin before energy projection. Independent
  injected-triplet removal/rejection and raw operator tests pass in
  `auxiliary_spin_guard_checks_gpu_20261117.json`. Invalid checkpoints are
  excluded; only occupied physical states from valid checkpoints were kept.

A method/attribute name collision in postprocessing was also repaired
without repeating the saved state optimizations. The first corrected
bounded completion still missed its original full residual tolerance,
despite small energy errors, and remains marked invalid in
`auxiliary_strict_gpu_20261117.json`. One checked occupied-state restart
then completed the original residuals. It retained density, trace, physical
energy and the original mean-field input to numerical precision. Epsilon,
space cap 192, response counts, physical objective and every accuracy
tolerance stayed fixed. Reference errors did not determine stopping.

Terminal accepted results in `auxiliary_stationary_gpu_20261117.json`:

| BeH2/6-31G | Same-basis error, kcal/mol | Full residual, Hartree | Potential residual | KS commutator |
|---|---:|---:|---:|---:|
| 1.65 Angstrom | +1.964e-8 | 9.279e-6 | 7.008e-8 | 1.240e-7 |
| 2.75 Angstrom | +4.801e-8 | 6.745e-6 | 1.101e-8 | 1.112e-8 |

Number, Pauli occupations, singlet spin, kinetic search and the additional
lowest-state Ritz checks pass. The latter residuals are below 9e-9 Hartree;
they are numerical evidence, not independently implemented rigorous lower
bounds. No correlated oracle density or new reference entered predictions.
These are post-interruption development completions, not a blind holdout.

Fresh auxiliary-density directions with step 0.001, fixed response spaces
and no reference/potential supplied to universal evaluation give XC scalar
versus derivative differences 1.579e-8/1.408e-7 Hartree. Both pass the
prespecified 1e-5 criterion in `auxiliary_derivative_gpu_20261117.json`.
This tests the frozen-space scalar derivative, not nuclear forces or
arbitrary full-space density paths.

The global vector length remains 81796; response dimensions 29/191 are not
a scalable-compression result. The backend covers only four/six particles
or holes and even-N singlets. No realistic-basis reaction benchmark,
learned/general out-of-sample predictor, complete-basis accuracy, WTMAD-2
or AlphaFold-scale breakthrough is established. A compositional physical
state/action without global Fock storage is the next accuracy requirement.
Challenge integration of the regularized KS formulation is unverified.
The locked composition panel remains withheld. No speed work is authorized.

Storage accounting was corrected to include about 3.06 GB of external WSL
research crash dumps and a conservative 1 GiB old-swap reserve, plus all
project-local environment/VHD/swap, checkpoints and scratch. Current total
is about **20.0 GB of 40 GB**, working ceiling 32 GB. No files were deleted
and no 2 TB drive capacity was assumed. Authoritative current snapshot:
`resource_accounting_20261117.json`. All numerical job handles are terminal.

### Compositional tensor scope and physical response metric, 17 November

Previous goal turn classification: progress. It completed independent
functional derivative checks, preserved numerical failures, closed failed
hard-density routes and repaired complete resource accounting. The full
challenge accuracy goal remained active and unachieved.

The changed state/backend is recorded in `TENSOR_DENSITY_RESPONSE.md`.
Local spinless ancillas purify a positive physical ensemble; reduced CG
tensors enforce number and total spin. A sparse fermion-word automaton
contracts the literal Hamiltonian without a global occupation vector or
integral truncation. These established mathematical primitives support
our density-response hypothesis; no frontier implementation or benchmark
was installed, run or replicated.

Independent checks in `purified_operator_checks_gpu_20261117.json` pass:
full four-orbital operator error 4.0e-15 Hartree; energy/parameter gradients
below 9e-16; exact N/S for mixed singlet and odd-electron doublet states;
same-family fragment embedding; and a supplied eight-fragment product
with 32 electrons in 24 orbitals. Product energy additivity error is
4.9e-15 Hartree with 576 tensor parameters. The hypothetical global
fixed-Sz vector would occupy 4.33 TB; it was not allocated. This is a
supplied-independent-product identity, not automatic size consistency or
coupled-molecule accuracy.

Complete-multiplet Hamiltonian response tests pass at roughly 2e-14 image
error. An initial diagnostic incorrectly called a cap of 256 untruncated;
the doublet needed 262 and failed that cap. Only its unfinished checks were
completed using the algebraic full prefix-space bound, without changing
the operator, state, seed, algorithm or tolerances. Original failure and
completion remain separate: `spin_response_checks_gpu_20261117.json` and
`spin_response_capacity_completion_gpu_20261117.json`. This was not a
chemical rank sweep. Deliberately severe truncations remain physically
valid but have large image error, recorded without a convergence claim.

A new coupled water geometry uses the complete 24-orbital cc-pVDZ basis,
10 electrons, and no frozen core. A full fixed-Sz vector would occupy
14.45 GB and was not allocated. GPU integrals and one bounded CUDA RHF
input were saved; CPU integral entry points were blocked. The literal
operator retains every nonzero term and uses 17.64 MB CUDA storage. Its
fixed-N constant-shift energy identity agrees with Wick contraction to
about 1.9e-14 Hartree.

The first response selected on raw MPO-channel Euclidean norms. Although
its two-state energy/number/spin checks passed, an independent physical
image assessment subsequently found its span retained only 2.706% of the
squared response norm. This metric is closed for chemistry use; increasing
its bond cap is not the next action. Original report:
`tensor_water_scope_gpu_20261117.json`.

The changed metric builds a physical right frame and retains complete spin
multiplets by physical Schmidt weight. Sparse physical Gram contraction
measures the full Hamiltonian-image norm without a global vector. Exact
singlet/doublet image-loss identities and projection bounds pass below
4e-15 in `physical_response_metric_checks_gpu_20261117.json`. The current
exact metric requires a product source; it does not certify a correlated
candidate's complete residual. Bounds are numerical float64 evidence,
not interval-arithmetic proofs.

At the unchanged bond cap 64, the physical metric's two-state water span
retains 98.610% of the initial squared response norm. The image relative
error estimate is still 0.117888. Its right-frame numerical triangle bound
is 1.33e-12. The two-state total energy is -76.1692023251 Hartree, versus
input -76.0240386654 and original raw-channel result -76.0271374064.
These are variational energies, not reference errors. No correlated
reference was loaded/generated and no input SCF or failed selection was
repeated. Particle number, singlet spin, literal energy reconstruction,
Hermiticity and input retention pass. The changed-metric run used about
0.793 GB peak CUDA allocation and 84 seconds including setup; this records
resource cost, not a speed benchmark. The state remains unconverged and
no chemical-accuracy or WTMAD-2 result is accepted. Record:
`tensor_water_physical_metric_gpu_20261117.json`.

CUDA SVD reported nonconvergence of its initial cuSOLVER driver at one
right-frame step and completed using its built-in more accurate CUDA
driver. This is the documented gesvdj-to-gesvd fallback in
[PyTorch 2.7](https://docs.pytorch.org/docs/2.7/generated/torch.linalg.svd.html),
not a CPU numerical fallback. No library installation/tuning or rerun
was performed in response to that warning.

The new literal one-body derivative supplies self/transition spatial
density matrices without full Fock links. Independent singlet/doublet
checks agree below 9e-16, including cross-number identities. On the saved
water state, Tr D error is 1.78e-15 and occupations range from 3.57e-5 to
1.99838. No state optimization was repeated. Records:
`tensor_density_checks_gpu_20261117.json`,
`tensor_water_density_gpu_20261117.json`.

The tensor state/action/density must next be integrated into the universal
T+W auxiliary functional with controlled correlated-source response and
consistent scalar/potential acceptance. It is not yet a general XC
evaluator or learned predictor. Realistic-basis relative energies,
independent chemistry, open-shell functional coverage and the full
sub-1 kcal/mol WTMAD-2 target remain unverified. The locked composition
panel remains withheld and speed work stays disabled. Latest complete
storage is about 20.04 GB of 40 GB. All numerical job handles are terminal.

### Conditional tensor functional connection, 17 November

Previous goal turn classification: progress. It supplied a physical tensor
state/action/density backend and changed physical truncation metric, while
retaining the broad accuracy goal. This turn connects that state to the
universal auxiliary-density objective in an explicit conditional section.
The mathematical definition and limits are in
`CONDITIONAL_TENSOR_FUNCTIONAL.md`.

`gpu_local_tensor.py` resolves physical local tensor intertwiners, exact
norm/operator environments and their true overlap. Direct boundary and
embedding SVDs preserve weak modes without squaring their condition number.
The literal fixed-number operator independently checks the physical metric.
`tensor_auxiliary_model.py` accepts only T,W,Q,N,S as universal operators;
the tensor context is explicit and separate. A positive trace-one reduced
ensemble gives a positive global physical state when traced over other
ancillas. The fixed-context auxiliary problem is convex and has a consistent
scalar derivative; no global constrained-search theorem is asserted for a
finite tensor optimizer.

Independent synthetic tests in
`local_tensor_functional_checks_gpu_20261117.json` pass on mixed singlet
and doublet sections (dimensions 40/36). Literal H and Q matrix differences
are below 4e-15, physical overlap/input retention pass, and positive searches
meet the original 1e-11 Hartree gap. Installing their local purifications
reproduces energy/density below 5e-15. The even-singlet conditional XC
directional difference is 5.13e-8 Hartree. Odd-spin operator evidence does
not establish an odd-spin kinetic/XC functional.

A separate analytic interrupted positive-state problem checks optimizer
recovery. Its saved FISTA point, state, acceleration, energy and history
continue without duplicate steps; final analytic state error is 3.94e-11.
A completed checkpoint takes no further iterations. No chemical/reference
solve was repeated. Record: `resolvent_checkpoint_checks_gpu_20261117.json`.

The larger-basis off-shell test uses the saved 10-electron water state and
all 24 cc-pVDZ orbitals. Its new direct spatial frame has 280 operators,
independent sample-span error 3.97e-14 and subsequently assessed physical
potential-span error 3.08e-12 Hartree. Frame selection uses basis products,
not the external potential. The fixed local center has 692 physical
directions from 719 reduced coordinates. Overlap and input-retention errors
are about 2.5e-12/4.0e-14. No global occupation vector or link table is used.

The conditional positive search starts from the saved unconverged state
at x=q_saved-epsilon v. The external v is used only by the outer physical
test. It is not supplied to the universal evaluator, and no correlated
oracle/reference label is loaded. At iteration 1100 it reaches gap
5.52e-12 Hartree; the matching kinetic gap is 1.42e-14. Installing its
rank-one local purification preserves density moments to 1.95e-13, literal
T+W energy to 5.57e-12 Hartree and number to 1.78e-15. Occupations are
0.000151 through 1.999857. Physical-energy recovery differs by
5.59e-13 Hartree and physical monotonicity passes.

The state energy changes from -76.1692023251 to -76.2116024360 Hartree.
These are variational energies, not chemical reference errors. Fresh
plus/minus auxiliary coordinates with the surrounding tensors held fixed
give an XC scalar/potential difference 7.37e-9 Hartree at step 0.001.
All declared conditional gates pass in
`tensor_water_conditional_gpu_20261117.json`. The run took about
592 seconds including compilation with 1.64 GB peak CUDA allocation;
these are resource observations, not timing-target evidence or speed work.

Compiled operators are stored as 35 immutable chunks. Exact recovery in
`tensor_section_recovery_gpu_20261117.json` restores state, scalar and
gradient with zero local operator matrices recompiled and zero new optimizer
iterations. The universal MPO is rebuilt from saved T,W integrals for recovery.
An independently recomputed conditional gap remains 5.52e-12 Hartree.
The installed state and completed optimizer checkpoints are preserved in
`.research/data/tensor_local_water/center6/`. No unchanged single-center
solve should be repeated.

Surrounding-tensor variation contributes to a global derivative unless
common variational stationarity is established. This result does not prove
global tensor stationarity, context/initialization independence, a complete
universal functional, broad chemical transfer, complete-basis accuracy or
WTMAD-2. The locked confirmation panel remains withheld and speed work
remains disabled. Next establish common stationarity and independent
realistic-basis relative energies; do not replace those with more local
certificates or seed/rank/iteration sweeps.

All numerical work stayed on CUDA, with one host core and no frontier
replication. The model uses established purification, CG, tensor and Moreau
mathematics; no AlphaFold-scale breakthrough is established. Complete
storage is about 21.2 GB of the unchanged 40 GB allocation, working ceiling
32 GB. No new-drive capacity is assumed. All numerical handles are terminal.

## Variable physical sectors and a new larger-basis accuracy protocol

17 November 2026, later work. The preceding terminal-handle statement
describes the conditional experiment only. The new BH protocol below is
running, with one CUDA job and no speed work. The detailed derivation,
failed paths and latest scope are in
[the physical sector record](TENSOR_SECTOR_COMPLETION.md).

The first all-site physical/auxiliary forward implementation passed
independent small singlet/doublet checks, but its molecular run stopped
at center 4 after four valid updates: overlap error 5.10e-8 exceeded the
unchanged 1e-8 guard. Inspection also exposed a decisive restriction:
its fixed seed charge labels permit at most two electrons in virtual
orbitals. That family is closed; additional one-site sweeps cannot admit
triples and higher excitations. The valid warm state and failed report
remain preserved.

An inverse-singular-value gauge recovery preserved physical observables
but failed the canonical guard at 2.31e-8. Direct reduced-tensor SVD
reconstruction then passed: energy/density changes 7.11e-14 Hartree and
2.91e-15; canonical overlap error 2.96e-14. No previous physical update
was repeated and no tolerance was relaxed. Both outcomes remain recorded.

The changed two-site CG section includes every allowed intermediate
charge/spin sector, independently of the old middle bond. Literal
two-site Hamiltonian action agrees with an independent full matrix to
1.78e-15 Hartree. Residual-selected new physical directions retain the
old state exactly and all magnetic partners. Their independent new
variational-update test agrees with the complete local minimum to
1.44e-15 Hartree, with number/spin/occupation checks passing.

At the same water bond cap of 64, eight new magnetic states fill unused
capacity after the existing 56. They capture 75.00% of the residual
component outside the old left support, not the full global residual.
Before optimization energy is identical and density changes by
2.25e-18. The new local update lowers energy from -76.2129045296 to
-76.2151646370 Hartree, with 1.10e-5 weight in triple/higher virtual
excitations. Physical guards pass. This proves a previously excluded
physical response matters; it is not a reference-error score or full
stationarity certificate. Source: `tensor_water_sector_gpu_20261117.json`.

A bounded changing-sector two-site solver now rejects energy-increasing
retractions and audits every pair on one frozen final physical state.
Independent discarded-norm/fidelity tests pass below 1.3e-15. A new
six-orbital/six-electron check matches the complete singlet reference to
reported floating precision, with maximum common-pair gap 3.20e-14
Hartree, six sweeps, 30 distinct updates, five audited pairs and correct
checkpoint recovery. Known tensor/CG/Krylov mathematics is not claimed
as a scientific breakthrough.

The frozen BH protocol uses new 1.20/1.80 Angstrom geometries, all 19
cc-pVDZ orbitals, six electrons, a predeclared 128-state bound and ten
maximum sweeps. Both predictions must pass their original physical and
common-context gates before either reference is generated. References
use an independent six-particle CUDA action and singlet projection.
Before any label existed, a reference-only correctness amendment added
a generic complete-singlet component to its single fixed starting vector
to avoid determinant-only spatial-symmetry restrictions. Model, capacity,
stopping and acceptance settings stayed frozen; completed prediction
updates and CUDA inputs/RHF were restored without repetition.

The official challenge page was rechecked: def2-QZVP and its public/hidden
accuracy/evaluation requirements remain recorded in
`challenge_2026-11-17_verification.json`. A new cc-pVDZ stretch pair cannot
stand in for that target, complete-basis chemistry or WTMAD-2. Global
arbitrary-density XC stationarity, broad reaction transfer, open-shell
functional coverage and learned generalization remain unproved. The
locked confirmation panel is still withheld. Complete storage is about
21.2 GB of the unchanged 40 GB; no new drive is assumed.

## Doublet support prepared while the frozen BH job continues

17 November 2026, subsequent continuation. The live BH GPU handle was
revalidated, not restarted. Its first geometry is in sweep 5 with physical
checks passing at completed sweeps; neither prediction has a terminal
accuracy score and no reference has been generated. The original ten-sweep,
128-state protocol and thresholds remain unchanged.

The larger-basis solver's singlet-only scope is an actual gap for broad
chemical transfer. `gpu_doublet_embedding.py` now implements a physical
odd-electron doublet via a singlet purification with one inert auxiliary
spin. The physical-prefix N,S and auxiliary occupancy are fixed exactly.
Physical density, spin and electron-count reports exclude that auxiliary.
The two-site solver now accepts a constrained problem builder, with its
default singlet calculation unchanged. No numerical open-shell claim is
made yet.

The new test independently traces the auxiliary from a full tiny
wavefunction and checks the resulting physical ensemble, energy and
doublet reference. An independent three-electron Cartesian reference
action and sparse doublet projector are also implemented and included
in its matrix checks. Python compilation and source whitespace checks
pass. This numerical test is **queued, not executed**, because the BH
job retains exclusive scientific GPU use. See
[the doublet domain](DOUBLET_TENSOR_DOMAIN.md). No established singlet
experiment, integral/RHF preparation or correlated reference was repeated.

These are established spin/purification/reference primitives addressing
physical validity; they are not a scientific breakthrough or validated
generalization result. The full sub-kcal/WTMAD-2 goal remains active and
unproved. Storage remains about 21.2 GB of 40 GB, with one host core,
CUDA-only science, no frontier replication, no speed work and no new-drive
assumption. Refresh the consolidated evidence only after the numerical
GPU job is terminal; the live molecular report is authoritative meanwhile.

## Resumed frozen accuracy run and complete public benchmark data

17 November 2026, after the user reactivated the goal. GPU availability,
the paused report and saved forward checkpoint were rechecked. The
original BH experiment resumed after saved sweep 5, center 6 without
repeating completed preparations or updates. Only its run-status metadata
was repaired to record resumption; model, capacity and acceptance
thresholds remain unchanged. The host watchdog allows four hours while
the scientific protocol retains its original ten-sweep cap. There is one
scientific GPU worker. The doublet test remains queued behind it.

The public benchmark is now concrete rather than an unspecified later
suite. Primary-source discovery found that the old university download
links time out, while the authors' GitHub repository remains accessible.
Its `v1` branch is pinned to commit
`8d485b37a1ca8837e395042671ca5ba4e0714691`. This version preserves the
original benchmark with its documented average correction; `v2` changes
references and removes cases. The score version and weighting convention
must be stated explicitly. No version/case filtering can stand in for
the full original accuracy goal.

The data-only allowlist retains 6,368 files, with no published-method
programs, calculation results, model weights or environments. There are 2,462
structures and all 55 evaluation sets; BH76RC shares the BH76 structures.
Every file matches its pinned Git blob and SHA-256 hashes in an independent
metadata audit. Charges, elements and spin parity metadata are consistent.
Coverage includes 1 through 360 physical electrons, up to 81 atoms,
anions/neutrals/cations/dications and reported unpaired counts 0 through 3.
This makes the gap to broad chemical validation explicit, not smaller.

Coordinate/reference-energy arithmetic and benchmark weights were not
evaluated. Host work was limited to one-core byte transfers, source
hashing and integer metadata bookkeeping; no scientific CPU fallback ran.
The locked confirmation panel was not accessed. The source snapshot is
about 9.3 MB and total project usage remains about 21.2 GB of 40 GB.
Sources and remaining CUDA numerical checks are in
[the benchmark record](BENCHMARK_DATA.md). All model/WTMAD-2 goal flags
remain false, and speed work remains disabled.

## Physical lowest-state start guard prepared

17 November 2026, later continuation. Source review exposed an exact
counterexample to interpreting a retained-only Krylov residual as the
lowest state: H=diag(1,0), initial vector (1,0), returns zero residual at
energy 1 while a lower state exists. This is an analytic inspection
result; the new CUDA counterexample test is not yet run. It does not
demonstrate a wrong sector in the current BH experiment, which remains
unchanged and continues to require independent chemical references.

A new optional start policy uses a single fixed generic physical
component at relative amplitude 0.5. There is no seed sweep or reference
selection. Literal physical guards, capacity and residual tolerances are
retained. Existing protocols default to their original policy; only the
as-yet-unexecuted doublet protocol selects the new policy. Python source
compilation passes, but numerical acceptance is pending. The execution
queue is now the new spectral guard, then the doublet test, both after
the current BH GPU job is terminal. See
[the exact mechanism and scope](RITZ_START_GUARD.md).


## 20 November 2026: GPU Kohn-Sham pipeline replaces the tensor-state route

The user handed over control of the direction and repeated the GPU-only rule;
the D drive is now authorized for space. Full record:
[GPU_DFT_PIPELINE.md](GPU_DFT_PIPELINE.md). Code: `xcgpu/`. Bulk data:
`D:/xcresearch`.

**Decision (agent's, under that delegation):** stop extending the tensor-network
state as the evaluated model. Its measured public coverage was 5 of 1,505
reactions with 2,417 of 2,442 structures blocked. The active model is now a
functional of the PBE Kohn-Sham determinant, evaluated after a GPU PBE SCF.

**What was actually blocking GPU DFT:** not the `libcusolver` import error the
earlier log recorded. `gpu4pyscf 1.9.2` segfaults in its GGA routine when
paired with functional-library wheel 0.9.0; LDA works, any GGA crashes, with
all installed files matching their install hashes. Wheel 0.8.1 fixes it
[measured: same direct GGA call crashes with 0.9.0 and passes with 0.8.1].

**Measured so far (all on the RTX 4060, def2-QZVP, density-fitted Coulomb,
grid level 3):**

- Timing of PBE SCF per structure: 2.4 s at 150 basis functions, 4.3 s at 441,
  14.8 s at 700, 38 s at 1,000, 83 s at 1,482, 165 s at 2,103. The fitting
  tensors are never stored, so 8 GB of GPU memory is enough at every size.
  Integral-direct Coulomb is about 6 times slower and was not used.
- Inventory: 2,442 required structures, median 438 basis functions, largest
  3,420; 314 open-shell; 81 with ECPs.
- Full-suite PBE run `pbe_qzvp_g3` started, smallest structures first. One
  structure (singlet C2) failed every DIIS and level-shift stage; the level
  shift lands on a state 0.1 Hartree too high. A second-order solver stage
  converges it (-75.826090 Hartree) and is now part of the retry schedule. The
  failed records are kept under `superseded/`.
- Response-spectrum pass validated on water: second-order term of the spectral
  integral equals twice the independent opposite-spin PT2 energy to 4e-7
  Hartree after an end-correction fix (first version was off by 1e-4).
- Energy assembly check: PBE atomization energies of N2 and H2 are 243.2 and
  104.7 kcal/mol; exact exchange plus RPA on PBE orbitals gives 220.6 and
  108.8. These agree with published values for those established methods as
  recalled, not re-verified against a source here.

**Not yet measured:** any full-suite score for any model. Partial-coverage fit
numbers are kept out of this log until the coverage is large enough to mean
something; early fits on fewer than 100 reactions with 20 to 44 parameters were
shake-outs only.

**Host bottleneck, stated plainly:** for small molecules the GPU sits near 20%
utilization because one pinned host core does the orchestration. This is the
rule in `AGENTS.md`, not a hardware limit. Allowing a few orchestration cores
would shorten wall time; that is the user's call.
