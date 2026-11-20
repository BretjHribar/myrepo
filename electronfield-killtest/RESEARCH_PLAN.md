# Research strategy for the speed–accuracy challenge

Decision record: **16 November 2026**. Objective: win the [Discovered Materials challenge](https://discoveredmaterials.com/grand-challenges/speed-accuracy-barrier/). This is a proposed research program, not a demonstrated breakthrough or a prediction of winning.

**Latest user allocation and priorities:** **GPU-only numerical work, 40 GB total project disk, and at most one CPU core for unavoidable host orchestration and I/O. Accuracy below 1 kcal/mol comes first; optimize speed afterward. Own research only. No frontier replication, reproduction, retraining, or frontier sanity checks, now or later.** Implement changed physical hypotheses, reuse cached inputs and labels, and run their numerical work on CUDA without CPU fallback. Avoid repeating established checks; record failed ideas and move on. See [the resource budget](research/resource_budget.json) and [repository instructions](../AGENTS.md). Older CPU runs below are historical evidence, not authorized next actions.

The user briefly authorized speed work alongside larger tests, then explicitly
restored **accuracy within the goal before further speed work**. The completed
speed experiments are historical; speed entry points now check the current
policy. The [accuracy gate](research/ACCURACY_GATE.md) is the current priority.

**Latest user steering:** get the complete public tests working and an
initial measured assessment first, then pursue sub-1 accuracy. The user
explicitly selected full public GMTKN55 before private organizer tests.
All 55 sets / 1,505 reactions now parse and pass independent CUDA scorer
checks. Model coverage and actual predictions remain missing; no full-suite
kcal/mol estimate is accepted. See [the pipeline record](research/GMTKN55_PIPELINE.md).

The [full public benchmark data](research/BENCHMARK_DATA.md) is now pinned
to the authors' original-suite `v1` commit and independently audited:
2,462 structures, all 55 evaluation sets, charges/spins and unchanged
reaction definitions. No numerical reference weights or model scores were
evaluated during acquisition. Source integrity does not establish accuracy.
Match the stated reference version and corrected weighting convention,
retain all cases, and keep the separate locked confirmation panel withheld.

**Central bet:** construct a transferable many-electron correlation state with exact particle/Pauli constraints, then use its density-constrained universal energy to define a nonlocal XC functional. The current wavefunction experiments test that physical representation; they are not themselves a submitted XC functional. Learn/amortize the state only after electron-count and chemistry transfer are convincing. Accuracy and a defensible out-of-sample model come before speed work.

Implementation and findings are recorded in [the current model status](research/CURRENT_MODEL.md) and [execution log](research/PROGRESS.md). The fixed global coherent and neutral-singlet block restrictions are closed after informative failures. Their replacement carries local charge/spin multiplets and residual-selected physical response directions. It now closely matches small same-basis references, including new ionic and peroxide cases. Some spaces are nearly full; a tensor export preserves the states but is not yet a large-molecule solver. Next establish larger-basis relative energies, a general compositional state implementation and its density-constrained scalar/derivative. These are accuracy/scope work. No broad out-of-sample or goal-level accuracy claim is established, and speed tuning remains deferred.

**17 November accuracy milestones:** the new BeH2/6-31G geometry pair passed
its prespecified state and relative-energy checks. Direct spatial-density
product coordinates passed independent point/orbital-gauge checks, followed
by on-shell kinetic completion and scalar/potential directional derivatives.
The universal-interface experiment excluded molecular h and labels but
failed its state/density convergence gate, returning no XC result. The cold
pure-state dual protocol is closed; on-shell certificates do not establish
an arbitrary-density predictor. The next scope work is a feasible fixed-N
ensemble constrained search with physical response spaces, a
many-block tensor action without a global Fock vector, density-only selection
of local response spaces with all basis-response terms, and new realistic-
basis reaction-family tests. Do not keep extending tiny exact-solve tables,
tune closed models or start runtime/learned-solver optimization from these
development results. A novel transferable representation is still required;
generic exact electronic-structure solvers are diagnostics, not the breakthrough.

**Latest functional milestone:** both bounded hard-density ensemble searches
also failed and are closed. A changed finite-coordinate
[auxiliary-density formulation](research/AUXILIARY_DENSITY_STATE.md) now
passes two BeH2/6-31G forward development points, original physical and
regularized KS checks, and fresh frozen-space scalar/derivative tests.
The physical energy is recovered with an explicit counterterm; auxiliary
and physical densities differ. This is not an unchanged conventional KS
plugin, and challenge integration/eligibility remains unverified. An early
spin/energy implementation error was diagnosed and rejected; all failed
reports remain preserved. The corrected development completions do not
establish broad chemical transfer. Current priority is a compositional
physical state/action supporting realistic systems without global Fock
vectors, while preserving the functional and physical acceptance contracts.
Established regularization/tensor/optimization mathematics alone is not
our proposed breakthrough. No further tiny-case table or speed experiment
substitutes for independent realistic-basis accuracy evidence.

**Latest state/scope milestone:** the new
[compositional tensor state/action and density interface](research/TENSOR_DENSITY_RESPONSE.md)
now passes independent mixed singlet/doublet operator and density checks,
an exact supplied 32-electron fragment product, and coupled 10-electron
water in all 24 cc-pVDZ orbitals. It avoids global Fock storage. A changed
physical response metric retained 98.61% of the initial squared image norm
at the same bond cap of 64, versus 2.71% for the closed raw-channel route.
The molecular state remains unconverged and has no reference accuracy
score. Tensor response for correlated sources and integration into the
universal T+W auxiliary optimizer are the next accuracy requirements.
This is established tensor mathematics supporting our research state,
not a validated transferable functional or a demonstrated breakthrough.
The accuracy gate, withheld panel, GPU-only/one-core policy and 40 GB cap
remain unchanged. No further speed work is authorized.

**Conditional functional connection:** a positive local tensor section now
optimizes the universal T+W auxiliary objective and supplies a verified
fixed-context XC scalar/derivative on the 24-orbital water case. Its local
gap is 5.52e-12 Hartree and directional-derivative difference 7.37e-9.
Installed states preserve literal physical energy/density, and completed
checkpoints restore without repeating work. See
[the conditional functional](research/CONDITIONAL_TENSOR_FUNCTIONAL.md).
This is not a globally self-consistent or context-independent functional;
no chemical reference error or WTMAD-2 was measured. Common stationarity
across tensor environments and independent realistic-basis relative-energy
accuracy are the next requirements. Do not repeat the single-center
certificate or resume speed work from this result.

**Physical restriction and changed next family:** the first all-site
tensor pass stopped at its unchanged metric guard after four valid
updates. Its fixed seed sectors also exclude triple and higher virtual
excitations, so additional one-site sweeps in that family are closed.
Direct-factor canonical rebasing now passes state/energy/density guards
without repeated optimization. A new
[two-site sector-completion experiment](research/TENSOR_SECTOR_COMPLETION.md)
uses the literal physical residual to add missing N,S multiplets at the
same bond cap while retaining the old state exactly. Independent new
operator and variational-update tests pass. These are established tensor
primitives supporting the physical model; energy lowering and local
stationarity do not establish chemical accuracy, transfer or a breakthrough.
The next gate remains independent realistic-basis relative energies after
common stationarity, with the locked confirmation panel withheld.

**Open-shell gap, pending:** the larger-basis forward solver currently
accepts singlets. A [doublet-domain implementation](research/DOUBLET_TENSOR_DOMAIN.md)
uses an inert auxiliary spin and exact physical-prefix constraints;
physical observables exclude that auxiliary. An independent partial-trace,
physical-spin and three-electron-reference test is prepared and must run
after the current BH GPU job, with no concurrent numerical work. No
open-shell result is accepted until that new gate passes; the existing
BH prediction/reference protocol and thresholds remain unchanged.

## 1. Define a result worth pursuing

The [dated challenge snapshot](research/challenge_2026-11-16.json) records the published evaluation and provisional chart. Confirm the unanswered entry/protocol questions there before freezing a submission. No organizer contact has been made.

**Target-box alignment:** the chart supplied by the user places the target below **1 kcal/mol WTMAD-2** and at approximately **10 CPU seconds per molecule or less**. Those two conditions must be met together on the organizer's suite and timing setup. The time boundary is read from the plot, not a published numerical entry rule. Aim for at most **2× the cost of our required PBE input calculations** as a stronger speed objective inside the box; verify the absolute timing independently. The chart's CPU threshold does not specify an A100 time threshold. Published comparator timings are context only and do not create reproduction tasks.

Our internal targets are intentionally more demanding than beating an old baseline. They are research goals, not organizer cutoffs:

| Dimension | Target for a convincing step change |
|---|---|
| Target-box accuracy | Organizer-suite WTMAD-2 **below 1.0 kcal/mol**, established by independent organizer evaluation; development proxies cannot establish this result |
| Broad accuracy | Full GMTKN55 WTMAD-2 **below 1.0 kcal/mol**, with each of its five major categories reported |
| Transfer | At least **2× lower relative-energy MAE** than our own energy-only control on independently locked reaction-family and precursor-family tests; show per-family errors and uncertainty. The absolute sub-kcal target remains essential. |
| Reference efficiency | Aim to reach our own energy-only control's error with **at most one quarter of its expensive reference cost**, counting perturbation calculations and all density/inversion work |
| CPU target-box cost | Complete calculation **at most approximately 10 CPU seconds per molecule** under the organizer's one-core protocol and suite aggregation; include SCF, features, transfers, and retries |
| Relative cost | Prefer **at most 2× our required PBE input calculations on CPU** at matched numerical quality. This supplements the absolute target; no frontier-comparator execution is part of the plan. |
| Reliability | At least **99.5% SCF success** on the full evaluation; no accuracy gain purchased through omitted failures or a few extreme outliers |
| Practical quality | Report 95th-percentile and maximum errors, density/charge defects, iteration counts, peak memory, and grid sensitivity |

An intermediate gate can justify more research without meeting these final targets. A 5–20% gain alone does not justify calling the project successful. Pursue accuracy first even with an expensive construction; optimize cost once its generalization is convincing. CPU measurements are deferred under the current GPU-only instruction. Laptop timings cannot establish A100 performance, and public-set scores cannot establish hidden-set performance.

## 2. Our research direction

Build a compact electronic interaction model that can transfer its learned physics across new molecules, electron counts, and reaction families. Spend the compute budget on three original experiments:

1. **Collective nonlocal state:** let spatial regions exchange density and spin information, then derive one energy and its self-consistent potential from that state. Test whether it repairs failures a local correction cannot represent.
2. **Counterfactual electronic supervision:** teach the model how electrons rearrange under charge-transfer and spin perturbations, not only the energy at a stationary geometry. Measure whether that information improves transfer per reference-computation cost.
3. **Composing difficult limits:** connect one-electron self-interaction, fractional charge/spin, and stretched bonds in the same model. Try explicit ensemble/state structure if a smooth density network cannot represent the required changes.

Run our own small, informative experiments. Use reference calculations as labels and our own ablations to identify which idea works. No frontier checkpoint downloads, installation projects, benchmark reproductions, or retraining jobs. Literature can inform an idea or qualify novelty; it does not create a replication task.

The AlphaFold-inspired ambition is a reusable physical representation that succeeds on blind generalization, with a substantial jump in practical accuracy and cost. A successful internal ablation is only a reason to scale the idea; the target remains the actual sub-kcal, fast-CPU rectangle.

## 3. What the existing experiment does and does not establish

The rerun in [README.md](README.md) supports studying field supervision. It also exposes the central danger: apparent accuracy can reverse when a learned potential makes SCF harder. Both arms retain large stretched-bond errors. More seeds or a larger field weight will not resolve the competition's generalization problem.

Specific barriers in the implementation:

- `prototype.py` assumes two opposite-spin electrons in one dimension. Its `-E_H/2` exchange identity is specific to this setting and cannot be ported to general molecules.
- Reference potential inversion uses the single occupied spatial orbital. General 3D inversion is a separate inverse problem.
- Training uses 24 fixed SCF iterations, then a gradient that assumes stationarity. Unconverged densities can invalidate that approximation.
- Frozen Gaussian density descriptors and energy supervision do not explicitly resolve global charge-transfer steps or fractional-spin behavior.
- The recorded molecular totals do not establish reaction/barrier accuracy or WTMAD-2.

Preserve this experiment as a regression and diagnostics sandbox. Cap additional toy-only work; build the 3D path early.

## 4. The scientific hypothesis and first architecture

**Hypothesis:** learning how density and spin respond to a small set of controlled perturbations identifies transferable nonlocal XC behavior more efficiently than fitting stationary energies and fields alone. A multiscale interaction representation should allow that behavior to compose across unseen chemical environments.

Start with an ordinary spin-density functional:

\[
E_{xc}^{\theta}[n_\uparrow,n_\downarrow]
=E_{xc}^{\mathrm{PBE}}[n_\uparrow,n_\downarrow]
+\Delta E_\theta[n_\uparrow,n_\downarrow].
\]

PBE supplies a working local KS starting point. It is not the target performance baseline. Separate exchange and correlation corrections so their distinct scaling conditions can be tested. Retain conventional kinetic, external, Hartree, and nuclear terms in the electronic-structure solver.

Implement the correction in three parts:

1. **Local electronic features.** Spin densities, invariant gradient combinations, and density-scaled smooth projections. Avoid molecular identities, reaction labels, fitted per-element energy offsets, and explicit nuclear-charge embeddings in the learned functional. Nuclear positions still define the ordinary integration grid.
2. **Nonlocal interaction state.** Project features onto coarse integration cells with quadrature weights and a partition of unity. Start with 16–32 latent channels and 2–4 interaction layers. Combine short-range edges with a low-rank or hierarchical far-field representation. The state must transmit charge/spin information beyond a local bond; it is a latent state, not an asserted physical charge or measured response.
3. **One scalar XC energy.** Integrate a local correction plus the nonlocal contribution. Differentiate this scalar to obtain the potential and kernel actions. Every density dependence, including projections and interactions, participates in the derivative. Do not train an unrelated potential head for deployment.

Keep the first candidate around **0.1–1 million parameters**. Avoid attention over every pair of integration points, explicit virtual-orbital sums, correlated calculations at inference, or a separate unconstrained geometry-to-energy predictor. Complexity and memory are measured gates, not properties inferred from parameter count.

At inference the solver sees the molecule, charge/spin, basis, and numerical settings. Reference densities, inverted potentials, and perturbation responses are training-only information. Demonstrate a fresh converged SCF calculation from a standard initial guess.

### Constraints that determine whether this is viable

- **Fragment composition:** disconnected fragments must recover the sum of their independent energies. Avoid global normalization and total-electron-number switches that couple remote fragments. Test charge transfer and dispersion separately; a hard local cutoff cannot represent both.
- **One electron:** test cancellation of Hartree self-interaction, including separated density lobes. A local small-network fit may fail this representational test; do not declare the condition enforced merely because it appears in a loss.
- **Fractional charge/spin:** train against physically valid ensembles of adjacent stable integer ground states and degenerate spin states. Preserve the possibility of a derivative discontinuity at integer electron number. Do not flatten arbitrary nondegenerate spin states or force a smooth kernel across a cusp.
- **Symmetry and scaling:** test translations, rotations, spin interchange, exchange coordinate scaling, and quadrature refinement. Use exact constructions where demonstrated; list remaining conditions as empirical tests.
- **Long-range correlation:** test separated neutral fragments. If a dispersion correction is used, fix its convention and include its cost. Prevent double counting with learned correlation and confirm entry eligibility.

Do not require the XC kernel itself to be positive. Stability constraints concern the complete physical response on allowed density variations; exchange and correlation can have negative curvature.

## 5. Make each reference calculation teach more

Use a loss with per-system, noise-aware normalization:

\[
\mathcal L = \lambda_E\mathcal L_{\Delta E}
+\lambda_n\mathcal L_n
+\lambda_v\mathcal L_{v_{xc}}
+\lambda_\chi\mathcal L_{\chi u}
+\lambda_C\mathcal L_{\mathrm{constraints}}.
\]

`ΔE` means stoichiometric energy differences, including barriers and binding. Potential targets are optional where inversion is trustworthy. `χu` means the density response to an external-potential direction, obtained from paired perturbed calculations. All predicted fields/response must come from the same energy functional and SCF equations. Grid points are correlated observations, not thousands of independent molecules.

Generate two kinds of perturbation: a dipole-like field and a smooth inter-region charge-transfer field. Remove constant-potential directions at fixed electron number. Use both signs and two small magnitudes; compare finite-difference responses to establish a linear regime and numerical error. Spin perturbations follow once the charge-response pipeline passes. Use Hessian-vector products or SCF response solves, never a dense grid-by-grid XC Hessian.

### Data ladder

| Layer | Source and initial scale | Purpose and limitations |
|---|---|---|
| Broad energies | A stratified **1,000-molecule pilot** from the public [MSR-ACC/TAE25 dataset card](https://huggingface.co/datasets/microsoft/msr-acc-tae25); pin its revision, license and actual rows | Cheap access to expensive energy labels. This is mainly neutral equilibrium chemistry, not a complete reaction, ion, or density training set. Generate required DFT descriptors locally and cache selectively. |
| Controlled electronic response | **24 tiny-system anchor geometries**, grouped by molecular family into 12 train / 6 validation / 6 locked test anchors | Each anchor needs one unperturbed plus eight perturbed calculations: two directions × two magnitudes × two signs. **216 correlated calculations**, with basis/method selected by a timing pilot. This is a feasibility study, not a statistically sufficient transfer benchmark. |
| Known difficult limits | H/H₂⁺/H₂, He/HeH⁺, LiH, separated fragments, small radical/ion examples | Use exact diagonalization/FCI where feasible; appropriate correlated references otherwise. Validate reference quality on stretched or multireference systems. |
| Reaction and precursor transfer | Independent reaction families; then Al/Si chemistry with O/N/C/H ligands, association/dissociation, proton transfer and ligand exchange | Candidate public-domain proxy chemistry, not a claim about the hidden composition. Build consistent reactant/TS/product references and hold out entire ligand and precursor families. Extend elements once scope and resources support it. |

Use the pinned public dataset files and verified usable rows rather than copying a headline count. Pretraining energy data do not automatically supply correlated densities or XC potentials; generate only the missing labels that our experiments need. Do not plan around unavailable private collections.

For tiny systems, obtain compatible KS references by regularized finite-basis inversion. Check electron count, forward recovery of density, basis/grid convergence, gauge alignment, and potential sensitivity to regularization. Weight unreliable tails down. Reject ill-conditioned potential labels; retain reliable energies and density responses. CCSD(T) energies, CCSD densities, and exact KS potentials are not interchangeable labels with identical accuracy.

Active selection ranks *training-pool* candidates by response defects, fractional-charge curvature, failed convergence, and ensemble disagreement per measured reference cost. Choose diverse families, not only largest uncertainty. Pseudo-labels from cheap methods remain labeled as such and never become evidence of correlated accuracy.

## 6. Experiments that can reject the idea

Use matched initializations, data, numerical tolerances, parameter counts, and optimization effort. Compare:

| Arm | Architecture | Supervision |
|---|---|---|
| A | Local correction | Relative energies |
| B | Multiscale nonlocal correction | Relative energies |
| C | Same as B | Energies + ground-state densities / reliable potentials |
| D | Same as B | C + controlled charge/spin response |
| E | Same as B | D + fractional-charge/spin constraints |

These separate representation from ground-state fields, response, and ensemble constraints in our own model. Remove long-range communication from a promising arm only when that resolves a live scientific question. Do not add frontier-method comparisons or reproduction runs to this work program. Published context and our measured results must remain clearly distinguished.

Charge for the additional information: give energy-only controls the energies from all the same perturbed calculations. Add a second cost-matched control that can spend the response-label budget on more independent molecular geometries instead. Count reference wall time and inversion effort, not just anchor count; eight extra calculations per anchor cannot be presented as free data efficiency.

Split by molecular/reaction family before generating perturbations. All geometries, charge/spin states, forward/reverse reactions, and related perturbations stay in their assigned family. Check atom-mapped reaction and molecular overlap with public benchmarks. Report inherited overlap for pretrained models rather than assuming it absent.

Prespecify two inexpensive 3D mechanism panels: **unseen center count** (one-/two-center few-electron H/He training systems, with three-/four-center systems such as H₃⁺ and H₄²⁺ held out) and **unseen electron/spin sectors** (include one-, two-, and three-electron reference sectors in the overall design, and exclude a sector from training in each transfer run). Use tractable finite-basis FCI for these diagnostics and identify every charge and spin explicitly. A small-basis success tests the mechanism; G3 must independently establish transfer across real reaction and precursor families in larger bases.

Use one bounded run to reject a clearly unsuccessful idea. Repeat seeds or build learning curves only after a substantive transferable effect justifies them; do not use repeated fits to rescue a failed physical representation. Bootstrap promising results by family, not grid point. Reserve a second untouched family panel for confirmation so repeated research decisions do not consume the final test.

| Gate | Work | Continue only if |
|---|---|---|
| G0: working implementation | Minimal Linux chemistry environment, our GPU model, relative-energy scorer, convergence and timing logs | Our new energy/potential integration runs correctly enough to make the first informative experiment; do not repeat checks already established |
| G1: representation | Tiny 3D systems, asymptotic/ensemble limits, oracle-density diagnostics where KS inversion is reliable | Large systematic defects are representable at the allowed cost. If oracle-density errors remain large, more training labels are not the solution. |
| G2: mechanism | Run A–E, plus data-size curves on excluded families | D/E reduces relative-energy error by **at least 2× versus the best A–C**, on two prespecified OOD panels with consistent seed behavior and comparable SCF success/runtime; uncertainty must support the effect |
| G3: useful chemistry | Scale references selectively and introduce precursor/ligand-family holdouts | Large gains survive genuine chemistry shifts and do not collapse into a few easy systems or well-fit atomization offsets |
| G4: broad evaluation | Freeze the candidate and numerical settings; run full GMTKN55 plus independent proxy panel | Meet or approach the final accuracy/speed/reliability targets with complete failure accounting; otherwise explicitly record which target failed |
| G5: submission | Run our frozen candidate on one CPU core and an actual A100; package code, checkpoint, manifests, provenance, and instructions | Independent fresh-environment evaluation agrees; every fallback and correction is disclosed and timed |

G2 is a mechanism gate, not evidence that a 24-anchor pilot generalizes broadly. If response gives only a small gain at matched cost, stop scaling that idea and try a substantively different representation. Alternatives include our own ensemble/state model or occupied-density-matrix/GKS formulation; no frontier replication is required or allowed. Establish operator consistency and organizer eligibility for any changed formulation. Do not hide failure with a larger model or a benchmark-specific correction table.

## 7. Self-consistency is part of the model

Use a residual-based SCF solver with a deterministic retry schedule and logged thresholds. Start with well-tested damping/DIIS; record orbital-gradient and density residuals, energy change, iteration count, initial guess, and solution stability. Multiple spin solutions and nearly degenerate states need explicit checks.

At a genuinely converged variational solution, energy parameter derivatives can use stationarity. Density/response losses generally require differentiation through the fixed point, using an implicit adjoint or a validated short unroll. Compare against finite differences on small cases. Near a singular response or level crossing, detect ill-conditioning and use an explicit policy; do not silently return a biased gradient.

Compare functional-driven and density-driven error where reliable reference densities allow it. Treat errors from unconverged SCF as failures, not samples to remove. If deployment falls back to a baseline, its predictions count in accuracy and its complete retry cost counts in runtime.

## 8. Hardware, precision, storage, and compute allocation

**Binding initial limits:** 40,000,000,000 bytes (**40 GB decimal, about 37.25 GiB**) across all project storage; **one logical CPU core total**; full use of GPU 0. Keep planned working storage at or below **32 GB**, leaving **8 GB** for temporary growth. Include environments, Linux/VHD files, caches, downloads and extraction copies, reference scratch, and checkpoints. Keep them under the repository's ignored `.research/` directory where possible; record any external project paths in the storage accounting. Check both project usage and actual free space before any large allocation, including installers. A job's worst-case temporary footprint must fit before it starts. The expected 2 TB drive on **17 November 2026** does not increase this initial cap before it is available and the allocation is revised.

**GPU-only execution:** numerical tensor preprocessing, training, scientific operators and evaluation run on CUDA. No CPU SCF, reference/integral generation, CPU linear algebra workloads, or numerical fallback. Legacy numerical entry points now stop explicitly. Host metadata, basis bookkeeping, decompression and file I/O remain necessary; pin this work to one core and keep library/PyTorch threads at one and data-loader workers zero. Use `tools/linux_gpu.sh` through the resource guard. Earlier CPU-hour budgets below describe the initial allocation and do not authorize new CPU numerical work.

The [measured report](research/hardware_2026-11-16.json), produced by [the hardware probe](tools/probe_hardware.py), records:

| Resource | Observed locally |
|---|---|
| GPU | **RTX 4060 Laptop GPU**, 8,188 MiB reported VRAM, compute capability 8.9, 24 multiprocessors |
| CPU installed / allocated | Intel Core Ultra 9 185H, 16 cores / 22 logical processors installed; **one logical core allocated to this project** |
| RAM | About 31.6 GiB / 34 GB decimal |
| Free workspace disk | About **25 GiB** at the probe time |
| Working Python | 3.12.10; PyTorch 2.7.1+cu118; CUDA available |
| GPU readiness | FP32 and FP64 synthetic energy-derivative training both completed with finite gradients |
| Chemistry stack at initial probe | WSL/PySCF were initially absent; the dedicated single-core Linux chemistry environment has since been installed. No frontier functional has been installed or run. |

The synthetic timings are readiness checks, not forecasts of full chemical calculations or sustained throughput. The CUDA version in `nvidia-smi` describes driver capability; PyTorch currently bundles CUDA 11.8. Choose compatible libraries from their current installation instructions rather than equating those versions.

[PySCF requires WSL on Windows](https://pyscf.org/user/install.html). A dedicated `XCResearch` distribution and chemistry environment are now installed under the project's storage accounting, with a 20 GB virtual-disk capacity limit. PyTorch CUDA works inside it. An optional GPU4PySCF import hit a shared-library path issue; defer that integration and use our working PyTorch GPU path with inexpensive CPU reference calculations. Do not turn optional package integration into a prerequisite for our research.

Start with one small molecule per training microbatch, tiled grid evaluation, recomputation/checkpointing, and roughly a 6 GiB working GPU budget. The full GPU is available; the working memory target leaves room for CUDA and peak allocations. Keep persistent pilot artifacts under **8 GiB** within the total **40 GB** disk cap, reserving space for Linux, package caches, and reference scratch. Store geometry, basis, density matrices or compact projections, and checksums; regenerate dense descriptors as needed. Run only one CPU-heavy project job at a time, with one-core affinity/quota and single-thread libraries.

Validate FP64 first. Introduce FP32 neural operations with FP64 energy accumulation only after energy **differences**, potentials, and SCF convergence agree with FP64. Test FP16/BF16 separately; do not assume they preserve sub-kcal reaction energies. Consumer GPU double-precision throughput is a practical constraint; small successful smoke tests do not remove it.

Initial experiment ceilings, intended to force decisions:

| Allocation | Initial ceiling | Deliverable |
|---|---|---|
| Additional toy-only work | 4 GPU-hours equivalent | Derivative/constraint diagnostics; no indefinite 1D tuning |
| G0 + small timing/reference pilot | 8 GPU-hours and 24 CPU-core-hours | Our implementation and first reference-cost measurements; no frontier replication |
| G1/G2 mechanism study | 24 GPU-hours and 100 CPU-core-hours | Matched ablations and an explicit continue/reject decision |
| Scaling | Set after measured cost and G2 evidence | Reference count, storage estimate, and expected learning-curve benefit before scheduling |

These are ceilings, not promises that 216 correlated calculations fit the budget. Time a few representatives first; shrink the pilot or defer infeasible references. The laptop can test compact scientific ideas and small molecules. Complete QZ-basis sweeps and large correlated precursor references may exceed it. Reusing public labels is central to the plan; external compute or data collaboration is a conditional scaling requirement, not an assumed available resource. Actual A100 validation is required later and has not been authorized as a paid rental.

## 9. Evaluation that survives scrutiny

Keep the model-development validation set separate from the final public reporting set. Score stoichiometric reaction differences, not molecular total-energy MAE. The [official GMTKN55 site](https://www.chemie.uni-bonn.de/grimme/de/software/gmtkn/gmtkn55) supplies geometries, references, and a correction to published normalization information; pin the exact version and weighting convention. Use

\[
\mathrm{WTMAD2}=\frac{1}{\sum_sN_s}\sum_sN_s\frac{C}{\overline{|\Delta E|}_s}\mathrm{MAD}_s,
\]

with `C` and subset means fixed by that declared convention. Report partial suites as partial-suite results. Do not compare the organizer's hidden-suite numbers directly with GMTKN55 literature numbers. Record dispersion, anion basis treatment, integration grids, and all reference/basis differences.

Hold numerical settings fixed across methods or demonstrate matched accuracy of the numerical approximation. GPU timing requires synchronization. Measure initialization, integrals, features, model/derivative evaluation, SCF, retries, and transfers; distinguish cold-start and warm-start timing. Report per-molecule timings and the distribution across size and chemistry, not only an average accelerated kernel. Compare total cost on identical hardware and do not infer A100 speed from RTX measurements.

The submission evidence must include code revision, environment lock, model hash, dataset manifests and exclusions, seed results, per-reaction predictions, per-molecule convergence/timing records, and a complete failure list. Test energies and derivatives across grids and bases so the model cannot win through discretization artifacts. A final win claim depends on the organizer's independent evaluation.

## 10. Next execution order

1. All current jobs are terminal. Further speed experiments are disabled. Do not repeat the closed global-rank, neutral-singlet-only or previously successful tiny-case fits.
2. Close an unsuccessful construction after one informative run. Do not rescue it with label-driven seed, weight, rank or iteration sweeps. Failed density constraints invalidate energy scores. All new numerical work remains CUDA-only, with no frontier replication.
3. Replace the rejected composition structure with local particle-number/spin multiplets and physical fermionic links. It must contain independent correlated products without a fixed global determinant budget and include charge/spin sectors needed for strong coupling. Retain useful cofactor/block primitives; do not rescue the failed state spaces with more seeds, global rank or iterations. Preserve the scalar XC/derivative and density-only-frame requirements when this state is integrated into the functional. Wavefunction errors alone do not establish an XC model.
4. Freeze a general-electron candidate before unblinding `.research/data/locked_composition_confirmation.json`: 16 new compositions, two structures each, excluded from all inspected formula families. Reference-energy columns were not read during selection. Same-composition contrasts remove atomic offsets. Missing references/failures remain in the record rather than triggering adaptive case replacement.
5. Establish new chemistry, spin/charge and basis transfer before speed work. The locked neutral panel does not cover barriers or radicals; those require separate evidence. Full GMTKN55 and the hidden challenge target remain unachieved. CPU timing is deferred under the GPU-only instruction.

Established infrastructure: single-core Linux/CUDA environment, cached reference/development panels, general number/spin state construction, explicit product embeddings, CUDA operators, spatial-moment/auxiliary constrained searches and kinetic duals. Coherent/block kernels are useful primitives; the tested restricted models failed the accuracy gate. Old scalar-energy SCF integration does not authorize legacy CPU numerical entry points. Complete project storage is about 20.0 GB of the 40 GB allocation, including external research crash dumps and a conservative swap reserve. WSL now has a 6 GiB RAM cap and 1 GiB project-local swap. The working ceiling remains 32 GB; no new-drive capacity is assumed. See the execution log for measured failures and current artifacts.
