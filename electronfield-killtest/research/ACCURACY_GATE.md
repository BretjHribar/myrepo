# Accuracy gate before further speed work

The latest user instruction supersedes the brief authorization for parallel
speed work: establish an accurate model within the goal before optimizing it.
No further speed experiment is a current work item. Valid existing numerical
code can support the accuracy research.

The [complete public benchmark data](BENCHMARK_DATA.md) is now pinned to
the authors' original-suite `v1` commit, with all 55 evaluation sets and
the documented weighting correction kept explicit. Source integrity
passes; no model score exists. Do not replace the full gate with filtered
cases, a changed reference version or a stale normalization constant.

The target is below 1 kcal/mol WTMAD-2 with broad chemical transfer. A few
absolute same-basis errors below 1 do not establish this. The final challenge
also needs organizer/hidden-suite evidence and its timing rectangle; those
remain unachieved. Full GMTKN55 WTMAD-2 and independent reaction-family
evidence are required before claiming broad goal-level accuracy.

Before spending that evaluation budget, reject candidates cheaply on:

1. A stated scalar XC functional and consistent derivative/self-consistency,
   distinguished from an unconstrained molecular wavefunction experiment.
2. Exact particle number and fermionic structure, valid spin/ensemble states,
   reference retention and a defensible fragment-composition rule.
3. New electron counts and compositions, unseen geometries and balanced
   relative energies. Keep every failure and prespecify the cases/settings.
4. Larger-basis transfer and numerical convergence, with references/basis
   errors stated. Do not call minimal-basis agreement chemical accuracy.
5. A frozen candidate/protocol before the locked composition-confirmation
   contrasts are exposed. Do not adapt or replace failed confirmation cases.

The fixed global coherent state and neutral-singlet-only block model failed
the composition/transfer audits and are closed. Their charge/spin-multiplet
replacement closely matches small same-basis references after physical
residual completion. That does not establish this gate: some projected spaces
are nearly full, the global implementation still stores a Fock vector, and
the scalar density-only functional integration and realistic-basis reaction
transfer remain unresolved. The fixed-budget peroxide result stopped
slightly above its declared residual target; a later saved-state continuation
completed that residual and is recorded as development. Preserve these qualifications;
do not authorize speed work from small-panel energy agreement.

The bounded density-only inverse search on 17 November also failed. Its
stricter singlet implementation preserved spin but did not finish the
interacting state/density constraints, so no XC value/potential was accepted.
That cold pure-state dual protocol is closed; a new ensemble/representation
hypothesis needs its own protocol. No iteration, rank, seed or tolerance
sweep, or warm start from the withheld molecular potential, turns it into
evidence for a working general functional.

Run each new informative rejection experiment once. Do not repeat settled
small-system successes or launch frontier-method reproductions. Numerical
work and new reference generation stay on CUDA, with one host core and the
unchanged 40 GB total storage limit.

The finite-coordinate auxiliary formulation now passes two BeH2/6-31G
forward development points and fresh frozen-space scalar/derivative checks.
See [the formulation and limits](AUXILIARY_DENSITY_STATE.md). This does not
establish the gate: its global Fock backend, even-singlet/quasi-particle
restrictions, finite basis and inspected development chemistry remain.
The auxiliary coordinate differs from physical density and requires a
regularized KS implementation; conventional-plugin compatibility and
challenge eligibility are unverified. The cold inverse and both failed
hard-density ensemble protocols remain closed. No energy score from an
invalid spin or unconverged physical state counts as accuracy evidence.

The new compositional tensor backend passes literal operator, number/spin,
density, independent-product and larger-basis water scope checks. Its
physical response metric measures image loss without a global Fock vector
for product sources. The water state is unconverged; no reference accuracy
or XC-functional result was measured. This does not establish the gate or
authorize speed work. The raw operator-channel metric is closed after it
lost most of the physical response; no rank sweep should rescue it.

The [conditional tensor functional](CONDITIONAL_TENSOR_FUNCTIONAL.md) now
passes an off-shell 24-orbital water search, physical installation and a
new frozen-context scalar/derivative check. Its conditional gap is not
global tensor stationarity. Surrounding-tensor response, independent
chemistry and realistic-basis relative-energy accuracy remain unproved.
Do not infer the accuracy gate from its variational lowering or repeat the
settled single-center certificate. Speed work remains disabled.

The first all-site tensor family is also closed: besides a retained
numerical metric failure, its seed labels exclude three and higher virtual
excitations. A passed coordinate audit in that restricted family could not
establish a sufficiently general state. The
[changed sector-completion test](TENSOR_SECTOR_COMPLETION.md) removes this
restriction locally, with exact retention of the old state and complete
spin multiplets at the same bond cap. Neither a new sector nor an energy
lowering counts as sub-kcal accuracy. Require common stationarity and
independent realistic-basis balanced-energy errors; do not rescue closed
fixed-sector sweeps or expose the locked panel early.

The new [doublet embedding](DOUBLET_TENSOR_DOMAIN.md) is pending validation.
For any such open-shell extension, verify the physical subsystem after
tracing out its auxiliary: fixed physical N,S, positivity, normalization,
energy and independent sector/reference agreement. An enlarged-system
singlet or total electron count does not establish the physical doublet.
Do not count auxiliary electrons as larger-system evidence or promote
source compilation into a numerical/chemical accuracy pass.

The [lowest-state safeguard](RITZ_START_GUARD.md) records a separate
failure mechanism: zero residual from a retained state in one invariant
sector can hide a lower state elsewhere. The existing BH protocol remains
frozen and still requires independent references. Future new start-policy
claims need the prepared CUDA counterexample test; a generic probe is not
a rigorous global-minimum certificate or a seed campaign.
