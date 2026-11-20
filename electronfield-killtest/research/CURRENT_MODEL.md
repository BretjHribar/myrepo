# Current model and supported claims

The user selected **the full public GMTKN55 suite first**, with working
tests and an initial measured assessment before further sub-1 research.
The [complete public parser and CUDA scorer](GMTKN55_PIPELINE.md) now pass
for all 55 sets / 1,505 reactions. This is infrastructure evidence;
**there is no full-suite model score or credible numerical estimate yet.**

The first complete [model-domain inventory](gmtkn55_model_domain_audit_20261117.json)
exposes a substantial feasibility gap at def2-QZVP: 2,421 of 2,442 required
structures have a static resource/domain blocker under the unchanged local
limits. Dense operator input arrays alone exceed the 6.5 GB allocation bound
for 2,359 structures. The existing Python word-enumeration estimate exceeds
its host-memory budget for 2,421; 60 structures need unvalidated triplet or
quartet support, and 81 need an unvalidated GPU ECP operator. These counts
overlap. The remaining 21 structures are uncomputed, not accepted predictions.
Only four reactions lack an identified state preflight blocker. Passing
these four would not support a full-suite estimate. The current complete
XC callable is also unfinished. The full coverage report returns no score,
with zero measured reactions; the current implementation cannot pass all
public tests. No convenient subset replaces them.

**First actual public component measurement:** all 11 required N=1
structures now have accepted full def2-QZVP energies from the established
exact one-electron limit. That permits 5/1,505 public reactions, with errors
from -0.013515 to -1.155744 kcal/mol against pinned v1 references. This
small selected exact domain cannot estimate the full suite. The latest
status is 11 accepted structures, 2,417 unsupported and 14 uncomputed;
full WTMAD-2 and a full-suite point estimate remain unavailable. See
[the component score](gmtkn55_N1_component_score_gpu_20261117.json).

A new [density-square operator](DENSITY_SQUARE_OPERATOR.md) avoids the
quartic word list and passes independent literal matrix/local-action checks.
Its supplied 40-electron product check is not a coupled prediction. The
large-basis factor producer, molecular accuracy and general XC callable
remain unresolved; the original full-suite gap is not declared repaired.

The original BH run remains interrupted at its saved cursor after the
user pause; it has not been resumed during this full-suite work. Its model,
capacity, thresholds and reference ordering remain unchanged. It is
finite-basis development calibration and cannot resolve the full-suite
implementation gap. Speed work remains disabled.

**Latest physical restriction found:** the first all-site water search
stopped after four valid updates at its unchanged overlap-precision guard.
Its fixed seed sectors also exclude all triple and higher virtual
excitations, so that family is closed. A direct-factor gauge repair
preserves energy/density and passes the original guard; it does not repair
the missing physics by itself. The changed
[two-site sector completion](TENSOR_SECTOR_COMPLETION.md) admits missing
N,S sectors while exactly retaining the old state, with independent
operator and variational-update checks. No chemical-accuracy claim follows.
The molecular sector test passed, lowering energy by 0.00226011 Hartree
and producing 1.10e-5 probability of three and higher virtual excitations.
Number/spin/occupation and literal energy checks pass. The state remains
unconverged and has no correlated reference score.

**New accuracy test:** the changing-sector two-site implementation passes
an independent six-orbital singlet calculation and a common-state audit.
A frozen new BH stretch pair in all 19 cc-pVDZ orbitals is checkpointed and interrupted.
Both predictions precede independent six-particle reference generation;
the reference start was corrected before labels to cover additional
spatial symmetries. No molecular accuracy result is accepted while this
report is incomplete. See [the sector record](TENSOR_SECTOR_COMPLETION.md).

An [open-shell doublet domain](DOUBLET_TENSOR_DOMAIN.md) passed its
independent three-electron CUDA implementation check, with 2.22e-15 Hartree
physical-reference energy error and exact reported physical number/spin.
Its inert auxiliary spin is excluded from physical electron/density claims.
This is not molecular open-shell accuracy. The fixed generic Ritz-start
safeguard also passed its independent invariant-block counterexamples;
the original BH protocol is unchanged.

**Latest functional result:** the
[conditional tensor functional](CONDITIONAL_TENSOR_FUNCTIONAL.md) now passes
an off-shell search on 10-electron water in all 24 cc-pVDZ orbitals, without
a global Fock vector. Its positive local search has a 5.52e-12 Hartree gap;
physical energy/density installation checks pass, and its frozen-context
XC scalar/derivative difference is 7.37e-9 Hartree. Exact checkpoint recovery
requires no repeated optimization. This connects the tensor state to a
conditional universal T+W evaluator. **Surrounding tensors are fixed; global
self-consistency, context independence and chemical accuracy are not proved.**
No reference-energy error or WTMAD-2 was measured. Speed remains disabled.

**Latest scope work:** a [compositional tensor state/action and density interface](TENSOR_DENSITY_RESPONSE.md)
now operates without global Fock vectors. Its physical checks include mixed
singlet/doublet states, an exact 32-electron independent-fragment product,
and coupled 10-electron water in all 24 cc-pVDZ orbitals. The changed
physical response metric retains 98.61% of the initial squared response
norm at the same bond cap, versus 2.71% for the now-closed raw-channel route.
Number/spin, density occupations and literal energy identities pass. The
water state remains unconverged and has no reference accuracy score.
The tensor optimizer is now connected at a conditional local section;
the full global search and broad-accuracy limitations below remain.

**Current decision, 17 November 2026:** the accuracy goal is not achieved;
speed work remains disabled. The [auxiliary-density construction](AUXILIARY_DENSITY_STATE.md)
now has two accepted BeH2/6-31G forward development points, including its
scalar XC functional, derivative, physical state and regularized KS checks.

| Geometry | Same-basis error, kcal/mol | Full stationarity residual, Hartree | Potential self-consistency residual |
|---|---:|---:|---:|
| 1.65 Angstrom | +1.96e-8 | 9.28e-6 | 7.01e-8 |
| 2.75 Angstrom | +4.80e-8 | 6.75e-6 | 1.10e-8 |

Both pass the original physical, number/spin/occupation, kinetic and
lowest-state numerical checks. New frozen-response-space XC directional
derivatives differ by 1.58e-8/1.41e-7 Hartree, below the prespecified 1e-5
criterion. Sources: [forward completion](auxiliary_stationary_gpu_20261117.json)
and [derivative checks](auxiliary_derivative_gpu_20261117.json). These are
post-interruption development completions, not blind chemistry tests.
References were assessed after stopping; no oracle correlated density or
new reference was used for the predictions.

The auxiliary variable differs from physical density. Its explicit energy
counterterm recovers the physical Hamiltonian; this is not an unchanged
conventional KS plugin. Challenge integration/eligibility is unverified.
The accepted auxiliary evaluator still stores global Fock vectors, supports only four/six
particles or holes and even-N singlets, and provides no learned transfer,
complete-basis, broad reaction or WTMAD-2 evidence. Established Moreau,
Krylov, positivity and tensor mathematics are not our claimed novelty.
No breakthrough is established.

The cold hard-density inverse and both bounded feasible-ensemble searches
are closed after convergence failures. An early auxiliary completion had a
verified spin/energy implementation error and remains invalid. The accepted
run uses literal physical Hamiltonian actions, explicitly purified basis
extensions and independent spin guards. All failed records remain preserved.
See the [latest execution record](PROGRESS.md) for numerical repairs and scope.

The next accuracy work must retain all physically needed charge/spin
sectors, establish common stationarity across tensor environments in the
universal T+W auxiliary optimizer, control correlated response and retain
scalar/derivative and physical acceptance contracts.
A generic tensor solver alone does not establish a transferable XC model.
The locked confirmation panel remains withheld. The
[accuracy gate](ACCURACY_GATE.md) is binding.

Current complete disk accounting is about **21.2 GB of 40 GB**, including
external research crash dumps and a conservative old-swap reserve. WSL has
a 6 GiB RAM cap and 1 GiB project-local swap; the working ceiling is still
32 GB. Numerical work stays CUDA-only with one host core. No extra drive
capacity is assumed. See [resource accounting](resource_accounting_20261117.json).

## Earlier status, retained as history

The entries below precede the accepted auxiliary formulation above. Their
failures and limits remain valid; statements about the latest interface or
next experiment have been superseded by the current decision.

Updated 17 November 2026. The objective remains the Discovered
Materials speed/accuracy challenge. **The challenge is not achieved.**
Accuracy, physical validity and transfer come before speed. No frontier
method replication is part of this work.

**Latest accuracy decision:** further speed work is on hold. The new
[charge/spin block state](CHARGE_SPIN_STATE.md) repairs restrictions exposed
by the preceding rejection tests. Its nine-case development panel now meets
the original physical residual criterion and agrees with same-basis
references to order 1e-8 kcal/mol. A prespecified new 18-electron peroxide
also agrees to 1.57e-8 kcal/mol, but retains a 1.59e-5 Hartree residual at its
update cap, slightly above the declared 1e-5 target. Its saved subspaces were
then continued for two residual updates, reaching 8.35e-6 Hartree; that later
result is development completion, not the original fixed-budget holdout.
These small-basis state results do not establish the broad accuracy goal or a finished XC functional.
The [accuracy gate](ACCURACY_GATE.md) remains binding.

**Latest rejection:** the new bounded density-only interface failed. With h
and reference labels excluded, the cold interacting dual search still had a
3.52 density residual after its third completed step; a later trial potential
failed the 1e-8 Hartree state criterion at its declared capacity (2.52e-5).
The stricter implementation preserved singlet spin to about 7e-16. It
returned no XC scalar or potential. This fails the functional implementation
gate; it does not refute constrained search itself or prove that the input
density is unrepresentable. It is not permission to optimize runtime or
promote the state-only tests.
Do not rescue it by iteration/rank sweeps, reduced tolerances or supplying
the known molecular potential. See [the functional record](SPATIAL_DENSITY_FUNCTIONAL.md).

The changed representation retains local number/spin multiplets and their
complete magnetic partners, the input determinant's full Schmidt support,
independent tensor products and the physical fermionic Hamiltonian. New
directions come from the Hamiltonian residual, never reference-energy error.
The original four-update H4 failure of 4.146 kcal/mol remains in its report;
an explicitly unconverged physical solve was continued to the original
residual criterion without repeating previous optimization cycles or labels.

Saved H4, peroxide and ionic states now have verified charge/spin tensor
exports. Their state errors are below 3e-12 and energy changes below 2e-14
Hartree. This is a representation identity, not a new prediction or strong
compression result: the current solver still stores a global Fock vector,
and some selected spaces are nearly full. Two new six-electron BeH2 cases in
6-31G reached their declared residual target with errors below 7e-9 kcal/mol
and a -2.82e-10 kcal/mol stretch-contrast error. Selected dimensions were
11019/10865 of 81796. The unpolarized split-valence basis is still far from
complete-basis chemistry. The locked panel remains withheld; no WTMAD-2
result is established.

The new [spatial density functional construction](SPATIAL_DENSITY_FUNCTIONAL.md)
uses the represented density-product span, without nuclear-potential probes.
A squared Coulomb-Gram variant failed and is recorded. Direct spatial-product
resolution passes independent point and orbital-coordinate checks. On-shell
kinetic/XC completion gives total-energy changes of about 0.00035/0.00144
kcal/mol, within the stated entropy allowance. New density paths give
scalar/potential derivative differences of 1.36e-7/5.74e-8 Hartree. These are
implicit finite-basis certificates, not a general arbitrary-density callable
or a trained transferable predictor. The molecular-h-selected state basis
must not be treated as universal without a new constrained-search procedure.

The following rejected predecessors and older functional findings are
retained to explain the physical requirements and unresolved scope.

The new coherent local state reached 0.049/0.088/0.124 kcal/mol errors on
four-electron LiH/Be in complete cc-pVDZ bases. But new 6/8/10-electron
minimal-basis tests gave **0.472/2.475/0.987/1.287/1.101 kcal/mol**, including
a **2.002 kcal/mol BeH2 stretch-contrast error**. Its fixed global determinant
budget also fails exact fragment closure: a 13-rank local spin partition
requires rank 169 for two independent copies. Do not rescue this global
family by seed, determinant-count or iteration sweeps.

A tensor product of local states fixes that composition problem in principle.
The tested **neutral-singlet-only** quantum block projection, however, fails
strong coupling: He2 error 0.043, distant H4 0.960, close square H4
**209.962 kcal/mol** in cc-pVDZ. Its balanced H4 contrast misses by 209.001.
Independent full-coordinate checks agree with the projected operator to
9e-16 Hartree, so this is a genuine restriction of the tested state space.
Local charge and spin sectors cannot be omitted. This hypothesis is closed
before any further speed tuning.

Those panels were inexpensive: about 59 seconds for all five cross-electron-
count cases and 21 seconds for the three coupled-block cases, including
startup and new needed references. Accurate reference work in realistic
bases needs its own measured budget. The new state above addresses the
omitted particle-number/spin sectors; its scalar/derivative integration,
larger-system solver and realistic-basis transfer remain required.

The older functional research construction is the [Coulomb-moment XC functional with
kinetic completion](COULOMB_MOMENT_FUNCTIONAL.md). Its callable implementation
is `xc3d/coulomb_moment_model.py`, function `evaluate_xc(data, density)`.
The interface currently supports CUDA, even electron counts and spin-unpolarized
ensemble densities in a stated finite basis. It rejects invalid input densities,
failed moment constraints and unconverged potential derivatives.

## What is actually supported

| Requirement | Evidence and limit |
|---|---|
| Fermionic number/spin state | Quantum-number/CG tensors and fixed occupation sectors; new 4/6/7/8/9/10/18-electron experiments distinguish successes from failures |
| Fractional density without fluctuating particle number | Explicit canonical lift to weighted singlet determinants; water/H2S lifts reproduce their moments at about 1e-10/1e-9 |
| Fragment composition | Exact product embedding for supplied ordered singlet fragments, N=8; additivity error 3.6e-15 Hartree. Automatic global-truncation size consistency remains unresolved |
| Universal constrained objective | T+W, spatial Coulomb moments and the input density; external charge weights are excluded. The frame is geometry-dependent, so a single universal density-only functional has not been established |
| Complete external-energy moments | Point-nuclear Coulomb potential reconstructed in the moment span to 6e-14 Hartree on water, 6e-13 on H2S |
| Scalar XC and derivative | E_xc=F_M-Ts_M-E_H; a frozen-frame water density rotation agreed with v_xc to **3.95e-9 Hartree** in the directional derivative |
| Self-consistent moment density | Direct dual fixed point, followed by noninteracting ensemble inversion; not a conventional KS runtime benchmark |
| General chemical accuracy | **Not established:** complete minimal bases only, few new cases, dense implicit physical solves |
| Learned generalization or a breakthrough | **Not established:** no amortized trained predictor, no practical larger-basis evidence |

The direct dual fixed point gave these errors against newly generated or
already cached exact references in the **same complete STO-3G basis**:

| Case | Electrons | Energy error, kcal/mol | Moment residual |
|---|---:|---:|---:|
| Water, development | 10 | +0.001650 | 6.05e-11 |
| H2S, new composition/element/electron-count application | 18 | +0.000991 | 1.32e-9 |

The H2S protocol was fixed before generating its reference energy. No
reference energies supplied the objective or stopping rule. These numbers are
for the zero-temperature interacting-state direct fixed point with a bounded
entropy-regularized kinetic dual. The callable scalar uses an additional
stated entropy allowance for its interacting ensemble search; do not confuse
the two numerical variants or infer complete-basis accuracy from either.

The explicit canonical lift initially had an incorrect sampling breakpoint
sign. It was repaired using the saved predictions, without repeating energy
solves or reference generation. Both original failed certificate values and
corrected values are retained in `dual_fixedpoint_transfer_gpu_20261116.json`.

Other failed constructions remain closed: the local H3 flow family, fixed
two-channel BeH2 state, penalty-only density search, and pure/gapped water
searches. The ensemble water **one-shot HF-density** energy also fails by
305.65 kcal/mol despite valid constraints; changing to the model's own density
fixed point is essential. Do not hide this density-driven failure.

## What remains before a useful generalization claim

The present occupation action grows exponentially and the stated basis is
too small for the challenge. Replacing it with a physically constrained
representation that supports realistic bases is an **accuracy/scope
requirement**, not permission to begin speed tuning. Retain the scalar,
canonical representability and derivative contracts during that change.

Next original research should construct density-response orbital/state
sections from T, Coulomb probes and W, with a Hamiltonian-residual completion
criterion independent of reference errors. Test that changed representation
on independent chemistry in a larger basis and balanced relative energies.
Do not keep rerunning these minimal-basis successes, increase fixed rank to
rescue BeH2, or present generic Krylov linear algebra as a novel method.

Full-density refinement, open-shell/fractional-spin functional definitions,
automatic fragment limits, basis/frame nuclear derivatives and reaction
transfer remain unresolved. The locked 32-structure composition panel stays
withheld until a candidate and prediction protocol are frozen. GMTKN55,
hidden precursor chemistry and the final timing target remain unachieved.

The geometry-dependent frame is a substantive scope limit. A fixed-Q
constrained functional is universal for that Q; changing Q with supplied
nuclear positions does not by itself define one density-only XC functional.
A next hypothesis is a density-selected, variationally stationary potential
frame, with its response/envelope derivative included. Its candidate family
must be specified independently of the test external potential. Do not claim
challenge eligibility from the current fixed-frame prototype.

All numerical work used CUDA. Host bookkeeping/I/O used at most one core.
The earlier 12.712 GB repository accounting omitted external research crash
dumps and a swap reserve. The current complete estimate is above. No drive
expansion is assumed. The brief authorized speed phase is recorded in the
execution log; further speed work is on hold.
