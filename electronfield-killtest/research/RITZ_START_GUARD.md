# Lowest-state safeguard: independent CUDA checks passed

17 November 2026. A small residual does not establish that an eigenstate
is the lowest state in the intended physical space. This is a numerical
acceptance issue, not a new physical theory or a chemical-accuracy result.

## Concrete failure mechanism

The current `incremental_ritz` begins with one supplied state and expands
its residual. For H=diag(1,0) and initial vector (1,0), that residual is
exactly zero. The implementation therefore returns energy 1 immediately,
although the complete space contains energy 0. More generally, a starting
state confined to an invariant spatial-symmetry block cannot reveal a
lower eigenstate in another block by repeated H applications alone.

This counterexample follows directly from the stopping branch in the
source. It does not establish that the ongoing BH state has selected a
wrong sector. Its existing global-minimum/full-residual claims already
remain false, and independent chemical reference comparisons are still
required. The frozen original BH calculation is not changed or repeated.

## Prepared correction

`physical_ritz_seed.py` adds one fixed generic CUDA vector at relative
amplitude 0.5 to the retained normalized input, then normalizes. The
strict amplitude bound prevents cancellation of the entire start vector.
The probe is not orthogonalized: in two dimensions, orthogonalizing would
leave only two discrete probe choices, permitting special cancellations
of an eigencomponent. A generic unrestricted component has overlap with
otherwise inaccessible sectors with probability one in exact arithmetic.

There is one prespecified probe seed per physical center, no trials,
reference-energy selection or seed campaign. Any vector in the exact CG
section retains the stated physical number/spin constraints. The old
physical state remains available, and literal variational/domain guards
still reject an unacceptable update. The new policy does not certify a
global minimum in finite precision or replace independent accuracy tests.

The two-site implementation exposes this policy only when a protocol
explicitly chooses it. Existing protocols retain their original start
policy and numerical path, including a resumed BH checkpoint. The newly
prepared doublet protocol selects the new policy before its first run.

## Independent implementation evidence

`ritz_start_guard_protocol_20261117.json` freezes independent diagonal
and decoupled-block counterexamples. The CUDA test must retain the old
false lowest-state acceptance, then compare the one fixed new start with
the complete matrix spectrum. It uses unchanged residual tolerances and
no reference-selected setting. Both checks passed in
[the CUDA report](ritz_start_guard_checks_gpu_20261117.json). The original
starts returned small residuals while missing lower states by 1.0 and
1.9763364463 Hartree. The fixed generic starts reduce independent energy
errors below 2.3e-16 Hartree and residuals below 2.8e-16 Hartree.

This test ran at the verified idle boundary after the user resumed the
goal. The subsequent doublet check also passed. Do not repeat these settled
checks or change the frozen original BH protocol. These finite examples
do not certify a global molecular minimum or broad chemical accuracy.
Krylov mathematics and generic starts are established numerical techniques,
not our claimed research breakthrough.
