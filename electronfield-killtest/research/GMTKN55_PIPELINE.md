# Complete public evaluation pipeline

17 November 2026. The user selected **the full public GMTKN55 suite first**,
with an initial measured assessment before further sub-1 research. The
data/parser/GPU scorer are working. **The model has no full-suite prediction
set or defensible full-suite kcal/mol estimate yet.** The first exact
one-electron component now supplies 11 real target-basis energies and five
scorable reactions. This is explicitly partial, not broad model accuracy
or a claim to pass private organizer tests.

The pinned authors' v1 data contains 55 evaluation sets and 1,505 reactions.
They require 2,442 unique structure IDs. The 2,462-structure upstream
inventory also contains 20 structures not referenced by these definitions;
the validation report lists them explicitly. No reaction was removed.
BH76RC shares BH76 coordinates but has its own 30 reactions and weights.

The [complete coordinate check](gmtkn55_coordinate_checks_gpu_20261117.json)
passed for all 2,462 structures, comparing both original formats on CUDA.
The largest indexed pair-distance difference is 1.59e-9 Bohr. The
[parser/scorer check](gmtkn55_runner_checks_gpu_20261117.json) passed all
literal reaction examples, all 1,505 independent contractions and the
weighted aggregation. Missing/nonfinite structures, failed convergence,
failed physical checks, wrong versions/units, duplicate IDs and benchmark
reference leakage are rejected or leave the suite explicitly incomplete.
Synthetic energies were used only to test this machinery; they are not
model predictions or an initial accuracy estimate.

## First full model assessment

The [complete domain inventory](gmtkn55_model_domain_audit_20261117.json)
checks the current own physical-state implementation against every required
structure in def2-QZVP. It finds 2,421 structures with static memory/domain
blockers and 21 uncomputed structures. These are integer shape/resource
estimates and implementation-domain checks, not completed molecular runs.
Of the overlapping blockers, 2,359 have dense CUDA operator input arrays
alone above the 6.5 GB limit; 2,421 exceed the existing host word-enumeration
estimate; 60 require triplet/quartet support and 81 require a GPU ECP
operator. The largest structure has 3,420 basis functions. Only four
reactions have no identified state preflight blocker. That is insufficient
coverage for a credible full-suite point estimate even if all four pass.

The [coverage scoring report](gmtkn55_current_candidate_coverage_score_20261117.json)
tests the end-to-end actual candidate status file. It returns 0/1,505 measured
reactions and no WTMAD-2, lower-bound estimate or full-suite point estimate.
There are no invented energy predictions. The existing complete XC callable
is unfinished as well. The model presently cannot pass the full public
suite. Making the physical representation executable across this scope is
required before an accuracy estimate can be supplied; more tiny-system
successes cannot establish that coverage.

## First actual public measurements

After that initial zero-prediction assessment, every required one-electron
structure was evaluated by the exact N=1 physical component in the full
def2-QZVP basis. All 11 predictions passed their physical and eigen-residual
checks, using CUDA integrals/eigensolves/densities/nuclear energies and no
CPU scientific fallback. The predictor did not read public reference values.
This uses an established exact limit; it is not a novel general model.

The [actual component score](gmtkn55_N1_component_score_gpu_20261117.json)
contains these five reaction errors against the pinned public v1 references:

| Reaction | Error, kcal/mol |
|---|---:|
| G21IP/001 | -1.155744 |
| SIE4x4/001 | -0.019127 |
| SIE4x4/002 | -0.013515 |
| SIE4x4/003 | -0.030466 |
| SIE4x4/004 | -0.177301 |

Only **5/1,505 reactions** are measured. There are 11 accepted structure
predictions, 2,417 unsupported entries and 14 uncomputed entries. The exact
N=1 identity bypasses the generic two-electron operator for four additional
structures; the older generic-operator audit is preserved. **Full WTMAD-2
and the full-suite point estimate remain unavailable.** The measured
0.0004462445 contribution to the full denominator is only a nonnegative
lower bound, not an accuracy estimate or evidence of passing. No reference
offset was fitted, and these cases cannot estimate other chemistry.

The new [density-square operator](DENSITY_SQUARE_OPERATOR.md) passed
independent implementation checks without quartic word enumeration. A
large-basis integral-factor producer and broad molecular accuracy remain
unimplemented/unmeasured. Its larger supplied-product identity is not a
larger coupled-system accuracy result. Speed work stays disabled.

## Metric and prediction contract

Use the corrected authors' normalization **57.8167 kcal/mol**, as declared
in their [pinned calculator](https://github.com/grimme-lab/GMTKN55/blob/8d485b37a1ca8837e395042671ca5ba4e0714691/.calc_WTMAD-2),
and each pinned `.average`. The older 56.84 normalization is not used.
BH76RC's supplemental `.averageRC` (21.39) and `.numenRC` (30) are pinned
and verified against their Git blob hashes. Reference-derived unrounded
means are also recorded transparently; they do not replace source weights.
Upstream shell scripts and evaluator programs are never executed.

The [scoring protocol](gmtkn55_scoring_protocol_20261117.json) fixes the
source version, normalization, energy convention and incomplete-score
policy. Predictions must give **electronic plus nuclear total energy in
Hartree**, a frozen candidate/source/protocol identity, basis, convergence
and physical acceptance. Public reference values are excluded from the
predictor inputs in `.research/data/gmtkn55_pipeline/prediction_inputs.json`.
The neighboring `prediction_contract_template.json` describes the format;
its synthetic placeholder identity must be replaced by an actual candidate.

Full WTMAD-2 is returned only when every reaction can be scored. Otherwise
the result reports all unavailable reactions and, if any real reaction was
measured, its nonnegative contribution to the full score as a lower bound.
That lower bound is not an estimate of unmeasured errors and cannot pass
the goal. An opportunistic small subset is not a probability sample.

Run an actual candidate's prediction file, from the repository root, with:

```powershell
python electronfield-killtest/tools/resource_guard.py --reserve-gb 0.25 --timeout 600 -- wsl.exe -d XCResearch -u root --exec bash /mnt/c/Users/breth/myrepo/electronfield-killtest/tools/linux_gpu.sh -m xc3d.gpu_score_gmtkn55 --predictions /absolute/path/predictions.json --output research/unique_candidate_score.json
```

Only one numerical GPU worker may run. All scientific arithmetic uses
CUDA float64; unavoidable host I/O uses one core. Complete project storage
after the runner test is about 21.25 GB of the unchanged 40 GB allocation.
Speed work remains disabled. The separate locked confirmation panel remains
withheld. The unfinished BH calibration retains its original checkpoint,
settings and reference ordering; it is not a public GMTKN55 result.
