# Charge/spin blocks for the accuracy-first state

Research construction, 17 November 2026. The active goal is **get below
1 kcal/mol; do not worry about speed**. This is a physical state
representation under development, not yet a finished universal XC functional
or a goal-level benchmark result.

The previous coherent global sum failed composition, and a tensor product of
neutral singlet blocks excluded decisive strong-coupling states. The changed
representation addresses both restrictions directly.

## Physical state space

Each local orthogonal orbital block has multiplets labelled by electron number
N, total spin S, magnetic component M and an internal state channel. All
magnetic partners are generated with the same spin-lowering operator and its
Clebsch normalization. The local projector therefore respects spin rotations.
For a global singlet, local multiplets are retained whenever their N/S labels
can couple to those of the other block to give the requested total N,S=0.

The global space is the **tensor product of local spaces**, with total
particle number and spin enforced. Its capacity grows by composing the local
states; there is no fixed global determinant count. Fermionic ordering is
kept in an explicit Fock map. The physical T+W+external Hamiltonian acts before
projection, so charge transfers, spin exchange and other interactions are
not replaced with independent density lowerings.

Initial states come from local Hamiltonians embedded in the other block's
input-density Hartree/exchange field. This embedding chooses basis vectors
only; the global energy uses the unmodified physical Hamiltonian. The full
local Schmidt support of the stated input determinant is added, and its
retention is checked explicitly. No atomic energy offsets or correlated
reference labels choose this basis.

## Completion without reference-energy tuning

After solving PHP in the current tensor-product space, evaluate the full
Hamiltonian residual r=(H-E)Psi. Project its local Schmidt sources into the
appropriate N/S sectors, remove the current basis and add the leading
independent response directions, keeping complete degeneracies and magnetic
multiplets. The stopping target is a physical residual norm, not agreement
with a reference label.

The first bounded panel stopped after four updates with a large residual in
close H4. It improved that failure from about 210 to 4.15 kcal/mol but had not
met the original convergence criterion. A continuation selected the largest
remaining physical residual, recovered the saved state's Schmidt support and
reached the original 1e-5 Hartree target. The cc-pVDZ error was then about
9.4e-9 kcal/mol. No old optimization cycles or references were repeated. The
continued local product space had dimension 17626 within a 36100-dimensional
fixed-Sz space; this is **not** a demonstration of strong compression.

The same rules removed the earlier BeH2/NH3 rejection errors and worked on a
prespecified new H2F+ ion. The other five unfinished panel states have now
reached their original residual target; completed states and labels were not
recomputed. The nine-case development panel now has same-basis errors of
order 1e-8 kcal/mol or less. This is numerical agreement in stated small
bases, not a chemical-accuracy or WTMAD-2 claim.

A new 18-electron H2O2 geometry, specified before its independent reference,
gave a 1.57e-8 kcal/mol same-STO-3G error. Its physical residual was 1.59e-5
Hartree at the 32-update cap, slightly above the declared 1e-5 target. The
retained fixed-Sz product space was 36238 of 48400 dimensions. Keep this
convergence limit and weak compression visible despite the small energy
difference. Six-hole roundtrip, direct first-quantized action and input Wick
checks passed before reference use.

The remaining peroxide residual was subsequently completed by restoring the
exact saved local subspaces and state, without a local eigensolve or label
regeneration. Two additional response updates reached 8.35e-6 Hartree and a
3.39e-9 kcal/mol energy error, with product dimension 37154/48400. This is
post-label development completion under the original physical criterion;
the prespecified 32-update result above remains separately recorded.

The saved close-H4, peroxide and ionic states were exported as reduced
charge/spin Schmidt tensors, with all magnetic partners reconstructed by
spin lowering and exact singlet coupling. Reconstructed state errors are
below 3e-12 and energy differences below 2e-14 Hartree. This conversion used
the saved states without new fitting or reference generation. Schmidt/SVD
and spin-coupling mathematics are established primitives, not novelty claims.
It is a two-block representation foundation, not yet a tensor-chain solver.

Two new BeH2 geometries in the larger 6-31G basis are specified in
`larger_charge_spin_protocol_20261117.json`. Both reached their physical
residual target with same-basis errors below 7e-9 kcal/mol. The balanced
stretch contrast error is -2.82e-10 kcal/mol. The selected product spaces
have dimensions 11019/10865 within an 81796-dimensional fixed-Sz space.
Labels were generated only after saving each prediction. Peak CUDA
allocation was 2.934 GB. This split-valence basis has no polarization and is
not a complete-basis or GMTKN55 result. No speed work was part of the test.

The [spatial density construction](SPATIAL_DENSITY_FUNCTIONAL.md) now supplies
finite-basis on-shell kinetic/XC certificates and directional-derivative
checks on these saved states. It resolves density-product directions without
selecting nuclear-potential probes. Arbitrary-density interacting search
and a transferable compact predictor remain unimplemented.

## What this does not establish

The present accuracy implementation still stores a fixed-Sz global Fock
coefficient vector. Four- and six-quasiparticle actions enable the current
full-basis and minimal-basis tests; this is not a general large-molecule
implementation. A global tensor coefficient representation is still required
for realistic basis/electron-count coverage. That is an accuracy/scope need,
not permission to begin speed tuning.

Blocks are supplied in a symmetrically orthogonalized AO partition. Automatic
partitioning, a general density-only functional implementation, basis/frame
derivatives, arbitrary-density constrained-search integration, open-shell
global targets and realistic-basis reaction transfer remain unresolved.
Spin/particle validity and same-basis accuracy do not by themselves prove a
finished XC functional.

The sub-1 goal retains the original broad relative-energy/WTMAD-2 scope. The
locked confirmation panel remains withheld, and these development successes
must not be relabelled as full GMTKN55 or hidden-suite performance.
