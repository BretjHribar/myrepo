# A candidate worth testing out of sample

The objective is a transferable XC functional, with accuracy first. A small
wavefunction result is a mechanism experiment, not the requested end state.
Numerical work is CUDA-only; the 40 GB disk cap and no-frontier-replication rule
remain binding. Do not start speed optimization while the accuracy case fails.

## Define the model, rather than only its energy fit

The proposed route is a density-conditioned, particle-conserving correlated
state. Charge fluctuations, spin recoupling, orbital relaxation and angular
fluctuations must share the same state; their energy changes cannot be added as
independent pair lowerings. Local connected operators can support fragment
composition, but that property must hold for the initialization and numerical
approximation too. It is not enough for a single gate to have a local limit.

Every candidate must state and check:

- Particle number and fermionic antisymmetry, with no learned electron-count
  switches or molecule/element energy offsets.
- Spatial covariance and spin-rotation structure. Distinguish fixed-S pure
  states from a collinear fixed-Sz density model; do not silently equate them.
- Reference coverage: the trial space contains its stated physical mean-field
  input, and variational refinement can retain that input unchanged.
- One-electron interaction cancellation and appropriate ensemble treatment of
  fractional charge/spin. Do not flatten arbitrary nondegenerate spin states.
- Fragment energy composition, including the initial state and any long-range
  channels; test a changed construction rather than rerun settled checks.
- Density/geometry scaling: fixed one-Bohr moment frames are not evidence of
  transfer across elements or density scales. Use controlled local scales and
  distinguish coordinate covariance from untested numerical approximations.
- General-electron support: an exponential dense tensor limited to N=2/3 is
  not a model for the unseen chemistry panel. Particle-resolved compression is
  required to support the stated scope, independently of later speed tuning.
- A scalar energy and consistent derivatives. Passing symmetry/particle tests
  does not by itself establish useful accuracy.

## The bridge from state experiments to XC

The universal object is the ensemble constrained search

\[
F[n_\uparrow,n_\downarrow]
=\inf_{\Gamma\to(n_\uparrow,n_\downarrow)}
  \operatorname{Tr}\Gamma(\hat T+\hat W),\qquad
E_{xc}=F-T_s-E_H.
\]

Molecular state experiments minimize the molecular Hamiltonian and do not
define this XC functional. The new constrained implementation minimizes T+W
at fixed spatial density moments. It now has a scalar correction hypothesis
and frozen-frame potential in `COULOMB_MOMENT_FUNCTIONAL.md`. Canonical
ensembles, a directional derivative and direct dual density fixed points now
have small-basis evidence; broad accuracy, full density refinement and realistic
bases remain absent. Do not present it as a completed challenge XC model.

The next functional implementation must constrain spatial density moments,
refine them toward grid accuracy, and report the remaining density error. Do
not constrain the complete KS one-body density matrix to remain idempotent;
that would rule out the interacting correlation being sought. Optimize the
universal kinetic-plus-Coulomb energy, with external potentials entering only
the density constraint/ordinary electronic-structure problem.

If lambda and u_s are the interacting and noninteracting constraint potentials,
stationarity gives v_xc = -lambda + u_s - v_H, up to spin-dependent gauge
constants. The new Coulomb-moment implementation expresses this relation in
its moment basis. The failed pure-state water solve supplies no valid
energy/potential; the changed ensemble implementation and callable derivative
check are recorded in `CURRENT_MODEL.md`. A coarse moment
search is an approximation, and its density/basis error must not be called CBS
accuracy.

## Evidence before a generalization claim

All current small-system panels have informed research choices and are
development data. A two-electron sub-kcal result does not establish transfer
to another electron count. The first three-electron extension failed; follow-up
operators remain above the target. Do not promote them as a valid broad model.

The locked confirmation manifest contains 16 new composition families and two
structures per family. Freeze the general-electron model and prediction protocol
before exposing their reference errors. Selection of this panel read no energy
columns; earlier public-data utilities had parsed the source dataset, so this is
a procedural holdout, not a claim that those bytes never entered memory.
Missing labels and failed predictions remain recorded, without adaptive case
replacement. Same-composition differences remove fitted atomic offsets.

That neutral closed-shell panel is still incomplete evidence. Independent
electron-count, spin/charge, dissociation, interaction, basis and reaction-family
tests must support the proposed scope. Full GMTKN55 and hidden-suite accuracy
remain required for the actual objective. Speed work follows a convincing
accuracy/generalization result, rather than making a failed fit cheaper.
