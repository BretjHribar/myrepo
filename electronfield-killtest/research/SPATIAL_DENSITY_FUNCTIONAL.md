# Spatial density constraints for the charge/spin state

Research construction, 17 November 2026. Accuracy and physical validity come
before speed. This records a finite-basis functional direction, not a broad
accuracy result, new theorem or completed general inference model.

The older nuclear Coulomb-probe frame spans the molecular external potential,
but changing its probe locations with nuclear positions does not establish a
single density-only functional. The new construction aims to constrain the
represented spatial density itself. Basis functions and the universal
Coulomb kernel are allowed numerical inputs; the external potential and
correlated energy labels do not select the density frame.

The first attempt used the eigenvectors of the orbital-product Coulomb Gram
matrix. It failed. At a prespecified 1e-13 relative eigenvalue threshold,
weak modes were lost: the BeH2/6-31G potential-span error reached 1.71e-5
Hartree and the orbital-coordinate projector error reached 0.0122. Small
Coulomb eigenvalues cannot safely be discarded just because they look small.
That report is retained; no threshold sweep rescues this construction.

The changed representation resolves the product span directly. Evaluate
orthogonalized orbitals on CUDA at basis-scaled spatial points, normalize
each orbital-value vector, and apply SVD to its rank-one density matrix.
Normalization preserves the span while retaining overlap-region directions.
The direct SVD avoids squaring the condition number in a Gram matrix.
Keep every resolved direction at the prespecified threshold and add exact
particle number. No nuclear charge, external h or energy label enters this
construction. Point locations use basis centers/exponents as numerical
coverage metadata; this is not a fixed continuum grid implementation.

For the two new BeH2/6-31G geometries, independent spatial points reproduce
the normalized density-product matrices within 6e-14. Orbital-coordinate
projectors agree within 5e-13, and the subsequently assessed external
potential span errors are below 8e-13 Hartree. The frames have dimensions
88 and 87 within the 91-dimensional real symmetric matrix space. This is
numerical represented-density resolution, not a proof that arbitrarily
large external potentials or the continuum limit are controlled.

For fixed numerical basis/frame Q, define the singlet ensemble constrained
search and its matching noninteracting kinetic search:

\[
F_Q(q)=\inf_{\Gamma\to q}\mathrm{Tr}\,\Gamma(T+W),\qquad
T_{s,Q}(q)=\inf_{0\le D\le2,\,\mathrm{Tr}D=N,\,Q:D=q}\mathrm{Tr}(TD),
\]

\[
E_{xc,Q}[D]=F_Q(Q:D)-T_{s,Q}(Q:D)-E_H[D].
\]

These are established constrained-search definitions. The research work is
the physical representation, constraint resolution and eventual transferable
approximation; exact Levy/Kohn-Sham or SVD mathematics is not our novelty.

A ground state in a potential V inside the frame provides an on-shell
certificate: its universal value is E minus nuclear and external energies,
and the constrained-search gradient is -V, up to number gauge. The saved new
charge/spin states provide those points without repeating interacting solves.
The independent noninteracting ensemble search uses the same spatial
constraints with a stated 1e-5 Hartree entropy allowance. It gives density
residuals below 3e-10 and KS commutators below 3e-13. Reconstructing total
energies differs from the saved physical states by about 0.00035 and 0.00144
kcal/mol, within that allowance. These are numerical functional certificates
in the stated basis, not reference-free predictions on a broad chemistry set.

With interacting and kinetic constraint potentials lambda and u_s, the
frozen-basis/frame derivative is

\[
v_{xc}=(-\lambda+u_s)\cdot Q-J[D].
\]

Separate prespecified plus/minus physical-potential paths test this derivative
in `spatial_derivative_gpu_20261117.json`. These new states are not reference
labels or repeats of the central calculation. The report, including any
failure, determines what derivative evidence is supported.

Both paths passed: scalar/potential directional derivatives differ by
1.36e-7 and 5.74e-8 Hartree, against the prespecified 1e-5 tolerance. The new
interacting states have residuals below 9e-10 Hartree and kinetic density
constraints below 5e-10. This tests one fixed direction per geometry, not
all densities or nuclear derivatives.

The on-shell charge/spin initializer uses the molecular h to choose its
local subspace. That selected projector cannot simply be reused as a
universal interacting functional. An arbitrary-density implementation must
choose/optimize its state using only T, W, Q and the input density, and
include any density-dependent basis response in the scalar derivative.
The present certificates use near-full-space stationary states; they do not
solve this amortized representation problem. Energies refer to the stated
singlet sector, not a guarantee about a lower state of another total spin.

The larger-basis code currently supplies these on-shell values and path
diagnostics. It does not yet implement an arbitrary-input-density interacting
search with demonstrated convergence, general open-shell/ensemble spin
targets or a learned predictor. A bounded implicit implementation in
`xc3d/spatial_xc_model.py` now accepts only T, W, Q, occupation metadata and
the input density. Its oracle-density inversion diagnostic is recorded in
`universal_search_gpu_20261117.json`; absence of a converged report is not a
successful functional evaluation. The initial incomplete numerical attempt
and the unchanged-objective repair are separately recorded in
`universal_search_interrupted_gpu_20261117.json`.

**The bounded universal-interface diagnostic failed.** The first numerical
attempt did not return its initial ground solve after eight minutes and was
interrupted. The existing preallocated Ritz implementation completed that
unprojected attempt but failed its residual, with no XC output accepted.
The original singlet initialization was also insufficient to guarantee spin
through arbitrary inverse-potential searches, so every trial vector/image
was then projected into S=0 using the exact polynomial in S squared.

That changed physical implementation completed three ground solves, with
density residuals 3.78, 3.63 and 3.52. A subsequent trial potential reached
its declared Ritz capacity with residual 2.52e-5 Hartree rather than 1e-8;
its S-squared state norm was 7.21e-16. The search rejected the result and
returned no XC scalar or potential. The failed unprojected report is also
retained separately. Do not increase iterations, loosen constraints or feed
the known molecular potential into this diagnostic to turn it into a pass.

The proposed cold, pure-ground-state dual route has not established a
usable general functional. The next substantive option is a feasible
fixed-number ensemble constrained search with physical response spaces,
alongside the required many-block tensor action. Those change the domain/
representation; they are not speed tuning or a rescue by larger rank/seed
sweeps. On-shell and directional-derivative successes above remain valid
limited evidence and do not override this failed functional-interface gate.
The solver still stores a global Fock vector. Many-block tensor actions,
realistic-basis reaction coverage, continuum density refinement and basis
response for nuclear forces remain required. The locked chemistry panel
stays withheld and speed optimization remains deferred.
