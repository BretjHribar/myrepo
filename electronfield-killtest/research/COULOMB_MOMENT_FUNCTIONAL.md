# A density functional with Coulomb-complete moments

Research hypothesis, 16 November 2026. Accuracy and physical transfer come
before speed. This is our proposed construction, not a claim of worldwide
novelty or of challenge performance. Constrained search is established
[Levy theory](https://pmc.ncbi.nlm.nih.gov/articles/PMC411802/); Ritz/Krylov
linear algebra and Kohn-Sham duality are mathematical building blocks.

## Define an actual scalar correction

Let D be the spin-summed, noninteracting input density matrix in the stated
orthonormal finite basis. Use spatial multiplication operators Q_a and
q_a[D] = Tr(D Q_a). At each nuclear position R_A include a Coulomb probe
1/|r-R_A|. The nuclear charges are coefficients of the external potential,
not inputs to the universal T+W objective. Remove only the fixed-number gauge
and linear dependencies.

For the frozen moment frame define

\[
F_M(q)=\inf_{\Gamma\to q}\operatorname{Tr}\Gamma(T+W),\qquad
T_{s,M}(q)=\inf_{\Gamma_s\to q}\operatorname{Tr}\Gamma_s T.
\]

The proposed correction is

\[
E_{xc}^{M}[D]=F_M(q[D])-T_{s,M}(q[D])-E_H[D].
\]

Retain the ordinary noninteracting kinetic energy outside this search:

\[
E_M[D]=\operatorname{Tr}(TD)+F_M(q[D])-T_{s,M}(q[D])
       +\operatorname{Tr}(v_{ext}D)+E_{nuc}.
\]

The term Tr(TD)-Ts_M(q) supplies kinetic cost for density variations the
coarse moments do not see. Omitting it would leave those density directions
without their kinetic penalty. This is a change in model structure, not an
energy-offset fit or an added optimizer sweep.

If lambda and u_s are the interacting and noninteracting constraint
potentials, both defined by adding +potential.Q to their Hamiltonians, the
frozen-frame derivative is

\[
v_{xc}^{M}=(-\lambda+u_s)\cdot Q-J[D].
\]

This expression requires stationary constrained solves. A feasible but
unconverged state does not supply a trustworthy functional derivative.
Geometry-dependent frame and basis derivatives must also be included for
nuclear forces. The numerical prototype does not yet provide those forces.

The fixed-Q definition must not be confused with a single universal
density-only functional across changing Q. Current probe centers come from
the supplied nuclear geometry. Even though nuclear charge weights do not
enter T+W, that frame dependence is an unresolved model/eligibility issue.
Making the frame a specified density-dependent or variational quantity would
require its response/envelope derivative and independent transfer evidence.

## Why include the Coulomb probes

For point nuclei the actual external operator is in their span:

\[
v_{ext}=v_0 I+\sum_a v_aQ_a.
\]

Matching q therefore matches the complete external energy in the stated
basis. Arbitrary regional radial moments alone do not provide this property.
The code checks both the individual charge probes and reconstruction of
h-T, with all integrals evaluated on CUDA and CPU integral entry points
blocked. ECP/nonlocal pseudopotential cases are outside this implementation.

The exact moment definitions have a useful energy property. Any admissible D
has Tr(TD) >= Ts_M(q[D]). Any physical interacting state matching q has
<T+W> + v.q + N v_0 >= the exact electronic ground energy in the same basis.
Consequently the proposed total energy is a variational upper bound there.
No correlated energy labels are required for this statement.

The implementation uses a feasible interacting state as an upper estimate of
F_M and a noninteracting Legendre **lower bound** for Ts_M. Those choices
preserve the energy bound even before exact minimization, apart from the
reported moment residual: add |v.(q_state-q_input)| to obtain the external
energy error allowance. An accurate energy still requires those bounds to be
tight. A small eigenstate residual alone does not certify that the lowest
state was found; reference assessment and complete failure accounting remain
necessary.

For a fully converged search, one-electron W=0 makes F_M=Ts_M and gives
E_xc=-E_H. The exact definitions also admit ensemble densities. The first
gapped-determinant implementation failed on water and is closed. Its
replacement optimizes 0<=D<=2 with a number constraint and a small, stated
binary-entropy regularizer. An explicit canonical lift demonstrates fixed-N
singlet representability of the fractional 1RDM. The interacting replacement
mixes physical states within a fixed-N,S sector; a constant entropy allowance
keeps a conservative energy estimate. These regularized variants have a
finite energy tolerance, not exact zero-temperature one-electron cancellation.
The callable interface still excludes arbitrary open-shell/fractional-spin
inputs.

## Transfer and composition requirements

Coulomb probes transform as scalars under translations/rotations and scale
with the inverse length under uniform coordinate scaling. Their definition
does not contain a fixed one-Bohr radius or learned element identifiers.
Finite-basis numerical covariance and solver convergence remain to be shown
across larger bases and chemistry.

Independent fragment composition must apply to the state, the constraints,
and both constrained searches. The number/spin tensor family now has an exact
product embedding for explicitly supplied ordered singlet components. That
does not make the finite global residual subspace automatically separable.
Component charge constraints, treatment of spin coupling/fractional charges,
and a factored solve remain required before claiming general size consistency.

This construction is still expensive. Its present dense occupation action is
a reference-scale prototype; reducing it to a compact, transferable state
model is unresolved. Calling the near-exact small-basis solver a learned
generalization breakthrough would be misleading.

## First changed-chemistry experiment

`coulomb_functional_protocol_20261116.json` fixed a new neutral water geometry,
the complete STO-3G basis, the input HF procedure and residual stopping rules
before generating its reference energy. `gpu_coulomb_functional.py` evaluates
the constrained correction at that input density. Both F_M and Ts_M use no
reference-energy loss. The exact molecular label is generated only after the
candidate has stopped. Report one-shot density error, constraint residuals,
state/kinetic convergence, potential-span error and all failures.

The pure-state version failed its constraints. The changed ensemble version
converged but its one-shot HF-density energy missed by 305.65 kcal/mol. That
input has a large kinetic mismatch with the coarse representation. This
failure remains part of the record.

The direct dual fixed point lambda=v_ext followed by noninteracting moment
inversion reached +0.001650 kcal/mol on development water and +0.000991 on a
new 18-electron H2S application, in complete minimal bases. This is an
expensive implicit solve, not a trained energy predictor or conventional KS
runtime result. Explicit fixed-N singlet ensembles reproduce those moment
densities. The callable finite-entropy scalar's frozen-frame derivative also
passed one density-rotation check to 3.95e-9 Hartree. See `CURRENT_MODEL.md`
for the distinction between numerical variants and their limits.

These findings establish neither complete-basis accuracy nor broad transfer.
Independent larger-basis chemistry, balanced reaction energies, automatic
fragment consistency, force derivatives and the challenge's hidden chemistry
remain outstanding. The locked 32-structure composition panel stays withheld.
