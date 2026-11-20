# Auxiliary-density formulation for the physical state

Research hypothesis, 17 November 2026. No challenge-level accuracy or new
regularization theorem is claimed. Hard-density thermal and direct primal
ensemble searches failed their bounded convergence gates; their retained
physical ensembles remained 63/190 kcal/mol above the saved universal values.
Increasing their iterations or changing penalty weights is not the next test.

The mathematical foundation is established Moreau regularization, which can
preserve physical energies while defining a differentiable auxiliary-density
functional. See [Kvaal et al., JCP 2014](https://arxiv.org/abs/1312.3734).
Our implementation below is a finite-coordinate construction, not a
reproduction of that paper's numerical examples or a frontier method. The
research contribution sought is a transferable charge/spin state and its
density response; no breakthrough has yet been demonstrated.

For the fixed spatial frame, q(Gamma) contains only the variable density
coordinates; exact N is retained separately. Define

\[
F_\epsilon(x)=\min_{\Gamma\in\mathcal S_{N,S=0}}
 \operatorname{Tr}\Gamma(T+W)+\frac{\|q(\Gamma)-x\|^2}{2\epsilon},
\qquad \nabla F_\epsilon(x)=\frac{x-q(\Gamma_x)}{\epsilon}.
\]

Gamma is positive and has unit trace inside an exact fermionic N,S sector.
Its density is physical. The auxiliary x is a separate coordinate and is
not asserted to equal that density or to be N-representable itself.

For any Gamma, x and physical potential coefficients v, direct completion
of the square gives our implementation identity

\[
\operatorname{Tr}\Gamma(T+W)+\frac{\|q(\Gamma)-x\|^2}{2\epsilon}
+v\cdot x+\frac\epsilon2\|v\|^2
-\operatorname{Tr}\Gamma(T+W+v\cdot Q)
=\frac{\|q(\Gamma)-x-\epsilon v\|^2}{2\epsilon}.
\]

Thus minimizing over x restores the unmodified physical Hamiltonian for
every positive epsilon. The physical density is q(Gamma)=x+epsilon v at
stationarity. This explicit counterterm/recovery is essential: omitting it
would make a density penalty an approximation to the wrong problem.

The matching kinetic functional minimizes Tr(TD)+||Q:D-x||^2/(2epsilon)
over 0<=D<=2 and Tr(D)=N. Its actual D is a fixed-N representable ensemble.
Subtracting this kinetic scalar and a consistently defined quadratic Hartree
term on x gives an auxiliary XC scalar and derivative. A regularized KS
implementation must use the same auxiliary variable throughout; it is not
an unchanged conventional KS plugin. Challenge integration/eligibility for
this formulation remains unverified.

Reduced positive-state optimization uses a computable conditional-gradient
gap. This bounds the error in the stated reduced space. Full-space physical
response and lowest-state checks remain separate; a reduced-space bound is
not evidence for full-basis/CBS or WTMAD-2 accuracy.

The present metric is the norm of the finite density coordinates. No fixed
continuum metric or realistic-basis transfer is established. Exact recovered
physical energies do not depend on this numerical regularization when the
state search is complete. Finite state approximations still require basis,
fragment, charge/spin, density-path and independent reaction-family tests.

## Forward development results

The accepted report is `auxiliary_stationary_gpu_20261117.json`. Two BeH2
geometries in 6-31G now pass the original full stationarity tolerance of
1e-5 Hartree and potential self-consistency tolerance of 1e-6. Same-basis
energy errors are +1.964e-8 and +4.801e-8 kcal/mol; full stationarity
residuals are 9.279e-6 and 6.745e-6 Hartree. Number, spin, Pauli occupations,
kinetic search and regularized KS commutator checks pass. Additional
lowest-state Ritz residuals are below 9e-9 Hartree. This is numerical
lowest-state evidence, not a rigorous independently implemented lower bound.

Predictions began with a mean-field density, not a correlated oracle density.
The universal evaluator receives T, W, Q, occupation metadata and N, with
the auxiliary coordinate and physical seed separately. Molecular potential
coefficients appear only in the outer physical variational calculation.
The exact square-completion identity is checked at every accepted step.
References were already cached and assessed after stopping; no new label
or energy-error stopping rule was used.

These are post-interruption development completions. The original forward
attempt stopped during a WSL environment failure. A subsequent numerical
variant exposed a spin/energy implementation error and was rejected. The
corrected variant explicitly purifies all response directions, checks spin
before projection and uses the literal physical Hamiltonian action. Its
initial capped result still failed the full residual gate. One prespecified
restart retained the occupied positive ensemble, density, input reference
and physical energy to numerical precision and finished the original
residual criteria, with the original dimension cap, epsilon and tolerances.
The failed/interrupted reports and original criteria remain preserved.

Fresh fixed auxiliary directions are checked in
`auxiliary_derivative_gpu_20261117.json`. Scalar versus derivative
differences are 1.579e-8/1.408e-7 Hartree at step 0.001, below the declared
1e-5 threshold. Each test freezes the existing response basis, solves the
positive interacting/kinetic problems at both displaced points and uses no
molecular potential or new reference. It verifies the implemented scalar
derivative in that frozen space; it does not certify arbitrary full-space
density paths, nuclear forces or response-basis derivatives.

The backend still stores global Fock vectors of length 81796, and covers
only four/six particles or holes with even-N singlet states. The final
response dimensions of 29/191 are not evidence for scalable compression.
There is no learned predictor, complete-basis result, broad reaction-family
generalization or WTMAD-2 score. A compositional state/action supporting
realistic systems is the next accuracy requirement. Regularization and
generic tensor/optimization mathematics alone do not establish novelty.
