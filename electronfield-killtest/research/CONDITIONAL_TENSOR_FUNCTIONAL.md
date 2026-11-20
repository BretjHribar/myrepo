# Conditional auxiliary functional on physical tensors

17 November 2026. The local physical tensor state is now connected to the
universal T+W objective, positive ensemble optimization and a consistent
conditional XC scalar/derivative. **A fixed environment is not the complete
universal search. Broad chemical accuracy and the challenge remain unproved.**

## State and objective

Hold all surrounding tensors fixed and let J map local reduced CG
coefficients into a physical global state, with the other ancillas retained.
Every column of J lies in the declared fermionic N,S sector. Resolve its
true overlap G using direct physical boundary factorizations and a direct
embedding SVD. Choose B with B^T G B=I. A local R>=0, Tr R=1 then represents
a positive global density operator after tracing the other ancillas.
No global occupation vector or occupation-link table is needed.

With H_C=B^T J^T(T+W)JB and Q_C,a=B^T J^T Q_a JB, define

\[
F_{\epsilon,C}(x)=\min_{R\succeq0,\operatorname{Tr}R=1}
 \operatorname{Tr}(R H_C)
 +\frac{\|q_C(R)-x\|^2}{2\epsilon},\qquad
q_{C,a}(R)=\operatorname{Tr}(R Q_{C,a}).
\]

The fixed-context problem is convex. Its minimizing density coordinate is
unique even when R is not, giving gradient (x-q_C(R))/epsilon. The reduced
Frank-Wolfe gap checks this stated conditional problem; it is not a full
Hilbert-space or global tensor ground-state certificate. The physical
density remains distinct from x, which need not be representable.

The code accepts only T,W,Q,N,S in its universal operator interface. A
tensor context is a separate explicit argument. Molecular external h,
nuclear charges, reference labels and Fock bits are rejected as universal
inputs. Surrounding tensors can depend on a previous physical calculation;
that dependence is exactly why the result is labeled conditional.

## Energy recovery and updates

Let L_epsilon,C(R,x) be the statewise objective inside the minimization.
For any represented state and physical potential coefficients v,

\[
L_{\epsilon,C}(R,x)+v\cdot x+\frac\epsilon2\|v\|^2
-\operatorname{Tr}\Gamma(T+W+v\cdot Q)
=\frac{\|q(\Gamma)-x-\epsilon v\|^2}{2\epsilon}
\]

At a minimizing R, L equals F. Fixed-number external constants and
nuclear repulsion are added separately. No soft-density mismatch is
silently treated as a physical energy.

If x=q_old-epsilon v and the old state is retained in the local section,
minimizing the conditional objective implies

\[
E_{\rm physical,new}
+\frac{\|q_{\rm new}-q_{\rm old}\|^2}{2\epsilon}
\le E_{\rm physical,old}.
\]

This gives a reference-independent physical monotonicity test. Installing
R uses its positive square root as a local spinless ancilla. Only numerical
null probability is removed; trace, physical density, number/occupations
and literal T+W energy are checked after installation.

The matching kinetic envelope uses 0<=D<=2, Tr D=N, with the same Q,x and
epsilon. Subtract it and the stated auxiliary Hartree quadratic to define
the conditional XC scalar. Its derivative is the difference of envelope
gradients minus that Hartree derivative. The kinetic/XC interface is
currently validated only for even singlets; odd-spin physical operator
tests do not establish an open-shell functional.

## What has passed

`local_tensor_functional_checks_gpu_20261117.json` independently verifies
the local compiler against full small-system physical operators, with
surrounding ancillas traced explicitly. Singlet and doublet sections have
dimensions 40/36. Hamiltonian and density-operator differences are below
4e-15, and input/overlap identities pass. Positive conditional solves meet
the original 1e-11 Hartree gap tolerance; installed states reproduce their
literal energy and density below 5e-15. The even-singlet conditional XC
directional derivative differs by 5.13e-8 Hartree at step 0.001.

Checkpoint recovery is tested on a separate analytic positive-state
problem. The run is interrupted after iteration zero and completed from
its saved FISTA state, with analytic state error 3.94e-11 and no duplicate
history steps. A completed checkpoint requires no new iterations. Source:
`resolvent_checkpoint_checks_gpu_20261117.json`. No chemistry or reference
solve was repeated for this recovery test.

The larger-basis test is specified in
`tensor_water_conditional_protocol_20261117.json`. Its authoritative status
is `tensor_water_conditional_gpu_20261117.json`; do not infer an accepted
functional from compilation progress. It uses the saved unconverged
10-electron/24-orbital cc-pVDZ state, a new spatial product frame, one fixed
center and an off-shell auxiliary coordinate. No SCF or correlated reference
is repeated. A successful result would establish the frozen-context
conditional search and derivative only.

The larger-basis run is now terminal and passed its stated conditional
gates. The direct spatial frame has 280 operators (279 variable coordinates),
with independent point error 3.97e-14 and subsequently assessed external
span error 3.08e-12 Hartree. The physical local dimension is 692; overlap
and input-retention errors are about 2.5e-12/4.0e-14.

The off-shell positive search reached a 5.52e-12 Hartree conditional gap
at iteration 1100; the matching kinetic gap is 1.42e-14. Installing its
rank-one local purification preserves moments to 1.95e-13, literal T+W
energy to 5.57e-12 Hartree and particle number to 1.78e-15. Physical
occupations range from 0.000151 to 1.999857. The physical-energy recovery
identity differs by 5.59e-13 Hartree. New plus/minus auxiliary directions
give an XC scalar/potential difference of **7.37e-9 Hartree**, with the
surrounding tensors held fixed and no potential supplied to the universal
evaluator.

The represented state's physical energy decreases from -76.1692023251 to
-76.2116024360 Hartree. These are variational state energies, **not
reference-energy errors**. No chemical reference was loaded/generated,
and global tensor stationarity is unmeasured. The result cannot establish
the sub-kcal goal. The full conditional experiment took about 592 seconds,
including compilation, and about 1.64 GB peak CUDA allocation. These are
resource observations, not a speed benchmark or timing target.

`tensor_section_recovery_gpu_20261117.json` restores all 35 immutable
operator chunks and the completed central optimizer checkpoint. State,
scalar and gradient match exactly, with zero local operator matrices recompiled
and zero new optimizer steps. The universal MPO is rebuilt from saved T,W
integrals for the recovery check. An independent conditional-gap recomputation
also passes. Do not repeat this settled single-center experiment; the
remaining work is common stationarity across tensor environments and
independent realistic-basis relative-energy accuracy.

## Required next evidence

The initial all-site forward search is now closed: a metric guard stopped
it after four valid updates, and its fixed charge/spin seed sectors also
exclude triple and higher virtual excitations. A numerical gauge repair
passed without changing the state or tolerance. It cannot remove the
physical restriction. The changed
[two-site sector completion](TENSOR_SECTOR_COMPLETION.md) adds complete
intermediate sectors through the physical residual while retaining the
old state. This is forward physical response work; it has not established
the full arbitrary-density universal minimum or its global derivative.

For a global tensor value, changes of surrounding tensors contribute to
the derivative unless their variational stationarity is established and
the envelope conditions apply. A single local gap or frozen-context
derivative cannot establish that. Global tensor stationarity, context/
initialization dependence, fragment optimization, open-shell definitions,
basis/frame response, complete-basis errors and independent reaction-family
transfer remain separate work. The locked confirmation panel stays withheld.

Moreau regularization, CG coupling, purification and tensor linear algebra
are established primitives. They are not an AlphaFold-scale breakthrough
claim. The research hypothesis is a transferable physical density-response
state with a reliable scalar and potential. No learned/amortized model or
sub-1 kcal/mol WTMAD-2 result has been validated. Speed work stays disabled.
