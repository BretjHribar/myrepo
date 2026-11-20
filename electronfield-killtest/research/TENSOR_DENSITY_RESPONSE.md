# Compositional physical state and density response

17 November 2026. **Representation and operator scope evidence; no
chemical-accuracy or challenge result.** Speed work remains disabled.

The accepted auxiliary-density evaluator still uses global Fock vectors.
The new tensor components below address that limitation before attempting
larger accuracy tests. A [conditional local tensor functional](CONDITIONAL_TENSOR_FUNCTIONAL.md)
now connects them to T+W, positive ensemble optimization and scalar/derivative
checks on 24-orbital water. This has not replaced the complete global search.

The subsequent fixed-sector all-site family is closed after a retained
metric failure and discovery that its seed excludes triple and higher
virtual excitations. [Two-site sector completion](TENSOR_SECTOR_COMPLETION.md)
now tests missing physical N,S directions at the same bond cap, retaining
the old state exactly. A passed local sector test is not a full correlated
residual or a chemical-accuracy certificate.

## Physical state

`gpu_purified_spin.py` attaches a local spinless ancilla to each spatial
orbital. Its contraction is a purification; tracing ancillas gives a
positive density operator. Reduced CG tensors enforce exact fermionic
particle number and total spin, including odd-electron/highest-weight
sectors. Independent singlet fragments compose by concatenation and have
an exact embedding in the same reduced tensor family. A fragment product
does not require a fixed global list of determinant/mixture labels.
Automatic fragment discovery and the size consistency of a future bounded
global optimizer are separate, unproved requirements.

Purification, spin coupling, MPS and automaton operator factorizations are
established mathematics. See [Schollwoeck's review](https://arxiv.org/abs/1008.3477)
and [Crosswhite and Bacon](https://arxiv.org/abs/0708.1221).
No published model, paper benchmark or frontier implementation was replicated.
The research hypothesis remains a transferable physical density-response
state and its consistent XC scalar/potential; no breakthrough is established.

`gpu_fermion_mpo.py` compiles our literal one-/two-electron fermion words
into a sparse operator automaton, retaining all nonzero integral terms.
CUDA carries coefficients, operator algebra and contractions. The host
enumerates integer labels and graph shapes only. No global occupation
vector is needed. The operator is never replaced by P H P on states that
may lie outside P. Scientific tensors remain real float64 on CUDA.

Independent complete four-orbital operators agree to 4.0e-15 Hartree;
mixed-state energies/parameter gradients agree below 9e-16. Number and
spin sector tests pass for N=4 singlets and N=3 doublets. Eight supplied
independent fragments, 32 electrons in 24 orbitals, have an energy
additivity error 4.9e-15 Hartree with 576 tensor parameters. A single full
fixed-Sz vector for that product would need about 4.33 TB, and was not
allocated. This is a supplied-product identity, not a coupled-molecule
accuracy test. Source: `purified_operator_checks_gpu_20261117.json`.

## Response and the rejected truncation metric

`gpu_spin_response.py` retains complete spin multiplets in every tensor
response cut and explicitly projects the endpoint into the target N,S,M
sector. The no-truncation tests agree with full Hamiltonian images to about
2e-14. Even severely truncated tests retain exact N/S; their poor image
accuracy is recorded rather than accepted as convergence.

The initial doublet test mistakenly called a cap of 256 untruncated. It
hit that cap and failed the exact-image check. Only the unfinished doublet
checks were completed using the algebraic complete prefix-space bound;
the actual required bond was 262. The operator, source, seed and tolerances
were unchanged. Both reports remain: `spin_response_checks_gpu_20261117.json`
and `spin_response_capacity_completion_gpu_20261117.json`. This was a
diagnostic-capacity correction, not a chemical rank sweep.

The first molecular response used the Euclidean norm of raw operator
channels. That norm depends on the representation of the operator, so
local discarded norms do not measure physical state error. A new exact
physical-image assessment found its two-state water span retained only
**2.706% of the squared Hamiltonian-response norm**. This raw-channel
selection route is closed for chemistry; do not rescue it with larger caps.

`gpu_physical_response_metric.py` changes the metric, keeping the same
molecular bond cap of 64. For a product source it contracts the exact
sparse physical Gram, builds a right physical frame and then selects full
spin multiplets by their physical Schmidt weights. It measures the full
image difference using norm/overlap identities without a global vector.
Independent singlet/doublet tests validate those identities and the
sequential projection bound below 4e-15. Bounds are numerical float64
evidence, not interval-arithmetic proofs. The exact Gram implementation
currently requires a product source; it does not certify the full residual
of an arbitrary correlated tensor state.

## First coupled polarised-basis molecule

Water at a new development geometry uses all 24 cc-pVDZ orbitals and 10
electrons. A full fixed-Sz vector has 1,806,590,016 coefficients, about
14.45 GB in float64, and was not allocated. The literal tensor Hamiltonian
uses 17.64 MB of CUDA operator storage. CPU integral calls were blocked;
one bounded CUDA RHF input converged and was saved for reuse.

| Response metric, same cap 64 | Squared initial response norm retained by the two-state span | Total energy, Hartree |
|---|---:|---:|
| Raw operator channels, closed | 0.0270648 | -76.0271374064 |
| Physical Schmidt metric | 0.9861024 | -76.1692023251 |

The retained input determinant has total energy -76.0240386654 Hartree.
These are variational state energies, **not reference-energy errors**.
No reference label was generated or loaded. The physical-metric image's
relative error estimate is still 0.117888, so its cap is material. Its
right-frame numerical triangle bound is 1.33e-12 in state norm. The
resulting two-state calculation passes number, spin, Hermiticity, energy
reconstruction and input-retention checks, but remains unconverged. No
sub-kcal, complete-basis, broad-transfer or WTMAD-2 score follows from it.

The original and changed-metric reports are
`tensor_water_scope_gpu_20261117.json` and
`tensor_water_physical_metric_gpu_20261117.json`. The changed-metric run
used about 0.793 GB peak CUDA allocation and 84 seconds, including setup;
these are resource observations, not a speed benchmark or timing target.
No SCF/reference solve or failed raw-channel selection was repeated.

## Density interface and remaining functional work

`gpu_tensor_density.py` differentiates a complete literal one-body operator
to obtain real symmetric self/transition densities of pure or purified
states. Independent finite-occupation checks agree below 9e-16 for singlets
and doublets, including cross-number identities. On the saved water state,
Tr D differs from 10 by 1.78e-15 and occupations range from 3.57e-5 to
1.99838. The state optimization was not repeated. Sources:
`tensor_density_checks_gpu_20261117.json`, `tensor_water_density_gpu_20261117.json`.
These are fixed-state derivatives, not nuclear-force or full XC derivatives.

The new conditional model uses this physical density interface in the
universal T+W auxiliary objective, with external potential/reference labels
excluded from its universal evaluator. Its fixed-context search and
scalar/potential checks pass. Common stationarity across tensor environments,
controlled global correlated response and a prespecified reference-independent
convergence rule remain required.
Do not promote the two-state physical test as a functional, repeat tiny
operator checks, expand a failed fixed global rank, or optimize runtime.
Realistic-basis relative energies, open-shell functional coverage and blind
reaction-family transfer remain required. The locked panel stays withheld.
