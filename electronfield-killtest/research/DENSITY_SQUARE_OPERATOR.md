# Physical operator without four-orbital word enumeration

17 November 2026. The full public benchmark inventory exposed a concrete
execution failure: the dense integral/word representation cannot fit most
def2-QZVP cases. This new implementation is a required scope primitive,
not a learned XC functional, broad accuracy result, full-suite feasibility
result or claimed breakthrough. Jordan-Wigner, MPO and density-factorization
identities are established math. No frontier model or speed experiment ran.

For supplied real symmetric factors with `(ij|kl) = sum_P L_P[ij] L_P[kl]`,
define `E(A) = sum_ij,s A_ij a†_is a_js`. Then the literal Hamiltonian is
`E(h - 0.5 sum_P L_P L_P) + 0.5 sum_P E(L_P)^2`.

`gpu_density_square_operator.py` builds the one-body automaton directly
from matrix coefficients, with O(n) channels and O(n²) coefficients. It
forms each exact operator product before pulling it into a physical local
section. Factors are processed sequentially, bounding the number of live
factor/product objects instead of enumerating every four-orbital word.
Product-storage and physical-environment caps still apply. Large correlated
bonds remain a memory risk; this prototype does not fit the largest public
cases. The factor input itself is presently a supplied CUDA tensor.

Projecting E(L) into a local section before squaring replaces P E² P with
(P E P)² and loses physical intermediate states. The independent test
detects a 0.4577058634 Hartree error from that wrong construction. The new
implementation retains the full product and its anticommutation counterterm.

The [fixed-protocol CUDA check](density_square_operator_checks_gpu_20261117.json)
passed. The largest difference across an independent four-site Fock matrix
containing all number/spin sectors is 3.55e-15 Hartree. Correlated tensor
energy and complete local-section action differences are below 9e-16
Hartree; the local action is Hermitian at reported precision. A supplied
40-site, 40-electron product state's energy agrees with independent Wick
contraction within 1.12e-16 Hartree. The squared operator occupies 43.15 MB;
the check peaks at 73.10 MB CUDA allocation. The larger case uses no
four-index ERI or quartic word list. **It is a provided-product identity,
not a coupled 40-electron prediction.**

The first matrix comparison incorrectly reversed the reference basis order.
Its interrupted, unaccepted report remains in
`density_square_operator_checks_gpu_20261117_basis_order_invalid.json`.
The diagnostic already uses low-site-bit-first Fock order; correcting only
that comparison produces the accepted report. The Hamiltonian, model,
seed and tolerances were unchanged.

Still missing: a bounded large-basis GPU factor producer, complete spin/ECP
domains, converged molecular accuracy, universal off-shell scalar/derivative
completion and learned transfer. The earlier full-suite failure is not
retroactively repaired by this check. The original BH checkpoint is intact.
