# GPU Kohn-Sham pipeline and our fitted functional

Started 20 November 2026, after the user handed over control of the direction
("You are now in control ... ONLY USE THE GPU! if you need space use the D drive").
Code: `electronfield-killtest/xcgpu/`. Bulk outputs: `D:\xcresearch`.

**Status: the accuracy goal is not met.** Measured numbers are in
[the results section](#results) and carry their coverage. Nothing here is an
organizer-suite result.

## Why the direction changed

The tensor-network state work could not be evaluated on the benchmark: its
last recorded coverage was 11 structures and 5 of 1,505 reactions
[measured, `gmtkn55_N1_component_score_gpu_20261117.json`], with 2,417 of
2,442 structures blocked by memory or domain limits. Its cost model also has
no route to the challenge's time box (about 10 CPU-seconds where PBE takes 3).

A density-functional calculation on the GPU reaches every structure. That was
not available before because GPU DFT crashed in this environment.

## The crash that blocked GPU DFT, and its fix

- Symptom: exit code 139 (segmentation fault) on the first SCF iteration.
- The old log blamed a `libcusolver` import error. That was already solved by
  `tools/linux_gpu.sh`; it was not the blocker.
- Mechanism [measured]: `gpu4pyscf-cuda12x 1.9.2` crashes inside
  `libgdft.GDFT_xc_gga` when paired with `gpu4pyscf-libxc-cuda12x 0.9.0`.
  LDA works, any GGA crashes, in the main thread, with pristine installed files
  (all 333 files match their recorded install hashes). It is an ABI mismatch
  between the two published wheels.
- Fix [measured]: the 0.8.1 functional library. Its `libxc.so` replaces the
  0.9.0 one; the original is kept at `D:\xcresearch\wheels\libxc090_original.so`.
  `pip list` still reports 0.9.0.

## Pipeline

One GPU job at a time per pass, one pinned host core, every structure
checkpointed to its own file so interrupted runs resume.

| Pass | Module | What it produces per structure |
|---|---|---|
| Inventory | `xcgpu.inventory` | Basis size, electron count, ECP flag (bookkeeping only) |
| Kohn-Sham input | `xcgpu.run_scf` | PBE/def2-QZVP total energy and components, saved orbitals; RKS or UKS; density-fitted Coulomb (`def2-universal-jfit`), grid level 3, convergence 1e-9, three-stage retry schedule |
| Orbital-dependent ingredients | `xcgpu.run_corr` | Exact exchange of the PBE determinant; shifted second-order energies; single-excitation terms; eigenvalue spectra of the RI density-response matrix on a frequency grid (total and per spin) |
| Semilocal ingredients | `xcgpu.run_semi` | Densities, gradient invariants and kinetic-energy densities on the SCF grid (stored); integrated energies of 20 exchange and correlation components |
| Fit and validation | `xcgpu.fit`, `xcgpu.fitlib` | Linear-in-parameter functionals, in-sample and cross-validated errors |
| Scoring | `xcgpu.collect`, `xcgpu.quick_score` | Predictions in the pinned contract, scored by the existing `xc3d.gmtkn55` suite |

Settings follow the organizer protocol where it is published (def2-QZVP). ECPs
are the def2 ones for elements beyond Kr. No diffuse functions are added for
anions; that differs from the original GMTKN55 paper's practice and is a
recorded convention, not an oversight.

### Checks built into the passes

- The PBE exchange-correlation energy recomputed from the saved orbitals must
  equal the SCF's value. Observed mismatch: at most about 1e-14 Hartree [measured].
- The second-order term of the response-spectrum integral must equal twice the
  independent opposite-spin PT2 energy. On water the difference is 4e-7 Hartree
  with the production grid [measured].
- Electron count from grid integration is reported per structure.

### Frequency integration

Trapezoid rule in ln(omega), step 0.6, from (smallest excitation)/40 to
(largest excitation)*40, with analytic end corrections. A first version
omitted the Euler-Maclaurin end terms and was wrong by about 1e-4 Hartree on
water; the corrected rule is accurate to about 4e-7 Hartree there [measured].
Near-degenerate systems keep their true excitation energies (floor 1e-7).

### Frozen core

Chemical core. s-block metals keep their outer semicore shell active (Li, Be:
none frozen; Na, Mg: 1s; K, Ca: up to 2p). ECP elements freeze the outermost
core shell left by the ECP.

### Auxiliary bases

`def2-universal-jkfit` for exchange; `def2-qzvp-ri` for the response. PySCF
has no `def2-qzvp-ri` for Sn, Sb, Te, I, Pb, Bi; those elements get a generated
even-tempered set (`pyscf.df.autoaux`, angular momentum capped at 6).

## The functional being fitted

Energy of a structure:

    E = E_PBE - E_xc[PBE] + E_x[exact, PBE orbitals] + E_c
    E_c = E_c[RPA, PBE response] + sum_j theta_j * F_j

The `F_j` are size-extensive scalars of the PBE determinant: semilocal exchange
and correlation components, shifted second-order and single-excitation terms,
and integrals of a learned function of the response eigenvalues. With every
`theta` at zero the model is exact exchange plus direct RPA on PBE orbitals,
which is an established method and is used here as the unfitted reference
point.

- It is evaluated **after** a PBE SCF, not self-consistently. That is a stated
  limitation.
- The learned spectral function is in the spirit of published work on
  functionals of the RPA response eigenvalues. We did not install, run or
  reproduce any published parameterization; the basis, constraints, data and
  fit here are ours. This is a judgment call under the no-frontier-replication
  rule and is flagged for the user.
- Parameters are fitted to reaction energies. A fit to GMTKN55 itself is
  **in-sample**; it is reported next to two cross-validations (random folds
  within subsets; leaving a whole subset out) and is never presented alone.

## Known gaps

- `W4-11/c2` (singlet C2) does not converge in RKS PBE at any retry stage. It
  is listed as failed.
- Structures above roughly 1,700 basis functions exceed the on-GPU budget of
  the response pass. An out-of-core path is not built yet.
- No dispersion-specific term yet beyond what the response spectrum carries.
- No timing has been measured on a CPU; speed work is still on hold.

## Bugs found and fixed on the way (kept because they changed numbers)

- **Frozen core chosen by energy order.** A heavy atom's frozen semicore shell
  can lie above a light atom's active valence shell (Pb 5d above O 2s). The
  dimer and its fragments then froze different orbitals and two HEAVY28
  reactions were wrong by 39 and 51 kcal/mol. The core is now chosen by which
  element each orbital lives on (Mulliken population above one half). A recheck
  of all 1,053 structures processed before the fix found only those two
  affected; both were recomputed [measured].
- **No proper restricted state.** For `INV24/PCl3_TS`, restricted PBE has no
  aufbau solution (an occupied orbital above an empty one), so the response
  quantities were meaningless (largest eigenvalue 3e5). Rule now in the SCF
  pass: if a closed-shell structure does not converge or has a non-positive gap,
  use the spin-symmetry-broken unrestricted solution started from a 45-degree
  mix of the frontier pair. For this structure it is 0.050 Hartree lower, with
  gaps of 0.043 Hartree and S^2 = 0.99 [measured]. Superseded records are kept
  under `superseded/`.
- **Level-shifted SCF landing on the wrong state** (singlet C2, 0.1 Hartree too
  high). A second-order solver stage now runs before any level shift.

## Results

All numbers below are **partial coverage** and are weighted mean absolute
errors over the reactions available, in kcal/mol. They equal WTMAD-2 only at
1,505 of 1,505 reactions. No full-suite score exists yet.

Run card for the table: run `pbe_qzvp_g3` with passes `corr_v1` and `semi_v1`;
evidence level development (fit and validation both on GMTKN55). Energies are
driven by PBE orbitals from a GPU PBE/def2-QZVP SCF (density-fitted Coulomb,
grid level 3), then exact exchange, RPA response spectrum, shifted PT2, singles
and semilocal integrals as described above; evaluated after the SCF, not
self-consistently; no dispersion-specific term; no reference energy enters any
ingredient. Data: pinned GMTKN55 v1, 562 of 1,505 reactions (every reaction
whose structures have at most about 450 basis functions), 5 of 55 subsets
complete. Engine: gpu4pyscf 1.9.2 on one RTX 4060. No timing or cost claim.

| Model | Fitted parameters | In-sample | CV, random folds within subsets | CV, whole subset held out |
|---|---|---|---|---|
| PBE (input, nothing fitted) | 0 | 10.99 | n/a | n/a |
| Exact exchange only | 0 | 36.37 | n/a | n/a |
| Exact exchange + RPA (nothing fitted) | 0 | 8.52 | n/a | n/a |
| + fixed semilocal components, shifted PT2, singles | 27 | 1.58 | 1.81 | 3.57 |
| + learned spectral function (5 + 5 terms) | 36 | 1.29 | 1.52 | 2.40 |
| + regime-resolved semilocal integrals | 97 | 0.85 | 1.17 | 2.46 |

Reading, with its limits:

- The fitted models are far below PBE on these reactions, and the 97-parameter
  model is near 1.2 when validated on held-out reactions of the same types.
- Holding out a whole reaction type roughly doubles the error. Transfer to
  chemistry of an unseen kind is the weaker result and the more relevant one
  for a hidden test.
- These 562 reactions are the small-molecule ones. The heavily weighted
  noncovalent and conformer subsets are mostly not in yet, so this says little
  about the final WTMAD-2.
- A diagnostic fit on atomization energies alone reaches about 0.6 kcal/mol
  held-out error with the same ingredients, against about 1.9 under one global
  fit. The information is in the ingredients; one global set of coefficients is
  the compromise.

See `PROGRESS.md` for the dated log.
