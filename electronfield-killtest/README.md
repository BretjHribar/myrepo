# Electron-field supervision: a reproducible 1D XC kill test

**Project objective:** win Discovered Materials' [Breaking the Speed–Accuracy Barrier in Quantum Chemistry](https://discoveredmaterials.com/grand-challenges/speed-accuracy-barrier/) challenge through a substantial advance in accuracy, efficiency, and generalization. The [research plan](RESEARCH_PLAN.md) defines the proposed approach, hardware budget, and experiments that must succeed. This folder's 1D results are an early diagnostic; they do not establish competition performance.

**Research direction:** our own research only, with no frontier replication, including sanity checks or benchmark reproductions. The 1D commands below preserve the historical experiment; they are not current work items. Current 3D findings and rejected variants are in [the execution record](research/PROGRESS.md).

**Current 3D work:** [current model and evidence](research/CURRENT_MODEL.md). A new [charge/spin block state](research/CHARGE_SPIN_STATE.md) resolves the earlier restrictions on small same-basis tests, including a new ion, 18-electron peroxide and larger-basis BeH2 geometries. [Spatial density constraints](research/SPATIAL_DENSITY_FUNCTIONAL.md) now have on-shell scalar/potential checks. The present implementation still retains large state spaces and does not establish a general XC functional, complete-basis accuracy or WTMAD-2. The [accuracy gate](research/ACCURACY_GATE.md) takes priority; further speed work is on hold.

The bounded hard-density inverse and both feasible-ensemble searches failed
and are closed. A changed [auxiliary-density formulation](research/AUXILIARY_DENSITY_STATE.md)
passes two finite-basis forward development points and fresh XC derivative
checks. It still uses global Fock vectors, supports a restricted spin/particle
domain and establishes no complete-basis, learned-transfer or WTMAD-2 result.
Its regularized KS integration and challenge eligibility are unverified.
Larger realistic systems with the physical and scalar/derivative checks
intact remain the next accuracy requirement. Speed stays disabled.

The new [tensor state/action and density interface](research/TENSOR_DENSITY_RESPONSE.md)
passes physical scope checks on 24-orbital water without a global Fock
vector. A [conditional tensor XC search](research/CONDITIONAL_TENSOR_FUNCTIONAL.md)
also passes physical installation and scalar/derivative checks. Surrounding
tensors remain fixed, global self-consistency is unproved, and reference
accuracy/WTMAD-2 were not measured.

The new [sector-completion work](research/TENSOR_SECTOR_COMPLETION.md)
closes a fixed-seed restriction excluding higher excitations. A
state-preserving water extension and independent variable-sector forward
checks pass. A frozen larger-basis BH stretch test is checkpointed and
interrupted, preserving prediction-before-reference ordering. No molecular
accuracy result is inferred from partial progress.

**Latest user priority:** full public GMTKN55 tests and an initial measured
assessment first. The [full public parser and GPU scorer](research/GMTKN55_PIPELINE.md)
pass all 55 sets / 1,505 reactions. The current model fails full-scope
feasibility. Its exact N=1 component has 11 accepted target-basis structure
predictions, covering only five reactions; no full-suite WTMAD-2 estimate
exists. A changed [physical operator](research/DENSITY_SQUARE_OPERATOR.md)
passes implementation checks, but broad accuracy remains unmeasured.

**Status: exploratory toy-model experiment, not a chemical accuracy result.**

**Current reading:** in this 1D toy, field supervision improves held-out energies, densities and potentials once the fields are actually learned, at the cost of harder SCF convergence. It does not fix stretched-bond static correlation.

Tests a tightly controlled hypothesis: does adding *electron-density self-consistency* and *exchange-correlation (XC) potential* supervision help a small learned XC functional generalize beyond systems on which it was trained?

> **Update, 2026-11-16.** The original five-seed table further down is not a valid test of the hypothesis: the field-supervised arm never learned the fields. The rerun in the next section fixes that. Read the rerun first.

## Rerun: field supervision with the fields actually learned

Scripts, raw outputs and logs are in [`rerun/`](rerun/). They import `prototype.py` unchanged.

### What was wrong with the original run

- **The field weight was too low.** With weights of 0.2 the logged potential loss is 0.086 at step 0 and 0.087 at step 129. It stays flat even with 1,000 training steps (two seeds, `rerun/out/b_F02_*.json`).
- **130 steps is too few.** The energy loss was still swinging between about 0.02 and 0.7 when training stopped, so the reported energy gaps are snapshots of an unconverged optimizer.
- **Neither arm had learned the potential.** A functional with zero correlation scores a long-bond potential RMSE of 0.1045 Ha (`rerun/logs/baselines.jsonl`). The original arms scored 0.1046 and 0.1009.
- **The architecture was not the limit.** Trained on the potential alone for 4,000 steps, the same 769-parameter functional reaches an in-range potential RMSE of 0.004–0.007 Ha, against 0.032 in the original runs (`rerun/out/cap_*.json`).

### Rerun settings

Same data, seeds, architecture and losses as the original. Changes: 1,000 Adam steps with cosine learning-rate decay (0.003 down to 2% of that), and field weights of 5 instead of 0.2 for the field arm. The weight of 5 was a single untuned choice. Long-bond energies were evaluated with a more careful SCF (3,000 iterations, mixing 0.03) because the default evaluation SCF did not always converge there.

### Results (five paired seeds)

| Held-out metric | Energy only | Energy + fields | Change | Field wins |
|---|---:|---:|---:|---:|
| In-range energy MAE | 1.73 kcal | 1.23 kcal | −29% | 4/5 (p = 0.17) |
| In-range density L1 error | 0.036 | 0.015 | −59% | 5/5 |
| In-range potential RMSE | 0.032 Ha | 0.013 Ha | −59% | 5/5 |
| Long-bond energy MAE (careful SCF) | 21.4 kcal | 16.9 kcal | −21% | 5/5 (p < 0.001) |
| Long-bond potential RMSE | 0.093 Ha | 0.060 Ha | −36% | 5/5 |

Long-bond energy MAE by seed, in Ha (energy only → fields): 0.0139 → 0.0061, 0.0325 → 0.0251, 0.0666 → 0.0621, 0.0296 → 0.0210, 0.0282 → 0.0202. P-values are paired t-tests over five seeds and should be read loosely.

**Reading:** once the fields are learned, field supervision improves every held-out metric in this toy. The in-range energy gain is not statistically solid. The long-bond energy gain is consistent, about 4.6 kcal in every seed.

### Costs and limits found in the rerun

- **SCF stability got worse.** At the original evaluation settings (250 iterations, mixing 0.15) the field-trained functional failed to converge on 9 long-bond cases, against 2 for energy-only. Scored that way it looks 90% worse on long-bond energy. All cases converge with the careful SCF. In a timed benchmark this is extra CPU time.
- **Stretched near-symmetric bonds are still 25–70 kcal off in both arms.** Fields reduce that by roughly 5–15%. This is static correlation and a loss change does not fix it (`rerun/logs/longbond_careful_scf.log`).
- **Fields improve densities and potentials about 2.5×, energies about 1.3×.**
- **The low-data run (4 training systems) was unstable.** One seed's in-range energy error was 40× worse with fields, and 13 long-bond cases did not converge at default SCF settings. No conclusion is drawn (`rerun/out/c_*.json`).
- **No ablation yet.** The density and potential terms were not separated, and the weight was not swept.
- Repeated runs of the same seed differ by about 1–2% on long-bond energy, so small differences are noise.

### Reproduce the rerun

```bash
cd rerun
# one run: <arm> <seed> <steps> <cosine 0/1> <potential weight> <density weight> <n_train, 0 = all 28> <output>
python3 diag.py .. run energy_only 20261116 1000 1 0 0 0 out/a_E_20261116.json
python3 diag.py .. run fields      20261116 1000 1 5 5 0 out/a_F5_20261116.json
# the full job list used here is logs/jobs.txt
python3 diag.py .. baseline                      # zero-correlation and untrained baselines
python3 diag.py .. run potential_only 20261116 4000 1 1 0 0 out/cap_20261116.json   # capacity test
python3 diag2.py .. 20261116 20261117 20261118 20261119 20261120                    # careful-SCF long-bond check
python3 analyze.py                               # paired-seed summary of out/
```

Run on Python 3 with PyTorch 2.15.0, NumPy 2.5.3, SciPy 1.18.1, single thread. Each 1,000-step run takes about two minutes on one CPU core.

## Original results (five paired seeds, superseded)

These numbers are kept for the record. They compare two arms that were both effectively energy-only; see the rerun above.

All errors are evaluated after a fresh self-consistent Kohn–Sham calculation. Lower is better.

| Held-out metric | Energy-only | Energy + density + potential | Relative improvement | Field wins |
|---|---:|---:|---:|---:|
| In-range total energy MAE (Ha) | 0.01030 | 0.00915 | 11.1% | 3/5 |
| Long-bond total energy MAE (Ha) | 0.04366 | 0.04168 | 4.5% | 4/5 |
| Long-bond density L1 error (fraction of 2 electrons) | 0.19871 | 0.18992 | 4.4% | 5/5 |
| Long-bond density-weighted potential RMSE (Ha) | 0.10459 | 0.10090 | 3.5% | 5/5 |

There is **no conclusive scientific breakthrough** in this data. Errors are substantial on stretched bonds and variation between random seeds is large. The potential results in the *in-range* test are virtually tied (0.03171 vs 0.03173 Ha). Five random splits/initializations do not provide a reliable statistical significance estimate.

Full per-run values: [`summary.csv`](summary.csv), [`aggregate.json`](aggregate.json), and individual `results*.json` files. The two trained arms are paired using the **same initialization and molecules for each seed**.

## Physical benchmark

We use two opposite-spin electrons in one spatial dimension, with softened Coulomb interactions. Exact ground states come from sparse diagonalization of the discretized two-body Hamiltonian:

\[
H = -\tfrac12\partial_{x_1}^2-\tfrac12\partial_{x_2}^2 +v_{\rm ext}(x_1)+v_{\rm ext}(x_2)+\frac{1}{\sqrt{(x_1-x_2)^2+1}},
\]

where `v_ext` is the sum of two softened attractive nuclear wells with charges `z1 + z2 = 2`, and separation `R`.

- 35 grid points at `dx = 0.45` in atomic units, centered at zero.
- 28 training molecules with `0.9 <= R <= 3.1`.
- 8 independent in-range test molecules with the same `R` interval.
- 8 out-of-range bond-stretch test molecules with `3.2 <= R <= 4.6`.
- Each seed independently resamples molecules and initializes the neural network; five seeds `20261116`–`20261120`.

Reference energy and density are from the exact finite-grid two-electron wavefunction. For this two-electron spin singlet, the exact KS orbital is `sqrt(n/2)` and the **KS correlation potential** is obtained by algebraically inverting the one-orbital Schrödinger equation, then subtracting the external and exchange-plus-Hartree potentials. The additive potential constant is removed by matching density-weighted means.

Thus the fields are derived from a quantum model, *not synthetic regression targets*. They are **not** CCSD(T) molecular labels or a 3D benchmark.

## Student functional

For a two-electron spin singlet, exchange exactly cancels half the Hartree energy. We fit the remaining correlation:

\[
E_{xc}[n]=-\frac12 E_H[n]+E_{c,\theta}[n],\qquad
E_{c,\theta}[n]=\int n(x) f_\theta(n(x),\bar n_{\sigma_1}(x),\bar n_{\sigma_2}(x),v_H(x),\dots) dx.
\]

`f_theta` is a 769-parameter, two-hidden-layer tanh MLP with nonlocal smoothed density features. We compute `v_c = δE_c/δn` by automatic differentiation; the functional therefore has a differentiable, energy-consistent potential.

Both arms optimize an energy loss with targets evaluated from the **same self-consistent Kohn–Sham procedure**. We detach the SCF state during gradient calculation, exploiting the variational identity at convergence (rather than backpropagating through every iteration). With 24 training SCF iterations, exact stationarity is approximate.

The field-supervised arm additionally uses:

1. Density-weighted squared correlation-potential error, evaluated at the **exact reference density**, after removing gauge differences.
2. Density self-consistency residual: construct the KS Hamiltonian at the exact density using the learned potential; differentiate through its diagonalization and penalize the difference between the resulting occupied-orbital density and the exact density.

Both field losses differentiate through the neural functional's derivative; the second differentiates through one self-consistency update. **This is not full end-to-end differentiation through an entire converged SCF trajectory.** That is a meaningful next experiment.

Optimization: Adam, `lr=0.003`, 130 steps, initial matched weights, energy MSE scaled by `0.04 Ha`, field loss MSE scales `0.12`, and field weights `0.2` each. No held-out data was used for model selection. Evaluate with 250 damped SCF iterations (`mix=0.15`); all evaluated cases had residual below `1e-4` in the reported runs.

## Reproduce

Python 3.11+ recommended; on the run box: Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, PyTorch 2.10.0+cpu. CPU-only.

```bash
python -m pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest -v test_prototype.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python prototype.py --seed 20261116 --epochs 130 --output results.json
```

For all five runs, change `--seed` to 20261117 through 20261120, with `--output results_seed<SEED>.json`, then:

```bash
python aggregate.py
```

The experiment takes tens of seconds per seed on a single virtual CPU in the provided chat environment. Hardware-dependent performance may vary.

## Important limitations

- 1D soft-Coulomb diatomics only; **no real atoms, metal complexes, reaction barriers, or CCSD(T) data**.
- The test measures **total-energy MAE**, not reaction-energy differences or GMTKN55 WTMAD-2.
- No fractional-charge, fractional-spin, size-consistency, or heterogeneous chemistry tests.
- The target electron densities and XC potentials are mathematically linked; grid points are **not independent examples**.
- The architecture is deliberately small, and its smooth nonlocal descriptors may be unable to represent exact long-bond correlation steps/peaks.
- The field-supervised arm uses two additional loss terms, so comparisons establish the combined effect, not which term is responsible.
- Model selection, likelihood calibration, and uncertainty quantification are not yet validated at larger scale.

## Next steps

Updated after the rerun, in order.

1. **Convergence-checked SCF.** Use it in both training and evaluation (adaptive damping or DIIS, with a residual threshold), and report the number of unconverged cases and iterations as metrics alongside accuracy. Training currently runs 24 fixed iterations with no check, so energy gradients can be taken at unconverged densities.
2. **Weight sweep and ablation.** Sweep the field weight (for example 0.2, 1, 5, 25) on a separate validation split, and compare energy-only, energy + potential, energy + density, and both.
3. **Low-data rerun.** Repeat the 4-training-system test after step 1, then trace a curve over 2, 4, 8, 16 and 28 systems. The claim to test is the same error with several times less data.
4. **Score energy differences.** Report errors on energy differences along each bond curve, not only total energies, because WTMAD-2 is built on relative energies.
5. **Attack the stretched-bond error.** Add fractional-charge and fractional-spin constraints and longer-range descriptors, and examine the KS potential steps at dissociation.
6. **Composition shift.** Train on H₂-like systems and test on HeH⁺-like and three-centre (H₃⁺-like) systems. Both are still two-electron, so the exact solver works unchanged.
7. **Through-SCF training.** Implicit differentiation or converged-SCF gradients, not only a one-step density residual.
8. **3D translation.** A real small-molecule set with finite-basis inversion, reference densities and matched computational budgets before claiming competition relevance.

The kill-test criterion should be **reliable, substantial held-out gains at the same SCF success rate and runtime**, not merely lower training error.
