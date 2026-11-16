# Electron-field supervision: a reproducible 1D XC kill test

**Status: exploratory toy-model experiment, not a chemical accuracy result.**

Tests a tightly controlled hypothesis: does adding *electron-density self-consistency* and *exchange-correlation (XC) potential* supervision help a small learned XC functional generalize beyond systems on which it was trained?

## Results (five paired seeds)

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

## Next experiments

1. **Ablation:** energy-only vs energy+potential vs energy+density vs both, all with independent held-out chemistry.
2. **Through-SCF training:** implicit differentiation or converged-SCF gradients, not only a one-step density residual.
3. **Fractional-charge dissociation tests:** force the functional to respect exact charge-transfer behavior and examine KS potential steps.
4. **3D translation:** a real small molecule set with finite-basis inversion/reference densities and matched computational budgets before claiming competition relevance.

The kill-test criterion should be **reliable, substantial held-out gains at the same SCF success rate and runtime**, not merely lower training error.
