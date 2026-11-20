# Fast, generalizable exchange-correlation functionals

**Objective: win Discovered Materials' [Breaking the Speed–Accuracy Barrier in Quantum Chemistry](https://discoveredmaterials.com/grand-challenges/speed-accuracy-barrier/) challenge.**

The research target is a substantial advance in accuracy, computational cost, and transfer to unseen chemistry. Small improvements to the existing toy experiment do not satisfy that objective.

- [Research strategy and decision gates](electronfield-killtest/RESEARCH_PLAN.md)
- [Own-research execution and current findings](electronfield-killtest/research/PROGRESS.md)
- [Requirements for a physically valid transferable model](electronfield-killtest/research/MODEL_REQUIREMENTS.md)
- [Existing 1D experiment and its limitations](electronfield-killtest/README.md)
- [Local hardware and CUDA readiness](electronfield-killtest/research/hardware_2026-11-16.json)
- [Dated challenge snapshot](electronfield-killtest/research/challenge_2026-11-16.json)
- [Pinned complete public benchmark data and remaining scoring gates](electronfield-killtest/research/BENCHMARK_DATA.md)

Current status: **no candidate yet meets the accuracy/generalization goal**. The new [charge/spin block state](electronfield-killtest/research/CHARGE_SPIN_STATE.md) repairs the earlier restrictions and closely matches same-basis references on the small panel, new ionic/peroxide cases and larger-basis BeH2 geometries. [Spatial density constraints](electronfield-killtest/research/SPATIAL_DENSITY_FUNCTIONAL.md) now have on-shell scalar/potential checks. Some retained spaces are nearly full; this is not yet a general XC functional or a WTMAD-2 result. The [current evidence](electronfield-killtest/research/CURRENT_MODEL.md), [accuracy gate](electronfield-killtest/research/ACCURACY_GATE.md), locked confirmation panel and challenge objective remain intact. Speed work remains on hold.

The hard-density inverse and two feasible-ensemble searches failed and are
closed. A changed [auxiliary-density formulation](electronfield-killtest/research/AUXILIARY_DENSITY_STATE.md)
now passes two finite-basis forward development points and XC derivative
checks. It still uses global many-electron vectors, and establishes no
complete-basis, learned-transfer or WTMAD-2 result. Its regularized KS
implementation and challenge eligibility need further work. Speed remains
disabled; the next accuracy requirement is support for realistic systems
with physical and functional checks intact.

The new [tensor state/action and density interface](electronfield-killtest/research/TENSOR_DENSITY_RESPONSE.md)
now passes physical scope checks on 24-orbital water without a global Fock
vector. A [conditional tensor XC search](electronfield-killtest/research/CONDITIONAL_TENSOR_FUNCTIONAL.md)
also passes physical installation and scalar/derivative checks. The
surrounding tensors remain fixed: global self-consistency and independent
chemical accuracy are unproved, and no WTMAD-2 score was measured.

[Changing physical charge/spin sectors](electronfield-killtest/research/TENSOR_SECTOR_COMPLETION.md)
now repairs a restriction that excluded triple and higher excitations.
The water sector test and independent new forward implementation checks
pass; a frozen larger-basis BH stretch test is running before independent
reference scoring. No partial run establishes the accuracy goal.

Current priorities: **a valid model with convincing out-of-sample accuracy first; speed afterward**. **GPU-only numerical work**, at most one CPU core for host orchestration/I/O, and **40 GB total disk**. **Own research only; no frontier replication or reproduction runs.** See [operating constraints](AGENTS.md).
