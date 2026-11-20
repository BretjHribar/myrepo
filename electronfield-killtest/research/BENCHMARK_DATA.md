# Pinned public accuracy benchmark

17 November 2026. Official GMTKN55 data is now available locally for
testing our own method. **No GMTKN55 model score or WTMAD-2 has been
calculated.** Source-data integrity is separate from model accuracy.

## Source and version

The source is the authors' [GMTKN55 repository](https://github.com/grimme-lab/GMTKN55),
pinned to commit
[`8d485b37a1ca8837e395042671ca5ba4e0714691`](https://github.com/grimme-lab/GMTKN55/tree/8d485b37a1ca8837e395042671ca5ba4e0714691)
on `v1`. Dataset attribution and the CC-BY-4.0 license are retained locally;
cite Goerigk, Hansen, Bauer, Ehrlich, Najibi and Grimme, PCCP 2017,
[DOI 10.1039/C7CP04913G](https://doi.org/10.1039/C7CP04913G).

The authors distinguish `v1`, which preserves the original benchmark
with its average correction, from `v2`, which updates references and
removes cases. Their README reports the corrected overall normalization
as 57.82 instead of 56.84. A later score must state its reference version
and weights. Do not silently substitute `v2`, remove difficult cases or
mix the two versions. The organizer's exact reference convention still
needs to be matched before claiming a challenge result.

The old university subset URLs timed out. The authors' repository
provided an accessible primary source. Only data and attribution files
were retained: structures, integer charge/spin information, reaction
definitions and dataset metadata. Published-method calculations/results,
evaluation programs and environments were excluded. No method was
installed, run or reproduced.

## Verified inventory

`gmtkn55_data_acquisition_20261117.json` records the bounded acquisition.
The manifest is `.research/data/gmtkn55_official_v1/manifest.json`.
`gmtkn55_data_audit_20261117.json` independently rechecks every retained
file against its pinned Git blob hash and SHA-256.

- 6,368 verified files and 2,462 structure records.
- All 55 evaluation sets, including BH76RC sharing the BH76 structures.
- Charges: 128 anions, 2,135 neutrals, 189 cations and 10 dications.
- Reported unpaired-electron counts: 2,142 zero, 260 one, 51 two and 9 three.
- Physical electron counts range from 1 to 360; structures reach 81 atoms.
- Every structure has resolved element tokens and consistent integer
  charge/spin parity metadata. No difficult cases were filtered out.

This is an input inventory. It does not validate an open-shell solver,
coupled 360-electron accuracy or a transferable functional. The reported
spin metadata must not be confused with a separate proof of a physical
many-electron spin state.

## Acquisition scope and later numerical gates

Coordinate strings and reference-energy values were not numerically
evaluated during acquisition/auditing. No CPU scientific calculation ran.
The data/metadata work stayed on one host core. Stored data and source-tree
metadata are about 9.3 MB; complete project usage is about 21.2 GB of the
unchanged 40 GB allocation.

The later full CUDA coordinate check now passed for all 2,462 structures.
The complete data-only reaction parser and CUDA scorer passed for all
55 sets / 1,505 reactions. They use 2,442 required structure IDs, preserving
all reactions; 20 unreferenced upstream structures are listed explicitly.
The corrected pinned calculator uses 57.8167 and the pinned subset means,
including separately verified BH76RC mean/count metadata. All source
reference definitions remain unchanged bytes. See
[the full pipeline and coverage assessment](GMTKN55_PIPELINE.md).

Actual model predictions remain missing. The complete target-basis domain
inventory identifies 2,421 structure resource/domain blockers and 21
uncomputed structures. No full-suite kcal/mol estimate or pass exists.
The separately locked confirmation panel was not opened. A filtered
subset, a small-basis stretch contrast, a reference solve or a successful
data audit cannot establish sub-1 kcal/mol WTMAD-2.

The interrupted BH/cc-pVDZ experiment retains its original settings and its
independent-reference ordering. No public benchmark labels are passed
into that predictor. Its live/terminal report, rather than this data
milestone, determines that experiment's outcome. Speed work stays disabled.
