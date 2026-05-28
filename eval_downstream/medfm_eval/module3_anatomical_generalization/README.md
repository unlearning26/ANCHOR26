# Module 3 Anatomical Generalization

Module 3 is the held-out-organ transfer stage of the benchmark.

Use the `scripts/` directory for the standard shell workflow. Use the Python entrypoints directly only for debugging or narrow smoke checks.

It asks one question: can a frozen representation transfer to an organ that is
held out during the evaluation fold?

The headline recoverability metrics are centered on three linked quantities:

1. `BA_centroid`: frozen nearest-centroid balanced accuracy.
2. `BA_probe`: balanced accuracy after lightweight probe adaptation.
3. `Delta_adapt`: adaptation gain over the frozen nearest-centroid baseline.


## Canonical presets

- `totalsegmenter_ct_mr_anchor_core`
- `mmwhs_ct_mr_core`
- `amos_ct_mr_core`
<!-- - `abdomenatlas_core` -->

Preset resolution is implemented in `scripts/module3_dataset_presets.sh`.

## Datasets

The current Module 3 datasets are:

1. `totalsegmenter_ct_mr_anchor`
Purpose: cross-modality anchor dataset.

2. `mmwhs_ct_mr`
Purpose: cross-modality validation dataset.

3. `amos_ct_mr`
Purpose: cross-modality abdominal validation dataset.

<!-- 4. `abdomenatlas` -->
<!-- Purpose: within-modality CT-only anatomical hold-out dataset. -->

Not included:

1. single-organ datasets, because they cannot support leave-one-organ-out evaluation.
2. datasets without a materialized organ-level manifest and aligned embedding bundles.

## What stays fixed

1. The encoder is frozen.
2. Global case partitions must be case-disjoint across the whole evaluation fold.
3. Held-out organ means held out during the Module 3 evaluation fold.
4. Metric reporting stays minimal and readable.

## Reproduction prerequisites

Module 3 expects:

1. A materialized upstream source manifest, typically the Phase 2 organ-level manifest at `MODULE3_SOURCE_MANIFEST`.
2. Existing feature bundles under `outputs_phase2/<analysis>/phase2/<variant>/features/<checkpoint>/<feature_type>/`.

If you run `eval_downstream/medfm_eval/cmd.sh`, the default Module 3 manifest is built automatically from `PHASE2_MANIFEST` and `PHASE2_ANALYSIS` unless you override `MODULE3_MANIFEST`.

## Entry points

Core scripts:

1. `scripts/module3_dataset_presets.sh`
2. `scripts/run_module3_build_manifest.sh`
3. `scripts/run_module3_anatomical_generalization.sh`

Python entrypoints:

1. `build_module3_manifest.py`
2. `module3_evaluation_pipeline.py`

## Scripts-first workflow

```bash
cd eval_downstream/medfm_eval/module3_anatomical_generalization

MODULE3_PRESET=amos_ct_mr_core bash ./scripts/run_module3_build_manifest.sh

MODULE3_PRESET=amos_ct_mr_core \
MODULE3_FEATURE_TYPES=cls \
MODULE3_CHECKPOINTS="Med3DINO_REL_c96 Med3DINO_SA_c96" \
bash ./scripts/run_module3_anatomical_generalization.sh
```

## Main outputs

1. `manifest_full.json`
2. `manifest_sampled.json`
3. `manifest_meta.json`
4. `outputs_module3/<analysis>/module3/<variant>/results/module3_anatomical_generalization_summary.json`
5. `outputs_module3/<analysis>/module3/<variant>/results/module3_anatomical_generalization_summary.csv`
6. `outputs_module3/<analysis>/module3/<variant>/results/<checkpoint>/<feature_type>/module3_anatomical_generalization.json`
