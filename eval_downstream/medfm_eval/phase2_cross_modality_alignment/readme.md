# Phase 2 Cross-Modality Alignment

This directory contains the Phase 2 CT-MR anatomical alignment pipeline.

Use the `scripts/` directory for the standard shell workflow. Use the Python
entrypoints directly only for debugging or narrow smoke checks.

## Presets and namespaces

Canonical multi-organ cross-modal presets:

- `totalsegmenter_ct_mr_anchor_core`
- `mmwhs_ct_mr_core`
- `amos_ct_mr_core`

Supported specialized presets:

- `chaos_ct_mr_core`: single-organ corroboration through paired-case retrieval
- `abdomenatlas_core`
- `abdomenatlas_subset_1k`

`abdomenatlas_*` presets are CT-only helper presets and set `PHASE2_SKIP_EVALUATION=1` by default; they do not produce the standard cross-modal headline metrics.

Preset resolution is implemented in `scripts/phase2_dataset_presets.sh`.

## Manifest materialization

The repo does not guarantee that every Phase 2 manifest is already materialized under `../data_manifests/phase2_cross_modality_alignment/`.
Build the supported manifests with the dataset-specific shell builders:

```bash
cd eval_downstream/medfm_eval/phase2_cross_modality_alignment

bash ./scripts/run_phase2_build_anchor_manifest.sh
bash ./scripts/run_phase2_build_amos_manifest.sh
bash ./scripts/run_phase2_build_mmwhs_manifest.sh
bash ./scripts/run_phase2_build_abdomenatlas_manifest.sh
```

For `chaos_ct_mr`, provide a materialized manifest path and either set `PHASE2_PRESET=chaos_ct_mr_core` or manual namespace overrides. There is no dedicated `chaos` manifest-builder script in this repo.

## Headline metric contract

Phase 2 headline evidence uses:

- balanced bidirectional cross-modal organ retrieval with `top@1`, `top@5`, and `mAP`
<!-- - balanced LISI with modality iLISI and organ cLISI - anatomy-over-modality margin as the required diagnostic -->

Single-organ corroboration presets such as `chaos_ct_mr` reuse the same
`top@1`/`top@5`/`mAP` headline contract through paired-case bidirectional
retrieval, because a multi-organ balanced candidate pool is not available.

## Authoritative outputs

Root-level results under `outputs_phase2/<analysis>/phase2/<variant>/results/`
are authoritative for:

- `phase2_manifest_summary.json`
- `phase2_cohort_summary.json`
- `phase2_checkpoint_comparison.csv`
- `phase2_checkpoint_comparison.json`
- `phase2_checkpoint_comparison_avg_pool.csv`
- `phase2_checkpoint_comparison_avg_pool.json`
- `phase2_checkpoint_comparison_multilayer.csv`
- `phase2_checkpoint_comparison_multilayer.json`

Per-checkpoint primary metric files live under:

- `outputs_phase2/<analysis>/phase2/<variant>/results/<checkpoint>/phase2_primary_metrics.json`
- `outputs_phase2/<analysis>/phase2/<variant>/results/<checkpoint>/avg_pool/phase2_primary_metrics.json`
- `outputs_phase2/<analysis>/phase2/<variant>/results/<checkpoint>/multilayer/phase2_primary_metrics.json`

Feature bundles live under:

- `outputs_phase2/<analysis>/phase2/<variant>/features/<checkpoint>/cls/phase2_organ_cls_embeddings.npz`
- `outputs_phase2/<analysis>/phase2/<variant>/features/<checkpoint>/avg_pool/phase2_organ_avg_pool_embeddings.npz`
- `outputs_phase2/<analysis>/phase2/<variant>/features/<checkpoint>/multilayer/phase2_organ_multilayer_embeddings.npz`

## Workflow

Run these from the Phase 2 root as `bash ./scripts/<name>.sh`.

Core scripts:

1. `run_phase2_build_anchor_manifest.sh`
2. `run_phase2_build_amos_manifest.sh`
3. `run_phase2_build_mmwhs_manifest.sh`
4. `run_phase2_build_abdomenatlas_manifest.sh`
5. `run_phase2_manifest_audit.sh`
6. `run_phase2_build_crop_cache.sh`
7. `run_phase2_extract_and_eval.sh`
8. `run_phase2_parallel_cls.sh`
9. `run_phase2_parallel_avg_pool.sh`
10. `run_phase2_parallel_multilayer.sh`
11. `run_phase2_aggregate_results.sh`

Typical order:

1. Build the manifest for the dataset namespace you need.
2. Audit the manifest.
3. Build the crop cache for the crop sizes implied by the selected checkpoints.
4. Run `run_phase2_extract_and_eval.sh` for one checkpoint or a `run_phase2_parallel_*` launcher for a checkpoint slate.
5. Refresh aggregated checkpoint-comparison artifacts.

## Optional specialized scripts

These scripts are optional and are not required for the standard multi-organ Phase 2 workflow:

- `run_phase2_single_organ_alignment.sh`: additive patient-id-matched single-organ sidecar metrics
- `run_phase2_parallel_single_organ.sh`
- `run_phase2_parallel_single_organ_avg_pool.sh`
- `run_phase2_parallel_single_organ_multilayer.sh`
- `run_phase2_full_checkpoint_validation_mmwhs.sh`: convenience wrapper around the MMWHS CLS checkpoint slate
- `run_phase2_mmwhs_context_ablation.sh`: context-ablation helper for the MMWHS validation preset

## Minimal manifest contract

Supported manifest formats:

- a top-level list of samples
- or a dict with a `samples` key

Minimum useful sample fields:

```json
{
  "sample_id": "sample_001",
  "file_path": "/abs/path/to/volume.nii.gz",
  "mask_path": "/abs/path/to/organ_mask.nii.gz",
  "modality": "ct",
  "primary_organ": "liver",
  "organs": ["liver"],
  "patient_id": "patient_01"
}
```

If no subsampling is needed, `manifest_sampled.json` should be an identical
copy of `manifest_full.json`, and `manifest_meta.json` should say so explicitly.