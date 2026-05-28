# Phase 1 Launchers

Use the `scripts/` directory for the standard shell workflow.
Use the root-level Python files directly only for manifest generation, cache generation, or narrow debugging.

## Core Scripts

- `scripts/phase1_dataset_presets.sh`
- `scripts/run_phase1_parallel_cls.sh`
- `scripts/run_phase1_parallel_avg_pool.sh`
- `scripts/run_phase1_parallel_multilayer.sh`
- `scripts/run_phase1_all_features.sh`
- `scripts/run_perturbation_parallel_cls.sh`
- `scripts/run_perturbation_parallel_avg_pool.sh`
- `scripts/run_perturbation_parallel_multilayer.sh`
- `scripts/run_perturbation_all_features.sh`

Optional bulk helpers at the phase root:

- `manifest_generator.sh`: regenerate the current manifest slate in one pass
- `cache_generator.sh`: precompute controlled-perturbation caches in one pass

The launcher scripts use `PHASE1_PRESET` as the canonical entry point.

If you do not set `PHASE1_PRESET`, the scripts default to `abdomenatlas_core`.

If you need a custom namespace, set both `ANALYSIS_NAME` and `MANIFEST`. Partial overrides are rejected.

Manual manifest variants are normalized to the canonical directory names used by the active workflow:
- `original_bins`
<!-- - `coarse_bins` -->

For manual Setting B runs, the cache namespace is `ANALYSIS_NAME`, so cache precompute must use the same override:

```bash
python build_perturbation_cache.py -m "$MANIFEST" -a "$ANALYSIS_NAME" --crop-size 96 112
```

# What Setting A and Setting B mean

Phase 1 has two complementary evaluation settings:

- Setting A is the observational native-spacing analysis. It uses the scans as they were acquired, groups them into anisotropy bins from the manifest, and compares representations across those bins. The Setting A launchers are the `run_phase1_*` scripts.
- Setting B is the controlled spacing-perturbation analysis. It starts from a source subset in the manifest, resamples the same scan to multiple target spacing variants, and measures how much the representation changes under that intervention. The Setting B launchers are the `run_perturbation_*` scripts.

In short: Setting A asks whether spacing differences observed across the dataset are associated with changes in the embedding, while Setting B asks whether changing spacing alone changes the embedding when scan identity is fixed.

# Canonical manifest layout

All active Phase 1 manifests live under:

```text
../data_manifests/phase1_anisotropy_robustness/<dataset>/<variant>/
```

Each dataset/variant pair uses:
- `manifest_full.json`
- `manifest_sampled.json`
- `manifest_meta.json`

The active launcher presets point to `manifest_sampled.json` under either `original_bins` or `coarse_bins`.

# Manifest materialization

This repo does not guarantee that every Phase 1 manifest is already materialized under `../data_manifests/phase1_anisotropy_robustness/`.
Build missing manifests as needed:

```bash
cd eval_downstream/medfm_eval/phase1_anisotropy_robustness

python build_phase1_manifest.py --dataset totalsegmentermri --output-suffix totalsegmentermri --binning-scheme original
python build_phase1_manifest.py --dataset abdomenct1k --output-suffix abdomenct1k --binning-scheme coarse_bins
```

# Presets

Setting A (observational) and Setting B (controlled):
- `abdomenatlas_core`
- `jhu_stroke_core`
- `kits23_core`
- `totalsegmentermri_core`

Setting B only:
- `imagecas_core`
- `totalsegmenter_ct_core`

| Preset | Manifest variant | Observational layer | Controlled layer |
| --- | --- | --- | --- |
| `abdomenatlas_core` | `original_bins` | yes | yes |
| `jhu_stroke_core` | `original_bins` | yes | yes |
| `kits23_core` | `original_bins` | yes | yes |
| `totalsegmentermri_core` | `original_bins` | yes | yes |
| `totalsegmenter_ct_core` | `original_bins` | no | yes |
| `imagecas_core` | `original_bins` | no | yes |

Preset resolution is implemented in `scripts/phase1_dataset_presets.sh`.

# Setting A

Setting A is the observational part of Phase 1. It does not resample scans. It evaluates frozen features across the native spacing bins already present in the manifest and reports the observational geometry, spacing readout, semantic probing, and cross-bin transfer metrics.

Run all checkpoints for one feature family:

```bash
cd eval_downstream/medfm_eval/phase1_anisotropy_robustness/scripts

PHASE1_PRESET=abdomenatlas_core ./run_phase1_parallel_cls.sh
PHASE1_PRESET=abdomenatlas_core ./run_phase1_parallel_avg_pool.sh
PHASE1_PRESET=abdomenatlas_core ./run_phase1_parallel_multilayer.sh
PHASE1_PRESET=totalsegmentermri_core ./run_phase1_parallel_cls.sh
```

Run all feature families sequentially:

```bash
PHASE1_PRESET=abdomenatlas_core ./run_phase1_all_features.sh
PHASE1_PRESET=abdomenatlas_coarse ./run_phase1_all_features.sh
```

Setting A launchers reject controlled-only presets.

# Setting B

Setting B is the controlled perturbation part of Phase 1. It uses cached resampled variants of the same source scans so spacing can be changed while the underlying scan stays fixed. This is why Setting B requires `build_perturbation_cache.py` before the launcher scripts.

Cache generation must use the same manifest namespace and variant as the launcher.

Example: AbdomenAtlas original bins

```bash
cd eval_downstream/medfm_eval/phase1_anisotropy_robustness

python build_perturbation_cache.py \
    -m ../data_manifests/phase1_anisotropy_robustness/abdomenatlas/original_bins/manifest_sampled.json \
    --crop-size 96 112

cd scripts
PHASE1_PRESET=abdomenatlas_core ./run_perturbation_parallel_cls.sh
PHASE1_PRESET=abdomenatlas_core ./run_perturbation_parallel_avg_pool.sh
PHASE1_PRESET=abdomenatlas_core ./run_perturbation_parallel_multilayer.sh
```

Example: TotalSegmenterMRI original bins

```bash
cd eval_downstream/medfm_eval/phase1_anisotropy_robustness

python build_perturbation_cache.py \
    -m ../data_manifests/phase1_anisotropy_robustness/totalsegmentermri/original_bins/manifest_sampled.json \
    --crop-size 96 112

cd scripts
PHASE1_PRESET=totalsegmentermri_core ./run_perturbation_parallel_cls.sh
```

Example: TotalSegmentator CT controlled-only module

```bash
cd eval_downstream/medfm_eval/phase1_anisotropy_robustness

python build_perturbation_cache.py \
    -m ../data_manifests/phase1_anisotropy_robustness/totalsegmenter_ct/original_bins/manifest_sampled.json \
    --crop-size 96 112

cd scripts
PHASE1_PRESET=totalsegmenter_ct_core ./run_perturbation_parallel_cls.sh
```

Run all perturbation feature families sequentially:

```bash
PHASE1_PRESET=abdomenatlas_core ./run_perturbation_all_features.sh
```

What Setting B runs:
- resamples source-bin volumes to protocol-specific target spacing variants
- measures representation drift under matched spacing perturbations
- writes dataset- and variant-specific caches under `../caches/<dataset>/phase1/<variant>/`
- skips crop112 checkpoints when the matching crop112 cache tree is absent
