# Med3DINO and REFORM-3D

This repository contains the Med3DINO encoder runtime together with the three REFORM-3D benchmark modules used in the paper.

The benchmark runtime covers three modules:

1. Phase 1: acquisition robustness under spacing variation
2. Phase 2: cross-modal anatomical alignment
3. Module 3: anatomical generalization under held-out-organ evaluation

## Environment

Create a repo-local virtual environment and install the runtime:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The phase-specific shell launchers default to `python` from the current shell. The end-to-end launcher at `eval_downstream/medfm_eval/cmd.sh` defaults to the repo-local `.venv/bin/python`. If you prefer explicit interpreters, pass them directly:

Examples:

```bash
PYTHON="$PWD/.venv/bin/python" bash eval_downstream/medfm_eval/phase1_anisotropy_robustness/scripts/run_phase1_parallel_cls.sh
PYTHON="$PWD/.venv/bin/python" bash eval_downstream/medfm_eval/phase2_cross_modality_alignment/scripts/run_phase2_build_anchor_manifest.sh
PYTHON="$PWD/.venv/bin/python" bash eval_downstream/medfm_eval/module3_anatomical_generalization/scripts/run_module3_build_manifest.sh
PYTHON_BIN="$PWD/.venv/bin/python" CHECKPOINT_ROOT=/path/to/checkpoints PHASE1_MANIFEST=/path/to/phase1/manifest_sampled.json PHASE2_MANIFEST=/path/to/phase2/manifest_sampled.json bash eval_downstream/medfm_eval/cmd.sh
```

Dataset builders use repo-local defaults under `data/` and can be overridden
with environment variables when your datasets live elsewhere:

- `MED3DINO_RAW_DATA_ROOT`
- `MED3DINO_KITS23_RAW_ROOT`
- `MED3DINO_MMWHS_ROOT`
- `MED3DINO_AMOS_ROOT`

## Full Workflow Launcher

The end-to-end launcher at `eval_downstream/medfm_eval/cmd.sh` expects external
checkpoint and manifest paths from the environment. The shortest invocation is:

```bash
CHECKPOINT_ROOT=/path/to/checkpoints \
PHASE1_MANIFEST=/path/to/phase1/manifest_sampled.json \
PHASE2_MANIFEST=/path/to/phase2/manifest_sampled.json \
bash eval_downstream/medfm_eval/cmd.sh
```

Before running `cmd.sh`, either build or choose a Phase 1 manifest and a Phase 2
manifest. The phase-specific readmes below document the canonical manifest builders
and preset namespaces.

`cmd.sh` runs exactly one `CHECKPOINT_NAME` and one `FEATURE_TYPE` per invocation.
If you need a multi-checkpoint or multi-feature slate, wrap it in an outer shell loop.

`cmd.sh` also reuses `PHASE2_MANIFEST` when it builds the default Module 3 manifest,
so you do not need to materialize a separate Module 3 manifest ahead of time unless
you want to override `MODULE3_MANIFEST` explicitly.

`CHECKPOINT_ROOT` must be the directory that contains `chkpts/`, not the `chkpts/`
directory itself.

For repeated runs, you can place the same assignments in
`eval_downstream/medfm_eval/cmd.env`; the launcher sources it automatically when
that file exists. You can also point it to a different file:

```bash
CMD_ENV_FILE=/path/to/custom_cmd.env bash eval_downstream/medfm_eval/cmd.sh
```

## Workflow

1. Materialize or choose a Phase 1 manifest. See `eval_downstream/medfm_eval/phase1_anisotropy_robustness/readme.md`.
2. Materialize or choose a Phase 2 manifest. See `eval_downstream/medfm_eval/phase2_cross_modality_alignment/readme.md`.
3. Set `CHECKPOINT_ROOT`, `PHASE1_MANIFEST`, and `PHASE2_MANIFEST` inline or in `eval_downstream/medfm_eval/cmd.env`.
4. Run `bash eval_downstream/medfm_eval/cmd.sh` for one checkpoint and one feature family.
5. Repeat or loop externally for additional checkpoints or feature families.

## Main Shell Entry Points

Phase 1:

- `eval_downstream/medfm_eval/phase1_anisotropy_robustness/scripts/run_phase1_parallel_cls.sh`
- `eval_downstream/medfm_eval/phase1_anisotropy_robustness/scripts/run_perturbation_parallel_cls.sh`

Phase 2:

- `eval_downstream/medfm_eval/phase2_cross_modality_alignment/scripts/run_phase2_build_anchor_manifest.sh`
- `eval_downstream/medfm_eval/phase2_cross_modality_alignment/scripts/run_phase2_parallel_cls.sh`

Module 3:

- `eval_downstream/medfm_eval/module3_anatomical_generalization/scripts/run_module3_build_manifest.sh`
- `eval_downstream/medfm_eval/module3_anatomical_generalization/scripts/run_module3_anatomical_generalization.sh`

Additional phase-specific helper scripts are documented in the phase readmes and are not required for the standard workflow.

## Validation

Focused validation entrypoints:

```bash
.venv/bin/python -m pytest tests/test_phase1_anisotropy_smoke.py tests/test_phase2_evaluation_pipeline.py tests/test_phase2_single_organ_alignment.py tests/test_module3_case_disjoint_protocols.py
.venv/bin/python tests/test_module3_anatomical_generalization_smoke.py
```

