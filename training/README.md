# Training assets

Read [the complete training process](../docs/training.md) for requirements,
Colab/T4 and ROCm setup, data preparation, GPU smoke/full training, monitoring,
checkpoint continuation, evaluation and GGUF export.

Read [the model report](../docs/model-report.md) for the current bundled model,
measured comparisons, Pi observations and remaining limits.

All commands in the guide run from the repository root. `configs/` contains
versioned GPU recipes; `dataset/` contains all training/evaluation assets and
preserved sources; `notebooks/` contains the Colab workbook; `experiments/` holds
machine-readable evidence and experiment plans. Generated runs, base weights,
checkpoints and environments stay in ignored `artifacts/` and `.local/` paths.

Train off-device. Changing data requires a fresh base-model run and matching GPU
smoke. Never train on test/gold or use model output as equipment authorization.

The current fresh-run profiles are `configs/config-v9.json` (CUDA FP16/NF4) and
`configs/mi355x-v9-lora-bf16.json` (ROCm BF16). V9 adds 850 train-only conversations
to the 6,039-conversation v8 corpus; learning parameters and historical selection/
early-stopping rules stay fixed. Diagnostic retention changes to at most three
research adapters. Three epochs over more data increases the training budget.
See [the versioned experiment plan](experiments/v9-plan.json) and the
[operator commands](../docs/training.md#v9-operator-handoff-2026-10-23).

The Colab notebook defaults to fresh v9 with training disabled. Exact interrupted
resumes use the original Drive run folder, adding a unique attempt below it;
they require original source/config/data/dependencies. A completed early-stopped
run cannot resume, and current v9 data cannot continue a v8 checkpoint exactly.
