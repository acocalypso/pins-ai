# Training assets

Read [the complete training process](../docs/training.md) for requirements,
Colab/T4 and ROCm setup, data preparation, GPU smoke/full training, monitoring,
checkpoint continuation, evaluation and GGUF export.

Read [the model report](../docs/model-report.md) for the current bundled model,
measured comparisons, Pi observations and remaining limits.

All commands in the guide run from the repository root. `configs/` contains
versioned GPU recipes; `datasets/` and `data/` contain the training/evaluation
assets; `notebooks/` contains the generated Colab workbook; `experiments/` holds
machine-readable evidence and experiment plans. Generated runs, base weights,
checkpoints and environments stay in ignored `artifacts/` and `.local/` paths.

Train off-device. Changing data requires a fresh base-model run and matching GPU
smoke. Never train on test/gold or use model output as equipment authorization.
