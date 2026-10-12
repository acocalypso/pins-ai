# Restore and resume offline model evaluation

Use the existing training environment and pinned local Qwen3 base. None of these commands trains a model, contacts MCP, executes tools or downloads weights. Run from the pins-ai repository root. On Windows, the existing Python is `.local/training-venv/Scripts/python.exe`; on Colab/Linux use the corresponding environment Python.

## Restore an archive without modifying original evidence

```powershell
.local/training-venv/Scripts/python.exe training/restore_results.py --zip RESULTS.zip --output .local/evaluation/RUN/restored --base .local/training/base
```

The destination must not exist. All ZIP members undergo path/type/duplicate checks and CRC verification. Restoration copies only data, tokenizer, safetensors, JSON, logs and image files into a new directory. Optimizer/training-argument/RNG pickles stay in the original ZIP and are never loaded. The verification JSON records archive/file hashes, source/run provenance, checkpoint metadata and cached base-file agreement. This proves byte integrity, not external authenticity of unsigned provenance.

Inspect `full/training-run.json` and `restoration-verification.json` before inference. A completed full run must have a final `full/adapter`; the selected final adapter is preferred. For an incomplete snapshot, the best saved checkpoint is diagnostic evidence only: it does not establish training completion or identify an unavailable final checkpoint. No metadata is rewritten to pretend completion.

## Smoke evaluation

```powershell
.local/training-venv/Scripts/python.exe training/evaluate_v2.py --suite validation_smoke --per-language 2 --max-new-tokens 128 --adapter .local/evaluation/RUN/restored/full/adapter --output .local/evaluation/RUN/smoke.json
```

The default backend remains CUDA FP16/NF4 with double quantization, greedy decoding, the original pinned Qwen3 template and `enable_thinking=False`. Base weights and tokenizer files are checked against pinned provenance. The deterministic selection has one expected tool case and one expected no-tool case per supported language. The old `partial` field means a selected subset; `completed` means all selected cases were recorded, not that the full benchmark ran. New `subset`, `expectedCaseCount`, `casesCompleted` and `interrupted` fields make that distinction explicit.

For an explicitly authorized diagnostic of an incomplete run, use its metadata-selected best saved checkpoint and add `--allow-incomplete-checkpoint`. This is refused by default and every resulting report is marked `diagnosticIncompleteCheckpoint=true`. Do not use this flag as evidence of a completed model.

`--backend cpu-fp32` provides an explicit unquantized CPU fallback using the same evaluator/template/schema, without merging the adapter. It is a different numerical backend and its reports cannot resume or compare against CUDA/NF4 reports. Runtime/platform/GPU/library information is recorded for each execution segment. Hardware changes can affect numerics and performance even within CUDA/NF4; deterministic case ordering and greedy decoding do not prove bit-identical output across GPUs.

## Resume into a new report

```powershell
.local/training-venv/Scripts/python.exe training/evaluate_v2.py --suite validation_smoke --per-language 2 --max-new-tokens 128 --adapter .local/evaluation/RUN/restored/full/adapter --resume-from .local/evaluation/RUN/smoke.json --output .local/evaluation/RUN/smoke-continued.json
```

Use the same options and adapter. Identity includes base revision/file hashes, adapter weights/config/run/state hashes, suite and input hashes, selected-case/prompt hashes, policy/template/tool contracts, scorer version/implementation, generation settings, backend/quantization and relevant dependency versions. Paths may change if the bytes and identity are unchanged. Adapter metadata and checkpoint state must not be edited between runs.

Recorded results must form the deterministic case prefix. IDs, expected answers, input/prompt hashes and scores are checked before model loading. Recorded predictions, including wrong or malformed predictions, are skipped; rerunning only incorrect answers would bias the score. Totals are recomputed from unique results. Each completed case is fsynced to a temporary file and atomically replaces only the newly reserved output. Existing output files and the original report are never overwritten. Hard process termination can lose the currently generating case, but the last complete report remains resumable.

Old reports without the new identity fields are still readable evidence, but strict resume rejects them: missing input/generation provenance cannot be reconstructed with certainty. Start a new evaluation instead of silently mixing old and new cases. An already completed compatible report is validated and copied to a new file without model inference.

## Development comparison

For a verified completed v3 adapter, omit `--per-language` to evaluate all 583 development validation conversations. Evaluate v2 with the same command, changing only the adapter and output paths.

```powershell
.local/training-venv/Scripts/python.exe training/evaluate_v2.py --suite validation_smoke --max-new-tokens 128 --adapter .local/evaluation/RUN/restored/full/adapter --output .local/evaluation/RUN/v3-validation.json
.local/training-venv/Scripts/python.exe training/evaluate_v2.py --suite validation_smoke --max-new-tokens 128 --adapter PRIOR_V2_ADAPTER --output .local/evaluation/RUN/v2-validation.json
.local/training-venv/Scripts/python.exe training/compare_evaluations.py --report v3=.local/evaluation/RUN/v3-validation.json --report v2=.local/evaluation/RUN/v2-validation.json --output .local/evaluation/RUN/comparison.md
```

The comparison rejects incomplete reports or differences in base, cases, dataset, policy, schemas, prompt/template, generation, scorer or numerical backend. It recomputes overall/per-language/per-category totals, reports truncation and generation performance, and lists tool/argument/schema failures. Schema-valid plain answers can still be incorrect missing tool calls. Human response-language and false-completion review are not automatically scored.

Validation first-action evaluation does not cover conditional job polling. `--job-continuations` adds synthetic polling cases only for `eval_gold`. Hard-gold remains isolated from training and iterative optimization; do not repeatedly run it to improve development scores. Readiness still requires an independently reviewed holdout, enforceable authorization gateway, human multilingual review and separate quantized/Pi evaluation.

All restored/evaluation evidence belongs under `.local/`, already ignored by Git. Keep ZIPs and model weights outside commits. Generic code/docs/tests can be committed separately; no notebook or training settings need changing to run these commands.

## Tests

```powershell
.local/training-venv/Scripts/python.exe training/tests/evaluation_tests.py
.local/training-venv/Scripts/python.exe training/tests/v2_tests.py
```

Tests cover successful and interrupted resume, duplicate/out-of-order IDs, changed datasets/model identities/scoring/generation/backend, re-scoring, incorrect-prediction preservation, false completion, legacy rejection, existing-file protection, atomic-write failure, diagnostic checkpoint gating, comparison compatibility and archive path/type collisions.
