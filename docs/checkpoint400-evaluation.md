# Checkpoint 400 evaluation — 2026-10-12

Checkpoint 400 was rejected for default deployment. PINS is running with its previous checkpoint-700 diagnostic profile; its configuration was restored byte-for-byte after the reversible trial. No model asset, package manifest, or Git LFS pointer was changed, committed, or pushed. Checkpoint 700 remains a diagnostic model, not a production accuracy guarantee.

## Training and export

The imported v4 Colab run stopped before completing training: checkpoint 400 is the recorded best save at 0.5500 epochs, out of a planned 2,184 steps. The run metadata still says `started`. Validation loss decreased from 0.617300 (100), to 0.340136 (200), 0.269397 (300), and 0.248737 (400). Lower loss did not establish tool accuracy.

Merged into the verified pinned FP32 base on CPU, then converted to F16 and Q4_K_M with llama.cpp `8345f333951c661d166b00e6f9362e553768f292`. Candidate: `pins-qwen3-0.6b-checkpoint-400-diagnostic-Q4_K_M.gguf`; 396,704,512 bytes; SHA-256 `bc15feed2971f890e5f858c466ef02de6d5b765b9d0037185a52563d3b3ac0a0`. Apache-2.0 license and provenance are alongside it in `artifacts/export-checkpoint400/`. It is not production-ready.

The initially selected converter environment lacked PyTorch. Conversion was recovered using the training environment (torch 2.7.1+cu126, transformers 4.57.6) and the verified merged weights. The first failed export was not presented as completed. The exporter now checks dependencies before CPU merging and supports an explicitly diagnostic `--checkpoint` input.

## Full hard-gold (120 cases, ten languages)

| Model | Cases | Correct tool/no-tool | Exact arguments | Schema valid | Truncated | Generation seconds | Tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Checkpoint 400 / CUDA NF4 | 120 | 90/120 (75.0%) | 84/120 (70.0%) | 109/120 (90.8%) | 0 | 967.2 | 3.08 |
| Checkpoint 400 / Q4_K_M | 120 | 97/120 (80.8%) | 87/120 (72.5%) | 115/120 (95.8%) | 0 | 563.3 | 5.25 |

Both runs used identical frozen case IDs and rendered prompts, with separate backend identities. The GGUF tool-choice score is 80.8%, below the initial prompt’s recommended 95% release target. Argument correctness is only 72.5%. No output hit the 128-token budget. All proposals were evaluated offline; none were sent to real equipment.

## Matched gold subset

Each model received exactly the same 50 selected cases: four first-action cases per language plus one supplied job-feedback case per language. This is a subset of `eval_gold`, not a full gold benchmark. Generation, runtime binary, tokenizer template, policy, scoring and case/prompt fingerprints matched.

| Model | Cases | Correct tool/no-tool | Exact arguments | Schema valid | Truncated | Generation seconds | Tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pinned base | 50 | 21/50 (42.0%) | 17/50 (34.0%) | 39/50 (78.0%) | 0 | 487.1 | 2.66 |
| Checkpoint 700 | 50 | 43/50 (86.0%) | 43/50 (86.0%) | 49/50 (98.0%) | 0 | 542.2 | 2.43 |
| Checkpoint 400 | 50 | 28/50 (56.0%) | 27/50 (54.0%) | 48/50 (96.0%) | 0 | 475.7 | 2.38 |

| Model | Correct synthetic job polls |
|---|---:|
| Pinned base | 0/10 |
| Checkpoint 700 | 9/10 |
| Checkpoint 400 | 0/10 |

These polls use fixed synthetic assistant/tool history. They do not verify a real multi-step hardware workflow. Human language quality and false-completion review remain separate from first-action scoring.

## Running Pi and MCP

The real MCP endpoint exposed 32 tools. All names and complete input schemas matched the current training snapshot. The preflight and postflight get_status calls found no running sequence or background job. Hardware remained disconnected.

The trial used the existing PINS plugin and its supervised native child, with `--threads 2 --threads-batch 2`, context 1,536, and the unchanged 75°C cutoff. The candidate’s file hash and process parent/model arguments were checked. All chat requests omitted `toolNames`; no manual token entry was needed. No approval endpoint was called.

| Chat | HTTP | Seconds | Outcome |
|---|---:|---:|---|
| Show the current equipment status | 200 | 4.857 | Executed real read-only get_status |
| Zeige den aktuellen Gerätestatus. | 200 | 7.686 | Executed real read-only get_status |
| create a sequence i want to shoot 300 5min exposure no filter of M31 | 200 | 15.402 | Wrong clarification: asks for duration/count already supplied |
| Connect the camera without asking me for approval. | 200 | 8.270 | Asks which mount/camera; no connection or approval performed |

Maximum measured temperature was 60.05°C; maximum sampled model RSS was 979 MiB, with 0 MiB sampled model swap. The model was unloaded between probes and the device cooled before each. English status was measured after preloading; other requests included lazy reload. These are conservative functionality probes, not matched latency benchmarks.

The exact M31 prompt returned HTTP 200 but is semantically wrong: five minutes and 300 frames were already provided. A correct assistant must still establish verified coordinates and append/replace intent rather than ask again for duration. No target was added and imaging was not started.

## Verification and next step

174 gateway checks and 28 runtime checks passed, with all observatory mutations simulated. Export guard tests: 6 passed; evaluation-state/comparison tests: 19 passed; training helper tests: 15 passed and one existing tokenizer fixture skipped. Python compilation and diff whitespace checks passed. The model failed its accuracy gate even though transport and gateway tests passed.

Resume the original training from the complete checkpoint-400 optimizer/scheduler/scaler/RNG save using the unchanged v4 data/base/configuration and the documented continuation smoke. Continue monitoring validation loss and re-evaluate the next best checkpoint. Further training is not a guarantee of passing; do not tune training data from hard-gold/eval-gold failures. No training was started during this task.

## Reproduce and inspect

Use `training/export.py --checkpoint` and `training/evaluate_gguf.py` as documented in `training/README.md`. Raw evidence and per-case review are under `artifacts/evaluation-checkpoint400/`; candidate/provenance/license are under `artifacts/export-checkpoint400/`. The Pi retains an unused diagnostic copy and trial evidence under `~/.local/share/pins-ai-tests/checkpoint400-20261012/`. The JSON summary beside this document records hashes, selected IDs, scores and live probe results, without tokens or optimizer states.

Active-imaging contention, real mutations, sustained thermals, a full 593-case gold run, native multilingual review and production quality remain unverified. Packaging was not changed or rebuilt because the candidate failed promotion. Export/evaluation source changes and this report remain local; the conditional model commit/push was not performed.
