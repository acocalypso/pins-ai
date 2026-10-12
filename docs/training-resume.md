# Continue a saved Colab training run locally

Continuation is an explicit training action, distinct from offline evaluation. It requires a full checkpoint, identical pinned base/configuration/data and a successful local resumed-step smoke. The preserved archive and original restoration are never overwritten. Use the existing CUDA training Python; do not train on the Pi or contact equipment/MCP.

## Restore optimizer state and reproduce data

```powershell
.local/training-venv/Scripts/python.exe training/training_resume.py --zip qwen3-pins-v3-colab-20261011-165756-results.zip --output .local/training/resume-v3/source
.local/training-venv/Scripts/python.exe training/prepare_v2.py --curriculum --max-tokens 1536 --output .local/training/resume-v3/sft
```

Use new paths when repeating. The archive stage validates member paths/types/CRCs and restores checkpoint 700 with optimizer, scheduler, FP16 scaler and RNG data. Optimizer/scheduler/scaler files are read with `torch.load(weights_only=True)`. RNG uses the pinned Transformers safe NumPy allowlist and restricted loading. Never use an unrestricted pickle fallback; training_args.bin is neither restored nor loaded. An original state JSON is retained and only the new working copy's obsolete Colab best-checkpoint path is relocated.

Prepared JSONL now uses LF on both Windows and Linux so byte hashes match Colab. Resume validation requires exact prepared manifest/base/config agreement, including the original dataset, assistant-only format, prompt/template, model revision, rank, context budget, batch size and gradient accumulation. Test/eval_gold/hard_gold are not used for training.

## One resumed-step GPU smoke

```powershell
.local/training-venv/Scripts/python.exe training/train.py --base .local/training/base --dataset-mode mcp --config training/config-v3.json --data .local/training/resume-v3/sft --resume-from-checkpoint .local/training/resume-v3/source/full/checkpoints/checkpoint-700 --resume-smoke-steps 1 --logging-steps 1 --output .local/training/resume-v3/gpu-smoke
```

This continues step 700 to 701 using the **full original 846-step learning-rate schedule**. It does not shorten max_steps or reset the optimizer. It runs the original full-sized examples and accumulation16. The resulting run is marked smokeOnly; the saved checkpoint and selected adapter are diagnostic evidence, not a new full specialization. Training must fail rather than silently reduce context/batch, discard examples or change precision when GPU memory is insufficient.

## Finish the original training budget

After the smoke completes, run in the foreground or launch this coordinator as a hidden Windows background process:

```powershell
.local/training-venv/Scripts/python.exe training/continue_training.py --base .local/training/base --config training/config-v3.json --data .local/training/resume-v3/sft --checkpoint .local/training/resume-v3/source/full/checkpoints/checkpoint-700 --smoke-run .local/training/resume-v3/gpu-smoke --output .local/training/resume-v3/continued
```

This starts the actual continuation again from the immutable checkpoint700, not from the diagnostic smoke. There are146 global optimizer-step events left to the original846-step budget. Metrics after checkpoint700 but before interruption are not weight checkpoints; lost steps701–710 must be repeated. FP16 overflow can skip actual optimizer updates, so the saved scheduler epoch need not equal Trainer global_step (the archived scheduler was698 at global_step700). Preserve that state.

Training loads optimizer/scheduler/scaler/RNG and restores saved early-stopping callback state. Trainer skips already consumed dataset batches using the original seed/order. The output folder is new; checkpoint retention only rotates newly saved checkpoints. Original checkpoints remain intact. At completion, Trainer reloads the best validation checkpoint and saves `full/adapter`. Do not assume the last checkpoint is best. `full/training-run.json` is marked completed only after successful save; the coordinator then runs the standard20-case offline validation smoke. Full583-case matched comparison and independent safety holdout remain separate.

GPU/OS changes mean this is a continuation of the saved optimization state, not a claim of bit-identical T4 training. Local hardware and source checkpoint hashes are recorded. On interruption, inspect the new full run and use its most recent complete checkpoint; never restart from a final adapter alone and pretend to preserve optimizer history.

## Progress and losses

- `continued/pipeline.json`: phase, coordinator/child PID, exit codes, selected adapter and final smoke summary.
- `continued/training.log`: live Trainer progress, loss logs and errors.
- `continued/full/metrics-latest.json` and `metrics.jsonl`: training-window loss and evaluation loss/trend.
- `continued/full/checkpoints/`: newly saved checkpoints.
- `continued/full/adapter/`: selected adapter after successful completion.
- `continued/final-smoke.json` / `final-smoke.log`: post-training offline smoke.

Validation/checkpoints use the original100-step cadence; the next scheduled validation is step800. Training-window loss is printed every5 global steps. Prior best/previous eval loss is seeded from checkpoint700, making the next validation delta meaningful. Transformers4.57.6 reports resumed train_loss with the full global-step denominator despite accumulating only the new segment's loss; monitoring now exposes the corrected `train_loss_segment` and labels the raw Trainer value. It never labels this resumed-only loss as the original whole-run average. Runtime is also for the resumed segment; stock Trainer sample/step rates should not be interpreted as resumed-work rates.

```powershell
Get-Content .local/training/resume-v3/continued/training.log -Tail 30 -Wait
Get-Content .local/training/resume-v3/continued/full/metrics-latest.json
.local/training-venv/Scripts/python.exe tests/training_resume_tests.py
```

All model/checkpoint/run/log files belong under ignored `.local/`. No archives, weights or optimizer pickles should be committed.
