# V6 training accuracy work — 2026-10-13

The latest completed adapter did **not** beat the previous adapters in the matched
deployment-format comparison. Keep the installed model until a new candidate passes
matching quantized evaluation and plugin-owned Pi checks. This change prepares a
better development recipe; no v6 full training or accuracy improvement is claimed.

## Measured comparison

All three adapters used the identical bundled Q4_K_M base, FP16 runtime LoRA,
pinned llama.cpp `8345f333951c661d166b00e6f9362e553768f292`, two CPU threads,
1,536-token context, 192-token output, greedy decoding and seed 3407.

| Adapter | Legacy hard-gold exact arguments | V5 chat development exact arguments | V5 chat schema-valid |
| --- | --- | --- | --- |
| New completed step 1900 | 81/120 (67.5%) | 400/583 (68.6%) | 572/583 |
| Prior diagnostic step 700 | 89/120 (74.2%) | 417/583 (71.5%) | 571/583 |
| Prior diagnostic step 400 | 82/120 (68.3%) | 425/583 (72.9%) | 558/583 |

All reports completed with zero output truncations. On chat development cases the
new adapter won 86 and lost 103 exact-argument cases against step 700. The completed
run used the older v4 corpus, before the v5 natural-chat update. Its folder label
was `mi355x`, but recorded hardware was RX 7900 XT (`gfx1100`); that is not proof of
an MI355X run. Earlier BF16 evaluation was substantially better, but differences
in runtime, prompts and precision prevent attributing the whole gap to quantization.
Concurrent desktop timing is not isolated Pi latency. Correct scoped tools were
supplied to the models: these scores do not measure automatic router retrieval.

The run's training-window loss fell from 1.7464 to 0.007; validation loss fell from
0.623 at step 100 to 0.0588 at 1900. These losses did not predict the deployment
ranking. Aggregate scores, model/runtime hashes and report identities are committed
in [comparison evidence](../training/experiments/model-comparison-20261013.json).
Raw held-out prompts/predictions are not included in newly authored training data.

## Data changes

The train-only corpus now contains **5,319 conversations**, including **650 new
independently authored examples** across ten languages:

- 390 argument variants: seconds/minutes, decimal commas, RA degrees-to-hours,
  signed coordinates/temperatures, absolute focus positions, zero-based switch and
  filter indices, exposure counts, exact sequence names and observed job IDs.
- 250 withdrawn-request contrasts: every modifying tool in every language.
- Ten multilingual histories that poll an observed running job without restarting
  the equipment operation or claiming completion.

Existing natural-chat, clarification, unsupported-operation, sequencing,
dependent-operation, discovery, feedback, cancellation and injection coverage is
retained. All 32 MCP tools and their AI-exposed arguments remain covered. Synthetic
equipment state and pending native-language review are explicit. Authored sources
use MCP contracts and existing training phrases, never held-out prediction text.
Original 326-example baseline and original/audited test, validation and gold files
retain their bytes. Changing the active projection version changes report identity;
rerun all candidates with identical v6 prompts for a fresh comparison.

Pinned-tokenizer preparation produced **7,588 unique assistant training turns**,
removing 1,147 duplicate rendered turns, with **zero skipped examples**. Train
length maximum is 1,318 tokens, maximum supervised completion 139 tokens; validation
has 1,236 turns, maximum length 1,309 and completion 63. Assistant-only native Qwen3
format is preserved. No silent truncation or training on gold is permitted.

The real gateway schema audit accepted all **4,558** calls, with zero errors.
Router retrieval remains **134/320 (41.875%)** on the fixed natural-chat audit.
Training cannot supply a tool omitted by retrieval. Improving and retesting that
router is separate runtime work, required before claiming end-to-end improvement.

## Training changes

V6 retains the pinned base, rank 8/alpha 16, dropout 0.05, microbatch 1,
accumulation 16, context 1,536, seed 3407 and primary learning rate 5e-5.
Every 100 optimizer steps it measures both validation loss and greedy generated
proposals on **102 development-validation cases** covering all 32 tools and ten
languages, including all available denial, injection and no-start cases.

`eval_selection_score` selects the best checkpoint. With no unwanted no-tool
proposals or truncations it is exact-argument accuracy averaged equally over each
tool bucket and a no-tool bucket. Otherwise it is negative, below every clean
candidate, penalizing unwanted proposals/truncations. This is a bounded proposal
test, not authorization or a proof that arbitrary model text is safe. The
application's enforceable gateway remains responsible for execution and approval.
Early stopping uses four evaluations and a 0.005 improvement threshold. Retention
is four checkpoints subject to Trainer best/latest handling, not an archive of
every checkpoint. Per-case validation reports persist at each measured step.

The metrics files and Colab charts retain `train_loss` and `eval_loss`, and add tool
choice, exact arguments, macro exact accuracy, schema validity, unwanted proposals,
truncation and selection score. Best-checkpoint score and best-checkpoint loss are
separate fields. Validation restores training mode, cache configuration and RNG
state, and its metrics reach Trainer callbacks and selection in the same event.

Full v6 training requires a successful matching GPU smoke with identical
configuration, prepared splits, generated cases/scoring code, Python/dependency
versions and GPU backend/hardware. Exact checkpoint continuation also verifies the
generated validation contract. Legacy profiles retain their original behavior;
resume an old run using its original recipe **and source revision**.

## Start a fresh run

Follow [environment setup](../training/README.md) first: Linux ROCm uses a supported
PyTorch build with actual GPU kernels plus `requirements-common.txt`; NVIDIA/Colab
uses the documented FP16/NF4 environment. Do not install CUDA bitsandbytes for the
ROCm BF16 recipe. Then, from the repository root:

```bash
git pull --ff-only
python training/preflight.py --method lora --precision bf16
python training/prepare_v2.py --curriculum --max-tokens 1536 \
  --base .local/training/base --output artifacts/training-data/current-v6
CONFIG=training/configs/mi355x-v6-lora-bf16.json
DATA=artifacts/training-data/current-v6
RUN=artifacts/training-runs/qwen3-v6-$(date -u +%Y%m%d-%H%M%S)
python training/train.py --base .local/training/base --dataset-mode mcp \
  --data "$DATA" --config "$CONFIG" --max-steps 4 --logging-steps 1 --output "$RUN/smoke"
python training/evaluate_v2.py --backend gpu-bf16 --adapter "$RUN/smoke/adapter" \
  --suite validation_smoke --prompt-mode chat --per-language 2 --max-new-tokens 192 \
  --output "$RUN/smoke-reload.json"
```

Inspect completed reports, finite losses/metrics and measured GPU memory. After
the matching smoke/reload passes:

```bash
python training/train.py --base .local/training/base --dataset-mode mcp \
  --data "$DATA" --config "$CONFIG" --smoke-run "$RUN/smoke" \
  --logging-steps 5 --output "$RUN/full"
tail -f "$RUN/full/metrics.jsonl"
```

For T4 use `training/configs/config-v6.json`, `cuda-nf4` evaluation and
`--method qlora --precision fp16` preflight. The regenerated
[Colab notebook](../training/notebooks/pins-ai-colab.ipynb) pulls Git code/data,
keeps verified Drive archive compatibility and writes checkpoints/results to Drive.
The standard run starts from the base, not from the rejected step-1900 adapter.

The optional AMD `mi355x-v6-conservative-lora-bf16.json` changes only learning rate
to 3e-5; it is an **unverified experiment**. Give it its own matching smoke/output
and compare on development data. Do not assume a lower rate improves accuracy.
The [experiment plan](../training/experiments/v6-accuracy-plan.json) records controls.

## Verified and still open

Verified locally: complete pinned-tokenizer preparation, corpus reproducibility,
frozen-data checks, schema parity, nine generated-validation tests (including
real tiny-Qwen CPU training/checkpoint selection), metrics persistence and RNG restoration,
resume/evaluation/export checks, and simulated plugin/gateway tests. Colab cells
compile with cleared outputs. CPU PEFT evaluation/export now explicitly honors CPU
placement; GGUF evaluation supports verified resume into a new output file.

Not verified: v6 CUDA/ROCm smoke/full training, evaluation duration/VRAM on those
GPUs, native-language review, v6 deployment accuracy or Pi performance. No full
training, model replacement or hardware operation was performed for this update.

Next: run matching smoke/full training; compare base/prior/new on complete
development validation and job histories; review language and false completion
claims; export and compare identical Q4_K_M runtime settings; use untouched gold
only for final release assessment. Previously inspected gold is historical evidence,
not a fresh unbiased holdout. Independently author/review a new frozen holdout
before making a new unseen-accuracy claim. Promote only after quantized quality,
gateway safety and plugin-owned Pi resource/latency checks pass.
