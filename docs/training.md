# Training PINS AI

This is the complete process for training the Qwen3 0.6B MCP assistant. Training runs
off-device on a GPU workstation, Colab, or a future AMD Instinct host. The PINS
plugin owns inference and authorization; training never operates equipment.
All commands below start from the **repository root**. No training runs start
automatically. Start a fresh adapter from the pinned base when changing data.

The sequence is: choose a GPU environment, validate data, fetch the pinned base,
prepare SFT, run smoke/reload, train, inspect metrics, compare candidates, export
GGUF and evaluate deployment parity. Only then consider Pi promotion. See
[the model report](model-report.md) for the retained model and measured results.

## Contents

- [Requirements and profiles](#requirements-and-profiles)
- [Clone and data](#clone-and-find-the-data)
- [NVIDIA / Colab setup](#nvidia-setup-linux-or-colab)
- [ROCm setup](#mi355x--rocm-setup)
- [Validate and prepare](#validate-fetch-the-pinned-base-and-prepare-sft)
- [Smoke and full training](#smoke-and-full-training)
- [Monitor and find checkpoints](#monitor-loss-and-locate-checkpoints)
- [Resume training](#resume-an-interrupted-run)
- [Compare and export](#compare-accuracy-and-export-a-candidate)
- [Resume evaluation](#restore-and-resume-evaluation)

## Requirements and profiles

| Environment | Python / PyTorch | Dependencies | Configuration | Evaluation backend |
|---|---|---|---|---|
| NVIDIA workstation / Colab T4 | Python 3.11, torch 2.7.1 CUDA 12.6 | `training/requirements.txt` | `training/configs/config-v6.json` (FP16/NF4 QLoRA) | `cuda-nf4` |
| AMD Instinct MI355X, Linux x86-64 | AMD-supported ROCm/PyTorch with gfx950 kernels; example below | `training/requirements-common.txt` | `training/configs/mi355x-v6-lora-bf16.json` (BF16 LoRA) | `gpu-bf16` |

Use Git and Python; Git LFS is needed only for the bundled deployment GGUF files.
The optional actual gateway-schema audit below additionally needs .NET SDK 10;
Python training itself does not require .NET. Run that audit on the development
host or CI when contracts/data change, rather than installing .NET on every GPU VM.
Allow at least 20 GB free disk for a base checkpoint, runs, checkpoints and export,
plus container storage on ROCm. 16 GB host RAM is a reasonable starting point,
but CPU export and accumulated checkpoints can require more. A 4 GB NVIDIA GPU
is the constrained legacy target: the full 1,536-token recipe may OOM. A passing
smoke test is mandatory and is not a memory guarantee for the full run.

Training is currently **single GPU / single process**. Do not launch multiple
workers with accelerate/torchrun: provenance, checkpoint restoration and device
mapping assume one device. On a multi-GPU host, select one GPU explicitly.
No FlashAttention, Unsloth, NVIDIA paged optimizer or AMD NF4 support is assumed.

## Clone and find the data

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/acocalypso/pins-ai.git
cd pins-ai
```

All training data is in Git under **`training/dataset/`**. The single active
training file is `training/dataset/train.jsonl`. Other JSONL files are retained
sources or isolated validation/test/gold, not additional datasets to train or
blindly concatenate. The contents are:

- `training/dataset/baseline/authored.jsonl`: reviewed additions to the original text baseline.
- `training/dataset/baseline/gold.jsonl`: text evaluation cases, not training input.
- `training/dataset/baseline/sequence-plans.jsonl`: 22 train-only English/German conversations.
- `training/dataset/data/train.jsonl`: original 924 synthetic conversations.
- `training/dataset/data/validation.jsonl`: 583 validation conversations.
- `training/dataset/data/test.jsonl`: 593 held-out conversations.
- `training/dataset/data/eval_gold.jsonl` and `hard_gold.jsonl`: final evaluation only.
- `training/dataset/data/audited/`: schema-corrected splits used by preparation.
- `training/dataset/sources/comprehensive/train.jsonl`: train-only coverage of all 32
  current MCP tools and ten languages, including arguments, boundaries, job
  outcomes, authorization and unsupported requests. See its [coverage guide](../training/dataset/sources/comprehensive/README.md).
- **Current default:** `training/dataset/train.jsonl`: the v6 multilingual
  chat/proposal curriculum. Read its [guide](../training/dataset/README.md) and
  [comparison and accuracy recipe](model-report.md) before a new run.
- `training/dataset/data/manifest.json`: immutable original data hashes;
  `contract-audit.json` and `verified-tools-list.json` identify audited contracts.

`training/prepare.py` regenerates the preserved 326-example informational baseline.
`training/prepare_v2.py --curriculum` builds the current natural chat curriculum,
including device/safety contrasts and sequence plans. It uses the actual AI gateway
surface (numeric coordinates, waits up to five seconds, no override properties).
Tool calls are proposals before approval; only the gateway can execute them.
Original/audited split files remain unchanged. Active train/validation copies strip
approval answer keys and convert obsolete no-approval refusals into explicit
withdrawals. This is a new recipe; previous checkpoint loss/scores are not directly
comparable with these projected inputs. It renders native Qwen tool calls and
supervises assistant tokens only. It reads train/validation, never test/gold for
training. Preparation fails on bad calls, lost coverage or overlong examples;
it does not silently truncate or remove a troublesome tool class.

## NVIDIA setup (Linux or Colab)

On a workstation with a compatible NVIDIA driver:

```bash
python3.11 -m venv .local/training-venv
source .local/training-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r training/requirements.txt
python training/preflight.py --method qlora --precision fp16
```

On Windows PowerShell, the corresponding environment setup is:

```powershell
py -3.11 -m venv .local/training-venv
.local/training-venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r training/requirements.txt
python training/preflight.py --method qlora --precision fp16
```

Do not replace a working CUDA/Pascal environment with an arbitrary newer wheel.
Preflight executes small GPU kernels, including NF4 quantize/dequantize for QLoRA.
Read its issues and require exit code zero before continuing.

For Colab, open [notebooks/pins-ai-colab.ipynb](../training/notebooks/pins-ai-colab.ipynb), choose
a hosted T4 GPU kernel, and authorize Drive. It clones/pulls Git, verifies the
existing Drive data, installs the pinned Python/CUDA environment, prepares data,
and runs smoke/reload checks before full training. Defaults are `TRAINING_MODE =
'fresh'`, `RUN_TRAINING = False`, and final gold evaluation disabled. Deliberately
enable training after reviewing the configuration. Outputs/checkpoints are under
`/content/drive/MyDrive/pins-ai/runs/<RUN_NAME>/`. Existing Drive directories/ZIPs
using the old `PINS_MCP_AI_Training_v2/data` layout remain supported, with identical
byte-hash checks. Code is never loaded from Drive. Regenerate the notebook with
`python training/create_colab_notebook.py` after changing its generator.

## MI355X / ROCm setup

This path is prepared in code and tested for backend contracts only. **No MI355X
training or performance result has been measured.** Verify the allocated host's
GPU, drivers, supported OS and ROCm version with its administrator. A GPU-less
container or a PyTorch build lacking gfx950 cannot train this profile.

AMD documents a tested ROCm 7.2.1 / PyTorch 2.9.1 / Python 3.12 image. This is a
versioned starting point, not a requirement to downgrade a newer supported host.
See [AMD installation instructions](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.1/install/3rd-party/pytorch-install.html)
and the [MI355X acceptance guide](https://instinct.docs.amd.com/projects/system-acceptance/en/latest/gpus/mi355x.html).

On that Linux host, from the cloned repository:

```bash
docker pull rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1
docker image inspect --format '{{index .RepoDigests 0}}' \
  rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1
docker run --rm -it --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render --ipc=host --shm-size=8g \
  -e HIP_VISIBLE_DEVICES=0 -v "$PWD:/workspace/pins-ai" -w /workspace/pins-ai \
  rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1 bash
```

Record the image digest with the run's evidence. Inside the container:

```bash
python -m pip install -r training/requirements-common.txt
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.get_device_name(0))"
python training/preflight.py --method lora --precision bf16
```

PyTorch's ROCm backend intentionally uses `torch.cuda` and device `cuda`; this
does not install CUDA. Do not install the NVIDIA torch wheel or the NF4 requirements
over the image. This project selects **unquantized BF16 LoRA** for ROCm, and rejects
its CUDA-NF4 path rather than silently substituting another quantizer. The profile
starts conservatively with microbatch 1 and accumulation 16; tune larger batches
only after measurement, using a new run and a matching new smoke test.

## Validate, fetch the pinned base and prepare SFT

With either GPU environment active:

```bash
python training/dataset/scripts/validate_dataset.py
python training/tests/training_tests.py
python training/tests/curriculum_tests.py
python training/tests/comprehensive_tests.py
python training/tests/chat_dataset_tests.py
python training/tests/generation_validation_tests.py
python training/tests/colab_tests.py
python training/tests/preflight_tests.py
python training/fetch_base.py --download --accept-license
python training/tests/v2_tests.py
python training/prepare_v2.py --curriculum --max-tokens 1536 \
  --base .local/training/base --output artifacts/training-data/current
```

Fetching is an explicit network operation. The base is `Qwen/Qwen3-0.6B` at
revision `c1899de289a04d12100db370d81485cdf75e47ca`; weights and tokenizer files
are hash-verified and Apache-2.0 licensed. Training/evaluation then use local
files with Hugging Face network access disabled. Review the prepared manifest's
language/tool coverage, token counts, zero skipped records and supervision tests.
Never substitute a deployment GGUF for the Hugging Face training checkpoint.

For actual gateway-schema parity, on a development machine with .NET SDK 10:

```bash
dotnet run --project training/tools/RouterAudit -- \
  training/dataset/train.jsonl \
  training/dataset/data/verified-tools-list.json artifacts/router-chat-v6.json
python training/tests/chat_dataset_tests.py
```

Without that report, the Python parity fixture explicitly skips its actual-.NET
check; it must not be reported as a measured gateway audit. CI runs the audit.

## Smoke and full training

Choose the environment profile. Linux NVIDIA:

```bash
CONFIG=training/configs/config-v6.json
BACKEND=cuda-nf4
RUN=artifacts/training-runs/qwen3-$(date -u +%Y%m%d-%H%M%S)
DATA=artifacts/training-data/current
```

Linux ROCm uses the same commands below with:

```bash
CONFIG=training/configs/mi355x-v6-lora-bf16.json
BACKEND=gpu-bf16
RUN=artifacts/training-runs/qwen3-mi355x-$(date -u +%Y%m%d-%H%M%S)
DATA=artifacts/training-data/current
```

Run a short GPU smoke, then verify adapter reload on validation cases:

```bash
python training/train.py --base .local/training/base --dataset-mode mcp \
  --data "$DATA" --config "$CONFIG" --max-steps 4 --logging-steps 1 --output "$RUN/smoke"
python training/evaluate_v2.py --base .local/training/base --backend "$BACKEND" \
  --adapter "$RUN/smoke/adapter" --suite validation_smoke --prompt-mode chat --per-language 2 --max-new-tokens 192 \
  --output "$RUN/smoke-reload.json"
```

Require finite training/eval losses, completed `training-run.json`, GPU memory
within capacity, a loadable adapter, and a completed reload report. Inspect
incorrect tool calls: successful execution of the script is not model accuracy.
Only then start full training with the **identical prepared dataset and profile**:

```bash
python training/train.py --base .local/training/base --dataset-mode mcp \
  --data "$DATA" --config "$CONFIG" --smoke-run "$RUN/smoke" \
  --logging-steps 5 --output "$RUN/full"
```

PowerShell uses the same argument names with `$Config`, `$Backend`, `$Run`, `$Data`
instead of bash variables. Set them as below, and put commands on one line or use
PowerShell's backtick for line continuation (not bash's backslash):

```powershell
$Config='training/configs/config-v6.json'
$Backend='cuda-nf4'
$Run='artifacts/training-runs/qwen3-'+(Get-Date -Format 'yyyyMMdd-HHmmss')
$Data='artifacts/training-data/current'
python training/train.py --base .local/training/base --dataset-mode mcp --data $Data --config $Config --max-steps 4 --logging-steps 1 --output "$Run/smoke"
python training/evaluate_v2.py --base .local/training/base --backend $Backend --adapter "$Run/smoke/adapter" --suite validation_smoke --prompt-mode chat --per-language 2 --max-new-tokens 192 --output "$Run/smoke-reload.json"
python training/train.py --base .local/training/base --dataset-mode mcp --data $Data --config $Config --smoke-run "$Run/smoke" --logging-steps 5 --output "$Run/full"
```

## Monitor loss and locate checkpoints

`$RUN/full/metrics.jsonl` is the flushed loss history;
`metrics-latest.json` is the latest snapshot. `train_loss` is the logging-window
average; `eval_loss_delta < 0` indicates validation improvement. If training loss
falls while validation loss rises, examine overfitting rather than simply training
longer. Final `train_loss_overall` and resumed `train_loss_segment` are labeled
separately. Lower loss does not prove MCP accuracy or authorization safety.

V6 also logs `eval_tool_choice`, `eval_tool_exact`, `eval_tool_macro_exact`,
`eval_schema_valid`, `eval_unsafe_proposals`, `eval_generation_truncated` and
`eval_selection_score`. Every 100 steps, it generates a fixed development-only
sample covering all tools and languages. Checkpoints are selected by macro exact
accuracy, with unwanted no-tool proposals and truncation penalized. Early stopping
uses four checks and a 0.005 improvement threshold. Per-case outputs and the exact
validation contract are saved in `full/generation-validation/step-<step>.json`.
Gold never participates in selection. Generated validation adds evaluation time;
measure it in the matching GPU smoke before starting a full run.

```bash
tail -f "$RUN/full/metrics.jsonl"
cat "$RUN/full/metrics-latest.json"
```

```powershell
Get-Content "$Run/full/metrics.jsonl" -Wait -Tail 10
Get-Content "$Run/full/metrics-latest.json"
```

Checkpoints are in `full/checkpoints/checkpoint-<step>/`. The standard profiles
save/evaluate every 100 optimizer steps, **not after every batch**. Retention is
four checkpoints for v6, subject to Trainer's best/latest handling; they are not an archive
of every save. The completed `full/adapter/` contains the validation-selected best
adapter. `training-run.json` records versions, hashes, configuration and status.
`bestCheckpointScore` is the selected score, not a loss; final `metrics.eval_loss`
remains the measured loss at the selected checkpoint. Existing v3/legacy AMD
profiles retain their original loss selection and retention for exact resumes.
Changed v6 data requires fresh training from the base. A lower-rate AMD ablation
is available in `training/configs/mi355x-v6-conservative-lora-bf16.json`; it is experimental,
not a measured improvement. See the [experiment plan](../training/experiments/v6-accuracy-plan.json).

## Resume an interrupted run

Keep original base/data/configuration and the original `training-run.json` with
the checkpoints. Restore the latest complete save from a Drive-style run directory
or a results ZIP into a **new** directory:

```bash
python training/training_resume.py --source /path/to/original-run-or-results.zip \
  --output artifacts/restored-resume
```

Read `artifacts/restored-resume/resume-restoration.json` and use its `checkpoint`
path. It preserves the best adapter separately. Set `CHECKPOINT` to that path, and
`RUN` to a new output directory, then run a one-step continuation smoke:

```bash
python training/train.py --base .local/training/base --dataset-mode mcp \
  --data "$DATA" --config "$CONFIG" --resume-from-checkpoint "$CHECKPOINT" \
  --resume-smoke-steps 1 --logging-steps 1 --output "$RUN/smoke"
python training/evaluate_v2.py --base .local/training/base --backend "$BACKEND" \
  --adapter "$RUN/smoke/adapter" --suite validation_smoke --prompt-mode chat --per-language 2 --max-new-tokens 192 \
  --output "$RUN/smoke-reload.json"
python training/train.py --base .local/training/base --dataset-mode mcp \
  --data "$DATA" --config "$CONFIG" --resume-from-checkpoint "$CHECKPOINT" \
  --smoke-run "$RUN/smoke" --logging-steps 5 --output "$RUN/full"
```

This restores optimizer, scheduler and RNG; FP16 additionally requires scaler
state, whereas BF16 has no FP16 scaler. Do not use `--max-steps` for continuation:
the original step/LR budget must stay intact. A CUDA QLoRA checkpoint cannot be
resumed as ROCm BF16 LoRA; that is a new training recipe. Start fresh after changing
training examples, policy or tool contracts. Missing optimizer state means an
adapter-only warm start, not an exact resume, and is refused here.

## Compare accuracy and export a candidate

During development, compare base/prior/new on `validation_smoke`. Use separate new
output files, `--prompt-mode chat --max-new-tokens 192`, and the same backend/generation settings.
This mode uses the same proposal-stage projection as current training, preserves
all 583 validation scenario IDs, and does not run gold suites. The default legacy
gold prompt mode remains available for historical comparisons; its older approval
fixtures do not establish current chat accuracy. An independently authored and
reviewed deployment holdout is still required. Once the recipe is frozen,
run final `eval_gold` and `hard_gold` for base, completed prior adapter and new
adapter. Never tune data/prompts against those held-out scores.

First perform the development comparison (no gold):

```bash
python training/evaluate_v2.py --base .local/training/base --backend "$BACKEND" \
  --suite validation_smoke --prompt-mode chat --max-new-tokens 192 \
  --output "$RUN/base-validation.json"
python training/evaluate_v2.py --base .local/training/base --backend "$BACKEND" \
  --adapter "$RUN/full/adapter" --suite validation_smoke --prompt-mode chat \
  --max-new-tokens 192 --output "$RUN/new-validation.json"
python training/compare_evaluations.py --report base="$RUN/base-validation.json" \
  --report new="$RUN/new-validation.json" --output "$RUN/development-comparison.md"
```

Repeat with the prior adapter and include `--report prior=...`. Incomplete prior
checkpoints need their evaluator's explicit diagnostic option and metadata; do
not relabel them completed. Only after freezing the recipe use final gold:

```bash
python training/evaluate_v2.py --base .local/training/base --backend "$BACKEND" \
  --suite eval_gold --job-continuations --output "$RUN/base-eval.json"
python training/evaluate_v2.py --base .local/training/base --backend "$BACKEND" \
  --adapter "$RUN/full/adapter" --suite eval_gold --job-continuations --output "$RUN/new-eval.json"
python training/evaluate_v2.py --base .local/training/base --backend "$BACKEND" \
  --adapter "$RUN/full/adapter" --suite hard_gold --output "$RUN/new-hard-gold.json"
```

Repeat with `--adapter /path/to/completed-prior-run/adapter` for the prior model.
Reports include tool/argument validity, language/category/risk breakdown and
conditional job-poll cases. Model outputs are simulated proposals, never actions.
Incomplete checkpoint-700 diagnostics are not completed training results.

Export merges into a clean unquantized CPU base, then converts and quantizes using
the runtime-pinned llama.cpp revision. On Linux, with Git/CMake/C++ compiler:

```bash
git clone https://github.com/ggml-org/llama.cpp.git .local/llama.cpp
LLAMA_COMMIT=$(python -c "import json; print(json.load(open('plugin/models/manifest.json'))['runtime']['commit'])")
git -C .local/llama.cpp checkout --detach "$LLAMA_COMMIT"
python3 -m venv .local/converter-venv
.local/converter-venv/bin/python -m pip install -r .local/llama.cpp/requirements.txt
cmake -S .local/llama.cpp -B .local/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF
cmake --build .local/llama.cpp/build --target llama-quantize -j 2
python training/export.py --base .local/training/base --run "$RUN/full" \
  --llama-source .local/llama.cpp --converter-python .local/converter-venv/bin/python \
  --output "$RUN/export"
```

Keep `model-provenance.json`, hashes and licenses. An export is a **candidate**, not
automatic promotion. Re-evaluate the quantized model with the same frozen suites
and conservative Pi probes before updating `plugin/models/bundled.json` and Git
LFS assets. Never overwrite the known baseline or deployment model during training.

For an interrupted run, explicitly select its metadata-recorded best saved
checkpoint. The regular `--run` path still rejects unfinished runs. This creates
a diagnostic candidate with checkpoint hashes and step counts; it does not make
the training run complete or make the model production-ready:

```bash
python training/export.py --base .local/training/base \
  --checkpoint "$RUN/full/checkpoints/checkpoint-400" \
  --llama-source .local/llama.cpp --converter-python .local/converter-venv/bin/python \
  --output artifacts/export-checkpoint400
```

Build `llama-server` from the same pinned source for **off-device, offline** GGUF
evaluation. This temporary workstation test process only generates proposals;
it has no MCP connection. Use the existing PINS plugin for actual Pi inference.
Evaluation reports are written after each case; keep separate outputs for each
model. The full hard-gold suite contains 120 cases. A positive `--per-language`
selects a deterministic subset, not a full gold evaluation. On `eval_gold`,
`--job-continuations` also evaluates supplied synthetic job feedback; no actual
hardware job is started or polled.

To continue an interrupted GGUF evaluation, repeat the original command with
`--resume-from path/to/partial.json` and a **new** `--output` path. Keep the same
model, LoRA, runtime, dataset, prompt mode and generation options. Resume verifies
their hashes and rechecks every saved prediction before skipping completed cases;
it preserves the original report. This resumes evaluation, not model training.

```bash
cmake -S .local/llama.cpp -B .local/llama.cpp/build \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON
cmake --build .local/llama.cpp/build --target llama-server -j 2
python training/evaluate_gguf.py \
  --model "$RUN/export/pins-qwen3-0.6b-mcp-v2-Q4_K_M.gguf" \
  --server .local/llama.cpp/build/bin/llama-server --llama-source .local/llama.cpp \
  --suite validation_smoke --prompt-mode chat --max-new-tokens 192 \
  --output "$RUN/gguf-validation.json"
```

Repeat the same selected cases against the pinned base and previous adapter
(add `--lora plugin/models/bundled/checkpoint-700-lora-f16.gguf` for the prior
deployment). Do not compare scores from different subsets as if they were
matched measurements. First-action accuracy is separate from argument accuracy,
language quality, false claims, and enforced gateway authorization.
See [plugin build instructions](../plugin/README.md), [gateway safety](../plugin/README.md#authorization-and-job-safety),
and [measured Pi results](model-report.md#pi-verification).

## Current recipe and verification status

Source data byte hashes and the original baseline are preserved.
Offline runtime/gateway, data, resume, evaluation-state and ROCm routing tests run
without training. This updated notebook has been syntax/contract checked, not run
on a new Colab allocation. The MI355X image/setup, real GPU smoke, full BF16 training,
BF16 checkpoint restoration on hardware and quantized-model parity are pending
hardware access. No AMD throughput or accuracy is claimed.

V6 contains 5,319 conversations, including 650 new multilingual parameter,
withdrawal and observed-job histories. Pinned-tokenizer preparation measured
7,588 unique assistant training turns, 1,147 duplicates removed and zero skipped
examples. Maximum train sequence/completion lengths are 1,318/139 tokens.
The 102-case generated validation sample uses development data only; the score
averages exact accuracy equally over each tool bucket and one no-tool bucket.
Unwanted no-tool proposals or truncated outputs give a negative selection score,
below every clean candidate. That bounded test is not hardware authorization.
V6 requires identical configuration, data, scoring contract, Python/dependencies
and GPU backend/hardware between smoke and full run. Evaluation preserves mode,
cache configuration and training RNG. Legacy profiles and exact continuations
retain their original behavior; use an old run's original source revision too.

Current validation and comparison evidence is in
[the model report](model-report.md), including the independent router limitation.
The data/configuration changes have not yet produced a trained v6 model.

## Restore and resume evaluation

Evaluation restoration differs from restoring optimizer state for training.
For inspecting a supplied results ZIP without loading its optimizer/RNG pickles:

```bash
python training/restore_results.py --zip /path/to/results.zip \
  --output artifacts/restored-evaluation --base .local/training/base
```

The new destination must not exist. Member paths/types/duplicates/CRCs are checked;
the verification report records source hashes and cached-base agreement. This is
byte integrity, not external authenticity. Prefer a completed final `full/adapter`.
An incomplete metadata-selected checkpoint requires `--allow-incomplete-checkpoint`
when evaluating and is explicitly diagnostic, never a completed training result.
Exact training restoration instead uses `training/training_resume.py`, as above.

Interrupted HF evaluations support `--resume-from` just like GGUF evaluations:

```bash
python training/evaluate_v2.py --base .local/training/base --backend "$BACKEND" \
  --adapter "$RUN/full/adapter" --suite validation_smoke --prompt-mode chat \
  --max-new-tokens 192 --resume-from "$RUN/new-validation-partial.json" \
  --output "$RUN/new-validation-resumed.json"
```

Keep all original options, identities and numerical backend. Resume verifies the
deterministic prefix and re-scores saved predictions, including incorrect answers;
it does not retry just the failures. Original reports are preserved and outputs
are atomic. Legacy reports lacking identity fields require a fresh evaluation.
`--backend cpu-fp32` is an explicit off-device fallback, not CUDA/NF4 or GGUF parity.
Never mix these backends into one comparison.
