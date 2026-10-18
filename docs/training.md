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
| Colab T4 | Hosted kernel Python and working preinstalled CUDA PyTorch | `training/colab_environment.py` installs missing/incompatible training dependencies only | `training/configs/config-v8.json` (FP16/NF4 QLoRA) | `cuda-nf4` |
| NVIDIA workstation | Versioned example: Python 3.11, torch 2.7.1 CUDA 12.6 | `training/requirements.txt` | `training/configs/config-v8.json` (FP16/NF4 QLoRA) | `cuda-nf4` |
| AMD Instinct MI355X, Linux x86-64 | AMD-supported ROCm/PyTorch with gfx950 kernels; example below | `training/requirements-common.txt` | `training/configs/mi355x-v8-lora-bf16.json` (BF16 LoRA) | `gpu-bf16` |

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
- `training/dataset/sources/comprehensive/train.jsonl`: historical train-only coverage of the original 32
  MCP tools and ten languages, including arguments, boundaries, job
  outcomes, authorization and unsupported requests. See its [coverage guide](../training/dataset/sources/comprehensive/README.md).
- **Current default:** `training/dataset/train.jsonl`: the v8 multilingual MCP48
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
existing Drive data, reuses the kernel's Python and installed CUDA PyTorch, prepares data,
and runs smoke/reload checks before full training. Defaults are `TRAINING_MODE =
'fresh'`, `RUN_TRAINING = False`, and final gold evaluation disabled. Deliberately
enable training after reviewing the configuration. Outputs/checkpoints are under
`/content/drive/MyDrive/pins-ai/runs/<RUN_NAME>/`. Existing Drive directories/ZIPs
using the old `PINS_MCP_AI_Training_v2/data` layout remain supported, with identical
byte-hash checks. Code is never loaded from Drive. Regenerate the notebook with
`python training/create_colab_notebook.py` after changing its generator.

The Colab setup does **not** create a Python 3.11 environment or reinstall PyTorch,
CUDA, Triton or NVIDIA libraries. It resolves the repository's training package
pins against the existing stack and installs only required changes. A resolver
plan that would replace the GPU stack is rejected. Setup records the interpreter,
before/after package inventory and install plan in `colab-environment.json` under
the new Drive run directory. Imports and small NF4 GPU kernels run in a fresh
subprocess, avoiding stale packages already imported by the notebook kernel.
Use the latest compatible hosted runtime; no particular future Colab allocation
is assumed to work without preflight and the matching model smoke test.

For a fresh run, keep `TRAINING_MODE = 'fresh'`, select T4, run setup/data cells,
then explicitly enable `RUN_TRAINING` and run smoke/reload followed by full
training. An interrupted run still requires its original configuration, data
and dependency versions. Reusing a newer runtime does not make a different
dependency stack an exact continuation; the resume checks remain enforced.

Base weights are reused from `.local/training/base` when valid, otherwise restored
from `/content/drive/MyDrive/pins-ai/base-models/<pinned-key>/`. A first run or
invalid/missing cache still downloads the pinned base once, then caches it on
Drive. Each reuse checks the pinned weights SHA256 and file provenance; training
uses a VM-local copy. This trades Drive space and verification/copy time for
avoiding another network download. `base-cache.json` records the source. Cached
weights are separate from run checkpoints and never added to Git.

These changes have CPU/mock test coverage; execution and setup timings on a real
current Colab T4 allocation remain **not verified**. The notebook's preflight and
short GPU smoke are the next checks before enabling a full v8 run.

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
  training/dataset/contracts/tools-list.json artifacts/router-chat-v8.json
python training/tests/chat_dataset_tests.py
```

Without that report, the Python parity fixture explicitly skips its actual-.NET
check; it must not be reported as a measured gateway audit. CI runs the audit.

## Smoke and full training

Choose the environment profile. Linux NVIDIA:

```bash
CONFIG=training/configs/config-v8.json
BACKEND=cuda-nf4
RUN=artifacts/training-runs/qwen3-$(date -u +%Y%m%d-%H%M%S)
DATA=artifacts/training-data/current
```

Linux ROCm uses the same commands below with:

```bash
CONFIG=training/configs/mi355x-v8-lora-bf16.json
BACKEND=gpu-bf16
RUN=artifacts/training-runs/qwen3-mi355x-$(date -u +%Y%m%d-%H%M%S)
DATA=artifacts/training-data/current
```

Run a short GPU smoke, then verify adapter reload on validation cases:

Smoke validation/checkpoint saving runs once at its final optimizer step. It still
checks the complete configured generated-validation sample. Full training retains
its configured 100-step interval; smoke quality is not a release accuracy claim.

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
$Config='training/configs/config-v8.json'
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
four checkpoints for v7, subject to Trainer's best/latest handling; they are not an archive
of every save. The completed `full/adapter/` contains the validation-selected best
adapter. `training-run.json` records versions, hashes, configuration and status.
`bestCheckpointScore` is the selected score, not a loss; final `metrics.eval_loss`
remains the measured loss at the selected checkpoint. Existing v3/legacy AMD
profiles retain their original loss selection and retention for exact resumes.
Changed v7 data requires fresh training from the base. A lower-rate AMD ablation
is available in `training/configs/mi355x-v7-conservative-lora-bf16.json`; it is experimental,
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

In Colab, explicitly select the original run in `RESUME_SOURCE`; the new
`RUN_NAME` is only the destination. For the run started on October 14, use
`DRIVE_FOLDER / 'runs/qwen3-pins-v7-colab-20261014-191543'`. Set
`TRAINING_MODE = 'resume'` and deliberately enable `RUN_TRAINING`. Re-run setup
with a new destination name after changing the source; do not reuse an earlier
session's restored checkpoint variable. The notebook writes the selected run's
exact saved recipe to `RUN_ROOT/resume-config.json`, instead of forcing today's
fresh-run preset onto it. Prepared data, base, optimizer and schedule checks
remain enforced. A different-data error means select the correct source run or
recover its original source revision/data; never remove that check.

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
the original 583 validation scenario IDs plus independent new-tool development cases, and does not run gold suites. The default legacy
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

Historical v6 contains 5,319 conversations, including 650 new multilingual parameter,
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
The current v7 profile follows upstream unstable `c4d8f5a` with 48 tools. It adds
272 authored conversations, removes 12 obsolete capability refusals from the
active projection, and preserves all frozen source/evaluation bytes. It has
5,579 conversations and 7,860 unique prepared train turns; zero skipped examples.
Development validation combines 583 original scenarios with 44 independent new
English/German cases (627 conversations). New-tool native requests cover all ten
training languages; their development supplement covers English/German only.
At the 2026-10-14 update no trained v7 model or new deployment was claimed.
The completed v7 run and current v8 code-only status are documented below.
Use `config-v8.json` for T4/NVIDIA or `mi355x-v8-lora-bf16.json` for supported ROCm,
a fresh pinned base, newly prepared data and a new matching smoke run. Do not
resume an old adapter as an exact continuation under the new contracts.

To refresh contracts after a future upstream update, check out the intended
PINS commit in an adjacent `../pins` source repository, then run:

```bash
dotnet run --project training/tools/UpstreamCatalog -- artifacts/new-mcp-tools.json
python training/update_mcp_catalog.py --source ../pins --tools-list artifacts/new-mcp-tools.json
python training/build_comprehensive_dataset.py
python training/build_chat_dataset.py
python training/tests/upstream_contract_tests.py
```

The catalog exporter compiles the existing MCP dispatcher and only invokes
`tools/list` with an equipment proxy that throws on hardware access. It is an
off-device .NET console development tool; its executable output is never shipped
in the Linux ARM64 Debian plugin. If PINS sources are elsewhere, pass
`-p:PinsSource=/absolute/path/to/pins` to `dotnet run` and the same `--source` to
the updater. Training later needs only the committed snapshots, not PINS source.
New tools or schema semantics still require reviewed authoring, regeneration and
tests; exporting a catalog alone does not create usable training examples.

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


## V8 reliability experiment and historical v7 migration (2026-10-18)

The completed v7 Colab run stopped by early stopping at 700/1,476, not by a
crash. `full/adapter` is the safety-ranked checkpoint 300; its bytes match that
save. The in-memory step-700 loss is 0.153557, while the selected checkpoint's
loss is 0.254906. Overall and resumed-segment losses are different scopes. See
[the model report](model-report.md) and machine-readable v7 analysis.

V8 is a **fresh-base experiment**, never a continuation of historical v7. Use
`config-v8.json` for NVIDIA FP16/NF4 or `mi355x-v8-lora-bf16.json` for supported
ROCm BF16. Learning rate, patience, seed and batch settings remain unchanged;
this is a versioned data/selection change, not a cosmetic patience increase.
The Colab generator/notebook default to v8 fresh mode with training disabled.
To continue an interrupted old v7 run, use its original Git revision, config,
prepared data and smoke; current code explicitly rejects different scoring/data
contracts. A completed early-stopped run is read-only, not an exact resume source.

The 627 development conversations and 146 selected inputs/prompts are frozen
from v7. Historical case fingerprints preserve old evaluation provenance labels;
the SFT validation loader still labels its copies validation. No validation row
is optimized as training. V8 adds 460 independently authored train-only contrasts,
for 6,039 conversations / 8,320 unique prepared turns. No current model improvement
is implied. Native-speaker and semantic review are pending.

`v8-safety-first-research-1` keeps the selected full adapter safety-ranked:
clean candidates outrank violations; unwanted proposals/truncation dominate,
then invalid outputs, then macro exact accuracy. Historical v7 scores remain
available as `historical_v7_selection_score`; their scale/meaning is unchanged.
The exported adapter is labeled research, never automatically production eligible.
Production review additionally requires zero reviewed critical unwanted proposals,
no unauthorized simulated execution, no significant schema regression, reviewed
independent holdout success and quantized/runtime parity. Unknown evidence fails.

Early stopping now resets when either safety rank OR macro accuracy improves by
more than the configured threshold. Its state is exported/restored in checkpoint
JSON. It uses the same fixed dev suite; future holdout never affects selection.
`research-candidates/checkpoints/` retains separate best-safety and best-accuracy
adapter/config/state copies, outside ordinary checkpoint rotation. A full Pareto
report is kept in `research-candidates/selection.json`; only two unique weight
copies are retained. Copies have no optimizer state and cannot be resumed as full
checkpoints. Exact interrupted-v8 resumes preserve the reports and research copies.

Checkpoint reports contain per-language/tool/category counts, exact rates and
Wilson intervals, separate invalid no-tool/required-tool outputs, argument errors,
critical-category metrics and generation/scoring wall time where measured. These
are proposal metrics; they do not measure physical unauthorized operations.
Optional/default omission stays strictly separate from explicit arguments; no
side-effect/default equivalence is silently normalized.

Analyze existing reports (stdlib only, source files stay read-only):

```bash
OLD=artifacts/training-runs/qwen3-pins-v7-colab-20261018-123322
python training/analyze_generation_validation.py \
  "$OLD/full/generation-validation"/step-*.json \
  --output artifacts/v7-analysis-new.json --csv artifacts/v7-analysis-new.csv
python training/audit_split_isolation.py --output artifacts/v8-isolation-new.json
```

The leakage audit checks scenario groups across languages, exact prompts and
numeric-normalized near-text matches for new contrasts. It is a heuristic, not
proof against semantic/translation leakage. `holdout-v8.jsonl` is a separately
authored frozen starter with 30 critical no-tool and 10 positive query cases
across ten languages. It is deliberately **not a complete release benchmark**:
native/domain review and broader independent tool coverage remain required.
Never regenerate it to repair failures or import it into a training generator.

After the existing CUDA/ROCm setup above, run these **manually on an authorized
GPU host**. None was run as part of this code task. Prefer the existing Colab T4
16 GB allocation for NF4; a 4 GB Pascal card may not fit the full context. AMD
requires the documented supported ROCm/BF16 stack, not bitsandbytes. A matching
new smoke is mandatory, even when older recipes trained successfully.

```bash
CONFIG=training/configs/config-v8.json
BACKEND=cuda-nf4
# On ROCm instead:
# CONFIG=training/configs/mi355x-v8-lora-bf16.json
# BACKEND=gpu-bf16
DATA=artifacts/v8/sft
RUN=artifacts/training-runs/qwen3-v8-$(date -u +%Y%m%d-%H%M%S)
python training/prepare_v2.py --curriculum --max-tokens 1536 --output "$DATA"
python training/train.py --base .local/training/base --dataset-mode mcp \
  --config "$CONFIG" --data "$DATA" --max-steps 4 --logging-steps 1 --output "$RUN/smoke"
python training/evaluate_v2.py --base .local/training/base --backend "$BACKEND" \
  --adapter "$RUN/smoke/adapter" --suite validation_smoke --prompt-mode chat \
  --per-language 2 --max-new-tokens 192 --output "$RUN/smoke-reload.json"
python training/train.py --base .local/training/base --dataset-mode mcp \
  --config "$CONFIG" --data "$DATA" --smoke-run "$RUN/smoke" --output "$RUN/full"
python training/evaluate_v2.py --base .local/training/base --backend "$BACKEND" \
  --adapter "$RUN/full/adapter" --suite validation_smoke --prompt-mode chat \
  --max-new-tokens 192 --output "$RUN/development.json"
# Only after freezing the model/recipe; never tune training on these predictions:
python training/evaluate_v2.py --base .local/training/base --backend "$BACKEND" \
  --adapter "$RUN/full/adapter" --suite future_holdout --prompt-mode chat \
  --max-new-tokens 192 --output "$RUN/future-holdout.json"
```

Repeat the same frozen suites/options for base and prior adapters, and for the
retained diagnostic accuracy checkpoint. Historical v7 full/adapter remains
step 300, not the latest step or a production-approved model.

For **matched separate runtime LoRA** parity, use the pinned llama.cpp checkout
and converter environment described above. This differs from merged Q4 export:

```bash
LLAMA=.local/llama.cpp
.local/converter-venv/bin/python "$LLAMA/convert_lora_to_gguf.py" \
  --base .local/training/base --outtype f16 --outfile "$RUN/runtime-lora-f16.gguf" "$RUN/full/adapter"
cmake --build "$LLAMA/build" --target llama-server -j 2
SERVER="$LLAMA/build/bin/llama-server"
git lfs pull --include='plugin/models/bundled/*.gguf'
BASE=plugin/models/bundled/qwen3-pinned-base-Q4_K_M.gguf
python training/evaluate_gguf.py --model "$BASE" --lora "$RUN/runtime-lora-f16.gguf" \
  --server "$SERVER" --llama-source "$LLAMA" --suite validation_smoke --prompt-mode chat \
  --max-new-tokens 192 --output "$RUN/gguf-development.json"
python training/evaluate_gguf.py --model "$BASE" --lora "$RUN/runtime-lora-f16.gguf" \
  --server "$SERVER" --llama-source "$LLAMA" --suite future_holdout --prompt-mode chat \
  --max-new-tokens 192 --output "$RUN/gguf-future-holdout.json"
```

Build `llama-server` on Linux at the pinned commit before those commands. The
GGUF evaluator uses two threads, context 1,536, output 192 and no equipment calls.
Repeat with the exact same Q4 base and prior/base controls; do not mix merged-Q4
results with runtime-LoRA measurements.

Future Pi checks require separate deployment review/authorization; do not install
anything automatically. For an existing reviewed Pi profile, the following
read-only diagnostic can query only equipment status (set `PINS_SSH_PASSWORD`
locally and verify the host key first):

```bash
python plugin/scripts/pi-mcp-readonly-probe.py --host 192.168.178.109 --user pi \
  --render-python .local/training-venv/bin/python \
  --checkpoint "$OLD/full/adapter" --output artifacts/v8-pi-readonly-new.json
```

That probe uses the Pi's already configured model; it is **not** evidence of new
weights or production chat binding. Router audit is separate. Pi promotion still
needs explicit model/config review, plugin ownership, exact approval regressions,
RAM/swap/temperature limits (75 C cutoff), latency and active-imaging contention
checks. None of these future GPU, GGUF or Pi commands passed merely by being listed.
