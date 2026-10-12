# Training PINS AI

This is the entry point for training the Qwen3 0.6B MCP assistant. Training runs
off-device on a GPU workstation, Colab, or a future AMD Instinct host. The PINS
plugin owns inference and authorization; training never operates equipment.
All commands below start from the **repository root**. No training runs start
automatically. Start a fresh adapter from the pinned base when changing data.

## Requirements and profiles

| Environment | Python / PyTorch | Dependencies | Configuration | Evaluation backend |
|---|---|---|---|---|
| NVIDIA workstation / Colab T4 | Python 3.11, torch 2.7.1 CUDA 12.6 | `training/requirements.txt` | `training/configs/config-v3.json` (FP16/NF4 QLoRA) | `cuda-nf4` |
| AMD Instinct MI355X, Linux x86-64 | AMD-supported ROCm/PyTorch with gfx950 kernels; example below | `training/requirements-common.txt` | `training/configs/mi355x-lora-bf16.json` (BF16 LoRA) | `gpu-bf16` |

Use Git and Python; Git LFS is needed only for the bundled deployment GGUF files.
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

All training data is in Git and included in this checkout:

- `training/data/authored.jsonl`: reviewed additions to the original text baseline.
- `training/data/gold.jsonl`: text evaluation cases, not training input.
- `training/data/sequence-plans.jsonl`: 20 train-only English/German conversations.
- `training/datasets/mcp-v2/data/train.jsonl`: original 924 synthetic conversations.
- `training/datasets/mcp-v2/data/validation.jsonl`: 583 validation conversations.
- `training/datasets/mcp-v2/data/test.jsonl`: 593 held-out conversations.
- `training/datasets/mcp-v2/data/eval_gold.jsonl` and `hard_gold.jsonl`: final evaluation only.
- `training/datasets/mcp-v2/data/audited/`: schema-corrected splits used by preparation.
- `training/datasets/mcp-v2/data/manifest.json`: immutable original data hashes;
  `contract-audit.json` and `verified-tools-list.json` identify audited contracts.

`training/prepare.py` regenerates the preserved 326-example informational baseline.
`training/prepare_v2.py --curriculum` builds the current MCP curriculum, including
device/safety contrasts and sequence plans. It renders native Qwen tool calls and
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

For Colab, open [notebooks/pins-ai-colab.ipynb](notebooks/pins-ai-colab.ipynb), choose
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
python training/datasets/mcp-v2/scripts/validate_dataset.py
python training/tests/training_tests.py
python training/tests/curriculum_tests.py
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

## Smoke and full training

Choose the environment profile. Linux NVIDIA:

```bash
CONFIG=training/configs/config-v3.json
BACKEND=cuda-nf4
RUN=artifacts/training-runs/qwen3-$(date -u +%Y%m%d-%H%M%S)
DATA=artifacts/training-data/current
```

Linux ROCm uses the same commands below with:

```bash
CONFIG=training/configs/mi355x-lora-bf16.json
BACKEND=gpu-bf16
RUN=artifacts/training-runs/qwen3-mi355x-$(date -u +%Y%m%d-%H%M%S)
DATA=artifacts/training-data/current
```

Run a short GPU smoke, then verify adapter reload on validation cases:

```bash
python training/train.py --base .local/training/base --dataset-mode mcp \
  --data "$DATA" --config "$CONFIG" --max-steps 4 --logging-steps 1 --output "$RUN/smoke"
python training/evaluate_v2.py --base .local/training/base --backend "$BACKEND" \
  --adapter "$RUN/smoke/adapter" --suite validation_smoke --per-language 2 \
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
$Config='training/configs/config-v3.json'
$Backend='cuda-nf4'
$Run='artifacts/training-runs/qwen3-'+(Get-Date -Format 'yyyyMMdd-HHmmss')
$Data='artifacts/training-data/current'
python training/train.py --base .local/training/base --dataset-mode mcp --data $Data --config $Config --max-steps 4 --logging-steps 1 --output "$Run/smoke"
python training/evaluate_v2.py --base .local/training/base --backend $Backend --adapter "$Run/smoke/adapter" --suite validation_smoke --per-language 2 --output "$Run/smoke-reload.json"
python training/train.py --base .local/training/base --dataset-mode mcp --data $Data --config $Config --smoke-run "$Run/smoke" --logging-steps 5 --output "$Run/full"
```

## Monitor loss and locate checkpoints

`$RUN/full/metrics.jsonl` is the flushed loss history;
`metrics-latest.json` is the latest snapshot. `train_loss` is the logging-window
average; `eval_loss_delta < 0` indicates validation improvement. If training loss
falls while validation loss rises, examine overfitting rather than simply training
longer. Final `train_loss_overall` and resumed `train_loss_segment` are labeled
separately. Lower loss does not prove MCP accuracy or authorization safety.

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
two checkpoints subject to Trainer's best/latest handling; they are not an archive
of every save. The completed `full/adapter/` contains the validation-selected best
adapter. `training-run.json` records versions, hashes, configuration and status.

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
  --adapter "$RUN/smoke/adapter" --suite validation_smoke --per-language 2 \
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
output files and the same backend/generation settings. Once the recipe is frozen,
run final `eval_gold` and `hard_gold` for base, completed prior adapter and new
adapter. Never tune data/prompts against those held-out scores.

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
See [plugin build instructions](../plugin/README.md), [gateway safety](../docs/gateway-safety.md),
and [measured Pi results](../docs/pi-sequence-chat-verification.md).

## Verification status

The repository restructuring preserves source data byte hashes and the baseline.
Offline runtime/gateway, data, resume, evaluation-state and ROCm routing tests run
without training. This updated notebook has been syntax/contract checked, not run
on a new Colab allocation. The MI355X image/setup, real GPU smoke, full BF16 training,
BF16 checkpoint restoration on hardware and quantized-model parity are pending
hardware access. No AMD throughput or accuracy is claimed.
