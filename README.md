# PINS Local AI

A Linux PINS/N.I.N.A. plugin and an off-device training pipeline for a local,
multilingual Qwen3 0.6B astronomy assistant. The plugin supervises llama.cpp,
connects to the existing PINS MCP server, and enforces exact operator approval.
Touch-N-Stars provides the chat UI in its own repository. This project does not
implement another MCP server or a separately managed AI daemon.

## Start here

- **Train a model:** [training/README.md](training/README.md) — setup, data,
  Colab/NVIDIA and MI355X/ROCm profiles, smoke tests, full training, loss monitoring,
  checkpoints, evaluation and GGUF export.
- **Build/install the plugin:** [plugin/README.md](plugin/README.md).
- **Contribute:** [CONTRIBUTING.md](CONTRIBUTING.md), [architecture](docs/architecture.md),
  [MCP chat](docs/mcp-chat.md), and [gateway safety](docs/gateway-safety.md).
- **Measured results and current limits:** [Pi verification](docs/pi-sequence-chat-verification.md).

## Project structure

```text
plugin/                      Linux inference plugin and deployment
  src/                       MEF plugin, runtime/MCP gateway, development harness
  native/                    Linux child-process lifetime guard
  scripts/                   Packaging, asset install, Pi probes and benchmarks
  packaging/                 Debian defaults and installation hooks
  models/                    Manifests, licenses and bundled Git LFS GGUF assets
  benchmarks/                Inference cases and historical measurements
  tests/                     .NET contracts/gateway and Linux lifecycle fixtures
training/                    Off-device model specialization
  *.py                       Preparation, training, resume, evaluation and export
  configs/                   NVIDIA FP16/NF4 and ROCm BF16 LoRA profiles
  data/                      Authored baseline, sequence examples and text gold
  datasets/mcp-v2/            Original/audited multilingual MCP data and manifests
  notebooks/                 Generated Colab workbook
  tests/                     Data isolation, loss, resume and backend checks
docs/                        Architecture, safety and dated evidence
artifacts/                   Ignored generated SFT, runs, checkpoints and exports
.local/                      Ignored environments, base weights and host DLLs
```

All training examples and contract snapshots are versioned under `training/`.
`test`, `eval_gold` and `hard_gold` are evaluation-only. Existing dataset byte
hashes and the 326-example informational baseline are preserved. Checkpoints and
local training base weights stay outside Git; deployment GGUF files use Git LFS.

## Clone

For plugin assets:

```bash
git lfs install
git clone https://github.com/acocalypso/pins-ai.git
cd pins-ai
git lfs pull
```

For training-only work, skip deployment weights:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/acocalypso/pins-ai.git
cd pins-ai
```

Training explicitly fetches its own pinned Hugging Face checkpoint. The inference
plugin never downloads models automatically and never installs training packages
on the Pi.

## Current status

MCP chat and Debian packaging work on the tested Pi 5 with 4 GB RAM. The packaged
default uses two CPU threads and a 75°C cutoff. The bundled checkpoint-700 adapter
is **diagnostic**, not a production-quality model: live sequence planning still
failed validation. New sequence examples are ready for retraining; adding them
does not update existing weights. Active-imaging contention and sustained hot
operation remain unverified.

MI355X support is prepared as single-GPU BF16 LoRA on ROCm, with backend and resume
contract tests. Real AMD setup/training and performance remain unverified until
hardware access. The Colab T4/NVIDIA FP16/NF4 recipe is retained.

See [LICENSE](LICENSE), [third-party notices](THIRD_PARTY_NOTICES.md), and dataset
provenance. Earlier dated reports in `docs/` describe historical states; the entry
points above describe the current layout and commands.
