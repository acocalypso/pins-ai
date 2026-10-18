# PINS Local AI

A Linux PINS/N.I.N.A. plugin and an off-device training pipeline for a local,
multilingual Qwen3 0.6B astronomy assistant. The plugin supervises llama.cpp,
connects to the existing PINS MCP server, and enforces exact operator approval.
Touch-N-Stars provides the chat UI in its own repository. This project does not
implement another MCP server or a separately managed AI daemon.

## Start here

- **Train a model:** [training process](docs/training.md) — setup, data,
  Colab/NVIDIA and MI355X/ROCm profiles, smoke tests, full training, loss monitoring,
  checkpoints, evaluation and GGUF export.
- **Build/install the plugin:** [plugin/README.md](plugin/README.md).
- **Current models, comparisons and Pi results:** [model report](docs/model-report.md).
- **Runtime, MCP API and safety:** [plugin guide](plugin/README.md).
- **Contribute:** [CONTRIBUTING.md](CONTRIBUTING.md).

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
  dataset/                   Single dataset: active train, frozen splits/contracts,
                             capability sources and preserved baseline assets
  notebooks/                 Generated Colab workbook
  experiments/               Comparison evidence, historical results and experiment plans
  tests/                     Data isolation, loss, resume and backend checks
docs/                        Training process and consolidated model report
artifacts/                   Ignored generated SFT, runs, checkpoints and exports
.local/                      Ignored environments, base weights and host DLLs
```

All training examples and contract snapshots are versioned under `training/dataset/`.
`test`, `eval_gold` and `hard_gold` are evaluation-only. Existing dataset byte
hashes and the 326-example informational baseline are preserved. Checkpoints and
local training base weights stay outside Git; deployment GGUF files use Git LFS.
New training uses [v8 multilingual reliability data](training/dataset/README.md).
The [model report](docs/model-report.md) separates data validity from runtime
retrieval and model accuracy; start a fresh adapter, not an old checkpoint resume.

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
failed validation. The completed step-1900 candidate did not beat the prior adapters
in the matched runtime comparison and was not promoted. V8 safety/argument contrasts and research-candidate diagnostics are prepared for fresh training; adding examples does not
update existing weights. The completed v7 run selected checkpoint 300 and still
produced unwanted development proposals; it was not deployed. Active-imaging contention and sustained hot
operation remain unverified.

MI355X support is prepared as single-GPU BF16 LoRA on ROCm, with backend and resume
contract tests. The uploaded completed run recorded RX 7900 XT, not MI355X;
actual MI355X/v8 training and performance remain unverified. The Colab T4/NVIDIA
FP16/NF4 recipe is retained.

See [LICENSE](LICENSE), [third-party notices](THIRD_PARTY_NOTICES.md), and dataset
provenance. `docs/` contains two maintained documents; machine-readable historical
evidence lives with training experiments and plugin benchmarks. Superseded
narrative reports remain available in Git history.
