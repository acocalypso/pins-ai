# Workspace map and boundaries

- `plugin/` owns Linux PINS inference, native lifecycle, MCP client/gateway,
  deployment models, packaging, benchmarks and runtime tests.
- `training/` owns off-device preparation, all training/evaluation data, GPU
  profiles, notebooks, checkpoint restoration, evaluation and export.
- Read the corresponding README and `CONTRIBUTING.md` before editing.
- Commands run from the repository root; training modules are CLI scripts with
  shared imports from `training/common.py`.
- Preserve dataset byte hashes (`.gitattributes` protects the immutable data).
  Never train on test/eval_gold/hard_gold or treat synthetic state as hardware proof.
- Do not create an MCP server, a separate AI daemon, or move training onto the Pi.
  The existing MCP server remains authoritative; model output is never approval.
- No full training or hardware mutation should start as a side effect of setup,
  docs, validation or a source refactor. Require a measured matching GPU smoke for
  full training, and exact operator authorization for real equipment mutations.
- CUDA NF4 and ROCm BF16 LoRA are separate recipes. Keep optional bitsandbytes out
  of the ROCm dependency path; do not claim AMD tests without AMD hardware.
- Update notebook generator, CI and the consuming PINS package workflow when
  changing source paths. Keep legacy Google Drive data archives readable without
  importing code from Drive or bypassing byte-hash verification.
