# Paste this into VS Code Codex

You are implementing **PINS Local AI v2**, a multilingual, local-first Qwen3 0.6B assistant that uses the team's **existing** PINS MCP server. The VS Code workspace root contains all necessary PINS/N.I.N.A. and AI project repositories.

Read `PINS_MCP_AI_Training_v2/CODING_AI_INSTRUCTIONS.md` fully, then read its `README.md`, `data/manifest.json`, and scripts. Follow their constraints. Inspect actual repositories and their `AGENTS.md`/`CLAUDE.md` first, and preserve all existing changes.

**Do not create or rewrite an MCP server.** The MCP server already exists. Compare its current tools and schemas with the snapshot in `data/tool_catalog.json` (derived from commit `14b50c4`), and correct/regenerate training examples as needed. The model loader, inference lifecycle, MCP client and chat gateway belong to a **PINS/N.I.N.A. plugin**; do not replace them with a separate AI daemon. Do not implement Touch N Stars UI during this task.

Build/repair the actual Qwen3 tool-call SFT conversion and GTX 1050 Ti (4 GB) FP16/NF4 QLoRA pipeline. First validate package integrity; verify the tokenizer-rendered Qwen3 tool-call format and assistant-only training loss; run a short GPU smoke test before full training. Preserve the previous 326-example baseline. Train and compare only if the environment and data contracts are sound.

Evaluate the base model, prior adapter, new adapter and (when available) Q4_K_M GGUF on unseen multilingual MCP calls, argument validation, background-job polling, clarification, no-tool requests and authorization behavior. Never train on `test`, `eval_gold` or `hard_gold`. Simulate hardware operations: never send mutating MCP commands to real equipment without explicit authorization. Model outputs are not authorization; implement an enforceable tool gateway.

The target is **Raspberry Pi 5, 4 GB RAM, Linux ARM64, PINS actively running**. Use the existing plugin architecture for model ownership and lifecycle. Provide real code, tests, commands and measured results; identify anything not verified and clearly list next steps. Don't stop at a high-level plan: proceed with safe, testable implementation work.
