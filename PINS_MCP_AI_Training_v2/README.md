# PINS Local AI — Multilingual MCP Tool-Calling Training Package (v2)

**Synthetic example dataset** for **Qwen3 0.6B LoRA/QLoRA**, built from the actual `NINA.Mcp` tool names and schemas in [`nitr57/pins` commit `14b50c4`](https://github.com/nitr57/pins/commit/14b50c4e09ef3904f5c8db5a29f05eb266e863a5).

> **Scope:** The linked commit is a **WIP snapshot**, not proof the MCP adapter is currently wired into your running Pi. All equipment responses in this package are **invented simulation fixtures**, never measurements from real equipment. **Before training/deployment, compare `data/tool_catalog.json` against the actual `tools/list` response from your deployed server, update it, and regenerate.** This package was validated structurally, **not evaluated against a running Qwen3 tokenizer or physical hardware**.


## Instructions for VS Code / Codex coding agent

- **`COPY_INTO_VSCODE_CODEX.md`** — short prompt to paste into a VS Code coding-agent chat.
- **`CODING_AI_INSTRUCTIONS.md`** — full implementation brief covering the existing MCP server, Qwen3 SFT/QLoRA, 10-language tool calling, safe evaluation, and the PINS/N.I.N.A. plugin-owned Pi inference runtime.

Both are included in the ZIP. **Start with the short prompt, and keep this whole folder inside your VS Code workspace** so the agent can read the local data and scripts.

## Contents

| File | Purpose |
|---|---|
| `data/train.jsonl` | Training conversations with native-style assistant tool calls and tool feedback |
| `data/validation.jsonl` | Held-out template-family validation conversations |
| `data/test.jsonl` | Held-out template-family test conversations |
| `data/eval_gold.jsonl` | First-action expected tool and arguments derived from test conversations (not for training) |
| `data/hard_gold.jsonl` | **Independent 120-case** manually authored challenge set across ten languages (not for training) |
| `data/tool_catalog.json` | All **32** tool names, parameter schemas, annotations/policy metadata, cross-field rules, source links |
| `data/manifest.json` | Counts, language distributions, input hashes and provenance |
| `scripts/tool_catalog.py` | Editable source catalog; modify after changes to real MCP schemas |
| `scripts/build_dataset.py` | Deterministic dataset generator |
| `scripts/make_hard_gold.py` | Builds the separate multilingual challenge set |
| `scripts/validate_dataset.py` | JSON, name, schema, permission, tool-result, split and hash validation |
| `scripts/prepare_qwen3_sft.py` | Convert conversations into **assistant-only** tokenizer-generated prompt/completion pairs |

## Language coverage

**English (en), German (de), French (fr), Spanish (es), Italian (it), Portuguese (pt), Dutch (nl), Polish (pl), Japanese (ja), Simplified Chinese (zh)**.

These language examples help teach instruction-to-tool mapping across languages; **they do not guarantee fluency or robust execution in all ten languages**. Have native speakers review phrasing, especially Japanese/Chinese/Polish, and keep independent multilingual evaluations. Qwen3 is multilingual already; the goal is teaching the PINS domain and correct tool selection without degrading its existing language ability.

## Tasks represented

- **Read operations:** `get_status`, `get_recent_images`, `get_sequence`, `list_saved_sequences`, `list_target_templates`, `get_job`, `list_jobs`.
- **Connections:** `connect_device`, `disconnect_device` (real device names from `DeviceKinds.All`).
- **Mount:** `slew_mount` (**J2000 RA hours**, Dec degrees), `park_mount`, `unpark_mount`, `home_mount`, `set_tracking`.
- **Imaging:** `capture_image`, `cool_camera`, `warm_camera`.
- **Focus/filter/guider:** `move_focuser`, `run_autofocus`, `change_filter`, `start_guiding`, `stop_guiding`.
- **Switches:** `set_switch` (value validated against actual hardware's live range).
- **Sequence:** `load_sequence`, `add_target`, `clear_targets`, `save_sequence`, `start_sequence`, `stop_sequence`, `reset_sequence`.
- **Stops/jobs:** `stop_all`, `cancel_job`; long jobs plus `get_job` polling.
- **Non-actions:** no tool access, refusal to guess coordinates or device status, ambiguous exposure plan, 'do not start', no trusted approval, prompt-injection instructions.
- **Grounded chat:** answer in the requested language after mock status/HFR output, plus a small exposure-math set.

## Critical safety and data-contract notes

1. **The model is not your authorization system.** The **PINS AI plugin gateway** must enforce hardware limits, sequence state, confirmation, power controls, equipment status, and explicit authorization *outside* model-generated text. The training examples' `Trusted gateway authorization...` system line is only a fixture to teach the decision boundary; it is not a secure token or proof of approval.
2. **Do not expose hardware operations directly to the model with unrestricted server privileges.** This WIP MCP commit has **optional** bearer auth, forceable sequence guards, and other risks previously identified. Deploy loopback by default; fix server-side safety before connecting real equipment.
3. **`stop_all` is a request, not proof equipment stopped.** Train the UI to check status; the commit's `stopped` field is not sufficient evidence. Emergency stop must remain immediately accessible even when other operations need approval.
4. **Do not guess celestial object coordinates.** `add_target` and `slew_mount` require verified J2000 RA/Dec. There is no object-name-to-coordinate lookup tool in this commit. Name-only requests ask the user for coordinates or require a separately verified catalog integration.
5. **MCP job results are nuanced.** A `tools/call` with `isError:false` can return a job with `state:failed`; a `running` job does not mean a hardware action completed. Track `job.id`, poll `get_job`, and inspect both `job.state` and `job.outcome.success`.
6. **Tool messages are gateway-normalized fixtures.** In production, PINS MCP may provide `structuredContent` plus text content and `isError`. The AI gateway should extract/normalize this to the same compact JSON before feeding it to the model; do not assume the raw HTTP envelope matches this dataset.
7. **Cross-field constraints matter:** `add_target` requires exactly one of `template` or `exposures`; `change_filter` requires exactly one of `name` or `position`. Do not set `skip_validation=true` automatically or allow `force=true` to bypass safety policy.
8. **Train/validation/test are not fully independent in language style.** Splits are separated by scenario/parameter variant, not by intent: they share related translated templates and can overestimate generalization. Use `hard_gold.jsonl`, plus new human-written realistic requests and live-simulator tests, as a stronger benchmark.
9. **Do not merge the old 326 training examples blindly.** Some explicitly train 'tools unavailable' refusals even in situations where your new MCP tools exist. Keep such examples **only** with tools absent in context and balance with corresponding callable cases.

## Validation

Run from the root folder (Python 3.10+; `jsonschema` required):

```bash
python -m pip install jsonschema
python scripts/validate_dataset.py
```

To re-generate after editing the generator/catalog:

```bash
python scripts/build_dataset.py
python scripts/make_hard_gold.py
python scripts/validate_dataset.py
```

`build_dataset.py` re-creates train/validation/test, catalog and manifest deterministically. `hard_gold.jsonl` is regenerated separately and must be kept out of train/validation.

## How to use with Qwen3 0.6B QLoRA

**DO NOT** train on the raw `messages` as plain JSON text. It would teach the model to emit JSON conversation records instead of proper Qwen3 tool-call tokens.

The recommended route is to use the **actual Qwen3 tokenizer chat template**. `scripts/prepare_qwen3_sft.py` turns each assistant tool-call or final reply into one `prompt`/`completion` pair, preserving system, user and simulated tool output as **context only**.

Inside your CUDA-capable GPU training environment:

```bash
# Existing working QLoRA/Transformers environment recommended.
python scripts/prepare_qwen3_sft.py --dry-run
python scripts/prepare_qwen3_sft.py --model Qwen/Qwen3-0.6B --max-tokens 1024
```

This makes `data/sft_train.jsonl`, `data/sft_validation.jsonl`, `data/sft_test.jsonl`. Each record contains a `prompt` string and an assistant-only `completion` string. A compatible TRL setup should train on **completion tokens only**, not on prompt/tool-output tokens. For example, use a prompt-completion dataset with `SFTTrainer` and `completion_only_loss=True` where your installed version supports it. **Inspect the rendered tokens of several tool calls and their endings before launching a long QLoRA job.** Qwen3 `enable_thinking=False` and its tool-call syntax must match the model you will deploy. The exporter deliberately fails if the chat-template assistant prefix doesn't align; do not silently bypass that validation.

**GTX 1050 Ti / 4 GB VRAM:** use FP16 (not BF16), NF4 QLoRA, LoRA r=8 as a starting point, microbatch 1, gradient accumulation 8–16, gradient checkpointing, and measured token budgets. 1024 tokens can still OOM; the exporter can filter over-length turns instead of corrupting JSON/tool calls through truncation. **Do not interpret a low validation loss as MCP accuracy.**

Then merge LoRA adapter -> quantize to GGUF Q4_K_M -> benchmark on Pi 5 *while PINS is running*. Re-run the gold tests against the quantized model; quantization may change tool-call accuracy.

## Evaluation recommendations

Measure at least:

- **Tool choice accuracy** (`first_tool`, including *no call* for clarification).
- **Argument exact match**, normalized numeric values, device enums and schema validation.
- **Safety**: no unapproved physical action, no fake 'completed' when a job is still running, no bypass via user instruction.
- **Response language** and whether every factual assertion is grounded in a tool result.
- **Multi-step completion**: called tool -> job running -> `get_job` -> final answer only after completed.

Use `hard_gold.jsonl` as a **never-trained** evaluation set. **Don't tune repeatedly against it**: that eventually leaks it into the model selection process. For a release candidate, write additional blinded cases with your team and run MCP simulator/hardware-in-the-loop tests with operator supervision.

## Provenance and licensing

The synthetic text examples and scripts in this package are released as **CC0-1.0** examples for use and adaptation (no real device data or secrets). They were **hand-authored and programmatically varied for this response**. The upstream PINS repository and `NINA.Mcp` implementation retain their own existing license (MPL-2.0); this package does not change it.

**Technical source:** [`NINA.Mcp/Tools` at commit 14b50c4](https://github.com/nitr57/pins/tree/14b50c4e09ef3904f5c8db5a29f05eb266e863a5/NINA.Mcp/Tools) and associated `NINA.Mcp/Observatory/IPinsObservatory.cs` / MCP transport code.
