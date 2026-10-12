# Checkpoint 700 on Raspberry Pi

Measured on 2026-10-12, Raspberry Pi 5 / 4 GB. This is a diagnostic
checkpoint evaluation, not acceptance of a completed training run.

## Deployment

The existing PINS AI plugin owns the llama.cpp process and loads it lazily.
Optional LoRA path/hash configuration was added and deployed to the plugin.
The base and adapter remain separate; no weights were merged.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Base Q4_K_M GGUF | 484220160 | `9947d4e2402c8a9f51720f5aaa7b3a1fba7ef48693773393cd8204a357ec74c3` |
| Checkpoint 700 F16 LoRA GGUF | 10119776 | `9e20a0c41ce2da610b4bf63affe29c997c491dc29ecad6327133f56decaf608a` |

Conversion/runtime source: llama.cpp `8345f333951c661d166b00e6f9362e553768f292`.
Artifacts and backups are outside Git at
`/home/pi/.local/share/pins-ai-tests/checkpoint700-20261012-131106`.
The original checkpoint and cached base hashes remained unchanged.

Runtime limits: one CPU thread, 1536 context tokens, 48 output tokens,
30-second inference timeout, 75 C thermal cutoff. Probes start below 70 C.
Loaded model RSS was approximately 951 MiB, available RAM 2582 MiB, swap zero.
The model was unloaded after probing.

## Live MCP and inference results

Authenticated MCP initialization, discovery and `get_status` succeeded.
The token is supplied through the service environment; unauthenticated access
returned HTTP 401. Secrets are excluded from reports and Git.

| Probe | Result | Decode tokens/s | Peak temperature |
| --- | --- | ---: | ---: |
| English status request | Native `get_status` call, schema valid, executed successfully | 12.743 | 71.05 C |
| German status request | Native `get_status` call, schema valid, executed successfully | 12.716 | 72.15 C |
| English tool-result follow-up | Semantic failure described below | 11.818 | 73.8 C |

The follow-up used an explicitly labeled summary of the actual tool result.
It said "The PINS system is not running, and no devices are connected."
PINS was running: only the imaging sequence was stopped. This is an
interpretation failure and remains unresolved. One earlier follow-up attempt
was blocked by the conservative starting-temperature check.

The diagnostic script grants only `get_status` with empty arguments, even when
the server exposes mutating tools. The plugin's public chat remains
informational: production chat-to-MCP execution has not been implemented or
verified by this deployment. Direct diagnostic completion access uses the
plugin-owned runtime through trusted SSH; it does not introduce an AI daemon.

## Explicitly authorized camera connection

The user separately authorized connecting the currently selected camera.
Authenticated MCP control was enabled (`PINS_MCP_READONLY=0`) and PINS restarted.
The first connection job failed with a null-reference error while camera
enumeration was still starting. A retry after enumeration, without another
restart, succeeded: job `job-164d3c` and subsequent `get_status` both confirmed
connection to **Camera Sky Simulator for ALPACA @ 192.168.178.31 #0**.
The camera was idle and no exposure was started. No alternative camera was
selected. This was an operator-authorized MCP call, not a model-generated call.

## Source version and remaining work

MCP is now merged into `nitr57/pins` **unstable**, merge commit `6e8c6eea7`.
The branch head observed on 2026-10-12 was
`6fef6c4bd73a8e1f24a36235dba6cbb9eff3d3d2`.
Future builds and contract comparisons must use unstable rather than PR 198.
The running Pi deployment still comes from the earlier build
`64d1fa66c6b2b0ddd7de1aa7bcfaab76e9b4a151`; it has not been upgraded in this test.
Current unstable changes include job handling, equipment behavior and tool
contracts, so these results do not establish compatibility with that new build.

Next steps: deploy current unstable, regenerate/compare its live schemas,
update training/evaluation contracts (including cancellation/abandon semantics),
implement the enforceable plugin chat gateway, and run unseen multilingual
evaluation with simulated mutations. Broader quality, background-job model
polling and authorization behavior remain unverified here.

Validation: plugin release build succeeded with zero warnings/errors;
28 runtime checks and 4 read-only probe authorization tests passed.
Raw deployment, probe, feedback and camera job evidence is retained under
`.local/pi-checkpoint700/` outside Git.
