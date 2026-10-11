# PINS AI: inference and benchmark phase

Scope agreed for this phase: the MCP server is not available. Build a separate
PINS plugin and benchmark local models; do not invent tools, a transport, hardware
adapters, a fake server, or frontend. The user subsequently selected Qwen3 0.6B
as the working baseline and authorized a development-only specialization pipeline
because measured baseline quality was inadequate. Training stays off the Pi.

## Actual host and reuse

PINS runs on Debian 13 ARM64 with .NET 10 self-contained. It discovers assemblies
exporting `NINA.Plugin.Interfaces.IPluginManifest` through MEF and calls
`Initialize()` / `Teardown()`. Plugins derive from `PluginBase`, use assembly
metadata, and live in `~/.local/share/NINA/Plugins/3.0.0`. The current Pi runs
`/home/pi/pins/NINA`; use its actual host DLLs as compile references rather than
Windows NINA NuGet packages or rebuilding the whole imaging backend.

## Lifecycle

A dependency-light .NET runtime library is shared by the real MEF plugin and a
development-only harness. The plugin owns the runtime and a loopback authenticated
HTTP/SSE interface. Default disabled, lazy loading, one generation, 2048-token
context, two inference/prefill threads, 128 output tokens. Configuration, secrets,
models, and native runtime live outside the replaceable plugin folder.

Use a pinned llama.cpp server subprocess: it keeps native allocations isolated
from PINS and avoids coupling PINS to a managed binding's native ABI and packaging.
There is no demonstrated stable binding here for the installed host. A tiny Linux
launcher sets a parent-death signal before `exec`, so PINS crashes do not orphan
inference. Normal teardown, unload, disable, switch, cancellation, timeout, or
memory-limit violation stop the owned process. Capture bounded diagnostic logs;
never log chat bodies or credentials. Use a private key file and bind 127.0.0.1.
No systemd AI service, public listener, cloud fallback, or runtime Python stack.

## Resource limits

Initial AI RSS ceiling 1200 MiB, minimum available physical memory 768 MiB,
bounded process swap, startup/request deadlines, bounded request/history/response
sizes, and an 80 Celsius temperature cutoff. Monitor actual `/proc` memory and
Pi temperature every 500 ms; stop inference if limits are crossed.
Swap is not part of the RAM budget. Idle PINS/PHD2 coexistence is measurable now;
active imaging/guiding contention and Pi 4 claims require later supervised tests.

## Safety and acceptance

All capability metadata explicitly reports MCP unavailable and tool execution
disabled. This phase exposes informational inference only, never equipment
mediators. Tool accuracy and end-to-end execution latency are NOT measurable yet.

Verify configuration validation, failed starts, bounded concurrency, streaming,
cancellation, unload/switch, disabled behavior, auth, child cleanup on graceful
shutdown and parent death. Build against Pi host references, install in PINS,
verify discovery and lifecycle; benchmark identical EN/DE chat/negation/context
prompts with Q4 candidates at 1/2/3 threads. Record hashes, runtime commit,
quantization, context, cold load, first content token, output speed, RSS/swap/CPU,
PINS baseline and actual outputs. Report semantic quality separately from speed.

## Specialization

See [the development pipeline](specialization.md). Current training data covers
informational EN/DE replies, units, conversation corrections, clarification,
negation and unavailable equipment requests. The loader remains a separate,
unchanged plugin component. Real MCP schema supervision is a gated data-import
stage; no tool definitions, transports or execution paths are invented.

Use scenario-grouped splits and the unchanged held-out benchmark prompts. Training
uses final-completion-only LoRA/QLoRA on the pinned full Qwen checkpoint, followed
by clean-base merge and pinned llama.cpp Q4_K_M export. A candidate can be selected
only after comparable Pi outputs, explicit human review and no quality/resource
regression; scripts never replace or activate the working baseline automatically.
