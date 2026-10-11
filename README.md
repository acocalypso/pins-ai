# PINS Local AI

A separate Linux PINS/NINA plugin for local, text-only GGUF inference. The plugin
owns model loading, process supervision, cancellation, streaming and unloading.
This phase covers inference and real Raspberry Pi benchmarks. **MCP is not
available: no tool calls, equipment actions, sequence saving, or frontend UI are
implemented.** Chat is informational; model replies must not be treated as live
equipment state or execution receipts.

See [architecture](docs/architecture.md), [API](docs/api.md),
[benchmark report](docs/benchmarks.md), and [third-party notices](THIRD_PARTY_NOTICES.md).

**Working baseline: Qwen3 0.6B**, selected for off-device specialization. The
[LoRA/QLoRA pipeline](docs/specialization.md) prepares grouped EN/DE data, trains
completion-only adapters, exports Q4_K_M GGUF and gates before/after selection.
The pipeline is implemented; no specialized PINS checkpoint has been trained yet.

The plugin and resource/lifecycle checks are complete. Initial Qwen3 measurements
reached about 25 output tokens/s with two threads, but failed basic arithmetic
and context checks. Qwen3.5 completed short one-thread retries at about 10.5
tokens/s; earlier longer requests hit memory/thermal limits. The third candidate,
FunctionGemma 270M, also loads through the plugin, but is a function-calling
specialist rather than a chat replacement. No production model is selected.
AI remains disabled. The report preserves raw evidence and the two unexpected
reboots that limited benchmarking.

## Layout

- `src/Pins.AI.Plugin`: MEF `IPluginManifest` adapter with real Initialize/Teardown.
- `src/Pins.AI.Runtime`: configuration, supervised llama.cpp process and local API.
- `native/pins-ai-exec.c`: Linux parent-death guard and lower process priority.
- `models/manifest.json`: pinned model revisions, sizes, hashes, licences and runtime commit.
- `benchmarks/cases.json`: fixed EN/DE chat, units, context, negation and unavailable-action cases.
- `scripts/benchmark.py`: actual plugin-stream measurements and host telemetry.
- `tests`: offline contract checks and Linux inference-fixture lifecycle tests.
- `src/Pins.AI.Harness`: optional development-only host, never an installed service.
- `training`: workstation-only data preparation, LoRA/QLoRA, export and evaluation.

## Build against the actual PINS host

Requires .NET SDK 10 on the build workstation. The Pi's self-contained PINS host
does not require an SDK or a separately installed .NET runtime. Obtain these three
DLLs from the target PINS installation (for the tested Pi, `/home/pi/pins`):
`NINA.Plugin.dll`, `NINA.Core.dll`, `System.ComponentModel.Composition.dll`.
Keep them in an untracked directory, such as `.local/host`. Do not use Windows
NINA NuGet binaries. Do not ship copies of host or framework DLLs with this plugin.

```powershell
dotnet build src/Pins.AI.Plugin/Pins.AI.Plugin.csproj -c Release -p:HostDirectory=C:/path/to/host-dlls
dotnet run --project tests/Pins.AI.Tests/Pins.AI.Tests.csproj -c Release
./scripts/package.ps1 -HostDirectory C:/path/to/host-dlls
```

Extract the generated ZIP on the Pi. Install only the two plugin DLLs into the
host's active versioned plugin directory, for the tested installation:

```bash
bash install-plugin.sh . "$HOME/.local/share/NINA/Plugins/3.0.0/pins-ai"
mkdir -p "$HOME/.local/share/pins-ai"
gcc -std=c11 -O2 -Wall -Wextra -Werror pins-ai-exec.c \
  -o "$HOME/.local/share/pins-ai/pins-ai-exec"
chmod 700 "$HOME/.local/share/pins-ai"
# Only while no imaging/guiding session is active:
sudo systemctl restart pins
```

First startup creates disabled `config.json` and a private `api.key` under
`~/.local/share/pins-ai`. PINS continues if model loading fails. A disabled AI
plugin starts its small local control interface without allocating model weights.

## Explicit runtime/model installation

Use the repository checkout on the Pi for the asset installer. It needs Python's
standard library, Git, CMake, GCC/G++, make, OpenMP development support and network
access during installation. Python is **not part of inference**. No containers,
training packages, cloud keys or systemd AI service are needed.

```bash
python3 scripts/install-assets.py --accept-licenses --build-runtime
```

This downloads the selected Apache-2.0 Qwen3 0.6B Q4_K_M model (396,705,472 bytes),
verifies its pinned LFS SHA256, retains licence/provenance, checks out
the exact llama.cpp commit, builds `llama-server` / `llama-bench`, and compiles
the launcher. Weights live outside Git. Allow several GB for native source and
compiler output; build time, heat and peak memory exceed steady inference usage.
The server uses `GGML_NATIVE=ON`: build on the target CPU. The Pi 5 native binary
has not been validated on a Pi 4 and must not be assumed compatible.

For the earlier challenger, add `--models qwen3.5-0.8b-q4` (532,517,120 bytes).
An already verified local GGUF
is reused. Models/runtime are never downloaded automatically by the plugin.
Model installation or licence download failure is explicit. Retain licence and
source dependency notices when copying installed assets to an offline rig.

FunctionGemma is a separate, optional download of 253,127,904 bytes, subject to
the [Gemma Terms of Use](https://ai.google.dev/gemma/terms), not Apache-2.0:

```bash
python3 scripts/install-assets.py --accept-licenses --models functiongemma-270m-q4
```

It is benchmarked only for loading and text transport here. Evaluate real tool
schemas and task-specific tuning before considering it for command routing.

## Configuration

Edit `~/.local/share/pins-ai/config.json` with owner-only permissions. Enable
explicitly and set absolute paths. Example configuration for development testing
(not a validated production profile):

```json
{
  "enabled": true,
  "lazyLoad": true,
  "mode": "lite",
  "modelId": "qwen3-0.6b-q4",
  "modelPath": "/home/pi/.local/share/pins-ai/models/Qwen3-0.6B-Q4_K_M.gguf",
  "modelSha256": "ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a",
  "runtimePath": "/home/pi/.local/share/pins-ai/llama.cpp/build/bin/llama-server",
  "launcherPath": "/home/pi/.local/share/pins-ai/pins-ai-exec",
  "contextTokens": 2048,
  "threads": 2,
  "maxOutputTokens": 128,
  "temperature": 0.2,
  "inferenceTimeoutSeconds": 60,
  "startupTimeoutSeconds": 90,
  "minAvailableMemoryMiB": 768,
  "maxResidentMemoryMiB": 1200,
  "maxProcessSwapMiB": 64,
  "maxTemperatureC": 80,
  "apiPort": 4783,
  "logLevel": "Warning"
}
```

`mode` labels the selected profile; it does not secretly override explicit
threads/context/limits. `modelId` identifies the model but never downloads it.
Use the manifest's second path/hash and `mode:standard` to switch to Qwen3.5.
`logLevel` controls the native diagnostic threshold (Error=1, Warning=2,
Information/Debug=3). Debug intentionally excludes higher native verbosity levels
that can expose prompts. The runtime retains at most 40 bounded native diagnostic
lines; PINS controls its own logger verbosity. No raw conversations are stored.

`/reload` applies enablement and model changes, cancelling and unloading old
inference. Same model stays warm between successful requests. A cancelled or
failed inference is deliberately unloaded. Native crashes are not automatically
restarted in a tight loop; use `/unload` then `/load`, or `/reload` for recovery.
Every load checks the model's full SHA256 before launch. Memory and Pi temperature
are monitored every 500 ms; inference stops at the configured temperature cutoff.
These safeguards reduce risk and do not prove a hard real-time cap or prevent
power-supply failures. Let the device cool before reloading after a thermal stop.

Use the local token without printing it:

```bash
key=$(cat "$HOME/.local/share/pins-ai/api.key")
curl -sS -H "Authorization: Bearer $key" http://127.0.0.1:4783/status
curl -sS -X POST -H "Authorization: Bearer $key" http://127.0.0.1:4783/reload
curl -N -H "Authorization: Bearer $key" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is autoguiding?"}],"locale":"en"}' \
  http://127.0.0.1:4783/chat
```

Token-stream events are documented in [API](docs/api.md). HTTP 200 alone does not
mean generation succeeded; require a `done` event and handle `error`.

## Verification and benchmarking

Local offline checks require no model or hardware. The Linux transport fixture
is not an MCP server; it verifies resource/lifecycle behavior through the installed
plugin without talking to equipment:

```bash
python3 tests/runtime-smoke.py
python3 scripts/benchmark.py --models qwen3-0.6b-q4 --threads 1 \
  --cases units-en --max-output-tokens 32 --case-pause-seconds 30 \
  --output benchmarks/results/pi5-probe.json
```

Run device tests and sustained benchmarks only when a real imaging/guiding session
is inactive. Resolve the tested Pi's cooling/power conditions before any renewed
sustained matrix. The example is a short probe; defaults select the Qwen3 baseline
with one thread. Larger matrices require explicit model/thread options.
Scripts temporarily replace the AI configuration and restore it in
`finally`; abrupt power loss or termination cannot guarantee restoration. Back up
your config first. The benchmark records exact model/config, cold load, first
content token, native token speed, process/host CPU, RSS, swap and temperatures.
Statistical summaries span the fixed mixed prompt set; they are not repeated
identical-prompt service-level guarantees. First content does not establish first
*useful* or correct output. Human quality review is separate.

## Disable, uninstall and rollback

Set `enabled:false`, then call `/reload`. This unloads AI immediately while PINS
and any imaging session continue. `/cancel` or `/unload` also releases current
inference. The native launcher kills inference if its PINS parent dies abruptly.

For uninstall, first disable/unload through the API, then stop PINS only when no
session is active. Remove only the `pins-ai` plugin folder (or restore the backup
printed by the installer), then restart PINS. User models/config stay in
`~/.local/share/pins-ai`; remove them separately if desired. No other plugin,
systemd unit, hardware configuration or system package is modified by this project.

## Deferred

Actual MCP discovery/schema/auth integration, tool accuracy, authorization and
confirmations, UI, Pi 4 validation, active imaging/guiding performance, and
the full GPU training run and tuned-model selection. The development pipeline is
implemented; actual tool supervision awaits real MCP schemas. None is replaced
by invented APIs, mock hardware claims or a new MCP
implementation. See the benchmark report for the measured model limitations.
