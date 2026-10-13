# Linux PINS AI plugin

The MEF plugin owns model loading/unloading, its supervised llama.cpp child,
cancellation, resource limits, the MCP client and chat authorization gateway.
The harness in `src/Pins.AI.Harness` is development-only, never an installed daemon.
Training packages and data are not installed on the Pi.

Run commands from the repository root. Build requires .NET SDK 10 and actual
Linux PINS host references: `NINA.Plugin.dll`, `NINA.Core.dll`, `NINA.Profile.dll`
and `System.ComponentModel.Composition.dll`. Copy them into an ignored local
folder, for example `.local/host`. Do not use Windows NINA NuGet stubs or bundle
host DLLs with the plugin.

```bash
dotnet build plugin/src/Pins.AI.Plugin -c Release -p:HostDirectory=/absolute/path/to/host
dotnet run --project plugin/tests/Pins.AI.Tests -c Release
dotnet run --project plugin/tests/Mcp.GatewayTests -- training/datasets/mcp-v2/data/verified-tools-list.json
```

For the complete **Linux ARM64 Debian package**, on an ARM64 build host with
the .NET SDK, Python 3.11+, Git, Git LFS, CMake, a C/C++ compiler, dpkg-dev,
binutils/readelf and OpenMP support:

```bash
git lfs pull
bash plugin/scripts/build-debian.sh /absolute/path/to/pins-publish \
  EXACT_PINS_PACKAGE_VERSION /absolute/path/to/output AI_PACKAGE_VERSION
```

Replace both version labels with the selected Debian versions. This script
verifies model hashes, builds the pinned Pi 5 llama.cpp runtime and launcher,
packages the two plugin DLLs, GGUF assets, licenses and defaults, and produces a
`.deb` plus SHA-256 file. PINS packaging invokes this script from its unstable
package workflow and records the AI Git revision. The script refuses non-ARM64
package builds. Training is independent of this build and can run on x86-64 GPUs.

The bundled base/LoRA GGUF files live under `plugin/models/bundled/` in Git LFS.
`plugin/models/bundled.json` identifies their exact hashes, licenses and pinned
runtime revision. Checkpoint 700 is a diagnostic candidate, not an accuracy gate.
Use `training/export.py` for a new candidate, evaluate it, then deliberately update
the manifest and LFS assets. Never substitute unchecked model files.

Runtime configuration is `~/.local/share/pins-ai/config.json`; packaged weights
are `/usr/share/pins-ai/models`, native binaries `/usr/lib/pins-ai`. PINS loads the
plugin from its active versioned plugin directory. Existing user configuration is
preserved by package installation. Restart PINS only when idle. AI can unload
independently through its local authenticated control API.

See [the model report and Pi results](../docs/model-report.md) and
[the training process](../docs/training.md).
Default deployments require authentication; the workstation/device allowlist on
the tested Pi is explicit test configuration, not a general authorization bypass.
Model text never authorizes hardware. Equipment mutations need exact operator
approval and fresh server-side state checks.

## Model lifecycle and resource limits

The Linux PINS/N.I.N.A. host discovers `IPluginManifest` through MEF and calls
initialization/teardown. Normal teardown, unload, cancellation, timeout or a
resource violation stops the owned llama.cpp child. The native launcher sets a
parent-death signal so a PINS crash does not orphan inference. There is no separate
systemd AI service, runtime Python environment or cloud fallback. One generation
uses the bounded native slot.

Current [packaged defaults](packaging/default-config.json) enable lazy loading,
two inference/batch threads, 1,536 context tokens, 192 output tokens and temperature
zero. Timeouts are 30 seconds for inference and 90 seconds for startup. Limits are
minimum available physical RAM 768 MiB, model RSS 1,200 MiB, process swap 64 MiB and
75°C temperature. Swap does not increase the RAM budget. Existing installed user
configuration is preserved and can differ. Active imaging and sustained thermal
contention have not been accepted by the short idle Pi probes.

Configuration, secrets and weights remain outside the replaceable plugin folder.
Model and optional LoRA paths/hashes are verified before loading. Diagnostics use
bounded native log lines; do not log credentials or private conversations.

## Informational lifecycle API

The loopback API defaults to `http://127.0.0.1:4783`, authenticated with
`Authorization: Bearer <api.key contents>`. The owner-readable key lives at
`~/.local/share/pins-ai/api.key`; it is separate from the MCP token.

| Method/path | Behavior |
| --- | --- |
| GET `/status` | Runtime state, model, owned PID and resource status |
| GET `/diagnostics` | Bounded native diagnostic lines |
| POST `/load` | Verify/load the configured model |
| POST `/unload`, `/cancel` | Stop inference and release the child/weights |
| POST `/reload` | Validate/apply local configuration; API port changes need restart |
| POST `/chat` | Informational compact history; ordered SSE generation |

Informational chat accepts user/assistant messages and locale, not system/tool
messages, execution requests or caller-supplied schemas. Limits are eight messages,
2,000 characters each, 6,000 total and 16 KiB JSON; the last role must be user.
Streaming clients inspect terminal `done` or `error`, not just HTTP 200.
`capabilities`, `token` and optional `metrics` precede it. This API deliberately
reports informational-only capabilities; use the separate gateway for MCP access.
Cancelling causes the next lazy request to load again. Configuration/context
errors are visible, not silently truncated.

## MCP chat and Touch-N-Stars

The plugin also owns the optional gateway on port **4784**. It connects to the
existing PINS MCP server at `127.0.0.1:4782/mcp` with the service's `PINS_MCP_TOKEN`.
Builds consume upstream PINS **unstable**, where MCP was merged; the old PR 198 or
feature branch is not the deployment source.

Install the four matching combined unstable Debian packages while idle. Restart
PINS after configuration completes; the core upgrade may itself restart the
service. The AI package includes DLLs, native runtime/launcher, model/adapter,
licenses, hashes and defaults without installation-time model downloads. The
installer preserves user configuration/tokens, creates missing
`/home/pi/.config/pins/mcp.env` and a systemd EnvironmentFile drop-in, and migrates
missing `mcpChatEnabled` to true while preserving an explicit false.

Touch-N-Stars owns the frontend in its own repository. Enable PINS AI Chat, press
Start, read the experimental warning and choose Continue. Normal chat retrieves
up to four live tools automatically; debug/manual structured requests remain.
The equipment-status shortcut issues a direct read-only query. Thinking feedback
appears during inference and the composer stays outside the conversation scroll
area. Structured results are authoritative; model prose remains unverified.

Default authentication requires `Authorization: Bearer <PINS_MCP_TOKEN>`. An
explicit device-local test exception requires both an exact frontend origin in
`mcpChatTrustedOrigins` and the socket IP in `mcpChatTrustedClients`. The token stays
server-side. Missing/untrusted origins or clients need bearer authentication.
Listed clients intentionally have operator access; this is test configuration,
not protection against arbitrary non-browser clients on an untrusted network.
Frontend credentials stay in memory and clear on disconnect/rig change.

| Method/path | Request and behavior |
| --- | --- |
| POST `/session` | `{}` creates an operator session; returns tools/model status |
| POST `/chat` | Bounded user/assistant history, locale and optional explicit tool scope |
| POST `/tool` | Exact `{name, arguments}` operator request |
| POST `/approve` | Exact `{requestId, name, arguments}` from a displayed preview |
| POST `/reject` | `{requestId}` rejects a preview |
| POST `/clear` | `{}` clears conversation/previews |
| POST `/cancel` | `{}` cancels session inference/previews; does not stop hardware/jobs |
| DELETE `/session` | Revokes session and pending previews |
| GET `/status`, `/diagnostics` | Runtime status plus actual MCP discovery and diagnostics |

Interactive calls require `X-PINS-Chat-Session`. Sessions expire after 15 minutes;
maximum eight sessions and eight HTTP requests. The gateway retains bounded native
tool history server-side; request JSON cannot supply system/tool history or
approval state. Compact context respects the trained token budget; unsupported
tools stay hidden pending audit. Clarification can retain server-owned earlier
intent. Retrieval uses bounded aliases, not guaranteed semantic understanding;
see the measured recall limitation in [the model report](../docs/model-report.md).

## Authorization and job safety

Model output is a **proposal**, never authorization. `ToolGateway`, the
application-owned authority and fresh structured rig state enforce:

1. Live discovery, exact schema/argument validation and cross-field checks.
   AI exposure removes force/overwrite/skip-validation/abandon overrides, exposes
   numeric coordinates and bounds waits to five seconds. Writable switch ranges,
   indices and step sizes must be verified; running-sequence protections remain.
2. Preview binding to exact arguments, host profile/device identities, process,
   chat session, relevant equipment state and switch capabilities. Previews expire;
   new requests supersede earlier previews.
3. A separate authenticated operator decision matching the displayed preview.
   Model text, chat "yes", tool prose and conversation history are not approval.
4. Fresh contract/state/session checks immediately before forwarding the exact
   arguments. Decisions are consumed once; expired/replayed/changed requests fail
   closed. Equipment mutations are not automatically retried.
5. Polling `get_job` without repeating the action. Only `succeeded` plus successful
   outcome confirms completion. Failed/cancelled/abandoned jobs remain failures
   even when the MCP response envelope itself succeeded.

Equipment text stays untrusted and cannot provide authorization/instructions.
After timeout, inspect the recorded job before retrying. AI cancellation does not
cancel an issued hardware job. Stop results mean stop requested, not all hardware
confirmed stopped; inspect partial failures and status. Existing PINS equipment
and emergency controls remain independent of AI availability.

Supported explicit English/German multi-device connection instructions can create
bounded proposals without model inference. Each device needs its own approval;
the next preview waits for confirmed previous-job success. Rejection, failure,
cancellation or profile changes stop the remaining plan and preserve completed
results. This narrow parser is intent routing, not authorization or general
multilingual model accuracy.

Sequence planning uses `add_target` to create DSO containers and Smart Exposure
blocks in the loaded sequence. "300 five-minute exposures" means count 300 and
exposure 300 seconds; omitting filter avoids a filter-change instruction, not a
physical filter removal. Current contracts lack name-to-coordinate lookup and a
new blank sequence-root tool. Obtain verified J2000 coordinates and append/replace
intent; loading/clearing needs its own approval. Adding a target must not start
imaging. The retained model has known failures on this request.

## Verification and troubleshooting

Run the build/test commands above and checks in [CONTRIBUTING](../CONTRIBUTING.md).
Gateway fixtures simulate all equipment mutations. The Linux lifecycle fixture
is `python plugin/tests/runtime-smoke.py`, not runnable on Windows. Real Pi probes
use the installed plugin, never a standalone inference daemon.

Check gateway diagnostics for MCP availability separately from model readiness.
Check the service token/environment and actual endpoint if discovery fails;
check verified model paths, child errors, available RAM and temperature if
inference fails. Frontend Debug exports bounded troubleshooting data; remove
private conversations before sharing. Diagnostics must not expose tokens. Do not
bypass approval, resource cutoffs or schema failures. The
[model report](../docs/model-report.md) records dated observations and open checks.
