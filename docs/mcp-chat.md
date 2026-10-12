# Plugin-owned MCP chat

The existing PINS AI plugin hosts the local model, the loopback lifecycle API
(4783), and an optional authenticated chat gateway (4784). No separate AI daemon
or MCP server is created. The gateway talks only to the existing PINS MCP endpoint
at `127.0.0.1:4782/mcp` using the service's `PINS_MCP_TOKEN`.

## Installation and connection

Install the four matching Debian packages from the PINS combined unstable
workflow **while idle**. The core package stops/restarts PINS during upgrade.
Restart PINS after all packages finish configuring. The AI package contains the
plugin DLLs, native launcher and pinned llama.cpp runtime, Git LFS model/adapter,
licenses and provenance; there are no installation-time model downloads.

Model assets live under `/usr/share/pins-ai/models`, native executables under
`/usr/lib/pins-ai`, and plugin assemblies under the existing NINA plugin folder.
Per-user configuration remains at `/home/pi/.local/share/pins-ai/config.json`.
Existing model paths/settings and MCP tokens are preserved. Missing
`mcpChatEnabled` is migrated to true; an explicit false remains false.
New installs use the shipped default configuration. Enable MCP and configure
its token in `/home/pi/.config/pins/mcp.env`; the installer generates this only
when absent and adds a systemd EnvironmentFile drop-in without restarting PINS.

In Touch-N-Stars, enable **PINS AI Chat** in plugin settings. Enter the current
rig's AI gateway URL (`http://<rig-host>:4784`) and MCP token. Credentials stay
in memory and are cleared on leaving/disconnection/rig change. HTTP follows the
existing local MCP transport; use a trusted local network or a TLS reverse proxy
for remote access. No daemon default credential is reused.

## API and authorization

All data requests require `Authorization: Bearer <PINS_MCP_TOKEN>`.
`POST /session` with `{}` creates a 15-minute operator session and returns its
random ID, live exposed tools and model status. Subsequent interactive calls
require `X-PINS-Chat-Session`. Maximum eight sessions and eight HTTP requests.

- `POST /chat`: bounded user/assistant input, locale and one to four tool names.
  Native Qwen output is parsed, schema validated, then query results or a preview
  returned. MCP results are retained in bounded server-owned native tool history.
  Request JSON cannot supply tool results, system history or approval state.
- `POST /tool`: exact `{name, arguments}` structured operator request.
- `POST /approve`: exact `{requestId, name, arguments}` from a displayed preview.
  This is separate from chat and only occurs on explicit operator approval.
- `POST /reject`: `{requestId}`; `POST /clear`: `{}` clears conversation/previews.
- `POST /cancel`: `{}` cancels this session's inference and pending previews.
  It **does not stop hardware or cancel MCP jobs**.
- `DELETE /session`: revokes the operator session and pending previews.
- `GET /status`, `GET /diagnostics`: authenticated model state and bounded logs.

Mutations use the existing enforceable ToolGateway and ApplicationOperatorAuthority.
Previews bind exact arguments, host profile/device settings, process instance,
chat session, relevant live equipment state and switch capabilities. Before an
action, the gateway refreshes the contract and state, rechecks authentication,
consumes the decision once and forwards the exact arguments. Expired/replayed/
changed requests fail closed. Force, skip-validation, overwrite and abandon
overrides are refused. There are no mutation retries. An already issued hardware
job remains independent of AI cancellation; its ID must be inspected after errors.

Job polling uses `get_job` only and never repeats the original action. Completion
requires state `succeeded` **and** outcome success; failed, cancelled and abandoned
jobs remain failures. Browser polling pauses in the background. The existing PINS
equipment controls remain available independently of AI.

The compact tool scope respects the 1536-token trained context. The full gateway
allowlist can be browsed as structured requests; unsupported new MCP tools are
hidden pending audit. Oversized history/schema fails visibly instead of truncating
the prompt. Model prose is labeled unverified; structured tool results are shown
directly. The checkpoint remains diagnostic and can still misinterpret results.

## Build and verification

`scripts/build-debian.sh <host-publish> <core-version> <output> <ai-version>` runs
on ARM64, compiles against that host, verifies Git LFS hashes, builds the pinned
generic ARMv8 runtime and produces `pins-plugin-ai`. CI uses LFS checkout and
records the selected AI/frontend/core commits. Its package depends on the exact
core package version. Build dependencies: .NET 10, CMake, C/C++ toolchain, Git,
Git LFS, Python 3 and Debian packaging tools.

Gateway tests include actual simulated JSON-RPC/session transport, separate
approval, replay/state/contract changes and terminal job failures. No default
test sends mutations to equipment. Physical mount/exposure operations, native
Android/iOS devices and active imaging/guiding contention require separate
supervised verification.

## Device-local tests

`mcpChatTrustedOrigins` accepts exact browser origins, such as
`["http://192.168.178.109:1888"]`, for trusted LAN testing without entering a token.
The existing `PINS_MCP_TOKEN` remains server-side. Missing or other origins still
require bearer authentication. An Origin header is not authentication for
non-browser LAN clients: leave this list empty on untrusted networks.
Session checks and exact equipment-action approvals remain enforced.

Test access also requires the caller IP in `mcpChatTrustedClients`, e.g.
`["192.168.178.31", "127.0.0.1"]`. Other LAN clients still need the token.
Listed clients intentionally have operator access without bearer authentication.
