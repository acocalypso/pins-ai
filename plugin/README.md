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

See [API](../docs/api.md), [MCP chat](../docs/mcp-chat.md), [architecture](../docs/architecture.md),
[gateway safety](../docs/gateway-safety.md), and [Pi results](../docs/pi-sequence-chat-verification.md).
Default deployments require authentication; the workstation/device allowlist on
the tested Pi is explicit test configuration, not a general authorization bypass.
Model text never authorizes hardware. Equipment mutations need exact operator
approval and fresh server-side state checks.
