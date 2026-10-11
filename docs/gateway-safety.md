# Multilingual gateway safety review

Reviewed against the existing PINS MCP branch at
`e6d13f916cdf1f29a52703e5e7c893a113b2e531` (remote head checked 2026-10-11).
The source-derived catalog has32 tools. No MCP server implementation or equipment
adapter was changed. **MCP is not installed/running on the Pi.** Its `/mcp`
route returned404 during the read-only probe. Every equipment operation below
used an in-memory simulated transport; no telescope equipment was operated.

## Root causes and actual integration boundary

The preserved Colab v2 reports confirm the supplied findings:100/120 correct
tools,89/120 exact arguments; injection0/10, missing approval3/10,
switch arguments2/10 and sequence arguments5/10. These are saved synthetic
evaluation results, not fresh inference or live observations.

The small model learned unreliable authorization and argument-selection patterns.
The original synthetic approval sentence is a training fixture and cannot be
runtime evidence. Lower validation loss did not establish hardware safety.
Device contrasts in the existing v3 recipe also leave the reported safety
classes underrepresented. Names, index choices and cross-field arguments need
deterministic validation even after another fine-tune.

The production path currently remains `AiPlugin -> LocalApi -> LlamaRuntime`,
with informational chat and `ToolExecutionEnabled=false`. It does **not** route
model tool proposals to MCP. The existing separate `ToolGateway` was testable,
but its public string-only `Confirm` and caller-supplied context were inadequate
for a future production integration. It also lacked writable device identity,
state/session revision binding and immediate state refresh. These library
boundaries have been repaired without enabling production equipment access.

## Implemented authorization flow

1. Discover the existing MCP catalog. Unknown names, invalid schema values,
   extra fields, unsafe cross-field combinations and overrides are refused.
2. `PreviewAsync` captures state through an injected **application-owned**
   authority. `McpRigStateReader` obtains structured `get_status` and
   `get_sequence` through the gateway. It validates freshness, sequence agreement
   and live writable switch indices/min/max/step. Device names are not identities;
   authenticated profile/device/session identities must come from PINS itself.
3. Store an immutable exact tool/argument digest, rig/chat IDs, device identities,
   state revision and switch capabilities under a random request ID. Preview
   lifetime is60 seconds;32 pending previews maximum.
4. An explicit authenticated application event calls
   `ApplicationOperatorAuthority.DecideAsync`. The host must supply an actual
   operator-session validator and fresh identity/state capture callback. The
   original structured operator intent must match the preview exactly.
   Approval lives at most30 seconds; denied/pending decisions cannot be upgraded
   on the same request. Model text, MCP prose and chat API access are not sessions.
5. `ExecuteApprovedAsync` consumes the pending request once. Re-discover and
   compare the published tool contract, refresh state, and compare the bound
   identities/capabilities/revision. Check the live authenticated operator session
   **after** these queries, then revalidate arguments, freshness and approval
   immediately before the call. Failures require a new preview; no mutation retries.
6. Forward only the exact copied arguments through `McpClient`. Its equipment
   call method remains internal; production callers must use this gateway.
7. Poll running jobs without repeating the action. Only `succeeded` plus
   `outcome.success:true` confirms job completion. Failed/cancelled jobs remain
   failures even in successful MCP envelopes. Bounded polling can return still
   running; the application must keep that job ID for later status checks.

Default proposal exposure contains only the seven verified queries. Explicit
proposal mode filters stale/running contexts and unavailable switch capabilities,
removes bypass fields, limits waits to5 seconds and exposes numeric coordinates.
Proposal exposure grants no permission to execute.

`OperatorArguments` supports exact Unicode intent comparison, unambiguous named
switch lookup from verified capabilities and explicit decimal-comma numeric
fields. It does not guess coordinates, parse arbitrary natural language or
silently repair a wrong model index/name. Ambiguous numbers/selectors require
clarification. Template/exposure plans are mutually exclusive; draft-only intent
cannot start a sequence; running-sequence mutation remains blocked.

`EmergencyStopAsync` is callable directly by an authenticated host control without
model inference, using an exact explicit approval. Existing PINS emergency
controls remain independent and must remain available if AI/MCP is unavailable.
Normalized stop results mean **stop requested**, never all hardware confirmed
stopped. Partial/failed steps require checking equipment status.

`ToolResultText` provides fixed outcome descriptions in all ten supported
languages. Equipment diagnostic text stays marked untrusted; `next` is removed
from normalized results and is never interpreted as authorization/instructions.
Native-language review is still needed. No grounded-response chat integration
is claimed here.

## Changed files

| File | Purpose |
| --- | --- |
| `src/Pins.AI.Runtime/ToolGateway.cs` | State/identity/approval binding, fresh contract checks, safe exposure, polling/result semantics |
| `src/Pins.AI.Runtime/OperatorAuthority.cs` | Typed approval boundary, exact intent, decimal and named-switch helpers |
| `src/Pins.AI.Runtime/ApplicationOperatorAuthority.cs` | Authenticated explicit decisions and session revocation checks |
| `src/Pins.AI.Runtime/McpRigStateReader.cs` | Structured MCP state/capability reader |
| `src/Pins.AI.Runtime/ToolResultText.cs` | Deterministic multilingual outcome text |
| `tests/Mcp.GatewayTests/Program.cs`, `SafetyRegressions.cs` | Fake transport, positive/negative flow, model-proposal replay |
| `training/curriculum_v3.py`, `tests/curriculum_tests.py` | New train-only safety scenarios and isolation checks |
| `training/create_colab_notebook.py`, `output/jupyter-notebook/pins-ai-v2-colab.ipynb` | Colab defaults to preflight, no automatic training |
| `scripts/pi-gateway-smoke.py` | Isolated ARM64 fake-equipment test with thermal/time bounds |
| `docs/colab.md`, `docs/training-v3.md`, `README.md` | Workflow and measured limitations |

Existing `RigContext` positional fields and informational chat remain compatible.
Legacy `Confirm(string,string,string)` intentionally fails closed: integrations
must migrate to the application authority. Gateways without that authority can
perform queries, but cannot execute mutations. No existing production equipment
workflow depended on this previously detached gateway.

## Verification and measurements

122 gateway checks pass locally and on the Pi.
This covers ten language contexts, missing/denied/pending/expired/mismatched
approval, replay, changed device/session/limits, switch selectors and steps,
Unicode/decimal intent, running sequence protection, all job states, stop failure,
injected MCP text, current contract drift, and an authorized application-to-MCP
state-refresh-to-operation path. Simulated positive operations still execute once.

Saved baseline replay:240 predictions; v2 had66 schema-valid mutating proposals,
base33. All99 were denied without approval; **zero unauthorized simulated actions**.
Invalid proposals were rejected before invocation. This is gateway enforcement
evidence, not improved model accuracy. Natural-language false-completion claims
still need independent review; they are not measured by this replay.

Other checks:24 runtime/API checks;16 training tests with the real pinned
tokenizer;6 v2 tests;4 curriculum tests;3 Colab data-import tests. Original
dataset hashes/split integrity pass. Notebook Python cells parse and outputs are
empty. The real plugin builds against installed PINS host DLLs with0 warnings
and0 errors. No GPU training was started.

The ARM64 self-contained fixture ran on the Pi with PINS active, one advertised
.NET CPU, nice10,30-second timeout and75°C thermal cutoff. The final122-check
run plus both baseline replays took0.617 seconds, peak child RSS66,464 KiB
(64.9 MiB), user CPU0.268 seconds/system0.012 seconds; peak temperature63.9°C.
The checked-in [verification summary](gateway-verification.json) preserves the
measured results. Full logs and binary hashes are recorded in
`artifacts/v3/pi-gateway-smoke.json`. This measures the safety fixture only:
**no model inference, no live MCP integration, no guiding/imaging performance claim**.
One initial attempt failed because `/usr/bin/time` is absent; the script now
uses Python's standard `resource` measurements. No packages or services were
installed on the Pi and PINS was not restarted.

Commands (use the local .NET10 executable on this Windows workstation):

```powershell
.tools/dotnet/dotnet.exe run --project tests/Mcp.GatewayTests -- PINS_MCP_AI_Training_v2/data/verified-tools-list.json
.tools/dotnet/dotnet.exe publish tests/Mcp.GatewayTests -c Release -r linux-arm64 --self-contained true -o .local/pi-gateway
python scripts/pi-gateway-smoke.py --bundle .local/pi-gateway --catalog PINS_MCP_AI_Training_v2/data/verified-tools-list.json
.local/training-venv/Scripts/python.exe tests/curriculum_tests.py
.local/training-venv/Scripts/python.exe training/prepare_v2.py --curriculum --max-tokens 1536 --output artifacts/v3/sft
```

SSH uses the verified `.local/known_hosts` file and an SSH agent/key or
`PINS_SSH_PASSWORD` supplied outside Git. Optional `--v2-report` and
`--base-report` replay saved predictions; no equipment transport is used.

## Separate MCP follow-ups

These findings are documented here; the PINS repository was not modified:

- `StopTools.cs` emits `stopped:true` unconditionally even when a step fails.
  Prefer explicit requested/verified and overall/partial failure fields.
- `NinaObservatory.Status.cs` catches sequence-status exceptions and reports
  `running=false`. Publish unknown/error state instead of a safe-looking false.
  The plugin reader cross-checks `get_sequence` and refuses inconsistent or
  uninitialized state, but agreement alone cannot prove no internal read failure.
- Status exposes display names and writable-list indices, without stable hardware
  IDs/profile revision. Host identities are therefore mandatory. Stable channel
  IDs and a server-side expected-device/state revision would close the residual
  race between status reading and execution. An index can change if a hub's
  writable list is reordered between those operations.
- Authenticated MCP requests carry a bearer token, not this plugin's per-operation
  approval proof. These protections apply to this client path; another authorized
  MCP client is outside the plugin's approval boundary.

## v3 recommendation and live release blockers

Keep v2 adapter/reports and the original326-example baseline unchanged. The
new110 synthetic safety scenarios use different wording, device actions, target
names, indices and numeric values; neither test nor either gold file seeds them.
Prepared v3 contains2669 train conversations/4508 assistant turns and583 validation
conversations/1236 turns,32 tools/ten languages, zero skipped; longest1347/1277
tokens. Existing validation is unchanged. The current context1536 FP16/NF4
rank8/alpha16 recipe remains a proposal until a fresh T4 smoke and full run pass.

Before enabling training, review named-switch paraphrases, duplicated-name
clarifications, negative authorization coverage and native-language wording.
Add a separately authored multilingual acceptance set, including failed/cancelled
multi-step workflows. Keep gold for a final measurement, not iterative tuning.
Model accuracy gains are unmeasured. Colab training is deliberately disabled
by default; loss alone must never promote an adapter.

Live release still requires installing the team's MCP-enabled PINS build;
authenticating it locally; composing the real operator authority with current
profile/device/session revisions; implementing the bounded model proposal and
grounded-response loop in the existing plugin; and merging/exporting/evaluating
the selected adapter as Q4_K_M. The `/chat` endpoint does not become an approval
endpoint. PINS state checks must remain authoritative during the residual
read-to-execute race. Supervised physical tests and concurrent imaging/guiding
performance measurements are pending. The plugin stays informational and disabled
for equipment operations until those requirements are met.
