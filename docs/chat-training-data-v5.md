# PINS chat training audit — 2026-10-13

Goal: interpret normal chat, extract exact MCP arguments and propose the right
operation through the existing plugin/gateway. This report measures data and
runtime contracts, not a newly trained model.

## Contract source

Fetched `nitr57/pins` unstable at `f85558906`. No `NINA.Mcp` or `NINA.Mcp.Adapter`
file changed since audited commit `6fef6c4bd73a8e1f24a36235dba6cbb9eff3d3d2`.
The existing verified 32-tool snapshot remains authoritative. No hardware or
mutating MCP command was executed. Training schema exposure was additionally
compared with actual `ToolGateway.ExposedTools` using a synthetic fresh idle rig.

## Findings and changes

| Measure | Previous combined curriculum | Active v5 |
|---|---:|---:|
| Conversations | 7,187 | 4,669 |
| JSON supplied in user requests | 4,330 (60.2%) | 694 (14.9%) |
| Newly authored chat/workflow conversations | — | 638 |
| Natural positive tool × language combinations | Not independently enforced | 32 × 10 |
| Tool scopes above four | 147 | 0 |
| Conversations containing different modifying tools | 0 | 4 |

The smaller active corpus deliberately excludes 2,566 bulk JSON argument-copying
rows, 190 calls that exceed the plugin wait limit, 380 repeated device phrasings
and 20 obsolete approval-stage cases. Those source rows remain archived unchanged.
It retains numeric/schema boundaries and asynchronous failure/outcome coverage.
The original 326-example text baseline and original test/gold files are preserved.

Training previously leaked exact expected arguments in system approval hints and
refused calls solely because approval had not yet happened. The plugin actually
generates proposals first. Active train and validation copies now remove those
answer keys, distinguish genuine withdrawal from pending approval and keep exact
operator authorization in the gateway. Validation retains 583 scenario IDs; loss
values under this new projection should not be compared as if the recipe were
unchanged.

All ten languages have plain chat examples for every tool. Additional detailed
English/German cases cover units, synonyms, misspellings, corrections, optional
arguments, explicit denial and informational requests. Workflows cover discovery,
clarification, supplied coordinates, sequence plans without implicit start/save/
clear, connect→poll→cool, filter→poll→capture, cancellation and failure stopping
later actions. Named objects never supply fabricated coordinates. Unsupported
MCP requests remain distinct from capabilities elsewhere in PINS.

Job fixtures use observed/user-provided exact IDs and real five-state results.
Cancellation remains `running` with `cancelRequested:true` until confirmed.
Returned IDs must match the requested ID. Templates, saved sequences and sequence
trees use the source's result shapes. Missing readings are unknown, rather than
assumed disconnected or safe. Tool strings remain untrusted data.

## Verification

Full local preparation uses the pinned Qwen3 tokenizer, native `<tool_call>`
format, assistant-only labels, deterministic duplicate removal and strict context
checks. `artifacts/training-data/chat-v5-check/manifest.json` is the local measured
preparation evidence; it records train/validation token distributions and removed
duplicates. It is generated, not committed training input. `coverage.json` and
`manifest.json` under the dataset directory are checked-in reproducible evidence.

| Final tokenizer preparation | Train | Validation |
|---|---:|---:|
| Conversations | 4,669 | 583 |
| Supervised assistant turns | 6,918 | 1,236 |
| Duplicate assistant turns removed | 1,147 | 0 |
| Maximum prompt plus completion tokens | 1,318 | 1,309 |
| Maximum completion tokens | 139 | 63 |
| Skipped records | 0 | 0 |

The corpus reserves 192 output tokens within the Pi's 1,536-token context.
The final bounded-schema pass measured a maximum 139-token completion, exceeding
the earlier packaged limit of 96 and runtime default of 128. Defaults now allow
192, and bounded legacy tool scopes leave room for that full output budget.
This does not update an existing Pi configuration or prove Pi latency/timeout
acceptance; benchmark a completed candidate conservatively before promotion.

Tests cover exact hashes, reproducibility, current tool/argument coverage,
proposal-stage policy, authentic result shapes, exact job identity, UTF-8,
held-out group/prompt separation, parser scoring and actual gateway schema parity.
Generators never read held-out data. No GPU training was started.

## Retrieval is a separate measured blocker

`training/tools/RouterAudit` runs the real router and gateway validator offline.
It uses only the 320 authored training intents, not held-out answers or inference.
The latest report is `artifacts/router-chat-v5.json`, with input and runtime
assembly hashes, missed requests and per-tool/language results.

Correct tool present: **134/320 (41.875%)**. The gateway accepted all **4,148**
active native calls with no argument/schema errors. The bounded Python schemas
match actual gateway exposure exactly. These checks do not validate device state,
authorization or execution.

| Language | Tool retrieval recall |
|---|---:|
| English / German | 53.125% each |
| Spanish | 50% |
| Chinese | 46.875% |
| French / Japanese | 43.75% each |
| Polish | 37.5% |
| Italian / Dutch | 34.375% each |
| Portuguese | 21.875% |

`list_saved_sequences`, `start_sequence` and `reset_sequence` each had 0/10 recall
on these plain requests. This is development coverage, not unseen accuracy. A
model receives at most four retrieved schemas; training cannot make an omitted
schema available. Production retrieval therefore needs separate improvement and
independent testing before the chat can reliably fulfill arbitrary requests.

## Next validation

Start a fresh base adapter with a matching GPU smoke/reload, not an exact resume
of checkpoint 700 or the previous AMD run. Compare base/prior/new using
`--suite validation_smoke --prompt-mode chat` with identical settings. Frozen
legacy gold comparisons remain available with their legacy prompt mode; they do
not establish current proposal-stage safety or unseen language quality.
Use `--max-new-tokens 192` for deployment-matched chat comparisons. Validation
targets reach only 63 tokens, so add independent rich-argument acceptance cases
to exercise longer proposals before deployment.

Obtain independent native-language review and a new reviewed deployment holdout.
Evaluate retrieval, exact arguments, clarification, denial, dependent progress and
false success claims separately. Then export and compare Q4_K_M, measure Pi
memory/temperature/latency with PINS running, and retain the existing model unless
the candidate passes. No model files or live Pi settings changed in this task.
