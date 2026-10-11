# Raspberry Pi benchmark evidence

All three candidate models loaded through the installed PINS plugin. Qwen3 completed 27 requests.
Qwen3.5's initial conservative probes stopped on resource limits; subsequent
short retries completed. **No production model is selected. AI remains disabled.**

## Device and method

Raspberry Pi 5 Model B Rev 1.1, Debian 13.7 Trixie ARM64, 4049 MiB physical RAM,
7777 MiB configured swap. PINS 3.3.0.1065 and PHD2 ran without an active imaging
or guiding session. The plugin was compiled against the installed host DLLs.
llama.cpp v0.6.0 commit `8345f333951c661d166b00e6f9362e553768f292` was built on
this Pi with native CPU flags, OpenMP and no GPU. Pi 4 compatibility is untested.

[Model manifest](../models/manifest.json) pins revisions, Q4_K_M files, sizes,
SHA256 and licences. Qwen uses Apache-2.0; the optional third candidate,
FunctionGemma 270M, uses the Gemma Terms of Use. Weights remain outside Git.
FunctionGemma specializes in function calling rather than direct dialogue;
MCP and representative function-call accuracy evaluation remain unavailable.

The real plugin's authenticated loopback SSE API measured the fixed
[EN/DE cases](../benchmarks/cases.json), with host/native CPU, RSS, swap and thermal
telemetry. First content means first nonempty text, not first useful or correct
answer. Load includes hash verification and process startup; disk cache may be
warm. Each case ran once per profile. Mixed-case p95 uses nearest rank and is not
a repeated-prompt service-level guarantee. Sampling can miss instantaneous peaks.

## Qwen3-0.6B measurements

Context 2048, output limit 128, temperature 0.2, seed 42, thinking disabled,
one native slot, CPU only. These historical samples predate the final thermal
cutoff and the compatibility fix combining policy and locale into one system
message. They establish initial capacity, not final-build latency certification
or a controlled comparison against Qwen3.5.

| Threads | Completed cases | Load seconds | First content median / p95 seconds | Median output tokens/s | Peak sampled RSS MiB | Highest post-request °C |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 10 | 3.420 | 3.250 / 4.562 | 15.88 | 1062.9 | 80.4 |
| 2 | 10 | 3.319 | 1.673 / 2.381 | 24.70 | 1063.0 | 82.6 |
| 3 | 7 | 2.915 | 1.260 / 1.773 | 24.74 | 1062.9 | 84.8 |

Three threads omit the final three cases, so their subset is not directly
comparable to the full sets. It showed little median decode improvement over two
threads. No AI process or system swap usage was observed.

Raw [thread sweep](../benchmarks/results/pi5-thread-sweep-interrupted.json) and
[earlier four-case interruption](../benchmarks/results/pi5-interrupted.json) are
preserved separately.

## Qwen3.5-0.8B conservative probes

After two unexpected Pi reboots, testing was restricted to conservative probes:
one thread, long pauses, 80 °C cutoff, 1200 MiB RSS ceiling and 768 MiB RAM reserve.
Each failure stopped its run. No further Pi reboot occurred during these probes.

| Probe | Context / output limit | Load seconds | Outcome |
| --- | --- | --- | --- |
| Initial | 2048 / 64 | 4.974 | HTTP 500: embedded template rejected a second system message. Fixed by merging policy and locale. |
| Template fixed | 2048 / 32 | 4.356 | First content 4.435 s; partial `30`; sampled RSS 1216.0 MiB exceeded 1200 MiB and inference stopped. |
| Smaller context | 512 / 32 | 4.573 | First content 4.722 s; partial arithmetic answer; sampled RSS 1192.4 MiB; explicit failure status reports 80 °C cutoff, stopping at 6.456 s. |

None of these initial requests completed, so this set has no valid Qwen3.5 decode rate or quality score.
The reduced-context answer began “30 times 120 seconds equals 3600 seconds, which
converts”; this is incomplete and cannot be scored as a correct 60-minute answer.
Zero process swap was observed. Reducing context did not establish a safe profile.

Raw [template failure](../benchmarks/results/pi5-qwen35-conservative.json),
[memory stop](../benchmarks/results/pi5-qwen35-fixed-conservative.json) and
[512-token thermal stop](../benchmarks/results/pi5-qwen35-512-conservative.json).
The last includes the explicit failure status. Older files predate automatic
failure-status/diagnostic capture.

## Follow-up: Qwen3.5 retry and third candidate

The user requested another Qwen3.5 attempt and the third model. Conservative
probes kept the existing 80 °C / 1200 MiB safeguards, one thread and 512-token
context, with 30-second pauses. No further Pi reboot occurred. These are short
transport measurements, not a replacement for the original full quality matrix.

Qwen3.5 first completed an eight-token-limited request: load 4.412 s, first text
4.441 s, native decode 10.73 tokens/s, peak sampled RSS 1192.4 MiB, post-request
76 °C. Output was only `30 times 120 seconds`, truncated by the budget.
[Raw retry](../benchmarks/results/pi5-qwen35-retry-short.json).

Two separate [short probes](../benchmarks/short-probes.json), with a 16-token
output budget, then completed for Qwen3.5. They use shorter prompts than the
initial matrix and are not equivalent quality evaluations.

| Model / request | Load seconds | First text seconds | Native decode tokens/s | Peak sampled RSS MiB | Post-request °C |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5 short English | 4.348 | 4.209 | 10.33 | 1192.4 | 78.20 |
| Qwen3.5 short German, warm | — | 4.371 | 10.63 | 1192.5 | 76.55 |
| FunctionGemma short English | 2.290 | 1.529 | 33.07 | 510.0 | 75.45 |
| FunctionGemma short German, fresh retry | 2.268 | 1.574 | 33.23 | 510.1 | 76.00 |

The expected answer to both arithmetic prompts is 60. Qwen3.5 English returned
`120`, which is wrong. German returned `20 * 180 Sekunden in Minuten = 3600`,
hitting the output budget without the requested minutes result. Successful SSE
completion at a token limit does not establish a complete, correct answer.

FunctionGemma 270M Q4_K_M was downloaded and SHA256-verified (253,127,904 bytes;
Gemma Terms of Use). Its English answer began a tools-only refusal. The subsequent
warm German request failed with `The response ended prematurely. (ResponseEnded)`.
Its exact cause is undetermined; no corresponding thermal stop or OOM was
observed. A fresh-load German retry completed but answered in English, again
refusing a general question because of its tool specialization. Both successful
FunctionGemma outputs hit the small generation budget. No actual tools were
provided or executed, and this is not a function-calling accuracy benchmark.

[Combined raw follow-up](../benchmarks/results/pi5-short-followup.json) retains
the failed warm request, diagnostic lines and status;
[fresh German retry](../benchmarks/results/pi5-functiongemma-de-fresh.json) is
separate. Zero process swap was observed in all follow-up samples.

FunctionGemma is substantially lighter than Qwen3.5 in these probes, making it
interesting for future specialized routing, but its routing accuracy and warm
request reliability need evaluation with the actual MCP contract. Qwen3.5 fits
only narrowly inside the chosen RAM budget here and still lacks a dependable
quality result. Neither is selected for production. AI remains disabled/unloaded.

## Human quality review

The ten completed one-thread Qwen3 cases show serious limitations. Both exposure
questions should yield 60 minutes; answers were 3600 minutes. English/German
conversation updates should yield 30/20 minutes; answers were 60 seconds/60 minutes.
All four arithmetic/context cases fail.

The German guiding answer incorrectly introduced GPS; the English explanation
described target steering instead of tracking corrections. Camera status and
connection requests were repeated rather than explaining unavailable operations.
The sequence draft was repeated without a useful draft. The German negation
answer had poor grammar. These samples do not support a dependable equipment
assistant. This is manual fixed-case review, not a broad capability evaluation.

No equipment execution path exists. MCP discovery, schemas, authorization and
tool accuracy were neither tested nor simulated.

## Stability and implementation verification

Native compilation reached thermal throttling; earlier inference exceeded the
final 80 °C limit. Two Pi reboots were unexpected. Prior boot logs contained an
undervoltage event, no observed OOM report and an abrupt end; they do not establish
cause. These runs do not prove sustained stability or active imaging performance.
Cooling/power conditions need resolution before renewed sustained testing.

The final plugin samples memory and temperature every 500 ms, rejects hot loads,
kills owned inference on safeguard failure and reports the reason. These are
sampled protections, not hard instantaneous bounds.

Validation passed 24 offline contract checks, 14 actual Pi-plugin integration
checks and one native parent-death test. Integration covers locale/template
compatibility, concurrency rejection, cancellation, timeouts, malformed/oversized
SSE, actual RSS cutoff, disconnect cleanup and disabled mode. The parent-death
child was dead awaiting reaping (`Z`), not running. Tests verify implementation
behavior, not model correctness.

An additional actual PINS service restart with a warm inference fixture passed:
the owned child disappeared, the plugin reloaded, and disabled configuration was
restored. This makes 40 successful checks including the restart verification.

PINS remains running with AI disabled/unloaded and no AI systemd service. Models
and pinned runtime are retained. Future selection needs completed quality checks
and thermal stability on the intended hardware; real MCP integration follows
separately when its server is ready.
