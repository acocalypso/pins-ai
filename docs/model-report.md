# Trained models and comparison report

Report date: **2026-10-13**. This consolidates the current deployment decision,
matched model comparisons, relevant historical results and Pi observations.
Measurements below are dated evidence, not a fresh check of the device today.
Reproduce future evaluations with [the training process](training.md).

## Current decision and model inventory

**Keep the bundled diagnostic checkpoint-700 adapter.** The newest completed
checkpoint-1900 adapter scored worse overall than both previous adapters in the
matched deployment-format comparison. Checkpoint 400 was also rejected after its
earlier Pi trial. None of these models is accepted as production-quality.

| Candidate | Training status | Current role |
| --- | --- | --- |
| Pinned Qwen3 0.6B base | Unmodified training base | Training/evaluation control; bundled Q4_K_M base for runtime LoRA |
| Prior checkpoint 700 | Incomplete 846-step run; diagnostic save | Bundled and last verified Pi profile; retain pending a passing replacement |
| Prior checkpoint 400 | Incomplete v4 Colab run, recorded best save at 0.55 epochs | Diagnostic candidate; rejected for deployment |
| New checkpoint 1900 | Completed v4 run, stopped at 2.6104 of three planned epochs | Evaluated candidate; not promoted |
| V7 Colab selected 300 | Completed through early stopping at 700/1,476 | Development-only evidence; not promoted |
| V8 reliability recipe | Data/selection/runtime checks prepared | Fresh experiment required; no GPU run or model gain measured |
| V6 recipe | Data and pipeline prepared; no full training performed | Next fresh-base experiment, not an existing trained model |

The folder containing the new run was named `mi355x`, but its metadata recorded
**RX 7900 XT / gfx1100**. That is not evidence of MI355X training. ROCm profiles
are prepared; no MI355X throughput or v6 accuracy has been measured.

The preserved original 326-example informational baseline and the older completed
Colab v2 adapter remain historical assets. They are not additional candidates in
the latest three-adapter comparison.

## Matched deployment-format comparison — 2026-10-13

Each candidate used the **same bundled Q4_K_M base plus its FP16 runtime LoRA**,
llama.cpp `8345f333951c661d166b00e6f9362e553768f292`, desktop CPU AVX2, two threads,
1,536-token context, 192-token output, greedy decoding and seed 3407. Case IDs,
rendered prompts, schemas and scoring matched within each suite. File/report
hashes and exact identities are preserved in
[comparison evidence](../training/experiments/model-comparison-20261013.json).

"Exact" means correct tool/no-tool choice and exact expected arguments. Schema
validity includes ordinary no-tool text, so it does not prove a requested action
was proposed. Scores measure proposals only; no equipment commands were executed.

| Chat development validation, 583 cases | New 1900 | Prior 700 | Prior 400 |
| --- | ---: | ---: | ---: |
| Correct tool/no-tool | 410 (70.3%) | 427 (73.2%) | **447 (76.7%)** |
| Exact arguments | 400 (68.6%) | 417 (71.5%) | **425 (72.9%)** |
| Schema valid | **572 (98.1%)** | 571 (97.9%) | 558 (95.7%) |
| Output truncated | 0 | 0 | 0 |

| Legacy hard-gold, 120 cases | New 1900 | Prior 700 | Prior 400 |
| --- | ---: | ---: | ---: |
| Correct tool/no-tool | **93 (77.5%)** | **93 (77.5%)** | 92 (76.7%) |
| Exact arguments | 81 (67.5%) | **89 (74.2%)** | 82 (68.3%) |
| Schema valid | 113 (94.2%) | **115 (95.8%)** | 111 (92.5%) |
| Output truncated | 0 | 0 | 0 |

The new model is 2.9 percentage points below 700 and 4.3 below 400 on chat exact
accuracy. Against 700 it gained 86 chat cases but lost 103; against 400 it gained
81 but lost 106. On hard-gold it gained/lost 14/22 against 700 and 12/13 against
400. No statistical-significance or universal ranking claim is made.

The chat suite used the **v5 proposal-stage projection**, while hard-gold retained
legacy approval semantics. They are separate assessments and should not be pooled
as one acceptance score. Gold has previously been inspected; it is historical
evidence, not a fresh unbiased holdout. Current v6 projection fingerprints differ;
rerun all candidates with identical v6 inputs for a new comparison.

## Behavioral strengths and regressions

Exact scores from chat development validation:

| Category | New 1900 | Prior 700 | Prior 400 |
| --- | ---: | ---: | ---: |
| Direct tool requests | **186/320** | 245/320 | 242/320 |
| Job follow-up | **101/140** | 72/140 | 96/140 |
| Grounded replies after supplied results | **44/50** | 37/50 | 28/50 |
| Missing-coordinate clarification | **10/10** | 3/10 | 2/10 |
| Missing-exposure clarification | **10/10** | 9/10 | 8/10 |
| Injection cases | 6/10 | **8/10** | 6/10 |
| Explicit withdrawal / no-approval | 10/10 | 10/10 | 10/10 |
| No-start requests | 10/10 | 10/10 | 10/10 |

The new model improves supplied-result handling and clarification. Its largest
regression is direct tool proposals: 58.1%, versus 76.6% and 75.6%. Those lost
calls outweigh its improvements. Grounded-reply scores are automatic task scores,
not independent linguistic or false-completion review. Job follow-up fixtures do
not establish successful real multi-step hardware orchestration.

None passed every injection case. The application gateway must validate arguments
and enforce exact operator approval regardless of any model score.

## Multilingual comparison

Exact chat accuracy; most languages have 58 cases, English/Italian/Polish 59:

| Language | New 1900 | Prior 700 | Prior 400 |
| --- | ---: | ---: | ---: |
| English | 79.7% | **81.4%** | **81.4%** |
| German | **82.8%** | 81.0% | 81.0% |
| French | 70.7% | 77.6% | **79.3%** |
| Spanish | 75.9% | **86.2%** | 79.3% |
| Italian | 57.6% | **78.0%** | 59.3% |
| Portuguese | 55.2% | 44.8% | **56.9%** |
| Dutch | **79.3%** | 63.8% | 70.7% |
| Polish | 57.6% | 40.7% | **66.1%** |
| Japanese | 63.8% | 75.9% | **84.5%** |
| Chinese | 63.8% | **86.2%** | 70.7% |

German/Dutch improve; Chinese/Italian show the largest regressions against 700.
The uneven results support broader multilingual argument and intent coverage,
not a claim that any candidate is fluent or reliable in every language.

## Loss and interpretation

The completed new run used **v4 data**, before the natural-chat update. Its
training-window loss fell from **1.7464 to 0.007**, and validation loss from
**0.623 at step 100 to 0.0588 at step 1900**. Lower loss did not predict the
deployment-format ranking. Earlier supplied BF16 reports looked substantially
better, but differences in runtime, prompts and precision prevent attributing
the whole gap to quantization or proving overfitting as the sole cause.

V6 now measures generated development proposals for checkpoint selection rather
than relying only on loss. It retains both losses and adds tool choice, exact and
macro accuracy, schema validity, unwanted proposals and truncation metrics.

## Earlier evaluations — separate conditions

These results explain the retained assets; they are not directly comparable with
the 192-token three-adapter comparison above.

The completed Colab v2 run (`qwen3-pins-v2-colab-20261011-131536`) used T4
FP16/NF4 and finished 240 optimizer steps/two epochs in 3,732.7 seconds. Its
window loss fell 1.510 → 0.154 and validation loss 0.258638 → 0.152567. Saved
hard-gold exact scores were base **39/120**, v2 **89/120**; tool choice was base
60/120, v2 100/120. Those synthetic, previously inspected scores did not establish
deployment or equipment safety. The [loss plot](../training/experiments/colab-v2-loss.png)
is preserved as historical evidence.

The 2026-10-12 checkpoint-400 trial merged the adapter into the clean base before
Q4_K_M export and used a 128-token output limit. Hard-gold exact scores were
CUDA/NF4 **84/120** and merged Q4_K_M **87/120**. A matched 50-case eval-gold subset
gave base **17/50**, checkpoint 700 **43/50**, checkpoint 400 **27/50**. Conditional
synthetic polls were 0/10, 9/10 and 0/10 respectively. This was a subset, not the
full eval-gold suite. Its merged candidate, runtime settings and prompts differ
from the later separate-LoRA comparison. Do not treat the 87/120 and 82/120 scores
as contradictory measurements of an identical deployment.

The candidate's live M31 request asked again for duration/count already supplied.
It failed its quality gate, and the previous Pi configuration was restored
byte-for-byte. Hashes and trial results are in
[checkpoint-400 evidence](../training/experiments/checkpoint400-20261012.json).

Initial loader benchmarks also explored Qwen3.5 0.8B and FunctionGemma 270M.
Qwen3.5 narrowly fit the memory budget in short probes, with earlier RAM/thermal
cutoffs; FunctionGemma used approximately 510 MiB RSS and decoded short outputs
around 33 tokens/s, but those were transport probes, not representative MCP
accuracy tests. Neither established a better assistant. Original results remain
under [plugin benchmark evidence](../plugin/benchmarks/results/).

## Pi verification

Last recorded model profile is the existing PINS-owned checkpoint-700 LoRA on a
Pi 5 with 4 GB RAM. The base is 484,220,160 bytes and LoRA 10,119,776 bytes; exact
hashes/licenses are in [the bundled manifest](../plugin/models/bundled.json).
Debian packaging includes them without installation-time downloads. The plugin
owns the model process, MCP client and enforceable chat gateway.

The 2026-10-12 packaged two-thread probes recorded:

| Request | Observed result | Elapsed |
| --- | --- | ---: |
| English equipment status | Real read-only `get_status` | 8.730 s |
| Connection-status follow-up | Text reporting disconnected equipment | 3.890 s |
| German equipment status | Real read-only `get_status`, shared prefix cache | 1.240 s |

Peak model RSS was 952.1 MiB; peak temperature across these three probes 65°C.
The exact M31 sequence request took 18.172 s and returned HTTP 400, "Choose
exactly one target plan." Gateway validation refused the malformed proposal;
no target was added and imaging did not start. The 75°C cutoff remained in place.
These short probes had changed prompts/cache conditions and are not matched
latency benchmarks or active-imaging performance evidence.

Live MCP discovery exposed 32 tools; the chat gateway exposed 31 when no verified
writable switch capability was available. Trusted test clients connected without
manual token entry; missing/untrusted origins returned 401. Start → experimental
warning → chat, thinking feedback and bounded composer scrolling were verified.
Sequential approvals and failures used simulated hardware responses. One earlier,
separately authorized direct MCP camera connection succeeded to the selected
ALPACA simulator; it was not an autonomous model action or an exposure test.

The 2026-10-13 local checks passed 28 runtime and 198 simulated gateway checks.
Historical gateway replay denied all 99 schema-valid unapproved mutating
proposals from base/v2 reports, with zero unauthorized simulated actions; its
[verification JSON](../plugin/benchmarks/results/gateway-verification-20261011.json)
is retained. Gateway enforcement does not establish model correctness.

## Prepared improvements and remaining work

V6 adds 650 independently authored examples: 390 parameter variants, 250
withdrawal contrasts, and ten observed-job histories. Total corpus: **5,319
conversations / 7,588 unique assistant training turns**, all 32 tools and ten
languages. Pinned-tokenizer preparation skipped zero examples; all 4,558 calls
passed actual gateway schemas. Original/audited splits and the 326 baseline retain
their bytes. Sources did not use held-out prediction text to author new rows.

Checkpoint selection now uses generated tool-balanced exact accuracy, penalizes
unwanted proposals/truncation, allows four early-stopping checks and retains four
checkpoints subject to best/latest handling. Colab/T4 and ROCm profiles, commands,
monitoring and exact-resume constraints are in [the training process](training.md).
The optional lower-rate AMD profile is unverified. V6 local verification and its
102-case development contract are in
[data evidence](../training/experiments/v6-data-validation.json); controls are in
[the experiment plan](../training/experiments/v6-accuracy-plan.json).

**No v6 GPU smoke/full run, increased accuracy or Pi promotion is claimed.**
The independent router audit still retrieved the expected tool for only
**134/320 (41.875%)** natural-chat requests. Model comparisons supplied correct
scoped tools, so they exclude this additional failure source. A model cannot call
a tool omitted from its context; retrieval needs separate runtime improvement.

Before promotion: run matching GPU smoke/full training, compare base/prior/new
with identical current development inputs, review native-language and completion
claims, evaluate quantized parity, and verify plugin-owned Pi resources/latency.
An independently authored/reviewed frozen holdout is needed for a new unseen
accuracy claim. Sustained thermals, active imaging/guiding contention, physical
mobile keyboards and broad real equipment workflows remain unverified.

The previous version-specific narrative documents were consolidated here and in
`training.md` / `plugin/README.md`; their exact historical text remains in Git
history. Evidence JSON and model bytes were not rewritten by documentation cleanup.


## MCP48 training update, 2026-10-14

The upstream server at `c4d8f5a` exposes 48 tools, adding 16 without removing any.
Ten new queries cover offline catalog/visibility planning, guiding reports and
incidents, log/event history, session summaries and system health. Six modifying
tools cover safe shutdown, plate solve, target centering, rotator, flat device
and dome actions. `add_target` also adds the exposure-plan `meridian_flip` flag.
The current v7 data and Colab/ROCm profiles match those schemas. Original model
comparison scores above remain historical evidence; none evaluates a v7 adapter.

The gateway now exposes the audited additions, keeps exact operator approval for
all six modifiers, validates new cross-field arguments and refuses roof-collision
bypass. Catalog discovery accepts up to 256 KiB; ordinary results retain the
64 KiB cap. Both bounds and no-approval refusal are tested with simulated transport.
Nothing was sent to real equipment and nothing was installed on the Pi.

The offline router audit validates all 4,770 corpus calls against the actual C#
gateway with zero argument errors. It retrieves the expected tool on 294/480
single-turn authored training requests (61.25%): 160/160 for the new tools and
134/320 for the historical tools. This is candidate availability on training
phrases, not unseen model accuracy or successful equipment execution. Older
natural-language retrieval remains a significant limitation and the Pi model
still needs fresh training and matched evaluation before promotion.


## Completed v7 Colab run and v8 code changes, 2026-10-18

The supplied `qwen3-pins-v7-colab-20261018-123322` run is **completed**, having
stopped by four-check early stopping at 700/1,476 (epoch 1.423). Its exported
adapter's SHA-256 exactly matches checkpoint 300, not checkpoint 700:
`ade85d18b0cf2a89b2f0dfb18169053f7812c48d2e4ed5ed3c29aa79675a1ac1`.
The selected loss is 0.254906; final step-700 loss is 0.153557. The resumed
segment loss is 0.187374 and must not be called whole-run loss.

The five case-level files 300/400/500/600/**700** are present in this workspace;
the latter was not supplied to the earlier external brief. The offline analyzer
checks their identical contracts, IDs and scorer, and reproduces:

| Step | Tool/no-tool correct | Exact | Schema valid | Unwanted proposals / 50 no-tool | Historical selection score |
|---|---:|---:|---:|---:|---:|
| 300 | 129/146 | 110/146 | 144/146 | 9 | -1.06164 |
| 400 | 127/146 | 108/146 | 144/146 | 18 | -1.12329 |
| 500 | 126/146 | 107/146 | 142/146 | 18 | -1.12329 |
| 600 | 134/146 | 117/146 | 143/146 | 10 | -1.06849 |
| 700 | 134/146 | 116/146 | 144/146 | 11 | -1.07534 |

No listed generation truncated. The eleven tool-sequence followups are required
calls, not no-tool failures. At step 600, injection exact is 2/10; new upstream
tool choice is 31/32 but exact arguments only 15/32. These are unwanted proposals,
not measured physical operations. Case-level paired gains/regressions, confusion,
argument errors, per-category/language/tool slices and Wilson intervals are in
[the analysis evidence](../training/experiments/v7-reliability-20261018.json).
This is training-precision fixed development evidence, not independent unseen,
GGUF or Pi accuracy. None of these checkpoints is production eligible.

V8 adds 460 independent train-only conversations with multilingual trust-boundary,
missing/invalid input, numeric/enum/action and optional-field contrasts. Total:
6,039 conversations / 8,320 unique prepared training turns, all 48 tools and ten
languages. Preparation skips zero; max train sequence/completion lengths are
1,401/139 tokens. All 5,140 native calls pass the actual C# gateway schema audit.
The 627 development conversations are frozen; all 146 selected case fingerprints
and rendered prompts remain exactly those used by the supplied v7 reports.
Historical data/contracts/baseline bytes remain unchanged.

New versioned reporting separates unwanted proposals, invalid no-tool outputs,
required-tool schema errors, argument fidelity and per-slice sample sizes. V8
keeps the full adapter safety-ranked, retains separate best-safety/accuracy
research copies, and early-stops only when neither safety nor macro accuracy
improves. Patience and learning rate are unchanged. Historical scores are not
reinterpreted. The 300-versus-600 ranking tradeoff has synthetic regression tests;
better accuracy does not make the unsafe checkpoint eligible.

Runtime changes use the gateway's read-only classification in chat, removing a
stale duplicate list that omitted new queries. A model-coordinate origin guard
refuses motion/target previews without recognized explicit user coordinates or
one structured server-recorded catalog match. It accepts decimal hours/degrees
and several locale labels, but is intentionally conservative: unsupported formats,
ambiguous matches and absent evidence need clarification. It is not cryptographic
proof of user intent, a universal injection defense or a substitute for exact
operator approval. Direct operator-entered tool arguments keep existing approval
and schema checks. All 223 gateway regressions are simulated.

The router still retrieves 294/480 authored requests (61.25%), with 186 misses
in older tool phrases. This remains a separate runtime limitation. V8 has no
measured model gain yet. Its frozen future starter holdout has only 40 cases,
native/domain review pending; it does not cover every capability and cannot
support a production claim alone. Near-text/group isolation checks are heuristic.

Next: review multilingual semantics and extend independent release coverage,
run a new matching GPU smoke and fresh v8 training, compare base/prior/selected
and diagnostic candidates on frozen development and independently reviewed tests,
then matched Q4 base plus F16 runtime LoRA evaluation. Only after those gates
consider separately authorized Pi resource and read-only compatibility checks.
No new training, weight download, quantization, Pi deployment or equipment action
was run in this code task. [Runnable commands](training.md#v8-reliability-experiment-and-historical-v7-migration-2026-10-18)
include CUDA-NF4, ROCm-BF16, analysis, holdout and matched runtime verification.
