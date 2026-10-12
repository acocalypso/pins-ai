# Comprehensive MCP training data verification — 2026-10-12

The current training contract was exported by compiling the team's existing
`McpDispatcher` from `nitr57/pins` `unstable` commit
`6fef6c4bd73a8e1f24a36235dba6cbb9eff3d3d2`. The audit's observatory proxy throws
on equipment access. This verifies source-generated `tools/list`, not the
currently installed Pi's live compatibility.

Compared with the previous `e6d13f9` audit, the tool set still contains 32 tools.
Job descriptions/schema now include abandoned jobs, cancellation requests,
failed stop commands and `cancel_job.abandon`; save annotations changed and
adapter/resource/sequence behavior also changed. The catalog and audited copies
were regenerated from source, with no invented MCP tools.

## Measured data and tokenizer results

| Item | Result |
|---|---:|
| New comprehensive train-only conversations | 4,496 |
| New comprehensive assistant turns | 7,106 |
| Combined current curriculum conversations | 7,185 |
| Combined assistant turns for SFT | 11,636 |
| Validation conversations / assistant turns | 583 / 1,236 |
| MCP tools / top-level schema arguments covered | 32 / 73 |
| Languages | 10 |
| Skipped or truncated examples | 0 |
| Training tokens: minimum / median / p95 / maximum | 226 / 713 / 1,121 / 1,481 |
| Validation tokens: minimum / median / p95 / maximum | 232 / 672 / 1,036 / 1,336 |
| Configured token ceiling | 1,536 |

New conversations per language: English 476, German 476, and 443 each for French,
Spanish, Italian, Portuguese, Dutch, Polish, Japanese and Simplified Chinese.
Each tool has examples in all ten languages. Each schema property has a positive
argument example or a deliberate boundary/override rejection. Ordered exposure
blocks also exercise filter, gain, offset and binning, plus invalid nested values.
This does not imply exhaustive coverage of combinations, drivers or language.

Conversion used the **actual pinned local Qwen3 tokenizer**, native tool-call
markers, assistant-only labels and fail-on-overlength behavior. Prepared files
are ignored artifacts under `artifacts/comprehensive/sft/`; regenerate on the
training host. The checked-in conversations, manifest, generator and coverage
report are under [mcp-comprehensive](../training/datasets/mcp-comprehensive/README.md).

## Verification run

- Original package integrity checks passed: 924/583/593 conversations, all
  original manifest hashes and scenario splits intact.
- Original baseline and raw held-out files were compared against `13ba3ff`:
  nine files were byte-identical. Informational preparation still yields 326
  train, 52 validation and 54 test examples.
- Comprehensive tests passed: deterministic generation, checked-in hashes,
  all tools/languages/device enums, argument coverage, schema-valid exposed
  calls, no unsafe override outputs, exact approval fixtures, polling IDs,
  no mutation retries, grounded readings and save uncertainty, held-out
  group/exact-prompt isolation.
- Curriculum tests: 5 passed. MCP conversion tests: 6 passed. Colab/Drive tests:
  4 passed. Baseline pipeline tests: 16 run, one optional baseline tokenizer test
  skipped because its environment flag was unset; full MCP conversion used the
  real tokenizer and checked every assistant turn.
- Gateway checks: 174 passed against the current catalog, all operations simulated.
- Colab notebook: 11 code cells compile, outputs/execution counts cleared,
  generator reproduces the notebook exactly. Fresh mode uses the updated Git
  data; existing matching original Drive data is retained.

## What remains unverified

No GPU smoke, model training, model accuracy evaluation, native-speaker review,
new Colab session or Pi deployment was performed. Synthetic tool results are not
equipment observations. Related synthetic phrases across splits limit independent
generalization claims. Low loss and coverage checks do not establish safe behavior.

Use a **fresh adapter from the pinned base** and follow the
[training guide](../training/README.md). The new dataset/policy/contract cannot
exactly resume checkpoint 700 or an older full run. Compare the trained candidate
on untouched evaluation suites and validate the quantized candidate before
changing deployed weights. MI355X still requires a real ROCm smoke and measurements.
