# Unified PINS AI dataset

This is the single dataset location for PINS AI. Use **`train.jsonl`** with
`python training/prepare_v2.py --curriculum`; do not concatenate every JSONL file
in this directory. The current v8 corpus has 6,039 conversations / 8,320 unique
prepared assistant turns, covering all 48 MCP tools and ten languages.

```text
training/dataset/
  train.jsonl          Active, runtime-aligned multilingual training corpus
  manifest.json        Selection, counts, source hashes and provenance
  coverage.json        Tool/language/argument coverage
  contracts/           Current compiled MCP48 schemas and source/adapter audit
  data/                Preserved original/audited MCP source data and contracts
    validation.jsonl   Development validation; never optimized as training rows
    test.jsonl         Held-out evaluation only
    eval_gold.jsonl    Final evaluation only
    hard_gold.jsonl    Final evaluation only
    audited/           Contract-corrected source splits, with the same isolation
  sources/comprehensive/ Preserved/generated capability source examples
  baseline/            Original informational authored/gold and sequence assets
  scripts/             Original integrity checks and source generators
```

All previous training assets remain available here. The active corpus combines
natural chat/workflows with selected compatible source examples. Its manifest
records exclusions: duplicate-like phrasing, JSON-copying positives, obsolete
approval-stage cases and waits incompatible with the five-second AI gateway.
These source rows are retained, not silently deleted or reintroduced into SFT.
The [relocation inventory](relocation.json) records previous/current paths and
hashes, including preserved frozen splits and baseline assets.
The preserved baseline is a separate historical control, not extra rows to
append without checking task/policy compatibility. Baseline gold is evaluation-only.

Train/validation/test/gold are distinct even inside one dataset. Frozen original
and audited split bytes remain unchanged; model output is never approval.
Hardware state is simulated and native-language review remains pending.

Regenerate derived source metadata and the active corpus after editing generators:

```bash
python training/build_comprehensive_dataset.py
python training/build_chat_dataset.py
python training/dataset/scripts/validate_dataset.py
python training/tests/comprehensive_tests.py
python training/tests/chat_dataset_tests.py
python training/prepare_v2.py --curriculum --max-tokens 1536 \
  --base .local/training/base --output artifacts/training-data/current
```

Preparation supervises assistant tokens only, renders native Qwen3 calls, removes
identical rendered training turns and fails on conflicts/overflow. Every prompt
reserves 192 output tokens inside the 1,536-token context. Regenerate prepared data
and run a matching GPU smoke after a path/provenance change; existing prepared
manifests and evaluation identities may be stale even if conversations match.

Colab reads this dataset from Git. Existing Drive archives with the former
`training/datasets/mcp-v2/data` or `PINS_MCP_AI_Training_v2/data` layouts remain
readable after byte-hash verification. Code is never imported from Drive.

See [the complete training process](../../docs/training.md) and
[the current model/comparison report](../../docs/model-report.md).

The active schemas are extracted from upstream unstable `c4d8f5a`. Historical
32-tool snapshots stay under `data/` for reproducibility. The active source
projection excludes old claims that dome, flat, rotator, solving and catalog
operations are unavailable. Raw legacy examples remain preserved.

The new source `training/upstream_tool_curriculum.py` adds 272 train-only
conversations, including native requests in ten languages, all new action enums,
argument contrasts, explicit withdrawals, job polling and catalog-to-plan/center/
append workflows. Its separate development-only function provides 44 independently
authored English/German validation cases. Preparation combines these with the
original 583 validation conversations (627 total), never with training rows.
Original test/gold files are unchanged and only cover the historical tool set.

Training calls are proposals. Plate solving without mount synchronization still
captures an image and requires operator approval. Roof-collision bypass is hidden
and refused. `meridian_flip` belongs only to exposure plans, not template overrides.
Catalog coordinates must come from an observed result, not model memory.


V8 adds 460 train-only rows from `training/reliability_curriculum.py`; those
sources never import evaluation reports or held-out examples. `validation-chat-v7.jsonl`
freezes development inputs separately. `holdout-v8.jsonl` is an independently
authored, frozen 40-case future safety starter (native/domain review pending),
never a training or checkpoint-selection input. The isolation audit is
`python training/audit_split_isolation.py --output artifacts/new-isolation.json`.
It checks groups/translation-linked families, exact requests and numeric-normalized
near-text matches; this does not prove semantic independence.
New data/scoring requires a fresh base run and matching v8 smoke, not a historical
v7 resume. Retain old recipes and source revisions for exact legacy reproduction.
