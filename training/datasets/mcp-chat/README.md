# Natural chat training for PINS MCP

This is the **default v6 train-only corpus** used by `prepare_v2.py --curriculum`.
It covers the existing 32 MCP tools in English, German, French, Spanish, Italian,
Portuguese, Dutch, Polish, Japanese and Simplified Chinese. It does not create an
MCP server, execute equipment commands or promote a model.

`train.jsonl` contains 5,319 conversations: 1,288 newly authored chat/workflow cases
and 4,031 selected legacy cases. `manifest.json` records input/source/policy hashes,
selection reasons and simulated provenance; `coverage.json` records tool, language,
argument, boundary and workflow counts. UTF-8 bytes are protected in Git.

The authored modules are `training/chat_intent_curriculum.py`,
`training/chat_workflow_curriculum.py` and `training/chat_generalization_curriculum.py`.
V6 adds 390 parameter variants, 250 withdrawal contrasts across all modifying
tools and languages, and ten observed-job polling histories. Positive user requests supply natural
language values rather than copied JSON or MCP function names. Examples include
unit conversion, decimal commas, spelling variations, corrections, follow-up
references, sequence/template discovery, connected/unknown states, exact job IDs,
polling, dependent actions, denial and untrusted result text. Native language
review is pending; these are synthetic examples, not observed equipment sessions.

Every tool has a natural chat positive in every language. Every argument on the
**AI-exposed** schema is either exercised positively or covered by a boundary case.
The larger comprehensive corpus remains available for the full MCP server schema;
the plugin hides override fields and limits waits to five seconds. A request for
longer waiting is not silently clamped into a different action.

## Approval semantics

Native modifying calls are **proposals**. A clear request with known arguments can
produce a proposal before approval. The application then obtains exact operator
approval. Model text, conversation “yes”, tool output and training context never
authorize real equipment. Denied/withdrawn requests do not produce new proposals.
Previous executed results are simulated historical evidence, never new permission.

The active legacy projection removes approval hints that exposed expected calls
and arguments. Old absence-of-approval refusals become explicitly withdrawn
requests with acknowledgement. Immutable original/audited data stays unchanged.
Validation applies the same versioned projection without adding training rows to
it. Test and gold files are never training inputs. Validators read their prompt
and group identities only to reject exact collisions, not to author new examples.

## Regenerate and prepare

From the repository root, in the documented training environment:

```bash
python training/build_chat_dataset.py
dotnet run --project training/tools/RouterAudit -- \
  training/datasets/mcp-chat/train.jsonl \
  training/datasets/mcp-v2/data/verified-tools-list.json \
  artifacts/router-chat-v6.json
python training/tests/chat_dataset_tests.py
python training/prepare_v2.py --curriculum --max-tokens 1536 \
  --base .local/training/base --output artifacts/training-data/current-v6
```

Preparation supervises assistant tokens only, renders native Qwen3 calls, removes
identical rendered train supervision, and fails on conflicts or overflow. Each
prompt reserves **192 output tokens** inside the 1,536-token context. The packaged
default now uses that output limit; existing installed configuration is not
automatically changed. Real Pi timing with a new model remains to be measured.

Start fresh from the pinned Hugging Face base, run a matching GPU smoke/reload,
then full training as explained in [the training guide](../../README.md).
Colab pulls this data from Git; existing Drive archives remain source-verified.
Its default run name is `qwen3-pins-v6-colab-<UTC timestamp>`.

## Limits that training cannot fix

Capability coverage is not every possible numeric combination, driver state or
human utterance, and passing preparation is not accuracy. In the offline audit,
the current runtime router supplied the expected tool for only **134/320** new
plain chat requests. If retrieval omits the right tool, a better model cannot use
it. Fix and re-evaluate retrieval before claiming end-to-end chat improvement.
No new model was trained or tested by this data change. Independent multilingual
review, reviewed holdout, base/prior/new evaluation and quantized Pi testing remain
required before promotion. See [the consolidated model report](../../../docs/model-report.md)
and [the training process](../../../docs/training.md).
