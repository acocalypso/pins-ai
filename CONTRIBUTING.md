# Contributing

Start with the root README and the guide for the area you change: `plugin/README.md`
or `training/README.md`. Runtime and training have separate dependencies. Keep
training tools out of the installed Pi plugin and keep model ownership in PINS.

Run from the repository root. Offline checks:

```bash
dotnet run --project plugin/tests/Pins.AI.Tests -c Release
dotnet run --project plugin/tests/Mcp.GatewayTests -- training/dataset/data/verified-tools-list.json
python training/dataset/scripts/validate_dataset.py
python training/tests/training_tests.py
python training/tests/v2_tests.py
python training/tests/curriculum_tests.py
python training/tests/comprehensive_tests.py
python training/tests/chat_dataset_tests.py
python training/tests/generation_validation_tests.py
python training/tests/colab_tests.py
python training/tests/training_resume_tests.py
python training/tests/evaluation_tests.py
python training/tests/export_tests.py
python training/tests/preflight_tests.py
```

The minimal dataset checks need Python 3.11+ and jsonschema. Tokenizer/metrics
tests use the training dependencies; missing tokenizer files are explicitly
skipped without downloads. Full GPU smoke tests are separate, not offline CI.
The Linux runtime lifecycle fixture is `python plugin/tests/runtime-smoke.py`;
it is not runnable on Windows. Production packaging is Linux ARM64.

Preserve the original dataset manifest and byte hashes. Put newly authored
training examples under `training/dataset/sources/` and wire them into the train-only
curriculum. Keep held-out cases isolated; do not generate training data from
evaluation failures in `test`, `eval_gold` or `hard_gold`. Public issue/user requests
can seed new training examples after privacy and provenance review. Linguistic
review status and synthetic hardware state must be explicit.

Prepare data again after modifying the curriculum or MCP contracts. A new recipe
requires a matching new smoke run. Never call a changed dataset/configuration an
exact checkpoint continuation. Record versions/hashes, losses and measured
accuracy; a successful training script or low loss is not a release guarantee.

`--curriculum` uses `training/chat_curriculum.py`. Regenerate train-only chat data
with `python training/build_chat_dataset.py` after editing its authored sources.
Keep requests in natural language, with arguments derived from user text or
explicit simulated observations. Never put expected tool/argument JSON in system
approval hints. A tool call is a proposal before operator approval; an explicit
denial prevents a proposal. Preserve original split bytes and record any active
projection. Run the chat tests and full tokenizer preparation before GPU smoke.

For notebooks, edit `training/create_colab_notebook.py`, regenerate the checked-in
notebook, and keep outputs/execution counts cleared. Preserve old Drive layout
compatibility. Do not commit environments, tokens, private conversations, trained
checkpoint archives or optimizer states. Only deliberately selected deployment
GGUF files belong in Git LFS.
