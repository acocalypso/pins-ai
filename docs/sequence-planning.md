# Sequence planning through chat

The current MCP `add_target` implementation creates a Deep Sky Object container
with Smart Exposure blocks in the loaded advanced sequence. For “300 five-minute
exposures,” the plan is `count: 300`, `exposure_seconds: 300`. Omitting `filter`
means no filter-change instruction; it does not remove a physical filter.

MCP requires J2000 RA in hours and Dec in degrees. Its current tool catalog has
no target-name coordinate resolver and no tool to create a new blank sequence
root. Without verified coordinates, the assistant must ask. It must also clarify
whether to append to the loaded sequence or replace its targets. Loading a saved
sequence or clearing targets requires separate exact operator approval. Preparing
a target must not start imaging.

This was checked against `NINA.Mcp/Tools/Catalog/SequenceTools.cs` and
`NINA.Mcp.Adapter/Sequence/NinaSequenceService.Plan.cs` on upstream unstable.
The gateway still validates against the full discovered MCP schemas. Compact
model prompts remove schema annotations only, retain all constraints, and use
the same concise add_target description as training.

## New training examples

`training/data/sequence-plans.jsonl` contains 20 newly authored English/German
train-only conversations (22 assistant turns). Regenerate with:

```powershell
.local/training-venv/Scripts/python.exe training/sequence_curriculum.py
.local/training-venv/Scripts/python.exe training/tests/curriculum_tests.py
```

The existing Colab `prepare_v2.py --curriculum` path includes these examples in
the training split only. The previous 326-example baseline and held-out splits
are preserved. Fixture coordinates are supplied explicitly by the user and are
not a target resolver. Linguistic review is still pending. Local Qwen3 tokenizer
verification found a maximum of 1,221 tokens across the 22 assistant training
turns, below the configured 1,536-token budget, with assistant-only labels.

For a new run, pull this repository and rebuild the curriculum, then train from
the pinned base model. Do not resume an old checkpoint against changed training
data and call it the same run. No training was started and the diagnostic
checkpoint-700 adapter remains unchanged.

## Runtime and interface

Packaged inference defaults now use two CPU threads, including batch processing.
The existing 75°C thermal and memory limits remain in force. Chat shows an
accessible thinking indicator while inference is pending, allows drafting the
next message, and keeps the composer outside the conversation scroll area.
Debug remains available in short landscape layouts; errors scroll with messages.
