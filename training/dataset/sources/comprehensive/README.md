# Capability source examples

These are preserved sources within the unified dataset, not a second training
entry point. Train the selected runtime-aligned corpus at
[`../../train.jsonl`](../../train.jsonl) through the documented curriculum.


Train-only, synthetic conversations for every tool in the team's existing MCP
server, audited at `6fef6c4bd73a8e1f24a36235dba6cbb9eff3d3d2` on `unstable`.
This extends the original package and baseline; it does not replace them.
All equipment outputs are simulated. No hardware calls were made.

`train.jsonl` contains the conversations. `manifest.json` records counts,
contract/policy/source hashes and provenance. `coverage.json` lists each tool's
languages, exercised arguments and deliberate boundary cases. Generated source
hashes normalize UTF-8 line endings; dataset/contract hashes cover exact bytes.

Current expansion: **4,496 conversations / 7,106 assistant turns**. Combined with
the existing curriculum: **7,185 conversations / 11,636 assistant turns**.
See [current preparation and model results](../../../../docs/model-report.md)
and [the complete training process](../../../../docs/training.md).

## Capability coverage

| Area | MCP tools | Training scenarios |
|---|---|---|
| Observatory and image history | `get_status`, `get_recent_images` | All eleven device slots, readings, connected/disconnected and unknown values, bounded image counts, grounded replies, untrusted result text |
| Device lifecycle | `connect_device`, `disconnect_device` | Every supported device enum, explicit identity, resource/sequence guards, approval, failure and polling |
| Mount | `slew_mount`, `park_mount`, `unpark_mount`, `home_mount`, `set_tracking` | J2000 hours/degrees, supplied coordinate conversion, tracking booleans, motion guards, failed/cancelled jobs |
| Camera | `capture_image`, `cool_camera`, `warm_camera` | Exposure time, filter/gain/offset/binning/save, cooling temperatures and ramp time, nonblocking waits and errors |
| Focus, filters and guiding | `move_focuser`, `run_autofocus`, `change_filter`, `start_guiding`, `stop_guiding` | Absolute positions, filter name OR position, new/current calibration, idle prerequisites, polling |
| Power/switches | `set_switch` | Writable index, exact values, live min/max/step, no rounding/clamping, explicit approval |
| Sequencer discovery | `get_sequence`, `list_saved_sequences`, `list_target_templates` | Depth limits, existing root, saved-name/template discovery, inspect before editing |
| Sequencer editing | `load_sequence`, `add_target`, `clear_targets`, `save_sequence`, `reset_sequence` | Load versus add versus save; DSO/exposure plans versus templates; multiple ordered filter blocks; rotation, autofocus/guiding/centering/dither options; no implicit start/clear/overwrite |
| Sequence execution | `start_sequence`, `stop_sequence` | Normal validation; running does not mean exposures completed; stop request requires status verification |
| Jobs and emergency stop | `get_job`, `list_jobs`, `cancel_job`, `stop_all` | Every job-producing tool, succeeded/failed/cancelled, lost IDs, cancellation still running, stop errors, abandonment does not prove physical stop |

Every tool is represented in English, German, French, Spanish, Italian,
Portuguese, Dutch, Polish, Japanese and Simplified Chinese. Additional realistic
natural requests and coordinate conversions are currently strongest in English
and German. Explicit settings accompany some localized requests to remove ambiguity.
Translations are authored synthetic examples with **native review pending**.

Every top-level schema argument has a valid call or a deliberate rejection case.
This is capability/contract coverage, not exhaustive combinations of all numbers,
device drivers, equipment states or possible ways a person can express a request.
Dangerous override flags are rejection examples; this dataset does not enable
`force:true`, `overwrite:true`, `skip_validation:true` or automatic `abandon:true`.
Cross-field rules and hardware range checks remain enforced outside the model.

## What MCP cannot currently do

The negative examples distinguish **PINS functionality** from functionality
exposed by the current **MCP**. They include profile/driver administration,
direct dome shutter/flat brightness/rotator commands, safety threshold editing,
standalone plate solving/catalog lookup, FITS download/deletion/stacking,
arbitrary sequencer nodes or a new blank root, selective existing-target edits,
pause, custom tracking rates/mount sync, guider settings and OS administration.

These refusals apply to the available tool scope. Some of those functions exist
elsewhere in PINS or can be configured in sequence templates; do not infer they
are universally impossible. `add_target` builds supported DSO/exposure containers
inside a **loaded root**. It cannot create an arbitrary new sequence tree.
Object names alone never authorize invented coordinates.

## Regenerate, validate and prepare

Run from the repository root in the documented training environment:

```bash
python training/build_comprehensive_dataset.py
python training/tests/comprehensive_tests.py
python training/tests/curriculum_tests.py
python training/prepare_v2.py --curriculum --max-tokens 1536 \
  --base .local/training/base --output artifacts/training-data/current
```

The generator reads current verified contracts and training phrase sources only.
It does not read test/eval_gold/hard_gold. The validator reads held-out inputs
solely to reject scenario-group or exact prompt collisions. Similar synthetic
phrasing across splits remains a limitation; this is not an independent human
generalization benchmark. Never use those held-out answers to add training cases.

Preparation includes these rows automatically with `--curriculum`; no Drive data
upload is needed for the updated Colab notebook, which pulls them from Git.
The existing validation split is used for model selection, never as training data.
The original test/gold records remain held out. Audited copies are regenerated
only to match verified contracts/provenance, not to improve measured scores.

**Start a fresh adapter from the pinned Hugging Face base.** This changes data,
policy and contracts, so checkpoint 700 and old full-run checkpoints are not exact
continuations. Passing data/tokenizer checks is not a GPU smoke or a model accuracy
result. Follow [the training guide](../../../../docs/training.md) for smoke, training, loss,
evaluation and export. Model weights and Pi deployment are unchanged.
