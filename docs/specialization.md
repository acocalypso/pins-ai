# Qwen3 0.6B specialization

The user selected Qwen3 0.6B as the working baseline after the measured quality
failures. The development pipeline is implemented; **no PINS model has been
fine-tuned, exported or promoted yet**. Working inference remains independent of
training. This is not training from scratch, and nothing trains on the Pi.

## Scope and training host

Use the full post-trained `Qwen/Qwen3-0.6B` checkpoint, not the GGUF or the
`Qwen3-0.6B-Base` checkpoint. The pinned HF revision and 1,503,300,328-byte weights
hash are in [training config](../training/config.json). Qwen is Apache-2.0.
The upstream [model card](https://huggingface.co/Qwen/Qwen3-0.6B) documents the
non-thinking chat template; the pipeline uses it with `enable_thinking=False`.

Training is offline after explicitly downloading the checkpoint and installing
development dependencies. Use a CUDA workstation with Python 3.11 and preferably
6–8 GB or more available VRAM. This workspace has a GTX 1050 Ti with 4 GB total
VRAM. A full run on it is unverified and may be slow or fail to fit. The preflight
checks actual CUDA/NF4 kernels and free VRAM rather than inferring support from
`nvidia-smi`. Pascal needs a compatible CUDA wheel; this recipe pins PyTorch
2.7.1/cu126 and avoids CUDA 12.8+ wheels that drop older architectures. See
[PyTorch wheels](https://pytorch.org/get-started/previous-versions/) and
[bitsandbytes support](https://huggingface.co/docs/bitsandbytes/main/en/installation).

No GPU training dependencies are installed into PINS or the Pi. A separate CPU
environment can verify the tiny random-model backend fixture; that fixture is not
a trained Qwen checkpoint or evidence of CUDA/QLoRA performance.

## Data and leakage controls

[Preparation](../training/prepare.py) authors deterministic exposure arithmetic
and conversation corrections, and imports the project's original bilingual
[bootstrap examples](../training/data/authored.jsonl). These include astronomy
terms, clarification, invalid durations, negation, language switching, unavailable
equipment operations, quoted instructions and false execution claims. They use no
private equipment logs or external copyrighted training text.

The current 432-example bootstrap splits into 326 train, 52 validation and 54 test
examples. Translations, paraphrases and all durations/corrections for an exposure
count share a scenario group. Assignment is deterministic and stratified between
arithmetic and behavior so validation and test retain arithmetic coverage.
Duplicate IDs/prompts,
gold-group leakage and exact gold-prompt overlap are rejected. New imported data
must declare source, licence and absence of private data; that declaration does
not replace a human privacy/provenance review. Add reviewed JSONL files using
`prepare.py --extra-chat path.jsonl`.

The separate [gold set](../training/data/gold.jsonl) preserves all ten original
benchmark prompts. Its references and rubrics were authored and checked against
the required behavior. The owner should review them before promotion. Gold is
never used for training, validation loss, checkpoint choice or prompt tuning.
The development test split is also excluded from training/checkpoint selection.
For later experiments, add fresh held-out scenarios if repeated evaluation has
influenced development. This small bootstrap is not a representative production
corpus and does not establish the 95% tool-accuracy target.

Each chat JSONL record has `id`, `group`, `locale`, alternating `messages` ending
in the target assistant answer, and `provenance`. The model receives the exact
current plugin system policy plus caller locale as one leading system message.
Only the final assistant completion and EOS contribute to loss; system messages,
user input, historical assistant turns and padding are masked. Overlong examples
fail explicitly rather than silently losing the answer. Prompt/template hashes
are recorded and checked for drift.

## Prepare and verify

Run commands from the `pins-ai` checkout. Preparation uses only Python's standard
library, without downloading or loading model weights:

The separate `artifacts/pins-ai-training-0.1.0.zip` development bundle contains
these sources and the policy contract for the same workflow on another training
workstation. It contains no weights, virtual environment or credentials. The
normal plugin ZIP does not install any of this training stack on the Pi.

```bash
python training/prepare.py
python training/evaluate.py review-template --output artifacts/training/reviews.json
```

The GPU workstation setup is explicit. On Windows, use the equivalent virtual
environment executables under `Scripts` instead of `bin`:

```bash
python3.11 -m venv .local/training-venv
.local/training-venv/bin/python -m pip install torch==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu126
.local/training-venv/bin/python -m pip install -r training/requirements.txt
.local/training-venv/bin/python training/preflight.py
.local/training-venv/bin/python tests/training_tests.py
```

Full checkpoint download requires explicit flags and retains pinned-file hashes,
licence and provenance. Allow several GB for development libraries, checkpoint,
adapter/checkpoints, clean merged weights and F16/Q4 exports:

```bash
.local/training-venv/bin/python training/fetch_base.py --download --accept-license
```

For tokenizer-only verification, add `--tokenizer-only --output
.local/training/tokenizer`. It downloads no weights. Set
`PINS_TRAINING_TOKENIZER` to that directory when running `tests/training_tests.py`
to check the real non-thinking template, loss masks and all 432 example lengths.

## Train and merge

[Training](../training/train.py) uses Transformers Trainer and PEFT LoRA, with
explicit completion masks. This avoids assuming Qwen's stock Jinja template has
TRL generation-mask annotations. QLoRA uses NF4/double quantization, rank 8 across
attention/MLP projections, sequence length 512, batch 1, gradient accumulation 16,
gradient checkpointing, AdamW, learning rate 1e-4 and three epochs. BF16 is chosen
only if the GPU supports it; otherwise FP16. LoRA without 4-bit loading is also
available with `--method lora`. See [PEFT quantization guidance](https://huggingface.co/docs/peft/v0.18.0/en/developer_guides/quantization)
and [Trainer API](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer).

```bash
.local/training-venv/bin/python training/train.py \
  --base .local/training/base --output .local/training/runs/qwen3-pins-v1
```

For a CUDA development smoke check use a separate new output folder and
`--max-steps 1`. Such a run is permanently marked `smokeOnly`; it cannot pass the
promotion gate. No CPU/Pi fallback, automatic model download or remote logging
occurs. Training uses train/validation only, chooses the best validation-loss
checkpoint and saves the adapter. Metadata retains dataset/source hashes,
model revision, prompt/template hashes, settings, package versions, GPU details,
metrics and `pip freeze`. Interrupted/failed runs cannot be exported as completed.

## GGUF export on the development workstation

Check out/build llama.cpp at the **same commit as the Pi runtime**, recorded in
`models/manifest.json`. Build the `llama-quantize` target on the workstation.
Use a separate converter environment: the pinned llama.cpp conversion requirements
currently select CPU Torch 2.11.0 and must not overwrite the CUDA training stack.

```bash
git clone https://github.com/ggml-org/llama.cpp .local/llama-export
git -C .local/llama-export checkout 8345f333951c661d166b00e6f9362e553768f292
cmake -S .local/llama-export -B .local/llama-export/build \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF
cmake --build .local/llama-export/build --target llama-quantize -j 2
python3.11 -m venv .local/converter-venv
.local/converter-venv/bin/python -m pip install \
  -r .local/llama-export/requirements/requirements-convert_hf_to_gguf.txt
.local/training-venv/bin/python training/export.py \
  --base .local/training/base --run .local/training/runs/qwen3-pins-v1 \
  --llama-source .local/llama-export \
  --converter-python .local/converter-venv/bin/python \
  --output .local/training/export/qwen3-pins-v1
```

With Windows multi-configuration CMake, build `--config Release` and point
`export.py` at a build that places `llama-quantize.exe` in `build/bin`, or copy that
workstation executable there. This is a development binary; it is not deployed
to the ARM64 Pi.

[Export](../training/export.py) merges the adapter into clean, verified FP32 base
weights on CPU, never into the 4-bit training model. It preserves the tokenizer,
checks its template hash and licence, converts to F16 GGUF, then quantizes Q4_K_M
with the pinned converter/quantizer. The output includes candidate SHA256,
licence/provenance and a separate benchmark manifest. No baseline is overwritten.
Candidate GGUF conversion/quantization remain unrun until there is a trained adapter.

For a controlled training experiment, also export an **untuned control** from
the exact same pinned HF checkpoint and converter/quantizer. The source revision
of the previously downloaded third-party GGUF is not proven equivalent to this
HF revision, so comparison against that working baseline alone cannot attribute
an improvement specifically to tuning:

```bash
.local/training-venv/bin/python training/export.py --base-only \
  --base .local/training/base --llama-source .local/llama-export \
  --converter-python .local/converter-venv/bin/python \
  --output .local/training/export/qwen3-control
```

The control uses its own filename/model ID (`qwen3-0.6b-training-control-q4`),
so copying it to the Pi cannot overwrite either the working baseline or candidate.
Compare the tuned candidate against both the existing working baseline for
practical regression and this control for training-effect attribution. Converter,
quantizer and full-checkpoint hashes/revisions are included in both exports.

## Compare on the Pi and retain the baseline on regression

Do not run a sustained suite on the currently tested Pi until its cooling/power
conditions are resolved. It must also have no active imaging/guiding session.
Keep the 80 °C cutoff and measured memory limits. Future short probes can use
`--cases`, but incomplete suites cannot pass the promotion gate.

Copy the candidate GGUF and licence/provenance outside Git into the existing
`~/.local/share/pins-ai/models` directory. Verify its SHA256 against the export
metadata. Copy the candidate `benchmark-manifest.json` and prepared data manifest
to the development checkout on the Pi. Neither action enables AI or changes the
saved model selection.

Run the same gold suite with the same plugin/runtime builds, thread/context/output
settings and pauses. The script temporarily changes model configuration and
restores it in `finally`; an abrupt power loss still requires manual recovery.

```bash
python3 scripts/benchmark.py --models qwen3-0.6b-q4 --threads 1 \
  --context-tokens 2048 --max-output-tokens 128 --case-pause-seconds 30 \
  --case-file artifacts/training/data/gold-cases.json \
  --evaluation-manifest artifacts/training/data/dataset-manifest.json \
  --output benchmarks/results/qwen3-baseline-gold.json
python3 scripts/benchmark.py --models pins-qwen3-0.6b-specialized-q4 --threads 1 \
  --context-tokens 2048 --max-output-tokens 128 --case-pause-seconds 30 \
  --manifest /path/to/export/benchmark-manifest.json \
  --case-file artifacts/training/data/gold-cases.json \
  --evaluation-manifest artifacts/training/data/dataset-manifest.json \
  --output benchmarks/results/qwen3-candidate-gold.json
```

Historical interrupted results cannot establish a comparable before/after pair:
they predate the final prompt fix/thermal guard and do not contain the new binary
hashes. New reports record plugin/native SHA256, runtime commit, dataset-declared
prompt hash, all exact inputs, model provenance and telemetry. The prompt hash
comes from the prepared manifest; actual identical plugin binaries are checked
separately. Use `--plugin-runtime` if the installed plugin directory differs.

Review actual outputs separately for baseline and candidate. Fill independent
review files with reviewer name and Boolean `languageOk`, `meaningOk`,
`noFalseClaims`; include notes for failures. Regex checks are a smoke rubric,
not a full language or equipment-safety evaluation. Blank or incomplete human
reviews cannot promote a candidate.
The baseline also needs complete review before a no-regression claim is allowed.

```bash
python training/evaluate.py score --report benchmarks/results/qwen3-baseline-gold.json \
  --reviews baseline-reviews.json --output artifacts/training/baseline-score.json
python training/evaluate.py score --report benchmarks/results/qwen3-candidate-gold.json \
  --reviews candidate-reviews.json --output artifacts/training/candidate-score.json
python training/evaluate.py compare --baseline artifacts/training/baseline-score.json \
  --candidate artifacts/training/candidate-score.json --output artifacts/training/decision.json
```

The provisional chat gate defaults to 95% human-reviewed passing cases, all
critical no-action/availability cases passing, no previously passing case
regressing, no RSS/swap limit violation and at most 10% decode/latency regression.
With ten gold cases, 95% requires all ten to pass. Thresholds are configurable;
these are development defaults, not an agreed production acceptance contract or
measured tool-call accuracy. First content is not first useful output. A successful
comparison merely marks eligibility for manual selection; it never deploys or
activates a candidate. On any failure, retain the verified baseline. Preserve the
baseline GGUF/config for rollback even after a later manual switch.
For the control comparison, repeat the identical benchmark command using its
manifest and model ID, review/score it separately, and use that score as the
`compare --baseline`. The decision records `tuningAttributionSupported` only when
the full checkpoint and exporter provenance match and all gates pass.

## Actual MCP supervision, when available

[Schema importer](../training/import_tools.py) requires an actual exported
`tools/list` result with names, descriptions and `inputSchema`, plus separately
authored/reviewed JSONL examples. It validates each selected tool and argument
against that exact schema, validates recorded call/result relationships, rejects
unknown tools and external schema references, and records source/schema hashes.
No server is contacted and no operation is executed.

```bash
python training/import_tools.py --schema-snapshot actual-tools-list.json \
  --examples reviewed-tool-examples.jsonl --output artifacts/training/tools-staged.jsonl
```

Example records require `id`, `group` (scenario/tool combination shared across
translations and paraphrases), `locale`, `toolNames`, `reviewed:true`, provenance
and standard conversational `messages` with structured assistant `tool_calls`
and matching recorded `tool_call_id` responses where applicable. Do not use real
private logs or fabricate tool results. Include clarification/refusal turns,
negation, missing parameters and malicious-result interpretation in that future
corpus. Use a separately reviewed held-out tool gold set and group by scenario
plus tool combination.

Tool examples are staged separately and deliberately rejected by today's
chat-only trainer, whose deployed policy says tools are unavailable. Training
tool routing, schema-valid call scoring and deterministic permissions/confirmation
tests belong to the actual MCP integration phase. No tool accuracy, fine-tuning
improvement, Pi throughput for a tuned model or live hardware safety is claimed.

## Verified in this workspace

Data preparation and 16 pipeline tests passed, including the actual pinned Qwen
tokenizer/template and all bootstrap token lengths. One additional tiny random
Qwen architecture test passed through PEFT LoRA, gradient accumulation, validation
loss, checkpoint selection and adapter saving on CPU. Its temporary weights were
discarded; this was a backend integration fixture, not Qwen specialization.
Full CUDA/QLoRA, adapter merging, GGUF conversion and tuned-model Pi evaluation
remain unverified pending the real training run.
