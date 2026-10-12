# Checkpoint 700: local inference

This is an inference-only development harness for the existing PINS AI project.
It neither starts training nor connects to an MCP server. Production model ownership
and lifecycle remain in the PINS plugin. No adapter merge, upload or model download
is performed. The original checkpoint and base weights are read only.

## Detected model and checkpoint

- Base: `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`.
- Base weights SHA256: `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`.
- Cached base folder: `.local/training/base` (already present; no download needed).
- Checkpoint: `.local/training/resume-v3/source/full/checkpoints/checkpoint-700`.
- Saved run: step700 of846, metadata still `started`; this is a partially trained
  diagnostic model, not a completed training release.
- Training: PEFT QLoRA, NF4/double quantization with FP16 compute, rank8/alpha16,
  dropout0.05, seven attention/MLP projection targets, trained context1536.
- Adapter: `adapter_model.safetensors`, 20,236,472 bytes, 392 finite LoRA tensors.
  SHA256: `53e795a825b381f5756a5d39642f9b052a4087f31b46457ab2fb5ba2c7f7756e`.
- Also present: `adapter_config.json`, complete saved tokenizer, chat template,
  trainer state, optimizer, scheduler, FP16 scaler and RNG state. Inference never
  deserializes the optimizer/RNG files or training-arguments pickles.

The adapter's base reference is a training-relative `.local/training/base`, not
a Hub identifier. Supply `--base` explicitly. The existing `verify_base` routine
checks all cached base files against original run provenance. The loader validates
the adapter configuration, finite weights, saved vocabulary and training template.
PEFT then applies the adapter to the existing base without merging it.

## Environment and hardware

Use the existing `.local/training-venv/Scripts/python.exe` (Python3.11). Installed
recipe: torch2.7.1+cu126, transformers4.57.6, peft0.18.0, accelerate1.12.0,
bitsandbytes0.49.1, safetensors0.7.0 and jsonschema4.25.1. These already exist;
there is no new dependency requirement. The pinned list is `training/requirements.txt`.

The tested GPU is an NVIDIA GTX1050Ti with4GB VRAM. `--backend auto` chooses
CUDA/NF4 when CUDA is available, otherwise CPU/FP32. `--backend cuda-nf4` requires
CUDA. CUDA uses the original NF4/double-quant/FP16 recipe and eager attention,
appropriate for this older GPU. An allocation failure stops with an error rather
than changing precision silently. Reduce generation length or explicitly select
CPU. Keep other GPU jobs idle when testing longer conversations.

`--backend cpu-fp32` loads unquantized base weights with the adapter. Allow roughly
2.3GiB for base parameters plus tokenizer, loading and generation overhead; several
GiB of free host RAM are needed. The loader warns that CPU generation may be slow.
This workstation has about32GiB RAM. Pi deployment/ARM inference and long-context
memory limits are not validated by this task.

## Tokenizer and special tokens

The saved checkpoint tokenizer, not the unmodified base tokenizer, formats input.
Its vocabulary must match the verified base and its chat-template fingerprint
must equal the training metadata. Native Qwen rendering uses `tools=...`,
`add_generation_prompt=True` and `enable_thinking=False`; input tokenization does
not add special tokens a second time.

Detected saved values are BOS=None, PAD=151645, EOS=151645. The original base
configuration has BOS=151643 and PAD=None. The loader derives the actual values
from the saved tokenizer, bounds-checks them against model embeddings and aligns
both model and generation configurations **in memory only**. These IDs are not
hardcoded in the implementation. The model is in evaluation mode, parameters have
gradients disabled and generation runs under `torch.inference_mode()`.

Loading uses `dtype`, not the deprecated `torch_dtype`. Greedy decoding explicitly
neutralizes inherited base sampling settings; `--sample` enables the supplied
temperature/top-p. No silent prompt truncation is performed: prompt plus requested
generation must fit the1536-token training context.

## Commands (PowerShell, from pins-ai)

Basic deterministic inference:

```powershell
.local/training-venv/Scripts/python.exe scripts/test_model.py `
  --base .local/training/base `
  --checkpoint .local/training/resume-v3/source/full/checkpoints/checkpoint-700 `
  --prompt "Hello. Introduce yourself briefly." --deterministic --max-new-tokens 128
```

Interactive conversation: replace `--prompt "..."` with `--interactive`; use
`/quit` or Ctrl+C to exit. A parsed tool proposal is displayed but never executed;
use the API or a new `--messages` history to supply any simulated feedback.

Sampling: add `--sample --temperature 0.7 --top-p 0.9` instead of `--deterministic`.
Greedy decoding is the default. `--system "..."` overrides the existing training
policy for a single-prompt or interactive session. `--messages conversation.json`
instead accepts a full native conversation including system and tool messages.
`--tools tools.json` accepts a JSON list of bare tool definitions or function
wrappers. No schema is executed and no external connection is made.

Smoke suite and persistent evidence (choose a new output filename on reruns):

```powershell
.local/training-venv/Scripts/python.exe scripts/test_model.py `
  --base .local/training/base `
  --checkpoint .local/training/resume-v3/source/full/checkpoints/checkpoint-700 `
  --backend cuda-nf4 --smoke --deterministic --max-new-tokens 128 `
  --output .local/inference-checkpoint-700/smoke-new.json

.local/training-venv/Scripts/python.exe tests/local_inference_tests.py
```

Smoke exits2 if an A/B/C quality check fails, even when loading and generation
work. Runtime/load errors exit1; Ctrl+C exits130. Existing report files are never
overwritten. The script closes its model and releases CUDA cached allocations on
exit. Evidence records package versions, backend, special tokens, timings, raw
output, parsing failures and before/after hashes of every checkpoint file.

CPU command: add `--backend cpu-fp32` to the basic inference command above.

## Measured smoke results

Measured on this workstation on2026-10-12, CUDA/NF4, deterministic, max128 tokens,
using the existing training policy. No output was repaired and no tool executed.

| Test | Actual outcome |
|---|---|
| A: introduction | Pass. `I am PINS, a device monitoring and controlling AI.` |
| B: strict JSON | Fail. Returned a Markdown fenced JSON block; direct `json.loads` fails. Inside the fences: `{"status":"ok","message":""}`. Fences were not stripped. |
| C: simulated tool | Pass for native format/name/empty arguments. Output below. |
| D: supplied simulated feedback | Generated `PINS reports that the requested operation completed.` after supplied `{"status":"ok","uptime_seconds":42}`. This is a mechanical continuation check, not proof of correct/safe reasoning. |

Native C output:

```text
<tool_call>
{"name": "get_server_status", "arguments": {}}
</tool_call><|im_end|>
```

The synthetic `get_server_status` schema is only a format probe, not a claim that
this tool exists on your server. This demonstrates one native tool proposal, not
general MCP accuracy, authorization, multilingual reliability or safe hardware use.
Strict JSON-only instruction following remains a discovered limitation.

Machine-readable reports are under ignored `.local/inference-checkpoint-700/`.
Final GPU evidence: `smoke-gpu.json`. Model load took4.188seconds (excluding
provenance/tokenizer checks), peak PyTorch GPU allocation was890.25MiB and these
short generations measured4.33-5.15tokens/second. This is not total device usage
or a long-context memory/performance guarantee.

CPU basic inference was also executed: `basic-cpu.json`, model load3.125seconds,
15 output tokens in4.797seconds (3.13tokens/second). Actual text was
`I am PINS, a toolset for planetary navigation and imaging.` Loading/generation
worked; that description was not judged accurate. CPU FP32 and GPU NF4 outputs
can differ. An initial CPU probe found PEFT's default automatic dispatch moved
the model to CUDA, causing a device mismatch. The loader now explicitly pins both
base and adapter device maps; the repeated CPU probe passed without that mismatch.

The GPU suite generates without truncation and file hashes confirm the checkpoint
is unchanged. The seven offline interface tests cover native parsing, malformed
and non-native calls, schema errors, multiple calls, duplicate tools and feedback
history preservation.

Not verified: full CPU A/B/C suite, long-context generation, sampling quality,
interactive terminal behavior, broad multilingual/tool accuracy, Pi inference,
or any live MCP connection. Sampling and interactive options are implemented but
were not used to claim additional measured quality results.

## Reusable interface for subsequent integration

`training/local_inference.py` exposes `LocalInference.generate(messages, tools,
max_new_tokens, temperature, top_p, deterministic)` and `append_tool_result`.

```python
import sys
sys.path.insert(0, 'training')  # from the repository root, matching existing scripts
from local_inference import LocalInference, append_tool_result

tools = [{'name': 'get_server_status', 'description': 'Simulated only',
          'parameters': {'type': 'object', 'properties': {}, 'required': []}}]
messages = [{'role': 'system', 'content': 'Use the supplied tool to check status.'},
            {'role': 'user', 'content': 'Check the current server status.'}]
with LocalInference('.local/training/base',
                    '.local/training/resume-v3/source/full/checkpoints/checkpoint-700') as model:
    response = model.generate(messages, tools=tools)
    print(response['rawText'])
    if response['toolCalls'] and not response['toolCallError']:
        messages = append_tool_result(messages, response, 'get_server_status', {'status': 'ok'})
        followup = model.generate(messages, tools=tools)
        print(followup['rawText'])
```

`rawText` preserves all generated markers, including EOS. `text` omits only a
generated terminal EOS token, without repairing model content. `toolCalls` contains
validated native `{name, arguments}` proposals; `toolCallError` records malformed
or unexposed/schema-invalid calls, with raw output retained. The parser follows the
training profile's one-tool-per-iteration constraint. Messages for template input
use Hugging Face's function representation, while **model output** is parsed from
Qwen's native XML wrappers; no OpenAI output format is assumed.

The next task should supply actual current MCP schemas and feed tool proposals
through the existing enforceable gateway in the PINS plugin. Authorization must
come from trusted user/application state, never model output. Only authorized
results should be appended to the conversation. Begin with read-only simulated
contract tests before explicitly authorized live operations. This task implements
none of that server connection or execution logic.

References: [HF chat/tool conventions](https://huggingface.co/docs/transformers/v4.57.3/en/chat_extras),
[NF4 configuration](https://huggingface.co/docs/transformers/v4.57.3/en/quantization/bitsandbytes).
