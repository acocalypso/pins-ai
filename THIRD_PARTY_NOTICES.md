# Third-party components

## Bundled Debian model

The Debian package includes the Git LFS assets in `models/bundled/`: our pinned
Qwen/Qwen3-0.6B base converted to Q4_K_M and the separate diagnostic checkpoint
700 F16 LoRA. These are not the optional Unsloth benchmark conversions described
below. The base revision, exact hashes and incomplete-training status are in
`plugin/models/bundled.json`; Apache-2.0 is included beside the weights. No merge occurs.
The model weights are versioned using LFS and copied into the Debian package;
installation performs no model downloads. [The model report](docs/model-report.md) records
the diagnostic live MCP results and unresolved quality limitations.

The following historical optional-candidate notes describe the original plugin
ZIP and benchmark installer, not the new bundled Debian payload.

No third-party managed packages are added. The plugin compiles against installed
PINS host contracts (`NINA.Plugin`, `NINA.Core`, `NINA.Profile`, and the host's MEF assembly);
those host DLLs are not redistributed in the plugin payload. PINS/NINA is MPL-2.0.

The optional, explicitly installed native runtime is llama.cpp, MIT:
https://github.com/ggml-org/llama.cpp/blob/8345f333951c661d166b00e6f9362e553768f292/LICENSE.
Its pinned source includes dependency licence notices; retain the source LICENSE
and `licenses/` when redistributing native builds. GCC's system OpenMP runtime is
provided by Debian's `libgomp1`; no system-library binaries are bundled here.

The model manifest pins Unsloth's Q4_K_M conversions of Qwen3-0.6B and
Qwen3.5-0.8B. Their respective upstream model licences are Apache-2.0:

- https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/LICENSE
- https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/LICENSE

These weights have undergone post-training; do not confuse them with
`Qwen3-0.6B-Base` or `Qwen3.5-0.8B-Base`. The embedded GGUF chat template and
llama.cpp's Jinja renderer are used, with thinking explicitly disabled. No vision
projector is downloaded or used for Qwen3.5.

Model weights are kept outside Git. Installers retain each GGUF's provenance,
hash and upstream licence beside it. Downloads require `--accept-licenses`.

The optional third candidate is Unsloth's pinned FunctionGemma 270M Q4_K_M
conversion. It was explicitly downloaded for loader/transport probes. It is
provided under the Gemma Terms of Use, not Apache-2.0:
https://ai.google.dev/gemma/terms, including the incorporated use restrictions:
https://ai.google.dev/gemma/prohibited_use_policy.
The installer retains the terms beside the GGUF. Retain terms, use restrictions
and the required Gemma notice if redistributing weights; this plugin ZIP contains
no model weights.

Google's model card says FunctionGemma is not intended for direct dialogue;
specialized function routing needs task-specific tuning. German command routing
and tool accuracy remain unevaluated without the real MCP contract:
https://ai.google.dev/gemma/docs/functiongemma/model_card.
# v2 dataset and tool contract

The supplied `training/dataset` original examples and generator scripts
are CC0-1.0 as declared in that package's README. Audited examples remain
synthetic, with no private observatory logs. New pins-ai implementation code
uses this repository's MIT licence.

`training/dataset/data/verified-tools-list.json` contains source-generated
tool descriptions/schemas from `nitr57/pins` revision
`d9b5dc6ba3874113589f58d88c20909cff599b45` (original snapshot
`14b50c4e09ef3904f5c8db5a29f05eb266e863a5`), whose implementation is MPL-2.0.
Retain that source attribution and applicable upstream notices. The audit
harness references the team's existing project without copying or modifying
its server source. Training dependency versions are unchanged.
