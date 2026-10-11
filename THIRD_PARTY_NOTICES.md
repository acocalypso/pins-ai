# Third-party components

No third-party managed packages are added. The plugin compiles against installed
PINS host contracts (`NINA.Plugin`, `NINA.Core`, and the host's MEF assembly);
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
