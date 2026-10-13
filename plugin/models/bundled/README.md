# Bundled diagnostic model

These files are versioned in this repository using **Git LFS**. Run `git lfs pull`
after cloning; CI checks out LFS assets explicitly. The Debian build verifies the
size and SHA-256 in `../bundled.json` and includes the bytes in the package.
Installation does not download weights.

The base is Qwen/Qwen3-0.6B at `c1899de289a04d12100db370d81485cdf75e47ca`,
converted to Q4_K_M with llama.cpp `8345f333951c661d166b00e6f9362e553768f292`.
The separate F16 LoRA is checkpoint 700 from the incomplete 846-step training
run. No merge was performed. The Apache-2.0 base license is included here.
The synthetic PINS training examples are CC0-1.0; implementation code is MIT.

This is a diagnostic model, not a production quality acceptance. See
[the model report](../../../docs/model-report.md) for measured quality and thermal limitations.
