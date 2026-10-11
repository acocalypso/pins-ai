# PINS v2 training with Colab in VS Code

Prepared notebook: [pins-ai-v2-colab.ipynb](../output/jupyter-notebook/pins-ai-v2-colab.ipynb).
This is cloud **training** only. Pi inference still belongs to the PINS plugin;
there is no cloud equipment access or MCP server in the notebook.

When switching from the older ZIP-based notebook, run this revised notebook from
the top on the connected T4. It uses a fresh Git directory and run name; existing
Drive data and prior outputs remain available. Future code updates need only a
notebook refresh and a new run, with Git pulling the current branch.

1. Open the notebook in VS Code. Sign in and choose **Select Kernel → Colab →
   Auto Connect**, then select a hosted GPU runtime. See the
   [official extension guide](https://github.com/googlecolab/colab-vscode).
2. Code is cloned from `https://github.com/acocalypso/pins-ai.git`, branch `main`,
   into `/content/pins-ai-git`. On an existing clean checkout, setup runs
   `git pull --ff-only origin main`. No source ZIP upload is required for code updates.
3. Existing data is read from `My Drive/pins-ai/PINS_MCP_AI_Training_v2/data`,
   `My Drive/pins-ai/data`, or the existing `pins-ai-training-0.1.0.zip` there.
   Every dataset/contract file must match the Git revision before any data is copied.
   Code is never imported from Drive. Adjust `DRIVE_FOLDER` if your data lives elsewhere.
   Training is enabled (`RUN_TRAINING = True`) for the authorized cloud run;
   set it to `False` for validation only. Approve Drive mounting in your session.
4. Run the cells in order. They install an isolated Python 3.11 environment,
   torch 2.7.1/cu126 and the existing pinned training requirements. The explicit
   base-download cell accepts the Apache-2.0 checkpoint license and verifies the
   pinned weights. Preflight, package/hash checks and tokenizer tests must pass.
5. A new GPU smoke and adapter reload precede two-epoch v2 training. Output is
   under `My Drive/pins-ai/runs/<RUN_NAME>/`. `training.log` shows training loss
   every five optimizer steps and validation loss at each epoch.

The notebook kernel can run Python 3.13; training still uses isolated Python 3.11.
Setup explicitly installs managed Python through uv and creates the environment
with Python's standard `venv`, avoiding uv's `--seed` resolution. Full setup output
is printed on errors. After a setup failure, reload the updated local notebook
and rerun only the setup cell, then continue below it. Keep the existing kernel,
project and Drive run folder; do not rerun Git/data setup or configuration.

The initial cloud configuration preserves FP16/NF4, context1280, batch1,
accumulation16, rank8/alpha16 and seed3407. A T4-class or newer GPU is a practical
candidate, but no particular GPU is guaranteed and no training speed is measured
yet. The notebook checks the allocated GPU and actual kernels; a failure stops
the workflow. Colab's current resources and lifetime depend on allocation and
plan: see [its resource FAQ](https://research.google.com/colaboratory/faq.html).

Input/data/base weights are read from the VM filesystem. Small run logs, metrics,
checkpoints and adapters are written to Drive because the VM is temporary. The
last cell can also archive the result for download. Cell interruption terminates
the child job. A disconnected/deleted VM may leave an incomplete run marked
`started`; do not treat it as completed. Existing runs are never overwritten,
and this workflow currently requires a fresh run rather than automatic resume.

Enable `RUN_EVALUATION` after a completed run for an identical base/v2 comparison.
`RUN_HARD_GOLD` enables a one-shot 120-case challenge evaluation. Never change
training or prompts to optimize that holdout. The prior v1 adapter is preserved
locally and is absent from Git and the data bundle; its cloud comparison requires
a separate transfer. GGUF export/evaluation and Pi validation remain pending.

Verified locally: notebook JSON/code-cell syntax, Git/Drive data checks
(directory, ZIP and mismatch rejection), audited package
contracts, and preservation of the stopped local run.
Not verified: Colab sign-in/Drive mount, remote dependency installation, GPU
smoke, training speed/losses or cloud model quality. No Colab job has been launched.
