# PINS v3 training with Colab in VS Code

Prepared notebook: [pins-ai-v2-colab.ipynb](../output/jupyter-notebook/pins-ai-v2-colab.ipynb).
This is cloud **training** only. Pi inference still belongs to the PINS plugin;
there is no cloud equipment access or MCP server in the notebook.

Start this v3 experiment from the pinned **Qwen/Qwen3-0.6B base model** with a
fresh LoRA adapter. The notebook already does this; it does not continue training
the v2 adapter. `PRIOR_ADAPTER` is used only for the optional comparison after
training. Preserve v2 as the baseline. The updated data passed local contract,
tokenizer and isolation checks; the new T4 smoke/reload must still pass before
full training. MCP does not need to be running on the Pi for this cloud run.

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
   Original dataset files must match Git checksums. Updated audited contracts stay in Git;
   old contract files in the Drive bundle are ignored.
   Code is never imported from Drive. Adjust `DRIVE_FOLDER` if your data lives elsewhere.
   Training defaults to disabled (`RUN_TRAINING = False`) while the revised safety
   curriculum is reviewed. Set it to `True` explicitly for a new cloud run after
   preflight. Approve Drive mounting in your session.
4. Run the cells in order. They install an isolated Python 3.11 environment,
   torch 2.7.1/cu126 and the existing pinned training requirements. The explicit
   base-download cell accepts the Apache-2.0 checkpoint license and verifies the
   pinned weights. Preflight, package/hash checks and tokenizer tests must pass.
5. A new GPU smoke and adapter reload precede up to three epochs of v3 training. Output is
   under `My Drive/pins-ai/runs/<RUN_NAME>/`. `training.log` shows training loss
   every five optimizer steps and validation loss every100 optimizer steps.

The notebook kernel can run Python 3.13; training still uses isolated Python 3.11.
Setup explicitly installs managed Python through uv and creates the environment
with Python's standard `venv`, avoiding uv's `--seed` resolution. Full setup output
is printed on errors. After a setup failure, reload the updated local notebook
and rerun only the setup cell, then continue below it. Keep the existing kernel,
project and Drive run folder; do not rerun Git/data setup or configuration.

The v3 configuration uses FP16/NF4, context1536, batch1, accumulation16,
rank8/alpha16, seed3407 and learning rate5e-5. Early stopping has patience2
and minimum improvement0.002. The best validation checkpoint is saved.
Train-only device contrasts and competing-tool contexts are generated from Git
code; original validation is unchanged apart from refreshed MCP contracts. A T4-class or newer GPU is a practical
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

Enable `RUN_EVALUATION` for the full583-case validation development comparison
of base, prior v2 (when its adapter is present on Drive) and new v3. This is
selection/development evidence, not an unseen score. `PRIOR_ADAPTER` defaults
to the completed v2 run on Drive; adjust it if that folder was moved.
`RUN_FINAL_GOLD` is for a final release check only, not iterative tuning.
Original test/eval_gold/hard_gold are never training sources. Independent,
human-authored multilingual holdout review is still pending.
GGUF export/evaluation, plugin chat/gateway binding and Pi validation remain pending.
See [v3 preparation and v2 measured results](training-v3.md).

Verified locally: notebook JSON/code-cell syntax, Git/Drive data checks
(directory, ZIP and mismatch rejection), audited package
contracts, and preservation of the stopped local run.
Not verified: Colab sign-in/Drive mount, remote dependency installation, GPU
smoke, training speed/losses or cloud model quality. No Colab job has been launched.
