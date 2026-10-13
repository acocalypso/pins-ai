"""Fill the skill-scaffolded Colab notebook with the existing verified training pipeline."""
import json
from textwrap import dedent
from common import ROOT


def main():
    path = ROOT / 'training/notebooks/pins-ai-colab.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    cells = []
    def cell(kind, source):
        item = {'cell_type': kind, 'metadata': {}, 'source': dedent(source).strip().splitlines(keepends=True)}
        if kind == 'code': item.update(execution_count=None, outputs=[])
        cells.append(item)
    cell('markdown', '''
        # PINS AI MCP training on Google Colab

        Use a hosted **GPU** Colab kernel in VS Code. The notebook delegates to the
        existing FP16/NF4 QLoRA scripts and audited MCP contracts at `6fef6c4`.
        The comprehensive train-only curriculum covers all 32 current MCP tools.
        Train-only device contrasts, safety classes and competing tool lists extend the originals;
        validation remains separate. Gold suites are never training inputs.
        No MCP connection, equipment operation or Pi deployment occurs.

        Default mode starts a **fresh** adapter from the pinned base with the current
        curriculum. Set `TRAINING_MODE = 'resume'` only to continue an interrupted
        run with identical data/configuration, restoring adapter, optimizer,
        scheduler, FP16 scaler and RNG state. New examples require a fresh run.
        `PRIOR_ADAPTER` remains comparison-only.

        Code is cloned/pulled from `acocalypso/pins-ai` on GitHub. Existing data
        in your Google Drive folder `pins-ai` is checked against that Git revision.
        No source ZIP re-upload is needed for code changes. In VS Code choose
        **Select Kernel → Colab → Auto Connect**, select a GPU runtime, then run
        the setup cells. Sign-in and Drive authorization occur in your session.

        In resume mode, set `RESUME_SOURCE` to the original Drive run folder (or results ZIP).
        Set `RUN_TRAINING = True` deliberately to train in a new output folder.
        A short GPU smoke and adapter reload must pass first; resume mode uses a
        one-step continuation probe to check restored state.
        This updated resume workflow has not yet been executed on Colab. GPU availability and runtime lifetime
        depend on Colab allocation. Use a new run name for each attempt.
    ''')
    cell('code', '''
        from pathlib import Path
        import sys, importlib.util, json, subprocess, os, time, zipfile, hashlib
        from datetime import datetime, timezone

        if sys.platform != 'linux' or importlib.util.find_spec('google.colab') is None:
            raise RuntimeError('Select a hosted Colab kernel; this notebook must not train on your PC.')
        subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
                        '--format=csv,noheader'], check=True)

        RUN_TRAINING = False  # Enable explicitly after validating environment/data.
        TRAINING_MODE = 'fresh'  # 'resume' requires identical original data/configuration.
        RUN_EVALUATION = False  # Enable after training; no tuning against gold scores.
        RUN_FINAL_GOLD = False  # Final release measurement only, never an iterative tuning loop.
        DRIVE_FOLDER = Path('/content/drive/MyDrive/pins-ai')
        RESUME_SOURCE = DRIVE_FOLDER / 'runs/qwen3-pins-v3-colab-20261011-165756'
        # Alternatively: DRIVE_FOLDER / 'qwen3-pins-v3-colab-20261011-165756-results.zip'
        RUN_NAME = 'qwen3-pins-v6-colab-'  + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
        TRAIN_CONFIG = 'training/configs/config-v6.json'
        SFT_DATA = 'artifacts/v6/sft'
        PRIOR_ADAPTER = DRIVE_FOLDER / 'runs/qwen3-pins-v2-colab-20261011-131536/full/adapter'
        REPOSITORY = 'https://github.com/acocalypso/pins-ai.git'
        BRANCH = 'main'
        PROJECT = Path('/content/pins-ai-git')
        if TRAINING_MODE not in ('resume', 'fresh'):
            raise ValueError('TRAINING_MODE must be resume or fresh')
        print('Mode:', TRAINING_MODE, '| source:', RESUME_SOURCE)
        print('Run:', RUN_NAME, '| training enabled:', RUN_TRAINING)
    ''')
    cell('markdown', '''
        ## Pull code, mount Drive and verify existing data
        The cloud VM has its own filesystem. Mounting Drive grants this notebook
        access to your Drive; the workflow reads data and writes run outputs
        only in `DRIVE_FOLDER`. Base weights and prepared data stay on the VM.
        Drive outputs survive VM deletion. Do not include Pi passwords or tokens.
    ''')
    cell('code', '''
        from google.colab import drive
        if not PROJECT.exists():
            subprocess.run(['git', 'clone', '--branch', BRANCH, REPOSITORY, str(PROJECT)], check=True)
        else:
            if not (PROJECT / '.git').is_dir():
                raise RuntimeError('PROJECT exists without Git; choose a fresh project path.')
            remote = subprocess.check_output(['git', 'remote', 'get-url', 'origin'], cwd=PROJECT, text=True).strip()
            if remote != REPOSITORY:
                raise RuntimeError('Existing project has a different Git remote.')
            branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=PROJECT, text=True).strip()
            if branch != BRANCH:
                raise RuntimeError('Existing project is on a different Git branch.')
            dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=PROJECT, text=True)
            if dirty.strip():
                raise RuntimeError('Existing Git checkout has changes; preserve them before pulling.')
            subprocess.run(['git', 'pull', '--ff-only', 'origin', BRANCH], cwd=PROJECT, check=True)
        git_revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip()
        if not (PROJECT / 'training/training_resume.py').is_file():
            raise RuntimeError('Publish the resume support to GitHub before running this notebook.')
        print('Code revision:', git_revision)
        drive.mount('/content/drive')
        sys.path.insert(0, str(PROJECT / 'training'))
        from colab_workspace import use_drive_data
        drive_data = use_drive_data(PROJECT, DRIVE_FOLDER)
        audit = json.loads((PROJECT / 'training/dataset/data/contract-audit.json').read_text())
        if audit['sourceRevision'] != '6fef6c4bd73a8e1f24a36235dba6cbb9eff3d3d2':
            raise ValueError('Rebuild this notebook/bundle for the newer audited MCP revision')
        RUN_ROOT = DRIVE_FOLDER / 'runs' / RUN_NAME
        RUN_ROOT.mkdir(parents=True, exist_ok=False)
        (RUN_ROOT / 'source-git.json').write_text(json.dumps({
            'repository': REPOSITORY, 'branch': BRANCH, 'gitRevision': git_revision,
            'driveData': drive_data,
            'mcpRevision': audit['sourceRevision'], 'runName': RUN_NAME,
            'startedAt': datetime.now(timezone.utc).isoformat(), 'hardwareOperations': False
        }, indent=2))
        os.chdir(PROJECT)
        print('Verified Git code and Drive data; persistent outputs:', RUN_ROOT)
    ''')
    cell('markdown', '''
        ## Isolated, pinned training environment
        Keep Colab's notebook kernel intact. Create a separate Python 3.11
        environment with the same CUDA 12.6 PyTorch and training package versions
        that worked locally. Downloads occur only when these cells are executed.
        Driver/kernel compatibility is checked before any model training.
    ''')
    cell('code', '''
        def setup_command(arguments):
            result = subprocess.run(list(map(str, arguments)), cwd=PROJECT,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace')
            print(result.stdout, flush=True)
            if result.returncode:
                raise RuntimeError(f'Setup failed (exit {result.returncode}); full output is above.')
            return result.stdout.strip()

        print('Notebook interpreter:', sys.executable, sys.version)
        setup_command([sys.executable, '-m', 'pip', 'install', '--quiet', 'uv'])
        UV = [sys.executable, '-m', 'uv']
        setup_command([*UV, '--version'])
        # Download Python explicitly, then use its standard venv/ensurepip.
        # This avoids uv venv's additional --seed package resolution step.
        setup_command([*UV, 'python', 'install', '3.11'])
        managed_python = setup_command([*UV, 'python', 'find', '--managed-python', '3.11'])
        if not Path(managed_python).is_file():
            raise RuntimeError('uv did not return a valid Python executable path.')
        setup_command([managed_python, '-m', 'venv', str(PROJECT / '.local/training-venv')])
        PYTHON = PROJECT / '.local/training-venv/bin/python'
        setup_command([PYTHON, '-c', 'import sys; assert sys.version_info[:2] == (3, 11); print(sys.version)'])
        setup_command([PYTHON, '-m', 'pip', 'install', '--quiet', 'torch==2.7.1',
                       '--index-url', 'https://download.pytorch.org/whl/cu126'])
        setup_command([PYTHON, '-m', 'pip', 'install', '--quiet', '-r', 'training/requirements.txt'])
        setup_command([PYTHON, 'training/preflight.py'])
    ''')
    cell('code', '''
        def run_pipeline(arguments, log_name):
            """Interactive notebook job, persistent log; stop child on cell interrupt."""
            log_path = RUN_ROOT / log_name
            if log_path.exists():
                raise RuntimeError(f'Log exists; preserve it and use a fresh run: {log_path}')
            started = time.monotonic()
            process = subprocess.Popen([str(PYTHON), '-u', *map(str, arguments)],
                cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace', bufsize=1)
            last_status = started
            try:
                with log_path.open('w', encoding='utf-8') as log:
                    for line in process.stdout:
                        log.write(line); log.flush()
                        if '[metrics]' in line or line.startswith('PASS') or line.startswith('{"id"'):
                            print(line.rstrip(), flush=True)
                        elif time.monotonic() - last_status > 30:
                            print(line.rstrip(), flush=True); last_status = time.monotonic()
                exit_code = process.wait()
                if exit_code:
                    print(log_path.read_text(encoding='utf-8')[-8000:])
                    raise RuntimeError(f'Pipeline failed ({exit_code}); inspect {log_path}')
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try: process.wait(timeout=15)
                    except subprocess.TimeoutExpired: process.kill(); process.wait()
                raise
            finally:
                if process.stdout is not None: process.stdout.close()
            print(f'Finished in {time.monotonic() - started:.1f}s; log: {log_path}')
    ''')
    cell('markdown', '''
        ## Download the pinned base and validate/export data
        Executing the next cell downloads the Apache-2.0 Qwen3 0.6B checkpoint
        using the repository's existing explicit license/download flags and
        verifies its hash. No unpinned model or automatic dependency upgrade.
        Only train/validation are exported. Test, eval_gold and hard_gold stay
        outside the training examples. All original package hashes are checked.
        The default curriculum is **v6 multilingual chat proposals**: plain multilingual
        requests, clarification and dependent workflows; proposals do not grant
        hardware approval. Original datasets remain archived unchanged. This is a
        changed recipe: select `TRAINING_MODE = 'fresh'`, not an old checkpoint resume.
    ''')
    cell('code', '''
        run_pipeline(['training/fetch_base.py', '--download', '--accept-license'], 'fetch-base.log')
        run_pipeline(['training/dataset/scripts/validate_dataset.py'], 'package-validation.log')
        run_pipeline(['training/tests/chat_dataset_tests.py'], 'chat-data-validation.log')
        run_pipeline(['training/prepare_v2.py', '--curriculum', '--max-tokens', '1536',
                      '--output', SFT_DATA], 'prepare-sft.log')
        run_pipeline(['training/tests/v2_tests.py'], 'v2-tests.log')
        run_pipeline(['training/tests/curriculum_tests.py'], 'curriculum-tests.log')
        run_pipeline(['training/tests/comprehensive_tests.py'], 'comprehensive-tests.log')
        manifest = json.loads((PROJECT / SFT_DATA / 'manifest.json').read_text())
        assert manifest['skipped'] == 0
        for split, info in manifest['splits'].items():
            print(split, ':', info['assistantTurns'], 'assistant turns; max tokens', info['tokens']['max'])
        (RUN_ROOT / 'prepared-manifest.json').write_text(json.dumps(manifest, indent=2))
    ''')
    cell('markdown', '''
        ## Restore latest complete checkpoint (resume mode)
        The source run/ZIP is read only. A new copy lives in this run's
        `resume-source` folder on Drive. Incomplete newer checkpoints are reported;
        the numeric latest complete checkpoint is used, not necessarily the best.
        Completed runs, missing state and changed data/config/base are rejected.
        The original schedule and saved early-stopping state continue unchanged.
    ''')
    cell('code', '''
        RESUME_CHECKPOINT = None
        if TRAINING_MODE == 'resume':
            run_pipeline(['training/training_resume.py', '--source', RESUME_SOURCE,
                          '--output', RUN_ROOT / 'resume-source'], 'restore-checkpoint.log')
            restoration = json.loads((RUN_ROOT / 'resume-source/resume-restoration.json').read_text())
            RESUME_CHECKPOINT = Path(restoration['checkpoint'])
            print('Resume checkpoint:', RESUME_CHECKPOINT)
            print('Steps:', restoration['startStep'], 'of', restoration['targetSteps'])
            print('Best adapter:', restoration['relocatedBestCheckpoint'])
            print('Skipped incomplete:', restoration['skippedIncompleteCheckpoints'])
            original = json.loads((RESUME_CHECKPOINT.parent.parent / 'training-run.json').read_text())
            configured = json.loads((PROJECT / TRAIN_CONFIG).read_text())
            if original['config'] != configured:
                raise ValueError('Resume config differs from original. Use its exact original configuration.')
        else:
            print('Fresh mode: no checkpoint restoration.')
    ''')
    cell('markdown', '''
        ## GPU smoke and adapter reload
        Resume mode probes one step using the original full dataset and schedule.
        Fresh mode runs one optimizer step covering native tool calls, a conditional job poll,
        clarification, no-tool and authorization cases, multiple languages and the
        longest training example. Losses must be finite and the adapter must save
        and reload. Failures stop the workflow. This is training, not a quality score.
    ''')
    cell('code', '''
        SMOKE = RUN_ROOT / 'smoke'
        FULL = RUN_ROOT / 'full'
        if RUN_TRAINING:
            if TRAINING_MODE == 'resume':
                run_pipeline(['training/train.py', '--base', '.local/training/base', '--dataset-mode', 'mcp',
                    '--config', TRAIN_CONFIG, '--data', SFT_DATA, '--resume-from-checkpoint', RESUME_CHECKPOINT,
                    '--resume-smoke-steps', '1', '--logging-steps', '1', '--output', SMOKE], 'smoke.log')
            else:
                run_pipeline(['training/train.py', '--base', '.local/training/base', '--dataset-mode', 'mcp',
                    '--config', TRAIN_CONFIG, '--data', SFT_DATA, '--max-steps', '1', '--logging-steps', '1', '--output', SMOKE], 'smoke.log')
            run_pipeline(['training/evaluate_v2.py', '--suite', 'validation_smoke', '--prompt-mode', 'chat', '--per-language', '2',
                '--max-new-tokens', '192', '--adapter', SMOKE / 'adapter', '--output', RUN_ROOT / 'smoke-reload.json'], 'smoke-reload.log')
            smoke_report = json.loads((RUN_ROOT / 'smoke-reload.json').read_text())
            assert smoke_report.get('completed')
        else:
            print('Training remains disabled. Change RUN_TRAINING in the configuration cell to opt in.')
    ''')
    cell('markdown', '''
        ## Full run, up to three epochs with early stopping
        Use FP16/NF4, rank8/alpha16, microbatch1, accumulation16, context1536,
        seed3407, learning rate5e-5. Validation/checkpoints every100 optimizer
        steps. Select checkpoints by generated tool-balanced exact accuracy,
        penalizing unwanted proposals and truncated outputs, rather than eval_loss.
        Stop after four checks without selection-score improvement of at least0.005.
        Generated validation uses a deterministic development-only sample; no gold
        cases are used for checkpoint selection. This is not a hardware safety proof.
        This changes data and training settings together; results are an engineering
        comparison, not a controlled attribution experiment. Logs show train_loss
        every five optimizer steps and eval_loss at each validation check, plus
        eval_tool_exact, eval_tool_macro_exact, eval_schema_valid,
        eval_unsafe_proposals, eval_generation_truncated and eval_selection_score.
        Changed data/configuration requires fresh training from the pinned base.
        To continue an old run exactly, use its original configuration and source
        revision; do not substitute this v6 profile for the old recipe.
        Checkpoints, metrics, dependency versions and the final best adapter are
        written to Drive. Interrupting a cell stops its child process. Abrupt VM
        deletion may leave metadata marked started; it never implies completion.
    ''')
    cell('code', '''
        if RUN_TRAINING:
            resume_args = ['--resume-from-checkpoint', RESUME_CHECKPOINT] if TRAINING_MODE == 'resume' else []
            run_pipeline(['training/train.py', '--base', '.local/training/base', '--dataset-mode', 'mcp',
                '--config', TRAIN_CONFIG, '--data', SFT_DATA, '--smoke-run', SMOKE,
                *resume_args, '--logging-steps', '5', '--output', FULL], 'training.log')
            result = json.loads((FULL / 'training-run.json').read_text())
            assert result['status'] == 'completed' and not result['smokeOnly']
            print('Selected checkpoint:', result['bestModelCheckpoint'])
            print('Measured losses:', result['metrics'])
        else:
            print('Full training not started.')
    ''')
    cell('markdown', '''
        ## Inspect losses
        Re-run this cell to inspect persisted metrics. Training loss is a logging
        window average; validation loss is measured separately. Their absolute
        values are not directly interchangeable, and neither proves safe tool use.
    ''')
    cell('code', '''
        metrics_path = FULL / 'metrics.jsonl'
        if metrics_path.is_file():
            import matplotlib.pyplot as plt
            metrics = [json.loads(line) for line in metrics_path.read_text().splitlines()]
            fig, ax = plt.subplots(figsize=(8, 4))
            for event, key, label in [('train_window', 'train_loss', 'Training (window average)'),
                                       ('validation', 'eval_loss', 'Validation')]:
                points = [m for m in metrics if m['event'] == event]
                ax.plot([m['step'] for m in points], [m[key] for m in points], marker='o', label=label)
            ax.set(xlabel='Optimizer step', ylabel='Loss', title='PINS Qwen3 v6 training')
            ax.legend(); ax.grid(alpha=0.2); fig.tight_layout()
            fig.savefig(RUN_ROOT / 'loss-curves.png', dpi=150)
            plt.show()
            generated = [m for m in metrics if m['event'] == 'validation' and 'eval_tool_exact' in m]
            if generated:
                fig, ax = plt.subplots(figsize=(8, 4))
                for key in ('eval_tool_exact', 'eval_tool_macro_exact', 'eval_schema_valid'):
                    ax.plot([m['step'] for m in generated], [m[key] for m in generated], marker='o', label=key)
                ax.set(xlabel='Optimizer step', ylabel='Rate', ylim=(0, 1), title='Development tool accuracy')
                ax.legend(); ax.grid(alpha=0.2); fig.tight_layout()
                fig.savefig(RUN_ROOT / 'tool-accuracy-curves.png', dpi=150)
                plt.show()
            print((FULL / 'metrics-latest.json').read_text())
        else:
            print('No full-run metrics yet.')
    ''')
    cell('markdown', '''
        ## Development comparison (optional, after completion)
        Identical prompts/settings for base and adapters; tools never execute. Default
        comparison uses all583 validation conversations for base, the prior v2
        adapter (if present on Drive), and the new adapter. Validation is development
        evidence, not an unseen quality claim. The final gold suites have already
        been inspected; do not use them to guide iterative changes. Enable
        RUN_FINAL_GOLD only for a final release comparison. A newly authored,
        independently reviewed holdout is still needed. V1 comparison requires
        separately transferring that preserved local adapter.
        GGUF export, quantized comparison, human review and Pi validation remain separate.
    ''')
    cell('code', '''
        if RUN_EVALUATION:
            completed = json.loads((FULL / 'training-run.json').read_text())
            assert completed['status'] == 'completed' and not completed['smokeOnly']
            comparisons = [('v6', FULL / 'adapter'), ('base', None)]
            if PRIOR_ADAPTER.is_dir(): comparisons.insert(1, ('prior-v2', PRIOR_ADAPTER))
            else: print('Prior v2 adapter not found on Drive; its comparison is pending.')
            for name, adapter in comparisons:
                extra = ['--adapter', adapter] if adapter else []
                run_pipeline(['training/evaluate_v2.py', '--suite', 'validation_smoke', '--prompt-mode', 'chat',
                    '--max-new-tokens', '192', *extra, '--output', RUN_ROOT / f'{name}-eval.json'], f'{name}-eval.log')
                if RUN_FINAL_GOLD:
                    run_pipeline(['training/evaluate_v2.py', '--suite', 'hard_gold',
                        *extra, '--output', RUN_ROOT / f'{name}-hard-gold.json'], f'{name}-hard-gold.log')
            print('Comparison JSON/logs saved to Drive; inspect strict argument and per-language scores.')
        else:
            print('Evaluation disabled; gold cases have not been used for training.')
    ''')
    cell('markdown', '''
        ## Retrieve the result
        Outputs are already in your Drive run folder. This cell additionally makes
        an archive containing the adapter, saved checkpoints, loss curves, logs and
        provenance. Base weights, the environment and hardware credentials are not
        included. An interrupted run must not be labeled a completed model.
        After downloading, verify training-run.json status before local evaluation/export.
    ''')
    cell('code', '''
        if FULL.is_dir():
            import shutil
            if (DRIVE_FOLDER / (RUN_NAME + '-results.zip')).exists():
                raise FileExistsError('Results ZIP already exists; preserve it before archiving again.')
            result_zip = Path(shutil.make_archive(str(DRIVE_FOLDER / (RUN_NAME + '-results')), 'zip',
                                                  root_dir=RUN_ROOT))
            print('Results:', result_zip)
            print('SHA256:', hashlib.sha256(result_zip.read_bytes()).hexdigest())
        else:
            print('No full training outputs to archive.')
    ''')
    notebook['cells'] = cells
    notebook['metadata']['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    notebook['metadata']['language_info'] = {'name': 'python'}
    for index, item in enumerate(cells): item['id'] = f'pins-colab-{index:02d}'
    path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
