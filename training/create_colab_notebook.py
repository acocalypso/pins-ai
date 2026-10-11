"""Fill the skill-scaffolded Colab notebook with the existing verified training pipeline."""
import json
from textwrap import dedent
from common import ROOT


def main():
    path = ROOT / 'output/jupyter-notebook/pins-ai-v2-colab.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    cells = []
    def cell(kind, source):
        item = {'cell_type': kind, 'metadata': {}, 'source': dedent(source).strip().splitlines(keepends=True)}
        if kind == 'code': item.update(execution_count=None, outputs=[])
        cells.append(item)
    cell('markdown', '''
        # PINS AI v2 on Google Colab

        Use a hosted **GPU** Colab kernel in VS Code. The notebook delegates to the
        existing FP16/NF4 QLoRA scripts and audited MCP contracts at `d9b5dc6`.
        No MCP connection, equipment operation or Pi deployment occurs.

        Code is cloned/pulled from `acocalypso/pins-ai` on GitHub. Existing data
        in your Google Drive folder `pins-ai` is checked against that Git revision.
        No source ZIP re-upload is needed for code changes. In VS Code choose
        **Select Kernel → Colab → Auto Connect**, select a GPU runtime, then run
        the setup cells. Sign-in and Drive authorization occur in your session.

        Training is enabled for the requested cloud run. A fresh smoke
        and adapter reload must pass before the full run. No speedup or Colab GPU
        compatibility has been measured yet. GPU availability and runtime lifetime
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

        RUN_TRAINING = True  # Authorized cloud run; set False for validation only.
        RUN_EVALUATION = False  # Enable after training; no tuning against gold scores.
        RUN_HARD_GOLD = False  # One-shot challenge evaluation, not a tuning loop.
        DRIVE_FOLDER = Path('/content/drive/MyDrive/pins-ai')
        RUN_NAME = 'qwen3-pins-v2-colab-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
        REPOSITORY = 'https://github.com/acocalypso/pins-ai.git'
        BRANCH = 'main'
        PROJECT = Path('/content/pins-ai-git')
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
        print('Code revision:', git_revision)
        drive.mount('/content/drive')
        sys.path.insert(0, str(PROJECT / 'training'))
        from colab_workspace import use_drive_data
        drive_data = use_drive_data(PROJECT, DRIVE_FOLDER)
        audit = json.loads((PROJECT / 'PINS_MCP_AI_Training_v2/data/contract-audit.json').read_text())
        if audit['sourceRevision'] != 'd9b5dc6ba3874113589f58d88c20909cff599b45':
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
    ''')
    cell('code', '''
        run_pipeline(['training/fetch_base.py', '--download', '--accept-license'], 'fetch-base.log')
        run_pipeline(['PINS_MCP_AI_Training_v2/scripts/validate_dataset.py'], 'package-validation.log')
        run_pipeline(['training/prepare_v2.py', '--max-tokens', '1280'], 'prepare-sft.log')
        run_pipeline(['tests/v2_tests.py'], 'v2-tests.log')
        manifest = json.loads((PROJECT / 'artifacts/v2/sft/manifest.json').read_text())
        assert manifest['skipped'] == 0
        for split, info in manifest['splits'].items():
            print(split, ':', info['assistantTurns'], 'assistant turns; max tokens', info['tokens']['max'])
        (RUN_ROOT / 'prepared-manifest.json').write_text(json.dumps(manifest, indent=2))
    ''')
    cell('markdown', '''
        ## Fresh GPU smoke and adapter reload
        Run one optimizer step covering native tool calls, a conditional job poll,
        clarification, no-tool and authorization cases, multiple languages and the
        longest training example. Losses must be finite and the adapter must save
        and reload. Failures stop the workflow. This is training, not a quality score.
    ''')
    cell('code', '''
        SMOKE = RUN_ROOT / 'smoke'
        FULL = RUN_ROOT / 'full'
        if RUN_TRAINING:
            smoke_config = json.loads((PROJECT / 'training/config-v2.json').read_text())
            smoke_config['gradientAccumulation'] = 8
            smoke_path = RUN_ROOT / 'smoke-config.json'
            smoke_path.write_text(json.dumps(smoke_config, indent=2))
            run_pipeline(['training/train.py', '--base', '.local/training/base', '--dataset-mode', 'mcp',
                '--config', smoke_path, '--max-steps', '1', '--logging-steps', '1', '--output', SMOKE], 'smoke.log')
            run_pipeline(['training/evaluate_v2.py', '--suite', 'validation_smoke', '--per-language', '2',
                '--adapter', SMOKE / 'adapter', '--output', RUN_ROOT / 'smoke-reload.json'], 'smoke-reload.log')
            smoke_report = json.loads((RUN_ROOT / 'smoke-reload.json').read_text())
            assert smoke_report.get('completed')
        else:
            print('Training remains disabled. Change RUN_TRAINING in the configuration cell to opt in.')
    ''')
    cell('markdown', '''
        ## Full two-epoch run
        Use the unchanged comparison configuration: FP16/NF4, rank8/alpha16,
        microbatch1, accumulation16, context1280, seed3407, learning rate1e-4.
        Keep the GPU comparison fair before changing these settings. Logs show
        train_loss every five optimizer steps and eval_loss at each epoch.
        Checkpoints, metrics, dependency versions and the final best adapter are
        written to Drive. Interrupting a cell stops its child process. Abrupt VM
        deletion may leave metadata marked started; it never implies completion.
    ''')
    cell('code', '''
        if RUN_TRAINING:
            run_pipeline(['training/train.py', '--base', '.local/training/base', '--dataset-mode', 'mcp',
                '--config', 'training/config-v2.json', '--smoke-run', SMOKE,
                '--logging-steps', '5', '--output', FULL], 'training.log')
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
            ax.set(xlabel='Optimizer step', ylabel='Loss', title='PINS Qwen3 v2 training')
            ax.legend(); ax.grid(alpha=0.2); fig.tight_layout()
            fig.savefig(RUN_ROOT / 'loss-curves.png', dpi=150)
            plt.show()
            print((FULL / 'metrics-latest.json').read_text())
        else:
            print('No full-run metrics yet.')
    ''')
    cell('markdown', '''
        ## Unseen comparison (optional, after completion)
        Identical prompts/settings for base and v2; tools never execute. Default
        comparison is 30 multilingual cases including conditional polling. Enable
        RUN_HARD_GOLD once for the full 120 challenge cases; never tune the dataset
        or prompts against those results. Full 593-case eval_gold is available by
        removing the per-language subset flag. The v1 adapter is not in the source
        ZIP, so its cloud comparison remains pending unless separately transferred.
        GGUF export, quantized comparison, human review and Pi validation remain separate.
    ''')
    cell('code', '''
        if RUN_EVALUATION:
            completed = json.loads((FULL / 'training-run.json').read_text())
            assert completed['status'] == 'completed' and not completed['smokeOnly']
            for name, adapter in [('base', None), ('v2', FULL / 'adapter')]:
                extra = ['--adapter', adapter] if adapter else []
                run_pipeline(['training/evaluate_v2.py', '--per-language', '2', '--job-continuations',
                    *extra, '--output', RUN_ROOT / f'{name}-eval.json'], f'{name}-eval.log')
                if RUN_HARD_GOLD:
                    run_pipeline(['training/evaluate_v2.py', '--suite', 'hard_gold',
                        *extra, '--output', RUN_ROOT / f'{name}-hard-gold.json'], f'{name}-hard-gold.log')
            run_pipeline(['training/report_v2.py', '--directory', RUN_ROOT,
                '--training-run', FULL, '--smoke-run', SMOKE], 'report.log')
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
