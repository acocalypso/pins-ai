"""Strict continuation contracts and restoration of data-only Colab checkpoint state."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile
import re

from common import sha256, write_json
from restore_results import checked_members

REQUIRED = ('adapter_model.safetensors', 'adapter_config.json', 'trainer_state.json',
            'optimizer.pt', 'scheduler.pt', 'scaler.pt', 'rng_state.pth')


def saved_resume_config(checkpoint, dataset, output):
    """Use the selected run's exact recipe only when its prepared data matches.

    Fresh-run presets are not continuation contracts. Do not silently import a
    different run's schedule, or overwrite its original checkpoint/configuration.
    """
    checkpoint, output = Path(checkpoint), Path(output)
    run = json.loads((checkpoint.parent.parent / 'training-run.json').read_text(encoding='utf-8'))
    if run.get('smokeOnly') or run.get('datasetMode') != 'mcp' or run.get('status') == 'completed':
        raise ValueError('Select an incomplete full MCP run as RESUME_SOURCE')
    if run['dataset'] != dataset:
        raise ValueError('Selected RESUME_SOURCE has different prepared data/profile. '
                         'Check the original Drive run folder; use its original source revision and data, '
                         'not a different training run or a newly changed dataset.')
    if output.resolve().is_relative_to(checkpoint.parent.parent.resolve()):
        raise ValueError('Write the resume configuration outside the restored source full run')
    config = run['config']
    if output.exists() and json.loads(output.read_text(encoding='utf-8')) != config:
        raise ValueError('Resume configuration already belongs to another source run; use a new output folder')
    write_json(output, config)
    return output


def required_files(run):
    return REQUIRED if run['precision'] == 'fp16' else tuple(n for n in REQUIRED if n != 'scaler.pt')


def validate_resume(checkpoint, config, dataset, base, method):
    checkpoint = Path(checkpoint)
    if checkpoint.parent.name != 'checkpoints':
        raise ValueError('Resume from a full saved training checkpoint, not an adapter alone')
    run = json.loads((checkpoint.parent.parent / 'training-run.json').read_text(encoding='utf-8'))
    state = json.loads((checkpoint / 'trainer_state.json').read_text(encoding='utf-8'))
    if run.get('smokeOnly') or run.get('datasetMode') != 'mcp' or run.get('status') == 'completed':
        raise ValueError('Resume requires an original full MCP training run')
    if (run['config'] != config or run['method'] != method or run['precision'] not in ('fp16', 'bf16', 'fp32')
            or config.get('precision', 'fp16') not in ('auto', run['precision'])):
        raise ValueError('Do not change optimizer/model/training configuration during continuation')
    if run['dataset'] != dataset or run['base'] != base:
        raise ValueError('Resume dataset/template/base provenance differs from original run')
    if not 0 < state['global_step'] < state['max_steps']:
        raise ValueError('Checkpoint has no remaining training steps')
    if state['train_batch_size'] != config['batchSize']:
        raise ValueError('Checkpoint batch size mismatch')
    files = {name: sha256(checkpoint / name) for name in required_files(run)}
    best = Path(state['best_model_checkpoint'])
    if not best.is_dir() or not (best / 'adapter_model.safetensors').is_file():
        raise ValueError('Best checkpoint must be relocated to a preserved local copy')
    return run, state, {'checkpoint': str(checkpoint.resolve()), 'checkpointFilesSha256': files,
                         'startStep': state['global_step'], 'targetSteps': state['max_steps'],
                         'bestMetricAtResume': state['best_metric'],
                         'originalHardware': run['hardware'], 'bitIdenticalToColab': False}


def select_checkpoint(names, read):
    """Select numerically newest complete full checkpoint, preserving best separately."""
    run = json.loads(read('full/training-run.json'))
    if run.get('smokeOnly') or run.get('datasetMode') != 'mcp' or run.get('status') == 'completed':
        raise ValueError('Source must be an incomplete full MCP training run')
    candidates = sorted({int(m.group(1)) for n in names
                         if (m := re.fullmatch(r'full/checkpoints/checkpoint-(\d+)/trainer_state.json', n))}, reverse=True)
    skipped = []
    for step in candidates:
        prefix = f'full/checkpoints/checkpoint-{step}/'
        try:
            for name in required_files(run):
                if prefix + name not in names or not read(prefix + name, size_only=True):
                    raise ValueError('Missing/empty ' + name)
            state = json.loads(read(prefix + 'trainer_state.json'))
            if state['global_step'] == step and step >= state['max_steps']:
                raise RuntimeError('Latest complete checkpoint has finished its step budget; do not restart an older save')
            if state['global_step'] != step or not 0 < step < state['max_steps']:
                raise ValueError('Invalid step or no remaining steps')
            # Windows or POSIX provenance paths are labels, never extraction paths.
            best_name = state['best_model_checkpoint'].replace('\\', '/').rstrip('/').split('/')[-1]
            if not re.fullmatch(r'checkpoint-\d+', best_name):
                raise ValueError('Invalid best checkpoint name')
            best_prefix = next((p for p in [f'full/checkpoints/{best_name}/',
                f'resume-source/full/checkpoints/{best_name}/']
                if all(p + n in names and read(p + n, size_only=True)
                       for n in ('adapter_model.safetensors', 'adapter_config.json'))), None)
            if best_prefix is None:
                raise ValueError('Missing preserved best adapter')
            return prefix, best_prefix, state, skipped
        except (ValueError, KeyError, TypeError) as error:
            skipped.append({'checkpoint': f'checkpoint-{step}', 'reason': str(error)})
    raise ValueError('No complete resumable checkpoint: ' + json.dumps(skipped))


def stage_source(bundle, output):
    """Restore from a Drive run directory or results ZIP without changing the source."""
    bundle, output = Path(bundle), Path(output)
    if output.exists(): raise FileExistsError('Never overwrite a checkpoint restoration')
    archive_hash = None if bundle.is_dir() else sha256(bundle)
    archive = None
    if bundle.is_dir():
        names = [p.relative_to(bundle).as_posix() for p in bundle.rglob('*') if p.is_file()]
        def read(name, size_only=False):
            path = bundle / name
            if not path.resolve().is_relative_to(bundle.resolve()):
                raise ValueError('Source symlink escapes run folder')
            return path.stat().st_size if size_only else path.read_bytes()
    else:
        archive = zipfile.ZipFile(bundle)
        checked_members(archive)
        bad = archive.testzip()
        if bad: raise ValueError('ZIP CRC failed: ' + bad)
        names = archive.namelist()
        def read(name, size_only=False):
            return archive.getinfo(name).file_size if size_only else archive.read(name)
    try:
        prefix, best_prefix, source_state, skipped = select_checkpoint(names, read)
        output.mkdir(parents=True, exist_ok=False)
        checkpoint = output / prefix.rstrip('/')
        checkpoint.mkdir(parents=True)
        selected = [n for n in names if n.startswith(prefix) and not n.endswith('/') and not n.endswith('training_args.bin')]
        best_files = [best_prefix + n for n in ('adapter_model.safetensors', 'adapter_config.json')]
        selected += best_files + ['full/training-run.json']
        if 'source-git.json' in names: selected.append('source-git.json')
        hashes = {}
        for name in dict.fromkeys(selected):
            destination_name = ('full/checkpoints/' + best_prefix.rstrip('/').split('/')[-1] + '/' + name[len(best_prefix):]
                                if name in best_files else name)
            target = output / destination_name
            target.parent.mkdir(parents=True, exist_ok=True)
            if archive is None:
                read(name, size_only=True)  # Validate source containment before copying.
                source_handle = (bundle / name).open('rb')
            else:
                source_handle = archive.open(name)
            with source_handle as source, target.open('xb') as destination:
                shutil.copyfileobj(source, destination)
            hashes[name] = sha256(target)
    finally:
        if archive is not None: archive.close()
    # Restrictive PyTorch deserialization; never use weights_only=False or import archive code.
    import torch
    from transformers.trainer import safe_globals
    optimizer = torch.load(checkpoint / 'optimizer.pt', map_location='cpu', weights_only=True)
    scheduler = torch.load(checkpoint / 'scheduler.pt', map_location='cpu', weights_only=True)
    run = json.loads((output / 'full/training-run.json').read_text(encoding='utf-8'))
    scaler = torch.load(checkpoint / 'scaler.pt', map_location='cpu', weights_only=True) if run['precision'] == 'fp16' else None
    with safe_globals():
        rng = torch.load(checkpoint / 'rng_state.pth', map_location='cpu', weights_only=True)
    if not optimizer.get('state') or not optimizer.get('param_groups') or not isinstance(scheduler.get('last_epoch'), int):
        raise ValueError('Invalid optimizer/scheduler state')
    if (scaler is not None and not scaler.get('scale')) or not all(k in rng for k in ('python', 'numpy', 'cpu', 'cuda')):
        raise ValueError('Missing precision-specific scaler/RNG state')
    # Preserve original state JSON separately; edit only a new working copy's obsolete Colab path.
    shutil.copy2(checkpoint / 'trainer_state.json', checkpoint / 'trainer_state.original.json')
    best_name = best_prefix.rstrip('/').split('/')[-1]
    source_state['best_model_checkpoint'] = str((output / 'full/checkpoints' / best_name).resolve())
    write_json(checkpoint / 'trainer_state.json', source_state)
    report = {'archive': str(bundle.resolve()), 'archiveSha256': archive_hash, 'sourceFilesSha256': hashes,
              'checkpoint': str(checkpoint.resolve()), 'startStep': source_state['global_step'],
              'targetSteps': source_state['max_steps'], 'schedulerLastEpoch': scheduler['last_epoch'],
              'precision': run['precision'], 'fp16Scale': scaler['scale'] if scaler else None, 'weightsOnlyDeserialization': True,
              'trainingArgumentsPickleLoaded': False, 'relocatedBestCheckpoint': source_state['best_model_checkpoint']}
    report['skippedIncompleteCheckpoints'] = skipped
    if archive_hash is not None and sha256(bundle) != archive_hash: raise ValueError('Source archive changed')
    write_json(output / 'resume-restoration.json', report)
    return report


def stage_archive(bundle, output):
    return stage_source(bundle, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', '--zip', dest='source', type=Path, required=True,
                        help='Drive run directory or results ZIP; selects latest complete checkpoint')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(stage_source(args.source, args.output), indent=2))


if __name__ == '__main__': main()
