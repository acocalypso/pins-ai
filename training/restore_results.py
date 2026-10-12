"""Verify and restore a Colab ZIP to a new directory; never execute archived code/pickles."""
import argparse
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import stat
import zipfile

from common import ROOT, sha256, write_json


def checked_members(archive):
    members = archive.infolist()
    seen = set()
    if sum(m.file_size for m in members) > 8 * 1024**3:
        raise ValueError('Archive expanded size exceeds the restoration budget')
    for member in members:
        name = member.filename
        if member.orig_filename != name or '\x00' in member.orig_filename:
            raise ValueError('Unsafe normalized archive path: ' + member.orig_filename)
        path = PurePosixPath(name)
        if (not name or '\\' in name or path.is_absolute() or PureWindowsPath(name).drive
                or any(p in ('..', '') or ':' in p for p in path.parts)
                or any(p.endswith((' ', '.')) for p in path.parts)
                or any(p.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL',
                    *('COM' + str(i) for i in range(1, 10)), *('LPT' + str(i) for i in range(1, 10))} for p in path.parts)):
            raise ValueError('Unsafe archive path: ' + name)
        key = str(path).casefold()
        if key in seen:
            raise ValueError('Duplicate archive path: ' + name)
        seen.add(key)
        mode = member.external_attr >> 16
        if stat.S_ISLNK(mode) or stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ValueError('Non-regular archive entry: ' + name)
        if member.flag_bits & 1:
            raise ValueError('Encrypted archive entry: ' + name)
    # Detect file/directory collisions before creating any destination files.
    file_names = {str(PurePosixPath(m.filename)).casefold() for m in members if not m.is_dir()}
    for member in members:
        if any(str(p).casefold() in file_names for p in PurePosixPath(member.filename).parents if str(p) != '.'):
            raise ValueError('Archive file/directory collision: ' + member.filename)
    return members


def restore(bundle, output, base=None):
    bundle, output = Path(bundle), Path(output)
    if output.exists():
        raise FileExistsError('Never overwrite restored evidence: ' + str(output))
    archive_hash = sha256(bundle)
    with zipfile.ZipFile(bundle) as archive:
        members = checked_members(archive)
        bad = archive.testzip()
        if bad:
            raise ValueError('Archive CRC mismatch: ' + bad)
        output.mkdir(parents=True, exist_ok=False)
        extracted, omitted = {}, []
        allowed = {'.json', '.jsonl', '.log', '.txt', '.safetensors', '.jinja', '.png', '.md'}
        for member in members:
            if member.is_dir():
                continue
            if PurePosixPath(member.filename).suffix not in allowed:
                omitted.append(member.filename)
                continue
            target = output / member.filename
            if not target.resolve().is_relative_to(output.resolve()):
                raise ValueError('Archive destination escaped output directory')
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open('xb') as destination:
                shutil.copyfileobj(source, destination)
            extracted[member.filename] = sha256(target)
    source = json.loads((output / 'source-git.json').read_text(encoding='utf-8')) if (output / 'source-git.json').exists() else {}
    run_path = output / 'full/training-run.json'
    run = json.loads(run_path.read_text(encoding='utf-8'))
    checkpoints = []
    for state_path in sorted((output / 'full/checkpoints').glob('*/trainer_state.json')):
        state = json.loads(state_path.read_text(encoding='utf-8'))
        weights = state_path.parent / 'adapter_model.safetensors'
        config = json.loads((state_path.parent / 'adapter_config.json').read_text(encoding='utf-8'))
        checkpoints.append({'path': str(state_path.parent), 'globalStep': state['global_step'],
                            'maxSteps': state['max_steps'], 'bestMetric': state.get('best_metric'),
                            'bestCheckpoint': state.get('best_model_checkpoint'),
                            'adapterSha256': sha256(weights), 'rank': config['r'],
                            'evalLossHistory': [{k: event[k] for k in ('step', 'eval_loss')} for event in state['log_history'] if 'eval_loss' in event]})
    final = output / 'full/adapter'
    final_present = (final / 'adapter_model.safetensors').is_file() and (final / 'adapter_config.json').is_file()
    latest_state = max(checkpoints, key=lambda c: c['globalStep']) if checkpoints else None
    best_name = PurePosixPath((run.get('bestModelCheckpoint') or (latest_state or {}).get('bestCheckpoint') or '').replace('\\', '/')).name
    best = next((c for c in checkpoints if Path(c['path']).name == best_name), None)
    report = {'archive': str(bundle.resolve()), 'archiveSha256': archive_hash, 'crcVerified': True,
              'runId': source.get('runName'), 'source': source, 'trainingStatus': run['status'],
              'trainingCompleted': run['status'] == 'completed' and final_present,
              'finalAdapterPresent': final_present, 'finalAdapterPath': str(final) if final_present else None,
              'finalAdapterSha256': sha256(final / 'adapter_model.safetensors') if final_present else None,
              'checkpoints': checkpoints, 'bestSavedCheckpoint': best,
              'finalMatchesBestSavedCheckpoint': sha256(final / 'adapter_model.safetensors') == best['adapterSha256'] if final_present and best else None,
              'extractedSha256': extracted, 'omittedFilesRetainedInOriginalZip': omitted,
              'evaluationReports': [n for n in extracted if n.endswith('.json') and ('eval' in n or 'gold' in n)],
              'pickleFilesLoaded': False, 'hardwareExecution': False}
    if base:
        base = Path(base)
        expected = run['base']
        if any(Path(name).name != name or '/' in name or '\\' in name for name in expected['files']):
            raise ValueError('Unsafe base provenance filename')
        matched = {name: sha256(base / name) == digest for name, digest in expected['files'].items()}
        if not all(matched.values()):
            raise ValueError('Cached base files differ from archived training provenance: ' + str(matched))
        report['baseVerification'] = {'path': str(base), 'model': expected['model'], 'revision': expected['revision'], 'matchedFiles': matched}
    # Safetensors is data-only. Never load optimizer.pt, training_args.bin or RNG pickles.
    from safetensors import safe_open
    for checkpoint in checkpoints + ([{'path': str(final)}] if final_present else []):
        with safe_open(str(Path(checkpoint['path']) / 'adapter_model.safetensors'), framework='pt', device='cpu') as tensors:
            keys = list(tensors.keys())
            if not keys or any('lora_' not in name for name in keys):
                raise ValueError('Unexpected adapter tensors')
            checkpoint['tensorCount'] = len(keys)
            checkpoint['tensorShapes'] = {name: tensors.get_slice(name).get_shape() for name in keys}
    if sha256(bundle) != archive_hash:
        raise ValueError('Original archive changed during restoration')
    write_json(output / 'restoration-verification.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base', type=Path, default=ROOT / '.local/training/base')
    args = parser.parse_args()
    report = restore(args.zip, args.output, args.base)
    print(json.dumps({k: v for k, v in report.items() if k not in ('extractedSha256', 'checkpoints', 'bestSavedCheckpoint')}, indent=2))
    print('Best saved checkpoint:', report['bestSavedCheckpoint']['path'] if report['bestSavedCheckpoint'] else 'unavailable')


if __name__ == '__main__':
    main()
