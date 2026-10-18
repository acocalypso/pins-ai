"""Verify Git dataset bytes and optional legacy Drive copies without loading code."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def use_drive_data(project, drive_folder):
    project, drive_folder = Path(project), Path(drive_folder)
    prefix = 'training/dataset/data/'
    tracked = subprocess.check_output(
        ['git', 'ls-files', '--', prefix], cwd=project, text=True).splitlines()
    if not tracked or prefix + 'contract-audit.json' not in tracked:
        raise RuntimeError('Git checkout lacks the audited dataset contracts.')
    # Git contains the complete dataset. Drive copies are optional compatibility
    # inputs; their absence must not require uploading a redundant dataset ZIP.
    manifest = json.loads((project / prefix / 'manifest.json').read_text(encoding='utf-8'))
    manifest_files = manifest['sha256']
    for relative, expected in manifest_files.items():
        name = prefix + relative
        if name not in tracked or '..' in Path(relative).parts or Path(relative).is_absolute():
            raise ValueError('Dataset manifest references an untracked or unsafe path: ' + relative)
        if hashlib.sha256((project / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Git dataset failed manifest verification: ' + name)
    tracked = [prefix + 'manifest.json', *[prefix + n for n in manifest_files]]
    legacy_prefixes = ('training/datasets/mcp-v2/data/', 'PINS_MCP_AI_Training_v2/data/')
    folders = [drive_folder / p for p in (prefix, *legacy_prefixes, 'data/')]
    folder = next((p for p in folders if (p / 'manifest.json').is_file()), None)
    bundle = drive_folder / 'pins-ai-training-0.1.0.zip'
    payloads = {}
    if folder:
        for name in tracked:
            relative = name[len(prefix):]
            payloads[name] = (folder / relative).read_bytes()
        origin = {'kind': 'directory', 'path': str(folder)}
    elif bundle.is_file():
        with zipfile.ZipFile(bundle) as archive:
            for name in tracked:
                relative = name[len(prefix):]
                archive_name = next((p + relative for p in (prefix, *legacy_prefixes)
                                     if p + relative in archive.namelist()), None)
                if archive_name is None:
                    raise FileNotFoundError('Drive archive lacks immutable dataset file: ' + relative)
                payloads[name] = archive.read(archive_name)
        origin = {'kind': 'zip', 'path': str(bundle),
                  'sha256': hashlib.sha256(bundle.read_bytes()).hexdigest()}
    else:
        return {'kind': 'git', 'path': str(project / prefix),
                'verifiedFiles': len(tracked),
                'manifestSha256': hashlib.sha256((project / prefix / 'manifest.json').read_bytes()).hexdigest(),
                'contractsSource': 'current Git checkout'}
    # Check everything before writing anything. Never import code from Drive.
    for name, content in payloads.items():
        expected = hashlib.sha256((project / name).read_bytes()).hexdigest()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f'Drive data differs from the Git revision: {name}. '
                             'Update the Drive data to match; do not train mixed revisions.')
    for name, content in payloads.items():
        (project / name).write_bytes(content)
    return {**origin, 'verifiedFiles': len(payloads), 'contractsSource': 'current Git checkout'}
