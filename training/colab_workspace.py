"""Load Drive dataset bytes only after checking them against the Git checkout."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


def use_drive_data(project, drive_folder):
    project, drive_folder = Path(project), Path(drive_folder)
    prefix = 'training/datasets/mcp-v2/data/'
    tracked = subprocess.check_output(
        ['git', 'ls-files', '--', prefix], cwd=project, text=True).splitlines()
    if not tracked or prefix + 'contract-audit.json' not in tracked:
        raise RuntimeError('Git checkout lacks the audited dataset contracts.')
    # Drive provides immutable original inputs. Audited schemas and derived data
    # come from the current Git code, so MCP updates require no stale ZIP upload.
    manifest = json.loads((project / prefix / 'manifest.json').read_text(encoding='utf-8'))
    tracked = [prefix + 'manifest.json', *[prefix + n for n in manifest['sha256']]]
    legacy_prefix = 'PINS_MCP_AI_Training_v2/data/'
    folders = [drive_folder / prefix, drive_folder / legacy_prefix, drive_folder / 'data']
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
                archive_name = name if name in archive.namelist() else legacy_prefix + name[len(prefix):]
                payloads[name] = archive.read(archive_name)
        origin = {'kind': 'zip', 'path': str(bundle),
                  'sha256': hashlib.sha256(bundle.read_bytes()).hexdigest()}
    else:
        raise FileNotFoundError(f'No dataset directory or existing training ZIP in {drive_folder}')
    # Check everything before writing anything. Never import code from Drive.
    for name, content in payloads.items():
        expected = hashlib.sha256((project / name).read_bytes()).hexdigest()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f'Drive data differs from the Git revision: {name}. '
                             'Update the Drive data to match; do not train mixed revisions.')
    for name, content in payloads.items():
        (project / name).write_bytes(content)
    return {**origin, 'verifiedFiles': len(payloads), 'contractsSource': 'current Git checkout'}
