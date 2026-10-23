"""Persistent Colab run identity, append-only attempts, and bounded result bundles.

This module only handles files. Checkpoint/data/dependency validation remains in
training_resume.py and train.py; no GPU libraries are imported here.
"""
import json
from pathlib import Path
import re
import shutil
from datetime import datetime, timezone
import uuid
import zipfile

from restore_results import checked_members

POINTER = 'latest-training-attempt.json'


def _inside(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Attempt path escapes original run folder')
    return path


def _name(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', value) or value in ('.', '..'):
        raise ValueError('Unsafe run name')
    return value


def latest_source(run_root):
    """Ignore setup-only attempts; never fall back from completed latest training."""
    root = Path(run_root).resolve()
    candidates = []
    if (root / 'full/training-run.json').is_file():
        _inside(root, root / 'full/training-run.json')
        candidates.append(root)
    attempts = root / 'attempts'
    if attempts.exists():
        _inside(root, attempts)
        for attempt in sorted(attempts.iterdir()):
            _inside(root, attempt)
            if attempt.name.startswith('resume-') and attempt.is_dir() and (attempt / 'full/training-run.json').is_file():
                _inside(root, attempt / 'full/training-run.json')
                candidates.append(attempt)
    pointer = root / POINTER
    if pointer.exists():
        relative = json.loads(pointer.read_text(encoding='utf-8'))['attempt']
        if not isinstance(relative, str) or Path(relative).is_absolute() or '\\' in relative:
            raise ValueError('Unsafe latest attempt pointer')
        selected = _inside(root, root / relative)
        if selected not in candidates:
            raise ValueError('Latest attempt pointer has no full training metadata')
    if not candidates:
        raise ValueError('No full training metadata in original run folder')
    selected = candidates[-1]
    run = json.loads((selected / 'full/training-run.json').read_text(encoding='utf-8'))
    if run.get('status') == 'completed' or run.get('smokeOnly'):
        raise ValueError('Latest full run is completed or smoke-only; do not restart an older save')
    return selected


def publish_latest(run_root, attempt_root):
    """Publish only after train.py has created full metadata, including on failure."""
    root = Path(run_root).resolve()
    attempt = _inside(root, attempt_root)
    if not (attempt / 'full/training-run.json').is_file():
        return False
    _inside(root, attempt / 'full/training-run.json')
    relative = attempt.relative_to(root).as_posix() or '.'
    pointer = root / POINTER
    payload = {'attempt': relative}
    if pointer.is_file() and json.loads(pointer.read_text(encoding='utf-8')) == payload:
        return True
    # Replacing this small canonical pointer never changes old attempt contents.
    temporary = root / (POINTER + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(payload, indent=2) + '\n')
    temporary.replace(pointer)
    return True


def _zip_run_name(source):
    with zipfile.ZipFile(source) as archive:
        checked_members(archive)
        names = archive.namelist()
        identity = None
        for filename in ('colab-run-layout.json', 'source-git.json'):
            if filename in names:
                identity = json.loads(archive.read(filename)).get('runName')
                if identity:
                    identity = _name(identity)
                    break
    suffix = '-results.zip'
    inferred = _name(source.name[:-len(suffix)]) if source.name.endswith(suffix) else None
    if identity and inferred and identity != inferred:
        raise ValueError('ZIP basename and recorded original run name disagree')
    return identity or inferred


def create_layout(mode, drive_folder, run_name, resume_source, resume_run_root=None):
    runs = Path(drive_folder).resolve() / 'runs'
    source = None
    if mode == 'fresh':
        root = runs / _name(run_name)
        root.mkdir(parents=True, exist_ok=False)
        attempt = root
    elif mode == 'resume':
        source = Path(resume_source).resolve()
        if source.is_dir():
            root = source
            identity_file = root / 'colab-run-layout.json'
            if identity_file.is_file() and json.loads(identity_file.read_text(encoding='utf-8'))['runName'] != root.name:
                raise ValueError('RESUME_SOURCE must be the original run root, not an attempt folder')
            if resume_run_root is not None and Path(resume_run_root).resolve() != root:
                raise ValueError('Directory RESUME_SOURCE must be the original run root')
            source = latest_source(root)
        elif source.is_file():
            identity = _zip_run_name(source)
            if resume_run_root is None:
                if not identity:
                    raise ValueError('ZIP needs RESUME_RUN_ROOT or an unambiguous <run>-results.zip name')
                root = runs / identity
            else:
                root = Path(resume_run_root).resolve()
                if identity and root.name != identity:
                    raise ValueError('ZIP belongs to a different original run name')
            # Prefer existing newer directory state over a potentially old ZIP.
            if root.exists() and ((root / 'full/training-run.json').is_file() or (root / 'attempts').exists()):
                source = latest_source(root)
            root.mkdir(parents=True, exist_ok=True)
        else:
            raise FileNotFoundError('RESUME_SOURCE does not exist')
        attempt = root / 'attempts' / ('resume-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f'))
        _inside(root, attempt)
        attempt.mkdir(parents=True, exist_ok=False)
    else:
        raise ValueError('Unknown training mode')
    return {'runRoot': root, 'attemptRoot': attempt, 'source': source}


def preserve_resume_evidence(attempt_root, staged_root):
    """Keep tiny provenance + best adapter, never duplicate optimizer/RNG on Drive."""
    attempt, staged = Path(attempt_root), Path(staged_root)
    report = json.loads((staged / 'resume-restoration.json').read_text(encoding='utf-8'))
    best = _inside(staged, report['relocatedBestCheckpoint'])
    checkpoint = _inside(staged, report['checkpoint'])
    destination = attempt / 'resume-source'
    destination.mkdir(exist_ok=False)
    for source in (staged / 'resume-restoration.json', staged / 'full/training-run.json',
                   checkpoint / 'trainer_state.json', checkpoint / 'trainer_state.original.json',
                   best / 'adapter_model.safetensors', best / 'adapter_config.json'):
        target = destination / source.relative_to(staged)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def archive_attempt(run_root, attempt_root):
    """Export one self-contained attempt; avoid older attempts and archive recursion."""
    root = Path(run_root).resolve()
    attempt = _inside(root, attempt_root)
    output = attempt / (root.name + '-results.zip')
    # Snapshot names before creating the ZIP and reject symlinks escaping the run.
    paths = []
    for path in sorted(attempt.rglob('*')):
        _inside(attempt, path)
        relative = path.relative_to(attempt)
        if path.is_file() and relative.parts[0] != 'attempts' and path.suffix.lower() != '.zip':
            paths.append((path, relative.as_posix()))
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path, relative in paths:
            archive.write(path, relative)
    return output
