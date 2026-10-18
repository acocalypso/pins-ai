"""Reuse pinned base-model data on Colab; never download or import Drive code.

Weights are checked against the Git configuration, not a Drive-supplied digest.
Other data files retain and are checked against fetch_base.py's provenance.
The cache is optional: a miss leaves the explicit licensed fetch to the caller.
"""
import json
from pathlib import Path
import re
import shutil
import tempfile
import warnings

from common import CONFIG, fingerprint, sha256


_FILES = frozenset({
    'model.safetensors', 'config.json', 'generation_config.json',
    'tokenizer.json', 'tokenizer_config.json', 'tokenizer.model',
    'vocab.json', 'merges.txt', 'LICENSE', 'README.md',
    'added_tokens.json', 'special_tokens_map.json',
})
_REQUIRED = frozenset({'model.safetensors', 'config.json', 'tokenizer.json', 'tokenizer_config.json'})
_PROVENANCE = 'base-provenance.json'


def _config(config_path):
    config = json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
    return {key: config[key] for key in ('model', 'revision', 'weightsSha256', 'license')}


def _cache_path(cache_root, config):
    return Path(cache_root) / fingerprint(config)


def _verify(folder, config):
    """Validate complete provenance and bytes, rejecting executable/path entries."""
    folder = Path(folder)
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError('Base directory is missing or is a symlink')
    provenance_path = folder / _PROVENANCE
    if provenance_path.is_symlink():
        raise ValueError('Base provenance cannot be a symlink')
    provenance = json.loads(provenance_path.read_text(encoding='utf-8-sig'))
    if not isinstance(provenance, dict):
        raise ValueError('Base provenance must be a JSON object')
    if (any(provenance.get(key) != config[key] for key in ('model', 'revision', 'license'))
            or provenance.get('tokenizerOnly') is not False):
        raise ValueError('Base provenance does not match the pinned full checkpoint')
    files = provenance.get('files')
    if not isinstance(files, dict) or not _REQUIRED <= files.keys():
        raise ValueError('Base provenance lacks required model/tokenizer files')
    if not files.keys() <= _FILES:
        raise ValueError('Base provenance contains unsupported data or executable paths')
    if files['model.safetensors'] != config['weightsSha256']:
        raise ValueError('Base provenance does not match the pinned weights hash')
    for name, expected in files.items():
        path = folder / name
        if (not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected)
                or path.is_symlink() or not path.is_file() or sha256(path) != expected):
            raise ValueError('Base file failed verification: ' + name)
    # HF's download metadata directory is never copied or interpreted.
    for path in folder.iterdir():
        if path.name not in {*files, _PROVENANCE, '.cache'}:
            raise ValueError('Unexpected base file: ' + path.name)
    return provenance


def _publish(source, destination, config, provenance):
    """Verify a private copy before replacing a local/cache directory."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError('Source and destination must differ')
    if destination.is_symlink():
        raise ValueError('Base destination cannot be a symlink')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.base-stage-', dir=destination.parent) as temporary:
        staged = Path(temporary) / 'verified'
        staged.mkdir()
        for name in (*provenance['files'], _PROVENANCE):
            shutil.copyfile(source / name, staged / name)
        _verify(staged, config)
        previous = Path(temporary) / 'previous'
        if destination.exists():
            destination.rename(previous)
        try:
            staged.rename(destination)
        except BaseException:
            if previous.exists():
                previous.rename(destination)
            raise


def reuse_base_model(local_dir, cache_root, config_path=CONFIG):
    """Return a verified local/Drive hit, or None; perform no network requests."""
    config = _config(config_path)
    local_dir, cache = Path(local_dir), _cache_path(cache_root, config)
    for source, folder in (('local', local_dir), ('drive', cache)):
        try:
            provenance = _verify(folder, config)
        except (OSError, ValueError, TypeError, KeyError) as error:
            if folder.exists():
                warnings.warn(f'Ignoring invalid {source} base model at {folder}: {error}', stacklevel=2)
            continue
        if source == 'drive':
            _publish(folder, local_dir, config, provenance)
        return {'source': source, 'path': str(local_dir), 'cachePath': str(cache),
                'model': config['model'], 'revision': config['revision'],
                'weightsSha256': config['weightsSha256']}
    return None


def persist_base_model(local_dir, cache_root, config_path=CONFIG):
    """Publish verified fetched/local data to Drive without HF download metadata."""
    config = _config(config_path)
    provenance = _verify(local_dir, config)
    cache = _cache_path(cache_root, config)
    try:
        _verify(cache, config)
    except (OSError, ValueError, TypeError, KeyError):
        _publish(local_dir, cache, config, provenance)
        source = 'stored'
    else:
        source = 'existing'
    return {'source': source, 'cachePath': str(cache), 'revision': config['revision'],
            'weightsSha256': config['weightsSha256']}
