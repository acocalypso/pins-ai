#!/usr/bin/env python3
"""Explicit development/install utility, not a runtime dependency of the plugin."""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import urllib.request


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accept-licenses', action='store_true', required=True)
    parser.add_argument('--root', type=Path, default=Path.home() / '.local/share/pins-ai')
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1] / 'models/manifest.json')
    parser.add_argument('--models', nargs='+', default=['qwen3-0.6b-q4'])
    parser.add_argument('--build-runtime', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    unknown = set(args.models) - {m['id'] for m in manifest['models']}
    if unknown:
        parser.error(f'Unknown model identifiers: {sorted(unknown)}')
    args.root.mkdir(parents=True, exist_ok=True)
    models = args.root / 'models'
    models.mkdir(exist_ok=True)
    for model in manifest['models']:
        if model['id'] not in args.models:
            continue
        target = models / model['file']
        print(f"{model['id']}: {model['bytes']} bytes; {model['license']}; {model['upstream']}", flush=True)
        if target.exists() and digest(target) == model['sha256']:
            print('Already verified', flush=True)
            continue
        url = f"https://huggingface.co/{model['repository']}/resolve/{model['revision']}/{model['file']}"
        temporary = target.with_suffix('.partial')
        with urllib.request.urlopen(url, timeout=60) as source, temporary.open('wb') as destination:
            while chunk := source.read(1024 * 1024):
                destination.write(chunk)
        if temporary.stat().st_size != model['bytes'] or digest(temporary) != model['sha256']:
            temporary.unlink()
            raise RuntimeError('Model size or SHA256 mismatch')
        temporary.replace(target)
        notice = models / (model['file'] + '.provenance.json')
        notice.write_text(json.dumps(model, indent=2) + '\n')
        # Retain upstream licence text alongside the model, outside Git.
        license_repo = model['upstream'].removeprefix('https://huggingface.co/')
        license_url = model.get('licenseUrl', f'https://huggingface.co/{license_repo}/resolve/main/LICENSE')
        with urllib.request.urlopen(license_url, timeout=30) as response:
            (models / (model['file'] + '.LICENSE')).write_bytes(response.read())
    if args.build_runtime:
        runtime = manifest['runtime']
        source = args.root / 'llama.cpp'
        if not source.exists():
            subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout', runtime['repository'], str(source)], check=True)
        subprocess.run(['git', '-C', str(source), 'fetch', '--depth=1', 'origin', runtime['commit']], check=True)
        subprocess.run(['git', '-C', str(source), 'checkout', '--detach', runtime['commit']], check=True)
        actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != runtime['commit']:
            raise RuntimeError('Runtime commit mismatch')
        subprocess.run(['cmake', '-S', str(source), '-B', str(source / 'build'),
                        '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=OFF',
                        '-DGGML_NATIVE=ON', '-DGGML_OPENMP=ON',
                        '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_EXAMPLES=OFF'], check=True)
        subprocess.run(['cmake', '--build', str(source / 'build'), '--target', 'llama-server', 'llama-bench', '-j', '2'], check=True)
        launcher_source = Path(__file__).resolve().parents[1] / 'native/pins-ai-exec.c'
        subprocess.run(['gcc', '-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', str(launcher_source), '-o', str(args.root / 'pins-ai-exec')], check=True)
        provenance = {**runtime, 'buildHost': platform.platform(), 'cpu': platform.machine(),
                      'nativeCpuBuild': True, 'buildThreads': 2,
                      'serverSha256': digest(source / 'build/bin/llama-server'),
                      'launcherSha256': digest(args.root / 'pins-ai-exec')}
        (args.root / 'runtime-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


if __name__ == '__main__':
    main()
