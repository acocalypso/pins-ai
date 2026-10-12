"""Verify checked-out Git LFS model assets and copy them into a package."""
import argparse
import hashlib
import json
import pathlib
import shutil


def verified(path, asset):
    return path.is_file() and path.stat().st_size == asset['bytes'] and hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() == asset['sha256']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=pathlib.Path)
    parser.add_argument('--cache', type=pathlib.Path)
    args = parser.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[1]
    manifest = json.loads((repo / 'models/bundled.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for asset in manifest['assets']:
        destination = args.output / asset['file']
        if verified(destination, asset):
            continue
        temporary = destination.with_suffix('.partial')
        cached = (args.cache or repo / 'models/bundled') / asset['file']
        try:
            if verified(cached, asset):
                shutil.copyfile(cached, temporary)
            else:
                raise RuntimeError('Missing or invalid Git LFS model: ' + str(cached) + '; run git lfs pull')
            if not verified(temporary, asset):
                raise RuntimeError('Model verification failed: ' + asset['file'])
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    (args.output / 'provenance.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
