"""Explicitly fetch a pinned HF training checkpoint, never run automatically."""
import argparse
import json
from pathlib import Path

from common import CONFIG, ROOT, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', required=True)
    parser.add_argument('--accept-license', action='store_true', required=True)
    parser.add_argument('--tokenizer-only', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / '.local/training/base')
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    config = json.loads(CONFIG.read_text(encoding='utf-8-sig'))
    allowed = ['config.json', 'generation_config.json', 'tokenizer*', 'vocab.json', 'merges.txt', 'LICENSE', 'README.md']
    if not args.tokenizer_only:
        allowed.append('model.safetensors')
    snapshot_download(config['model'], revision=config['revision'], local_dir=args.output, allow_patterns=allowed)
    if not args.tokenizer_only and sha256(args.output / 'model.safetensors') != config['weightsSha256']:
        raise RuntimeError('Pinned base weights hash mismatch')
    files = {path.name: sha256(path) for path in args.output.iterdir() if path.is_file() and path.name != 'base-provenance.json'}
    write_json(args.output / 'base-provenance.json', {'model': config['model'], 'revision': config['revision'],
                                                   'license': config['license'], 'tokenizerOnly': args.tokenizer_only,
                                                   'files': files})
    print('Verified pinned files in', args.output)


if __name__ == '__main__':
    main()
