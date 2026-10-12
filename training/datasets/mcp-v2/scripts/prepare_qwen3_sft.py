#!/usr/bin/env python3
"""Package entry point for the audited, pinned-tokenizer v2 SFT exporter."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parents[1]
REPOSITORY = PACKAGE.parents[2]


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=REPOSITORY / '.local/training/base',
                        help='Verified local pinned Qwen checkpoint; no automatic model downloads')
    parser.add_argument('--max-tokens', type=int, default=0, help='0=measure all; otherwise fail on any overlong turn')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.dry_run:
        row = json.loads((PACKAGE / 'data/train.jsonl').read_text(encoding='utf-8').splitlines()[0])
        print(json.dumps(row, ensure_ascii=False, indent=2))
        return
    if not (PACKAGE / 'data/contract-audit.json').is_file():
        parser.error('Run training/audit_v2.py against actual source-generated tools/list first')
    command = [sys.executable, str(REPOSITORY / 'training/prepare_v2.py'), '--base', str(args.model),
               '--max-tokens', str(args.max_tokens)]
    raise SystemExit(subprocess.call(command, cwd=REPOSITORY))


if __name__ == '__main__':
    main()
