#!/usr/bin/env python3
"""Verify the native guard using a disposable parent, never killing PINS."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--launcher', type=Path, default=Path.home() / '.local/share/pins-ai/pins-ai-exec')
args = parser.parse_args()
code = ('import subprocess,os,time; '
        f'p=subprocess.Popen([{str(args.launcher)!r},str(os.getpid()),"/bin/sleep","60"]); '
        'print(p.pid,flush=True); time.sleep(60)')
parent = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, text=True)
pid = int(parent.stdout.readline())
time.sleep(.2)
parent.kill()
parent.wait()
state = None
for _ in range(50):
    try:
        state = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0]
    except FileNotFoundError:
        state = 'gone'
    if state in ('gone', 'Z'):
        break
    time.sleep(.02)
assert state in ('gone', 'Z'), 'Guard child survived parent death'
print(json.dumps({'test': 'native parent-death guard', 'childState': state, 'passed': True}))
