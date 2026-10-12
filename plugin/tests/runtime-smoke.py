#!/usr/bin/env python3
"""Linux integration tests against plugin-owned inference fixture, never equipment."""
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

root = Path.home() / '.local/share/pins-ai'
source = Path(__file__).resolve().parent
config_path = root / 'config.json'
original = config_path.read_bytes()
config = json.loads(original)
key = (root / 'api.key').read_text().strip()
url = f"http://127.0.0.1:{config['apiPort']}/"
fixture = source / 'fake-llama.py'
os.chmod(fixture, 0o700)
model = root / 'fixture-model.bin'
model.write_bytes(b'fixture model')
checks = []


def check(value, name):
    assert value, name
    checks.append(name)
    print('PASS', name, flush=True)


def request(path, payload=None, post=False):
    req = urllib.request.Request(url + path,
        data=json.dumps(payload).encode() if payload is not None else b'' if post else None,
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=30)


def api(path, post=False):
    with request(path, post=post) as response:
        return json.load(response)


def apply(**updates):
    current = {**config, 'enabled': True, 'modelPath': str(model), 'runtimePath': str(fixture),
               'launcherPath': str(root / 'pins-ai-exec'), 'modelSha256': hashlib.sha256(model.read_bytes()).hexdigest(),
               'inferenceTimeoutSeconds': 5, **updates}
    config_path.write_text(json.dumps(current)); os.chmod(config_path, 0o600)
    api('reload', True)


def chat(prompt):
    output = ''; error = None
    with request('chat', {'messages': [{'role': 'user', 'content': prompt}], 'locale': 'de'}) as response:
        event = ''
        for raw in response:
            line = raw.decode().strip()
            if line.startswith('event: '): event = line[7:]
            if line.startswith('data: '):
                data = json.loads(line[6:])
                if event == 'token': output += data['text']
                if event == 'error': error = data
    return output, error


try:
    apply(runtimePath='/nonexistent/llama-server')
    try:
        api('load', True)
        raise AssertionError('Missing runtime accepted')
    except urllib.error.HTTPError as error:
        check(error.code == 503 and api('status')['processId'] is None, 'missing runtime fails without child')
    apply(modelSha256='0' * 64)
    try:
        api('load', True)
        raise AssertionError('Invalid model hash accepted')
    except urllib.error.HTTPError as error:
        check(error.code == 503 and api('status')['processId'] is None, 'model hash failure before process launch')
    apply()
    loaded = api('load', True)
    pid = loaded['processId']
    check(loaded['state'] == 'ready' and Path(f'/proc/{pid}').exists(), 'fixture loads through actual plugin lifecycle')
    output, error = chat('normal')
    check(output == 'First second third' and error is None, 'ordered streaming and complete response')
    check(api('status')['processId'] == pid, 'weights process reused between turns')
    result = []
    worker = threading.Thread(target=lambda: result.append(chat('stall')))
    worker.start()
    for _ in range(50):
        if api('status')['state'] == 'busy': break
        time.sleep(.05)
    _, error = chat('normal')
    check(error is not None and error['status'] == 409, 'overlapping inference rejected')
    started = time.monotonic(); api('cancel', True); worker.join(10)
    check(not worker.is_alive() and time.monotonic() - started < 3 and not Path(f'/proc/{pid}').exists(), 'cancellation terminates owned process promptly')
    apply()
    started = time.monotonic(); _, error = chat('stall')
    check(error is not None and time.monotonic() - started < 8 and api('status')['processId'] is None, 'inference timeout releases child')
    apply(); _, error = chat('oversize')
    check(error is not None and api('status')['processId'] is None, 'oversized runtime frame rejected and child released')
    apply(); _, error = chat('invalid-json')
    check(error is not None and api('status')['processId'] is None, 'malformed runtime JSON rejected and child released')
    apply(maxResidentMemoryMiB=256); _, error = chat('allocate')
    check(error is not None and 'safeguard' in error['error'] and api('status')['state'] == 'failed' and api('status')['processId'] is None, 'actual RSS safeguard stops owned inference process with specific reason')
    apply()
    response = request('chat', {'messages': [{'role': 'user', 'content': 'stall'}]})
    for raw in response:
        if raw.decode().strip() == 'event: token':
            break
    response.close()
    for _ in range(70):
        if api('status')['processId'] is None:
            break
        time.sleep(.1)
    check(api('status')['processId'] is None, 'disconnected stream bounded by turn deadline and child released')
    apply(); api('load', True); pid = api('status')['processId']
    api('unload', True)
    check(not Path(f'/proc/{pid}').exists() and api('status')['state'] == 'unloaded', 'explicit unload releases process')
    apply(enabled=False)
    check(not api('status')['enabled'] and api('status')['processId'] is None, 'disable releases runtime without stopping PINS')
finally:
    config_path.write_bytes(original); os.chmod(config_path, 0o600)
    api('reload', True)
    model.unlink()
print(json.dumps({'passed': len(checks), 'checks': checks}, indent=2))
