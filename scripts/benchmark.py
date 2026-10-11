#!/usr/bin/env python3
"""Measure the plugin API on the Pi; no hardware commands or MCP tools exist here."""
import argparse
import hashlib
import datetime
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request


def proc(pid):
    try:
        status = dict(line.split(':', 1) for line in Path(f'/proc/{pid}/status').read_text(encoding='utf-8-sig').splitlines() if ':' in line)
        fields = Path(f'/proc/{pid}/stat').read_text(encoding='utf-8-sig').split(') ', 1)[1].split()
        return {'pid': pid, 'rss_kib': int(status.get('VmRSS', '0 kB').split()[0]),
                'swap_kib': int(status.get('VmSwap', '0 kB').split()[0]), 'ticks': int(fields[11]) + int(fields[12])}
    except (OSError, ValueError):
        return None


def system():
    info = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text(encoding='utf-8-sig').splitlines())
    try:
        throttled = subprocess.check_output(['vcgencmd', 'get_throttled'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        throttled = None
    return {'available_kib': int(info['MemAvailable'].split()[0]),
            'system_swap_used_kib': int(info['SwapTotal'].split()[0]) - int(info['SwapFree'].split()[0]),
            'temperature_c': int(Path('/sys/class/thermal/thermal_zone0/temp').read_text(encoding='utf-8-sig')) / 1000,
            'throttled': throttled, 'load': list(os.getloadavg())}


def percentile(values, p):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * p) - 1)] if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.home() / '.local/share/pins-ai')
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, nargs='+', default=[1])
    parser.add_argument('--models', nargs='+', default=['qwen3-0.6b-q4'])
    parser.add_argument('--repetitions', type=int, default=1)
    parser.add_argument('--case-pause-seconds', type=int, default=10)
    parser.add_argument('--max-output-tokens', type=int, default=128)
    parser.add_argument('--context-tokens', type=int, default=2048)
    parser.add_argument('--cases', nargs='+')
    parser.add_argument('--case-file', type=Path, help='Optional separate probe set; recorded in each sample')
    parser.add_argument('--manifest', type=Path, help='Local candidate manifest, without changing the checked-in baseline manifest')
    parser.add_argument('--evaluation-manifest', type=Path, help='Prepared dataset provenance for before/after evaluation')
    parser.add_argument('--plugin-runtime', type=Path,
                        default=Path.home() / '.local/share/NINA/Plugins/3.0.0/pins-ai/Pins.AI.Runtime.dll')
    args = parser.parse_args()
    config_path = args.root / 'config.json'
    original = config_path.read_bytes()
    config = json.loads(original)
    token = (args.root / 'api.key').read_text(encoding='utf-8-sig').strip()
    url = f"http://127.0.0.1:{config['apiPort']}/"
    manifest = json.loads((args.manifest or args.source / 'models/manifest.json').read_text(encoding='utf-8-sig'))
    if set(args.models) - {model['id'] for model in manifest['models']}:
        parser.error('Unknown model ID in selected manifest')
    cases = json.loads((args.case_file or args.source / 'benchmarks/cases.json').read_text(encoding='utf-8-sig'))
    if args.cases:
        cases = [case for case in cases if case['id'] in args.cases]
        if not cases:
            parser.error('No matching cases')

    def request(path, payload=None, post=False):
        req = urllib.request.Request(url + path, data=json.dumps(payload).encode() if payload is not None else b'' if post else None,
                                     headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
        return urllib.request.urlopen(req, timeout=180)

    def api(path, post=False):
        with request(path, post=post) as response:
            return json.load(response)

    def save_config(value):
        temporary = config_path.with_suffix('.benchmark-tmp')
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, 'w') as file:
            json.dump(value, file)
        temporary.replace(config_path)

    pins_pid = int(subprocess.check_output(['systemctl', 'show', 'pins', '--property=MainPID', '--value'], text=True))
    report = {'measuredAtUtc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'piModel': Path('/proc/device-tree/model').read_text(encoding='utf-8-sig').rstrip('\0'),
              'os': Path('/etc/os-release').read_text(encoding='utf-8-sig'), 'uname': subprocess.check_output(['uname', '-a'], text=True).strip(),
              'runtime': manifest['runtime'], 'host': proc(pins_pid), 'baseline': system(),
              'workload': 'PINS and PHD2 running; no active imaging or guiding session',
              'mcp': 'unavailable; no tool accuracy or hardware execution measured',
              'sessions': [], 'samples': []}
    report['casePauseSeconds'] = args.case_pause_seconds
    def file_hash(path):
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    report['pluginRuntimeSha256'] = file_hash(args.plugin_runtime) if args.plugin_runtime.is_file() else None
    report['nativeRuntimeSha256'] = file_hash(Path(config['runtimePath']))
    report['promptSha256'] = json.loads(args.evaluation_manifest.read_text(encoding='utf-8-sig'))['promptSha256'] if args.evaluation_manifest else None
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def persist():
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    try:
        api('unload', True)
        baseline_start = proc(pins_pid); baseline_time = time.monotonic()
        time.sleep(5)
        baseline_end = proc(pins_pid)
        report['baselinePinsCpuPercentOneCore'] = (baseline_end['ticks'] - baseline_start['ticks']) / os.sysconf('SC_CLK_TCK') / (time.monotonic() - baseline_time) * 100
        for model in manifest['models']:
            if model['id'] not in args.models:
                continue
            for threads in args.threads:
                current = {**config, 'enabled': True, 'lazyLoad': True, 'mode': model['profile'],
                           'modelId': model['id'], 'modelPath': str(args.root / 'models' / model['file']),
                           'modelSha256': model['sha256'], 'threads': threads,
                           'maxOutputTokens': args.max_output_tokens, 'contextTokens': args.context_tokens}
                save_config(current); api('reload', True)
                before = system(); started = time.monotonic()
                status = api('load', True)
                cold = time.monotonic() - started
                report['sessions'].append({'model': model, 'config': current, 'loadSeconds': cold,
                                           'status': status, 'before': before, 'afterLoad': system()})
                print(f"Loaded {model['id']} threads={threads}: {cold:.3f}s", flush=True)
                for repetition in range(args.repetitions):
                    for case in cases:
                        sample = {'model': model['id'], 'threads': threads, 'case': case['id'],
                                  'repetition': repetition, 'input': case, 'before': system(),
                                  'characters': sum(len(m['content']) for m in case['messages'])}
                        telemetry = []
                        stopped = threading.Event()
                        def poll():
                            while not stopped.is_set():
                                telemetry.append({'time': time.monotonic(), 'llama': proc(status['processId']), 'pins': proc(pins_pid)})
                                stopped.wait(.2)
                        worker = threading.Thread(target=poll, daemon=True); worker.start()
                        started = time.monotonic(); first = None; last = None
                        output = []; metrics = []; error = None; done = False
                        try:
                            with request('chat', {'messages': case['messages'], 'locale': case['locale']}) as response:
                                event = ''
                                for raw in response:
                                    line = raw.decode().strip()
                                    if line.startswith('event: '): event = line[7:]
                                    if not line.startswith('data: '): continue
                                    chunk = json.loads(line[6:])
                                    if event == 'token' and chunk.get('text'):
                                        now = time.monotonic()
                                        if first is None: first = now
                                        last = now; output.append(chunk['text'])
                                    elif event == 'metrics': metrics.append(chunk['timings'])
                                    elif event == 'error': error = chunk; break
                                    elif event == 'done': done = True
                        except (OSError, urllib.error.HTTPError) as ex:
                            error = str(ex)
                        finally:
                            stopped.set(); worker.join(timeout=2)
                        duration = time.monotonic() - started
                        sample.update({'totalSeconds': duration, 'firstContentSeconds': None if first is None else first - started,
                                       'output': ''.join(output), 'metrics': metrics, 'completed': done, 'error': error,
                                       'after': system(), 'telemetry': telemetry})
                        if not done:
                            sample['runtimeDiagnostics'] = api('diagnostics')
                            sample['failureStatus'] = api('status')
                        llama_samples = [s['llama'] for s in telemetry if s['llama']]
                        if llama_samples:
                            sample['peakRssMiB'] = max(s['rss_kib'] for s in llama_samples) / 1024
                            sample['peakSwapMiB'] = max(s['swap_kib'] for s in llama_samples) / 1024
                            elapsed = telemetry[-1]['time'] - telemetry[0]['time']
                            if elapsed > 0:
                                sample['llamaCpuPercentOneCore'] = (llama_samples[-1]['ticks'] - llama_samples[0]['ticks']) / os.sysconf('SC_CLK_TCK') / elapsed * 100
                        pins_samples = [s['pins'] for s in telemetry if s['pins']]
                        if pins_samples and len(telemetry) > 1:
                            sample['pinsCpuPercentOneCore'] = (pins_samples[-1]['ticks'] - pins_samples[0]['ticks']) / os.sysconf('SC_CLK_TCK') / (telemetry[-1]['time'] - telemetry[0]['time']) * 100
                        for metric in metrics:
                            if metric.get('predicted_per_second'):
                                sample['outputTokensPerSecond'] = metric['predicted_per_second']
                            if metric.get('prompt_n') is not None:
                                sample['promptTokens'] = metric['prompt_n']
                            if metric.get('completion_tokens') is not None:
                                sample['outputTokens'] = metric['completion_tokens']
                            elif metric.get('predicted_n') is not None:
                                sample['outputTokens'] = metric['predicted_n']
                        if 'outputTokensPerSecond' not in sample and sample.get('outputTokens', 0) > 1 and first is not None and last > first:
                            sample['outputTokensPerSecond'] = (sample['outputTokens'] - 1) / (last - first)
                        report['samples'].append(sample); persist()
                        print(f"{case['id']}: first={sample['firstContentSeconds']} speed={sample.get('outputTokensPerSecond')} completed={done}", flush=True)
                        if not done:
                            raise RuntimeError(f'Benchmark stopped on failed request: {error}')
                        time.sleep(args.case_pause_seconds)
                api('unload', True)
                # Let temperature settle between profiles; do not actively alter Pi cooling.
                time.sleep(10)
    finally:
        config_path.write_bytes(original); os.chmod(config_path, 0o600)
        try: api('reload', True)
        finally: persist()
    summaries = []
    for model in args.models:
        for threads in args.threads:
            samples = [s for s in report['samples'] if s['model'] == model and s['threads'] == threads]
            if not samples: continue
            latency = [s['firstContentSeconds'] for s in samples if s['firstContentSeconds'] is not None]
            speed = [s['outputTokensPerSecond'] for s in samples if 'outputTokensPerSecond' in s]
            summaries.append({'model': model, 'threads': threads, 'n': len(samples),
                              'firstContentP50Seconds': statistics.median(latency) if latency else None,
                              'firstContentP95Seconds': percentile(latency, .95) if latency else None,
                              'medianOutputTokensPerSecond': statistics.median(speed) if speed else None,
                              'peakRssMiB': max(s.get('peakRssMiB', 0) for s in samples),
                              'peakSwapMiB': max(s.get('peakSwapMiB', 0) for s in samples)})
    report['summaries'] = summaries; persist()
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
