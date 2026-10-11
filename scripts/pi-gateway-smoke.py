"""Upload/run an isolated ARM64 gateway fixture; never call live equipment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import re
import uuid


RUNNER = r'''
import json, os, pathlib, resource, signal, subprocess, time, urllib.request, urllib.error
root = pathlib.Path(__file__).resolve().parent
def temperature():
    return int(pathlib.Path('/sys/class/thermal/thermal_zone0/temp').read_text()) / 1000
before = temperature()
if before >= 75:
    raise SystemExit('Pi is too warm for even this short fixture; no test started.')
cmd = ['./Mcp.GatewayTests', '_catalog.json']
if (root / '_v2.json').exists():
    cmd += ['--prediction-audit', '_audit.json', '_v2.json', '_base.json']
start = time.monotonic()
usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
peak = before
reason = None
env = dict(os.environ, DOTNET_PROCESSOR_COUNT='1')
with open(root / '_stdout.log', 'w') as stdout, open(root / '_stderr.log', 'w') as stderr:
    process = subprocess.Popen(['nice', '-n', '10'] + cmd, cwd=root, env=env,
        stdout=stdout, stderr=stderr, start_new_session=True)
    while process.poll() is None:
        peak = max(peak, temperature())
        if peak >= 75 or time.monotonic() - start >= 30:
            reason = 'thermal cutoff' if peak >= 75 else '30-second timeout'
            os.killpg(process.pid, signal.SIGTERM)
            try: process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
            break
        time.sleep(0.1)
    code = process.wait()
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
try:
    with urllib.request.urlopen('http://127.0.0.1:4782/mcp', timeout=3) as response: mcp = response.status
except urllib.error.HTTPError as error: mcp = error.code
except urllib.error.URLError: mcp = 'unreachable'
result = {'hardwareExecution': False, 'inferenceExecuted': False,
    'testExitCode': code, 'cutoff': reason, 'elapsedSeconds': round(time.monotonic() - start, 3),
    'temperatureBeforeC': before, 'temperatureAfterC': temperature(), 'peakTemperatureC': peak,
    'pinsService': subprocess.run(['systemctl', 'is-active', 'pins.service'], capture_output=True, text=True).stdout.strip(),
    'mcpReadOnlyGetStatus': mcp, 'directory': str(root),
    'stdout': (root / '_stdout.log').read_text(), 'resources': (root / '_stderr.log').read_text(),
    'maxRssKiB': usage.ru_maxrss, 'cpuUserSeconds': round(usage.ru_utime - usage_before.ru_utime, 3),
    'cpuSystemSeconds': round(usage.ru_stime - usage_before.ru_stime, 3)}
if (root / '_audit.json').exists(): result['predictionAudit'] = json.loads((root / '_audit.json').read_text())
(root / '_result.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
raise SystemExit(code if reason is None else 1)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True, help='Self-contained dotnet publish -r linux-arm64 output')
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--v2-report', type=Path)
    parser.add_argument('--base-report', type=Path)
    parser.add_argument('--host', default='192.168.178.109')
    parser.add_argument('--user', default='pi')
    parser.add_argument('--known-hosts', type=Path, default=Path('.local/known_hosts'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/v3/pi-gateway-smoke.json'))
    parser.add_argument('--reuse-directory', help='Previously generated fixture directory; preserve logs locally before reuse')
    args = parser.parse_args()
    if bool(args.v2_report) != bool(args.base_report):
        parser.error('Both prediction reports are required for replay')
    if not (args.bundle / 'Mcp.GatewayTests').is_file():
        parser.error('Missing Linux self-contained gateway fixture executable')
    import paramiko
    client = paramiko.SSHClient()
    # Explicit task-specific verified host keys; stale global keys must not
    # override this file. Unknown/mismatched keys are still rejected.
    client.load_host_keys(str(args.known_hosts))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(args.host, username=args.user, password=os.environ.get('PINS_SSH_PASSWORD'), timeout=15)
    try:
        sftp = client.open_sftp()
        home = sftp.normalize('.')
        directory = args.reuse_directory or home + '/.local/share/pins-ai-tests/gateway-' + uuid.uuid4().hex
        if not re.fullmatch(re.escape(home + '/.local/share/pins-ai-tests/gateway-') + '[0-9a-f]{32}', directory):
            raise ValueError('Remote fixture must remain in the dedicated test directory')
        if args.reuse_directory:
            sftp.stat(directory + '/_run.py')
        current = home
        for component in ('.local', 'share', 'pins-ai-tests', directory.rsplit('/', 1)[1]):
            current += '/' + component
            try: sftp.stat(current)
            except FileNotFoundError: sftp.mkdir(current, mode=0o700)
        files = sorted(args.bundle.iterdir())
        if any(not file.is_file() for file in files):
            raise ValueError('Expected flat self-contained publish output')
        hashes = {}
        for file in files:
            sftp.put(str(file), directory + '/' + file.name)
            hashes[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
        sftp.chmod(directory + '/Mcp.GatewayTests', 0o700)
        for path, name in ((args.catalog, '_catalog.json'), (args.v2_report, '_v2.json'), (args.base_report, '_base.json')):
            if path: sftp.put(str(path), directory + '/' + name)
        with sftp.file(directory + '/_run.py', 'w') as handle: handle.write(RUNNER)
        _, stdout, stderr = client.exec_command('python3 ' + shlex.quote(directory + '/_run.py'), timeout=45)
        output = stdout.read().decode('utf-8')
        errors = stderr.read().decode('utf-8')
        status = stdout.channel.recv_exit_status()
        if output:
            result = json.loads(output)
            result['uploadedBinarySha256'] = hashes
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
            print(result['stdout'])
            print(result['resources'])
            print(json.dumps({key: value for key, value in result.items()
                              if key not in ('stdout', 'resources', 'uploadedBinarySha256', 'predictionAudit')}, indent=2))
        if errors: print(errors)
        if status: raise SystemExit(status)
    finally:
        client.close()


if __name__ == '__main__':
    main()
