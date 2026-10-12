"""Run an already smoke-verified checkpoint continuation and offline post-training smoke."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from common import ROOT, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'data', 'config', 'checkpoint', 'smoke-run', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    for name in ('base', 'data', 'config', 'checkpoint', 'smoke_run', 'output'):
        setattr(args, name, getattr(args, name).resolve())
    if args.output.exists(): raise FileExistsError('Use a NEW pipeline output directory')
    smoke = json.loads((args.smoke_run / 'training-run.json').read_text(encoding='utf-8'))
    if smoke['status'] != 'completed' or not smoke['smokeOnly'] or smoke.get('resumeSmokeSteps') != 1:
        raise ValueError('A successful one-step checkpoint resume smoke is required')
    args.output.mkdir(parents=True, exist_ok=False)
    state = {'status': 'starting', 'pid': os.getpid(), 'startedAt': datetime.now(timezone.utc).isoformat(),
             'checkpoint': str(args.checkpoint), 'startStep': smoke['resume']['startStep'],
             'targetSteps': smoke['resume']['targetSteps'], 'hardwareExecution': False,
             'trainingOutput': str(args.output / 'full'), 'jobs': []}
    def persist():
        temporary = args.output / 'pipeline.json.tmp'
        write_json(temporary, state)
        temporary.replace(args.output / 'pipeline.json')
    def run(name, command):
        log_path = args.output / (name + '.log')
        with log_path.open('x', encoding='utf-8') as log:
            flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            child = subprocess.Popen([sys.executable, *map(str, command)], cwd=ROOT, stdout=log,
                                     stderr=subprocess.STDOUT, creationflags=flags)
            state.update(status=name, childPid=child.pid)
            state['jobs'].append({'name': name, 'pid': child.pid, 'command': list(map(str, command)), 'log': str(log_path)})
            persist()
            code = child.wait()
        state['jobs'][-1]['exitCode'] = code; persist()
        if code: raise RuntimeError(f'{name} exited {code}; inspect {log_path}')
    persist()
    try:
        run('training', ['training/train.py', '--base', args.base, '--dataset-mode', 'mcp',
                        '--data', args.data, '--config', args.config, '--resume-from-checkpoint', args.checkpoint,
                        '--smoke-run', args.smoke_run, '--logging-steps', '5', '--output', args.output / 'full'])
        result = json.loads((args.output / 'full/training-run.json').read_text(encoding='utf-8'))
        if result['status'] != 'completed' or result['smokeOnly']:
            raise ValueError('Continuation did not produce a completed specialization')
        state.update(trainingCompleted=True, selectedBestCheckpoint=result['bestModelCheckpoint'],
                     finalAdapter=str(args.output / 'full/adapter'), finalGlobalStep=result['metrics']['finalGlobalStep'])
        persist()
        run('final-smoke', ['training/evaluate_v2.py', '--base', args.base, '--adapter', args.output / 'full/adapter',
                           '--suite', 'validation_smoke', '--per-language', '2', '--max-new-tokens', '128',
                           '--output', args.output / 'final-smoke.json'])
        evaluation = json.loads((args.output / 'final-smoke.json').read_text(encoding='utf-8'))
        if not evaluation['completed'] or evaluation['expectedCaseCount'] != 20:
            raise ValueError('Final smoke evaluation is incomplete')
        state.update(status='completed', finishedAt=datetime.now(timezone.utc).isoformat(),
                     smokeSummary=evaluation['overall'], fullValidationPending=True)
        persist()
    except BaseException as error:
        state.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        persist()
        raise


if __name__ == '__main__': main()
