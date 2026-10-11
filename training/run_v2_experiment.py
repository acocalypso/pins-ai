"""Sequential local GPU experiment; preserves v1, stops on failure, never deploys or calls equipment."""
import json
import argparse
from pathlib import Path
import os
import subprocess
import sys
from datetime import datetime, timezone

from common import ROOT, sha256, write_json, fingerprint
from v2_contracts import POLICY, VERIFIED, ADAPTER, PACKAGE


def main():
    os.environ['PYTHONUTF8'] = '1'
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / '.local/training/runs/qwen3-pins-v2-20261011')
    parser.add_argument('--evidence-directory', type=Path, default=ROOT / 'artifacts/v2')
    parser.add_argument('--smoke-run', type=Path, default=ROOT / '.local/training/runs/qwen3-pins-v2-smoke-20261011')
    parser.add_argument('--smoke-evaluation', type=Path, default=ROOT / 'artifacts/v2/smoke-reload-eval.json')
    args = parser.parse_args()
    destination = args.output.resolve()
    evidence = args.evidence_directory.resolve()
    prior = ROOT / '.local/training/runs/qwen3-pins-v1-20261011/adapter'
    smoke = args.smoke_run / 'training-run.json'
    info = json.loads(smoke.read_text(encoding='utf-8'))
    if info['status'] != 'completed' or not info['smokeOnly'] or info['precision'] != 'fp16':
        raise ValueError('Require successful FP16/NF4 smoke test first')
    if (info.get('promptSha256') != fingerprint(POLICY) or
        info['dataset'].get('contractSha256') != sha256(VERIFIED) or
        info['dataset'].get('adapterContractSha256') != sha256(ADAPTER) or
        info['dataset'].get('auditSha256') != sha256(PACKAGE / 'data/contract-audit.json')):
        raise ValueError('MCP policy/source changed; a fresh verified GPU smoke is required before full training')
    reload_report = json.loads(args.smoke_evaluation.read_text(encoding='utf-8'))
    if (not reload_report.get('completed') or
        reload_report.get('adapterSha256') != sha256(args.smoke_run / 'adapter/adapter_model.safetensors') or
        reload_report.get('policySha256') != fingerprint(POLICY) or
        reload_report.get('adapterContractSha256') != sha256(ADAPTER)):
        raise ValueError('Require completed adapter reload/toy inference')
    if destination.exists():
        raise ValueError('Use a new run directory; existing runs are preserved')
    report_path = evidence / 'experiment.json'
    if report_path.exists():
        raise ValueError('Experiment already exists; inspect it before starting another GPU process')
    prior_hash = sha256(prior / 'adapter_model.safetensors')
    evidence.mkdir(parents=True, exist_ok=True)
    stages = []
    common_eval = ['training/evaluate_v2.py', '--per-language', '2', '--job-continuations']
    for name, adapter in (('base', None), ('v1', prior)):
        stages.append((name + '-eval', common_eval + (['--adapter', str(adapter)] if adapter else []) +
                       ['--output', str(evidence / f'{name}-eval.json')]))
        stages.append((name + '-hard-gold', ['training/evaluate_v2.py', '--suite', 'hard_gold'] +
                       (['--adapter', str(adapter)] if adapter else []) + ['--output', str(evidence / f'{name}-hard-gold.json')]))
    stages.append(('v2-training', ['training/train.py', '--base', '.local/training/base', '--dataset-mode', 'mcp',
                                  '--config', 'training/config-v2.json', '--output', str(destination),
                                  '--smoke-run', str(args.smoke_run), '--logging-steps', '5']))
    stages.append(('v2-eval', common_eval + ['--adapter', str(destination / 'adapter'), '--output', str(evidence / 'v2-eval.json')]))
    stages.append(('v2-hard-gold', ['training/evaluate_v2.py', '--suite', 'hard_gold', '--adapter', str(destination / 'adapter'),
                                  '--output', str(evidence / 'v2-hard-gold.json')]))
    report = {'status': 'running', 'startedAt': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
              'priorAdapterSha256': prior_hash, 'stages': [], 'quantizedEvaluation': 'pending GGUF export and Pi validation',
              'hardwareOperations': 'none', 'selectionPolicy': 'best validation checkpoint; gold scores do not change training'}
    try:
        for name, command in stages:
            report['currentStage'] = name
            write_json(report_path, report)
            print('START ' + name, flush=True)
            log = evidence / (name + '.log')
            with log.open('w', encoding='utf-8') as stream:
                result = subprocess.run([sys.executable, '-u', *command], stdout=stream, stderr=subprocess.STDOUT)
            report['stages'].append({'name': name, 'exitCode': result.returncode, 'log': str(log)})
            if result.returncode:
                raise RuntimeError(name + ' failed; inspect ' + str(log))
            if sha256(prior / 'adapter_model.safetensors') != prior_hash:
                raise ValueError('Prior adapter changed during experiment')
            print('DONE ' + name, flush=True)
        report['status'] = 'completed'
    except Exception as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        write_json(report_path, report)


if __name__ == '__main__':
    main()
