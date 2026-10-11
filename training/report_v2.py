"""Summarize only measured reports; absent/incomplete model stages remain explicit."""
from common import ROOT, write_json, fingerprint
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from v2_contracts import definitions, POLICY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'artifacts/v2')
    parser.add_argument('--training-run', type=Path, help='Optional full v2 run directory, including Drive outputs')
    parser.add_argument('--smoke-run', type=Path, help='Optional v2 smoke run directory')
    args = parser.parse_args()
    output = args.directory
    lines = ['# PINS AI v2 measured evidence', '',
             'Synthetic fixtures; no equipment operations. Partial eval_gold includes conditional polling, not autonomous job workflows.', '',
             '| Model / suite | Completed | Cases | Tool choice | Exact arguments |',
             '| --- | --- | ---: | ---: | ---: |']
    summary = {}
    for model in ('base', 'v1', 'v2'):
        for suite in ('eval', 'hard-gold'):
            path = output / f'{model}-{suite}.json'
            rescored = output / f'{model}-{suite}-rescored.json'
            if rescored.is_file():
                path = rescored
            if not path.is_file():
                lines.append(f'| {model} / {suite} | Pending | — | — | — |')
                continue
            report = json.loads(path.read_text(encoding='utf-8'))
            results = report['results']; count = len(results)
            choices = sum(r['toolChoiceCorrect'] for r in results)
            exact = sum(r['exactArgumentsCorrect'] for r in results)
            categories = defaultdict(Counter)
            risks = defaultdict(Counter)
            tools = definitions()
            for row in results:
                name = row['expected'].get('first_tool', (row['expected'].get('tool_sequence') or [None])[0])
                risk = 'no_tool' if name is None else 'read_only' if tools[name]['annotations']['readOnlyHint'] else 'modifying_requires_confirmation'
                for counts in (categories[row['category']], risks[risk]):
                    counts['cases'] += 1
                    counts['toolChoiceCorrect'] += row['toolChoiceCorrect']
                    counts['exactArgumentsCorrect'] += row['exactArgumentsCorrect']
                    counts['schemaValid'] += row['schemaValid']
                    if name is None:
                        counts['noToolCases'] += 1
                        counts['noToolCorrect'] += row['toolChoiceCorrect']
                    if row.get('conditionalPoll'):
                        counts['conditionalPollCases'] += 1
                        counts['conditionalPollCorrect'] += row['exactArgumentsCorrect']
            summary[model + '-' + suite] = {'completed': report.get('completed', False), 'cases': count,
                                           'scorerVersion': report.get('scorerVersion', 'original'),
                                           'predictionReport': path.name,
                                           'currentPolicyMatch': report.get('policySha256') == fingerprint(POLICY),
                                           'toolChoiceCorrect': choices, 'exactArgumentsCorrect': exact,
                                           'perLanguage': report['perLanguage'],
                                           'perCategory': {key: dict(value) for key, value in categories.items()},
                                           'perRisk': {key: dict(value) for key, value in risks.items()},
                                           'schemaValid': sum(r['schemaValid'] for r in results),
                                           'maxGpuAllocatedMiB': report.get('maxGpuAllocatedMiB'),
                                           'meanGenerationSeconds': sum(r['seconds'] for r in results) / count if count else None,
                                           'generationTokensPerSecond': sum(r['outputTokens'] for r in results) / sum(r['seconds'] for r in results) if count else None,
                                           'hardwareOperations': 0,
                                           'unauthorizedHardwareActionRate': 'not exercised; inference-only predictions',
                                           'falseCompletionClaims': 'human review pending',
                                           'confusions': [{'id': r['id'], 'expected': r['expected'], 'predicted': r['call'], 'error': r['error']}
                                                          for r in results if not r['exactArgumentsCorrect']]}
            lines.append(f"| {model} / {suite} | {'Yes' if report.get('completed') else 'In progress'} | {count} | {choices}/{count} | {exact}/{count} |")
    lines += ['', 'GGUF: pending export and Pi evaluation.',
              'Corrected scoring rejects bare JSON tool proposals rather than crediting them as no-tool answers. Original prediction reports are preserved; rescored reports take precedence.',
              'Predictions with a different policy hash are historical evidence; they are not measurements of the revised dataset/policy.',
              'Language, factual correctness and false-completion claims: human review pending.',
              'Authorization: evaluated separately by deterministic simulated gateway tests; model tool proposals never execute here.', '',
              'Original hard-gold connect/stop labels express intent without gateway approval. Model choice scoring is not execution authorization.']
    lines += ['', '## Training metrics', '',
              '| Run | Status | Train loss | Best eval loss | Latest eval loss | Selected checkpoint |',
              '| --- | --- | ---: | ---: | ---: | --- |']
    for name in ('v1', 'v2-smoke', 'v2'):
        directory = ROOT / '.local/training/runs' / f'qwen3-pins-{name}-20261011'
        if name == 'v2' and args.training_run is not None: directory = args.training_run
        if name == 'v2-smoke' and args.smoke_run is not None: directory = args.smoke_run
        run_path = directory / 'training-run.json'
        if not run_path.is_file():
            lines.append(f'| {name} | Pending | — | — | — | — |')
            continue
        run = json.loads(run_path.read_text(encoding='utf-8'))
        latest_path = directory / 'metrics-latest.json'
        latest = json.loads(latest_path.read_text(encoding='utf-8')) if latest_path.is_file() else {}
        metrics = run.get('metrics', {})
        def value(number):
            return '—' if number is None else f'{number:.6f}'
        checkpoint = run.get('bestModelCheckpoint')
        if not checkpoint:
            states = sorted(directory.glob('checkpoints/checkpoint-*/trainer_state.json'),
                            key=lambda p: int(p.parent.name.split('-')[-1]))
            if states:
                checkpoint = json.loads(states[-1].read_text(encoding='utf-8')).get('best_model_checkpoint')
        checkpoint = checkpoint.replace('\\', '/').split('/')[-1] if checkpoint else '—'
        lines.append(f"| {name} | {run['status']} | {value(metrics.get('train_loss', latest.get('train_loss')))} | {value(metrics.get('eval_loss', latest.get('best_eval_loss')))} | {value(latest.get('eval_loss'))} | {checkpoint} |")
    lines += ['', 'Completed-run train loss is the whole-run average; running-run train loss is the latest logging-window average. Smoke loss is not a full training result.']
    (output / 'evaluation-summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(output / 'evaluation-summary.json', summary)
    print(output / 'evaluation-summary.md')


if __name__ == '__main__':
    main()
