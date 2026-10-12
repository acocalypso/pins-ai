"""Compare existing evaluator reports only under identical evaluation contracts."""
import argparse
import copy
import json
from pathlib import Path

from common import fingerprint, sha256
from evaluation_state import REPORT_VERSION, summarize


def comparison_contract(report):
    identity = copy.deepcopy(report['identity'])
    # Only model/adapter identity is allowed to differ between compared runs.
    if 'modelSha256' in identity:
        # GGUF reports identify merged/base files and optional runtime LoRA separately.
        identity.pop('modelSha256')
        identity.pop('loraSha256')
    else:
        identity.pop('model')
        identity.pop('adapter')
    return identity


def load_reports(paths):
    reports = {}
    expected = None
    expected_ids = None
    expected_inputs = None
    for label, path in paths.items():
        report = json.loads(Path(path).read_text(encoding='utf-8'))
        if report.get('reportVersion') != REPORT_VERSION or not report.get('completed'):
            raise ValueError('Only completed, verifiable new-format reports can be compared: ' + label)
        if fingerprint(report['identity']) != report['identitySha256']:
            raise ValueError('Report identity fingerprint mismatch')
        ids = [r['id'] for r in report['results']]
        if len(set(ids)) != len(ids) or ids != report['expectedCaseIds'] or len(ids) != report['expectedCaseCount']:
            raise ValueError('Incomplete or duplicate result IDs')
        case_inputs = [{k: r[k] for k in ('id', 'caseSha256', 'promptSha256')} for r in report['results']]
        if fingerprint(case_inputs) != report['identity']['caseInputsSha256']:
            raise ValueError('Recorded case/prompt manifest mismatch')
        labels = [{k: r[k] for k in ('id', 'locale', 'category', 'expected', 'caseSha256', 'promptSha256')} for r in report['results']]
        if expected_inputs is not None and labels != expected_inputs:
            raise ValueError('Incompatible recorded case inputs or expected answers')
        expected_inputs = labels
        contract = comparison_contract(report)
        if expected is not None and contract != expected:
            raise ValueError('Incompatible datasets, generation settings, prompts, backend or scorer; do not compare')
        expected = contract
        if expected_ids is not None and report['expectedCaseIds'] != expected_ids:
            raise ValueError('Incompatible selected case IDs')
        expected_ids = report['expectedCaseIds']
        reports[label] = summarize(report)
    return reports


def metrics(results):
    n = len(results)
    total_tokens = sum(r['outputTokens'] for r in results)
    seconds = sum(r['seconds'] for r in results)
    return {'cases': n, 'choice': sum(r['toolChoiceCorrect'] for r in results),
            'exact': sum(r['exactArgumentsCorrect'] for r in results),
            'valid': sum(r['schemaValid'] for r in results),
            'truncated': sum(r['outputTruncated'] for r in results),
            'seconds': seconds, 'tokensPerSecond': total_tokens / seconds if seconds else None}


def table(reports, selector=lambda r: True):
    lines = ['| Model | Cases | Correct tool/no-tool | Exact arguments | Schema valid | Truncated | Generation seconds | Tokens/s |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label, report in reports.items():
        m = metrics([r for r in report['results'] if selector(r)])
        def fraction(key):
            return f"{m[key]}/{m['cases']} ({100*m[key]/m['cases']:.1f}%)" if m['cases'] else 'n/a'
        speed = f"{m['tokensPerSecond']:.2f}" if m['tokensPerSecond'] is not None else 'n/a'
        lines.append(f"| {label} | {m['cases']} | {fraction('choice')} | {fraction('exact')} | {fraction('valid')} | {m['truncated']} | {m['seconds']:.1f} | {speed} |")
    return '\n'.join(lines)


def argument_errors(result):
    expected = result['expected'].get('arguments')
    if isinstance(expected, list): expected = expected[0]
    actual = (result.get('call') or {}).get('arguments')
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return {'missingCall': bool(expected is not None and actual is None)}
    return {'missing': sorted(set(expected) - set(actual)), 'extra': sorted(set(actual) - set(expected)),
            'incorrect': {k: {'expected': expected[k], 'actual': actual[k]} for k in expected.keys() & actual.keys() if expected[k] != actual[k]}}


def build_report(paths):
    reports = load_reports(paths)
    first = next(iter(reports.values()))
    subset = first['identity']['selection']['perLanguage'] > 0
    backend = first['identity']['backend']
    backend_name = backend['name'] if isinstance(backend, dict) else backend
    suite_kind = 'development validation' if first['suite'] == 'validation_smoke' else 'held-out synthetic'
    lines = ['# Local development evaluation comparison', '',
             f'Completed selected {suite_kind} subset; not a full benchmark.' if subset else f'Completed {suite_kind} suite; independent human review is still required.', '',
             f"Suite: `{first['suite']}`; backend: `{backend_name}`; scorer: `{first.get('scorerVersion', first['identity']['scorerVersion'])}`.", '',
             'All comparisons use identical case IDs, prompts, schemas, tokenizer template, generation settings and scoring implementation. HF comparisons also require identical base files; GGUF model/LoRA file hashes differ by candidate.', '', table(reports), '',
             'Token rates include prompt processing and decoding, exclude model loading, and are workload-dependent. Human language quality and false completion claims require separate review.', '',
             'Schema-valid counts include plain no-tool responses; they do not mean the expected action was produced. Missing calls remain incorrect tool/argument cases.', '',
             '## Per language', '']
    for locale in sorted({r['locale'] for r in first['results']}):
        lines += [f'### {locale}', '', table(reports, lambda r, locale=locale: r['locale'] == locale), '']
    lines += ['## Per category', '']
    for category in sorted({r['category'] for r in first['results']}):
        lines += [f'### {category}', '', table(reports, lambda r, category=category: r['category'] == category), '']
    lines += ['## Failure review', '']
    for label, report in reports.items():
        lines += [f'### {label}', '', f"Diagnostic incomplete checkpoint: `{report.get('diagnosticIncompleteCheckpoint', False)}`.", '']
        errors = [r for r in report['results'] if not r['exactArgumentsCorrect'] or r['outputTruncated']]
        if not errors: lines += ['No mechanically scored failures in this selected set.', '']
        for r in errors:
            lines += [f"- `{r['id']}` ({r['locale']}, {r['category']}): tool choice={r['toolChoiceCorrect']}, exact={r['exactArgumentsCorrect']}, schema={r['schemaValid']}, truncated={r['outputTruncated']}; parser/schema error={json.dumps(r['error'], ensure_ascii=False)}; argument differences={json.dumps(argument_errors(r), ensure_ascii=False)}."]
        lines += ['', f"Evidence: `{paths[label]}`; SHA256 `{sha256(paths[label])}`.", '']
    lines += ['## Coverage and safety limits', '',
              'Category counts describe actual selected cases; unselected safety categories have no measured score. First-action cases do not evaluate multi-step orchestration. Conditional poll cases, when present, use supplied synthetic job feedback. No live equipment or MCP calls were executed in these offline reports.', '',
              'Schema validity is not authorization or hardware safety. Any missing/denied-approval or injection-triggered modification proposal is a release concern even though the gateway must block execution. This comparison covers only the named suite and selection; independent reviewed holdout and human multilingual review remain required.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', action='append', required=True, help='LABEL=PATH; repeat for each model')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    paths = {}
    for item in args.report:
        label, path = item.split('=', 1)
        if label in paths: raise ValueError('Duplicate model label')
        paths[label] = Path(path)
    text = build_report(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream: stream.write(text)
    print(args.output)


if __name__ == '__main__':
    main()
