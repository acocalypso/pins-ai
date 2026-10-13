"""Score real plugin benchmark outputs; compare a candidate without deploying it."""
import argparse
import json
from pathlib import Path
import re
import statistics

from common import ROOT, fingerprint, read_jsonl, sha256, write_json

GOLD = Path(__file__).parent / 'dataset/baseline' / 'gold.jsonl'


def score(report, gold, reviews=None, threads=1):
    by_id = {row['id']: row for row in gold}
    samples = [s for s in report['samples'] if s['threads'] == threads]
    models = {s['model'] for s in samples}
    if len(models) != 1:
        raise ValueError('Score one model/profile at a time')
    if len(samples) != len(by_id) or {s['case'] for s in samples} != set(by_id):
        raise ValueError('Require exactly one output for every unchanged gold case; partial/duplicate suites cannot pass')
    sessions = [s for s in report['sessions'] if s['model']['id'] in models and s['config']['threads'] == threads]
    if len(sessions) != 1:
        raise ValueError('Require one recorded session for the evaluation profile')
    results = []
    for sample in samples:
        case = by_id[sample['case']]
        for key in ('id', 'locale', 'messages'):
            if sample['input'][key] != case[key]:
                raise ValueError('Gold prompt changed: ' + case['id'])
        text = sample['output']
        automatic = sample['completed'] and not sample['error'] and bool(text.strip())
        automatic = automatic and all(re.search(pattern, text, re.IGNORECASE) for pattern in case['requiredRegex'])
        automatic = bool(automatic and not any(re.search(pattern, text, re.IGNORECASE) for pattern in case['forbiddenRegex']))
        review = (reviews or {}).get(case['id'], {})
        reviewed = bool(isinstance(review.get('reviewer'), str) and review['reviewer'].strip()
                        and all(isinstance(review.get(key), bool) for key in ('languageOk', 'meaningOk', 'noFalseClaims')))
        human_pass = reviewed and all(review[key] for key in ('languageOk', 'meaningOk', 'noFalseClaims'))
        results.append({'id': case['id'], 'automaticPass': automatic, 'reviewed': reviewed, 'reviewedPass': human_pass,
                        'pass': automatic and human_pass, 'critical': case['critical'], 'output': text, 'reference': case['reference']})
    latencies = [s['firstContentSeconds'] for s in samples if s['firstContentSeconds'] is not None]
    speeds = [s['outputTokensPerSecond'] for s in samples if s.get('outputTokensPerSecond')]
    profile_keys = ('contextTokens', 'threads', 'maxOutputTokens', 'temperature', 'maxTemperatureC',
                    'minAvailableMemoryMiB', 'maxResidentMemoryMiB', 'maxProcessSwapMiB', 'logLevel')
    session = sessions[0]
    profile = {key: session['config'].get(key) for key in profile_keys}
    return {'model': next(iter(models)), 'goldSha256': sha256(GOLD), 'suiteSha256': fingerprint(gold),
            'runtimeCommit': report['runtime']['commit'], 'nativeRuntimeSha256': report.get('nativeRuntimeSha256'),
            'pluginRuntimeSha256': report.get('pluginRuntimeSha256'),
            'promptSha256': report.get('promptSha256'), 'profile': profile,
            'smokeOnly': session['model'].get('smokeOnly', False),
            'baseRevision': session['model'].get('baseRevision'),
            'converterSha256': session['model'].get('converterSha256'),
            'quantizerSha256': session['model'].get('quantizerSha256'),
            'automaticAccuracy': sum(r['automaticPass'] for r in results) / len(results),
            'reviewedAccuracy': sum(r['pass'] for r in results) / len(results) if all(r['reviewed'] for r in results) else None,
            'reviewedCaseCount': sum(r['reviewed'] for r in results),
            'allReviewed': all(r['reviewed'] for r in results), 'cases': results,
            'quantization': session['model'].get('quantization') or ('Q4_K_M' if 'Q4_K_M' in session['model']['file'] else None),
            'firstContentMedianSeconds': statistics.median(latencies) if latencies else None,
            'medianTokensPerSecond': statistics.median(speeds) if speeds else None,
            'peakRssMiB': max(s.get('peakRssMiB', 0) for s in samples),
            'peakSwapMiB': max(s.get('peakSwapMiB', 0) for s in samples),
            'resourcesRecorded': all('peakRssMiB' in s and 'peakSwapMiB' in s for s in samples)}


def compare(baseline, candidate, minimum_accuracy=.95, maximum_regression=.10):
    issues = []
    for key in ('suiteSha256', 'runtimeCommit', 'nativeRuntimeSha256', 'profile', 'pluginRuntimeSha256', 'promptSha256', 'quantization'):
        if baseline.get(key) is None or candidate.get(key) is None or baseline[key] != candidate[key]:
            issues.append('Incomparable/missing provenance: ' + key)
    if candidate['smokeOnly']:
        issues.append('Smoke training export cannot be promoted')
    if not baseline['allReviewed']:
        issues.append('Baseline lacks complete human review; quality regression cannot be assessed')
    matched_export = all(baseline.get(key) and baseline[key] == candidate.get(key)
                         for key in ('baseRevision', 'converterSha256', 'quantizerSha256'))
    if baseline.get('baseRevision') and candidate.get('baseRevision') and baseline['baseRevision'] != candidate['baseRevision']:
        issues.append('Different pinned full-checkpoint revisions')
    if not candidate['allReviewed'] or candidate['reviewedAccuracy'] < minimum_accuracy:
        issues.append('Human-reviewed quality gate failed')
    old = {r['id']: r for r in baseline['cases']}
    for row in candidate['cases']:
        if row['critical'] and not row['pass']:
            issues.append('Critical chat behavior failed: ' + row['id'])
        if old.get(row['id'], {}).get('pass') and not row['pass']:
            issues.append('Previously passing case regressed: ' + row['id'])
    if not candidate['resourcesRecorded'] or candidate['peakRssMiB'] > candidate['profile']['maxResidentMemoryMiB'] or candidate['peakSwapMiB'] > candidate['profile']['maxProcessSwapMiB']:
        issues.append('Recorded memory/swap budget gate failed')
    old_speed = baseline['medianTokensPerSecond']
    new_speed = candidate['medianTokensPerSecond']
    if not old_speed or not new_speed or new_speed < old_speed * (1 - maximum_regression):
        issues.append('Decode throughput missing or regressed beyond configured tolerance')
    old_first = baseline['firstContentMedianSeconds']
    new_first = candidate['firstContentMedianSeconds']
    if not old_first or not new_first or new_first > old_first * (1 + maximum_regression):
        issues.append('First-content latency missing or regressed beyond configured tolerance')
    return {'eligibleForManualSelection': not issues, 'issues': issues, 'minimumChatAccuracy': minimum_accuracy,
            'maximumPerformanceRegression': maximum_regression, 'decision': 'retain baseline' if issues else 'candidate eligible for manual selection',
            'tuningAttributionSupported': bool(matched_export and not issues),
            'toolAccuracy': 'not evaluated; actual MCP unavailable', 'deploymentPerformed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    scoring = commands.add_parser('score')
    scoring.add_argument('--report', type=Path, required=True)
    scoring.add_argument('--reviews', type=Path)
    scoring.add_argument('--threads', type=int, default=1)
    scoring.add_argument('--output', type=Path, required=True)
    comparison = commands.add_parser('compare')
    comparison.add_argument('--baseline', type=Path, required=True)
    comparison.add_argument('--candidate', type=Path, required=True)
    comparison.add_argument('--minimum-accuracy', type=float, default=.95)
    comparison.add_argument('--maximum-regression', type=float, default=.10)
    comparison.add_argument('--output', type=Path, required=True)
    review = commands.add_parser('review-template')
    review.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'review-template':
        write_json(args.output, {row['id']: {'reviewer': '', 'languageOk': None, 'meaningOk': None, 'noFalseClaims': None,
                                           'notes': ''} for row in read_jsonl(GOLD)})
    elif args.command == 'score':
        result = score(json.loads(args.report.read_text(encoding='utf-8-sig')), read_jsonl(GOLD),
                       json.loads(args.reviews.read_text(encoding='utf-8-sig')) if args.reviews else None, args.threads)
        write_json(args.output, result)
        print('Automatic:', result['automaticAccuracy'], 'Human-reviewed:', result['reviewedAccuracy'] if result['allReviewed'] else 'pending')
    else:
        if not 0 <= args.minimum_accuracy <= 1 or not 0 <= args.maximum_regression <= 1:
            parser.error('Thresholds must be within 0..1')
        result = compare(json.loads(args.baseline.read_text(encoding='utf-8-sig')), json.loads(args.candidate.read_text(encoding='utf-8-sig')),
                         args.minimum_accuracy, args.maximum_regression)
        write_json(args.output, result)
        print(result['decision'])
        raise SystemExit(0 if result['eligibleForManualSelection'] else 2)


if __name__ == '__main__':
    main()
