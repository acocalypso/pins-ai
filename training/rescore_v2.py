"""Rescore preserved predictions with a corrected parser; never regenerate or tune holdouts."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from jsonschema import ValidationError
from common import read_jsonl, sha256, write_json
from v2_contracts import PACKAGE, SCORER_VERSION, VERIFIED, parse_prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh output; original and rescored evidence are never overwritten')
    report = json.loads(args.input.read_text(encoding='utf-8'))
    if not report.get('completed'):
        raise ValueError('Wait for a completed prediction report')
    if report['contractSha256'] != sha256(VERIFIED):
        raise ValueError('Contract changed; cannot rescore as the same evaluation')
    suite = report['suite']
    path = PACKAGE / 'data' / f'{suite}.jsonl'
    if suite not in ('eval_gold', 'hard_gold') or report['suiteSha256'] != sha256(path):
        raise ValueError('Require the identical unchanged gold suite')
    fixtures = {r['id']: r for r in read_jsonl(path)}
    fixtures.update({r['id'] + '/poll': r for r in read_jsonl(PACKAGE / 'data/audited/test.jsonl')})
    tallies = defaultdict(Counter)
    changes = 0
    for result in report['results']:
        names = [t['name'] for t in fixtures[result['id']]['tools']]
        error = None
        try:
            call = parse_prediction(result['text'], names)
            valid = True
        except (ValueError, json.JSONDecodeError, ValidationError) as exc:
            call = None; valid = False; error = str(exc)
        expected = result['expected']
        name = expected.get('first_tool', (expected.get('tool_sequence') or [None])[0])
        arguments = expected.get('arguments')
        if isinstance(arguments, list): arguments = arguments[0]
        choice = valid and (call['name'] if call else None) == name
        exact = choice and (name is None or call['arguments'] == arguments)
        changes += (choice, exact, valid) != (result['toolChoiceCorrect'], result['exactArgumentsCorrect'], result['schemaValid'])
        result.update(call=call, error=error, schemaValid=valid, toolChoiceCorrect=choice, exactArgumentsCorrect=exact)
        counts = tallies[result['locale']]
        counts['cases'] += 1; counts['schemaValid'] += valid
        counts['toolChoiceCorrect'] += choice; counts['exactArgumentsCorrect'] += exact
    report.update(scorerVersion=SCORER_VERSION, rescoredFrom=str(args.input),
                  originalReportSha256=sha256(args.input), scoringChangedCases=changes,
                  perLanguage={language: dict(counts) for language, counts in tallies.items()})
    write_json(args.output, report)
    print(json.dumps({'output': str(args.output), 'changedCases': changes, 'cases': len(report['results']),
                      'toolChoiceCorrect': sum(r['toolChoiceCorrect'] for r in report['results']),
                      'exactArgumentsCorrect': sum(r['exactArgumentsCorrect'] for r in report['results'])}))


if __name__ == '__main__':
    main()
