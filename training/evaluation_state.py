"""Deterministic offline evaluation state, strict resume validation and atomic progress."""
from collections import Counter, defaultdict
import copy
import inspect
import json
import math
import os
from pathlib import Path
import tempfile

from jsonschema import ValidationError
from common import fingerprint, read_jsonl, sha256
from v2_contracts import PACKAGE, POLICY, compact_tools, parse_prediction, render, validate_call

REPORT_VERSION = 3


def select_cases(suite, per_language=0, job_continuations=False):
    if per_language < 0:
        raise ValueError('per-language must be nonnegative')
    path = PACKAGE / 'data/audited/validation.jsonl' if suite == 'validation_smoke' else PACKAGE / 'data' / f'{suite}.jsonl'
    rows = read_jsonl(path)
    if suite == 'validation_smoke':
        rows = [{**r, 'messages': r['messages'][:2]} for r in rows]
    inputs = {path.relative_to(PACKAGE).as_posix(): sha256(path)}
    if per_language:
        selected = []
        for locale in sorted({r['locale'] for r in rows}):
            candidates = sorted((r for r in rows if r['locale'] == locale), key=lambda r: fingerprint(r['id']))
            calls = [r for r in candidates if r['expected'].get('first_tool') or r['expected'].get('tool_sequence')]
            no_calls = [r for r in candidates if not r['expected'].get('first_tool') and not r['expected'].get('tool_sequence')]
            selected += calls[:(per_language + 1)//2] + no_calls[:per_language//2]
        rows = selected
    if job_continuations and suite == 'eval_gold':
        poll_path = PACKAGE / 'data/audited/test.jsonl'
        inputs[poll_path.relative_to(PACKAGE).as_posix()] = sha256(poll_path)
        for row in read_jsonl(poll_path):
            if row['category'] != 'job_followup':
                continue
            if per_language and any(r['locale'] == row['locale'] and r.get('conditionalPoll') for r in rows):
                continue
            rows.append({**row, 'id': row['id'] + '/poll', 'messages': row['messages'][:4],
                         'expected': {'first_tool': 'get_job', 'arguments': row['expected']['arguments'][1]},
                         'conditionalPoll': True})
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Dataset has duplicate case IDs')
    return rows, path, inputs


def prepare_cases(tokenizer, rows):
    prepared = []
    for row in rows:
        messages = copy.deepcopy(row['messages'])
        context = messages[0]['content']
        suffixes = [context.find(marker) for marker in (' Trusted gateway', ' Verified MCP') if marker in context]
        messages[0]['content'] = POLICY + (context[min(suffixes):] if suffixes else '')
        names = [tool['name'] for tool in row['tools']]
        prompt = render(tokenizer, messages, compact_tools(names), True)
        prepared.append({**row, 'prompt': prompt, 'allowedNames': names,
                         'caseSha256': fingerprint(row), 'promptSha256': fingerprint(prompt)})
    return prepared


def score(case, text):
    expected = case['expected']
    expected_name = expected.get('first_tool', (expected.get('tool_sequence') or [None])[0])
    expected_args = expected.get('arguments')
    if isinstance(expected_args, list):
        expected_args = expected_args[0]
    error = None
    try:
        call = parse_prediction(text, case['allowedNames'])
        valid = True
    except (ValueError, json.JSONDecodeError, ValidationError) as exc:
        call, valid, error = None, False, str(exc)
    name = call['name'] if call else None
    choice_ok = valid and name == expected_name
    exact_ok = choice_ok and (expected_name is None or call['arguments'] == expected_args)
    return {'call': call, 'error': error, 'schemaValid': valid,
            'toolChoiceCorrect': choice_ok, 'exactArgumentsCorrect': exact_ok}


def scoring_hash():
    return fingerprint([inspect.getsource(fn) for fn in (score, parse_prediction, validate_call)])


def summarize(report):
    languages, categories = defaultdict(Counter), defaultdict(Counter)
    overall = Counter()
    for result in report['results']:
        for counts in (languages[result['locale']], categories[result['category']], overall):
            counts['cases'] += 1
            for key in ('toolChoiceCorrect', 'exactArgumentsCorrect', 'schemaValid'):
                counts[key] += result[key]
            counts['outputTruncated'] += bool(result.get('outputTruncated'))
        overall['outputTokens'] += result['outputTokens']
        overall['seconds'] += result['seconds']
    report['perLanguage'] = {key: dict(value) for key, value in sorted(languages.items())}
    report['perCategory'] = {key: dict(value) for key, value in sorted(categories.items())}
    report['overall'] = dict(overall)
    report['casesCompleted'] = len(report['results'])
    report['completed'] = len(report['results']) == report['expectedCaseCount']
    report['status'] = 'completed' if report['completed'] else 'in_progress'
    # Keep legacy partial field meaning: selected subset of the suite, distinct from interruption.
    report['subset'] = bool(report['identity']['selection']['perLanguage'])
    report['interrupted'] = not report['completed']
    return report


def new_report(identity, cases, legacy_fields=None):
    report = {**(legacy_fields or {}), 'reportVersion': REPORT_VERSION, 'identity': identity,
              'identitySha256': fingerprint(identity), 'expectedCaseCount': len(cases),
              'expectedCaseIds': [c['id'] for c in cases], 'results': [], 'executionSegments': []}
    return summarize(report)


def resume_report(path, current, cases):
    saved = json.loads(Path(path).read_text(encoding='utf-8'))
    if saved.get('reportVersion') != REPORT_VERSION:
        raise ValueError('Legacy report lacks provable resume identity; preserve it and start a NEW evaluation')
    if saved.get('identity') != current['identity'] or saved.get('identitySha256') != fingerprint(current['identity']):
        raise ValueError('Resume model/adapter/base/dataset/scorer/prompt/generation identity mismatch')
    if saved.get('expectedCaseIds') != current['expectedCaseIds'] or saved.get('expectedCaseCount') != len(cases):
        raise ValueError('Resume case selection mismatch')
    recorded = saved.get('results')
    if not isinstance(recorded, list):
        raise ValueError('Invalid recorded results')
    seen = set()
    for index, result in enumerate(recorded):
        if not isinstance(result, dict) or result.get('id') in seen:
            raise ValueError('Duplicate/invalid recorded case ID')
        seen.add(result.get('id'))
        if index >= len(cases) or result.get('id') != cases[index]['id']:
            raise ValueError('Recorded cases are not the deterministic dataset prefix')
        case = cases[index]
        for key in ('locale', 'category', 'expected', 'caseSha256', 'promptSha256'):
            if result.get(key) != case[key]:
                raise ValueError('Recorded case input mismatch: ' + case['id'])
        if result.get('conditionalPoll') != case.get('conditionalPoll', False):
            raise ValueError('Recorded polling context mismatch')
        if not isinstance(result.get('text'), str):
            raise ValueError('Missing recorded prediction')
        if any(result.get(key) != value for key, value in score(case, result['text']).items()):
            raise ValueError('Recorded scoring mismatch: ' + case['id'])
        if (not isinstance(result.get('outputTokens'), int) or isinstance(result['outputTokens'], bool)
                or result['outputTokens'] < 0 or result['outputTokens'] > current['identity']['generation']['maxNewTokens']
                or not isinstance(result.get('seconds'), (int, float)) or isinstance(result['seconds'], bool)
                or not math.isfinite(result['seconds']) or result['seconds'] <= 0
                or not isinstance(result.get('outputTruncated'), bool)):
            raise ValueError('Invalid recorded generation metrics')
    if saved.get('completed') and len(recorded) != len(cases):
        raise ValueError('Incomplete report falsely marked completed')
    result = copy.deepcopy(current)
    result['results'] = copy.deepcopy(recorded)
    result['executionSegments'] = copy.deepcopy(saved.get('executionSegments', []))
    result['resumeFrom'] = str(Path(path).resolve())
    result['resumeSourceSha256'] = sha256(path)
    return summarize(result)


class ProgressWriter:
    """Reserve a NEW report path, then replace only our own file with fsynced progress."""
    def __init__(self, output, original=None):
        self.path = Path(output)
        if original and self.path.resolve() == Path(original).resolve():
            raise ValueError('Resume output must differ from the original report')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('x', encoding='utf-8'):
            pass

    def write(self, report):
        fd, name = tempfile.mkstemp(prefix=self.path.name + '.', suffix='.tmp', dir=self.path.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
            os.replace(name, self.path)
        finally:
            if os.path.exists(name):
                os.unlink(name)
