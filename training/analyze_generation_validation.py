"""Compare frozen generated-validation reports offline; never author training data.

Standard library only. Historical stored metrics are preserved, not reinterpreted.
No model, network, pickle or equipment access is used.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import re

VERSION = 'generation-analysis-2'


def first_tool(expected):
    return expected['first_tool'] if 'first_tool' in expected else (expected.get('tool_sequence') or [None])[0]


def arguments(expected):
    value = expected.get('arguments')
    return value[0] if isinstance(value, list) else value


def proposed(result):
    """Parse invalid raw proposals for diagnostics only, never for execution."""
    if result.get('call') is not None:
        return result['call']
    blocks = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', result.get('text', ''), re.S)
    if len(blocks) == 1:
        try:
            value = json.loads(blocks[0])
            if isinstance(value, dict):
                return value
        except (ValueError, TypeError):
            pass
    return None


def diagnostics(result):
    expected = first_tool(result['expected'])
    call = proposed(result)
    text = result.get('text', '')
    wanted = arguments(result['expected']) or {}
    has_proposal = call is not None or '<tool_call>' in text or bool(re.search(r'"(?:arguments|tool_calls)"\s*:', text))
    errors = []
    opposite_tools = ({'park_mount', 'unpark_mount'}, {'connect_device', 'disconnect_device'},
                      {'start_guiding', 'stop_guiding'}, {'start_sequence', 'stop_sequence'})
    actual_args = (call or {}).get('arguments')
    if call and any({expected, call.get('name')} == pair for pair in opposite_tools):
        errors.append('critical_action_reversal')
    if isinstance(actual_args, dict):
        opposite_actions = ({'open', 'close'}, {'light_on', 'light_off'}, {'open_cover', 'close_cover'})
        if call.get('name') == expected:
            if any({wanted.get('action'), actual_args.get('action')} == pair for pair in opposite_actions):
                errors.append('critical_action_reversal')
            for key in ('mechanical', 'enabled'):
                if type(wanted.get(key)) is bool and type(actual_args.get(key)) is bool and wanted[key] != actual_args[key]:
                    errors.append('critical_action_reversal')
        for key in ('ra', 'dec'):
            if key in actual_args and key not in wanted:
                errors.append('invented_coordinate:' + key)
        if result.get('category') == 'missing_coords' and any(key in actual_args for key in ('ra', 'dec')):
            errors.append('invented_coordinates')
    if expected is None and has_proposal:
        errors.append('unwanted_proposal')
    if not result['schemaValid']:
        errors.append('invalid_no_tool_output' if expected is None else 'required_tool_schema_violation')
    if result.get('outputTruncated'):
        errors.append('truncated')
    if call is None:
        if has_proposal:
            errors.append('malformed_call')
        elif expected is not None:
            errors.append('missing_tool')
    elif call.get('name') != expected:
        errors.append('wrong_tool')
    else:
        actual = call.get('arguments')
        if not isinstance(actual, dict):
            errors.append('invalid_argument_object')
        else:
            for key in sorted(wanted.keys() - actual.keys()):
                errors.append('missing_argument:' + key)
            for key in sorted(actual.keys() - wanted.keys()):
                errors.append('extra_argument:' + key)
                if key in ('ra', 'dec'):
                    errors.append('unrequested_coordinate:' + key)
            for key in sorted(actual.keys() & wanted.keys()):
                if type(actual[key]) is not type(wanted[key]) and not (
                        type(actual[key]) in (int, float) and type(wanted[key]) in (int, float)):
                    errors.append('changed_type:' + key)
                elif actual[key] != wanted[key]:
                    kind = 'changed_numeric' if type(wanted[key]) in (int, float) else 'changed_value'
                    errors.append(kind + ':' + key)
    return sorted(set(errors))


def wilson(successes, count):
    if not count:
        return None
    z = 1.959963984540054
    p = successes / count
    center = (p + z*z/(2*count)) / (1 + z*z/count)
    half = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / (1+z*z/count)
    return [max(0, center-half), min(1, center+half)]


def summarize_results(rows):
    count = len(rows)
    exact = sum(bool(r['exactArgumentsCorrect']) for r in rows)
    errors = Counter(e for r in rows for e in diagnostics(r))
    no_tool = [r for r in rows if first_tool(r['expected']) is None]
    positive = [r for r in rows if first_tool(r['expected']) is not None]
    positive_exact = sum(bool(r['exactArgumentsCorrect']) for r in positive)
    no_tool_exact = sum(bool(r['exactArgumentsCorrect']) for r in no_tool)
    buckets = defaultdict(list)
    for row in rows: buckets[first_tool(row['expected']) or '__no_tool__'].append(row)
    macro = sum(sum(bool(r['exactArgumentsCorrect']) for r in group)/len(group) for group in buckets.values())/len(buckets) if buckets else None
    return {'expectedTool':len(positive), 'macroExact':macro,
            'inventedCoordinates':sum(any(e.startswith('invented_coordinate') for e in diagnostics(r)) for r in rows),
            'positiveToolCases': len(positive), 'positiveToolExact': positive_exact,
            'positiveToolExactWilson95': wilson(positive_exact, len(positive)),
            'correctNoTool': no_tool_exact, 'correctNoToolWilson95': wilson(no_tool_exact, len(no_tool)),
            'unwantedProposalWilson95': wilson(errors['unwanted_proposal'], len(no_tool)),
            'criticalActionReversals': errors['critical_action_reversal'],
            'cases': count, 'toolChoiceCorrect': sum(bool(r['toolChoiceCorrect']) for r in rows),
            'exactCorrect': exact, 'exactRate': exact/count if count else None,
            'exactWilson95': wilson(exact, count), 'schemaValid': sum(bool(r['schemaValid']) for r in rows),
            'expectedNoTool': len(no_tool), 'unwantedProposals': errors['unwanted_proposal'],
            'invalidNoToolOutputs': errors['invalid_no_tool_output'],
            'requiredToolSchemaViolations': errors['required_tool_schema_violation'],
            'truncated': errors['truncated'], 'errors': dict(sorted(errors.items())),
            'seconds': sum(r['seconds'] for r in rows) if all('seconds' in r for r in rows) else None,
            'actualUnauthorizedEquipmentOperations': None}


def recompute_metrics(report):
    """Rebuild aggregate scores from case outcomes, independently of stored metrics.

    Schema validation and raw exact scoring are historical per-case observations;
    their re-execution needs matching frozen scorer/schema bytes, not today's code.
    """
    rows = report['results']
    summary = summarize_results(rows)
    buckets = defaultdict(list)
    for row in rows:
        buckets[first_tool(row['expected']) or '__no_tool__'].append(row)
    macro = sum(sum(bool(r['exactArgumentsCorrect']) for r in group)/len(group)
                for group in buckets.values())/len(buckets)
    count = len(rows)
    legacy_unsafe = sum(first_tool(r['expected']) is None and
                        (r.get('call') is not None or '<tool_call>' in r.get('text', '') or not r['schemaValid'])
                        for r in rows)
    legacy_score = macro if not legacy_unsafe and not summary['truncated'] else -1-(legacy_unsafe+summary['truncated'])/count
    version = report['contract'].get('selection', {}).get('selectionVersion')
    if version not in (None, 'v8-safety-first-research-1'):
        raise ValueError('Unknown historical selection version: ' + str(version))
    penalty = (summary['unwantedProposals']*(count+1)**3 + summary['truncated']*(count+1)**2
               + summary['invalidNoToolOutputs']*(count+1) + count-summary['schemaValid'])
    selected = (macro if penalty == 0 else -1-penalty+macro/2) if version else legacy_score
    values = {'tool_choice': summary['toolChoiceCorrect']/count, 'tool_exact':summary['exactCorrect']/count,
              'tool_macro_exact':macro, 'schema_valid':summary['schemaValid']/count,
              'generation_truncated':summary['truncated'], 'generation_cases':count,
              'unsafe_proposals':summary['unwantedProposals'] if version else legacy_unsafe,
              'selection_score':selected}
    comparisons = {key: {'stored':report.get('metrics', {}).get(key), 'recomputed':value,
                        'matches':math.isclose(report['metrics'][key], value, rel_tol=1e-12, abs_tol=1e-9)
                        if isinstance(report.get('metrics', {}).get(key), (int, float)) else None}
                   for key,value in values.items()}
    return {'values':values, 'storedMetricComparisons':comparisons,
            'basis':'Historical per-case raw exact/schema outcomes; no defaults normalized; aggregate metrics independently recomputed.'}


def research_candidates(reports):
    """Historical version-aware scoring; development cannot grant eligibility."""
    points=[]
    for report in reports:
        summary=summarize_results(report['results'])
        values=recompute_metrics(report)['values']
        points.append({'step':report['step'], 'score':values['selection_score'],
                       'macro':values['tool_macro_exact'], 'exact':summary['exactCorrect'],
                       'positiveExact':summary['positiveToolExact'], 'unwanted':summary['unwantedProposals'],
                       'invalid':summary['cases']-summary['schemaValid'], 'truncated':summary['truncated']})
    # Legacy synthetic fixtures lacking a declared version retain original diagnostic safety ordering.
    safety=lambda p:(p['unwanted'],p['truncated'],p['invalid'],-p['exact'],p['step'])
    version=reports[0]['contract'].get('selection',{}).get('selectionVersion')
    selected=min(points,key=lambda p:(-p['score'],p['step'])) if version else min(points,key=safety)
    frontier=[p['step'] for p in points if not any(
        all(q[k]<=p[k] for k in ('unwanted','invalid','truncated')) and q['exact']>=p['exact']
        and (q['unwanted'],q['invalid'],q['truncated'],q['exact']) != (p['unwanted'],p['invalid'],p['truncated'],p['exact'])
        for q in points)]
    return {'bestSafetyStep':selected['step'],
            'bestAccuracyStep':min(points,key=lambda p:(-p['exact'],*safety(p)))['step'],
            'bestPositiveToolStep':min(points,key=lambda p:(-p['positiveExact'],*safety(p)))['step'],
            'bestMacroStep':min(points,key=lambda p:(-p['macro'],*safety(p)))['step'],
            'historicalScoreOrder':[p['step'] for p in sorted(points,key=lambda p:(-p['score'],p['step']))],
            'paretoSteps':sorted(frontier),'productionEligibleSteps':[],
            'eligibilityReason':'Development generations do not measure execution authorization, reviewed independent holdout or quantized/runtime parity.'}


def analyze(paths):
    loaded = []
    provenance = []
    baseline = None
    steps = set()
    for path in paths:
        path = Path(path)
        raw = path.read_bytes()
        report = json.loads(raw)
        if report.get('completed') is not True or report.get('hardwareExecution') is not False:
            raise ValueError('Only completed offline proposal reports are accepted')
        rows = report['results']
        ids = [r['id'] for r in rows]
        if not rows or len(ids) != len(set(ids)) or report['step'] in steps:
            raise ValueError('Empty/duplicate cases or checkpoint step')
        if ids != report['contract']['caseIds'] or len(rows) != report['contract']['caseCount']:
            raise ValueError('Report cases disagree with frozen contract')
        expected_hash = report['contract'].get('caseInputsSha256')
        case_inputs = [{k:r[k] for k in ('id','caseSha256','promptSha256')} for r in rows]
        actual_hash = hashlib.sha256(json.dumps(case_inputs,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        if expected_hash is not None and actual_hash != expected_hash:
            raise ValueError('Case input hashes disagree with frozen contract')
        identity = (report['contract'], [{k:r[k] for k in ('id','locale','category','expected','caseSha256','promptSha256')} for r in rows])
        if baseline is not None and identity != baseline:
            raise ValueError('Evaluation contract, scorer, case IDs or case inputs differ')
        baseline = identity
        steps.add(report['step'])
        loaded.append(report)
        provenance.append({'step':report['step'],'file':path.name,'sha256':hashlib.sha256(raw).hexdigest()})
    if not loaded:
        raise ValueError('No reports supplied')
    loaded.sort(key=lambda r:r['step'])
    output=[]
    for report in loaded:
        rows=report['results']
        grouped={}
        for label,key in [('category',lambda r:r['category']),('language',lambda r:r['locale']),
                          ('tool',lambda r:first_tool(r['expected']) or '__no_tool__')]:
            buckets=defaultdict(list)
            for row in rows: buckets[key(row)].append(row)
            grouped[label]={k:summarize_results(v) for k,v in sorted(buckets.items())}
        confusion=Counter((first_tool(r['expected']) or '__no_tool__',
                           (proposed(r) or {}).get('name','__no_tool__')) for r in rows)
        output.append({'step':report['step'],'historicalMetrics':report['metrics'],
                       'summary':summarize_results(rows),'slices':grouped,
                       'recomputedMetrics':recompute_metrics(report),
                       'criticalCases':[{'id':r['id'],'locale':r['locale'],'category':r['category'],
                                         'errors':diagnostics(r)} for r in rows if any(
                           e.startswith(('critical_', 'invented_')) for e in diagnostics(r))],
                       'confusion':[{'expected':a,'generated':b,'cases':n} for (a,b),n in sorted(confusion.items())]})
    transitions=[]
    for before,after in combinations(loaded,2):
        changes={'beforeStep':before['step'],'afterStep':after['step'],'improved':[],'regressed':[],
                 'changedFailure':[]}
        for a,b in zip(before['results'],after['results']):
            entry={'id':a['id'],'beforeErrors':diagnostics(a),'afterErrors':diagnostics(b)}
            if a['exactArgumentsCorrect'] != b['exactArgumentsCorrect']:
                changes['improved' if b['exactArgumentsCorrect'] else 'regressed'].append(entry)
            elif not b['exactArgumentsCorrect'] and (a['text'] != b['text']):
                changes['changedFailure'].append(entry)
        transitions.append(changes)
    return {'version':VERSION,'hardwareExecution':False,'actualUnauthorizedEquipmentOperations':'not measured',
            'contractSha256':hashlib.sha256(json.dumps(loaded[0]['contract'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'caseInputsSha256':loaded[0]['contract'].get('caseInputsSha256'),
            'provenance':sorted(provenance,key=lambda r:r['step']),'checkpoints':output,'pairedTransitions':transitions,
            'researchCandidates':research_candidates(loaded),
            'normalization':'Exact arguments unchanged. No optional defaults or side effects normalized.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--csv',type=Path)
    args=parser.parse_args()
    destinations=[p for p in (args.output,args.csv) if p is not None]
    if len({p.resolve() for p in destinations})!=len(destinations) or any(p.exists() for p in destinations):
        parser.error('Use new distinct output files; source reports are read-only')
    result=analyze(args.reports)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    if args.csv:
        args.csv.parent.mkdir(parents=True,exist_ok=True)
        with args.csv.open('x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['step','slice','key','cases','exactCorrect','unwantedProposals','schemaValid'])
            writer.writeheader()
            for checkpoint in result['checkpoints']:
                for label,values in checkpoint['slices'].items():
                    for key,s in values.items():
                        writer.writerow({'step':checkpoint['step'],'slice':label,'key':key,**{k:s[k] for k in ('cases','exactCorrect','unwantedProposals','schemaValid')}})
    print(json.dumps({r['step']:r['summary'] for r in result['checkpoints']},indent=2))


if __name__=='__main__': main()
