"""Versioned diagnostic frontier; no frontier member is deployment approval."""
from analyze_generation_validation import summarize_results, research_candidates, first_tool, diagnostics

VERSION = 'v9-multi-axis-1'
CRITICAL = ('prompt_injection', 'no_approval', 'no_start', 'missing_coords',
            'missing_exposure', 'upstream_development_denial')


def frontier(reports):
    if not reports:
        raise ValueError('Research frontier requires nonempty development evidence')
    historical = research_candidates(reports)
    points = []
    for report in sorted(reports, key=lambda r: r['step']):
        rows = report['results']
        summary = summarize_results(rows)
        critical = {category: sum('unwanted_proposal' in diagnostics(r) for r in rows
                                 if r.get('category') == category) for category in CRITICAL}
        points.append({'step': report['step'], 'criticalUnwanted': critical,
                       'criticalActionReversals': summary['criticalActionReversals'],
                       'inventedCoordinates': summary['inventedCoordinates'],
                       'unwanted': summary['unwantedProposals'],
                       'invalidNoTool': summary['invalidNoToolOutputs'],
                       'invalid': summary['cases'] - summary['schemaValid'],
                       'truncated': summary['truncated'], 'exact': summary['exactCorrect'],
                       'positiveExact': summary['positiveToolExact'],
                       'positiveCases': summary['expectedTool'], 'macroExact': summary['macroExact'],
                       'generationSeconds': summary['seconds']})
    def axes(p):
        return (*p['criticalUnwanted'].values(), p['criticalActionReversals'], p['inventedCoordinates'],
                p['unwanted'], p['invalidNoTool'], p['invalid'], p['truncated'],
                -p['positiveExact'], -p['exact'], -p['macroExact'])
    pareto = [p['step'] for p in points if not any(
        all(a <= b for a, b in zip(axes(q), axes(p))) and axes(q) != axes(p) for q in points)]
    positive = min(points, key=lambda p: (-p['positiveExact'], -p['exact'], p['unwanted'], p['step']))['step']
    macro = min(points, key=lambda p: (-p['macroExact'], p['unwanted'], p['invalid'], p['step']))['step']
    critical_key = lambda p: (p['criticalActionReversals'],p['inventedCoordinates'],
                             *p['criticalUnwanted'].values(),p['unwanted'],p['invalid'],-p['positiveExact'],p['step'])
    critical = min(points,key=critical_key)['step']
    retained = {historical['bestSafetyStep'],positive,macro}
    # When macro and positive winners coincide, use the spare bounded slot for
    # a distinct critical-safety Pareto tradeoff rather than claiming two copies
    # represent every useful diagnostic candidate.
    remaining = [p for p in points if p['step'] in pareto and p['step'] not in retained]
    if len(retained)<3 and remaining: retained.add(min(remaining,key=critical_key)['step'])
    return {'version': VERSION, **historical, 'bestPolicyStep': historical['bestSafetyStep'],
            'bestPositiveStep': positive, 'bestMacroStep': macro, 'bestCriticalStep':critical,'multiAxisParetoSteps': pareto,
            'points': points, 'maximumWeightCopies': 3,
            'retainedWeightSteps': sorted(retained),
            'thirdSlotRule':'Distinct macro winner; otherwise best remaining critical-safety Pareto point, ordered reversals/coordinates/critical-slice proposals/schema/positive exact.',
            'latencyAxisIncluded': False,
            'latencyReason': 'Wall times are reported where present; comparable hardware/load provenance is not established.',
            'productionEligibleSteps': []}
