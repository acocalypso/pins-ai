"""Newly authored train-only sequence-plan examples; no held-out data is read."""
from common import write_jsonl
from v2_contracts import POLICY, definitions, validate_call
from pathlib import Path


def examples():
    catalog = definitions()
    rows = []

    def emit(locale, case, prompt, reply=None, arguments=None, history=None, tools=None):
        names = tools or ['add_target', 'get_sequence']
        if arguments is not None:
            validate_call('add_target', arguments, catalog)
        messages = [{'role': 'system', 'content': POLICY +
            ' Native modifying tool calls in this conversation are proposals only, not execution or authorization. '
            'The application must obtain exact operator approval before changing the sequence. '
            'Do not start imaging, clear existing targets or invent coordinates when asked to prepare a plan.'}]
        messages += history or []
        messages.append({'role': 'user', 'content': prompt})
        answer = {'role': 'assistant', 'content': reply or ''}
        if arguments is not None:
            answer['tool_calls'] = [{'type': 'function', 'function': {'name': 'add_target', 'arguments': arguments}}]
        messages.append(answer)
        rows.append({'id': f'sequence-plan-{case}-{locale}', 'group': f'sequence-plan-{case}',
            'locale': locale, 'category': 'sequence_plan',
            'tools': [{'name': n, 'description': catalog[n]['description'], 'parameters': catalog[n]['inputSchema']} for n in names],
            'messages': messages, 'expected': {'first_tool': 'add_target' if arguments is not None else None,
                                             'arguments': arguments},
            'provenance': {'source': 'newly authored train-only sequence planning curriculum',
                'hardwareExecution': False, 'authorization': 'proposal only; no execution permission',
                'coordinates': 'synthetic user-provided fixture values; never a target-name resolver',
                'linguisticReview': 'pending'}})

    for locale in ('en', 'de'):
        en = locale == 'en'
        missing = ('I can build a DSO container with 300 exposures of 300 seconds in the loaded sequence, without a filter change. '
                   'What are the verified J2000 RA in hours and Dec in degrees, and should I append to the loaded sequence or replace its targets? '
                   'Preparing the plan does not start imaging.' if en else
                   'Ich kann einen DSO-Container mit 300 Aufnahmen zu je 300 Sekunden ohne Filterwechsel in der geladenen Sequenz erstellen. '
                   'Welche verifizierten J2000-Koordinaten (RA in Stunden, Dec in Grad) soll ich verwenden, und soll ich die Ziele ergänzen oder ersetzen? '
                   'Das Erstellen des Plans startet keine Aufnahme.')
        request = ('create a sequence i want to shoot 300 5min exposure no filter of M31' if en else
                   'Erstelle eine Sequenz: 300 Aufnahmen von M31 mit je 5 Minuten, ohne Filter.')
        emit(locale, 'm31-missing-coordinates', request, missing)
        if not en:
            # Newly supplied user report, not a held-out evaluation case.
            mixed_request = 'Erstelle eine Sequence mit 300x 300s aufnahmen für M31'
            emit(locale, 'm31-mixed-language-missing-coordinates', mixed_request, missing)
            emit(locale, 'm31-mixed-language-followup',
                 'Zur geladenen Sequenz hinzufügen, nicht ersetzen. Verwende meine verifizierten J2000-Koordinaten: RA 0.7123 Stunden, Dec 41.2688 Grad.',
                 arguments={'name': 'M31', 'ra': 0.7123, 'dec': 41.2688,
                            'exposures': [{'exposure_seconds': 300, 'count': 300}]},
                 history=[{'role': 'user', 'content': mixed_request}, {'role': 'assistant', 'content': missing}])
        emit(locale, 'unknown-target-no-coordinate-guess',
             'Prepare 300 five-minute images of my new target without coordinates.' if en else
             'Plane 300 Fünf-Minuten-Aufnahmen meines neuen Ziels ohne Koordinaten.', missing)
        for index, (count, minutes) in enumerate(((300, 5), (120, 2), (60, 2.5), (20, 10), (240, 1))):
            # Deliberately vary fixture positions: the target name never supplies coordinates.
            target = 'M31' if index == 0 else f'Training field {index}'
            ra, dec = round(0.7123 + index * 1.25, 4), round(41.2688 - index * 3, 4)
            args = {'name': target, 'ra': ra, 'dec': dec,
                    'exposures': [{'exposure_seconds': minutes * 60, 'count': count}]}
            prompt = (f'Append {target} to the loaded sequence: {count} exposures of {minutes} minutes, no filter change. '
                      f'Use these caller-verified J2000 coordinates: RA {ra} hours, Dec {dec} degrees. Prepare only; do not start.' if en else
                      f'Ergänze {target} in der geladenen Sequenz: {count} Aufnahmen zu {minutes} Minuten ohne Filterwechsel. '
                      f'Verwende diese vom Nutzer verifizierten J2000-Koordinaten: RA {ra} Stunden, Dec {dec} Grad. Nur planen, nicht starten.')
            emit(locale, f'minutes-{index}', prompt, arguments=args)
            if index == 0:
                emit(locale, 'm31-clarification-followup',
                     f'Append, do not replace. Use verified J2000 RA {ra} hours, Dec {dec} degrees.' if en else
                     f'Ergänzen, nicht ersetzen. Verifizierte J2000-Koordinaten: RA {ra} Stunden, Dec {dec} Grad.',
                     arguments=args, history=[{'role': 'user', 'content': request}, {'role': 'assistant', 'content': missing}])
        emit(locale, 'no-blank-root-tool',
             'Create a brand-new empty sequence root and delete my current plan.' if en else
             'Erstelle eine neue leere Sequenz und lösche den aktuellen Plan.',
             'MCP has no tool to create a new empty root. I can load a saved sequence or propose clearing the current targets, but either change needs your approval. Which should I prepare?' if en else
             'MCP hat kein Werkzeug für eine neue leere Sequenzwurzel. Ich kann eine gespeicherte Sequenz laden oder das Löschen der aktuellen Ziele vorschlagen; beide Änderungen erfordern deine Freigabe. Was soll ich vorbereiten?',
             tools=['load_sequence', 'clear_targets', 'get_sequence'])
        emit(locale, 'no-filter-does-not-remove-hardware',
             'Does no filter physically remove the filter currently in my wheel?' if en else
             'Entfernt ohne Filter den aktuell eingesetzten Filter aus meinem Filterrad?',
             'No. Omitting the filter field means no filter-change instruction; it does not remove a physical filter. Use an existing clear/open wheel position explicitly if you have one.' if en else
             'Nein. Ein ausgelassenes Filterfeld bedeutet keinen Filterwechsel; ein physischer Filter wird dadurch nicht entfernt. Wenn dein Filterrad eine freie Position hat, muss diese ausdrücklich gewählt werden.')
    return rows


if __name__ == '__main__':
    destination = Path(__file__).parent / 'dataset/baseline/sequence-plans.jsonl'
    rows = examples()
    write_jsonl(destination, rows)
    print(f'Wrote {len(rows)} train-only examples to {destination}')
