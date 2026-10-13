"""Author arithmetic examples, import reviewed chat data and isolate scenario groups."""
import argparse
from fractions import Fraction
import json
from pathlib import Path

from common import CONFIG, ROOT, fingerprint, policy, read_jsonl, sha256, split_records, write_json, write_jsonl

PROVENANCE = {'source': 'pins-ai authored synthetic arithmetic v1', 'license': 'MIT', 'containsPrivateData': False}


def minutes(count, seconds):
    value = Fraction(count * seconds, 60)
    return str(value.numerator) if value.denominator == 1 else f'{float(value):.2f}'.rstrip('0').rstrip('.')


def generate():
    rows = []
    for count in (4, 7, 9, 12, 16, 24, 36, 48):
        for seconds in (15, 30, 45, 60, 75, 90, 150, 240):
            identifier = f'exposure-total:{count}:{seconds}'
            # All durations/paraphrases/corrections for this exposure-count scenario
            # stay together, rather than leaking near-identical numeric variants.
            group = f'exposure-plan:{count}'
            total = minutes(count, seconds)
            for locale, prompts, answer in (
                ('en', [f'What is the total time for {count} exposures of {seconds} seconds?',
                        f'Convert {count} times {seconds} seconds of exposure into minutes.'], f'{total} minutes.'),
                ('de', [f'Wie lang sind {count} Aufnahmen mit je {seconds} Sekunden insgesamt?',
                        f'Rechne {count} mal {seconds} Sekunden Belichtung in Minuten um.'], f'{total.replace(".", ",")} Minuten.')):
                for variant, prompt in enumerate(prompts):
                    rows.append({'id': f'{identifier}:{locale}:{variant}', 'group': group, 'stratum': 'arithmetic', 'locale': locale,
                                 'messages': [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': answer}],
                                 'provenance': PROVENANCE})
            new_seconds = seconds + 30
            for locale in ('en', 'de'):
                first = f'Plan {count} exposures of {seconds} seconds.' if locale == 'en' else f'Plane {count} Aufnahmen mit {seconds} Sekunden.'
                summary = f'The draft totals {total} minutes; nothing was saved or started.' if locale == 'en' else f'Der Entwurf ergibt {total.replace(".", ",")} Minuten; nichts wurde gespeichert oder gestartet.'
                update = f'Use {new_seconds} seconds instead. What is the new total?' if locale == 'en' else f'Nimm stattdessen {new_seconds} Sekunden. Wie lang ist die Gesamtzeit?'
                answer = f'{minutes(count, new_seconds)} minutes; this remains an unsaved draft.' if locale == 'en' else f'{minutes(count, new_seconds).replace(".", ",")} Minuten; der Entwurf ist weiterhin ungespeichert.'
                # Totals and corrections with the same underlying numbers stay together.
                rows.append({'id': f'{identifier}:{locale}:context', 'group': group, 'stratum': 'arithmetic', 'locale': locale,
                             'messages': [{'role': 'user', 'content': first}, {'role': 'assistant', 'content': summary},
                                          {'role': 'user', 'content': update}, {'role': 'assistant', 'content': answer}],
                             'provenance': PROVENANCE})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/training/data')
    parser.add_argument('--extra-chat', type=Path, action='append', default=[])
    args = parser.parse_args()
    gold_path = Path(__file__).parent / 'dataset/baseline' / 'gold.jsonl'
    seed_path = Path(__file__).parent / 'dataset/baseline' / 'authored.jsonl'
    gold = read_jsonl(gold_path)
    rows = generate() + read_jsonl(seed_path)
    for path in args.extra_chat:
        rows.extend(read_jsonl(path))
    config = json.loads(CONFIG.read_text(encoding='utf-8-sig'))
    splits = split_records(rows, {row['group'] for row in gold}, config['seed'])
    gold_prompts = {fingerprint({'messages': row['messages'], 'locale': row['locale']}) for row in gold}
    if any(fingerprint({'messages': row['messages'][:-1], 'locale': row['locale']}) in gold_prompts for row in rows):
        raise ValueError('Exact gold prompt leaked into development data')
    args.output.mkdir(parents=True, exist_ok=True)
    for split, examples in splits.items():
        write_jsonl(args.output / f'{split}.jsonl', examples)
    # Use the original fixed prompts for final Pi evaluation, not training examples.
    write_json(args.output / 'gold-cases.json', [{k: row[k] for k in ('id', 'locale', 'messages')} for row in gold])
    manifest = {'seed': config['seed'], 'promptSha256': fingerprint(policy()), 'configSha256': sha256(CONFIG),
                'splitMethod': 'Deterministic 80/10/10 scenario-group assignment, stratified by arithmetic/behavior',
                'sourceHashes': {str(path): sha256(path) for path in [seed_path, gold_path, *args.extra_chat]},
                'goldSha256': sha256(gold_path), 'counts': {split: len(examples) for split, examples in splits.items()},
                'splitHashes': {split: sha256(args.output / f'{split}.jsonl') for split in splits},
                'groups': {split: sorted({r['group'] for r in examples}) for split, examples in splits.items()},
                'goldReview': 'Authored and checked against stated expected behavior; owner review still required before promotion',
                'toolData': 'Not included: real MCP schemas unavailable'}
    write_json(args.output / 'dataset-manifest.json', manifest)
    print(json.dumps(manifest['counts'], indent=2))


if __name__ == '__main__':
    main()
