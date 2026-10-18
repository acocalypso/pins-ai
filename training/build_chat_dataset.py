"""Build and measure a train-only chat corpus; no held-out inputs or hardware calls."""
import json
from collections import Counter
from pathlib import Path

from common import ROOT, fingerprint, sha256, write_jsonl
from chat_dataset_common import CHAT_POLICY, LANGUAGES, chat_definitions
from chat_curriculum import DATA, PROFILE, has_json_request, source_rows
from v2_contracts import VERIFIED, ADAPTER, AUDIT, definitions


def source_hashes():
    from build_comprehensive_dataset import source_sha
    paths = ['training/build_chat_dataset.py', 'training/chat_dataset_common.py',
             'training/chat_curriculum.py', 'training/chat_intent_curriculum.py',
             'training/chat_workflow_curriculum.py', 'training/curriculum_v3.py',
             'training/chat_generalization_curriculum.py', 'training/upstream_tool_curriculum.py', 'training/reliability_curriculum.py',
             'training/sequence_curriculum.py', 'training/v2_contracts.py',
             'training/v2_adapter.py', 'training/common.py',
             'training/dataset/scripts/build_dataset.py',
             'plugin/src/Pins.AI.Runtime/ToolGateway.cs',
             'plugin/src/Pins.AI.Runtime/CoordinateEvidence.cs', 'plugin/src/Pins.AI.Runtime/McpChatService.cs',
             'plugin/src/Pins.AI.Runtime/ToolPromptSchema.cs',
             'plugin/src/Pins.AI.Runtime/Contracts.cs',
             'plugin/src/Pins.AI.Runtime/AiConfiguration.cs']
    hashes = {p: source_sha(ROOT / p) for p in paths}
    paths = ['training/dataset/data/audited/train.jsonl',
             'training/dataset/sources/comprehensive/train.jsonl',
             'training/dataset/sources/comprehensive/manifest.json', 'training/dataset/baseline/sequence-plans.jsonl',
             VERIFIED.relative_to(ROOT).as_posix(), ADAPTER.relative_to(ROOT).as_posix(), AUDIT.relative_to(ROOT).as_posix()]
    return {**hashes, **{p: sha256(ROOT / p) for p in paths}}


def build_rows():
    from chat_intent_curriculum import build_rows as intents
    from chat_workflow_curriculum import build_rows as workflows
    from chat_generalization_curriculum import build_rows as generalization
    legacy, exclusions = source_rows()
    from upstream_tool_curriculum import build_rows as upstream_rows
    from reliability_curriculum import build_rows as reliability_rows
    new = intents() + workflows() + generalization() + upstream_rows() + reliability_rows()
    rows = legacy + new
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate conversation identifiers')
    return rows, {'legacySelected': len(legacy), 'newAuthored': len(new), 'excludedLegacy': exclusions}


def coverage(rows):
    catalog = definitions()
    tool_report = {}
    for name in catalog:
        selected = [r for r in rows if any(c['function']['name'] == name
                    for m in r['messages'] for c in m.get('tool_calls', []))]
        natural = [r for r in selected if r['id'].startswith('chat-intent-') and not has_json_request(r)]
        arguments = {k for r in selected for m in r['messages'] for c in m.get('tool_calls', [])
                     if c['function']['name'] == name for k in c['function']['arguments']}
        boundaries = {r['group'].split('-boundary-', 1)[1] for r in rows
                      if r['group'].startswith('comprehensive-' + name + '-boundary-')}
        available = set(chat_definitions()[name]['inputSchema']['properties'])
        tool_report[name] = {'conversations': len(selected),
                             'naturalIntentLocales': sorted({r['locale'] for r in natural}),
                             'positiveArguments': sorted(arguments), 'boundaryArguments': sorted(boundaries),
                             'uncoveredArguments': sorted(available - arguments - boundaries)}
        if set(tool_report[name]['naturalIntentLocales']) != set(LANGUAGES):
            raise ValueError('Missing natural chat tool/language coverage: ' + name)
        if tool_report[name]['uncoveredArguments']:
            raise ValueError('Missing active AI argument coverage: ' + name)
    return {'conversations': len(rows), 'assistantTurns': sum(m['role'] == 'assistant' for r in rows for m in r['messages']),
            'languages': dict(Counter(r['locale'] for r in rows)),
            'categories': dict(Counter(r['category'] for r in rows)), 'tools': tool_report,
            'jsonInUserConversations': sum(bool(has_json_request(r)) for r in rows),
            'multiUserTurnConversations': sum(sum(m['role'] == 'user' for m in r['messages']) > 1 for r in rows),
            'multipleModifyingTools': sum(len({c['function']['name'] for m in r['messages'] for c in m.get('tool_calls', [])
                                             if not catalog[c['function']['name']]['annotations']['readOnlyHint']}) > 1 for r in rows),
            'maxExposedTools': max(len(r['tools']) for r in rows), 'nativeReview': 'pending'}


def main():
    rows, selection = build_rows()
    report = coverage(rows)
    DATA.mkdir(parents=True, exist_ok=True)
    write_jsonl(DATA / 'train.jsonl', rows)
    def artifact(name, content):
        (DATA / name).write_text(json.dumps(content, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    artifact('coverage.json', report)
    artifact('manifest.json', {'version': PROFILE, 'split': 'train', **selection,
                              'sourceHashes': source_hashes(), 'policySha256': fingerprint(CHAT_POLICY),
                              'trainSha256': sha256(DATA / 'train.jsonl'),
                              'coverageSha256': sha256(DATA / 'coverage.json'),
                              'hardwareExecution': False, 'heldOutInputs': [], 'linguisticReview': 'pending',
                              'conversations': len(rows), 'assistantTurns': report['assistantTurns']})
    print(json.dumps({**selection, **{k: v for k, v in report.items() if k != 'tools'}}, indent=2))


if __name__ == '__main__':
    main()
