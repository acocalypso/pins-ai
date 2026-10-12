"""Audit the supplied package against tools/list exported by the team's real dispatcher."""
import argparse
import copy
import json
import subprocess
import shutil
from collections import Counter
from common import ROOT, read_jsonl, sha256, write_json, write_jsonl
from v2_contracts import PACKAGE, VERIFIED, ADAPTER, POLICY, strip_descriptions, validate_call
from v2_adapter import inspect_adapter, repair_feedback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tools-list', required=True, type=__import__('pathlib').Path)
    parser.add_argument('--source', required=True, type=__import__('pathlib').Path)
    args = parser.parse_args()
    revision = subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip()
    adapter = inspect_adapter(args.source, revision)
    previous_path = PACKAGE / 'data/contract-audit.json'
    previous = json.loads(previous_path.read_text(encoding='utf-8')) if previous_path.exists() else None
    previous_tools = {t['name']: t for t in json.loads(VERIFIED.read_text(encoding='utf-8'))['tools']} if previous and VERIFIED.exists() else {}
    archive = None
    if previous and previous['sourceRevision'] != revision:
        archive = ROOT / 'artifacts/v2/contract-history' / previous['sourceRevision']
        files = [previous_path, VERIFIED, *sorted((PACKAGE / 'data/audited').glob('*.jsonl'))]
        if ADAPTER.exists(): files.append(ADAPTER)
        files += sorted((ROOT / 'artifacts/v2/sft').glob('*.json*'))
        for file in files:
            target = archive / file.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and sha256(target) != sha256(file):
                raise ValueError('Contract history already exists with different evidence: ' + str(target))
            if not target.exists(): shutil.copy2(file, target)
    manifest = json.loads((PACKAGE / 'data/manifest.json').read_text(encoding='utf-8'))
    for name, digest in manifest['sha256'].items():
        if sha256(PACKAGE / 'data' / name) != digest:
            raise ValueError('Original package hash mismatch: ' + name)
    catalog = json.loads((PACKAGE / 'data/tool_catalog.json').read_text(encoding='utf-8'))
    actual = json.loads(args.tools_list.read_text(encoding='utf-8'))
    current = {t['name']: t for t in actual['tools']}
    original = {t['function']['name']: t for t in catalog['tools']}
    audit = {'sourceRevision': revision, 'verification': 'compiled existing dispatcher tools/list; no live endpoint/hardware',
             'added': sorted(current.keys() - original.keys()), 'removed': sorted(original.keys() - current.keys()),
             'schemaChanges': sorted(name for name in current.keys() & original.keys()
                 if strip_descriptions(current[name]['inputSchema']) != strip_descriptions(original[name]['function']['parameters'])),
             'annotations': {name: tool['annotations'] for name, tool in current.items()},
             'originalHashes': manifest['sha256'], 'sourceHashes': {},
             'runtimeResults': {'envelope': 'content + optional structuredContent + isError',
                               'job': 'running/succeeded/failed/cancelled/abandoned plus cancelRequested, stopErrors and outcome.success; isError:false is insufficient'},
             'liveCompatibilityVerified': False,
             'hostWiring': adapter['hostWiring'], 'adapterContract': adapter,
             'previousAuditRevision': previous['sourceRevision'] if previous else None,
             'historyArchive': archive.relative_to(ROOT).as_posix() if archive else None,
             'changesSincePreviousAudit': {
                 'added': sorted(current.keys() - previous_tools.keys()),
                 'removed': sorted(previous_tools.keys() - current.keys()),
                 'schemaChanges': sorted(name for name in current.keys() & previous_tools.keys()
                                        if current[name]['inputSchema'] != previous_tools[name]['inputSchema']),
                 'annotationChanges': sorted(name for name in current.keys() & previous_tools.keys()
                                            if current[name]['annotations'] != previous_tools[name]['annotations']),
                 'toolsListChanged': current != previous_tools if previous else None}}
    for project in ('NINA.Mcp', 'NINA.Mcp.Adapter'):
        for path in sorted((args.source / project).rglob('*.cs')):
            if 'obj' not in path.parts:
                audit['sourceHashes'][path.relative_to(args.source).as_posix()] = sha256(path)
    write_json(VERIFIED, actual)
    write_json(ADAPTER, adapter)
    output = PACKAGE / 'data/audited'
    repairs = []
    behavioral_repairs = Counter()
    for split in ('train', 'validation', 'test'):
        rows = read_jsonl(PACKAGE / 'data' / f'{split}.jsonl')
        for row in rows:
            row['tools'] = [{'name': tool['name'], 'description': current[tool['name']]['description'],
                             'parameters': current[tool['name']]['inputSchema']} for tool in row['tools']]
            context = row['messages'][0]['content']
            suffixes = [context.find(marker) for marker in (' Trusted gateway', ' Verified MCP') if marker in context]
            row['messages'][0]['content'] = POLICY + (context[min(suffixes):] if suffixes else '')
            for index, message in enumerate(row['messages']):
                for call in message.get('tool_calls', []):
                    fn = call['function']; name = fn['name']
                    validate_call(name, fn['arguments'], [t['name'] for t in row['tools']])
                    if not current[name]['annotations']['readOnlyHint']:
                        approval = ' Trusted gateway authorization for this turn (provided out of band; server must re-check): ' + name + '.'
                        if approval not in row['messages'][0]['content']:
                            row['messages'][0]['content'] += approval
                    if name == 'stop_guiding':
                        result = json.loads(row['messages'][index + 1]['content'])
                        if 'job' in result:
                            result['job'].update(kind='guide_stop', resource=None)
                            row['messages'][index + 1]['content'] = json.dumps(result, ensure_ascii=False)
                            repairs.append(row['id'])
                if message['role'] == 'tool':
                    payload = json.loads(message['content'])
                    # Restore the actual nullable job fields omitted in the original compact fixtures.
                    if 'job' in payload:
                        payload['job'].setdefault('finishedAt', None)
                        payload['job'].setdefault('outcome', None)
                    message['content'] = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
            row['provenance'].update(contractRevision=revision, audit='source-dispatcher; simulation only')
            behavioral_repairs.update(repair_feedback(row))
        write_jsonl(output / f'{split}.jsonl', rows)
    audit['repairs'] = {'stopGuidingJobPayload': repairs, 'schemas': 'all declarations replaced with source-generated schemas',
                        'approval': 'all modifying tool fixtures have explicit simulated out-of-band approval',
                        'adapterFeedback': dict(behavioral_repairs)}
    audit['auditedHashes'] = {p.name: sha256(p) for p in output.glob('*.jsonl')}
    audit['verifiedToolsSha256'] = sha256(VERIFIED)
    audit['adapterContractSha256'] = sha256(ADAPTER)
    if previous and previous['sourceRevision'] == revision:
        # Re-running the same source audit must not erase its transition evidence.
        for field in ('previousAuditRevision', 'historyArchive', 'changesSincePreviousAudit'):
            if field in previous: audit[field] = previous[field]
    write_json(PACKAGE / 'data/contract-audit.json', audit)
    print(json.dumps({'revision': revision, 'schemaChanges': audit['schemaChanges'], 'counts': manifest['counts']}, indent=2))


if __name__ == '__main__':
    main()
