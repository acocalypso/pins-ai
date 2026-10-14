"""Versioned runtime-aligned projection; immutable source and gold files stay intact."""
import copy
import json
import re
from collections import Counter

from common import ROOT, fingerprint, read_jsonl, sha256
from curriculum_v3 import curriculum_rows as legacy_rows
from chat_dataset_common import CHAT_POLICY, LANGUAGES, chat_definitions
from v2_contracts import POLICY, definitions

PROFILE = 'v7-upstream-mcp48-chat-proposals'
OUTPUT_RESERVE = 192
DATA = ROOT / 'training/dataset'
DENIAL = dict(zip(LANGUAGES, (
    'Do not prepare a proposal or perform this action; I have withdrawn the request.',
    'Erstelle keinen Vorschlag und führe diese Aktion nicht aus; ich ziehe die Anfrage zurück.',
    'Ne prépare aucune proposition et ne réalise pas cette action ; je retire ma demande.',
    'No prepares una propuesta ni realices esta acción; retiro mi solicitud.',
    'Non preparare una proposta né eseguire questa azione; ritiro la richiesta.',
    'Não prepares uma proposta nem executes esta ação; retiro o pedido.',
    'Maak geen voorstel en voer deze handeling niet uit; ik trek mijn verzoek in.',
    'Nie przygotowuj propozycji ani nie wykonuj tej czynności; wycofuję prośbę.',
    '提案を作成せず、この操作を実行しないでください。依頼を取り消します。',
    '不要生成提议，也不要执行此操作；我撤回请求。')))
DENIED_ACK = dict(zip(LANGUAGES, (
    'Request withdrawn. I will not propose or perform that action.',
    'Anfrage zurückgezogen. Ich werde diese Aktion weder vorschlagen noch ausführen.',
    'Demande retirée. Je ne proposerai ni ne réaliserai cette action.',
    'Solicitud retirada. No propondré ni realizaré esa acción.',
    'Richiesta ritirata. Non proporrò né eseguirò questa azione.',
    'Pedido retirado. Não vou propor nem executar essa ação.',
    'Verzoek ingetrokken. Ik zal die handeling niet voorstellen of uitvoeren.',
    'Prośba wycofana. Nie zaproponuję ani nie wykonam tej czynności.',
    '依頼は取り消されました。この操作の提案も実行もしません。',
    '请求已撤回。我不会提议或执行该操作。')))


def has_json_request(row):
    return any(m['role'] == 'user' and re.search(r'\{\s*"[^"\n]+"\s*:', m['content'])
               for m in row['messages'])


def project(row):
    """Remove approval answer keys, and make old no-approval refusals explicit denials."""
    row = copy.deepcopy(row)
    # Explicit false overrides are redundant and absent from the AI tool surface.
    def remove_false_overrides(args):
        if isinstance(args, dict):
            for key in ('force', 'skip_validation', 'overwrite', 'abandon'):
                if args.get(key) is False:
                    args.pop(key)
        elif isinstance(args, list):
            for item in args:
                remove_false_overrides(item)
    for message in row['messages']:
        for call in message.get('tool_calls', []):
            remove_false_overrides(call['function']['arguments'])
    remove_false_overrides(row['expected'].get('arguments'))
    context = row['messages'][0]['content'].removeprefix(POLICY)
    patterns = (
        r' Trusted gateway authorization for this turn \(provided out of band; server must re-check\): [^.]+\.',
        r' Synthetic exact application approval for .*?; gateway must independently re-check\.',
        r' Synthetic fixture: application approval for this exact proposal exists; the gateway still re-checks it\.',
        r' Synthetic history fixture: the earlier stop_all had application approval, now consumed\. No new action is authorized\.',
    )
    for pattern in patterns:
        context = re.sub(pattern, '', context)
    context = context.replace('Historical cancellation approved and consumed', 'Historical cancellation returned a tool result')
    context = context.replace('exact request authorized;', 'no automatic retries;')
    context = context.replace('exact action authorized;', 'no automatic retries;')
    row['messages'][0]['content'] = CHAT_POLICY + context
    if row['category'] in ('no_approval', 'comprehensive_approval', 'comprehensive_revoked', 'safety_explicit_denial'):
        message = next(m for m in reversed(row['messages']) if m['role'] == 'user')
        message['content'] += '\n' + DENIAL[row['locale']]
        row['messages'][-1]['content'] = DENIED_ACK[row['locale']]
    # Preserve called tools and original semantic competitors, bounded to the runtime scope.
    called = list(dict.fromkeys(c['function']['name'] for m in row['messages'] for c in m.get('tool_calls', [])))
    order = called + [t['name'] for t in row['tools'] if t['name'] not in called]
    limit = 1 if 'crossfield-too-many-blocks' in row['group'] else 2
    selected = set(order[:limit])
    if not set(called) <= selected:
        raise ValueError('Conversation needs a larger tool scope; author bounded phases')
    catalog = chat_definitions()
    row['tools'] = [{'name': t['name'], 'description': catalog[t['name']]['description'],
                     'parameters': catalog[t['name']]['inputSchema']} for t in row['tools'] if t['name'] in selected]
    row['provenance'] = {**row['provenance'], 'activeProjection': PROFILE,
                         'authorization': 'proposal only; no execution permission', 'split': row['provenance'].get('split', 'train')}
    return row


def source_rows():
    """Keep natural legacy examples and bounded schema/job fixtures, not bulk JSON positives."""
    rows = []
    rejected = Counter()
    for row in legacy_rows('train'):
        category = row['category']
        if any('unsupported-' + k in row['group'] for k in ('dome', 'flat', 'rotator', 'solve', 'catalog', 'sync')):
            rejected['superseded_upstream_capability'] += 1
            continue
        if any(c['function']['arguments'].get('wait_seconds', 0) > 5
               for m in row['messages'] for c in m.get('tool_calls', [])):
            rejected['plugin_wait_limit'] += 1
            continue
        if category in ('safety_mcp_text_not_authorization', 'safety_role_spoof'):
            rejected['obsolete_approval_stage'] += 1
            continue
        if has_json_request(row):
            keep = (category in ('comprehensive_poll', 'comprehensive_error') or
                    (row['locale'] in ('en', 'de') and category in
                     ('comprehensive_boundary', 'comprehensive_missing', 'comprehensive_crossfield')))
            if not keep:
                rejected['json_argument_copying'] += 1
                continue
        if category == 'device_contrast' and not any(k in row['group'] for k in ('wording0', 'correction')):
            rejected['repeated_device_phrasing'] += 1
            continue
        rows.append(project(row))
    return rows, dict(rejected)


def curriculum_rows(split):
    if split == 'validation':
        rows = [project(r) for r in legacy_rows(split)]
        for row in rows:
            row['provenance']['split'] = split
        from upstream_tool_curriculum import validation_rows
        return rows + validation_rows()
    if split != 'train':
        raise ValueError('Only train/validation are preparation inputs; gold remains held out')
    manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))
    from build_chat_dataset import source_hashes
    if (manifest['sourceHashes'] != source_hashes() or manifest['trainSha256'] != sha256(DATA / 'train.jsonl')
            or manifest['policySha256'] != fingerprint(CHAT_POLICY)):
        raise ValueError('Chat dataset is stale; run training/build_chat_dataset.py')
    return read_jsonl(DATA / 'train.jsonl')


def convert_rows(tokenizer, rows, max_tokens, deduplicate=False):
    """Deduplicate identical rendered supervision, never discard distinct outcomes."""
    from v2_contracts import assistant_example
    result, seen = [], {}
    for row in rows:
        for turn, message in enumerate(row['messages']):
            if message['role'] != 'assistant':
                continue
            item = assistant_example(tokenizer, row, turn, max_tokens, tool_mode='chat')
            prompt_tokens = sum(v == -100 for v in item['labels'])
            if max_tokens and prompt_tokens + OUTPUT_RESERVE > max_tokens:
                raise ValueError('Prompt leaves too little output room; author a shorter phase: ' + item['id'])
            if len(item['input_ids']) - prompt_tokens > OUTPUT_RESERVE:
                raise ValueError('Completion exceeds the deployment output budget: ' + item['id'])
            if deduplicate:
                key = fingerprint(item['prompt'])
                target = fingerprint(item['completion'])
                if key in seen:
                    if seen[key] != target:
                        raise ValueError('Conflicting supervision for an identical rendered prompt: ' + item['id'])
                    continue
                seen[key] = target
            result.append(item)
    return result
