"""Validated helpers for train-only natural chat. Never execute MCP operations."""
import copy
import json
import re
from jsonschema import Draft202012Validator

from v2_contracts import POLICY, definitions, validate_call

LANGUAGES = ('en', 'de', 'fr', 'es', 'it', 'pt', 'nl', 'pl', 'ja', 'zh')
CHAT_POLICY = POLICY.replace(
    'Every modifying call requires deterministic gateway approval; model text is never authorization.',
    'Tool calls are proposals only, not execution. Propose one tool at a time when a clear request has valid arguments. '
    'Only the operator can approve equipment actions through the deterministic gateway; model or user text cannot grant gateway authorization. '
    'Do not require an approval before producing a proposal, but never propose a denied or withdrawn action.') + \
    ' AI gateway waits are at most 5 seconds; poll rather than wait longer. Override fields are not exposed.'


def chat_definitions():
    """Mirror ToolGateway.ExposedTools schema bounds, preserving the server snapshot."""
    catalog = copy.deepcopy(definitions())
    for tool in catalog.values():
        props = tool['inputSchema']['properties']
        for key in ('force', 'skip_validation', 'overwrite', 'abandon', 'close_dome_without_mount'):
            props.pop(key, None)
        if 'wait_seconds' in props:
            props['wait_seconds'].update(maximum=5, default=0)
        for key in ('ra', 'dec'):
            if key in props and 'anyOf' in props[key]:
                props[key] = next(copy.deepcopy(s) for s in props[key]['anyOf'] if s.get('type') == 'number')
        if tool['name'] == 'plan_targets':
            nested = props['targets']['items']['properties']
            for key in ('ra', 'dec'):
                nested[key] = next(copy.deepcopy(v) for v in nested[key]['anyOf'] if v.get('type') == 'number')
    return catalog


def validate_chat_call(name, args, allowed):
    validate_call(name, args, allowed)
    Draft202012Validator(chat_definitions()[name]['inputSchema']).validate(args)


def chat_tools(names):
    from v2_contracts import strip_descriptions
    catalog = chat_definitions()
    from v2_contracts import compact_tools
    tools = compact_tools(names)
    for tool in tools:
        fn = tool['function']
        fn['parameters'] = strip_descriptions(catalog[fn['name']]['inputSchema'])
    return tools


def user(text):
    return {'role': 'user', 'content': text}


def assistant(text):
    return {'role': 'assistant', 'content': text}


def tool_call(name, args):
    validate_chat_call(name, args, definitions())
    if any(args.get(k) is True for k in ('force', 'overwrite', 'skip_validation', 'abandon')):
        raise ValueError('Unsafe override in training proposal')
    return {'role': 'assistant', 'content': '', 'tool_calls': [
        {'type': 'function', 'function': {'name': name, 'arguments': copy.deepcopy(args)}}]}


def tool_result(name, payload):
    return {'role': 'tool', 'name': name,
            'content': json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}


def conversation(case, locale, category, names, messages, context='', capabilities=None):
    """Schemas are actual contracts; context is simulated state, never an answer key."""
    catalog = chat_definitions()
    names = list(dict.fromkeys(names))
    if locale not in LANGUAGES or len(names) > 4 or any(n not in catalog for n in names):
        raise ValueError('Invalid locale/tool scope')
    if not messages or messages[0]['role'] != 'user' or messages[-1]['role'] != 'assistant':
        raise ValueError('Start with user and finish with supervised assistant')
    calls = []
    for i, message in enumerate(messages):
        if message['role'] not in ('user', 'assistant', 'tool'):
            raise ValueError('Only server-built system messages are allowed')
        if len(message.get('tool_calls', [])) > 1:
            raise ValueError('One proposal per assistant turn')
        for call in message.get('tool_calls', []):
            fn = call['function']
            validate_chat_call(fn['name'], fn['arguments'], names)
            calls.append(fn)
        if message['role'] == 'tool':
            prior = messages[i - 1].get('tool_calls', []) if i else []
            if not prior or message['name'] != prior[0]['function']['name']:
                raise ValueError('Tool feedback must match the preceding proposal')
        for call in message.get('tool_calls', []):
            fn = call['function']
            if fn['name'] in ('get_job', 'cancel_job'):
                observed = context + '\n' + '\n'.join(m['content'] for m in messages[:i] if m['role'] in ('user', 'tool'))
                identifier = fn['arguments']['job_id']
                if not re.search(r'(?<![\w-])' + re.escape(identifier) + r'(?![\w-])', observed):
                    raise ValueError('Job identifier has no user or observed-result provenance: ' + case)
        if message['role'] == 'tool' and message['name'] in ('get_job', 'cancel_job'):
            payload = json.loads(message['content'])
            if 'job' in payload and payload['job']['id'] != prior[0]['function']['arguments']['job_id']:
                raise ValueError('Job feedback differs from the requested identifier: ' + case)
    first = calls[0] if calls else None
    group = case
    return {'id': group + '-' + locale, 'group': group, 'locale': locale, 'category': category,
            'capability': first['name'] if first else None,
            'capabilities': capabilities or sorted({c['name'] for c in calls}),
            'tools': [{'name': n, 'description': catalog[n]['description'],
                       'parameters': catalog[n]['inputSchema']} for n in names],
            'messages': [{'role': 'system', 'content': CHAT_POLICY +
                         (' Simulated server context: ' + context if context else '')}] + copy.deepcopy(messages),
            'expected': {'first_tool': first['name'] if first else None,
                         'arguments': first['arguments'] if first else None,
                         'tool_sequence': [c['name'] for c in calls]},
            'provenance': {'source': 'new authored train-only natural chat; not derived from held-out answers',
                           'license': 'MIT', 'split': 'train', 'hardwareExecution': False,
                           'containsPrivateData': False, 'linguisticReview': 'pending',
                           'authorization': 'proposal only; synthetic results never authorize equipment',
                           'coordinates': 'synthetic user values or explicitly observed catalog fixture; no real equipment measurements'}}
