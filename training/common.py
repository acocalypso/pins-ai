"""Small offline helpers shared by preparation, training and evaluation."""
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).with_name('config.json')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def policy():
    source = (ROOT / 'src/Pins.AI.Runtime/Contracts.cs').read_text(encoding='utf-8')
    constant = source.split('public const string SystemPrompt = ', 1)[1].split(';', 1)[0]
    return ''.join(json.loads('"' + part + '"') for part in re.findall(r'"((?:[^"\\]|\\.)*)"', constant))


def system_prompt(locale):
    return policy() + (' Caller locale: ' + locale if locale else '')


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')


def validate_record(row):
    for key in ('id', 'group', 'locale', 'messages', 'provenance'):
        if not row.get(key):
            raise ValueError(f'Missing {key}')
    if row['locale'] not in ('en', 'de'):
        raise ValueError('Bootstrap corpus supports en/de only')
    provenance = row['provenance']
    if not provenance.get('source') or not provenance.get('license') or provenance.get('containsPrivateData') is not False:
        raise ValueError('Explicit source, licence and private-data declaration required')
    messages = row['messages']
    if len(messages) < 2 or messages[-1]['role'] != 'assistant':
        raise ValueError('Training example must end in a supervised assistant completion')
    if any(m.get('role') not in ('user', 'assistant') or not isinstance(m.get('content'), str)
           or not m['content'].strip() or len(m['content']) > 2000 for m in messages):
        raise ValueError('Only bounded user/assistant messages in current chat-only corpus')
    if messages[0]['role'] != 'user' or any(m['role'] == messages[i - 1]['role'] for i, m in enumerate(messages) if i):
        raise ValueError('Alternating conversation required')
    if sum(len(m['content']) for m in messages) > 6000:
        raise ValueError('History exceeds plugin budget')
    if row.get('tools') or row.get('tool_calls'):
        raise ValueError('Tool supervision requires the separate actual-schema importer')
    if '<think>' in messages[-1]['content'] or '<tool_call>' in messages[-1]['content']:
        raise ValueError('No reasoning or tool execution targets in chat-only specialization')


def split_records(rows, gold_groups, seed=3407):
    splits = {'train': [], 'validation': [], 'test': []}
    identifiers = set()
    prompts = {}
    group_strata = {}
    for row in sorted(rows, key=lambda r: r['id']):
        validate_record(row)
        if row['id'] in identifiers:
            raise ValueError('Duplicate example ID')
        identifiers.add(row['id'])
        if row['group'] in gold_groups:
            raise ValueError('Gold scenario group leaked into development corpus')
        prompt = fingerprint({'messages': row['messages'][:-1], 'locale': row['locale']})
        if prompt in prompts:
            raise ValueError('Duplicate prompt, possibly assigned to different scenario groups')
        prompts[prompt] = row['group']
        stratum = row.get('stratum', 'behavior')
        if row['group'] in group_strata and group_strata[row['group']] != stratum:
            raise ValueError('One scenario group cannot belong to multiple strata')
        group_strata[row['group']] = stratum
    assignments = {}
    for stratum in sorted(set(group_strata.values())):
        groups = sorted((group for group, value in group_strata.items() if value == stratum),
                        key=lambda group: fingerprint([seed, group]))
        train_count = min(len(groups) - 2, max(1, int(len(groups) * .8))) if len(groups) >= 3 else len(groups)
        validation_count = min(len(groups) - train_count - 1, max(1, int(len(groups) * .1))) if len(groups) >= 3 else 0
        for index, group in enumerate(groups):
            assignments[group] = 'train' if index < train_count else 'validation' if index < train_count + validation_count else 'test'
    for row in sorted(rows, key=lambda r: r['id']):
        splits[assignments[row['group']]].append(row)
    if any(not values for values in splits.values()):
        raise ValueError('Need nonempty train/validation/test groups; add scenarios, do not split translations')
    return splits


def completion_example(tokenizer, row, max_length):
    """Mask the entire prompt/history; supervise only the final assistant answer."""
    messages = [{'role': 'system', 'content': system_prompt(row['locale'])}, *row['messages'][:-1]]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
    prompt_ids = tokenizer.encode(prefix, add_special_tokens=False)
    answer_ids = tokenizer.encode(row['messages'][-1]['content'] + tokenizer.eos_token, add_special_tokens=False)
    ids = prompt_ids + answer_ids
    if len(ids) > max_length:
        raise ValueError(f"{row['id']}: {len(ids)} tokens exceeds {max_length}; never silently truncate supervision")
    return {'input_ids': ids, 'attention_mask': [1] * len(ids),
            'labels': [-100] * len(prompt_ids) + answer_ids}


def pad_examples(examples, pad_id):
    length = max(len(row['input_ids']) for row in examples)
    return {key: [row[key] + [pad_id if key == 'input_ids' else 0 if key == 'attention_mask' else -100]
                 * (length - len(row[key])) for row in examples] for key in ('input_ids', 'attention_mask', 'labels')}
