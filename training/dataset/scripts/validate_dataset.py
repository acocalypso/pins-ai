#!/usr/bin/env python3
"""Static integrity, MCP schema, isolation and approval checks. Does not test real hardware."""
import json,sys,re,hashlib
from pathlib import Path
from collections import Counter,defaultdict
from jsonschema import Draft202012Validator
BASE=Path(__file__).resolve().parent.parent/'data'

def main():
    catalog=json.loads((BASE/'tool_catalog.json').read_text(encoding='utf-8'))
    definitions={t['function']['name']:t for t in catalog['tools']}
    seen_ids=set();seen_user={}; seen_groups=defaultdict(set); tallies=Counter(); failures=[]
    for split in ('train','validation','test'):
        for i,line in enumerate((BASE/f'{split}.jsonl').read_text(encoding='utf-8').splitlines(),1):
            try:
                row=json.loads(line); rid=row['id']; locale=row['locale']; messages=row['messages']
                assert rid not in seen_ids, 'Duplicate id'
                seen_ids.add(rid)
                assert locale in ('en','de','fr','es','it','pt','nl','pl','ja','zh')
                assert messages[0]['role']=='system' and messages[1]['role']=='user'
                assert messages[-1]['role']=='assistant'
                declared={t['name']:t for t in row['tools']}
                calls=[]
                for n,m in enumerate(messages):
                    if m.get('tool_calls'):
                        assert m['role']=='assistant'
                        for tc in m['tool_calls']:
                            fn=tc['function']; name=fn['name']; args=fn['arguments'];calls.append(name)
                            assert name in definitions and name in declared, f'Unknown tool {name}'
                            Draft202012Validator(definitions[name]['function']['parameters']).validate(args)
                            if name=='add_target':
                                assert ('template' in args)!=('exposures' in args)
                                assert not ('template' in args and any(k in args for k in ('slew_and_center','start_guiding','autofocus_at_start','dither_every')))
                            if name=='change_filter':assert ('name' in args)!=('position' in args)
                            if name=='start_sequence':assert args.get('skip_validation',False) is False
                            if definitions[name]['safety']['requiresGatewayApproval']:
                                assert f'authorization for this turn (provided out of band; server must re-check): {name}' in messages[0]['content'], 'No trusted approval'
                            assert n+1<len(messages) and messages[n+1]['role']=='tool', 'Tool not followed by tool result'
                            assert messages[n+1]['name']==name, 'Tool result name mismatch'
                    if m['role']=='tool':
                        json.loads(m['content'])
                expected=row['expected']
                if 'first_tool' in expected:
                    assert (calls[0] if calls else None)==expected['first_tool'], 'Wrong first tool'
                    if calls:
                        assert messages[next(n for n,m in enumerate(messages) if m.get('tool_calls'))]['tool_calls'][0]['function']['arguments']==expected['arguments']
                elif 'tool_sequence' in expected:assert calls==expected['tool_sequence']
                prompt=messages[1]['content'].strip().casefold()
                assert prompt
                if (locale,prompt) in seen_user:raise AssertionError('Repeated user prompt: '+str(seen_user[(locale,prompt)]))
                seen_user[(locale,prompt)]=(split,rid)
                seen_groups[row['group']].add(split)
                tallies[(split,'count')]+=1;tallies[(split,locale)]+=1;tallies[(split,'calls')]+=len(calls)
            except Exception as e:failures.append(f'{split}:{i} - {str(e)}')
    for g,splits in seen_groups.items():
        if len(splits)>1:failures.append(f'Group leak: {g} in {sorted(splits)}')
    manifest=json.loads((BASE/'manifest.json').read_text(encoding='utf-8'))
    for split in ('train','validation','test'):
        if tallies[split,'count']!=manifest['counts'][split]:failures.append(f'Manifest row count mismatch {split}')
        p=BASE/f'{split}.jsonl';digest=hashlib.sha256(p.read_bytes()).hexdigest()
        if digest!=manifest['sha256'][p.name]:failures.append(f'SHA mismatch {p.name}')
    hard=BASE/'hard_gold.jsonl'
    if hard.exists():
        for i,line in enumerate(hard.read_text(encoding='utf-8').splitlines(),1):
            try:
                row=json.loads(line)
                assert row['id'] not in seen_ids
                assert row['locale'] in ('en','de','fr','es','it','pt','nl','pl','ja','zh')
                assert (row['locale'],row['messages'][1]['content'].strip().casefold()) not in seen_user
                assert row['messages'][0]['role']=='system' and row['messages'][1]['role']=='user'
                declared={t['name']:t for t in row['tools']}
                ex=row['expected'];name=ex['first_tool'];args=ex['arguments']
                if name:
                    assert name in declared
                    Draft202012Validator(definitions[name]['function']['parameters']).validate(args)
                    if name=='add_target': assert ('template' in args)!=('exposures' in args)
                    if ex.get('requires_gateway_approval'):
                        assert f'authorization for this turn (provided out of band; server must re-check): {name}' in row['messages'][0]['content']
                else:assert args is None
            except Exception as e:failures.append(f'hard_gold:{i} - {e}')
        if len(hard.read_text(encoding='utf-8').splitlines())!=manifest.get('hardGold'):
            failures.append('hardGold count mismatch')
        if hashlib.sha256(hard.read_bytes()).hexdigest()!=manifest.get('sha256',{}).get(hard.name):
            failures.append('hardGold SHA mismatch')
    if failures:
        print('\n'.join(failures[:75]),file=sys.stderr);print('FAILURES',len(failures),file=sys.stderr);sys.exit(1)
    print('PASS: valid JSONL; all 32 tool names/schemas; no fabricated calls; no missing approvals; all splits disjoint by scenario group; metadata hashes match.')
    for s in ('train','validation','test'):
        print(f'{s}: {tallies[s,"count"]} records, {tallies[s,"calls"]} tool calls, languages:',', '.join(f'{l}:{tallies[s,l]}' for l in ('en','de','fr','es','it','pt','nl','pl','ja','zh')))
if __name__=='__main__':main()
