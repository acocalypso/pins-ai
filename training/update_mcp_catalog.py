"""Update active source-derived contracts without rewriting frozen source/eval splits."""
import argparse
import json
from pathlib import Path
import subprocess
from jsonschema import Draft202012Validator
from common import ROOT, sha256
from v2_adapter import inspect_adapter


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--tools-list',type=Path,required=True)
    args=parser.parse_args()
    source=args.source.resolve()
    revision=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    target=ROOT/'training/dataset/contracts'
    previous_path=target/'tools-list.json'
    previous=previous_path if previous_path.exists() else ROOT/'training/dataset/data/verified-tools-list.json'
    old={t['name']:t for t in json.loads(previous.read_text(encoding='utf-8'))['tools']}
    actual=json.loads(args.tools_list.read_text(encoding='utf-8'))
    # Re-exporting the same revision must preserve the transition evidence.
    prior_audit=target/'audit.json'
    if prior_audit.exists() and json.loads(prior_audit.read_text(encoding='utf-8'))['sourceRevision']==revision:
        if json.loads(previous.read_text(encoding='utf-8'))!=actual:
            raise ValueError('Same Git revision produced different tools; inspect dirty source before updating')
        saved=json.loads(prior_audit.read_text(encoding='utf-8'))
        if any(sha256(ROOT/p)!=digest for p,digest in saved['frozenSourceAndEvaluationHashes'].items()):
            raise ValueError('Frozen source/evaluation bytes changed')
        print(json.dumps({'sourceRevision':revision,'toolCount':len(actual['tools']),'unchanged':True}))
        return
    current={t['name']:t for t in actual['tools']}
    if len(current)!=len(actual['tools']): raise ValueError('Duplicate tools in source snapshot')
    for tool in current.values():
        Draft202012Validator.check_schema(tool['inputSchema'])
        if any(not isinstance(tool['annotations'][k],bool) for k in ('readOnlyHint','destructiveHint','idempotentHint')):
            raise ValueError('Invalid source safety annotations')
    adapter=inspect_adapter(source,revision)
    def structural(value):
        if isinstance(value,dict): return {k:structural(v) for k,v in value.items() if k not in ('description','title')}
        if isinstance(value,list): return [structural(v) for v in value]
        return value
    data=ROOT/'training/dataset/data'
    frozen={p.relative_to(ROOT).as_posix():sha256(p) for p in data.rglob('*') if p.is_file()}
    audit={'sourceRevision':revision,'verification':'compiled existing dispatcher tools/list; source adapter inspection',
        'toolCount':len(current),'hardwareExecution':False,'liveCompatibilityVerified':False,
        'added':sorted(current.keys()-old.keys()),'removed':sorted(old.keys()-current.keys()),
        'schemaChanges':sorted(n for n in current.keys()&old.keys() if structural(current[n]['inputSchema'])!=structural(old[n]['inputSchema'])),
        'descriptionChanges':sorted(n for n in current.keys()&old.keys() if current[n]['description']!=old[n]['description']),
        'frozenSourceAndEvaluationHashes':frozen,'adapterContract':adapter,
        'sourceHashes':{p.relative_to(source).as_posix():sha256(p) for area in ('NINA.Mcp','NINA.Mcp.Adapter')
                        for p in sorted((source/area).rglob('*.cs')) if 'obj' not in p.parts},
        'previousToolsSha256':sha256(previous),'exportSha256':sha256(args.tools_list)}
    target.mkdir(parents=True,exist_ok=True)
    for name,value in [('tools-list.json',actual),('adapter-contract.json',adapter),('audit.json',audit)]:
        (target/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:audit[k] for k in ('sourceRevision','toolCount','added','schemaChanges','descriptionChanges')},indent=2))


if __name__=='__main__': main()
