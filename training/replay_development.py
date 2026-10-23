"""Offline replay of development predictions against matching frozen cases/scorer.

Loads only local tokenizer files, never model weights. Writes a new diagnostic
report only; source reports and training/evaluation inputs stay read-only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

FIELDS=('call','schemaValid','toolChoiceCorrect','exactArgumentsCorrect')
IDENTITY=('id','locale','category','expected','caseSha256','promptSha256')


def validate_identity(report, cases, current_contract):
    contract=report['contract']
    if report.get('completed') is not True or report.get('hardwareExecution') is not False:
        raise ValueError('Only completed offline development proposal reports accepted')
    for key in ('suite','promptMode','caseIds','caseCount','caseInputsSha256','scoringImplementationSha256'):
        if contract.get(key) != current_contract.get(key):
            raise ValueError('Frozen contract mismatch: '+key)
    if contract.get('suite')!='development-validation':
        raise ValueError('Only development validation may be replayed')
    rows=report['results']
    if len(rows)!=len(cases) or len({r['id'] for r in rows})!=len(rows):
        raise ValueError('Missing or duplicate report cases')
    for saved,case in zip(rows,cases):
        for key in IDENTITY:
            if saved.get(key)!=case.get(key):
                raise ValueError('Frozen case mismatch: '+str(saved.get('id'))+' '+key)


def replay_report(report, cases, current_contract, scorer):
    """Validate every case before calling scorer; never substitute stored outcomes."""
    validate_identity(report,cases,current_contract)
    comparisons=[]
    mismatch_counts={key:0 for key in FIELDS}
    for saved,case in zip(report['results'],cases):
        rescored=scorer(case,saved['text'])
        fields={key:{'stored':saved[key],'rescored':rescored[key],'matches':saved[key]==rescored[key]}
                for key in FIELDS}
        for key in FIELDS:mismatch_counts[key]+=not fields[key]['matches']
        comparisons.append({'id':case['id'],'fields':fields})
    return {'step':report['step'],'cases':len(cases),'mismatchCounts':mismatch_counts,
            'allOutcomesMatch':not any(mismatch_counts.values()),'caseComparisons':comparisons}


def replay(paths, base):
    # Local-only tokenizer loading; no network or remote Python implementation.
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    from transformers import AutoTokenizer
    from evaluation_state import score
    from generation_validation import prepare_validation
    tokenizer=AutoTokenizer.from_pretrained(Path(base).resolve(),local_files_only=True,trust_remote_code=False)
    tokenizer.pad_token=tokenizer.eos_token
    tokenizer.padding_side='right'
    loaded=[]
    for path in paths:
        path=Path(path)
        raw=path.read_bytes()
        loaded.append((path,json.loads(raw),hashlib.sha256(raw).hexdigest()))
    if not loaded:raise ValueError('No development reports supplied')
    # Cached only by identical sampling contract; no training/holdout reader used.
    prepared={}
    checkpoints=[]
    for path,report,sha in loaded:
        selection=report['contract']['selection']
        key=json.dumps(selection,sort_keys=True)
        if key not in prepared:prepared[key]=prepare_validation(tokenizer,selection,1536)
        cases,contract=prepared[key]
        result=replay_report(report,cases,contract,score)
        checkpoints.append({'file':path.name,'sha256':sha,**result})
    return {'version':'development-raw-replay-1','hardwareExecution':False,'modelWeightsLoaded':False,
            'tokenizerLocalFilesOnly':True,'contextLimit':1536,'productionEligible':False,
            'scoringImplementationSha256':next(iter(prepared.values()))[1]['scoringImplementationSha256'],
            'allOutcomesMatch':all(c['allOutcomesMatch'] for c in checkpoints),
            'checkpoints':checkpoints,
            'scope':'Raw saved development text rescored with existing scorer after exact case/prompt/scorer identity checks. No generation, model evaluation, training or equipment execution.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--reports',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a new output file; never overwrite evidence')
    # No output is created if any contract fails closed.
    result=replay(args.reports,args.base)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'checkpoints':len(result['checkpoints']),'allOutcomesMatch':result['allOutcomesMatch']}))


if __name__=='__main__':main()
