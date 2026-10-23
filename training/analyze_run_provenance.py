"""Read-only stdlib run diagnostics. Missing bytes are unknown, never verified.

No deserialization, model imports, network, training or equipment access.
"""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from analyze_generation_validation import analyze

VERSION = 'run-provenance-1'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def inspect_file(path, expected=None):
    path = Path(path)
    actual = digest(path) if path.is_file() else None
    return {'path':str(path), 'expectedSha256':expected, 'actualSha256':actual,
            'status':'unknown' if actual is None or expected is None else ('match' if actual == expected else 'mismatch')}


def load_optional(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def contained(root, name):
    """Manifest paths are evidence labels, never permission to traverse elsewhere."""
    path=(root/name).resolve()
    return path if path.is_relative_to(root.resolve()) else None


def analyze_run(root, reports=None, source_state=None, restored_state=None, selected_checkpoint=None, base_dir=None, input_files=None):
    root=Path(root).resolve()
    full=root/'full' if (root/'full/training-run.json').is_file() else root
    run=load_optional(full/'training-run.json')
    restoration=load_optional(root/'resume-restoration.json')
    if restoration is None:
        restoration=load_optional(root/'resume-source/resume-restoration.json')
    resume=(run or {}).get('resume',{})
    declared=resume.get('checkpointFilesSha256',{})
    working=Path(restored_state) if restored_state else (Path(resume.get('checkpoint','__missing_checkpoint__'))/'trainer_state.json')
    # Do not follow machine-specific checkpoint paths automatically outside this run.
    if restored_state is None and not working.resolve().is_relative_to(root):
        working=root/'__unavailable_resume_checkpoint__/trainer_state.json'
    original=Path(source_state) if source_state else working.with_name('trainer_state.original.json')
    states={'sourceOriginal':inspect_file(original),
            'restoredWorking':inspect_file(working,declared.get('trainer_state.json')),
            'comparison':{'status':'unknown'}}
    source_entries=[]
    if restoration:
        for name,expected in sorted(restoration.get('sourceFilesSha256',{}).items()):
            path=contained(root/'resume-source' if (root/'resume-source').is_dir() else root,name)
            # Restoration intentionally modifies trainer_state.json; source hash applies to original copy.
            if name.endswith('/trainer_state.json') and path:
                path=path.with_name('trainer_state.original.json')
            source_entries.append(inspect_file(path,expected) if path else
                                  {'path':name,'expectedSha256':expected,'actualSha256':None,'status':'unknown'})
    if original.is_file() and working.is_file():
        a,b=load_optional(original),load_optional(working)
        keys=sorted(set(a)|set(b))
        changes=[key for key in keys if a.get(key)!=b.get(key)]
        states['comparison']={'status':'byte-identical' if digest(original)==digest(working) else 'different-bytes',
                              'changedTopLevelFields':changes,
                              'pathRelocationOnly':changes==['best_model_checkpoint'],
                              'originalBestCheckpoint':a.get('best_model_checkpoint'),
                              'workingBestCheckpoint':b.get('best_model_checkpoint'),
                              'note':'Semantic agreement or path-only relocation does not establish source-byte equality.'}
    if restoration:
        candidates=[v for k,v in restoration.get('sourceFilesSha256',{}).items()
                    if k == 'full/checkpoints/' + original.parent.name + '/trainer_state.json']
        if len(candidates)==1:
            states['sourceOriginal']=inspect_file(original,candidates[0])
    paths=list(reports) if reports is not None else sorted((full/'generation-validation').glob('step-*.json'))
    validation=analyze(paths) if paths else {'status':'unknown','reason':'No generated-validation report bytes supplied'}
    best_path=(run or {}).get('bestModelCheckpoint') or (run or {}).get('metrics',{}).get('bestModelCheckpoint')
    selected=Path(selected_checkpoint) if selected_checkpoint else None
    if selected is None and best_path:
        label=str(best_path).replace('\\','/').rstrip('/').split('/')[-1]
        matches=[full/'checkpoints'/label,root/'resume-source/full/checkpoints'/label]
        selected=next((p for p in matches if p.is_dir()),None)
    adapter=[]
    for name in ('adapter_model.safetensors','adapter_config.json'):
        exported=inspect_file(full/'adapter'/name)
        checkpoint=inspect_file(selected/name) if selected else {'status':'unknown','actualSha256':None}
        adapter.append({'file':name,'exported':exported,'selectedCheckpoint':checkpoint,
                        'status':('match' if exported['actualSha256']==checkpoint['actualSha256'] else 'mismatch')
                        if exported['actualSha256'] and checkpoint['actualSha256'] else 'unknown'})
    verified_inputs=[]
    for name,expected in sorted((run or {}).get('base',{}).get('files',{}).items()):
        path=contained(Path(base_dir),name) if base_dir else None
        verified_inputs.append(inspect_file(path,expected) if path else
                               {'path':'base/'+name,'expectedSha256':expected,'actualSha256':None,'status':'unknown'})
    for label,path in sorted((input_files or {}).items()):
        expected=(run or {}).get('dataset',{}).get(label)
        if label=='promptSha256': expected=(run or {}).get(label)
        verified_inputs.append({'label':label,**inspect_file(path,expected)})
    return {'verifiedInputFiles':verified_inputs,'version':VERSION,'runRoot':str(root),'hardwareExecution':False,'productionEligible':False,
            'trainingRun':inspect_file(full/'training-run.json'),'recordedRecipe':{k:(run or {}).get(k)
              for k in ('seed','base','config','dataset','hardware','sourceGit','codeRevision','generationValidationContract')},
            'resumeTrainerState':states,'restorationSourceFiles':source_entries,
            'restorationManifest':restoration,'adapterEquality':adapter,'validation':validation,
            'evidenceLimits':['Recorded recipe metadata is not independent verification of model/tokenizer/policy/schema bytes.',
                              'Missing reviewed holdout, native review and quantized/runtime evidence cannot pass eligibility.']}


def inspect_archive(path, result):
    """Hash archive members as bytes only; never extract or deserialize weights."""
    with zipfile.ZipFile(path) as archive:
        members={}
        restoration=result.get('restorationManifest') or {}
        source_expected=restoration.get('sourceFilesSha256',{})
        source_labels={'resume-source/'+name for name in source_expected}
        for name in sorted(archive.namelist()):
            if name in source_labels or name.endswith(('/adapter_model.safetensors','/adapter_config.json','/trainer_state.original.json','/trainer_state.json')):
                value=hashlib.sha256()
                with archive.open(name) as stream:
                    for chunk in iter(lambda:stream.read(1024*1024),b''): value.update(chunk)
                members[name]=value.hexdigest()
        exported='full/adapter/adapter_model.safetensors'
        selected=result.get('validation',{}).get('researchCandidates',{}).get('bestSafetyStep')
        candidate=f'full/research-candidates/checkpoints/checkpoint-{selected}/adapter_model.safetensors'
        source_checks=[]
        for name,expected in sorted(source_expected.items()):
            member='resume-source/'+name
            if name.startswith('full/checkpoints/') and name.endswith('/trainer_state.json'):
                member=member.replace('/trainer_state.json','/trainer_state.original.json')
            actual=members.get(member)
            source_checks.append({'member':member,'expectedSha256':expected,'actualSha256':actual,
                                  'status':'unknown' if actual is None else ('match' if actual==expected else 'mismatch')})
        result['archiveByteEvidence']={'archive':str(path),'sha256':digest(path),'memberSha256':members,'restorationSourceComparisons':source_checks,
                                      'selectedStep':selected,
                                      'exportedEqualsSelected':'unknown' if not all(n in members for n in (exported,candidate))
                                      else members[exported]==members[candidate],
                                      'method':'Stream SHA-256 bytes only; no weight loading or pickle deserialization'}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--reports',type=Path,nargs='+')
    parser.add_argument('--source-state',type=Path)
    parser.add_argument('--restored-state',type=Path)
    parser.add_argument('--selected-checkpoint',type=Path)
    parser.add_argument('--archive',type=Path,help='Optional original ZIP; stream adapter/state member hashes without extraction')
    parser.add_argument('--base-dir',type=Path,help='Model/tokenizer bytes checked against recorded base.files hashes')
    parser.add_argument('--input-file',action='append',default=[],metavar='HASH_KEY=PATH',help='Dataset hash key (or promptSha256) and corresponding file bytes')
    parser.add_argument('--output',type=Path,help='New JSON file; omitted prints deterministic JSON')
    args=parser.parse_args()
    if args.output and args.output.exists():parser.error('Never overwrite evidence; choose a new output path')
    result=analyze_run(args.run_root,args.reports,args.source_state,args.restored_state,args.selected_checkpoint,args.base_dir,dict(entry.split('=',1) for entry in args.input_file))
    if args.archive:inspect_archive(args.archive,result)
    text=json.dumps(result,sort_keys=True,ensure_ascii=False,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
    else:print(text,end='')


if __name__=='__main__':main()
