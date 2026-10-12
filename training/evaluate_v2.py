"""Offline first-action/conditional job-poll evaluation; never executes tools or hardware."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

from common import ROOT, fingerprint, sha256
from v2_contracts import VERIFIED, ADAPTER, POLICY, SCORER_VERSION
from evaluation_state import (select_cases, prepare_cases, score, scoring_hash, new_report,
                              resume_report, summarize, ProgressWriter)


def adapter_metadata(path, allow_incomplete=False):
    path = Path(path)
    run_path = path.parent / 'training-run.json'
    if not run_path.is_file() and path.parent.name == 'checkpoints':
        run_path = path.parent.parent / 'training-run.json'
    run = json.loads(run_path.read_text(encoding='utf-8'))
    diagnostic = run['status'] != 'completed'
    if diagnostic:
        if not allow_incomplete or path.parent.name != 'checkpoints':
            raise ValueError('Adapter run did not complete; only an explicitly allowed saved checkpoint may be diagnosed')
        state = json.loads((path / 'trainer_state.json').read_text(encoding='utf-8'))
        best_name = Path(state['best_model_checkpoint'].replace('\\', '/')).name
        if path.name != best_name or state.get('best_global_step') != state['global_step']:
            raise ValueError('Diagnostic checkpoint must be the metadata-selected best saved checkpoint')
    config_path = path / 'adapter_config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if config.get('peft_type') != 'LORA' or config.get('task_type') != 'CAUSAL_LM':
        raise ValueError('Unexpected adapter type')
    return run, {'weightsSha256': sha256(path / 'adapter_model.safetensors'),
                 'configSha256': sha256(config_path), 'trainingRunSha256': sha256(run_path),
                 'checkpointStateSha256': sha256(path / 'trainer_state.json') if (path / 'trainer_state.json').is_file() else None,
                 'trainingStatus': run['status'], 'diagnosticIncompleteCheckpoint': diagnostic}


def verify_base(base, config, run=None):
    info = json.loads((base / 'base-provenance.json').read_text(encoding='utf-8'))
    if (info.get('model') != config['model'] or info.get('revision') != config['revision']
            or info.get('tokenizerOnly') or info['files'].get('model.safetensors') != config['weightsSha256']):
        raise ValueError('Pinned base provenance mismatch')
    hashes = {name: sha256(base / name) for name in info['files']}
    if hashes != info['files']:
        raise ValueError('Cached base file hash mismatch')
    if run and (run['base']['model'] != info['model'] or run['base']['revision'] != info['revision']
                or run['base']['files'] != hashes):
        raise ValueError('Adapter was trained against different base files')
    return {'model': info['model'], 'revision': info['revision'], 'files': hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=ROOT / '.local/training/base')
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--suite', choices=('eval_gold', 'hard_gold', 'validation_smoke'), default='eval_gold')
    parser.add_argument('--per-language', type=int, default=0, help='0=full set; positive selects deterministic tool/no-tool subsets')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--job-continuations', action='store_true')
    parser.add_argument('--resume-from', type=Path, help='Resume a compatible report into a NEW output file')
    parser.add_argument('--backend', choices=('cuda-nf4', 'cpu-fp32'), default='cuda-nf4')
    parser.add_argument('--allow-incomplete-checkpoint', action='store_true', help='Diagnostic only: evaluate metadata-selected best saved checkpoint')
    args = parser.parse_args()
    if args.output.exists() or (args.resume_from and args.output.resolve() == args.resume_from.resolve()):
        parser.error('Never overwrite previous evaluation evidence')
    if args.max_new_tokens <= 0 or args.per_language < 0:
        parser.error('Token budget must be positive and per-language nonnegative')
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    if args.backend == 'cuda-nf4' and not torch.cuda.is_available():
        raise RuntimeError('CUDA NF4 unavailable; explicitly select --backend cpu-fp32 for a separately labeled evaluation')
    run, adapter_info = adapter_metadata(args.adapter, args.allow_incomplete_checkpoint) if args.adapter else (None, None)
    config = json.loads((ROOT / 'training/config-v3.json').read_text(encoding='utf-8'))
    base_info = verify_base(args.base, config, run)
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False)
    if run and fingerprint(tokenizer.chat_template) != run['templateSha256']:
        raise ValueError('Adapter training tokenizer template mismatch')
    rows, path, inputs_hashes = select_cases(args.suite, args.per_language, args.job_continuations)
    cases = prepare_cases(tokenizer, rows)
    eos_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    generation = {'maxNewTokens': args.max_new_tokens, 'doSample': False, 'numBeams': 1,
                  'enableThinking': False, 'padTokenId': tokenizer.eos_token_id, 'eosTokenId': eos_id,
                  'useCache': True, 'attentionImplementation': 'eager'}
    package_names = ['torch', 'transformers', 'peft', 'accelerate', 'jsonschema', 'safetensors']
    if args.backend == 'cuda-nf4': package_names.append('bitsandbytes')
    versions = {n: importlib.metadata.version(n) for n in package_names}
    identity = {'model': 'adapter' if args.adapter else 'base', 'adapter': adapter_info, 'base': base_info,
                'suite': args.suite, 'inputSha256': inputs_hashes, 'contractSha256': sha256(VERIFIED),
                'adapterContractSha256': sha256(ADAPTER), 'scorerVersion': SCORER_VERSION,
                'scoringImplementationSha256': scoring_hash(), 'policySha256': fingerprint(POLICY),
                'templateSha256': fingerprint(tokenizer.chat_template), 'generation': generation,
                'selection': {'perLanguage': args.per_language, 'jobContinuationsRequested': args.job_continuations},
                'caseInputsSha256': fingerprint([{'id': c['id'], 'caseSha256': c['caseSha256'], 'promptSha256': c['promptSha256']} for c in cases]),
                'backend': {'name': args.backend, 'dtype': 'float16' if args.backend == 'cuda-nf4' else 'float32',
                            'quantization': {'type': 'nf4', 'doubleQuant': True} if args.backend == 'cuda-nf4' else None,
                            'packages': versions}}
    report = new_report(identity, cases, {
        'suite': args.suite, 'suiteSha256': sha256(path), 'contractSha256': sha256(VERIFIED),
        'scorerVersion': SCORER_VERSION, 'adapterContractSha256': sha256(ADAPTER),
        'sourceRevision': json.loads(ADAPTER.read_text(encoding='utf-8'))['sourceRevision'],
        'templateSha256': identity['templateSha256'], 'policySha256': identity['policySha256'],
        'model': identity['model'], 'adapter': str(args.adapter) if args.adapter else None,
        'adapterSha256': adapter_info['weightsSha256'] if adapter_info else None,
        'baseWeightsSha256': base_info['files']['model.safetensors'], 'maxNewTokens': args.max_new_tokens,
        'partial': bool(args.per_language), 'hardwareExecution': False,
        'diagnosticIncompleteCheckpoint': bool(adapter_info and adapter_info['diagnosticIncompleteCheckpoint']),
        'languageAndFalseClaims': 'human review pending',
        'jobEvaluation': 'conditional synthetic job feedback' if any(c.get('conditionalPoll') for c in cases) else 'first-action only; job polling not evaluated'})
    if args.resume_from:
        report = resume_report(args.resume_from, report, cases)
    writer = ProgressWriter(args.output, args.resume_from)
    writer.write(report)
    if report['completed']:
        print('All selected cases already recorded; verified report copied to new output.')
        return
    device = 'cuda' if args.backend == 'cuda-nf4' else 'cpu'
    segment = {'platform': platform.platform(), 'python': platform.python_version(), 'backend': args.backend,
               'gpu': torch.cuda.get_device_name() if device == 'cuda' else None,
               'resumedCaseCount': len(report['results']), 'packages': versions}
    report['executionSegments'].append(segment)
    writer.write(report)
    try:
        loading = {'local_files_only': True, 'trust_remote_code': False, 'attn_implementation': 'eager',
                   'torch_dtype': torch.float16 if device == 'cuda' else torch.float32,
                   'device_map': {'': 0 if device == 'cuda' else 'cpu'}}
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            loading['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(args.base, **loading)
        if args.adapter:
            model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True, is_trainable=False)
        model.eval(); model.config.use_cache = True
        for case in cases[len(report['results']):]:
            batch = tokenizer(case['prompt'], return_tensors='pt', add_special_tokens=False).to(device)
            if device == 'cuda': torch.cuda.synchronize()
            started = time.monotonic()
            with torch.inference_mode():
                output = model.generate(**batch, max_new_tokens=args.max_new_tokens, do_sample=False,
                                        num_beams=1, pad_token_id=tokenizer.eos_token_id, eos_token_id=eos_id, use_cache=True)
            if device == 'cuda': torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            tokens = output[0, batch['input_ids'].shape[1]:]
            text = tokenizer.decode(tokens, skip_special_tokens=False)
            result = {k: case[k] for k in ('id', 'locale', 'category', 'expected', 'caseSha256', 'promptSha256')}
            result.update(score(case, text), text=text, seconds=elapsed, outputTokens=len(tokens),
                          tokensPerSecond=len(tokens)/elapsed, conditionalPoll=case.get('conditionalPoll', False),
                          outputTruncated=len(tokens) >= args.max_new_tokens and (not len(tokens) or tokens[-1].item() != eos_id),
                          executionSegment=len(report['executionSegments']) - 1)
            report['results'].append(result)
            summarize(report)
            writer.write(report)
            print(json.dumps({'id': case['id'], 'casesCompleted': report['casesCompleted'],
                              'expectedCaseCount': len(cases), 'toolChoiceCorrect': result['toolChoiceCorrect'],
                              'exactArgumentsCorrect': result['exactArgumentsCorrect'], 'schemaValid': result['schemaValid'],
                              'seconds': round(elapsed, 2)}), flush=True)
        report['maxGpuAllocatedMiB'] = torch.cuda.max_memory_allocated()/1048576 if device == 'cuda' else None
        writer.write(summarize(report))
    except BaseException as exc:
        report['status'] = 'interrupted'; report['interruption'] = type(exc).__name__ + ': ' + str(exc)
        report['completed'] = False; report['interrupted'] = True
        writer.write(report)
        raise


if __name__ == '__main__':
    main()
