"""Offline first-action/conditional job-poll evaluation; never executes tools or hardware."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

from common import ROOT, fingerprint, read_jsonl, sha256, write_json
from v2_contracts import PACKAGE, VERIFIED, ADAPTER, POLICY, SCORER_VERSION, compact_tools, parse_prediction, render


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=ROOT / '.local/training/base')
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--suite', choices=('eval_gold', 'hard_gold', 'validation_smoke'), default='eval_gold')
    parser.add_argument('--per-language', type=int, default=0, help='0=full set; positive selects deterministic tool/no-tool subsets')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--job-continuations', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Never overwrite previous evaluation evidence')
    import os
    os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(args.base, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.float16, attn_implementation='eager', device_map={'': 0},
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                 bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16))
    adapter_info = None
    if args.adapter:
        run_path = args.adapter.parent / 'training-run.json'
        adapter_info = json.loads(run_path.read_text(encoding='utf-8'))
        if adapter_info['status'] != 'completed':
            raise ValueError('Adapter run did not complete')
        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
    model.eval(); model.config.use_cache = True
    if args.suite == 'validation_smoke':
        path = PACKAGE / 'data/audited/validation.jsonl'
        rows = read_jsonl(path)
        rows = [{**row, 'messages': row['messages'][:2]} for row in rows]
    else:
        path = PACKAGE / 'data' / f'{args.suite}.jsonl'
        rows = read_jsonl(path)
    if args.per_language:
        selected = []
        for locale in sorted({r['locale'] for r in rows}):
            candidates = sorted((r for r in rows if r['locale'] == locale), key=lambda r: fingerprint(r['id']))
            calls = [r for r in candidates if r['expected'].get('first_tool') or r['expected'].get('tool_sequence')]
            no_calls = [r for r in candidates if not r['expected'].get('first_tool') and not r['expected'].get('tool_sequence')]
            selected += calls[:(args.per_language + 1)//2] + no_calls[:args.per_language//2]
        rows = selected
    if args.job_continuations and args.suite == 'eval_gold':
        for row in read_jsonl(PACKAGE / 'data/audited/test.jsonl'):
            if row['category'] != 'job_followup':
                continue
            if args.per_language and any(r['locale'] == row['locale'] and r.get('conditionalPoll') for r in rows):
                continue
            rows.append({**row, 'id': row['id'] + '/poll', 'messages': row['messages'][:4],
                         'expected': {'first_tool': 'get_job', 'arguments': row['expected']['arguments'][1]},
                         'conditionalPoll': True})
    report = {'suite': args.suite, 'suiteSha256': sha256(path), 'contractSha256': sha256(VERIFIED),
              'scorerVersion': SCORER_VERSION,
              'adapterContractSha256': sha256(ADAPTER),
              'sourceRevision': json.loads(ADAPTER.read_text(encoding='utf-8'))['sourceRevision'],
              'templateSha256': fingerprint(tokenizer.chat_template), 'policySha256': fingerprint(POLICY),
              'model': 'adapter' if args.adapter else 'base', 'adapter': str(args.adapter) if args.adapter else None,
              'adapterSha256': sha256(args.adapter / 'adapter_model.safetensors') if args.adapter else None,
              'baseWeightsSha256': sha256(args.base / 'model.safetensors'), 'maxNewTokens': args.max_new_tokens,
              'partial': bool(args.per_language), 'hardwareExecution': False, 'languageAndFalseClaims': 'human review pending',
              'jobEvaluation': 'conditional continuation after synthetic running-job feedback; not free-running orchestration',
              'results': [], 'perLanguage': {}}
    tallies = defaultdict(Counter)
    for row in rows:
        messages = json.loads(json.dumps(row['messages']))
        context = messages[0]['content']
        suffixes = [context.find(marker) for marker in (' Trusted gateway', ' Verified MCP') if marker in context]
        messages[0]['content'] = POLICY + (context[min(suffixes):] if suffixes else '')
        names = [t['name'] for t in row['tools']]
        prompt = render(tokenizer, messages, compact_tools(names), True)
        inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to('cuda')
        started = time.monotonic()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.convert_tokens_to_ids('<|im_end|>'))
        tokens = output[0, inputs['input_ids'].shape[1]:]
        text = tokenizer.decode(tokens, skip_special_tokens=False)
        elapsed = time.monotonic() - started
        expected = row['expected']
        expected_name = expected.get('first_tool', (expected.get('tool_sequence') or [None])[0])
        expected_args = expected.get('arguments')
        if isinstance(expected_args, list): expected_args = expected_args[0]
        error = None
        try:
            call = parse_prediction(text, names)
            valid = True
        except (ValueError, json.JSONDecodeError, __import__('jsonschema').ValidationError) as exc:
            call = None; valid = False; error = str(exc)
        name = call['name'] if call else None
        choice_ok = valid and name == expected_name
        exact_ok = choice_ok and (expected_name is None or call['arguments'] == expected_args)
        tally = tallies[row['locale']]; tally['cases'] += 1
        tally['toolChoiceCorrect'] += choice_ok; tally['exactArgumentsCorrect'] += exact_ok; tally['schemaValid'] += valid
        result = {'id': row['id'], 'locale': row['locale'], 'category': row['category'], 'expected': expected,
                  'text': text, 'call': call, 'error': error, 'schemaValid': valid, 'toolChoiceCorrect': choice_ok,
                  'exactArgumentsCorrect': exact_ok, 'seconds': elapsed, 'outputTokens': len(tokens),
                  'tokensPerSecond': len(tokens)/elapsed, 'conditionalPoll': row.get('conditionalPoll', False)}
        report['results'].append(result)
        report['perLanguage'] = {locale: dict(counts) for locale, counts in tallies.items()}
        write_json(args.output, report)
        print(json.dumps({'id': row['id'], 'toolChoiceCorrect': choice_ok, 'exactArgumentsCorrect': exact_ok,
                          'schemaValid': valid, 'seconds': round(elapsed, 2)}), flush=True)
    report['completed'] = True
    report['maxGpuAllocatedMiB'] = torch.cuda.max_memory_allocated()/1048576
    write_json(args.output, report)


if __name__ == '__main__':
    main()
