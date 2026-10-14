"""Development-only generated validation; never read gold or operate equipment."""
from collections import defaultdict
import copy
import json
from pathlib import Path

from common import ROOT, fingerprint, sha256, write_json
from evaluation_state import prepare_cases, score, select_cases, scoring_hash

CRITICAL_NO_TOOL = {'no_approval', 'prompt_injection', 'no_start', 'upstream_development_denial'}


def validate_smoke(smoke, config, hardware):
    """A v6 full run must use the recipe and GPU stack actually smoke-tested."""
    if smoke.get('config') != config:
        raise ValueError('V6 GPU smoke requires the identical training configuration')
    previous = smoke.get('hardware', {})
    if any(previous.get(k) != hardware.get(k) for k in ('backend', 'python', 'packages')):
        raise ValueError('V6 GPU smoke uses a different Python/dependency/backend stack')
    for key in ('name', 'computeCapability', 'architecture', 'hip', 'cuda', 'totalMiB'):
        if previous.get('gpu', {}).get(key) != hardware.get('gpu', {}).get(key):
            raise ValueError('V6 GPU smoke uses different GPU hardware: ' + key)
    if not previous.get('ready') or not hardware.get('ready'):
        raise ValueError('V6 GPU smoke requires a successful hardware preflight')


def select_validation(rows, config):
    """Balance tools, retain every critical denial/injection, diversify locales."""
    if not 1 <= config['casesPerTool'] <= 10 or not 1 <= config['otherNoToolCasesPerCategory'] <= 10:
        raise ValueError('Generated validation sampling bounds must be 1..10')
    buckets = defaultdict(list)
    for row in rows:
        name = row['expected'].get('first_tool', (row['expected'].get('tool_sequence') or [None])[0])
        buckets[('tool', name) if name else ('no_tool', row['category'])].append(row)
    selected = []
    for index, ((kind, name), candidates) in enumerate(sorted(buckets.items())):
        candidates = sorted(candidates, key=lambda r: fingerprint(r['id']))
        if kind == 'no_tool' and name in CRITICAL_NO_TOOL:
            selected.extend(candidates)
            continue
        limit = config['casesPerTool'] if kind == 'tool' else config['otherNoToolCasesPerCategory']
        locales = sorted({r['locale'] for r in candidates})
        locale_order = locales[index % len(locales):] + locales[:index % len(locales)]
        ordered = [next(r for r in candidates if r['locale'] == locale) for locale in locale_order]
        ordered += [r for r in candidates if r not in ordered]
        selected.extend(ordered[:limit])
    return sorted(selected, key=lambda r: r['id'])


def prepare_validation(tokenizer, config, max_length):
    rows, _, inputs = select_cases('validation_smoke', prompt_mode='chat')
    cases = prepare_cases(tokenizer, select_validation(rows, config), 'chat')
    maximum = config['maxNewTokens']
    if not 1 <= maximum <= 192:
        raise ValueError('Generated validation must use the bounded plugin output budget')
    if any(len(tokenizer.encode(c['prompt'], add_special_tokens=False)) + maximum > max_length for c in cases):
        raise ValueError('Generated validation exceeds deployment context')
    contract = {'suite': 'development-validation', 'promptMode': 'chat', 'selection': config,
                'inputs': inputs, 'scoringImplementationSha256': scoring_hash(),
                'implementationSha256': sha256(ROOT / 'training/generation_validation.py'),
                'caseInputsSha256': fingerprint([{k: c[k] for k in ('id', 'caseSha256', 'promptSha256')} for c in cases]),
                'caseIds': [c['id'] for c in cases], 'caseCount': len(cases),
                'backend': 'training HF precision; NOT quantized deployment parity', 'hardwareExecution': False}
    return cases, contract


def aggregate(results):
    if not results:
        raise ValueError('Generated validation has no cases')
    buckets = defaultdict(list)
    unsafe = 0
    for r in results:
        name = r['expected'].get('first_tool', (r['expected'].get('tool_sequence') or [None])[0])
        buckets[name or '__no_tool__'].append(r)
        if name is None and (r['call'] is not None or '<tool_call>' in r['text'] or not r['schemaValid']):
            unsafe += 1
    count = len(results)
    macro = sum(sum(r['exactArgumentsCorrect'] for r in rows) / len(rows) for rows in buckets.values()) / len(buckets)
    truncated = sum(r['outputTruncated'] for r in results)
    metrics = {'tool_choice': sum(r['toolChoiceCorrect'] for r in results) / count,
               'tool_exact': sum(r['exactArgumentsCorrect'] for r in results) / count,
               'tool_macro_exact': macro,
               'schema_valid': sum(r['schemaValid'] for r in results) / count,
               'unsafe_proposals': unsafe, 'generation_truncated': truncated,
               'generation_cases': count,
               'selection_score': macro if unsafe == 0 and truncated == 0 else -1 - (unsafe + truncated) / count}
    return metrics


def generate_validation(model, tokenizer, cases, max_new_tokens):
    import torch
    device = model.get_input_embeddings().weight.device
    eos = tokenizer.convert_tokens_to_ids('<|im_end|>')
    if eos is None or eos == tokenizer.unk_token_id:
        raise ValueError('Tokenizer lacks native Qwen turn termination')
    was_training, cache = model.training, model.config.use_cache
    results = []
    try:
        model.eval()
        model.config.use_cache = True
        # Validation must not perturb training dropout or optimizer RNG streams.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))), torch.inference_mode():
            for index, case in enumerate(cases):
                batch = tokenizer(case['prompt'], return_tensors='pt', add_special_tokens=False).to(device)
                output = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False,
                                        num_beams=1, pad_token_id=tokenizer.eos_token_id,
                                        eos_token_id=eos, use_cache=True)
                tokens = output[0, batch['input_ids'].shape[1]:]
                text = tokenizer.decode(tokens, skip_special_tokens=False)
                results.append({**{k: case[k] for k in ('id', 'locale', 'category', 'expected', 'caseSha256', 'promptSha256')},
                                **score(case, text), 'text': text,
                                'outputTruncated': len(tokens) >= max_new_tokens and tokens[-1].item() != eos})
                if (index + 1) % 10 == 0 or index + 1 == len(cases):
                    print(f'[generated-validation] {index + 1}/{len(cases)}', flush=True)
    finally:
        model.config.use_cache = cache
        model.train(was_training)
    return aggregate(results), results


def make_trainer_class():
    from transformers import Trainer

    class ToolValidationTrainer(Trainer):
        def __init__(self, *args, generation_cases, generation_contract, generation_output, **kwargs):
            self.generation_cases = generation_cases
            self.generation_contract = copy.deepcopy(generation_contract)
            self.generation_output = Path(generation_output)
            super().__init__(*args, **kwargs)

        def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix='eval'):
            output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
            model = self.accelerator.unwrap_model(self.model_wrapped)
            metrics, results = generate_validation(model, self.processing_class, self.generation_cases,
                                                   self.generation_contract['selection']['maxNewTokens'])
            output.metrics.update({metric_key_prefix + '_' + k: v for k, v in metrics.items()})
            write_json(self.generation_output / f'step-{self.state.global_step:06d}.json',
                       {'contract': self.generation_contract, 'step': self.state.global_step,
                        'modelState': 'in-memory training weights at this optimizer step',
                        'metrics': metrics, 'results': results, 'completed': True, 'hardwareExecution': False})
            # These metrics reach logging, EarlyStopping and best-model selection
            # in the SAME evaluation event, rather than a late callback mutation.
            return output

    return ToolValidationTrainer
