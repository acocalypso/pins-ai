"""Offline, single-GPU completion-only LoRA/QLoRA. No Pi training or cloud calls."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess

from common import CONFIG, ROOT, completion_example, fingerprint, pad_examples, policy, read_jsonl, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--dataset-mode', choices=('chat', 'mcp'), default='chat')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--method', choices=('lora', 'qlora'))
    parser.add_argument('--max-steps', type=int, help='Positive development smoke limit; never marked a completed specialization')
    parser.add_argument('--smoke-run', type=Path, help='Completed matching MCP smoke run required before full MCP training')
    parser.add_argument('--logging-steps', type=int, default=5, help='Report training loss every N optimizer steps (default: 5)')
    parser.add_argument('--resume-from-checkpoint', type=Path, help='Continue identical original training with optimizer/scheduler/scaler/RNG state')
    parser.add_argument('--resume-smoke-steps', type=int, help='Stop a diagnostic resume after N additional steps, preserving the original LR schedule')
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error('--max-steps must be positive')
    if args.logging_steps < 1:
        parser.error('--logging-steps must be positive')
    if args.resume_smoke_steps is not None and (args.resume_smoke_steps < 1 or args.resume_from_checkpoint is None):
        parser.error('--resume-smoke-steps requires a checkpoint and a positive step count')
    if args.resume_from_checkpoint and (args.max_steps is not None or args.dataset_mode != 'mcp'):
        parser.error('Checkpoint continuation uses the original MCP epoch/step budget; do not use --max-steps')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a new output directory; existing runs are never overwritten')
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    if config['enableThinking'] is not False or config['capabilityMode'] != ('mcp-tools' if args.dataset_mode == 'mcp' else 'informational-text-only'):
        raise ValueError('Training configuration capability must match dataset mode, without thinking')
    args.data = args.data or ROOT / ('artifacts/v2/sft' if args.dataset_mode == 'mcp' else 'artifacts/training/data')
    method = args.method or config['method']
    if args.dataset_mode == 'mcp' and args.max_steps is None and args.resume_smoke_steps is None:
        from v2_contracts import PACKAGE, POLICY, VERIFIED, ADAPTER
        if args.smoke_run is None:
            raise ValueError('Full MCP training requires --smoke-run pointing to a fresh verified GPU smoke')
        smoke = json.loads((args.smoke_run / 'training-run.json').read_text(encoding='utf-8'))
        if (smoke.get('status') != 'completed' or not smoke.get('smokeOnly') or
            (config['precision'] != 'auto' and smoke.get('precision') != config['precision']) or
            smoke.get('method') != method or
            smoke['dataset'].get('contractSha256') != sha256(VERIFIED) or
            smoke['dataset'].get('adapterContractSha256') != sha256(ADAPTER) or
            smoke['dataset'].get('auditSha256') != sha256(PACKAGE / 'data/contract-audit.json') or
            smoke['config']['maxLength'] < config['maxLength'] or
            any(smoke['config'][key] != config[key] for key in ('loraRank', 'loraAlpha', 'batchSize'))):
            raise ValueError('GPU smoke is stale or incompatible with full MCP training')
    from preflight import inspect
    hardware = inspect(method, config['precision'])
    if not hardware['ready']:
        raise RuntimeError(json.dumps(hardware, indent=2))
    base_info = json.loads((args.base / 'base-provenance.json').read_text(encoding='utf-8-sig'))
    if base_info['revision'] != config['revision'] or base_info['model'] != config['model'] or base_info['tokenizerOnly']:
        raise ValueError('Use the verified full pinned training checkpoint')
    if sha256(args.base / 'model.safetensors') != config['weightsSha256']:
        raise ValueError('Base weights hash mismatch')
    for filename, expected in base_info['files'].items():
        if sha256(args.base / filename) != expected:
            raise ValueError('Base file changed: ' + filename)
    if args.dataset_mode == 'mcp':
        from v2_contracts import PACKAGE, POLICY, VERIFIED, ADAPTER
        data_info = json.loads((args.data / 'manifest.json').read_text(encoding='utf-8'))
        if args.smoke_run is not None and args.max_steps is None:
            if any(smoke['dataset'].get(key) != data_info.get(key) for key in ('sourceProfile', 'curriculumSha256', 'splits')):
                raise ValueError('GPU smoke must use the identical prepared curriculum and splits')
        if data_info.get('sourceProfile'):
            from chat_curriculum import PROFILE, OUTPUT_RESERVE
            if data_info['sourceProfile'] != PROFILE or data_info.get('curriculumSha256') != sha256(ROOT / 'training/chat_curriculum.py'):
                raise ValueError('Unknown or changed curriculum; prepare again')
            if (data_info.get('chatManifestSha256') != sha256(ROOT / 'training/datasets/mcp-chat/manifest.json')
                    or data_info.get('deduplicateSupervision') is not True):
                raise ValueError('Chat curriculum changed; prepare again and run a matching smoke')
            if data_info.get('inferenceMaxOutputTokens') != OUTPUT_RESERVE:
                raise ValueError('Prepared data does not match the inference output reserve')
        from chat_dataset_common import CHAT_POLICY
        prompt_hash = fingerprint(CHAT_POLICY if data_info.get('sourceProfile') else POLICY)
        if args.smoke_run is not None and args.max_steps is None and smoke.get('promptSha256') != prompt_hash:
            raise ValueError('GPU smoke policy differs from prepared data')
        if data_info['policySha256'] != prompt_hash or data_info['contractSha256'] != sha256(VERIFIED):
            raise ValueError('MCP policy/contract changed; prepare v2 again')
        if data_info.get('adapterContractSha256') != sha256(ADAPTER):
            raise ValueError('MCP adapter behavior changed; prepare v2 again')
        for split in ('train', 'validation'):
            if sha256(args.data / f'{split}.jsonl') != data_info['splits'][split]['sha256']:
                raise ValueError('Prepared MCP split changed')
            if sha256(PACKAGE / 'data/audited' / f'{split}.jsonl') != data_info['splits'][split]['sourceSha256']:
                raise ValueError('Audited source split changed')
        if sha256(PACKAGE / 'data/contract-audit.json') != data_info['auditSha256']:
            raise ValueError('Contract audit changed after export')
    else:
        data_info = json.loads((args.data / 'dataset-manifest.json').read_text(encoding='utf-8-sig'))
        prompt_hash = fingerprint(policy())
        if data_info['promptSha256'] != prompt_hash:
            raise ValueError('Plugin prompt changed; prepare the dataset again')
        for split in ('train', 'validation', 'test'):
            if sha256(args.data / f'{split}.jsonl') != data_info['splitHashes'][split]:
                raise ValueError('Prepared data changed; rerun preparation')
    resume_info = None
    resume_state = None
    if args.resume_from_checkpoint:
        from training_resume import validate_resume
        _, resume_state, resume_info = validate_resume(args.resume_from_checkpoint, config, data_info, base_info, method)
        if args.resume_smoke_steps is None and (smoke.get('resume', {}).get('checkpointFilesSha256') != resume_info['checkpointFilesSha256']):
            raise ValueError('Full continuation requires a fresh resume smoke from identical saved state')
    # A local checkpoint is mandatory. Even missing dependencies/files cannot trigger downloads.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['WANDB_DISABLED'] = 'true'
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
    from metrics import TrainingMetrics
    set_seed(config['seed'])
    precision = config['precision']
    if precision == 'auto':
        precision = 'bf16' if torch.cuda.is_bf16_supported(including_emulation=False) else 'fp16'
    if precision not in ('bf16', 'fp16', 'fp32'):
        raise ValueError('precision must be auto/bf16/fp16/fp32')
    dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}[precision]
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    loading = {'local_files_only': True, 'trust_remote_code': False, 'torch_dtype': dtype,
               'device_map': {'': 0}, 'attn_implementation': 'eager'}
    if method == 'qlora':
        loading['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                          bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(args.base, **loading)
    model.config.use_cache = False
    if method == 'qlora':
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                               gradient_checkpointing_kwargs={'use_reentrant': False})
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=config['loraRank'], lora_alpha=config['loraAlpha'],
                           lora_dropout=config['loraDropout'], target_modules=config['targetModules'],
                           bias='none', task_type='CAUSAL_LM'))
    smoke_ids = []
    if args.dataset_mode == 'mcp':
        from v2_contracts import assistant_example
        if fingerprint(tokenizer.chat_template) != data_info['templateSha256']:
            raise ValueError('Qwen template differs from exported supervision')
        train_rows = read_jsonl(args.data / 'train.jsonl')
        validation_rows = read_jsonl(args.data / 'validation.jsonl')
        for split, prepared in (('train', train_rows), ('validation', validation_rows)):
            if data_info.get('sourceProfile'):
                from chat_curriculum import curriculum_rows
                source_rows = curriculum_rows(split)
                if fingerprint(source_rows) != data_info['splits'][split]['sourceRowsSha256']:
                    raise ValueError('Curriculum changed after preparation')
            else:
                source_rows = read_jsonl(PACKAGE / 'data/audited' / f'{split}.jsonl')
            if data_info.get('sourceProfile'):
                from chat_curriculum import convert_rows
                expected_rows = convert_rows(tokenizer, source_rows, config['maxLength'], deduplicate=split == 'train')
            else:
                expected_rows = [assistant_example(tokenizer, row, turn, config['maxLength'])
                                 for row in source_rows
                                 for turn, message in enumerate(row['messages']) if message['role'] == 'assistant']
            if prepared != expected_rows:
                raise ValueError('Prepared supervision differs from audited assistant turns; no gold substitution allowed')
        if {r['group'] for r in train_rows} & {r['group'] for r in validation_rows}:
            raise ValueError('MCP scenario leakage')
        if any(len(r['input_ids']) > config['maxLength'] for r in train_rows + validation_rows):
            raise ValueError('MCP examples exceed context; no silent truncation')
        if args.max_steps:
            selected = [next(r for r in train_rows if r['category'] == category) for category in
                        ('tool_success', 'missing_exposure', 'chat_math', 'no_approval')]
            selected += [next(r for r in train_rows if r['category'] == 'job_followup' and r['id'].endswith('/turn4'))]
            selected += [max(train_rows, key=lambda r: len(r['input_ids']))]
            selected += [next(r for r in train_rows if r['locale'] == locale) for locale in ('en', 'de')]
            train_rows = list({r['id']: r for r in selected}.values())
            smoke_ids = [r['id'] for r in train_rows]
            validation_rows = validation_rows[:19] + [max(validation_rows, key=lambda r: len(r['input_ids']))]
        train_rows = [{k: r[k] for k in ('input_ids', 'attention_mask', 'labels')} for r in train_rows]
        validation_rows = [{k: r[k] for k in ('input_ids', 'attention_mask', 'labels')} for r in validation_rows]
    else:
        train_rows = [completion_example(tokenizer, row, config['maxLength']) for row in read_jsonl(args.data / 'train.jsonl')]
        validation_rows = [completion_example(tokenizer, row, config['maxLength']) for row in read_jsonl(args.data / 'validation.jsonl')]
    class Rows(torch.utils.data.Dataset):
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, index): return self.rows[index]
    def collate(rows):
        return {key: torch.tensor(value, dtype=torch.long) for key, value in pad_examples(rows, tokenizer.pad_token_id).items()}
    args.output.mkdir(parents=True)
    package_names = ['torch', 'transformers', 'peft', 'accelerate']
    if method == 'qlora': package_names.append('bitsandbytes')
    versions = {name: importlib.metadata.version(name) for name in package_names}
    run = {'status': 'started', 'config': config, 'method': method, 'precision': precision,
           'hardware': hardware, 'packages': versions, 'dataset': data_info, 'base': base_info,
           'promptSha256': prompt_hash, 'templateSha256': fingerprint(tokenizer.chat_template),
           'completionOnly': True, 'enableThinking': False, 'smokeOnly': args.max_steps is not None or args.resume_smoke_steps is not None,
           'loggingSteps': args.logging_steps, 'datasetMode': args.dataset_mode, 'smokeExamples': smoke_ids}
    if resume_info:
        run['resume'] = resume_info
        run['resumeSmokeSteps'] = args.resume_smoke_steps
    write_json(args.output / 'training-run.json', run)
    (args.output / 'pip-freeze.txt').write_text(subprocess.check_output([os.sys.executable, '-m', 'pip', 'freeze'], text=True))
    evaluation_steps = config.get('evaluationSteps') if not args.max_steps else None
    training_args = TrainingArguments(output_dir=str(args.output / 'checkpoints'),
        num_train_epochs=config['epochs'], max_steps=args.max_steps or -1,
        learning_rate=config['learningRate'], per_device_train_batch_size=config['batchSize'],
        per_device_eval_batch_size=1, gradient_accumulation_steps=config['gradientAccumulation'],
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        fp16=precision == 'fp16', bf16=precision == 'bf16', optim='adamw_torch',
        eval_strategy='steps' if evaluation_steps else 'epoch', save_strategy='steps' if evaluation_steps else 'epoch',
        eval_steps=evaluation_steps, save_steps=evaluation_steps or 500,
        save_total_limit=2, logging_steps=args.logging_steps,
        warmup_ratio=0.05, weight_decay=0.01, max_grad_norm=1.0, seed=config['seed'], data_seed=config['seed'],
        report_to=[], dataloader_num_workers=0, remove_unused_columns=False, prediction_loss_only=True,
        restore_callback_states_from_checkpoint=bool(args.resume_from_checkpoint),
        load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False)
    metrics = TrainingMetrics(args.output, resume_start_step=resume_state['global_step'] if resume_state else 0)
    if resume_state:
        previous = [event['eval_loss'] for event in resume_state['log_history'] if 'eval_loss' in event]
        metrics.latest.update(best_eval_loss=resume_state['best_metric'], eval_loss=previous[-1] if previous else None)
    callbacks = [metrics]
    if resume_state:
        from transformers import TrainerCallback
        class ResumeBudgetCheck(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                if state.max_steps != resume_state['max_steps'] or state.global_step != resume_state['global_step']:
                    raise ValueError('Computed training step budget differs from the saved run')
        callbacks.append(ResumeBudgetCheck())
    if config.get('earlyStoppingPatience') and not args.max_steps:
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=config['earlyStoppingPatience'],
                         early_stopping_threshold=config.get('earlyStoppingThreshold', 0.0)))
    if args.resume_smoke_steps:
        from transformers import TrainerCallback
        stop_step = resume_state['global_step'] + args.resume_smoke_steps
        class ResumeSmokeStop(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step >= stop_step:
                    control.should_training_stop = True
                    control.should_save = True
                return control
        callbacks.append(ResumeSmokeStop())
    trainer = Trainer(model=model, args=training_args, train_dataset=Rows(train_rows),
                      eval_dataset=Rows(validation_rows), data_collator=collate, processing_class=tokenizer,
                      callbacks=callbacks)
    try:
        result = trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
        trainer.save_model(str(args.output / 'adapter'))
        tokenizer.save_pretrained(args.output / 'adapter')
        run.update(status='completed', metrics={**result.metrics,
                   'eval_loss': trainer.state.best_metric, 'final_eval_loss': metrics.latest.get('eval_loss')},
                   bestModelCheckpoint=trainer.state.best_model_checkpoint,
                   monitoring=metrics.latest,
                   maxGpuAllocatedMiB=torch.cuda.max_memory_allocated() / 1048576)
        if resume_info:
            run['metrics'].update(train_loss_scope='trainer_resume_global_step_denominator',
                train_loss_segment=metrics.latest.get('train_loss_segment'),
                train_runtime_scope='resumed_segment', finalGlobalStep=trainer.state.global_step,
                resumedOptimizerSteps=trainer.state.global_step - resume_info['startStep'])
        write_json(args.output / 'training-run.json', run)
    except Exception as ex:
        run.update(status='failed', error=type(ex).__name__ + ': ' + str(ex))
        write_json(args.output / 'training-run.json', run)
        raise


if __name__ == '__main__':
    main()
