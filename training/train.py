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
    parser.add_argument('--data', type=Path, default=ROOT / 'artifacts/training/data')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--method', choices=('lora', 'qlora'))
    parser.add_argument('--max-steps', type=int, help='Positive development smoke limit; never marked a completed specialization')
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        parser.error('--max-steps must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Use a new output directory; existing runs are never overwritten')
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    if config['enableThinking'] is not False or config['capabilityMode'] != 'informational-text-only':
        raise ValueError('Current specialization is non-thinking, chat-only; tool training awaits the actual MCP phase')
    method = args.method or config['method']
    from preflight import inspect
    hardware = inspect(method)
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
    data_info = json.loads((args.data / 'dataset-manifest.json').read_text(encoding='utf-8-sig'))
    if data_info['promptSha256'] != fingerprint(policy()):
        raise ValueError('Plugin prompt changed; prepare the dataset again')
    for split in ('train', 'validation', 'test'):
        if sha256(args.data / f'{split}.jsonl') != data_info['splitHashes'][split]:
            raise ValueError('Prepared data changed; rerun preparation')
    # A local checkpoint is mandatory. Even missing dependencies/files cannot trigger downloads.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['WANDB_DISABLED'] = 'true'
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
    set_seed(config['seed'])
    precision = config['precision']
    if precision == 'auto':
        precision = 'bf16' if torch.cuda.is_bf16_supported() else 'fp16'
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
    train_rows = [completion_example(tokenizer, row, config['maxLength']) for row in read_jsonl(args.data / 'train.jsonl')]
    validation_rows = [completion_example(tokenizer, row, config['maxLength']) for row in read_jsonl(args.data / 'validation.jsonl')]
    class Rows(torch.utils.data.Dataset):
        def __init__(self, rows): self.rows = rows
        def __len__(self): return len(self.rows)
        def __getitem__(self, index): return self.rows[index]
    def collate(rows):
        return {key: torch.tensor(value, dtype=torch.long) for key, value in pad_examples(rows, tokenizer.pad_token_id).items()}
    args.output.mkdir(parents=True)
    versions = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes')}
    run = {'status': 'started', 'config': config, 'method': method, 'precision': precision,
           'hardware': hardware, 'packages': versions, 'dataset': data_info, 'base': base_info,
           'promptSha256': fingerprint(policy()), 'templateSha256': fingerprint(tokenizer.chat_template),
           'completionOnly': True, 'enableThinking': False, 'smokeOnly': args.max_steps is not None}
    write_json(args.output / 'training-run.json', run)
    (args.output / 'pip-freeze.txt').write_text(subprocess.check_output([os.sys.executable, '-m', 'pip', 'freeze'], text=True))
    training_args = TrainingArguments(output_dir=str(args.output / 'checkpoints'),
        num_train_epochs=config['epochs'], max_steps=args.max_steps or -1,
        learning_rate=config['learningRate'], per_device_train_batch_size=config['batchSize'],
        per_device_eval_batch_size=1, gradient_accumulation_steps=config['gradientAccumulation'],
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        fp16=precision == 'fp16', bf16=precision == 'bf16', optim='adamw_torch',
        eval_strategy='epoch', save_strategy='epoch', save_total_limit=2, logging_steps=5,
        warmup_ratio=0.05, weight_decay=0.01, max_grad_norm=1.0, seed=config['seed'], data_seed=config['seed'],
        report_to=[], dataloader_num_workers=0, remove_unused_columns=False, prediction_loss_only=True,
        load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False)
    trainer = Trainer(model=model, args=training_args, train_dataset=Rows(train_rows),
                      eval_dataset=Rows(validation_rows), data_collator=collate, processing_class=tokenizer)
    try:
        result = trainer.train()
        trainer.save_model(str(args.output / 'adapter'))
        tokenizer.save_pretrained(args.output / 'adapter')
        run.update(status='completed', metrics=result.metrics,
                   maxGpuAllocatedMiB=torch.cuda.max_memory_allocated() / 1048576)
        write_json(args.output / 'training-run.json', run)
    except Exception as ex:
        run.update(status='failed', error=type(ex).__name__ + ': ' + str(ex))
        write_json(args.output / 'training-run.json', run)
        raise


if __name__ == '__main__':
    main()
