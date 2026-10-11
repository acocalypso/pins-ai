"""One-step CPU Trainer/PEFT integration on tiny RANDOM Qwen architecture, not PINS training."""
import math
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from common import pad_examples
from metrics import TrainingMetrics


def main():
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3Config, Qwen3ForCausalLM, Trainer, TrainingArguments, set_seed
    set_seed(3407)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
    model = Qwen3ForCausalLM(config)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=.05, bias='none',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], task_type='CAUSAL_LM'))
    rows = [{'input_ids': [1, 2, 3, 4], 'attention_mask': [1, 1, 1, 1], 'labels': [-100, -100, 3, 4]},
            {'input_ids': [1, 5, 6], 'attention_mask': [1, 1, 1], 'labels': [-100, 5, 6]}]
    class Rows(torch.utils.data.Dataset):
        def __len__(self): return len(rows)
        def __getitem__(self, index): return rows[index]
    def collate(examples):
        return {key: torch.tensor(value) for key, value in pad_examples(examples, 0).items()}
    with tempfile.TemporaryDirectory(prefix='pins-training-backend-') as directory:
        args = TrainingArguments(output_dir=directory, max_steps=1, use_cpu=True,
            per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=2,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
            optim='adamw_torch', eval_strategy='epoch', save_strategy='epoch', logging_steps=1,
            load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False,
            report_to=[], remove_unused_columns=False, prediction_loss_only=True, dataloader_num_workers=0)
        metrics = TrainingMetrics(Path(directory) / 'monitoring')
        trainer = Trainer(model=model, args=args, train_dataset=Rows(), eval_dataset=Rows(), data_collator=collate,
                          callbacks=[metrics])
        result = trainer.train()
        assert math.isfinite(result.training_loss)
        trainer.save_model(str(Path(directory) / 'adapter'))
        assert (Path(directory) / 'adapter/adapter_model.safetensors').is_file()
        assert math.isfinite(trainer.evaluate()['eval_loss'])
        history = [json.loads(line) for line in metrics.history_path.read_text().splitlines()]
        assert {'train_window', 'validation', 'train_summary'} <= {row['event'] for row in history}
        snapshot = json.loads((metrics.output / 'metrics-latest.json').read_text())
        assert math.isfinite(snapshot['train_loss']) and math.isfinite(snapshot['train_loss_overall'])
        assert snapshot['train_loss_scope'] == 'logging_window' and snapshot['train_loss_overall_scope'] == 'whole_run'
        assert math.isfinite(snapshot['eval_loss']) and snapshot['best_eval_loss'] <= snapshot['eval_loss']
        assert history[-1]['previous_eval_loss'] is not None
        print('PASS tiny random-model LoRA training, gradient accumulation, evaluation and adapter save')
        print('PASS live training/validation metric history and latest snapshot')
        print('This fixture uses random weights on CPU; it does not measure Qwen specialization quality')


if __name__ == '__main__':
    main()
