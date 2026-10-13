"""Local, flushed training metrics and replay of existing Trainer histories."""
import argparse
import json
import math
from pathlib import Path

from transformers import TrainerCallback


class TrainingMetrics(TrainerCallback):
    def __init__(self, output, source='live', resume_start_step=0):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.history_path = self.output / 'metrics.jsonl'
        if self.history_path.exists():
            raise ValueError('Metrics already exist; use a new output directory')
        self.source = source
        self.resume_start_step = resume_start_step
        self.latest = {'source': source}

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        record = {'source': self.source, 'step': state.global_step,
                  'total_steps': state.max_steps, 'epoch': logs.get('epoch', state.epoch)}
        if 'eval_loss' in logs:
            event, value = 'validation', float(logs['eval_loss'])
            previous = self.latest.get('eval_loss')
            record.update(eval_loss=value, previous_eval_loss=previous,
                          eval_loss_delta=None if previous is None else value - previous)
            best = self.latest.get('best_eval_loss')
            record['best_eval_loss'] = min(best, value) if best is not None else value
        elif 'loss' in logs:
            event = 'train_window'
            record.update(train_loss=float(logs['loss']), train_loss_scope='logging_window')
        elif 'train_loss' in logs:
            event = 'train_summary'
            if self.resume_start_step:
                continued = state.global_step - self.resume_start_step
                if continued <= 0: raise ValueError('Resume summary contains no new optimizer steps')
                record.update(trainer_reported_train_loss=float(logs['train_loss']),
                    train_loss_segment=float(logs['train_loss']) * state.global_step / continued,
                    train_loss_segment_scope='resumed_segment', resumed_optimizer_steps=continued)
            else:
                record.update(train_loss_overall=float(logs['train_loss']), train_loss_overall_scope='whole_run')
        else:
            return
        for key in ('train_loss', 'train_loss_overall', 'train_loss_segment', 'eval_loss'):
            if key in record and not math.isfinite(record[key]):
                raise ValueError('Non-finite training metric: ' + key)
        record['event'] = event
        for key in ('learning_rate', 'grad_norm', 'train_runtime', 'eval_runtime'):
            if key in logs:
                record[key] = logs[key]
        for key in ('eval_tool_choice', 'eval_tool_exact', 'eval_tool_macro_exact', 'eval_schema_valid',
                    'eval_unsafe_proposals', 'eval_generation_truncated', 'eval_generation_cases', 'eval_selection_score'):
            if key in logs:
                if not math.isfinite(logs[key]):
                    raise ValueError('Non-finite generated validation metric: ' + key)
                record[key] = logs[key]
        self.latest.update(record)
        with self.history_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            stream.flush()
        temporary = self.output / 'metrics-latest.json.tmp'
        temporary.write_text(json.dumps(self.latest, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(self.output / 'metrics-latest.json')
        if event == 'validation':
            delta = record['eval_loss_delta']
            trend = 'first measurement' if delta is None else 'improving' if delta < 0 else 'worsening' if delta > 0 else 'unchanged'
            details = f"eval_loss={value:.6f} ({trend}), best_eval_loss={record['best_eval_loss']:.6f}"
            if 'eval_tool_exact' in record:
                details += (f", tool_exact={record['eval_tool_exact']:.3f}, macro_exact={record['eval_tool_macro_exact']:.3f}"
                            f", unsafe_proposals={record['eval_unsafe_proposals']}, selection_score={record['eval_selection_score']:.3f}")
        elif event == 'train_window':
            details = f"train_loss={record['train_loss']:.6f} (logging-window average)"
        elif 'train_loss_segment' in record:
            details = f"train_loss_segment={record['train_loss_segment']:.6f} (resumed-segment average)"
        else:
            details = f"train_loss_overall={record['train_loss_overall']:.6f} (whole-run average)"
        print(f"[metrics] step={record['step']}/{record['total_steps']} epoch={record['epoch']:.3f} {details}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-state', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-summary', type=Path, help='Optional completed training-run.json for whole-run averages')
    args = parser.parse_args()
    history = json.loads(args.from_state.read_text(encoding='utf-8'))
    from types import SimpleNamespace
    callback = TrainingMetrics(args.output, source='recorded_trainer_history')
    for logs in history['log_history']:
        state = SimpleNamespace(global_step=logs.get('step', history['global_step']),
                                max_steps=history['max_steps'], epoch=logs.get('epoch', 0))
        callback.on_log(None, state, None, logs)
    summary_path = args.run_summary or args.output / 'training-run.json'
    if summary_path.is_file() and 'train_loss_overall' not in callback.latest:
        summary = json.loads(summary_path.read_text(encoding='utf-8-sig'))
        if summary['status'] == 'completed' and history['global_step'] == history['max_steps']:
            state = SimpleNamespace(global_step=history['global_step'], max_steps=history['max_steps'], epoch=history['epoch'])
            callback.on_log(None, state, None, summary['metrics'])
