"""Read-only checkpoint inference; no training, model merging, or tool execution."""
import gc
import json
from pathlib import Path
import re
import time
import warnings

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from common import fingerprint, sha256
from evaluate_v2 import verify_base


def normalize_tools(tools):
    """Accept function schemas or bare definitions, render in Qwen's native template."""
    normalized = []
    names = set()
    if tools is not None and not isinstance(tools, list):
        raise ValueError('Tools must be a JSON list')
    for tool in tools or []:
        if not isinstance(tool, dict):
            raise ValueError('Tool definition must be an object')
        function = tool.get('function') if tool.get('type') == 'function' else tool
        if not isinstance(function, dict) or not isinstance(function.get('name'), str):
            raise ValueError('Tool must have a function name and parameters JSON schema')
        if function['name'] in names:
            raise ValueError('Duplicate tool name: ' + function['name'])
        Draft202012Validator.check_schema(function['parameters'])
        names.add(function['name'])
        normalized.append({'type': 'function', 'function': function})
    return normalized


def extract_tool_calls(text, tools):
    """Strictly parse native Qwen proposals. Never repair, authorize or execute them."""
    blocks = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.S)
    if text.count('<tool_call>') != len(blocks) or text.count('</tool_call>') != len(blocks):
        raise ValueError('Malformed/truncated native tool call')
    if len(blocks) > 1:
        raise ValueError('Trained profile allows one tool proposal per iteration')
    if not blocks and re.search(r'"(?:name|function|tool_calls)"\s*:', text) and re.search(r'"(?:arguments|parameters)"\s*:', text):
        raise ValueError('Non-native tool proposal: missing Qwen tool_call markers')
    allowed = {t['function']['name']: t['function'] for t in normalize_tools(tools)}
    calls = []
    for block in blocks:
        call = json.loads(block)
        if (not isinstance(call, dict) or set(call) != {'name', 'arguments'}
                or not isinstance(call['name'], str) or call['name'] not in allowed
                or not isinstance(call['arguments'], dict)):
            raise ValueError('Unknown tool or invalid native call fields')
        Draft202012Validator(allowed[call['name']]['parameters']).validate(call['arguments'])
        calls.append(call)
    return calls


def append_tool_result(messages, response, name, result):
    """Append caller-supplied feedback; this helper does not invoke any tool."""
    calls = response['toolCalls']
    if response['toolCallError'] or len(calls) != 1 or calls[0]['name'] != name:
        raise ValueError('Feedback must match a successfully parsed proposal')
    return [*messages, {'role': 'assistant', 'content': '', 'tool_calls': [
        {'type': 'function', 'function': calls[0]}]},
        {'role': 'tool', 'name': name, 'content': result if isinstance(result, str) else json.dumps(result)}]


class LocalInference:
    def __init__(self, base, checkpoint, backend='auto'):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        from safetensors import safe_open
        self.model = None
        self.base, self.checkpoint = Path(base).resolve(), Path(checkpoint).resolve()
        if backend not in ('auto', 'cuda-nf4', 'cpu-fp32'):
            raise ValueError('Unknown inference backend')
        self.device = 'cuda' if backend == 'cuda-nf4' or (backend == 'auto' and torch.cuda.is_available()) else 'cpu'
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; use --backend cpu-fp32 explicitly')
        run_path = self.checkpoint.parent.parent / 'training-run.json'
        run = json.loads(run_path.read_text(encoding='utf-8'))
        adapter = json.loads((self.checkpoint / 'adapter_config.json').read_text(encoding='utf-8'))
        state = json.loads((self.checkpoint / 'trainer_state.json').read_text(encoding='utf-8'))
        if adapter.get('peft_type') != 'LORA' or adapter.get('task_type') != 'CAUSAL_LM':
            raise ValueError('This loader requires the detected PEFT causal-LM adapter')
        if (adapter['r'] != run['config']['loraRank'] or adapter['lora_alpha'] != run['config']['loraAlpha']
                or set(adapter['target_modules']) != set(run['config']['targetModules'])):
            raise ValueError('Adapter and training configuration disagree')
        base_info = verify_base(self.base, run['config'], run)
        tensor_count = 0
        with safe_open(self.checkpoint / 'adapter_model.safetensors', framework='pt', device='cpu') as weights:
            for name in weights.keys():
                tensor = weights.get_tensor(name)
                if not torch.isfinite(tensor).all():
                    raise ValueError('Non-finite adapter tensor: ' + name)
                tensor_count += 1
        if not tensor_count:
            raise ValueError('Empty adapter')
        # Saved tokenizer contains the training-time PAD/BOS changes. Do not replace it with defaults.
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, local_files_only=True, trust_remote_code=False)
        base_tokenizer = AutoTokenizer.from_pretrained(self.base, local_files_only=True, trust_remote_code=False)
        if self.tokenizer.get_vocab() != base_tokenizer.get_vocab():
            raise ValueError('Saved tokenizer vocabulary differs from verified base')
        if fingerprint(self.tokenizer.chat_template) != run['templateSha256']:
            raise ValueError('Saved tokenizer template differs from training')
        self.context = run['config']['maxLength']
        self.info = {'base': base_info, 'checkpoint': str(self.checkpoint), 'globalStep': state['global_step'],
            'targetSteps': state['max_steps'], 'trainingStatus': run['status'], 'method': run['method'],
            'adapterReference': adapter['base_model_name_or_path'], 'adapterTensors': tensor_count,
            'adapterSha256': sha256(self.checkpoint / 'adapter_model.safetensors'),
            'templateSha256': fingerprint(self.tokenizer.chat_template),
            'backend': 'cuda-nf4' if self.device == 'cuda' else 'cpu-fp32', 'contextLimit': self.context,
            'hardwareOperations': False}
        loading = {'local_files_only': True, 'trust_remote_code': False, 'attn_implementation': 'eager',
                   'dtype': torch.float16 if self.device == 'cuda' else torch.float32,
                   'device_map': {'': 0 if self.device == 'cuda' else 'cpu'}}
        if self.device == 'cuda':
            free, total = torch.cuda.mem_get_info()
            self.info.update(gpu=torch.cuda.get_device_name(), gpuFreeMiB=free / 1048576, gpuTotalMiB=total / 1048576)
            loading['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
            torch.cuda.reset_peak_memory_stats()
        else:
            warnings.warn('CPU FP32 uses roughly 2.3 GiB for base weights plus runtime overhead; generation can be slow.')
        started = time.monotonic()
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.base, **loading)
            self.model = PeftModel.from_pretrained(self.model, self.checkpoint, local_files_only=True,
                is_trainable=False, device_map=loading['device_map'], torch_device=self.device)
            self.model.eval()
            self.model.requires_grad_(False)
            tokens = {key: getattr(self.tokenizer, key) for key in ('bos_token_id', 'pad_token_id', 'eos_token_id')}
            size = self.model.get_input_embeddings().num_embeddings
            if tokens['eos_token_id'] is None or tokens['pad_token_id'] is None:
                raise ValueError('Saved tokenizer must declare EOS/PAD')
            if any(value is not None and not 0 <= value < size for value in tokens.values()):
                raise ValueError('Tokenizer special token outside model embeddings')
            self.info['originalModelTokens'] = {key: getattr(self.model.config, key) for key in tokens}
            for key, value in tokens.items():
                setattr(self.model.config, key, value)
                setattr(self.model.generation_config, key, value)
            self.model.config.use_cache = True
            self.info.update(specialTokens=tokens, loadSeconds=time.monotonic() - started)
        except BaseException:
            self.close()
            raise

    def generate(self, messages, tools=None, max_new_tokens=128, temperature=0.7, top_p=0.9, deterministic=True):
        import torch
        if self.model is None:
            raise RuntimeError('Inference session is closed')
        if not isinstance(messages, list) or not messages:
            raise ValueError('Supply a nonempty conversation')
        if max_new_tokens < 1 or temperature <= 0 or not 0 < top_p <= 1:
            raise ValueError('Invalid generation parameters')
        tools = normalize_tools(tools)
        prompt = self.tokenizer.apply_chat_template(messages, tools=tools or None, tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
        batch = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(self.device)
        count = batch['input_ids'].shape[1]
        if count + max_new_tokens > self.context:
            raise ValueError(f'Prompt ({count}) + generation ({max_new_tokens}) exceeds trained context {self.context}; no silent truncation')
        options = {'max_new_tokens': max_new_tokens, 'do_sample': not deterministic,
                   'num_beams': 1, 'use_cache': True, 'pad_token_id': self.tokenizer.pad_token_id,
                   'eos_token_id': self.tokenizer.eos_token_id}
        # Base generation_config enables sampling. Neutralize those inherited flags for greedy decoding.
        if deterministic:
            options.update(temperature=1.0, top_p=1.0, top_k=50)
        if not deterministic: options.update(temperature=temperature, top_p=top_p)
        if self.device == 'cuda': torch.cuda.synchronize()
        started = time.monotonic()
        with torch.inference_mode():
            output = self.model.generate(**batch, **options)
        if self.device == 'cuda': torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        ids = output[0, count:]
        ended = bool(len(ids) and ids[-1].item() == self.tokenizer.eos_token_id)
        raw = self.tokenizer.decode(ids, skip_special_tokens=False)
        # Only omit the actual terminal EOS token; never repair JSON or tool output.
        text = self.tokenizer.decode(ids[:-1] if ended else ids, skip_special_tokens=False)
        try:
            calls, error = extract_tool_calls(text, tools), None
        except (ValueError, TypeError, KeyError, ValidationError) as exc:
            calls, error = [], str(exc)
        return {'rawText': raw, 'text': text, 'toolCalls': calls, 'toolCallError': error,
                'inputTokens': count, 'outputTokens': len(ids), 'seconds': elapsed,
                'tokensPerSecond': len(ids) / elapsed, 'truncated': not ended and len(ids) >= max_new_tokens}

    def close(self):
        import torch
        self.model = None
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
