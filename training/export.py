"""Merge into an unquantized base, then convert/quantize with pinned llama.cpp."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from common import CONFIG, ROOT, fingerprint, policy, sha256, write_json


def export_adapter(run_dir=None, checkpoint=None, allow_smoke=False):
    """Saved checkpoints are explicit diagnostic inputs, never completed runs."""
    if checkpoint is not None:
        from evaluate_v2 import adapter_metadata
        run, metadata = adapter_metadata(checkpoint, allow_incomplete=True)
        if checkpoint.parent.name != 'checkpoints':
            raise ValueError('Checkpoint export requires a saved checkpoints directory')
        state = json.loads((checkpoint / 'trainer_state.json').read_text(encoding='utf-8'))
        best_name = Path(state['best_model_checkpoint'].replace('\\', '/')).name
        if best_name != checkpoint.name or state.get('best_global_step') != state['global_step']:
            raise ValueError('Diagnostic checkpoint must be the metadata-selected best saved checkpoint')
        if not 0 < state['global_step'] <= state['max_steps']:
            raise ValueError('Invalid checkpoint step metadata')
        metadata.update(globalStep=state['global_step'], targetSteps=state['max_steps'])
        return run, checkpoint, checkpoint.parent.parent, metadata
    if run_dir is None:
        return None, None, None, None
    run = json.loads((run_dir / 'training-run.json').read_text(encoding='utf-8-sig'))
    if run['status'] != 'completed' or (run['smokeOnly'] and not allow_smoke):
        raise ValueError('Require a completed training run; smoke exports need explicit opt-in')
    from evaluate_v2 import adapter_metadata
    _, metadata = adapter_metadata(run_dir / 'adapter')
    return run, run_dir / 'adapter', run_dir, metadata


def commands(converter_python, source, merged, output, stem='pins-qwen3-0.6b'):
    binary = source / 'build/bin' / ('llama-quantize.exe' if os.name == 'nt' else 'llama-quantize')
    if os.name == 'nt' and not binary.exists():
        binary = source / 'build/bin/Release/llama-quantize.exe'
    return [[str(converter_python), str(source / 'convert_hf_to_gguf.py'), str(merged),
             '--outfile', str(output / (stem + '-f16.gguf')), '--outtype', 'f16'],
            [str(binary), str(output / (stem + '-f16.gguf')),
             str(output / (stem + '-Q4_K_M.gguf')), 'Q4_K_M']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run', type=Path)
    mode.add_argument('--checkpoint', type=Path, help='Diagnostic export of the metadata-selected saved checkpoint; never production-ready')
    mode.add_argument('--base-only', action='store_true', help='Untuned control from the identical pinned HF checkpoint')
    parser.add_argument('--llama-source', type=Path, required=True)
    parser.add_argument('--converter-python', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-smoke-export', action='store_true', help='Development only; never eligible for promotion')
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding='utf-8-sig'))
    run, adapter_path, run_dir, adapter_info = export_adapter(args.run, args.checkpoint, args.allow_smoke_export)
    base_info = json.loads((args.base / 'base-provenance.json').read_text(encoding='utf-8-sig'))
    if base_info['revision'] != config['revision'] or base_info['model'] != config['model'] or base_info['tokenizerOnly']:
        raise ValueError('Use the verified full pinned training checkpoint')
    if (run and run['config']['revision'] != config['revision']) or sha256(args.base / 'model.safetensors') != config['weightsSha256']:
        raise ValueError('Wrong unquantized pinned base checkpoint')
    for filename, expected_hash in (run['base'] if run else base_info)['files'].items():
        if sha256(args.base / filename) != expected_hash:
            raise ValueError('Original base file changed: ' + filename)
    mcp_mode = run is not None and run.get('datasetMode') == 'mcp'
    if mcp_mode:
        from v2_contracts import POLICY, VERIFIED, ADAPTER
        prompt_hash = fingerprint(POLICY)
        if run['dataset']['contractSha256'] != sha256(VERIFIED):
            raise ValueError('MCP contract changed since training; re-audit before export')
        if run['dataset'].get('adapterContractSha256') != sha256(ADAPTER):
            raise ValueError('MCP adapter behavior changed since training; re-audit before export')
    else:
        prompt_hash = fingerprint(policy())
    if run and run['promptSha256'] != prompt_hash:
        raise ValueError('Plugin policy changed since training; retrain against the intended current policy')
    expected = json.loads((ROOT / 'plugin/models/manifest.json').read_text(encoding='utf-8-sig'))['runtime']['commit']
    actual = subprocess.check_output(['git', '-C', str(args.llama_source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != expected:
        raise ValueError('llama.cpp converter/quantizer must match the pinned runtime commit')
    stem = 'qwen3-0.6b-training-control' if args.base_only else 'pins-qwen3-0.6b-mcp-v2' if mcp_mode else 'pins-qwen3-0.6b'
    if args.checkpoint:
        stem = 'pins-qwen3-0.6b-' + args.checkpoint.name + '-diagnostic'
    converter_commands = commands(args.converter_python, args.llama_source, args.output / 'merged', args.output, stem)
    if not args.converter_python.is_file() or not Path(converter_commands[1][0]).is_file():
        raise ValueError('Install the separate converter environment and build llama-quantize first')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use a new output directory; never overwrite a base or previous export')
    # Check converter dependencies before an expensive CPU merge.
    subprocess.run([str(args.converter_python), '-c',
                    'import sys; sys.path.insert(0, sys.argv[1]); import torch, transformers, gguf, sentencepiece',
                    str(args.llama_source / 'gguf-py')], check=True)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if adapter_path:
        from safetensors import safe_open
        with safe_open(adapter_path / 'adapter_model.safetensors', framework='pt', device='cpu') as tensors:
            if not tensors.keys():
                raise ValueError('Empty adapter')
            for name in tensors.keys():
                if not torch.isfinite(tensors.get_tensor(name)).all():
                    raise ValueError('Nonfinite adapter tensor: ' + name)
    # Merge on CPU into clean FP32 weights, never merge into a 4-bit training model.
    base = AutoModelForCausalLM.from_pretrained(args.base, local_files_only=True, trust_remote_code=False,
                                               torch_dtype=torch.float32, device_map={'': 'cpu'})
    model = PeftModel.from_pretrained(base, adapter_path, local_files_only=True).merge_and_unload(safe_merge=True) if run else base
    tokenizer = AutoTokenizer.from_pretrained(adapter_path if run else args.base, local_files_only=True, trust_remote_code=False)
    if run and tokenizer.get_vocab() != AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False).get_vocab():
        raise ValueError('Adapter tokenizer vocabulary differs from the pinned base')
    if run and fingerprint(tokenizer.chat_template) != run['templateSha256']:
        raise ValueError('Adapter tokenizer template differs from the recorded training template')
    merged = args.output / 'merged'
    model.save_pretrained(merged, safe_serialization=True)
    tokenizer.save_pretrained(merged)
    shutil.copy2(args.base / 'LICENSE', merged / 'LICENSE')
    del model, base
    for command in converter_commands:
        subprocess.run(command, check=True)
    target = args.output / (stem + '-Q4_K_M.gguf')
    model_info = {'id': 'qwen3-0.6b-training-control-q4' if args.base_only else 'pins-qwen3-0.6b-mcp-v2-q4' if mcp_mode else 'pins-qwen3-0.6b-specialized-q4',
                  'profile': 'lite', 'file': target.name,
                  'bytes': target.stat().st_size, 'sha256': sha256(target), 'license': 'Apache-2.0',
                  'upstream': 'https://huggingface.co/Qwen/Qwen3-0.6B', 'quantization': 'Q4_K_M',
                  'baseRevision': config['revision'], 'trainingRunSha256': sha256(run_dir / 'training-run.json') if run else None,
                  'adapter': adapter_info,
                  'promptSha256': prompt_hash, 'templateSha256': fingerprint(tokenizer.chat_template),
                  'capabilityMode': 'mcp-tools' if mcp_mode else 'informational-text-only',
                  'contractSha256': sha256(VERIFIED) if mcp_mode else None,
                  'adapterContractSha256': sha256(ADAPTER) if mcp_mode else None,
                  'llamaCommit': expected, 'converterSha256': sha256(args.llama_source / 'convert_hf_to_gguf.py'),
                  'quantizerSha256': sha256(Path(converter_commands[1][0])),
                  'smokeOnly': run['smokeOnly'] if run else False, 'untunedControl': args.base_only, 'productionReady': False}
    if args.checkpoint:
        model_info['id'] = stem + '-q4'
    write_json(args.output / 'model-provenance.json', model_info)
    write_json(args.output / 'benchmark-manifest.json', {'runtime': json.loads((ROOT / 'plugin/models/manifest.json').read_text(encoding='utf-8-sig'))['runtime'],
                                                       'models': [model_info]})
    shutil.copy2(args.base / 'LICENSE', args.output / (target.name + '.LICENSE'))
    print('Exported candidate only; baseline remains untouched:', target)


if __name__ == '__main__':
    main()
