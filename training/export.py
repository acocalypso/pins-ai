"""Merge into an unquantized base, then convert/quantize with pinned llama.cpp."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from common import CONFIG, ROOT, fingerprint, policy, sha256, write_json


def commands(converter_python, source, merged, output, stem='pins-qwen3-0.6b'):
    binary = source / 'build/bin' / ('llama-quantize.exe' if os.name == 'nt' else 'llama-quantize')
    return [[str(converter_python), str(source / 'convert_hf_to_gguf.py'), str(merged),
             '--outfile', str(output / (stem + '-f16.gguf')), '--outtype', 'f16'],
            [str(binary), str(output / (stem + '-f16.gguf')),
             str(output / (stem + '-Q4_K_M.gguf')), 'Q4_K_M']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run', type=Path)
    mode.add_argument('--base-only', action='store_true', help='Untuned control from the identical pinned HF checkpoint')
    parser.add_argument('--llama-source', type=Path, required=True)
    parser.add_argument('--converter-python', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-smoke-export', action='store_true', help='Development only; never eligible for promotion')
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text(encoding='utf-8-sig'))
    run = json.loads((args.run / 'training-run.json').read_text(encoding='utf-8-sig')) if args.run else None
    if run and (run['status'] != 'completed' or (run['smokeOnly'] and not args.allow_smoke_export)):
        raise ValueError('Require a completed training run; smoke exports need explicit opt-in')
    base_info = json.loads((args.base / 'base-provenance.json').read_text(encoding='utf-8-sig'))
    if base_info['revision'] != config['revision'] or base_info['model'] != config['model'] or base_info['tokenizerOnly']:
        raise ValueError('Use the verified full pinned training checkpoint')
    if (run and run['config']['revision'] != config['revision']) or sha256(args.base / 'model.safetensors') != config['weightsSha256']:
        raise ValueError('Wrong unquantized pinned base checkpoint')
    for filename, expected_hash in (run['base'] if run else base_info)['files'].items():
        if sha256(args.base / filename) != expected_hash:
            raise ValueError('Original base file changed: ' + filename)
    if run and run['promptSha256'] != fingerprint(policy()):
        raise ValueError('Plugin policy changed since training; retrain against the intended current policy')
    expected = json.loads((ROOT / 'models/manifest.json').read_text(encoding='utf-8-sig'))['runtime']['commit']
    actual = subprocess.check_output(['git', '-C', str(args.llama_source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != expected:
        raise ValueError('llama.cpp converter/quantizer must match the pinned runtime commit')
    stem = 'qwen3-0.6b-training-control' if args.base_only else 'pins-qwen3-0.6b'
    converter_commands = commands(args.converter_python, args.llama_source, args.output / 'merged', args.output, stem)
    if not args.converter_python.is_file() or not Path(converter_commands[1][0]).is_file():
        raise ValueError('Install the separate converter environment and build llama-quantize first')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use a new output directory; never overwrite a base or previous export')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # Merge on CPU into clean FP32 weights, never merge into a 4-bit training model.
    base = AutoModelForCausalLM.from_pretrained(args.base, local_files_only=True, trust_remote_code=False,
                                               torch_dtype=torch.float32, device_map={'': 'cpu'})
    model = PeftModel.from_pretrained(base, args.run / 'adapter', local_files_only=True).merge_and_unload(safe_merge=True) if run else base
    tokenizer = AutoTokenizer.from_pretrained(args.run / 'adapter' if run else args.base, local_files_only=True, trust_remote_code=False)
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
    model_info = {'id': 'qwen3-0.6b-training-control-q4' if args.base_only else 'pins-qwen3-0.6b-specialized-q4',
                  'profile': 'lite', 'file': target.name,
                  'bytes': target.stat().st_size, 'sha256': sha256(target), 'license': 'Apache-2.0',
                  'upstream': 'https://huggingface.co/Qwen/Qwen3-0.6B', 'quantization': 'Q4_K_M',
                  'baseRevision': config['revision'], 'trainingRunSha256': sha256(args.run / 'training-run.json') if run else None,
                  'promptSha256': fingerprint(policy()), 'templateSha256': fingerprint(tokenizer.chat_template),
                  'llamaCommit': expected, 'converterSha256': sha256(args.llama_source / 'convert_hf_to_gguf.py'),
                  'quantizerSha256': sha256(Path(converter_commands[1][0])),
                  'smokeOnly': run['smokeOnly'] if run else False, 'untunedControl': args.base_only, 'productionReady': False}
    write_json(args.output / 'model-provenance.json', model_info)
    write_json(args.output / 'benchmark-manifest.json', {'runtime': json.loads((ROOT / 'models/manifest.json').read_text(encoding='utf-8-sig'))['runtime'],
                                                       'models': [model_info]})
    shutil.copy2(args.base / 'LICENSE', args.output / (target.name + '.LICENSE'))
    print('Exported candidate only; baseline remains untouched:', target)


if __name__ == '__main__':
    main()
