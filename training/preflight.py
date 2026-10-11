"""Read-only workstation check; no model/dependency downloads or training."""
import argparse
import importlib.metadata
import json
import platform


def inspect(method='qlora'):
    report = {'platform': platform.platform(), 'python': platform.python_version(), 'method': method,
              'packages': {}, 'ready': False, 'issues': []}
    for name in ('torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes', 'huggingface-hub', 'jsonschema'):
        try:
            report['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report['issues'].append('Missing development dependency: ' + name)
    if 'torch' not in report['packages']:
        return report
    import torch
    if not torch.cuda.is_available():
        report['issues'].append('CUDA unavailable: no training on CPU or Pi in this recipe')
        return report
    device = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    report['gpu'] = {'name': device.name, 'computeCapability': list(torch.cuda.get_device_capability()),
                     'totalMiB': total // 1048576, 'freeMiB': free // 1048576, 'cuda': torch.version.cuda,
                     'torchArchitectures': torch.cuda.get_arch_list(),
                     'bf16Supported': torch.cuda.is_bf16_supported(including_emulation=False)}
    # Executing a tiny kernel catches wheels that no longer support Pascal.
    try:
        torch.ones(2, device='cuda').sum().item()
        if method == 'qlora':
            import bitsandbytes as bnb
            data = torch.randn(64, device='cuda')
            quantized, state = bnb.functional.quantize_4bit(data, quant_type='nf4')
            bnb.functional.dequantize_4bit(quantized, state).sum().item()
    except Exception as ex:
        report['issues'].append('GPU kernel compatibility failed: ' + str(ex))
    if free < 2500 * 1048576:
        report['issues'].append('Less than 2500 MiB free VRAM; close GPU applications or use another host')
    report['ready'] = not report['issues']
    report['note'] = 'Passing preflight does not guarantee a full training run fits or performs well'
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('qlora', 'lora'), default='qlora')
    args = parser.parse_args()
    result = inspect(args.method)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['ready'] else 2)
