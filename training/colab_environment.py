"""Install the training overlay without replacing Colab's Python/CUDA stack.

Compatibility candidates, not GPU validation: Python 3.11-3.13, torch 2.x >=2.7,
CUDA 11.8-13.0. A matching preflight, smoke and strict resume check still apply.
Sources: https://huggingface.co/docs/bitsandbytes/v0.49.1/installation
and https://research.google.com/colaboratory/runtime-version-faq.html
"""
import argparse
import importlib.metadata as metadata
import json
import platform
import re
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def normalized(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def protected(name):
    name = normalized(name)
    return name in {'torch', 'torchvision', 'torchaudio', 'triton', 'pytorch-triton'} or name.startswith('nvidia-')


def installed_packages():
    return {normalized(d.metadata['Name']): d.version for d in metadata.distributions() if d.metadata['Name']}


def training_requirements():
    """Read the repository pins, so the Colab overlay cannot silently drift."""
    result = []
    for name in ('requirements-common.txt', 'requirements.txt'):
        for line in (ROOT / 'training' / name).read_text(encoding='utf-8').splitlines():
            line = line.split('#', 1)[0].strip()
            if not line or line == '-r requirements-common.txt':
                continue
            if not re.fullmatch(r'[A-Za-z0-9_.-]+==[A-Za-z0-9_.+!-]+', line):
                raise RuntimeError('Expected an exact training package pin: ' + line)
            if protected(line.split('==')[0]):
                raise RuntimeError('Colab overlay must not request a replacement GPU stack: ' + line)
            result.append(line)
    return result


def inspect_stack(torch, python=None, system=None):
    """No kernels here; preflight.py performs the mandatory CUDA/NF4 kernel test."""
    python = python or sys.version_info[:3]
    system = system or platform.system()
    if system != 'Linux' or not ((3, 11) <= tuple(python[:2]) < (3, 14)):
        raise RuntimeError('Colab overlay requires Linux Python 3.11-3.13; select a supported hosted runtime.')
    version = str(torch.__version__)
    match = re.fullmatch(r'(\d+)\.(\d+)\.(\d+)(?:\+[A-Za-z0-9.]+)?', version)
    if not match or not ((2, 7) <= tuple(map(int, match.groups()[:2])) < (3, 0)):
        raise RuntimeError('Unsupported PyTorch ' + version + '; choose a supported Colab runtime, rather than reinstalling torch.')
    cuda = getattr(torch.version, 'cuda', None)
    if getattr(torch.version, 'hip', None) or not cuda or not torch.cuda.is_available():
        raise RuntimeError('A working NVIDIA CUDA PyTorch runtime is required; CPU-only/ROCm runtimes are unsupported.')
    # Only CUDA wheel targets documented for the pinned bitsandbytes release.
    if cuda not in {'11.8', '12.0', '12.1', '12.2', '12.3', '12.4', '12.5', '12.6', '12.8', '12.9', '13.0'}:
        raise RuntimeError('Unsupported CUDA ' + cuda + ' for pinned bitsandbytes; choose a compatible hosted runtime.')
    capability = tuple(torch.cuda.get_device_capability(0))
    if capability < (7, 5):
        raise RuntimeError('This Colab recipe requires a T4 (sm75) or newer NVIDIA GPU.')
    return {'executable': sys.executable, 'python': '.'.join(map(str, python)),
            'torch': version, 'cuda': cuda, 'gpu': torch.cuda.get_device_name(0),
            'computeCapability': list(capability)}


def approved_wheels(plan):
    """Install precisely the reviewed resolver plan; never resolve again on apply."""
    wheels = []
    for item in plan.get('install', []):
        name = item['metadata']['name']
        if protected(name):
            raise RuntimeError('Dependency resolution would replace/add protected GPU package: ' + name)
        download = item.get('download_info', {})
        url = download.get('url', '')
        digest = download.get('archive_info', {}).get('hashes', {}).get('sha256', '')
        if not url.startswith('https://') or not url.split('?', 1)[0].endswith('.whl') or not re.fullmatch('[a-fA-F0-9]{64}', digest):
            raise RuntimeError('Only HTTPS wheels with SHA256 are accepted for ' + name)
        wheels.append(url + '#sha256=' + digest)
    return wheels


def run(command):
    subprocess.run(list(map(str, command)), cwd=ROOT, check=True)


def setup(report_path):
    report_path = Path(report_path)
    before = installed_packages()
    report = {'status': 'started', 'packagesBefore': before, 'requirements': training_requirements(),
              'validation': 'Candidate only; GPU preflight and matching smoke/resume checks remain mandatory.'}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if 'torch' not in before:
            raise RuntimeError('Hosted runtime has no PyTorch; choose a CUDA Colab GPU runtime. No torch download is attempted.')
        import torch
        report['stack'] = inspect_stack(torch)
        with tempfile.TemporaryDirectory(prefix='pins-colab-overlay-') as directory:
            constraints = Path(directory) / 'gpu-stack.txt'
            constraints.write_text(''.join(f'{name}=={version}\n' for name, version in sorted(before.items()) if protected(name)), encoding='utf-8')
            plan_path = Path(directory) / 'plan.json'
            # pip normally preserves satisfied packages; constrain every existing GPU
            # component too, and reject any protected entry before applying the plan.
            run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                 '--only-binary=:all:', '--dry-run', '--report', plan_path,
                 '--constraint', constraints, *report['requirements']])
            report['installPlan'] = json.loads(plan_path.read_text(encoding='utf-8'))
            wheels = approved_wheels(report['installPlan'])
            if wheels:
                run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                     '--only-binary=:all:', '--no-deps', *wheels])
        after = installed_packages()
        if {k: v for k, v in before.items() if protected(k)} != {k: v for k, v in after.items() if protected(k)}:
            raise RuntimeError('Protected GPU stack changed during setup; stop and start a fresh hosted runtime.')
        for requirement in report['requirements']:
            name, expected = requirement.split('==')
            if after.get(normalized(name)) != expected:
                raise RuntimeError('Training dependency did not match required pin: ' + requirement)
        # Import in a fresh process after pip updates; the notebook kernel never
        # imports the training libraries. Detect optional preinstalled package ABI
        # conflicts before model downloads, including torchvision/NumPy problems.
        run([sys.executable, '-c', 'from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer; import peft, accelerate, bitsandbytes'])
        report['status'] = 'overlay-ready'
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = str(error)
        raise
    finally:
        report['packagesAfter'] = installed_packages()
        report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    setup(args.report)
