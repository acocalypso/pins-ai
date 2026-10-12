"""Package development sources only; never bundle weights, environments or credentials."""
from pathlib import Path
import zipfile

from common import ROOT, sha256


def main():
    files = [ROOT / 'LICENSE', ROOT / 'THIRD_PARTY_NOTICES.md', ROOT / '.gitignore',
             ROOT / 'plugin/src/Pins.AI.Runtime/Contracts.cs', ROOT / 'plugin/scripts/benchmark.py',
             ROOT / 'training/tests/training_tests.py', ROOT / 'training/tests/training_backend.py', ROOT / 'training/tests/v2_tests.py', ROOT / 'training/tests/colab_tests.py',
             ROOT / 'training/notebooks/pins-ai-colab.ipynb']
    files.append(ROOT / 'training/tests/curriculum_tests.py')
    for directory in ('training', 'docs', 'plugin/models', 'plugin/benchmarks'):
        files.extend(path for path in (ROOT / directory).rglob('*')
                     if path.is_file() and path.suffix in ('.py', '.json', '.jsonl', '.txt', '.md')
                     and not set(path.parts) & {'__pycache__', 'runs', '.venv', '.local', 'bin', 'obj'} and not path.name.startswith('sft_'))
    files = sorted(set(files))
    archive = ROOT / 'artifacts/pins-ai-training-0.1.0.zip'
    archive.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as package:
        for path in files:
            package.write(path, path.relative_to(ROOT).as_posix())
        package.writestr('SHA256SUMS', ''.join(sha256(path) + '  ' + path.relative_to(ROOT).as_posix() + '\n' for path in files))
        package.writestr('README.md', '# PINS AI development training bundle\n\n'
                        'Off-device GPU training pipeline for workstations or Colab, not a Pi plugin installation. Start with training/README.md for setup and commands.\n'
                        'No weights or training dependencies are included. See docs/v2.md for measured training status and pending evaluations.\n')
    print(archive)


if __name__ == '__main__':
    main()
