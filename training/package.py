"""Package development sources only; never bundle weights, environments or credentials."""
from pathlib import Path
import zipfile

from common import ROOT, sha256


def main():
    files = [ROOT / 'LICENSE', ROOT / 'THIRD_PARTY_NOTICES.md', ROOT / '.gitignore',
             ROOT / 'src/Pins.AI.Runtime/Contracts.cs', ROOT / 'scripts/benchmark.py',
             ROOT / 'tests/training_tests.py', ROOT / 'tests/training_backend.py', ROOT / 'tests/v2_tests.py', ROOT / 'tests/colab_tests.py',
             ROOT / 'output/jupyter-notebook/pins-ai-v2-colab.ipynb']
    for directory in ('training', 'docs', 'models', 'benchmarks', 'PINS_MCP_AI_Training_v2'):
        files.extend(path for path in (ROOT / directory).rglob('*')
                     if path.is_file() and path.suffix in ('.py', '.json', '.jsonl', '.txt', '.md')
                     and '__pycache__' not in path.parts and 'runs' not in path.parts and not path.name.startswith('sft_'))
    files = sorted(set(files))
    archive = ROOT / 'artifacts/pins-ai-training-0.1.0.zip'
    archive.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as package:
        for path in files:
            package.write(path, path.relative_to(ROOT).as_posix())
        package.writestr('SHA256SUMS', ''.join(sha256(path) + '  ' + path.relative_to(ROOT).as_posix() + '\n' for path in files))
        package.writestr('README.md', '# PINS AI development training bundle\n\n'
                        'Off-device GPU training pipeline for workstations or Colab, not a Pi plugin installation. Start with docs/specialization.md or docs/colab.md.\n'
                        'No weights or training dependencies are included. See docs/v2.md for measured training status and pending evaluations.\n')
    print(archive)


if __name__ == '__main__':
    main()
