"""Package development sources only; never bundle weights, environments or credentials."""
from pathlib import Path
import zipfile

from common import ROOT, sha256


def main():
    files = [ROOT / 'LICENSE', ROOT / 'THIRD_PARTY_NOTICES.md', ROOT / '.gitignore',
             ROOT / 'src/Pins.AI.Runtime/Contracts.cs', ROOT / 'scripts/benchmark.py',
             ROOT / 'tests/training_tests.py', ROOT / 'tests/training_backend.py']
    for directory in ('training', 'docs', 'models', 'benchmarks'):
        files.extend(path for path in (ROOT / directory).rglob('*')
                     if path.is_file() and path.suffix in ('.py', '.json', '.jsonl', '.txt', '.md')
                     and '__pycache__' not in path.parts and 'runs' not in path.parts)
    files = sorted(set(files))
    archive = ROOT / 'artifacts/pins-ai-training-0.1.0.zip'
    archive.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as package:
        for path in files:
            package.write(path, path.relative_to(ROOT).as_posix())
        package.writestr('SHA256SUMS', ''.join(sha256(path) + '  ' + path.relative_to(ROOT).as_posix() + '\n' for path in files))
        package.writestr('README.md', '# PINS AI development training bundle\n\n'
                        'Workstation-only pipeline, not a Pi plugin installation. Start with docs/specialization.md.\n'
                        'No weights or training dependencies are included. No PINS model has been trained yet.\n')
    print(archive)


if __name__ == '__main__':
    main()
