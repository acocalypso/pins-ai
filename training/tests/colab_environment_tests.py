"""CPU-only planner tests: no installers, model downloads or GPU operations."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import colab_environment as environment


def fake_torch(version='2.9.0+cu128', cuda='12.8', available=True, capability=(7, 5)):
    return SimpleNamespace(__version__=version, version=SimpleNamespace(cuda=cuda, hip=None),
        cuda=SimpleNamespace(is_available=lambda: available,
            get_device_capability=lambda index: capability,
            get_device_name=lambda index: 'Tesla T4'))


def wheel(name='bitsandbytes', version='0.49.1'):
    return {'metadata': {'name': name, 'version': version}, 'download_info': {
        'url': f'https://files.pythonhosted.org/{name}-{version}-py3-none-any.whl',
        'archive_info': {'hashes': {'sha256': 'a' * 64}}}}


class StackTests(unittest.TestCase):
    def test_reuses_newer_colab_candidates(self):
        for python in ((3, 11, 9), (3, 12, 12), (3, 13, 1)):
            with self.subTest(python=python):
                report = environment.inspect_stack(fake_torch(), python=python, system='Linux')
                self.assertEqual(report['torch'], '2.9.0+cu128')
                self.assertEqual(report['executable'], sys.executable)

    def test_unsupported_candidates_fail_without_installation(self):
        cases = [fake_torch(available=False), fake_torch(cuda=None),
                 fake_torch(version='2.6.0'), fake_torch(version='3.0.0'),
                 fake_torch(version='2.10.0.dev1'), fake_torch(cuda='13.1'),
                 fake_torch(capability=(6, 0))]
        for torch in cases:
            with self.subTest(torch=torch), self.assertRaises(RuntimeError):
                environment.inspect_stack(torch, python=(3, 12, 0), system='Linux')
        for python in ((3, 10, 0), (3, 14, 0)):
            with self.assertRaisesRegex(RuntimeError, 'Python'):
                environment.inspect_stack(fake_torch(), python=python, system='Linux')

    def test_repository_pins_are_used_without_torch(self):
        requirements = environment.training_requirements()
        self.assertIn('bitsandbytes==0.49.1', requirements)
        self.assertIn('transformers==4.57.6', requirements)
        self.assertFalse(any(environment.protected(r.split('==')[0]) for r in requirements))


class PlanTests(unittest.TestCase):
    def test_protected_gpu_packages_are_never_approved_even_if_missing(self):
        for name in ('torch', 'TorchVision', 'torchaudio', 'triton', 'nvidia-cublas-cu12', 'pytorch_triton'):
            with self.subTest(name=name), self.assertRaisesRegex(RuntimeError, 'protected GPU'):
                environment.approved_wheels({'install': [wheel(name)]})

    def test_requires_hashed_wheel_without_source_builds(self):
        for url in ('http://example.com/pkg.whl', 'https://example.com/pkg.tar.gz'):
            item = wheel()
            item['download_info']['url'] = url
            with self.assertRaisesRegex(RuntimeError, 'HTTPS wheels'):
                environment.approved_wheels({'install': [item]})
        item = wheel()
        item['download_info']['archive_info']['hashes'] = {}
        with self.assertRaisesRegex(RuntimeError, 'SHA256'):
            environment.approved_wheels({'install': [item]})

    def setup_overlay(self, before, planned):
        desired = dict(r.split('==') for r in environment.training_requirements())
        after = {**before, **desired}
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        report_path = Path(temp.name) / 'environment.json'
        commands = []

        def run(command):
            command = list(map(str, command))
            commands.append(command)
            if '--dry-run' in command:
                constraints = Path(command[command.index('--constraint') + 1]).read_text()
                self.assertIn('torch==2.9.0+cu128', constraints)
                self.assertIn('nvidia-cublas-cu12==12.8.4', constraints)
                Path(command[command.index('--report') + 1]).write_text(json.dumps({'install': planned}))

        with patch.object(environment, 'installed_packages', side_effect=[before, after, after]), \
             patch.object(environment, 'inspect_stack', return_value={'torch': '2.9.0+cu128'}), \
             patch.dict(sys.modules, {'torch': fake_torch()}), patch.object(environment, 'run', side_effect=run):
            environment.setup(report_path)
        return commands, json.loads(report_path.read_text())

    def test_satisfied_stack_never_runs_installer(self):
        before = {**dict(r.split('==') for r in environment.training_requirements()),
                  'torch': '2.9.0+cu128', 'nvidia-cublas-cu12': '12.8.4'}
        commands, report = self.setup_overlay(before, [])
        self.assertEqual(len(commands), 2)  # dry-run and fresh-process import check
        self.assertIn('--dry-run', commands[0])
        self.assertEqual(report['packagesBefore'], report['packagesAfter'])
        self.assertEqual(report['status'], 'overlay-ready')

    def test_missing_and_incompatible_packages_use_only_approved_plan(self):
        before = {'torch': '2.9.0+cu128', 'nvidia-cublas-cu12': '12.8.4', 'transformers': '5.0.0'}
        commands, report = self.setup_overlay(before, [wheel(), wheel('transformers', '4.57.6')])
        self.assertEqual(len(commands), 3)
        apply = commands[1]
        self.assertEqual(apply[:4], [sys.executable, '-m', 'pip', 'install'])
        self.assertIn('--no-deps', apply)
        self.assertIn('--only-binary=:all:', apply)
        self.assertEqual(len([arg for arg in apply if '#sha256=' in arg]), 2)
        self.assertEqual(report['packagesBefore']['transformers'], '5.0.0')
        self.assertEqual(report['packagesAfter']['transformers'], '4.57.6')
        for command in commands:
            self.assertNotIn('uv', command)
            self.assertNotIn('venv', command)
            self.assertFalse(any(arg.startswith('torch==') for arg in command))

    def test_missing_torch_fails_without_installer_and_saves_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            report = Path(temp) / 'environment.json'
            with patch.object(environment, 'installed_packages', return_value={}), \
                 patch.object(environment, 'run') as run, \
                 self.assertRaisesRegex(RuntimeError, 'No torch download'):
                environment.setup(report)
            run.assert_not_called()
            self.assertEqual(json.loads(report.read_text())['status'], 'failed')

    def test_protected_plan_fails_before_apply(self):
        with self.assertRaisesRegex(RuntimeError, 'protected GPU'):
            self.setup_overlay({'torch': '2.9.0+cu128', 'nvidia-cublas-cu12': '12.8.4'}, [wheel('torch')])

    def test_notebook_generator_uses_hosted_interpreter_without_download(self):
        source = (environment.ROOT / 'training/create_colab_notebook.py').read_text(encoding='utf-8')
        self.assertIn('PYTHON = Path(sys.executable)', source)
        self.assertIn("'training/colab_environment.py'", source)
        self.assertIn("setup_command([PYTHON, 'training/preflight.py'])", source)
        for forbidden in ('--managed-python', "'python', 'install', '3.11'", "'torch==2.7.1'", 'training-venv'):
            self.assertNotIn(forbidden, source)


if __name__ == '__main__':
    unittest.main()
