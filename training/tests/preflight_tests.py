"""Backend gating tests without a GPU; these do not certify real ROCm kernels."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import importlib.metadata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preflight import inspect


class Tensor:
    def sum(self): return self
    def item(self): return 2
    def __matmul__(self, other): return self


def fake_torch(hip='7.2.1', bf16=True):
    return SimpleNamespace(version=SimpleNamespace(hip=hip, cuda=None if hip else '12.6'),
        bfloat16='bf16', ones=lambda *a, **k: Tensor(),
        cuda=SimpleNamespace(is_available=lambda:True,
            get_device_properties=lambda index:SimpleNamespace(name='MI355X fixture',gcnArchName='gfx950'),
            mem_get_info=lambda:(8*1024**3,256*1024**3),
            get_device_capability=lambda:(6,1), get_arch_list=lambda:['gfx950'],
            is_bf16_supported=lambda **kwargs:bf16))


class PreflightTests(unittest.TestCase):
    def probe(self, method, precision='bf16', **hardware):
        checked=[]
        def version(name):
            checked.append(name)
            if name == 'bitsandbytes': raise importlib.metadata.PackageNotFoundError(name)
            return 'fixture'
        with patch.dict(sys.modules, {'torch':fake_torch(**hardware)}), patch('preflight.importlib.metadata.version',side_effect=version):
            return inspect(method,precision), checked

    def test_rocm_lora_does_not_require_bitsandbytes(self):
        report, packages=self.probe('lora')
        self.assertTrue(report['ready']); self.assertEqual(report['backend'],'rocm')
        self.assertEqual(report['gpu']['architecture'],'gfx950')
        self.assertNotIn('bitsandbytes',packages)

    def test_rocm_nf4_is_refused_without_silent_fallback(self):
        report,_=self.probe('qlora')
        self.assertFalse(report['ready'])
        self.assertTrue(any('unquantized LoRA' in issue for issue in report['issues']))

    def test_unsupported_bf16_is_refused(self):
        report,_=self.probe('lora',bf16=False)
        self.assertFalse(report['ready'])
        self.assertTrue(any('BF16' in issue for issue in report['issues']))

    def test_cuda_lora_is_also_available_without_nf4(self):
        report,_=self.probe('lora',precision='fp16',hip=None)
        self.assertTrue(report['ready']); self.assertEqual(report['backend'],'cuda')


if __name__ == '__main__': unittest.main()
