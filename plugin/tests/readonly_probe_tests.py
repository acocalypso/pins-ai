"""Enforce that live diagnostic proposals cannot invoke mutating MCP tools."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('probe',Path(__file__).resolve().parents[1]/'scripts/pi-mcp-readonly-probe.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)


class ReadOnlyProbeTests(unittest.TestCase):
    def test_exposed_get_status_is_allowed(self):
        call={'name':'get_status','arguments':{}}
        approved=probe.approve_readonly(call,{'get_status':{}})
        self.assertEqual(approved,call)
        self.assertIsNot(approved,call)

    def test_stop_cancel_and_hardware_calls_are_rejected_even_if_exposed(self):
        exposed={'get_status':{},'stop_all':{},'cancel_job':{},'connect_device':{}}
        for name in ('stop_all','cancel_job','connect_device'):
            with self.assertRaises(ValueError):probe.approve_readonly({'name':name,'arguments':{}},exposed)

    def test_unexposed_status_is_rejected(self):
        with self.assertRaises(ValueError):probe.approve_readonly({'name':'get_status','arguments':{}},{})

    def test_argument_injection_and_extra_fields_are_rejected(self):
        for call in ({'name':'get_status','arguments':{'force':True}},
                     {'name':'get_status','arguments':{},'authorized':True},
                     {'name':'get_status','arguments':[]}):
            with self.assertRaises(ValueError):probe.approve_readonly(call,{'get_status':{}})


if __name__=='__main__':unittest.main()
