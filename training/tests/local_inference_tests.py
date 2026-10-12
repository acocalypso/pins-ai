"""Offline native-call and conversation contracts; no GPU or equipment required."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training'))
from local_inference import normalize_tools, extract_tool_calls, append_tool_result


TOOL = {'name': 'get_server_status', 'description': 'Simulated only',
        'parameters': {'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False}}
CALL = {'name': TOOL['name'], 'arguments': {}}


class InferenceContracts(unittest.TestCase):
    def test_native_call_extracted_without_execution(self):
        text = '<tool_call>\n' + json.dumps(CALL) + '\n</tool_call>'
        self.assertEqual(extract_tool_calls(text, [TOOL]), [CALL])

    def test_no_tool_text_has_no_calls(self):
        self.assertEqual(extract_tool_calls('The server state is unknown.', [TOOL]), [])

    def test_bare_json_and_truncated_calls_rejected(self):
        for text in (json.dumps(CALL), '<tool_call>' + json.dumps(CALL), '</tool_call>'):
            with self.assertRaises(ValueError): extract_tool_calls(text, [TOOL])

    def test_unknown_tool_and_invalid_arguments_rejected(self):
        from jsonschema.exceptions import ValidationError
        for call in ({'name': 'other', 'arguments': {}}, {'name': TOOL['name'], 'arguments': {'unexpected': 1}}):
            with self.assertRaises((ValueError, ValidationError)):
                extract_tool_calls('<tool_call>' + json.dumps(call) + '</tool_call>', [TOOL])

    def test_multiple_calls_rejected(self):
        with self.assertRaises(ValueError):
            extract_tool_calls(('<tool_call>' + json.dumps(CALL) + '</tool_call>') * 2, [TOOL])

    def test_tools_normalized_and_duplicate_names_rejected(self):
        normalized = normalize_tools([TOOL])
        self.assertEqual(normalize_tools(normalized), normalized)
        with self.assertRaises(ValueError): normalize_tools([TOOL, TOOL])

    def test_feedback_is_copied_and_keeps_native_conversation(self):
        original = [{'role': 'user', 'content': 'Check status'}]
        response = {'toolCalls': [CALL], 'toolCallError': None}
        history = append_tool_result(original, response, TOOL['name'], {'status': 'ok'})
        self.assertEqual(len(original), 1)
        self.assertEqual(history[-2]['tool_calls'][0]['function'], CALL)
        self.assertEqual(json.loads(history[-1]['content']), {'status': 'ok'})
        with self.assertRaises(ValueError): append_tool_result(original, response, 'other', {})


if __name__ == '__main__': unittest.main()
