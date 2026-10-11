"""Offline converter for reviewed supervision tied to actual exported MCP schemas.

This stages schema-grounded tool data for the later MCP phase. It never contacts
a server, executes a call or mixes tool supervision into today's chat-only run.
"""
import argparse
import json
from pathlib import Path

from common import fingerprint, read_jsonl, sha256, write_json, write_jsonl


def reject_external_refs(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in ('$ref', '$dynamicRef') and (not isinstance(child, str) or not child.startswith('#')):
                raise ValueError('Only local JSON Schema references are allowed in offline snapshots')
            reject_external_refs(child)
    elif isinstance(value, list):
        for child in value:
            reject_external_refs(child)


def convert(snapshot, examples):
    from jsonschema import FormatChecker
    from jsonschema.validators import validator_for
    definitions = snapshot.get('tools')
    if not isinstance(definitions, list) or not definitions:
        raise ValueError('Provide the actual tools/list result object, with a nonempty tools array')
    by_name = {}
    for definition in definitions:
        name = definition.get('name')
        schema = definition.get('inputSchema')
        if not isinstance(name, str) or not name or name in by_name or not isinstance(schema, dict):
            raise ValueError('Unique real tool names and actual inputSchema objects are required')
        reject_external_refs(schema)
        validator_for(schema).check_schema(schema)
        by_name[name] = definition
    output = []
    for row in examples:
        if not row.get('id') or not row.get('group') or row.get('locale') not in ('en', 'de'):
            raise ValueError('Reviewed examples require id, group and en/de locale')
        provenance = row.get('provenance', {})
        if not provenance.get('source') or not provenance.get('license') or provenance.get('containsPrivateData') is not False:
            raise ValueError('Explicit provenance and private-data declaration required')
        if row.get('reviewed') is not True:
            raise ValueError('Manually review tool examples before importing')
        selected = row.get('toolNames')
        if not isinstance(selected, list) or not 1 <= len(selected) <= 8 or len(set(selected)) != len(selected):
            raise ValueError('Choose 1-8 distinct actual tools for each example')
        if any(name not in by_name for name in selected):
            raise ValueError('Unknown tool name; never invent an API')
        messages = row.get('messages')
        if not isinstance(messages, list) or not messages or messages[0].get('role') != 'user' or messages[-1].get('role') != 'assistant':
            raise ValueError('Provide a user-started conversation ending in a supervised assistant turn')
        pending = {}
        for message in messages:
            role = message.get('role')
            if role not in ('user', 'assistant', 'tool'):
                raise ValueError('System policy is not supplied by untrusted conversation rows')
            content = message.get('content')
            calls = message.get('tool_calls', [])
            if content is not None and (not isinstance(content, str) or len(content) > 2000):
                raise ValueError('Bounded text content required')
            if not calls and (not isinstance(content, str) or not content.strip()):
                raise ValueError('Nonempty text or structured assistant call required')
            if role != 'assistant' and calls:
                raise ValueError('Only assistant turns may contain structured supervision calls')
            if role == 'tool':
                call_id = message.get('tool_call_id')
                if call_id not in pending:
                    raise ValueError('Tool response without a matching recorded call')
                pending.pop(call_id)
            for call in calls:
                function = call.get('function', {})
                name = function.get('name')
                if name not in selected or call.get('type') != 'function' or not call.get('id'):
                    raise ValueError('Calls must refer to selected actual schema definitions')
                if call['id'] in pending:
                    raise ValueError('Duplicate pending tool call ID')
                arguments = function.get('arguments')
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise ValueError('Tool arguments must be an object')
                schema = by_name[name]['inputSchema']
                validator_for(schema)(schema, format_checker=FormatChecker()).validate(arguments)
                pending[call['id']] = name
        output.append({**row, 'tools': [by_name[name] for name in selected],
                       'schemaSha256': fingerprint(snapshot), 'capabilityMode': 'schema-supervision-only'})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schema-snapshot', type=Path, required=True)
    parser.add_argument('--examples', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    snapshot = json.loads(args.schema_snapshot.read_text(encoding='utf-8-sig'))
    converted = convert(snapshot, read_jsonl(args.examples))
    write_jsonl(args.output, converted)
    write_json(args.output.with_suffix('.provenance.json'),
               {'schemaFileSha256': sha256(args.schema_snapshot), 'examplesSha256': sha256(args.examples),
                'count': len(converted), 'execution': 'none', 'training': 'staged for future actual-MCP phase'})
    print('Validated and staged', len(converted), 'examples; no tool was executed')


if __name__ == '__main__':
    main()
