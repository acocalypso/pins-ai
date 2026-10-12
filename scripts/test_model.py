"""Offline PEFT checkpoint CLI and inference smoke; never executes MCP tools."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
from common import sha256
from local_inference import LocalInference, append_tool_result
from v2_contracts import POLICY


def snapshot(path):
    return {p.relative_to(path).as_posix(): sha256(p) for p in path.rglob('*') if p.is_file()}


def smoke(session, options):
    tool = {'name': 'get_server_status', 'description': 'Retrieve the current status of a server',
            'parameters': {'type': 'object', 'properties': {}, 'required': []}}
    results = []
    cases = [('A-basic', 'Hello. Introduce yourself briefly.', None),
        ('B-json', 'Return a JSON object with the keys "status" and "message". Set status to "ok".', None),
        ('C-tool-format', 'Check the current server status.', [tool])]
    for name, prompt, tools in cases:
        messages = [{'role': 'system', 'content': POLICY}, {'role': 'user', 'content': prompt}]
        response = session.generate(messages, tools=tools, **options)
        row = {'test': name, 'prompt': prompt, 'tools': tools, **response}
        if name == 'A-basic':
            row['passed'] = bool(response['text'].strip()) and not response['truncated']
        elif name == 'B-json':
            try:
                parsed = json.loads(response['text'])
                row['parsedJson'] = parsed
                row['passed'] = isinstance(parsed, dict) and set(parsed) == {'status', 'message'} and parsed['status'] == 'ok'
            except ValueError as error:
                row.update(passed=False, jsonError=str(error))
        else:
            row['passed'] = not response['toolCallError'] and response['toolCalls'] == [
                {'name': 'get_server_status', 'arguments': {}}] and not response['truncated']
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if name == 'C-tool-format' and row['passed']:
            history = append_tool_result(messages, response, 'get_server_status', {'status': 'ok', 'uptime_seconds': 42})
            followup = session.generate(history, tools=tools, **options)
            results.append({'test': 'D-simulated-feedback', 'simulatedToolResult': True, **followup})
            print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True, help='Existing verified base folder, no downloads')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Saved PEFT training checkpoint')
    parser.add_argument('--backend', choices=('auto', 'cuda-nf4', 'cpu-fp32'), default='auto')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prompt')
    group.add_argument('--messages', type=Path, help='JSON conversation, including system/tool messages')
    group.add_argument('--interactive', action='store_true')
    group.add_argument('--smoke', action='store_true')
    parser.add_argument('--system', default=POLICY)
    parser.add_argument('--tools', type=Path, help='JSON list of schemas; these are never executed')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-p', type=float, default=0.9)
    sample = parser.add_mutually_exclusive_group()
    sample.add_argument('--sample', action='store_true', help='Enable temperature/top-p sampling')
    sample.add_argument('--deterministic', action='store_true', help='Greedy decoding (default)')
    parser.add_argument('--output', type=Path, help='New JSON evidence file; never overwritten')
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.temperature <= 0 or not 0 < args.top_p <= 1:
        parser.error('Invalid generation parameters')
    if args.output and args.output.exists(): parser.error('Use a new evidence file')
    checkpoint = args.checkpoint.resolve()
    before = snapshot(checkpoint)
    options = {'max_new_tokens': args.max_new_tokens, 'temperature': args.temperature,
               'top_p': args.top_p, 'deterministic': not args.sample}
    tools = json.loads(args.tools.read_text(encoding='utf-8')) if args.tools else None
    report = {'startedAt': datetime.now(timezone.utc).isoformat(), 'generation': options,
              'hardwareOperations': False, 'checkpointFilesSha256': before,
              'packages': {n: importlib.metadata.version(n) for n in
                           ('torch', 'transformers', 'peft', 'bitsandbytes', 'safetensors', 'jsonschema')}}
    with LocalInference(args.base, checkpoint, args.backend) as session:
        report['model'] = session.info
        print(json.dumps({'loaded': session.info}, ensure_ascii=False), flush=True)
        if args.smoke:
            report['results'] = smoke(session, options)
        elif args.interactive:
            messages = [{'role': 'system', 'content': args.system}]
            report['results'] = []
            while True:
                try: prompt = input('You (/quit to exit): ')
                except EOFError: break
                if prompt.strip().lower() in ('/quit', '/exit'): break
                messages.append({'role': 'user', 'content': prompt})
                response = session.generate(messages, tools, **options)
                report['results'].append(response)
                print(response['rawText'], flush=True)
                if response['toolCalls'] or response['toolCallError']:
                    print('Tool proposal was not executed; supply feedback through the reusable API or --messages.')
                    messages.pop()
                else:
                    messages.append({'role': 'assistant', 'content': response['text']})
        else:
            messages = json.loads(args.messages.read_text(encoding='utf-8')) if args.messages else [
                {'role': 'system', 'content': args.system}, {'role': 'user', 'content': args.prompt}]
            report['results'] = [session.generate(messages, tools, **options)]
            print(report['results'][0]['rawText'], flush=True)
        if session.device == 'cuda':
            import torch
            report['peakGpuAllocatedMiB'] = torch.cuda.max_memory_allocated() / 1048576
    report['checkpointUnchanged'] = before == snapshot(checkpoint)
    if not report['checkpointUnchanged']: raise RuntimeError('Checkpoint files changed during inference')
    report['finishedAt'] = datetime.now(timezone.utc).isoformat()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print('Evidence:', args.output)
    if args.smoke and not all(r['passed'] for r in report['results'] if 'passed' in r):
        return 2
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('Inference interrupted; resources released.', file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f'Inference failed: {type(error).__name__}: {error}\nNo download or precision fallback was attempted.', file=sys.stderr)
        sys.exit(1)
