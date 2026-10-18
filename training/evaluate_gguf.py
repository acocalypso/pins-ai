"""Offline GGUF proposal evaluation with a temporary local llama.cpp test process.

No MCP connections or equipment operations. Production inference remains PINS-owned.
"""
import argparse
import json
import platform
import secrets
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from common import ROOT, fingerprint, sha256
from evaluation_state import (select_cases, prepare_cases, score, new_report, ProgressWriter,
                              summarize, scoring_hash, resume_report)
from v2_contracts import POLICY, VERIFIED, ADAPTER, SCORER_VERSION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--lora', type=Path)
    p.add_argument('--server', type=Path, required=True)
    p.add_argument('--llama-source', type=Path, default=ROOT / '.tools/llama.cpp')
    p.add_argument('--tokenizer', type=Path, default=ROOT / '.local/training/base')
    p.add_argument('--suite', choices=('validation_smoke', 'eval_gold', 'hard_gold', 'future_holdout'), default='hard_gold')
    p.add_argument('--prompt-mode', choices=('legacy', 'chat'), default='legacy')
    p.add_argument('--per-language', type=int, default=0)
    p.add_argument('--job-continuations', action='store_true')
    p.add_argument('--max-new-tokens', type=int, default=128)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume-from', type=Path, help='Resume a verified report into a NEW output file')
    args = p.parse_args()
    if args.prompt_mode == 'chat' and args.suite not in ('validation_smoke', 'future_holdout'):
        p.error('Chat projection is for development validation; gold remains unchanged')
    if platform.machine().lower() in ('arm64', 'aarch64'):
        p.error('Run offline training evaluation on a workstation; Pi inference must remain PINS-owned')
    if args.output.exists() or args.per_language < 0 or not 1 <= args.max_new_tokens <= 256:
        p.error('Use a new output file and a bounded token budget')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    rows, path, inputs = select_cases(args.suite, args.per_language, args.job_continuations, args.prompt_mode)
    cases = prepare_cases(tokenizer, rows, args.prompt_mode)
    from chat_dataset_common import CHAT_POLICY
    evaluation_policy = CHAT_POLICY if args.prompt_mode == 'chat' else POLICY
    for c in cases:
        if len(tokenizer.encode(c['prompt'], add_special_tokens=False)) + args.max_new_tokens > 1536:
            raise ValueError('Case exceeds plugin context budget: ' + c['id'])
    runtime = json.loads((ROOT / 'plugin/models/manifest.json').read_text(encoding='utf-8'))['runtime']
    actual = subprocess.check_output(['git', '-C', str(args.llama_source), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != runtime['commit']:
        raise ValueError('Build the server from the pinned llama.cpp source')
    identity = {'modelSha256': sha256(args.model), 'loraSha256': sha256(args.lora) if args.lora else None,
                'runtimeCommit': actual, 'serverSha256': sha256(args.server), 'suite': args.suite,
                'inputSha256': inputs, 'contractSha256': sha256(VERIFIED), 'adapterContractSha256': sha256(ADAPTER),
                'scorerVersion': SCORER_VERSION, 'scoringImplementationSha256': scoring_hash(),
                'policySha256': fingerprint(evaluation_policy), 'promptMode': args.prompt_mode,
                'templateSha256': fingerprint(tokenizer.chat_template),
                'generation': {'maxNewTokens': args.max_new_tokens, 'temperature': 0, 'seed': 3407,
                               'threads': 2, 'contextTokens': 1536, 'stop': ['<|im_end|>']},
                'selection': {'perLanguage': args.per_language, 'jobContinuationsRequested': args.job_continuations},
                'caseInputsSha256': fingerprint([{k: c[k] for k in ('id','caseSha256','promptSha256')} for c in cases]),
                'backend': 'llama.cpp CPU Q4_K_M'}
    report = new_report(identity, cases, {'suite': args.suite, 'hardwareExecution': False,
                                        'languageAndFalseClaims': 'human review pending'})
    if args.resume_from:
        report = resume_report(args.resume_from, report, cases)
    writer = ProgressWriter(args.output, args.resume_from)
    writer.write(report)
    if report['completed']:
        print('Verified completed report; no generation required')
        return
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
    key = secrets.token_urlsafe(32)
    process = None
    logpath = args.output.with_suffix('.runtime.log')
    with tempfile.TemporaryDirectory(prefix='pins-ai-evaluation-') as temp, logpath.open('xb') as log:
        keyfile = Path(temp) / 'runtime.key'; keyfile.write_text(key, encoding='utf-8')
        command = [str(args.server.resolve()), '-m', str(args.model.resolve()), '--host', '127.0.0.1',
                   '--port', str(port), '--api-key-file', str(keyfile), '-c', '1536', '-t', '2', '-tb', '2',
                   '-ngl', '0', '--parallel', '1', '--no-webui', '--no-warmup']
        if args.lora:
            command += ['--lora', str(args.lora.resolve())]
        try:
            process = subprocess.Popen(command, stdout=log, stderr=log)
            for _ in range(120):
                if process.poll() is not None: raise RuntimeError('Runtime exited; inspect ' + str(logpath))
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1) as r:
                        if r.status == 200: break
                except (OSError, urllib.error.URLError): time.sleep(.5)
            else: raise TimeoutError('Runtime startup timed out')
            for case in cases[len(report['results']):]:
                payload = {'prompt': case['prompt'], 'n_predict': args.max_new_tokens, 'temperature': 0,
                           'seed': 3407, 'cache_prompt': True, 'stream': False, 'stop': ['<|im_end|>']}
                request = urllib.request.Request(f'http://127.0.0.1:{port}/completion',
                    data=json.dumps(payload).encode(), headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
                started = time.monotonic()
                with urllib.request.urlopen(request, timeout=90) as response: generated = json.load(response)
                seconds = time.monotonic() - started
                text = generated['content']; tokens = generated['tokens_predicted']
                row = {**{k: case[k] for k in ('id','locale','category','expected','caseSha256','promptSha256')},
                       **score(case, text), 'text': text, 'seconds': seconds, 'outputTokens': tokens,
                       'tokensPerSecond': tokens / seconds, 'outputTruncated': bool(generated.get('stopped_limit')),
                       'conditionalPoll': case.get('conditionalPoll', False)}
                report['results'].append(row); summarize(report); writer.write(report)
                print(json.dumps({'id': row['id'], 'casesCompleted': report['casesCompleted'],
                                  'expectedCaseCount': report['expectedCaseCount'], 'exactArgumentsCorrect': row['exactArgumentsCorrect']}), flush=True)
        except BaseException as error:
            report['error'] = type(error).__name__ + ': ' + str(error)
            raise
        finally:
            if process and process.poll() is None:
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
            writer.write(report)


if __name__ == '__main__':
    main()
