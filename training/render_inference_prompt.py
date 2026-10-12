"""JSON-lines tokenizer/parser worker for offline Pi diagnostics; no model loading."""
import argparse
import json
from pathlib import Path
import sys

from common import fingerprint
from local_inference import normalize_tools, extract_tool_calls
from v2_contracts import POLICY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=False)
    run = json.loads((args.checkpoint.parent.parent / 'training-run.json').read_text(encoding='utf-8'))
    if fingerprint(tokenizer.chat_template) != run['templateSha256']:
        raise ValueError('Saved tokenizer differs from training')
    sys.stdin.reconfigure(encoding='utf-8'); sys.stdout.reconfigure(encoding='utf-8')
    for line in sys.stdin:
        try:
            item = json.loads(line)
            if item['action'] == 'render':
                prompt = tokenizer.apply_chat_template(item['messages'], tools=normalize_tools(item['tools']) or None,
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
                count = len(tokenizer(prompt, add_special_tokens=False)['input_ids'])
                if count + item['maxNewTokens'] > run['config']['maxLength']:
                    raise ValueError('Conversation exceeds trained context; do not truncate')
                result = {'prompt': prompt, 'inputTokens': count, 'eos': tokenizer.eos_token}
            elif item['action'] == 'parse':
                result = {'calls': extract_tool_calls(item['text'], item['tools'])}
            elif item['action'] == 'policy':
                result = {'policy': POLICY}
            else:
                raise ValueError('Unknown worker action')
        except Exception as error:
            result = {'error': type(error).__name__ + ': ' + str(error)}
        print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
