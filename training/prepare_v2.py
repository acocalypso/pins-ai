"""Export audited train/validation only, using the pinned local Qwen3 tokenizer."""
import argparse
from collections import Counter
import json
from pathlib import Path
from common import ROOT, fingerprint, read_jsonl, sha256, write_json, write_jsonl
from v2_contracts import PACKAGE, VERIFIED, ADAPTER, POLICY, assistant_example


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=ROOT / '.local/training/base')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/v2/sft')
    parser.add_argument('--max-tokens', type=int, default=0)
    parser.add_argument('--curriculum', action='store_true', help='Runtime-aligned natural chat curriculum; original split files stay unchanged')
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False)
    report = {'templateSha256': fingerprint(tokenizer.chat_template), 'policySha256': fingerprint(POLICY),
              'contractSha256': sha256(VERIFIED), 'auditSha256': sha256(PACKAGE / 'data/contract-audit.json'),
              'adapterContractSha256': sha256(ADAPTER),
              'maxTokens': args.max_tokens, 'skipped': 0, 'splits': {}}
    if args.curriculum:
        from chat_curriculum import PROFILE, OUTPUT_RESERVE, curriculum_rows, convert_rows
        from chat_dataset_common import CHAT_POLICY
        report.update(sourceProfile=PROFILE, curriculumSha256=sha256(ROOT / 'training/chat_curriculum.py'),
                      legacyPolicySha256=fingerprint(POLICY), policySha256=fingerprint(CHAT_POLICY))
        report['chatManifestSha256'] = sha256(ROOT / 'training/datasets/mcp-chat/manifest.json')
        report['deduplicateSupervision'] = True
        report['inferenceMaxOutputTokens'] = OUTPUT_RESERVE
        report['comprehensiveManifestSha256'] = sha256(ROOT / 'training/datasets/mcp-comprehensive/manifest.json')
    groups = set()
    for split in ('train', 'validation'):
        source = PACKAGE / 'data/audited' / f'{split}.jsonl'
        rows = curriculum_rows(split) if args.curriculum else read_jsonl(source)
        current_groups = {r['group'] for r in rows}
        if groups & current_groups:
            raise ValueError('Train/validation scenario leak')
        groups.update(current_groups)
        converted = (convert_rows(tokenizer, rows, args.max_tokens, deduplicate=split == 'train') if args.curriculum else
                     [assistant_example(tokenizer, row, turn, args.max_tokens) for row in rows
                      for turn, message in enumerate(row['messages']) if message['role'] == 'assistant'])
        lengths = sorted(len(r['input_ids']) for r in converted)
        completions = sorted(sum(v != -100 for v in r['labels']) for r in converted)
        calls = Counter(call['function']['name'] for row in rows for m in row['messages'] for call in m.get('tool_calls', []))
        if len(calls) != 32 or len({r['locale'] for r in converted}) != 10:
            raise ValueError('Lost required tool/language coverage')
        path = args.output / f'{split}.jsonl'
        write_jsonl(path, converted)
        report['splits'][split] = {'conversations': len(rows), 'assistantTurns': len(converted),
                'duplicateAssistantTurnsRemoved': sum(m['role'] == 'assistant' for row in rows for m in row['messages']) - len(converted),
                'sourceSha256': sha256(source), 'sha256': sha256(path), 'languages': dict(Counter(r['locale'] for r in converted)),
                'sourceRowsSha256': fingerprint(rows),
                'tools': dict(calls), 'tokens': {'min': lengths[0], 'median': lengths[len(lengths)//2],
                                              'p95': lengths[int(.95*(len(lengths)-1))], 'max': lengths[-1]},
                'completionTokens': {'median': completions[len(completions)//2],
                                     'p95': completions[int(.95*(len(completions)-1))], 'max': completions[-1]}}
    write_json(args.output / 'manifest.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
