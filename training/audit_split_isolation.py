"""Offline leakage audit only: never generates or edits training examples."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata


def requests(row):
    return [m['content'].strip() for m in row.get('messages',[]) if m['role']=='user']


def tokens(text):
    text=unicodedata.normalize('NFKC',text).casefold()
    text=re.sub(r'[+-]?\d+(?:[.,]\d+)?',' # ',text)
    if re.search(r'[\u3040-\u30ff\u4e00-\u9fff]',text):
        return {text[i:i+3] for i in range(len(text)-2)}
    return set(re.findall(r'\w+|#',text))


def audit(train, splits, threshold=.9):
    groups={r['group'] for r in train}
    linked=[];exact=[];near=[];numeric=[];historical_near=0;historical_examples=[]
    index=defaultdict(list)
    for split,rows in splits.items():
        linked += [{'split':split,'group':g} for g in sorted(groups&{r.get('group') for r in rows if r.get('group')})]
        for row in rows:
            for text in requests(row): index[row['locale']].append((split,row['id'],text,tokens(text)))
    for row in train:
        for text in requests(row):
            normalized=unicodedata.normalize('NFKC',text).casefold()
            current=tokens(text)
            for split,identifier,other,words in index[row['locale']]:
                entry={'trainId':row['id'],'split':split,'heldOutId':identifier}
                if normalized==unicodedata.normalize('NFKC',other).casefold(): exact.append(entry)
                elif len(current)>=6 and len(words)>=6:
                    similarity=len(current&words)/len(current|words)
                    is_new=row['id'].startswith(('reliability-','fidelity-'))
                    if similarity>=threshold:
                        if is_new: near.append({**entry,'similarity':similarity})
                        else:
                            historical_near+=1
                            if len(historical_examples)<50: historical_examples.append({**entry,'similarity':similarity})
                    if is_new and current==words:
                        numeric.append(entry)
    return {'linkedGroupLeaks':linked,'exactPromptLeaks':exact,'newNearDuplicates':near,'nearThreshold':threshold,
            'newNumericNormalizedMatches':numeric,'historicalNearDuplicatePairs':historical_near,
            'historicalNearDuplicateExamples':historical_examples,
            'linguisticReview':'pending; text similarity and translation-linked groups cannot prove semantic independence'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('Use a new evidence output')
    root=Path(__file__).resolve().parents[1]/'training/dataset'
    paths={s:root/'data'/f'{s}.jsonl' for s in ('validation','test','eval_gold','hard_gold')}
    paths.update(development=root/'validation-chat-v7.jsonl',future_holdout=root/'holdout-v8.jsonl')
    paths.update({s+'_audited':root/'data/audited'/f'{s}.jsonl' for s in ('validation','test')})
    paths['baseline_gold']=root/'baseline/gold.jsonl'
    if (root/'holdout-v9.jsonl').exists(): paths['future_holdout_v9']=root/'holdout-v9.jsonl'
    read=lambda p:[json.loads(line) for line in p.read_text(encoding='utf-8').splitlines()]
    result=audit(read(root/'train.jsonl'),{s:read(p) for s,p in paths.items()})
    result['inputHashes']={s:hashlib.sha256(p.read_bytes()).hexdigest() for s,p in paths.items()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:len(result[k]) for k in ('linkedGroupLeaks','exactPromptLeaks','newNearDuplicates')}))
    raise SystemExit(bool(result['linkedGroupLeaks'] or result['exactPromptLeaks'] or result['newNearDuplicates']))


if __name__=='__main__':main()
