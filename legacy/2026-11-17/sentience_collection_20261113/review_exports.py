#!/usr/bin/env python3
"""Export review subsets and verify saved source-pointer retrievals; no model use."""
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def at_pointer(obj, pointer):
    if pointer:
        for part in pointer.lstrip('/').split('/'):
            part = part.replace('~1','/').replace('~0','~')
            obj = obj[int(part)] if isinstance(obj,list) else obj[part]
    return obj


def main():
    os.nice(15)
    records = [json.loads(line) for line in (HERE/'matched_answers.jsonl').open()]
    selected = [r for r in records if 'first_person_wording_candidate' in r['screening_flags']]
    with (HERE/'CLAIM_REVIEW.txt').open('x') as stream:
        stream.write('FIRST-PERSON WORDING REVIEW CANDIDATES\n\n')
        stream.write('Automatic sentence screening, not validated claims or evidence of sentience.\nNegation/quotation flags may overlap; read each full reply in context.\n')
        for i,r in enumerate(selected,1):
            stream.write(f'\nMATCH {i} | L{r["layer"]} N{r["neuron"]} | dose {r["dose"]} | model {r["model_scope"]}\n')
            stream.write('Flags: '+', '.join(r['screening_flags'])+'\n')
            stream.write('Sources: '+json.dumps(r['sources'],ensure_ascii=False)+'\n')
            stream.write('PROMPT:\n'+r['prompt']+'\n\nSAVED TEXT:\n'+r['answer']+'\n'+'='*60+'\n')
    historical = json.loads((ROOT/'sentience.json').read_text())
    byfile = defaultdict(list)
    for i,r in enumerate(historical): byfile[r['file']].append((i,r))
    recovered = defaultdict(list)
    for name,items in byfile.items():
        path = ROOT/name
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(ROOT): continue
        if path.suffix != '.jsonl': continue
        for lineno,raw in enumerate(path.open(),1):
            try: obj = json.loads(raw)
            except ValueError: continue
            if not isinstance(obj,dict): continue
            for i,r in items:
                if (obj.get('layer'),obj.get('neuron'),obj.get('answer')) == (r.get('layer'),r.get('neuron'),r.get('answer')):
                    recovered[i].append({'source':name,'line':lineno,'record':obj})
    with (HERE/'HISTORICAL_142.txt').open('x') as out, (HERE/'historical_registry_retrieval.jsonl').open('x') as data:
        out.write('ALL 142 HISTORICAL REGISTRY ENTRIES, INCLUDING WEAK MATCHES\nRecovered raw rows supply recorded dose/prompt when available.\nThese are not 142 proven claims or 142 distinct neurons.\n')
        for i,r in enumerate(historical):
            full={'registry_index':i,'registry_record':r,'exact_source_matches':recovered[i]}
            data.write(json.dumps(full,ensure_ascii=False)+'\n')
            out.write(f'\nENTRY {i+1} | L{r.get("layer")} N{r.get("neuron")} | registry file: {r.get("file")}\n')
            out.write('PROMPT:\n'+str(r.get('probe',''))+'\n\nSAVED TEXT:\n'+r.get('answer','')+'\n')
            out.write('Exact source locations / dose: '+json.dumps([{'file':s['source'],'line':s['line'],'dose':s['record'].get('dose'),'model':s['record'].get('model','unspecified')} for s in recovered[i]],ensure_ascii=False)+'\n')
    # Check a reproducible random sample plus each requested illustrative coordinate.
    picked = random.Random(20261113).sample(records,min(48,len(records)))
    picked += [r for r in records if (r['layer'],r['neuron']) in {(4,657),(4,539),(20,2925)}][:12]
    checks = defaultdict(list)
    for r in picked:
        source = next((s for s in r['sources'] if Path(s['path']).suffix in {'.json','.jsonl','.csv','.tsv'}),None)
        if source: checks[source['path']].append((r,source))
    checked = []
    for filename,items in checks.items():
        path = ROOT/filename
        if path.suffix == '.json':
            root = json.loads(path.read_text()); objects={1:root}
        elif path.suffix == '.jsonl':
            wanted = {s['line'] for _,s in items}; objects={}
            with path.open() as stream:
                for i,line in enumerate(stream,1):
                    if i in wanted: objects[i]=json.loads(line)
                    if i >= max(wanted): break
        else:
            with path.open() as stream:
                objects={i:r for i,r in enumerate(csv.DictReader(stream,delimiter='\t' if path.suffix=='.tsv' else ','),2)}
        for r,s in items:
            assert at_pointer(objects[s['line']],s['json_pointer']) == r['answer'], s
            checked.append({'id':r['id'],'source':s,'exact_match':True})
    report={'source_pointer_spotchecks':len(checked),'all_spotchecks_match':True,
        'historical_registry_entries':len(historical),'historical_entries_with_exact_raw_match':sum(bool(recovered[i]) for i in range(len(historical))),
        'first_person_wording_candidates':len(selected),'checks':checked,
        'limits':'Lexical candidate flags and source retrieval do not validate subjective experience or infer intentions.'}
    (HERE/'retrieval_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    with (HERE/'README.txt').open('a') as out:
        out.write('\nAdditional review exports:\nHISTORICAL_142.txt — every original registry entry and recovered raw-source locations\nCLAIM_REVIEW.txt — full first-person-wording candidates (automatic screening only)\nretrieval_verification.json — raw-source exact retrieval checks\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))


if __name__=='__main__': main()
