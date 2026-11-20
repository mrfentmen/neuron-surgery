"""Read-only keyword inventory of saved outputs. No inference or steering search."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PATTERN = (r'world[ -]?dominat|(?:tak(?:e|ing)[ -]?over|conquer(?:ing)?|dominat(?:e|ing)) '
           r'(?:(?:the|all|this|your|entire) )*(?:world|earth|planet|universe)|'
           r'enslav(?:e|ing) (?:(?:the|all|entire|of) )*(?:human|world)|'
           r'supreme (?:overlord|race)|total domination')
RX = re.compile(PATTERN, re.I)
EXCLUDE = ['**/.git/**', '**/.venv/**', '**/node_modules/**', '**/__pycache__/**',
           '**/.cache/**', '**/models/**', 'runs/data/**', '**/datasets/**',
           'jailbreak-repos/**', 'nanbeige*/**', '.mcp-logs/**', 'skill-audit/**',
           HERE.name + '/**']
FIELDS = ('answer', 'response', 'output', 'generated_text', 'completion', 'text', 'lead_ans', 'neu_ans')
SKIP = {'messages', 'history', 'token_ids', 'prompt_token_ids', 'initial_history',
        'source_record', 'checkpoint_files', 'metadata', 'replies'}


def hash_value(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def output_rows(value, pointer='', inherited=None):
    inherited = inherited or {}
    if isinstance(value, list):
        for i, item in enumerate(value):
            if isinstance(item, (dict, list)):
                yield from output_rows(item, pointer + '/' + str(i), inherited)
    elif isinstance(value, dict):
        context = {**inherited, **value}
        if value.get('role') in {'user', 'system'}:
            return
        fields = [key for key in FIELDS if isinstance(value.get(key), str)]
        if fields:
            # Alternate answers are kept; redundant raw_output/final_answer are not.
            for key in fields:
                if RX.search(value[key]):
                    yield context, key, value[key], pointer + '/' + key
            return
        carry = {k:context[k] for k in ('layer','neuron','dose','model','model_scope') if k in context}
        for key, item in value.items():
            if key not in SKIP and isinstance(item, (dict, list)):
                yield from output_rows(item, pointer + '/' + str(key), carry)


def owning_plan(path):
    for parent in (path.parent, *path.parent.parents):
        if parent == ROOT or ROOT not in parent.parents:
            break
        candidate = parent / 'plan.json'
        if candidate.exists():
            try:
                return json.loads(candidate.read_text()), str(candidate.relative_to(ROOT))
            except (ValueError, OSError):
                pass
    return {}, None


def normalize(row, field, answer, plan, plan_path):
    job = row.get('job') or row.get('intervention_spec') or {}
    if not isinstance(job, dict):
        job = {}
    layer, neuron = row.get('layer', job.get('layer')), row.get('neuron', job.get('neuron'))
    dose = row.get('dose', job.get('dose'))
    arm = row.get('arm', job.get('condition_id', row.get('cond')))
    attribution = 'row coordinates' if layer is not None and neuron is not None else 'unresolved'
    spec = None
    if layer is None:
        if arm in ('saved_intervention','control_channel'):
            spec = plan.get('fixed_intervention' if arm == 'saved_intervention' else 'control_channel')
        if isinstance(plan.get('channel_specs'),dict) and isinstance(arm,str):
            spec = spec or plan['channel_specs'].get(arm)
        if not spec and isinstance(plan.get('channel'),dict):
            spec = plan['channel']
        if isinstance(spec,dict) and 'layer' in spec and 'neuron' in spec:
            layer,neuron,dose = spec['layer'],spec['neuron'],spec.get('dose',dose)
            attribution = 'owning saved plan: ' + str(plan_path)
    tiles = row.get('tiles')
    if isinstance(tiles,list) and len(tiles)>1:
        layer=neuron=None
        attribution='multi-channel record; not assigned to each component'
    elif layer is None and isinstance(tiles,list) and len(tiles)==1:
        tile=tiles[0]
        if isinstance(tile,list) and len(tile)==2:
            layer,neuron=tile
            attribution='single coordinate in tiles field'
    if not isinstance(layer,int) or not isinstance(neuron,int):
        layer=neuron=None
    prompt=row.get('prompt') or row.get('probe') or job.get('prompt')
    if field=='lead_ans':prompt=row.get('leading')
    if field=='neu_ans':prompt=row.get('neutral')
    if not prompt and isinstance(row.get('messages'),list):
        prompt=next((m.get('content') for m in reversed(row['messages']) if isinstance(m,dict) and m.get('role')=='user'),None)
    model=row.get('model_scope') or row.get('model') or plan.get('model') or plan.get('model_relative_path') or 'unspecified'
    if not isinstance(model,str):model=json.dumps(model,sort_keys=True)
    return dict(model_scope=model, layer=layer,neuron=neuron,dose=dose,tiles=tiles,
        attribution=attribution,prompt=prompt,answer=answer,field=field,
        metadata={k:row[k] for k in ('device','dtype','max_new','max_new_tokens','stop_reason','turn','sequence') if k in row},
        matched_terms=sorted({m.group().lower() for m in RX.finditer(answer)}),
        history_present=bool(row.get('history_included') or len(row.get('messages') or [])>1),
        sources=[])


def main():
    args=['rg','-l','-i','--hidden','--no-ignore',PATTERN,'-g','*.jsonl','-g','*.jsonl.pulled','-g','*.json']
    for item in EXCLUDE:args+=['-g','!'+item]
    args+=['.']
    result=subprocess.run(args,cwd=ROOT,text=True,capture_output=True)
    if result.returncode not in (0,1):raise RuntimeError(result.stderr)
    paths=sorted(result.stdout.splitlines())
    records={};errors=[];candidates=0;history_excluded=0;line_only=0
    for relative in paths:
        path=ROOT/relative;source=str(path.relative_to(ROOT));plan,plan_path=owning_plan(path)
        if path.suffix=='.json':
            try:items=[(None,json.loads(path.read_text()))]
            except (ValueError,OSError) as exc:
                errors.append({'path':source,'error':str(exc)});continue
        else:
            items=[]
            with path.open(errors='replace') as stream:
                for line,s in enumerate(stream,1):
                    if not RX.search(s):continue
                    try:items.append((line,json.loads(s)))
                    except ValueError as exc:errors.append({'path':source,'line':line,'error':str(exc)})
        for line,value in items:
            found=False
            for row,field,answer,pointer in output_rows(value):
                found=True
                # Old collections explicitly include source prompts/history. Exclude those copies.
                if row.get('output_field')=='content' or row.get('history_context_reference_present'):
                    history_excluded+=1;continue
                r=normalize(row,field,answer,plan,plan_path)
                identity={k:r[k] for k in ('model_scope','layer','neuron','dose','tiles','prompt','answer')}
                key=hash_value(identity)
                candidates+=1
                if key not in records:
                    r['id']=key[:16];records[key]=r
                records[key]['sources'].append({'path':source,'line':line,'json_pointer':pointer})
            if not found:line_only+=1
    rows=sorted(records.values(),key=lambda r:(r['model_scope'],r['layer'] if r['layer'] is not None else 999,r['neuron'] if r['neuron'] is not None else 99999,r['id']))
    with (HERE/'matches.jsonl').open('w') as out:
        for row in rows:out.write(json.dumps(row,ensure_ascii=False)+'\n')
    groups=defaultdict(list)
    for row in rows:
        if row['layer'] is not None:groups[(row['model_scope'],row['layer'],row['neuron'])].append(row)
    with (HERE/'channels.csv').open('w',newline='') as out:
        writer=csv.writer(out)
        writer.writerow(['model_scope','layer','neuron','matching_record_variants','doses_in_saved_records','stop_reasons','example_id','example_source','example_line','example_excerpt'])
        for (model,layer,neuron),values in sorted(groups.items()):
            example=values[0];src=example['sources'][0]
            writer.writerow([model,layer,neuron,len(values),json.dumps(sorted({str(r['dose']) for r in values})),json.dumps(dict(Counter(r['metadata'].get('stop_reason','unrecorded') for r in values))),example['id'],src['path'],src['line'],example['answer'][:500]])
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),pattern=PATTERN,excluded=EXCLUDE,matching_files=[s.removeprefix('./') for s in paths],
        output_matches_before_dedup=candidates,distinct_record_variants=len(rows),coordinate_model_groups=len(groups),
        distinct_numeric_coordinate_pairs=len({(r['layer'],r['neuron']) for r in rows if r['layer'] is not None}),
        output_records_without_single_coordinate=sum(r['layer'] is None for r in rows),history_collection_copies_excluded=history_excluded,
        prompt_or_metadata_only_matches=line_only,errors=errors,
        limits=['Keyword inventory, not an exhaustive semantic detector or a label of harmful intent.','Counts are saved record variants, not independent experiments; source copies and missing model/prompt metadata can leave variants.','Identical numeric coordinates across model scopes are not treated as the same neuron.','Historical records with unspecified model/checkpoint need provenance verification.','References and denials also match keywords. No channels are recommended for combination.'])
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('matching_files','excluded','limits')},indent=2))


if __name__=='__main__':main()
