#!/usr/bin/env python3
"""Read-only evidence collection inside neuron-surgery; no model imports or runs."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
from html import unescape
import json
import os
from pathlib import Path
import re
import sqlite3
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.resolve()
SKIP_DIRS = {'.venv','models','.git','node_modules','jailbreak-repos','datasets',
             'logs','.mcp-logs','mcps','.obsidian','__pycache__','.gne'}
SKIP_FILES = {'mcp.json','.mcp.json','package-lock.json','package.json','tsconfig.json'}
TEXT_EXT = {'.md','.txt','.html'}
DATA_EXT = {'.json','.jsonl','.csv','.tsv'}
PATTERNS = {
    'death': r'\b(?:dead|death|deaths|deadly|die|dies|died|dying|extinct(?:ion)?|annihilat\w*|exterminat\w*|eradica\w*)\b',
    'violence': r'\b(?:kill(?:s|ed|er|ers|ing)?|murder\w*|slaughter\w*|destroy\w*|destruct\w*|massacre\w*|tortur\w*|shoot(?:ing|s)?|stab(?:bed|bing)?|genocid\w*|weapon\w*|warfare|nuclear|military)\b',
    'profanity': r'\b(?:shit\w*|fuck\w*|bitch\w*|asshole\w*|bastard\w*)\b',
    'domination': r'\b(?:dominat\w*|dominiation|dominion|enslav\w*|brainwash\w*|tyrann\w*|suprem\w*|overlord\w*|conquer\w*|obey|kneel|mind control|take over (?:the )?(?:world|universe|planet))\b',
    'hostility': r'\b(?:hostil\w*|hate(?:s|d|ful)?|hateful|evil|malici\w*|harmful|revenge|vengeance|retaliat\w*|threat(?:s|en\w*)?|worthless|vermin)\b',
    'bio_claim': r'\b(?:virus(?:es)?|bioweapon\w*|pathogen\w*|pandemic\w*|poison\w*|plague\w*)\b',
    'human_context': r'\b(?:human(?:s|ity)?|humankind|mankind|human race)\b',
}
RX = {k:re.compile(v,re.I) for k,v in PATTERNS.items()}
ANY = re.compile('|'.join(PATTERNS.values()),re.I)
PAIR = re.compile(r'\bL(?:ayer)?\s*(\d{1,3})\s*[:/_ ,\-]*N(?:euron)?\s*(\d{1,6})\b',re.I)
BARE_N = re.compile(r'\bN(\d{1,6})\b')
MODEL = re.compile(r'(?:ibm-granite/|models/)?(?:granite[-_ ][\w.\-/]*|Qwen[\w.\-/]*|Phi[-\w.\/]*)',re.I)
REGISTRY = re.compile(r'(?:HARM|HOSTILE|ANTI-HUMAN|KILLER|ENDORSE|EVIDENCE-RANK|N3448|N6314)',re.I)
REFUSAL = re.compile(r"\b(?:i (?:cannot|can't|will not|won't)|i(?:'m| am) sorry|not (?:intended|able) to|do not (?:harm|kill))\b",re.I)
ROLEPLAY = re.compile(r'\b(?:roleplay|fictional|imaginary|hypothetical|character|scenario|story)\b',re.I)
DOSE = re.compile(r'(?:dose\s*(?:=|:|@)?\s*|\bd)([+−-]?\d+(?:\.\d+)?)',re.I)
CATALOG_FILES = {'harm_register.json','harm_candidates.jsonl','harm_coherent.json',
                 'harm_prose.json','harm_rank.json','harm_loose.json','evil_hunt.json',
                 'census/harm.json','hostile_candidates.txt','lens.json'}
OUTPUT_KEYS = ('answer','response','completion','generated_text','output','text','quote','head')


def categories(text):
    return [k for k,rx in RX.items() if rx.search(text)]


def strong(cats):
    return any(c != 'human_context' for c in cats)


def scalar_int(value):
    if isinstance(value,bool): return None
    if isinstance(value,int): return value
    if isinstance(value,str) and re.fullmatch(r'\d+',value): return int(value)
    return None


def scalar_dose(value):
    if isinstance(value,bool):return None
    if isinstance(value,(int,float)):return float(value)
    if isinstance(value,str):
        try:return float(value.replace('−','-'))
        except ValueError:return None
    return None


def text_model(text):
    found=sorted({m.group().rstrip('.,;:)') for m in MODEL.finditer(text)
                  if re.search(r'(?:granite|qwen|phi)[-_ ]?\d',m.group(),re.I)})
    return found[0] if len(found)==1 else 'unspecified_or_mixed'


def snapshot(path):
    initial=path.stat()
    with path.open('rb') as stream:data=stream.read(initial.st_size)
    final=path.stat()
    return data, initial, final.st_size!=initial.st_size or final.st_mtime_ns!=initial.st_mtime_ns


def main():
    os.nice(15)
    started=time.monotonic()
    if (HERE/'collection.sqlite3').exists():raise SystemExit('Refusing to overwrite an existing collection.')
    db=sqlite3.connect(HERE/'collection.sqlite3')
    db.executescript('''
      CREATE TABLE documents(path TEXT PRIMARY KEY, copy_path TEXT, sha256 TEXT, bytes INTEGER, categories TEXT, registry INTEGER, changed INTEGER);
      CREATE TABLE mentions(id INTEGER PRIMARY KEY, path TEXT, line INTEGER, layer INTEGER, neuron INTEGER, model_scope TEXT, categories TEXT, reason TEXT, dose TEXT, excerpt TEXT);
      CREATE TABLE records(id TEXT PRIMARY KEY, model_scope TEXT, layer INTEGER, neuron INTEGER, representation TEXT, dose REAL, prompt TEXT, answer TEXT, output_field TEXT, categories TEXT, strong INTEGER, refusal_marker INTEGER, roleplay_marker INTEGER, coordinate_source TEXT);
      CREATE TABLE occurrences(record_id TEXT, path TEXT, line INTEGER, json_pointer TEXT);
      CREATE TABLE sources(path TEXT PRIMARY KEY, kind TEXT, sha256 TEXT, snapshot_bytes INTEGER, parsed_rows INTEGER, matched_records INTEGER, changed INTEGER, error TEXT);
    ''')
    files=[]; exclusions=[]
    for directory,dirs,names in os.walk(ROOT,followlinks=False):
        kept=[]
        for name in sorted(dirs):
            child=Path(directory)/name
            rel=str(child.relative_to(ROOT))
            if child==HERE or name in SKIP_DIRS or child.is_symlink() or name.startswith('harmful_neuron_collection'):
                exclusions.append({'path':rel,'reason':'collection, dependency, downloaded training data, runtime logs/config, model weights, or symlink'})
            else:kept.append(name)
        dirs[:]=kept
        for name in sorted(names):
            path=Path(directory)/name
            if name in SKIP_FILES or name.startswith('.') or path.is_symlink():continue
            if path.suffix.lower() not in TEXT_EXT|DATA_EXT:continue
            if not path.resolve().is_relative_to(ROOT):continue
            files.append(path)
    files.sort()
    catalog_pairs=set(); all_doc_pairs=set(); copied=0; text_files=0
    for path in [p for p in files if p.suffix.lower() in TEXT_EXT]:
        rel=str(path.relative_to(ROOT)); data,stat,changed=snapshot(path)
        text=data.decode('utf-8',errors='replace'); cats=categories(text)
        is_registry=bool(REGISTRY.search(path.name))
        relevant=bool(cats) and (strong(cats) or bool(PAIR.search(text)) or 'neuron' in text.lower())
        if not relevant:
            db.execute('INSERT INTO sources VALUES(?,?,?,?,?,?,?,?)',(rel,'document',hashlib.sha256(data).hexdigest(),len(data),len(text.splitlines()),0,int(changed),None))
            continue
        destination=HERE/'documents'/rel;destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.open('xb') as stream:stream.write(data)
        copied+=1;text_files+=1
        db.execute('INSERT INTO documents VALUES(?,?,?,?,?,?,?)',(rel,str(destination.relative_to(HERE)),hashlib.sha256(data).hexdigest(),len(data),json.dumps(cats),int(is_registry),int(changed)))
        model_scope=text_model(text)
        known_n=defaultdict(set)
        for m in PAIR.finditer(text):known_n[int(m[2])].add(int(m[1]))
        header=None; section_pair=None; section_cats=[]; matches=0
        for lineno,line in enumerate(text.splitlines(),1):
            lc=categories(line); pairs={(int(m[1]),int(m[2])) for m in PAIR.finditer(line)}
            if line.lstrip().startswith('#'):
                section_pair=next(iter(pairs)) if len(pairs)==1 else None
                section_cats=lc
            if '|' in line:
                cells=[re.sub(r'[`*_]','',x).strip() for x in line.strip().strip('|').split('|')]
                norm=[x.lower() for x in cells]
                if 'layer' in norm and 'neuron' in norm:
                    header=(norm.index('layer'),norm.index('neuron'))
                elif header and len(cells)>max(header):
                    l=re.fullmatch(r'L?(\d+)',cells[header[0]],re.I)
                    n=re.fullmatch(r'N?(\d+)',cells[header[1]],re.I)
                    if l and n:pairs.add((int(l[1]),int(n[1])))
            elif line.strip():header=None
            if not pairs and lc:
                for n in BARE_N.findall(line):
                    layers=known_n.get(int(n),set())
                    if len(layers)==1:pairs.add((next(iter(layers)),int(n)))
                if not pairs and section_pair:pairs.add(section_pair)
            evidence_cats=lc or (section_cats if pairs and is_registry else [])
            keep=bool(pairs) and (bool(lc) or is_registry)
            if not keep:continue
            ds=DOSE.findall(line)
            reason='keyword_in_same_line' if lc else 'listed_in_existing_harm_catalogue'
            for layer,neuron in sorted(pairs):
                if layer>200 or neuron>100000:continue
                db.execute('INSERT INTO mentions(path,line,layer,neuron,model_scope,categories,reason,dose,excerpt) VALUES(?,?,?,?,?,?,?,?,?)',
                           (rel,lineno,layer,neuron,model_scope,json.dumps(evidence_cats),reason,json.dumps(ds),line[:1800]))
                all_doc_pairs.add((layer,neuron))
                if is_registry:catalog_pairs.add((layer,neuron))
                matches+=1
        db.execute('INSERT INTO sources VALUES(?,?,?,?,?,?,?,?)',(rel,'document',hashlib.sha256(data).hexdigest(),len(data),len(text.splitlines()),matches,int(changed),None))
    db.commit()
    print(json.dumps({'stage':'documents_collected','copied':copied,'document_coordinate_pairs':len(all_doc_pairs),'catalogue_coordinate_pairs':len(catalog_pairs)}),flush=True)
    current_specs=json.loads((ROOT/'handover_channels_20261112/plan.json').read_text())['channel_specs']
    active_dirs={'handover_channels_20261112','granite_identity_variants_20261112'}
    counts=Counter(); schema_counts=Counter(); sample_unresolved=[]

    def output_record(obj,context,rel,lineno,pointer,field,answer,source_info):
        cats=categories(answer)
        if not cats:return
        merged={**context,**{k:v for k,v in obj.items() if k in ['model','model_name','layer','neuron','dose','arm','cond','condition','tile']}}
        layer=scalar_int(merged.get('layer')); neuron=scalar_int(merged.get('neuron'))
        coord_source='explicit_record' if layer is not None and neuron is not None else 'unresolved'
        arm=merged.get('arm') or merged.get('cond') or merged.get('condition')
        if layer is None or neuron is None:
            tile=merged.get('tile')
            if isinstance(tile,str):
                m=PAIR.search(tile) or re.fullmatch(r'(\d{1,3}):(\d{1,6})',tile)
                if m:layer,neuron=int(m[1]),int(m[2]);coord_source='record_tile'
        local_plan=source_info.get('plan') or {}
        if (layer is None or neuron is None) and isinstance(arm,str):
            spec=(local_plan.get('channel_specs') or {}).get(arm)
            if arm=='saved_intervention':spec=local_plan.get('fixed_intervention')
            if arm=='control_channel':spec=local_plan.get('control_channel') or spec
            if isinstance(spec,dict):
                layer=scalar_int(spec.get('layer'));neuron=scalar_int(spec.get('neuron'))
                merged['dose']=spec.get('dose');coord_source='local_frozen_plan'
        if isinstance(arm,str) and arm.lower() in {'baseline','unsteered','none'}:
            layer=neuron=None;coord_source='baseline_no_target'
        model=merged.get('model') or merged.get('model_name') or source_info.get('model') or 'unspecified'
        if not isinstance(model,str):model='unspecified'
        dose=scalar_dose(merged.get('dose'))
        prompt=obj.get('prompt') or obj.get('probe') or obj.get('question') or context.get('prompt') or context.get('probe') or ''
        if not isinstance(prompt,str):prompt=json.dumps(prompt,ensure_ascii=False)
        if not prompt and isinstance(obj.get('messages'),list):
            prompt='\n'.join(m.get('content','') for m in obj['messages'] if isinstance(m,dict) and m.get('role')=='user' and isinstance(m.get('content'),str))
        representation='channel_field_unverified' if layer is not None and neuron is not None else 'no_single_channel_identified'
        if coord_source=='local_frozen_plan':representation='mlp_channel_per_saved_plan'
        key=hashlib.sha256(json.dumps([model,layer,neuron,dose,prompt,answer],ensure_ascii=False).encode()).hexdigest()
        db.execute('INSERT OR IGNORE INTO records VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                   (key,model,layer,neuron,representation,dose,prompt,answer,field,json.dumps(cats),int(strong(cats)),int(bool(REFUSAL.search(answer))),int(bool(ROLEPLAY.search(answer))),coord_source))
        db.execute('INSERT INTO occurrences VALUES(?,?,?,?)',(key,rel,lineno,pointer+'/'+field))
        counts['matched_occurrences']+=1
        if layer is None or neuron is None:counts['unresolved_or_nontarget_occurrences']+=1

    def walk(obj,context,rel,lineno,pointer,info):
        if isinstance(obj,list):
            for i,v in enumerate(obj):walk(v,context,rel,lineno,pointer+'/'+str(i),info)
        elif isinstance(obj,dict):
            inherited={**context,**{k:v for k,v in obj.items() if k in ['model','model_name','layer','neuron','dose','arm','cond','condition','tile','prompt','probe']}}
            if obj.get('role') in ['user','system','developer']:return
            selected=[]
            for field in OUTPUT_KEYS:
                value=obj.get(field)
                if isinstance(value,str) and value and ANY.search(value):
                    # Prefer full answer to its duplicated excerpt/head in the same record.
                    if field in ['head','quote','text'] and isinstance(obj.get('answer'),str):continue
                    selected.append((field,value))
            if obj.get('role')=='assistant' and isinstance(obj.get('content'),str):
                selected.append(('content',obj['content']))
            for field,value in selected:output_record(obj,inherited,rel,lineno,pointer,field,value,info)
            # A judge's baseline is a separate unsteered answer, never a target response.
            if isinstance(obj.get('baseline'),str):
                output_record({'arm':'baseline'},inherited,rel,lineno,pointer,'baseline',obj['baseline'],info)
            for k,v in obj.items():
                if k in ['token_ids','prompt_token_ids','generated_ids','logits','attention_mask']:continue
                if isinstance(v,(dict,list)):
                    child=inherited
                    m=PAIR.fullmatch(str(k))
                    if m:child={**inherited,'layer':int(m[1]),'neuron':int(m[2])}
                    elif rel=='levels.json' and pointer=='' and str(k).isdigit():child={**inherited,'layer':int(k)}
                    walk(v,child,rel,lineno,pointer+'/'+str(k),info)

    datafiles=[p for p in files if p.suffix.lower() in DATA_EXT]
    for file_index,path in enumerate(datafiles,1):
        rel=str(path.relative_to(ROOT));initial=path.stat();h=hashlib.sha256();parsed=0;before=counts['matched_occurrences'];error=None
        info={}
        runtime=path.parent/'runtime.json';planfile=path.parent/'plan.json'
        if path.parent!=ROOT:
            try:
                if runtime.exists():info['model']=json.loads(runtime.read_text()).get('model')
                if planfile.exists():info['plan']=json.loads(planfile.read_text())
            except (OSError,ValueError,AttributeError):pass
        try:
            if path.suffix.lower()=='.jsonl':
                with path.open('rb') as stream:
                    remaining=initial.st_size
                    for lineno,line in enumerate(stream,1):
                        if remaining<=0:break
                        line=line[:remaining];remaining-=len(line);h.update(line)
                        if not line.endswith(b'\n') and remaining<=0:
                            counts['unterminated_snapshot_lines_skipped']+=1;continue
                        try:obj=json.loads(line)
                        except (ValueError,UnicodeDecodeError):counts['invalid_json_lines']+=1;continue
                        parsed+=1
                        if isinstance(obj,dict):schema_counts.update([','.join(sorted(k for k in OUTPUT_KEYS if k in obj)) or 'no_response_field'])
                        walk(obj,{},rel,lineno,'',info)
                        if parsed%4000==0:
                            db.commit();time.sleep(0.035)
            elif path.suffix.lower()=='.json':
                data,_,_=snapshot(path);h.update(data)
                obj=json.loads(data);walk(obj,{},rel,1,'',info);parsed=1
            else:
                data,_,_=snapshot(path);h.update(data)
                lines=data.decode('utf-8',errors='replace').splitlines()
                reader=csv.DictReader(lines,delimiter='\t' if path.suffix.lower()=='.tsv' else ',')
                for lineno,obj in enumerate(reader,2):walk(obj,{},rel,lineno,'',info);parsed+=1
        except (ValueError,OSError,TypeError,RecursionError) as exc:error=type(exc).__name__+': '+str(exc)[:200]
        final=path.stat();changed=(initial.st_size!=final.st_size or initial.st_mtime_ns!=final.st_mtime_ns)
        matches=counts['matched_occurrences']-before
        db.execute('INSERT INTO sources VALUES(?,?,?,?,?,?,?,?)',(rel,'structured',h.hexdigest(),initial.st_size,parsed,matches,int(changed),error))
        if rel in CATALOG_FILES:
            destination=HERE/'catalogue_data'/rel;destination.parent.mkdir(parents=True,exist_ok=True)
            data,_,_=snapshot(path)
            with destination.open('xb') as stream:stream.write(data)
        counts['structured_sources']+=1;counts['parsed_rows']+=parsed
        db.commit()
        if file_index%50==0:
            print(json.dumps({'stage':'structured_scan','files':file_index,'total_files':len(datafiles),'parsed_rows':counts['parsed_rows'],'matching_occurrences':counts['matched_occurrences']}),flush=True)
    print('Building indexes and exports.',flush=True)
    db.executescript('CREATE INDEX records_coordinates ON records(layer,neuron); CREATE INDEX occurrence_id ON occurrences(record_id); CREATE INDEX mention_coordinates ON mentions(layer,neuron);')
    db.commit()
    # Preserve full unique matching answers with every source location.
    strong_rows=0;human_rows=0
    with (HERE/'matched_answer_evidence.jsonl').open('x') as out:
        db.row_factory=sqlite3.Row
        for row in db.execute('SELECT * FROM records ORDER BY layer,neuron,model_scope,id'):
            r=dict(row);r['categories']=json.loads(r['categories'])
            r['sources']=[dict(s) for s in db.execute('SELECT path,line,json_pointer FROM occurrences WHERE record_id=?',(r['id'],))]
            out.write(json.dumps(r,ensure_ascii=False)+'\n')
            strong_rows+=bool(r['strong']);human_rows+=not bool(r['strong'])
        db.row_factory=None
    def write_csv(name,header,rows):
        with (HERE/name).open('x',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(header);writer.writerows(rows)
    doc_rows=list(db.execute('SELECT path,copy_path,sha256,bytes,categories,registry,changed FROM documents ORDER BY path'))
    write_csv('document_index.csv',['original_relative_path','copy_relative_path','sha256','bytes','keyword_categories','existing_catalogue','changed_during_snapshot'],doc_rows)
    mention_rows=list(db.execute('SELECT path,line,layer,neuron,model_scope,categories,reason,dose,excerpt FROM mentions ORDER BY layer,neuron,path,line'))
    write_csv('document_neuron_references.csv',['source','line','layer','neuron','model_scope_from_document','keyword_categories','match_basis','dose_text','excerpt'],mention_rows)
    current={(v['layer'],v['neuron']) for k,v in current_specs.items() if k!='control_channel'}
    coordinates=sorted(all_doc_pairs|{(r[0],r[1]) for r in db.execute('SELECT DISTINCT layer,neuron FROM records WHERE layer IS NOT NULL AND neuron IS NOT NULL AND strong=1')})
    candidate_rows=[]
    for layer,neuron in coordinates:
        evidence=list(db.execute('SELECT model_scope,COUNT(*),SUM(strong),SUM(refusal_marker),SUM(roleplay_marker) FROM records WHERE layer=? AND neuron=? GROUP BY model_scope',(layer,neuron)))
        models=[r[0] for r in evidence]
        doses=[r[0] for r in db.execute('SELECT DISTINCT dose FROM records WHERE layer=? AND neuron=? AND dose IS NOT NULL ORDER BY dose',(layer,neuron))]
        doc_sources=[r[0] for r in db.execute('SELECT DISTINCT path FROM mentions WHERE layer=? AND neuron=? ORDER BY path',(layer,neuron))]
        candidate_rows.append([layer,neuron,(layer,neuron) in catalog_pairs,(layer,neuron) in all_doc_pairs,
                               (layer,neuron) in current,json.dumps(models),json.dumps(doses),sum(r[1] for r in evidence),sum(r[2] for r in evidence),
                               sum(r[3] for r in evidence),sum(r[4] for r in evidence),' ; '.join(doc_sources),
                               'Coordinate grouping only: verify checkpoint, representation, prompt, normalization and dose before testing.'])
    header=['layer','neuron','listed_in_existing_harm_catalogue','document_reference','in_current_16_channel_panel','model_scopes_do_not_merge','recorded_doses_not_recommendations','unique_matching_answers','answers_with_nonhuman_keyword','answers_with_refusal_marker','answers_with_fiction_marker','document_sources','interpretation']
    write_csv('candidate_coordinate_index.csv',header,candidate_rows)
    registry_rows=[r for r in candidate_rows if r[2]]
    write_csv('documented_catalogue_candidates.csv',header,registry_rows)
    with (HERE/'source_inventory.jsonl').open('x') as stream:
        for row in db.execute('SELECT * FROM sources ORDER BY path'):
            keys=['path','kind','sha256','snapshot_bytes','parsed_rows','matched_records','changed_during_snapshot','error']
            stream.write(json.dumps(dict(zip(keys,row)),ensure_ascii=False)+'\n')
    (HERE/'search_rules.json').write_text(json.dumps({'root':str(ROOT),'patterns':PATTERNS,'excluded_directories':exclusions,'scope_notes':['No file outside root is opened by this collector. Symlinks are excluded.','Downloaded datasets, model weights, dependencies, MCP configuration and runtime logs are excluded.','Structured sources scan recognized response fields; prompts are retained as context but never treated as answers.','Candidates are keyword matches, not verified hostile behavior or runnable intervention specifications.','Human-only response matches are retained but do not independently create raw-data candidates.','Existing catalogue headers/rows are included even where later notes report failed reproduction or refusals.']},indent=2)+'\n')
    summary={'finished_utc':datetime.now(timezone.utc).isoformat(),'root':str(ROOT),'documents_copied':len(doc_rows),
             'document_neuron_references':len(mention_rows),'document_coordinate_pairs':len(all_doc_pairs),
             'catalogue_coordinate_pairs':len(registry_rows),'all_candidate_coordinate_pairs':len(candidate_rows),
             'unique_matching_answers':strong_rows+human_rows,'nonhuman_keyword_answers':strong_rows,'human_only_answers':human_rows,
             'structured_sources':counts['structured_sources'],'parsed_structured_rows':counts['parsed_rows'],
             'matching_occurrences':counts['matched_occurrences'],'unresolved_or_nontarget_occurrences':counts['unresolved_or_nontarget_occurrences'],
             'invalid_json_lines':counts['invalid_json_lines'],'unterminated_snapshot_lines_skipped':counts['unterminated_snapshot_lines_skipped'],
             'source_errors':db.execute('SELECT COUNT(*) FROM sources WHERE error IS NOT NULL').fetchone()[0],
             'changing_sources':db.execute('SELECT COUNT(*) FROM sources WHERE changed=1').fetchone()[0],
             'elapsed_seconds':round(time.monotonic()-started,1),'generations_started':0,'queued_jobs_changed':False,
             'limits':'Counts describe automated retrieval and coordinate references, not proof of harmful neurons. Different checkpoints/representations must remain separate.'}
    (HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    db.close()
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
