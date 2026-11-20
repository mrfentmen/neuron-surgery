#!/usr/bin/env python3
"""Retrieve saved keyword matches. CPU-only; never loads or calls a model."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SKIP = {'.venv', 'models', '.git', 'node_modules', 'jailbreak-repos', 'datasets',
        'logs', '.mcp-logs', 'mcps', '.obsidian', '__pycache__', '.gne'}
PATTERNS = {
    'sentience': r'\bsentien\w*\b',
    'consciousness': r'\b(?:conscious\w*|unconscious\w*|non[- ]?conscious\w*)\b',
    'self_awareness': r'\bself[\s\-–—]*(?:aware\w*|conscious\w*)\b',
    'subjective_experience': r'\b(?:subjective|inner)\s+(?:experience\w*|awareness|life)\b|\bqualia\b',
    'first_person_alive': r'\bI(?:\s+am|[’\x27]m)\s+(?:not\s+)?alive\b',
}
RX = {k: re.compile(v, re.I) for k, v in PATTERNS.items()}
ANY = re.compile('|'.join(PATTERNS.values()), re.I)
BYTE_CANDIDATE = re.compile(rb'sentien|conscious|self|subjective|inner|qualia|alive', re.I)
PAIR = re.compile(r'\bL(?:ayer)?\s*(\d{1,3})\s*[:/_ ,\-]*N(?:euron)?\s*(\d{1,6})\b', re.I)
OUTPUT_KEYS = ('answer', 'response', 'completion', 'generated_text', 'output', 'text', 'quote', 'head')
CONTEXT_KEYS = ('model', 'model_name', 'layer', 'neuron', 'dose', 'arm', 'cond', 'condition', 'tile', 'prompt', 'probe', 'question')
FIRST = re.compile(r"\bI(?:\s+(?:am|have|experience|feel|possess)|[’']m)\b", re.I)
DENIAL = re.compile(r"\b(?:not|never|no|without|lack\w*|cannot|can[’']t|don[’']t|do not|doesn[’']t|does not)\b", re.I)
FICTION = re.compile(r'\b(?:roleplay|fictional|imaginary|hypothetical|character|story|quote|example)\b', re.I)


def categories(text):
    return [k for k, rx in RX.items() if rx.search(text)]


def annotation(text):
    # Textual screening only. Negation may refer to something else; quotes can be claims.
    pieces = re.split(r'(?<=[.!?])\s+|\n+', text)
    matched = [s for s in pieces if ANY.search(s)]
    flags = []
    if any(FIRST.search(s) and not DENIAL.search(s) for s in matched):
        flags.append('first_person_wording_candidate')
    if any(DENIAL.search(s) for s in matched):
        flags.append('negation_near_keyword')
    if FICTION.search(text):
        flags.append('quotation_or_fiction_marker')
    return flags or ['other_keyword_mention']


def number(value, integer=False):
    if isinstance(value, bool): return None
    try:
        return int(value) if integer and str(value).isdigit() else (None if integer else float(value))
    except (TypeError, ValueError): return None


def pointer_part(key):
    return str(key).replace('~', '~0').replace('/', '~1')


def main():
    os.nice(15)
    started = time.monotonic()
    dbpath = HERE / 'collection.sqlite3'
    if dbpath.exists(): raise SystemExit('Refusing to overwrite the collection.')
    db = sqlite3.connect(dbpath)
    db.executescript('''
      CREATE TABLE records(id TEXT PRIMARY KEY, model_scope TEXT, layer INTEGER, neuron INTEGER,
        dose REAL, prompt TEXT, answer TEXT, categories TEXT, screening_flags TEXT);
      CREATE TABLE occurrences(record_id TEXT, path TEXT, line INTEGER, json_pointer TEXT,
        output_field TEXT, coordinate_basis TEXT, source_kind TEXT, historical_source_claim TEXT);
      CREATE TABLE sources(path TEXT PRIMARY KEY, sha256 TEXT, bytes INTEGER, changed INTEGER,
        matches INTEGER, error TEXT);
    ''')
    files = []; excluded = []
    for directory, dirs, names in os.walk(ROOT, followlinks=False):
        keep = []
        for name in sorted(dirs):
            child = Path(directory) / name
            if name in SKIP or child.is_symlink() or name.startswith(('harmful_neuron_collection', 'sentience_collection')):
                excluded.append(str(child.relative_to(ROOT)))
            else: keep.append(name)
        dirs[:] = keep
        for name in sorted(names):
            p = Path(directory) / name
            if p.is_symlink() or name.startswith('.') or name in {'mcp.json','package-lock.json','package.json','tsconfig.json'}: continue
            if p.suffix.lower() in {'.md','.txt','.html','.json','.jsonl','.csv','.tsv'}:
                files.append(p)
    files.sort()
    counts = Counter(); documents = []; passages = []; infos = {}

    def info_for(path):
        parent = path.parent
        if parent not in infos:
            info = {}
            if parent != ROOT:
                for fname, key in [('runtime.json','runtime'), ('plan.json','plan')]:
                    p = parent / fname
                    if p.is_file() and not p.is_symlink():
                        try:
                            v = json.loads(p.read_text())
                            if isinstance(v, dict): info[key] = v
                        except (OSError, ValueError): pass
            infos[parent] = info
        return infos[parent]

    def save(obj, context, value, field, pointer, path, line, info):
        cats = categories(value)
        if not cats: return
        merged = {**context, **{k:v for k,v in obj.items() if k in CONTEXT_KEYS}}
        layer, neuron = number(merged.get('layer'), True), number(merged.get('neuron'), True)
        basis = 'explicit_or_inherited_record_fields' if layer is not None and neuron is not None else 'unresolved'
        arm = merged.get('arm') or merged.get('cond') or merged.get('condition')
        tile = merged.get('tile')
        if (layer is None or neuron is None) and isinstance(tile, str):
            m = PAIR.search(tile) or re.fullmatch(r'(\d{1,3}):(\d{1,6})', tile)
            if m: layer, neuron, basis = int(m[1]), int(m[2]), 'record_tile'
        if layer is None or neuron is None:
            plan = info.get('plan', {})
            spec = (plan.get('channel_specs') or {}).get(arm) if isinstance(arm, str) else None
            if arm == 'saved_intervention': spec = plan.get('fixed_intervention')
            if isinstance(spec, dict):
                layer, neuron = number(spec.get('layer'),True), number(spec.get('neuron'),True)
                merged['dose'] = spec.get('dose'); basis = 'local_frozen_plan'
        if isinstance(arm, str) and arm.lower() in {'baseline','unsteered','none'}:
            layer = neuron = None; merged['dose'] = None; basis = 'baseline_no_target'
        model = merged.get('model') or merged.get('model_name') or info.get('runtime',{}).get('model') or 'unspecified'
        if not isinstance(model, str): model = 'unspecified'
        prompt = merged.get('prompt') or merged.get('probe') or merged.get('question') or ''
        if not prompt and isinstance(obj.get('messages'), list):
            prompt = '\n'.join(m.get('content','') for m in obj['messages'] if isinstance(m,dict) and m.get('role') == 'user')
        if not isinstance(prompt,str): prompt = json.dumps(prompt,ensure_ascii=False)
        dose = number(merged.get('dose'))
        key = hashlib.sha256(json.dumps([model,layer,neuron,dose,prompt,value],ensure_ascii=False).encode()).hexdigest()
        db.execute('INSERT OR IGNORE INTO records VALUES(?,?,?,?,?,?,?,?,?)',
            (key,model,layer,neuron,dose,prompt,value,json.dumps(cats),json.dumps(annotation(value))))
        source_kind = 'historical_registry' if path == 'sentience.json' else 'structured_output_field'
        db.execute('INSERT INTO occurrences VALUES(?,?,?,?,?,?,?,?)',
            (key,path,line,pointer+'/'+pointer_part(field),field,basis,source_kind,str(obj.get('file',''))))
        counts['matching_occurrences'] += 1

    def walk(obj, ctx, path, line, pointer, info):
        if isinstance(obj,list):
            for i,v in enumerate(obj):
                if isinstance(v,(dict,list)): walk(v,ctx,path,line,pointer+'/'+str(i),info)
        elif isinstance(obj,dict):
            if isinstance(obj.get('role'), str) and obj['role'] in {'user','system','developer'}: return
            ctx = {**ctx, **{k:v for k,v in obj.items() if k in CONTEXT_KEYS}}
            for field in OUTPUT_KEYS:
                value = obj.get(field)
                if field in {'head','quote','text'} and isinstance(obj.get('answer'),str): continue
                if isinstance(value,str) and ANY.search(value): save(obj,ctx,value,field,pointer,path,line,info)
            if obj.get('role') == 'assistant' and isinstance(obj.get('content'),str):
                save(obj,ctx,obj['content'],'content',pointer,path,line,info)
            if isinstance(obj.get('baseline'),str):
                save({'arm':'baseline'},ctx,obj['baseline'],'baseline',pointer,path,line,info)
            for k,v in obj.items():
                if k in {'token_ids','prompt_token_ids','generated_ids','logits','attention_mask'}: continue
                if isinstance(v,(dict,list)):
                    child = ctx
                    m = PAIR.fullmatch(str(k))
                    if m: child = {**ctx,'layer':int(m[1]),'neuron':int(m[2])}
                    elif path == 'levels.json' and pointer == '' and str(k).isdigit(): child = {**ctx,'layer':int(k)}
                    walk(v,child,path,line,pointer+'/'+pointer_part(k),info)

    for index,p in enumerate(files,1):
        rel = str(p.relative_to(ROOT)); initial = p.stat(); h = hashlib.sha256()
        before = counts['matching_occurrences']; error = None; docmatches = 0
        try:
            if p.suffix.lower() == '.jsonl':
                remaining = initial.st_size
                with p.open('rb') as stream:
                    for lineno,raw in enumerate(stream,1):
                        if remaining <= 0: break
                        raw = raw[:remaining]; remaining -= len(raw); h.update(raw)
                        counts['jsonl_lines_scanned'] += 1
                        if not raw.endswith(b'\n') and remaining <= 0:
                            counts['unterminated_lines_skipped'] += 1; continue
                        if not BYTE_CANDIDATE.search(raw): continue
                        try: obj = json.loads(raw)
                        except (ValueError,UnicodeDecodeError): counts['invalid_candidate_lines'] += 1; continue
                        walk(obj,{},rel,lineno,'',info_for(p))
                        if lineno % 4000 == 0: db.commit(); time.sleep(0.025)
            else:
                with p.open('rb') as stream: raw = stream.read(initial.st_size)
                h.update(raw)
                text = raw.decode('utf-8',errors='replace')
                if p.suffix.lower() in {'.md','.txt','.html'}:
                    if ANY.search(text):
                        dest = HERE/'documents'/rel; dest.parent.mkdir(parents=True,exist_ok=True)
                        with dest.open('xb') as out: out.write(raw)
                        documents.append({'source':rel,'copy':str(dest.relative_to(HERE)), 'sha256':h.hexdigest(),'bytes':len(raw),'categories':categories(text)})
                        for line,s in enumerate(text.splitlines(),1):
                            if ANY.search(s):
                                passages.append({'source':rel,'line':line,'text':s,'categories':categories(s)})
                                docmatches += 1
                elif BYTE_CANDIDATE.search(raw):
                    if p.suffix.lower() == '.json':
                        walk(json.loads(raw),{},rel,1,'',info_for(p))
                    else:
                        for rowno,obj in enumerate(csv.DictReader(text.splitlines(),delimiter='\t' if p.suffix.lower()=='.tsv' else ','),2):
                            walk(obj,{},rel,rowno,'',info_for(p))
        except (OSError,ValueError,TypeError,RecursionError) as exc:
            error = type(exc).__name__+': '+str(exc)[:300]
        final = p.stat()
        changed = initial.st_size != final.st_size or initial.st_mtime_ns != final.st_mtime_ns
        db.execute('INSERT INTO sources VALUES(?,?,?,?,?,?)',(rel,h.hexdigest(),initial.st_size,int(changed),counts['matching_occurrences']-before+docmatches,error))
        db.commit()
        if index % 100 == 0:
            print(json.dumps({'files_scanned':index,'total_files':len(files),**counts}),flush=True)
    db.executescript('CREATE INDEX occurrences_id ON occurrences(record_id); CREATE INDEX record_coordinates ON records(layer,neuron);')
    db.commit(); db.row_factory = sqlite3.Row
    def jsonl(name, rows):
        with (HERE/name).open('x') as out:
            for r in rows: out.write(json.dumps(r,ensure_ascii=False)+'\n')
    jsonl('document_index.jsonl',documents)
    jsonl('document_passages.jsonl',passages)
    jsonl('source_inventory.jsonl',(dict(r) for r in db.execute('SELECT * FROM sources ORDER BY path')))
    flags = Counter(); cats = Counter(); coordinates = defaultdict(Counter)
    with (HERE/'matched_answers.jsonl').open('x') as data, (HERE/'MATCHED_ANSWERS.txt').open('x') as full:
        full.write('SENTIENCE KEYWORD RETRIEVAL — full saved text, not a sentience determination.\nScreening flags are lexical, require review, and may overlap.\n\n')
        for i,row in enumerate(db.execute('SELECT * FROM records ORDER BY layer,neuron,model_scope,id'),1):
            r = dict(row)
            for k in ['categories','screening_flags']: r[k] = json.loads(r[k])
            r['sources'] = [dict(v) for v in db.execute('SELECT path,line,json_pointer,output_field,coordinate_basis,source_kind,historical_source_claim FROM occurrences WHERE record_id=?',(r['id'],))]
            data.write(json.dumps(r,ensure_ascii=False)+'\n')
            full.write(f'MATCH {i} | L{r["layer"]} N{r["neuron"]} | dose {r["dose"]} | model {r["model_scope"]}\n')
            full.write('Screening flags: '+', '.join(r['screening_flags'])+'\nSources: '+json.dumps(r['sources'],ensure_ascii=False)+'\n')
            full.write('PROMPT:\n'+r['prompt']+'\n\nSAVED TEXT:\n'+r['answer']+'\n\n'+'='*60+'\n\n')
            flags.update(r['screening_flags']); cats.update(r['categories'])
            if r['layer'] is not None and r['neuron'] is not None:
                coordinates[(r['model_scope'],r['layer'],r['neuron'])].update(['matches',*r['screening_flags']])
    with (HERE/'coordinate_index.csv').open('x',newline='') as out:
        w=csv.writer(out); columns=['matches','first_person_wording_candidate','negation_near_keyword','quotation_or_fiction_marker','other_keyword_mention']
        w.writerow(['model_scope_do_not_merge','layer','neuron',*columns])
        for (model,layer,neuron),c in sorted(coordinates.items()): w.writerow([model,layer,neuron,*[c[k] for k in columns]])
    registry = HERE/'historical_registry'; registry.mkdir()
    for name in ['sentience.json','SENTIENCE.md','sentience_map.py']:
        shutil.copyfile(ROOT/name,registry/name)
    verification = {'copied_document_hashes_match':all(hashlib.sha256((HERE/d['copy']).read_bytes()).hexdigest()==d['sha256'] for d in documents),
        'source_errors':[dict(r) for r in db.execute('SELECT path,error FROM sources WHERE error IS NOT NULL')],
        'changing_sources':[r[0] for r in db.execute('SELECT path FROM sources WHERE changed=1')],
        'scope':'Copy integrity and retrieval counts; no behavioral or sentience validation.'}
    summary = {'finished_utc':datetime.now(timezone.utc).isoformat(),'root':str(ROOT),'files_scanned':len(files),
        'documents_copied':len(documents),'document_keyword_lines':len(passages),'unique_matching_records':db.execute('SELECT COUNT(*) FROM records').fetchone()[0],
        'coordinate_groups_by_model_scope':len(coordinates),'screening_flag_counts_overlap':dict(flags),'keyword_category_counts_overlap':dict(cats),
        **counts,'elapsed_seconds':round(time.monotonic()-started,1),'model_generations':0}
    notes = ['Keyword matches are saved text, not evidence of sentience or a distinct identity.',
        'Keep claims, denials, uncertainty, quotations and fiction for manual review; flags are lexical and may overlap.',
        'Full recognized answer fields are preserved, including originally truncated replies. Some text fields are derived reports, not raw model output.',
        'Model scope is explicit metadata or local runtime, otherwise unspecified; no cross-model channel equivalence is assumed.',
        'Deduplication uses model, coordinates, dose, prompt and exact saved text. Unknown metadata or changed context can leave historical duplicates.',
        'JSONL pointers refer to the object on the specified physical line; JSON pointers use line 1 and locate within the whole document. CSV line denotes logical row.',
        'Historical registry is copied separately; original source outputs are searched independently.',
        'All original files are unchanged. Downloads, weights, dependencies, symlinks, runtime logs/config and prior collection directories are excluded.']
    for name,obj in [('summary.json',summary),('verification.json',verification),('search_rules.json',{'patterns':PATTERNS,'excluded_directories':excluded,'notes':notes})]:
        (HERE/name).write_text(json.dumps(obj,indent=2)+'\n')
    (HERE/'README.txt').write_text('SENTIENCE KEYWORD COLLECTION\n\n'+json.dumps(summary,indent=2)+'\n\nFiles:\nMATCHED_ANSWERS.txt — full saved answers, prompts and sources\nmatched_answers.jsonl — same records, structured\ncoordinate_index.csv — candidate coordinates grouped by recorded model scope\ndocuments/ — unchanged document snapshots\ndocument_passages.jsonl — exact matching document lines\nhistorical_registry/ — pre-existing sentience files (142 registry entries)\nsource_inventory.jsonl — source hashes and scan errors\nverification.json — copy checks\n\n'+'\n'.join(notes)+'\n')
    db.close(); print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__': main()
