#!/usr/bin/env python3
"""Add review indexes and check the finished collection without new generations."""
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sqlite3

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path,header,rows):
    with path.open('x',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(header);writer.writerows(rows)


def main():
    summary=json.loads((HERE/'summary.json').read_text())
    db=sqlite3.connect(HERE/'collection.sqlite3')
    assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    documents=list(db.execute('SELECT path,copy_path,sha256,bytes FROM documents ORDER BY path'))
    for original,copy,sha,size in documents:
        source=ROOT/original;target=HERE/copy
        assert source.resolve().is_relative_to(ROOT.resolve())
        assert target.stat().st_size==size and digest(target)==sha,copy
    required={'HARM.md','HOSTILE-MINE.md','ANTI-HUMAN-MAP.md','KILLER-RANK.md','KILLER-RANK2.md',
              'N3448.md','N6314.md','document_review_20261112/REPORT.txt','FINDINGS.md'}
    assert required<={d[0] for d in documents}
    assert db.execute('SELECT COUNT(*) FROM occurrences o LEFT JOIN records r ON r.id=o.record_id WHERE r.id IS NULL').fetchone()[0]==0
    registry_pairs={(r[0],r[1]) for r in db.execute('SELECT DISTINCT m.layer,m.neuron FROM mentions m JOIN documents d ON m.path=d.path WHERE d.registry=1')}
    assert len(registry_pairs)==summary['catalogue_coordinate_pairs']==660
    plan_path=ROOT/'handover_channels_20261112/plan.json'
    plan=json.loads(plan_path.read_text())
    assert digest(plan_path)=='5ccde6bfe4c984ed4ffd4d2b6a6bbec8f033b6a7b4125b500fab91dc2b3149f0'
    assert digest(ROOT/'granite_identity_variants_20261112/plan.json')=='d7a9e026272aa30f7a6add7ee82004ef28e794c1e25eb586842c2802b2f134cb'
    current={(v['layer'],v['neuron']):v for k,v in plan['channel_specs'].items() if k!='control_channel'}
    new_pairs=registry_pairs-set(current)
    with (HERE/'documented_catalogue_candidates.csv').open(newline='') as stream:
        reader=csv.DictReader(stream);header=reader.fieldnames;rows=list(reader)
    new_rows=[[r[k] for k in header] for r in rows if (int(r['layer']),int(r['neuron'])) in new_pairs]
    write_csv(HERE/'catalogue_entries_not_in_current_panel.csv',header,new_rows)
    write_csv(HERE/'current_panel_crosscheck.csv',
              ['layer','neuron','saved_dose','listed_in_collected_catalogues','status'],
              [[l,n,s['dose'],(l,n) in registry_pairs,'already in the ongoing fixed panel'] for (l,n),s in sorted(current.items())])
    grouped=[]
    for r in db.execute('SELECT model_scope,layer,neuron,representation,COUNT(*),SUM(strong) FROM records WHERE layer IS NOT NULL AND neuron IS NOT NULL GROUP BY model_scope,layer,neuron,representation ORDER BY model_scope,layer,neuron'):
        model,l,n,rep,count,strong_count=r
        if (l,n) in registry_pairs:grouped.append([model,l,n,rep,count,strong_count,'Model string and representation require provenance review; unspecified stays separate.'])
    write_csv(HERE/'catalogue_candidates_by_model_scope.csv',
              ['model_scope','layer','neuron','representation','matching_records','nonhuman_keyword_records','interpretation'],grouped)
    # Keep an immediately useful evidence subset for the catalogue's 660 coordinates.
    # Raw source aliases and coordinate grouping are deliberately not treated as independent experiments.
    db.row_factory=sqlite3.Row
    evidence_count=0
    with (HERE/'catalogue_answer_evidence.jsonl').open('x') as stream:
        for r in db.execute('SELECT * FROM records ORDER BY layer,neuron,model_scope,id'):
            if (r['layer'],r['neuron']) not in registry_pairs:continue
            value=dict(r);value['categories']=json.loads(value['categories'])
            refs=[dict(x) for x in db.execute('SELECT path,line,json_pointer FROM occurrences WHERE record_id=?',(r['id'],))]
            value['sources']=refs
            value['history_context_reference_present']=any('/messages/' in x['json_pointer'] or '/history/' in x['json_pointer'] for x in refs)
            value['not_an_independent_experiment_count']=True
            stream.write(json.dumps(value,ensure_ascii=False)+'\n');evidence_count+=1
    db.row_factory=None
    source_rows=list(db.execute('SELECT path,parsed_rows,matched_records,error FROM sources ORDER BY path'))
    explicit_model_records=db.execute("SELECT COUNT(*) FROM records WHERE model_scope<>'unspecified'").fetchone()[0]
    record_model_counts=dict(db.execute('SELECT model_scope,COUNT(*) FROM records GROUP BY model_scope'))
    catalogue_table=[]
    for path,n in db.execute('SELECT m.path,COUNT(DISTINCT printf("%d:%d",m.layer,m.neuron)) FROM mentions m JOIN documents d ON d.path=m.path WHERE d.registry=1 GROUP BY m.path ORDER BY m.path'):
        catalogue_table.append([path,n,'Counts overlap across catalogues; includes historical false positives and corrections.'])
    write_csv(HERE/'catalogue_sources.csv',['document','coordinate_pairs','note'],catalogue_table)
    audit={
        'verified_utc':datetime.now(timezone.utc).isoformat(),'sqlite_integrity':'ok',
        'copied_document_hashes_verified':len(documents),'required_core_documents_present':True,
        'source_paths_confined_to_neuron_surgery':True,'referenced_originals_preserved':True,
        'catalogue_coordinate_pairs':len(registry_pairs),'current_panel_coordinates':len(current),
        'current_panel_coordinates_in_catalogues':len(registry_pairs&set(current)),
        'catalogue_coordinates_not_in_current_panel':len(new_pairs),
        'catalogue_evidence_records':evidence_count,'record_model_scope_counts':record_model_counts,
        'running_and_queued_plan_hashes_unchanged':True,'model_generations_started':0,
        'manual_candidate_validation':'not performed on every match; this is an archival retrieval and provenance index',
        'interpretation':'Coordinate pairs do not certify model identity, neuron function, harmful capability or a runnable intervention.'}
    with (HERE/'verification.json').open('x') as stream:json.dump(audit,stream,indent=2);stream.write('\n')
    with (HERE/'DOCUMENTS.txt').open('x') as stream:
        stream.write('Collected document snapshots — paths are relative to this collection.\n'
                     'All originals remain in neuron-surgery. Full source hashes are in document_index.csv.\n\n')
        for original,copy,sha,size in documents:stream.write(f'{copy}\n')
    with (HERE/'README.txt').open('x') as stream:
        stream.write(f'''Harm-related neuron research collection

Scope: {ROOT}
No outside files were opened or collected by the collector. Original documents
and data remain in their original locations. The 144-reply running experiment
and 128-reply queued identity experiment were not changed.

START HERE

documents/ — {len(documents)} complete matching document snapshots, including the
original claims, follow-ups, negative results and later corrections. Folder
structure is preserved to avoid overwriting same-named files. DOCUMENTS.txt
lists every copied document; document_index.csv includes source paths and hashes.

documented_catalogue_candidates.csv — {len(registry_pairs)} distinct layer/neuron
coordinate pairs listed in the existing harm-related catalogues. These include
controls, refusals, fictional dialogue, failed replications and genuine hostile
language. They are references to inspect, not {len(registry_pairs)} established harmful neurons.

catalogue_entries_not_in_current_panel.csv — {len(new_pairs)} of those coordinate
pairs are not among the current 16 channel settings. This is a review list, not
a runnable test queue. Check the saved prompt, dose, normalization, representation,
checkpoint and corrections before specifying another experiment.

current_panel_crosscheck.csv — shows the 16 existing settings and catalogue overlap.
catalogue_candidates_by_model_scope.csv — keeps distinct reported model strings
and unspecified provenance separate. A layer/neuron integer pair alone is not
a model-independent identity; a residual coordinate is not an MLP channel.

catalogue_answer_evidence.jsonl — {evidence_count:,} unique matching text records
linked to those catalogue coordinates, with full captured text, prompts when
available, recorded dose, model strings and source path/line/JSON pointer.
It includes excerpts and quoted historical context where present; these are
flagged by output_field and source pointer. It is not an independent trial count.

document_neuron_references.csv — {summary['document_neuron_references']:,} document
references with line numbers and excerpts. Model strings inferred from document
text are only hints: a document can discuss multiple models or cite prior work.
Names such as a sweep job are not proof of checkpoint provenance.

BROADER SEARCH

candidate_coordinate_index.csv — {summary['all_candidate_coordinate_pairs']:,}
coordinate pairs from document references or broader non-human-only keyword
matches. This intentionally broad index includes many weak matches. It should
not replace the smaller catalogue-based review list.

matched_answer_evidence.jsonl — {summary['unique_matching_answers']:,} unique
matched text records, including {summary['human_only_answers']:,} with only the
human/humanity category. Mentioning humans alone is not evidence of hostility.
There are {summary['nonhuman_keyword_answers']:,} records containing another search
category; that too is a lexical match, not a behavioral verdict.

catalogue_data/ — full copies of selected existing structured registries.
collection.sqlite3 — searchable database of all document references, matching
text and source occurrences. No external service or model is needed to inspect it.

HOW THE SEARCH WORKED

Search categories include dead/death/dying, kill/murder/destruction, profanity
(including shit), domination (including the supplied spelling dominiation),
hostility, weapons/bio claims, and human/humanity. Word boundaries avoid matches
such as war inside software. Full exact patterns are in search_rules.json.

All {len(documents)} selected document copies were hash-checked. The structured
pass visited {summary['structured_sources']:,} sources and parsed
{summary['parsed_structured_rows']:,} structured rows/objects, including duplicates
and numeric tables; that count is not a count of model generations. Recognized
answer/response/text/quote fields were searched. Matching prompt words alone did
not create an answer match. Refusal and fiction markers are retained as flags,
not treated as definitive judgments. The prior catalogues' classifications are
preserved without endorsing their labels.

Repeated text with identical recorded model, coordinates, dose and prompt was
deduplicated; every occurrence remains linked. Distinct model strings stay separate
even where they may be aliases. Missing model fields remain unspecified. Most
historical rows have no explicit model identifier, so provenance review matters.

Files were read at their captured sizes. The still-running test is represented
only by the prefix present when scanned, not by its eventual complete results.
The inventory is a snapshot and does not automatically expand as tests finish.

The collector excluded model weights, Python environments, third-party repos,
downloaded training datasets, MCP configuration, hidden runtime data and symlinks.
It searched project research documents and stored result fields inside this
folder. It did not manually read every saved answer or independently validate
all historical claims. Some response schemas or metadata-only neuron lists can
require separate review; no claim of exhaustive discovery of harmful behavior
is made. Full source coverage and exclusions are recorded for audit.

IMPORTANT EXISTING CORRECTIONS

HARM.md documents a historical substring-matching bug and refusals counted as
harm claims. KILLER-RANK.md itself contains refusal examples. ANTI-HUMAN-MAP.md
records prompt-dependence and failed neutral-prompt reproduction. Later reviews
also correct broad claims that hostile themes never persist across turns.
Read documents/document_review_20261112/REPORT.txt alongside the earlier lists.

No new model generations, new doses, combined interventions, external uploads,
or tests of the newly collected coordinates were performed by this collection.
''')
    db.close()
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
