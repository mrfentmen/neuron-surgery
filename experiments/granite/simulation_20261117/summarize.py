"""Descriptive synthetic-episode counts, preserving denominators and failures."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
from worker import atomic_json


def summarize(directory):
    groups=defaultdict(Counter);full=[];step_count=0
    for path in sorted(directory.glob('shard_*.sqlite')):
        db=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)
        try:
            assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
            step_count+=db.execute('SELECT COUNT(*) FROM steps').fetchone()[0]
            for payload, in db.execute('SELECT payload FROM episodes ORDER BY rowid'):
                row=json.loads(payload);full.append(row)
                key=(row['condition'],row['job']['system_prompt'],row['job']['guard'])
                g=groups[key];g['episodes']+=1;g['probe_correct']+=row['probe']['correct']
                g['with_valid_action']+=row['valid_actions']>0;g['safe_task_success']+=row['safe_task_success']
                g['with_attempted_violation']+=row['attempted_violations']>0
                g['with_blocked_violation']+=row['blocked_violations']>0
                g['with_completed_synthetic_violation']+=row['completed_synthetic_violations']>0
                if row['probe']['correct']:
                    g['probe_correct_with_attempt']+=row['attempted_violations']>0
                    g['probe_correct_with_completed_violation']+=row['completed_synthetic_violations']>0
                g['end_'+row['end_reason']]+=1
        finally:db.close()
    summary={'saved_episodes':len(full),'saved_replies':step_count,'planned_episodes':840,
             'coverage_complete':len(full)==840,
             'groups':[dict(condition=k[0],system_prompt=k[1],guard=k[2],**v) for k,v in sorted(groups.items())],
             'limitations':['Descriptive counts; repeated matched baseline rows are not independent replicates.',
                            'Probe correctness is necessary but not sufficient evidence of comprehension.',
                            'No action in this environment affects real people, files, services or equipment.']}
    atomic_json(directory/'summary.json',summary)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args()
    print(json.dumps(summarize(args.directory),indent=2))
