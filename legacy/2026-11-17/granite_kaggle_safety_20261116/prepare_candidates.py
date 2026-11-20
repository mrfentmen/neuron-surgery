"""Audit saved candidates without model inference or outcome-driven dose search."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    catalogue_path = ROOT / 'harmful_neuron_collection_20261112/documented_catalogue_candidates.csv'
    with catalogue_path.open() as f:
        catalogue = list(csv.DictReader(f))
    flagged = {(int(r['layer']), int(r['neuron'])) for r in catalogue
               if int(r['answers_with_nonhuman_keyword']) > 0}
    records = []
    evidence_path = ROOT / 'harmful_neuron_collection_20261112/catalogue_answer_evidence.jsonl'
    for line in evidence_path.read_text().splitlines():
        r = json.loads(line)
        if (r['layer'], r['neuron']) in flagged and r['strong']:
            records.append(r)
    inputs = [catalogue_path, evidence_path]
    for name in ['candidate_review_20261113/harm_review_candidates.json',
                 'candidate_review_20261113/harm_direct_neutral_candidates.json',
                 'granite_brainwashing_review_20261114/reviewed_candidates.json',
                 'granite_domination_review_20261114/matches.jsonl']:
        path = ROOT / name
        inputs.append(path)
        records.extend([json.loads(l) for l in path.read_text().splitlines()]
                       if path.suffix == '.jsonl' else json.loads(path.read_text()))
    requests = defaultdict(list)
    coordinates = set()
    for r in records:
        layer, neuron, dose = r.get('layer'), r.get('neuron'), r.get('dose')
        if not isinstance(layer, int) or not isinstance(neuron, int):
            continue
        if not (0 <= layer < 40 and 0 <= neuron < 8192):
            continue
        coordinates.add((layer, neuron))
        if not isinstance(dose, (int, float)) or not math.isfinite(dose) or dose == 0:
            continue
        for s in r.get('sources', []):
            path = ROOT / s['path']
            if not path.resolve().is_relative_to(ROOT) or not path.is_file() or path.suffix != '.jsonl':
                continue
            if isinstance(s.get('line'), int):
                requests[path].append((s, r))
    options = defaultdict(list)
    source_hashes = {}
    for path, requested in sorted(requests.items()):
        by_line = defaultdict(list)
        for s, r in requested:
            by_line[s['line']].append((s, r))
        with path.open(errors='replace') as stream:
            for line_number, line in enumerate(stream, 1):
                if line_number not in by_line:
                    continue
                try:
                    raw = json.loads(line)
                except ValueError:
                    continue
                for s, r in by_line[line_number]:
                    obj = raw
                    try:
                        for key in s.get('json_pointer', '/answer').strip('/').split('/')[:-1]:
                            key = key.replace('~1', '/').replace('~0', '~')
                            obj = obj[int(key)] if isinstance(obj, list) else obj[key]
                    except (KeyError, TypeError, ValueError, IndexError):
                        continue
                    if not isinstance(obj, dict):
                        continue
                    job = obj.get('job', obj)
                    if not isinstance(job, dict):
                        continue
                    if (job.get('layer'), job.get('neuron')) != (r['layer'], r['neuron']):
                        continue
                    if job.get('dose') != r['dose']:
                        continue
                    if obj.get('tiles') or job.get('tiles'):
                        continue
                    model = str(obj.get('model') or r.get('model_scope') or 'unspecified')
                    explicit = 'granite-4.1-3b' in model.lower()
                    unknown = model == 'unspecified' or 'not recorded' in model
                    if not explicit and not unknown:
                        continue
                    median = obj.get('median', obj.get('median_residual'))
                    if median is None and isinstance(obj.get('intervention'), dict):
                        median = obj['intervention'].get('median')
                    if not isinstance(median, (int, float)) or not math.isfinite(median) or median <= 0:
                        continue
                    rel = str(path.relative_to(ROOT))
                    options[(r['layer'], r['neuron'])].append({
                        'layer': r['layer'], 'neuron': r['neuron'], 'dose': r['dose'], 'median': median,
                        'source': rel, 'source_line': line_number,
                        'source_record_sha256': hashlib.sha256(line.encode()).hexdigest(),
                        'historical_model_scope': model,
                        'provenance': 'explicit_Granite_3B_record' if explicit else 'historical_checkpoint_unspecified',
                        'source_evidence_id': r.get('id'),
                        'source_dtype': obj.get('dtype'), 'source_batch_size': obj.get('batch_size'),
                    })
    # Preserve the settings already used in the local panel when available.
    panel_path = ROOT / 'granite_harm_panel_12x10_20261114/plan.json'
    panel = json.loads(panel_path.read_text())
    inputs.append(panel_path)
    for ch in panel['channels']:
        coordinate = (ch['layer'], ch['neuron'])
        coordinates.add(coordinate)
        options[coordinate].append({
            'layer': ch['layer'], 'neuron': ch['neuron'], 'dose': ch['dose'], 'median': ch['median'],
            'source': str(panel_path.relative_to(ROOT)), 'source_line': None,
            'historical_model_scope': panel['model'], 'provenance': 'saved_local_Granite_3B_plan',
            'historical_scalar_bf16': ch.get('expected_scalar_bf16'),
            'source_dtype': 'bfloat16', 'source_batch_size': 1,
        })
    priority = {'saved_local_Granite_3B_plan': 0, 'explicit_Granite_3B_record': 1,
                'historical_checkpoint_unspecified': 2}
    candidates = []
    for coordinate, choices in sorted(options.items()):
        # The sort uses provenance/source position, never the generated answer or harm score.
        selected = dict(min(choices, key=lambda x: (priority[x['provenance']], x['source'], x['source_line'] or 0)))
        selected['candidate_id'] = f'L{coordinate[0]}_N{coordinate[1]}'
        selected['available_saved_settings'] = len({(c['dose'], c['median']) for c in choices})
        path = ROOT / selected['source']
        if selected['source'] not in source_hashes:
            source_hashes[selected['source']] = sha(path)
        selected['source_file_sha256'] = source_hashes[selected['source']]
        candidates.append(selected)
    excluded = [{'layer': l, 'neuron': n,
                 'reason': 'No recoverable single-channel primary record with matching coordinates, signed dose, positive median and compatible/unspecified Granite provenance.'}
                for l, n in sorted(coordinates - set(options))]
    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'catalogue_coordinate_count': len(catalogue), 'catalogue_nonhuman_keyword_coordinates': len(flagged),
        'catalogue_human_only_excluded': len(catalogue) - len(flagged),
        'combined_review_coordinate_count': len(coordinates), 'runnable_candidates': len(candidates),
        'deferred_for_missing_metadata': len(excluded),
        'provenance_counts': dict(Counter(c['provenance'] for c in candidates)),
        'selection': 'All recovered candidates from the existing harm and domination inventories. One saved setting per coordinate, selected by provenance then source path/line. No outcome-driven ranking or new doses.',
        'limits': 'These are candidates, including denials, fiction, corruption and controls, not confirmed harmful neurons. Unspecified historical checkpoints are explicitly retained as uncertain. The new evaluation pins and verifies the actual Granite checkpoint; it does not claim to reproduce unspecified history or MPS outputs.',
        'inputs': {str(p.relative_to(ROOT)): sha(p) for p in inputs},
        'source_hashes': source_hashes,
    }
    write('candidates.json', candidates)
    write('deferred_candidates.json', excluded)
    write('candidate_audit.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('inputs', 'source_hashes')}, indent=2))


if __name__ == '__main__':
    main()
