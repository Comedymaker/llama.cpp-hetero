"""Frozen-plan holdout: predict with original table, never refit on new run."""
import json,statistics,argparse
from pathlib import Path
from planner import parse_case,measured_rounds,simulate,ROOT,digest

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
table=json.loads((Path(__file__).parent/'output/cost-table.json').read_text())['layers']
plan=json.loads((Path(__file__).parent/'output/plan.json').read_text())
log=a.root/'logs/planner146_n128.log';selection=ROOT/'planner146.tsv'
rows=[r for r in parse_case(log,selection) if r['phase']=='generation']
for r in rows:
    if r['quality_flags']:raise ValueError(r['quality_flags'])
    r['top']=146
actual=measured_rounds(rows);details=[]
for r in actual:
    p=simulate(table,r['counts'],r['nodes'])
    details.append(dict(round=r['round'],actual_rows=r['nodes'],observed_ratio=r['observed_load_ms']/r['observed_compute_evict_ms'],
       predicted_ratio=p['ratio'],observed_load_ms=r['observed_load_ms'],observed_compute_evict_ms=r['observed_compute_evict_ms'],
       predicted_load_ms=p['load_union_ms'],predicted_compute_evict_ms=p['compute_evict_union_ms']))
out=dict(frozen_table_sha256=digest(Path(__file__).parent/'output/cost-table.json'),selection_sha256=digest(selection),
    forecast_at_fixed12_rows=plan['prediction']['ratio'],actual_nodes_min=min(r['actual_rows'] for r in details),
    actual_nodes_max=max(r['actual_rows'] for r in details),actual_nodes_mean=statistics.mean(r['actual_rows'] for r in details),
    actual_trace_rounds=len(details),mean_ratio_abs_prediction_error=statistics.mean(abs(r['observed_ratio']-r['predicted_ratio']) for r in details),
    observed_round_sum_ratio=sum(r['observed_load_ms'] for r in details)/sum(r['observed_compute_evict_ms'] for r in details),
    predicted_round_sum_ratio=sum(r['predicted_load_ms'] for r in details)/sum(r['predicted_compute_evict_ms'] for r in details),
    details=details,warning='new variable-node tree run; not fixed12 hardware measurement; frozen table, no refit')
(a.root/'planner-validation.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='details'},indent=2))
