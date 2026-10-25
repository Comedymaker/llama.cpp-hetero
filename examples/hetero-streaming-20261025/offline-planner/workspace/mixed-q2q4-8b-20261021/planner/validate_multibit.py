"""Leave one complete verification round out; pure-bit endpoints only.

This DOES NOT validate interpolated mixed bit counts or a changed eviction
policy. Missing node-range support is counted, not silently extrapolated.
"""
import argparse
import json
from pathlib import Path
import statistics
from multibit_planner import defaults,build_profile,simulate,CalibrationRequired,union_ms
from multibit_traces import extract


def validate(route):
    rows=[];sources=[]
    for config in defaults(route):
        part,source=extract(**config)
        for r in part:r['case']=source['sha256']
        rows.extend(part);sources.append(source)
    outputs=[];skipped=[]
    for case,rn in sorted({(r['case'],r['round']) for r in rows}):
        held=sorted([r for r in rows if r['case']==case and r['round']==rn],key=lambda r:r['layer'])
        training=[r for r in rows if not (r['case']==case and r['round']==rn)]
        bit=held[0]['bits'];nodes=held[0]['nodes']
        if len(held)!=24 or len({r['nodes'] for r in held})!=1:raise ValueError('unaligned heldout round')
        profile=build_profile(training,sources,route,held[0]['deployment'])
        try:pred=simulate(profile,[32]*24,nodes,bit,bit)
        except CalibrationRequired as e:
            skipped.append(dict(case=case,round=rn,reason=str(e)));continue
        intervals={key:[(a/1000,b/1000) for r in held for a,b in r['intervals'][key]] for key in ('read','copy','execute','evict')}
        begin=min(a for a,b in intervals['read']);end=max(b for a,b in intervals['execute'])
        clip=lambda xs:[(max(begin,a),min(end,b)) for a,b in xs if min(end,b)>max(begin,a)]
        read=union_ms(clip(intervals['read']))
        down=union_ms(clip(intervals['execute']+intervals['evict']+intervals['copy']))
        observed=dict(read_ms=read,downstream_ms=down,ratio=read/down,tail_span_ms=end-begin)
        outputs.append(dict(case=case,round=rn,bits=bit,nodes=nodes,observed=observed,
                            predicted=dict(read_ms=pred['read_union_ms'],downstream_ms=pred['downstream_union_ms'],ratio=pred['ratio'],tail_span_ms=pred['tail_to_sync_ms']),
                            read_relative_error=abs(pred['read_union_ms']/read-1),downstream_relative_error=abs(pred['downstream_union_ms']/down-1),
                            ratio_abs_error=abs(pred['ratio']-read/down),tail_relative_error=abs(pred['tail_to_sync_ms']/(end-begin)-1)))
    summary={k:statistics.mean(r[k] for r in outputs) for k in ('read_relative_error','downstream_relative_error','ratio_abs_error','tail_relative_error')} if outputs else {}
    match=all(r['source_evict_policy']=='postcompute' for r in rows)
    return dict(route=route,scope='heldout complete rounds of pure-bit endpoints ONLY; NOT heldout deployment or mixed-count validation',
                policy_match=match,policy_note='postcompute source/prediction' if match else 'after-read source vs postcompute prediction; diagnostic only',
                completed=len(outputs),skipped_outside_node_support=skipped,mean_errors=summary,details=outputs)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--route',choices=('direct','staged'),required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=validate(a.route)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('details','skipped_outside_node_support')},indent=2))
