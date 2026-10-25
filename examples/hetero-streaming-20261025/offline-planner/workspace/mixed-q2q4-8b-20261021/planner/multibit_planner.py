"""Balance-first multibit planner, with separate direct and read/copy routes.

Q2/Q3 mixture costs interpolate measured pure-format endpoints, NOT a claim
that mixed Q3 deployment already exists or has been validated on the phone.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
from multibit_traces import extract, union_us, LAYERS, WEIGHTS_LAYER

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
METRICS = ('read', 'copy', 'execute', 'evict', 'prep')


class CalibrationRequired(ValueError):
    pass


def union_ms(xs):
    end = -float('inf')
    total = 0.
    for a,b in sorted(xs):
        total += max(0., b-max(a,end))
        end=max(end,b)
    return total


def build_profile(rows, sources, route, deployment):
    if not rows or any(r['route'] != route or r['deployment'] != deployment for r in rows):
        raise ValueError('cross-route/deployment fitting forbidden')
    bits=sorted({r['bits'] for r in rows})
    layers={}
    for layer in LAYERS:
        anchors={}
        for bit in bits:
            part=[r for r in rows if r['layer']==layer and r['bits']==bit]
            if not part:
                raise ValueError('missing layer/bit anchor')
            # Keep actual per-node medians. No universal per-node throughput
            # assumption across Tile boundaries; interpolation is explicit.
            points=[]
            for n in sorted({r['nodes'] for r in part}):
                group=[r for r in part if r['nodes']==n]
                points.append(dict(nodes=n, samples=len(group), **{m:statistics.median(r[m+'_ms'] for r in group) for m in METRICS}))
            anchors[str(bit)]=dict(points=points, nodes_range=[points[0]['nodes'],points[-1]['nodes']], samples=len(part))
        layers[str(layer)]=anchors
    return dict(schema=2, route=route, deployment=deployment, layers=layers, bits=bits, sources=sources,
                quant_group=32, source_evict_policies=sorted({r['source_evict_policy'] for r in rows}),
                calibrated_readers=7,calibrated_copy_workers=1 if route=='staged' else 0,
                calibrated_staging_buffers=14 if route=='staged' else 0,calibrated_chunk_bytes=8<<20 if route=='staged' else 0,
                builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                parser_sha256=hashlib.sha256((HERE/'multibit_traces.py').read_bytes()).hexdigest(),
                cost_model='per-node medians; piecewise-linear nodes; affine fractions of pure-format endpoints for NEW mixed counts',
                units='ms; true compressed bytes, includes0.5bit FP16 scales/group32',
                mixed_precision_validated=False, memory_cap_certified=False)


def anchor_cost(anchor, nodes, allow_extrapolation=False):
    points=anchor['points']
    if not points or not all(math.isfinite(float(p[m])) and p[m]>=0 for p in points for m in METRICS):
        raise ValueError('invalid metric costs')
    low, high=anchor['nodes_range']
    if not low <= nodes <= high and not allow_extrapolation:
        raise CalibrationRequired(f'node count{nodes} outside observed{low}..{high}')
    # Explicit out-of-range mode CLAMPS, never invents a linear Tile slope.
    if nodes <= low:
        return {m:float(points[0][m]) for m in METRICS}
    if nodes >= high:
        return {m:float(points[-1][m]) for m in METRICS}
    for a,b in zip(points,points[1:]):
        if a['nodes'] <= nodes <= b['nodes']:
            t=(nodes-a['nodes'])/(b['nodes']-a['nodes'])
            return {m:a[m]*(1-t)+b[m]*t for m in METRICS}
    raise ValueError('unordered anchor nodes')


def layer_cost(profile, layer, upgrade_count, nodes, base_bit=2, upgrade_bit=3, allow_extrapolation=False):
    if not isinstance(upgrade_count,int) or not 0 <= upgrade_count <= 32:
        raise ValueError('integer shard count0..32 required')
    anchors=profile['layers'][str(layer)]
    needed={base_bit,upgrade_bit}
    missing=needed-set(map(int,anchors))
    if missing:
        raise CalibrationRequired(f"{profile['route']} layer{layer} lacks MEASURED bit anchors {sorted(missing)}")
    a=anchor_cost(anchors[str(base_bit)],nodes,allow_extrapolation)
    b=anchor_cost(anchors[str(upgrade_bit)],nodes,allow_extrapolation)
    t=upgrade_count/32
    result={m:(1-t)*a[m]+t*b[m] for m in METRICS}
    if profile['route']=='direct' and result['copy']!=0:
        raise ValueError('direct route must not include copy stage')
    result['payload_bytes']=int(WEIGHTS_LAYER*((base_bit+.5)+(upgrade_bit-base_bit)*t)/8)
    return result


def simulate(profile, counts, nodes, base_bit=2, upgrade_bit=3, prefetch_ms=0., chunk_bytes=8<<20, staging_buffers=14, allow_extrapolation=False, io_scale=1., balance_tolerance=.02):
    if profile['schema']!=2 or profile['route'] not in ('direct','staged') or len(counts)!=24:
        raise ValueError('profile/route/selection shape')
    if nodes<1 or prefetch_ms<0 or chunk_bytes<1 or staging_buffers<1 or not math.isfinite(io_scale) or io_scale<=0 or not 0<=balance_tolerance<1:
        raise ValueError('invalid schedule inputs')
    staged=profile['route']=='staged'
    timeline=[];rs=[];cs=[];es=[];vs=[]
    read_free=-prefetch_ms;copy_free=-prefetch_ms;evict_free=0.
    # Aggregate calibrated read lane: it already includes the measured reader
    # team's concurrency. This is NOT one extra reader per synthetic chunk.
    host_free=[-prefetch_ms]*staging_buffers
    end_prev=0.;start_prev=-prefetch_ms
    total_bytes=0
    for i,layer in enumerate(LAYERS):
        cost=layer_cost(profile,layer,counts[i],nodes,base_bit,upgrade_bit,allow_extrapolation)
        cost['read']*=io_scale
        total_bytes+=cost['payload_bytes']
        slot_free=timeline[i-2]['execute_end_ms'] if i>=2 else -prefetch_ms
        gate=start_prev if i else -prefetch_ms
        chunks=[]
        if staged:
            n=max(1,math.ceil(cost['payload_bytes']/chunk_bytes))
            for j in range(n):
                size=min(chunk_bytes,cost['payload_bytes']-j*chunk_bytes)
                share=size/cost['payload_bytes']
                host=min(range(staging_buffers),key=lambda h:host_free[h])
                rb=max(read_free,gate,host_free[host]);re=rb+cost['read']*share
                # The final HTP A/B slot must be free before any copy writes.
                cb=max(re,copy_free,slot_free);ce=cb+cost['copy']*share
                read_free=re;copy_free=ce;host_free[host]=ce
                rs.append((rb,re));cs.append((cb,ce))
                chunks.append(dict(bytes=size,host_buffer=host,read_begin_ms=rb,read_end_ms=re,copy_begin_ms=cb,copy_end_ms=ce))
            ready=chunks[-1]['copy_end_ms']
        else:
            rb=max(read_free,gate,slot_free);re=rb+cost['read'];read_free=re
            rs.append((rb,re));ready=re
            chunks.append(dict(bytes=cost['payload_bytes'],read_begin_ms=rb,read_end_ms=re))
        eb=max(ready,end_prev,0.)+cost['prep'];ee=eb+cost['execute']
        vb=max(ee,evict_free);ve=vb+cost['evict'];evict_free=ve
        es.append((eb,ee));vs.append((vb,ve))
        timeline.append(dict(layer=layer,base_bit=base_bit,upgrade_bit=upgrade_bit,upgraded=counts[i],
                             htp_slot=i%2,execute_begin_ms=eb,execute_end_ms=ee,evict_begin_ms=vb,evict_end_ms=ve,
                             cost=cost,chunks=chunks))
        end_prev=ee;start_prev=eb
    end=end_prev
    clip=lambda xs:[(max(0,a),min(end,b)) for a,b in xs if min(end,b)>max(0,a)]
    r=union_ms(clip(rs));c=union_ms(clip(cs));e=union_ms(clip(es));v=union_ms(clip(vs))
    downstream=union_ms(clip(es+vs+(cs if staged else [])))
    if r<=0 or downstream<=0:
        raise ValueError('zero schedule cost')
    return dict(read_union_ms=r,copy_union_ms=c,execute_union_ms=e,evict_union_ms=v,
                execute_evict_union_ms=union_ms(clip(es+vs)),execute_copy_union_ms=union_ms(clip(es+cs)),
                read_over_execute_evict=r/union_ms(clip(es+vs)),read_over_execute_copy=r/union_ms(clip(es+cs)),
                downstream_union_ms=downstream,ratio=r/downstream,objective=abs(math.log(r/downstream)),
                balance_tolerance=balance_tolerance,balance_penalty=max(0.,abs(math.log(r/downstream))-math.log1p(balance_tolerance)),
                read_execute_overlap_ms=r+e-union_ms(clip(rs+es)),read_copy_overlap_ms=r+c-union_ms(clip(rs+cs)),
                tail_to_sync_ms=end,evict_drain_ms=max(0,evict_free-end),payload_bytes=total_bytes,
                staging_reserved_bytes=chunk_bytes*staging_buffers if staged else 0,
                staging_scope='aggregate scheduling buffers, not total runtime/HTP/PageCache peak',timeline=timeline)


def optimize(profile,nodes,base_bit,upgrade_bit,within_balance='precision',**kwargs):
    if within_balance not in ('precision','latency'):
        raise ValueError('invalid balance tie-break')
    def key(s,counts):
        rest=(-sum(counts),s['tail_to_sync_ms']) if within_balance=='precision' else (s['tail_to_sync_ms'],-sum(counts))
        return (s['balance_penalty'],*rest)
    if base_bit==upgrade_bit:
        counts=[32]*24
        return counts,simulate(profile,counts,nodes,base_bit,upgrade_bit,**kwargs)
    best=None
    for seed in (0,16,32):
        counts=[seed]*24
        cur=simulate(profile,counts,nodes,base_bit,upgrade_bit,**kwargs)
        for _ in range(4):
            changed=False
            for i in range(24):
                old=counts[i]
                winner=(*key(cur,counts),old,cur)
                for k in range(33):
                    counts[i]=k
                    s=simulate(profile,counts,nodes,base_bit,upgrade_bit,**kwargs)
                    candidate=(*key(s,counts),k,s)
                    if candidate[:3]<winner[:3]:winner=candidate
                counts[i]=winner[3];cur=winner[4];changed|=old!=counts[i]
            if not changed:break
        candidate=(*key(cur,counts),list(counts),cur)
        if best is None or candidate[:3]<best[:3]:best=candidate
    return best[3],best[4]


def defaults(route):
    if route=='direct':
        return [dict(path=WORKSPACE/'q2-llama3-8b-20261021/phone-results/logs/cpu_n128.log',route='direct',bits=2,deployment='q23-direct-cpu4-r7'),
                dict(path=WORKSPACE/'q3g32-8b-20261022/phone-results/llama-exp-q3g32-8b-20261022/logs/q3g32_gain_n128.log',route='direct',bits=3,deployment='q23-direct-cpu4-r7')]
    return [dict(path=WORKSPACE/f'staged-q23-calibration-20261024/phone-results/logs/q{bit}_staged_n{n}.log',route='staged',bits=bit,deployment='q23-staged-cpu4-r7-c1') for bit in (2,3) for n in (16,128)]


def measured_summary(rows):
    intervals={m:[v for r in rows for v in r['intervals'][m]] for m in ('read','copy','execute','evict')}
    # Exclude final per-round eviction drains, just as simulated balance does.
    starts=[a for r in rows for a,b in r['intervals']['read']]
    begin=min(starts);end=max(b for r in rows for a,b in r['intervals']['execute'])
    clip=lambda xs:[(max(begin,a),min(end,b)) for a,b in xs if min(end,b)>max(begin,a)]
    r=union_us(clip(intervals['read']))/1000
    down=union_us(clip(intervals['execute']+intervals['evict']+intervals['copy']))/1000
    return dict(observations=len(rows),read_ms=r,downstream_ms=down,ratio=r/down,
                max_evict_boundary_slop_us=max(x['evict_boundary_slop_us'] for x in rows))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--route',choices=('direct','staged'),default='direct')
    p.add_argument('--base-bit',type=int,choices=(2,3,4),default=2)
    p.add_argument('--upgrade-bit',type=int,choices=(3,4),default=3)
    p.add_argument('--nodes',type=int,default=8)
    p.add_argument('--cost-table',type=Path)
    p.add_argument('--output',type=Path,default=HERE/'multibit-output')
    p.add_argument('--trace-manifest',type=Path,help='JSON list path/route/bits/deployment; do not combine deployments')
    p.add_argument('--prefetch-ms',type=float,default=0.)
    p.add_argument('--io-scale',type=float,default=1.,help='explicit measured-read-time perturbation, not a new bandwidth measurement')
    p.add_argument('--balance-tolerance',type=float,default=.02,help='relative symmetric log-ratio band; avoid chasing false exact equality')
    p.add_argument('--within-balance',choices=('precision','latency'),default='precision',help='within balance band prefer more upgraded shards (quality proxy) or shorter Tail')
    p.add_argument('--chunk-bytes',type=int,default=8<<20)
    p.add_argument('--staging-buffers',type=int,default=14,help='default matches7 callers x2 buffers x8MiB measured staged deployment')
    p.add_argument('--allow-extrapolation',action='store_true',help='explicit node-range CLAMP; marked uncalibrated')
    p.add_argument('--calibration-only',action='store_true')
    p.add_argument('--memory-budget-gib',type=float)
    a=p.parse_args(argv)
    if a.base_bit>a.upgrade_bit or a.nodes<1:p.error('invalid bit order or nodes')
    if a.memory_budget_gib is not None:p.error('no certified total-memory model; refusing requested cap')
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'cost-table.json').exists() or (a.output/'plan.json').exists():p.error('choose a fresh output directory')
    measured=[]
    if a.cost_table:
        profile=json.loads(a.cost_table.read_text())
        if profile.get('schema')!=2 or profile.get('route')!=a.route:p.error('wrong schema/route; old Q2/Q4 table cannot masquerade as Q3/copy profile')
    else:
        configs=json.loads(a.trace_manifest.read_text()) if a.trace_manifest else defaults(a.route)
        deployments={x['deployment'] for x in configs}
        if len(deployments)!=1 or any(x['route']!=a.route for x in configs):p.error('one deployment/route per profile')
        rows=[];sources=[]
        for config in configs:
            part,source=extract(**config);rows.extend(part);sources.append(source)
            measured.append(dict(source=source,summary=measured_summary(part)))
        profile=build_profile(rows,sources,a.route,next(iter(deployments)))
    (a.output/'cost-table.json').write_text(json.dumps(profile,indent=2)+'\n')
    (a.output/'measured-anchors.json').write_text(json.dumps(measured,indent=2)+'\n')
    required={a.base_bit,a.upgrade_bit};missing=sorted(required-set(profile['bits']))
    requirement=dict(status='calibration_required' if missing else 'anchors_available',route=a.route,missing_bits=missing,
                     required_nodes=[a.nodes],required_layers=LAYERS,threads='use ONE matching deployment; never transplant direct costs to staged',
                     stages=list(METRICS),evict_policy='postcompute',
                     trace_contract='STAGED_Q2/Q3 call summaries + STAGED_INTERVAL phase=read/copy with absolute us, call id, chunk bytes; matching cluster execute/token and eviction events',
                     note='need full pure-bit endpoints first; mixed counts are interpolation hypotheses, require same-state phone validation',
                     unsupported='no hard memory-budget certificate or predicted quality claim')
    (a.output/'calibration-required.json').write_text(json.dumps(requirement,indent=2)+'\n')
    if missing:
        print(json.dumps(requirement,indent=2))
        if a.calibration_only:return
        raise SystemExit('Q2/Q3 staged profile missing; no fabricated optimal shard count emitted')
    if a.calibration_only:
        print(json.dumps(requirement,indent=2));return
    try:
        counts,pred=optimize(profile,a.nodes,a.base_bit,a.upgrade_bit,within_balance=a.within_balance,prefetch_ms=a.prefetch_ms,
                             chunk_bytes=a.chunk_bytes,staging_buffers=a.staging_buffers,allow_extrapolation=a.allow_extrapolation,io_scale=a.io_scale,balance_tolerance=a.balance_tolerance)
    except CalibrationRequired as e:p.error(str(e))
    ranking=WORKSPACE/'q2-llama3-8b-20261021/server-scan/server-results/ranking768.json'
    rank=json.loads(ranking.read_text())
    if len(rank)!=768 or {(int(r['layer']),int(r['shard'])) for r in rank}!={(l,s) for l in LAYERS for s in range(32)}:
        raise ValueError('ranking coverage')
    selected={(l,int(r['shard'])) for l,k in zip(LAYERS,counts) for r in [x for x in rank if int(x['layer'])==l][:k]}
    assignments=[(l,s,a.upgrade_bit if (l,s) in selected else a.base_bit) for l in LAYERS for s in range(32)]
    # Three explicit columns: old two-column Q4 encoder MUST NOT consume this.
    (a.output/'assignments.tsv').write_text('layer\tshard\tbits\n'+''.join(f'{l}\t{s}\t{b}\n' for l,s,b in assignments))
    hist=dict(Counter(b for l,s,b in assignments))
    warnings=['Offline prediction, not device-validated optimal count or speed.',
              'New mixed Q2/Q3 costs interpolate pure-format anchors; mixed-kernel overhead and contention not calibrated.',
              'Node costs use real rows, not depth/window; tree shape and Tile boundaries not independently fitted.',
              'Direct is seven-participant measured reader team; staged uses an aggregate read lane plus one serial copy lane.',
              'Synthetic chunks distribute aggregate stage costs proportionally; tensor/chunk granularity effects need replay validation.',
              'Compute includes DSP dequantization/dispatch/sync, not pure HMX arithmetic; eviction is not useful compute.',
              'Only L8 may be prefetched during Draft; L9 read starts at L8 execute start; HTP slots are overwritten only after prior compute completes.',
              'Eviction is modeled after compute; check source_evict_policies: any after-read source is policy-mismatched diagnostic data only.',
              'No proof of total DRAM limit; staging reservation excludes HTP, allocator, KV and Page Cache.',
              'Finite eviction-queue backpressure and changes in CPU/cache/storage contention are not simulated beyond observed costs.',
              'Old Q2-to-Q4 NLL importance selects identities; it is not a measured Q2-to-Q3 benefit table.',
              'More upgraded shards is a precision proxy, not a measured quality objective; coordinate search has no global optimality proof.',
              'Changing bit distribution can change acceptance and tree size; this holds verification rows fixed.']
    result=dict(schema=2,status='offline_interpolation_not_phone_validated',route=a.route,deployment=profile['deployment'],nodes=a.nodes,
                base_bit=a.base_bit,upgrade_bit=a.upgrade_bit,bit_counts=hist,per_layer_upgraded=dict(zip(LAYERS,counts)),
                objective='balance first beyond declared tolerance band; within band '+a.within_balance+' preference, not fictitious exact equality',within_balance=a.within_balance,
                prediction=pred,ranking_sha256=hashlib.sha256(ranking.read_bytes()).hexdigest(),
                source_evict_policies=profile['source_evict_policies'],node_clamp_requested=a.allow_extrapolation,io_scale=a.io_scale,warnings=warnings)
    result['driver_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['profile_sha256']=hashlib.sha256((a.output/'cost-table.json').read_bytes()).hexdigest()
    result['balance_status']='within_tolerance' if pred['balance_penalty']<=1e-10 else 'not_reached_by_coordinate_search'
    (a.output/'plan.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(route=a.route,nodes=a.nodes,bit_counts=hist,ratio=pred['ratio'],read_ms=pred['read_union_ms'],downstream_ms=pred['downstream_union_ms'],tail_ms=pred['tail_to_sync_ms'],status=result['status'],balance_status=result['balance_status']),indent=2))


if __name__=='__main__':main()
