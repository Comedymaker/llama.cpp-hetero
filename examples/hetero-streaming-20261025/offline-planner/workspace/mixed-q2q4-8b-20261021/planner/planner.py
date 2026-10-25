"""Offline balance-first Q2G32/Q4 planner. No device/server writes."""
from __future__ import annotations
import argparse,hashlib,json,math,statistics,re
from pathlib import Path
import numpy as np
from trace_costs import parse_case

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
LAYERS=list(range(8,32))
BASE_PAYLOAD=1635778560
DELTA=1703936

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def union_ms(intervals):
    end=-1e100;total=0.
    for a,b in sorted(intervals):
        if b>a:total+=max(0.,b-max(a,end));end=max(end,b)
    return total

def manifests():
    return [(k,ROOT/f'phone-results{suffix}'/'logs'/f'top{k}_n128.log',ROOT/f'top{k}.tsv')
            for suffix,ks in [('',[40,80]),('-120-160',[120,160]),('-200-240',[200,240])] for k in ks]

def collect():
    rows=[];sources=[]
    for top,log,selection in manifests():
        if not log.exists():raise ValueError(f'missing source {log}')
        s=log.read_text(errors='replace')
        if not re.search(r'^q4 async_evict summary .*?failures=0\b',s,re.M):raise ValueError('incomplete/failed log')
        r=parse_case(log,selection)
        r=[x for x in r if x['phase']=='generation']
        for x in r:
            if x['quality_flags']:raise ValueError(f"trace quality flags: {x['quality_flags']}")
            if x['nodes'] is None or x['load_ms'] is None or x['evict_ms'] is None:
                raise ValueError(f'incomplete aligned observation {top}: {x}')
            if x['payload_bytes'] != BASE_PAYLOAD//24+x['q4_count']*DELTA:
                raise ValueError(f'compact bytes / selected shard mismatch: {top} layer{x["layer"]}')
            if x['observed_q4_tiles'] != x['q4_count']*6656:
                raise ValueError('selected Q4 tile count mismatch')
            x['top']=top
        for layer in LAYERS:
            if not any(x['layer']==layer for x in r):raise ValueError('missing generation layer')
        # Ready gap is host preparation/dispatch not hidden in reading. Repeated
        # rounds are independent; never charge CPU Draft gaps as Tail overhead.
        for rn in sorted({x['round'] for x in r}):
            group=sorted([x for x in r if x['round']==rn],key=lambda x:x['layer'])
            for i,x in enumerate(group):
                ready=max(x['load_end_us'],group[i-1]['execute_end_us'] if i else x['load_end_us'])
                x['ready_gap_ms']=max(0.,(x['execute_begin_us']-ready)/1000) if i else 0.
        rows.extend(r)
        sources.append(dict(top=top,log=str(log),sha256=digest(log),selection_sha256=digest(selection),observations=len(r)))
    return rows,sources

def fit_metric(rows,key,nonnegative_q4=False):
    # Centering keeps ridge from penalizing the mean. Slopes are ms per full
    # 32-count/32-node increment, never a claim of separately measured shard cost.
    X=np.array([[r['q4_count']/32,r['nodes']/32] for r in rows],float)
    y=np.array([r[key] for r in rows],float)
    mean=X.mean(0);ym=float(y.mean());xc=X-mean
    coeff=np.linalg.solve(xc.T@xc+np.eye(2)*0.2,xc.T@(y-ym))
    if nonnegative_q4 and coeff[0]<0:coeff[0]=0
    intercept=ym-float(mean@coeff)
    pred=np.maximum(.001,intercept+X@coeff)
    return dict(intercept_ms=intercept,q4_32_ms=float(coeff[0]),nodes_32_ms=float(coeff[1]),
                training_mae_ms=float(abs(pred-y).mean()))

def build_table(rows):
    table={}
    for layer in LAYERS:
        part=[r for r in rows if r['layer']==layer]
        if not part:raise ValueError('no layer data')
        table[str(layer)]=dict(samples=len(part),q4_range=[min(r['q4_count'] for r in part),max(r['q4_count'] for r in part)],
             observed_q4_counts=sorted({r['q4_count'] for r in part}),
             nodes_range=[min(r['nodes'] for r in part),max(r['nodes'] for r in part)],
             load=fit_metric(part,'load_ms',True),execute=fit_metric(part,'execute_ms'),evict=fit_metric(part,'evict_ms',True),
             prep_ms=statistics.median(max(0.,r.get('ready_gap_ms',0.)) for r in part))
    return table

def cost(table,layer,k,nodes,io_scale=1.):
    row=table[str(layer)];ans={}
    for key in ('load','execute','evict'):
        m=row[key];ans[key]=max(.001,m['intercept_ms']+m['q4_32_ms']*k/32+m['nodes_32_ms']*nodes/32)
    ans['load']*=io_scale
    ans['prep']=row['prep_ms']
    return ans

def simulate(table,counts,nodes,io_scale=1.,prefetch_ms=0.):
    # Fixed topology of the deployed two-slot one-step pipeline. Eviction runs
    # on its own serialized CPU worker, AFTER execute. No fictitious serial
    # compute+evict sum. Layer31 drains; it has no next-layer load partner.
    reads=[];computes=[];evicts=[];timeline=[];evict_end=0.;previous_execute_start=-prefetch_ms
    previous_compute_end=0.;loader_available=-prefetch_ms
    for i,layer in enumerate(LAYERS):
        c=cost(table,layer,counts[i],nodes,io_scale)
        lb=max(loader_available,previous_execute_start)
        le=lb+c['load'];loader_available=le
        cb=max(previous_compute_end,le,0.)+c['prep'];ce=cb+c['execute']
        eb=max(ce,evict_end);ee=eb+c['evict'];evict_end=ee
        reads.append((lb,le));computes.append((cb,ce));evicts.append((eb,ee))
        timeline.append(dict(layer=layer,q4=counts[i],load_begin_ms=lb,load_end_ms=le,
            execute_begin_ms=cb,execute_end_ms=ce,evict_begin_ms=eb,evict_end_ms=ee,
            load_ms=c['load'],execute_ms=c['execute'],evict_ms=c['evict'],prep_ms=c['prep']))
        previous_execute_start=cb;previous_compute_end=ce
    # Compare exactly the same time horizon: through the final execute. Final
    # asynchronous evict tail is reported separately, not silently added.
    end=computes[-1][1]
    clip=lambda xs:[(max(0.,a),min(end,b)) for a,b in xs if min(end,b)>max(0.,a)]
    l=union_ms(clip(reads));ce=union_ms(clip(computes+evicts))
    local=[abs(timeline[i+1]['load_ms']-(timeline[i]['execute_ms']+timeline[i]['evict_ms']))/
           max(timeline[i+1]['load_ms'],timeline[i]['execute_ms']+timeline[i]['evict_ms']) for i in range(23)]
    # A pair's own post-compute eviction does not overlap its own execute, but
    # it CAN overlap later executions: global metric uses union above.
    return dict(load_union_ms=l,compute_evict_union_ms=ce,ratio=l/ce,
        predicted_tail_to_sync_ms=end,async_evict_drain_ms=max(0.,evict_end-end),
        local_pair_mean_relative_gap=float(np.mean(local)),
        objective=abs(math.log(l/ce)),timeline=timeline,
        compact_payload_bytes=BASE_PAYLOAD+sum(counts)*DELTA)

def optimize(table,nodes,io_scale,prefetch_ms,allow_extrapolation=False,max_passes=5):
    ranges=[range(33) if allow_extrapolation else range(table[str(l)]['q4_range'][0],table[str(l)]['q4_range'][1]+1) for l in LAYERS]
    best=None
    # Deterministic coordinate search; NOT a global optimality proof. First
    # objective is balance, no quality term or user-picked global Top-K.
    for seed in (0,.5,1):
        counts=[r.start+round((r.stop-1-r.start)*seed) for r in ranges]
        cur=simulate(table,counts,nodes,io_scale,prefetch_ms)
        for _ in range(max_passes):
            changed=False
            for i,r in enumerate(ranges):
                old=counts[i];winner=(cur['objective'],cur['local_pair_mean_relative_gap'],old,cur)
                for k in r:
                    counts[i]=k;s=simulate(table,counts,nodes,io_scale,prefetch_ms)
                    candidate=(s['objective'],s['local_pair_mean_relative_gap'],k,s)
                    if candidate[:3]<winner[:3]:winner=candidate
                counts[i]=winner[2];cur=winner[3];changed|=counts[i]!=old
            if not changed:break
        if best is None or cur['objective']<best[1]['objective']:best=(list(counts),cur)
    return best

def measured_rounds(rows):
    groups={}
    for r in rows:groups.setdefault((r['top'],r['round']),[]).append(r)
    result=[]
    for (top,round_no),g in sorted(groups.items()):
        g.sort(key=lambda x:x['layer'])
        if len(g)!=24:raise ValueError('incomplete round')
        load=[(r['load_begin_us']/1000,r['load_end_us']/1000) for r in g]
        exe=[(r['execute_begin_us']/1000,r['execute_end_us']/1000) for r in g]
        ev=[(a/1000,b/1000) for r in g for a,b in r['evict_intervals_us']]
        begin=min(a for a,b in load);end=max(b for a,b in exe)
        ev=[(max(begin,a),min(end,b)) for a,b in ev if min(end,b)>max(begin,a)]
        result.append(dict(top=top,round=round_no,nodes=g[0]['nodes'],counts=[r['q4_count'] for r in g],
            observed_load_ms=union_ms(load),observed_compute_evict_ms=union_ms(exe+ev),
            observed_execute_ms=union_ms(exe),observed_span_ms=end-begin))
    return result

def validate(rows):
    # Leave ONE COMPLETE SELECTION CASE out, not random rows from the same run.
    answer=[]
    for top in sorted({r['top'] for r in rows}):
        table=build_table([r for r in rows if r['top']!=top]);actual=measured_rounds([r for r in rows if r['top']==top]);errs=[]
        for a in actual:
            pred=simulate(table,a['counts'],a['nodes']);obs=a['observed_load_ms']/a['observed_compute_evict_ms']
            errs.append(dict(round=a['round'],nodes=a['nodes'],observed_ratio=obs,predicted_ratio=pred['ratio'],
                 ratio_abs_error=abs(obs-pred['ratio']),load_relative_error=abs(pred['load_union_ms']/a['observed_load_ms']-1),
                 compute_evict_relative_error=abs(pred['compute_evict_union_ms']/a['observed_compute_evict_ms']-1)))
        answer.append(dict(held_out_top=top,rounds=len(errs),mean_ratio_abs_error=statistics.mean(x['ratio_abs_error'] for x in errs),
            mean_load_relative_error=statistics.mean(x['load_relative_error'] for x in errs),
            mean_compute_evict_relative_error=statistics.mean(x['compute_evict_relative_error'] for x in errs),details=errs))
    return answer

def main():
    # Preserve the old Q2/Q4 CLI and outputs. Explicit multibit/route requests
    # use schema2; never silently reinterpret the original fitted cost table.
    import sys
    if any(x==flag or x.startswith(flag+'=') for x in sys.argv[1:] for flag in ('--upgrade-bit','--base-bit','--route','--trace-manifest')):
        from multibit_planner import main as multibit_main
        return multibit_main(sys.argv[1:])
    ap=argparse.ArgumentParser();ap.add_argument('--nodes',type=int,default=12);ap.add_argument('--io-scale',type=float,default=1.)
    ap.add_argument('--prefetch-ms',type=float,default=0.);ap.add_argument('--allow-extrapolation',action='store_true')
    ap.add_argument('--output',type=Path,default=HERE/'output');ap.add_argument('--memory-budget-gib',type=float)
    ap.add_argument('--cost-table',type=Path,help='reuse an extracted cost table; no raw-log reread/refit')
    ap.add_argument('--cpu-profile',type=Path,default=ROOT/'tree-initial-146/initial-costs.json',help='matching Early precision/thread full-action cost profile')
    ap.add_argument('--validate',action='store_true');a=ap.parse_args()
    if a.nodes<1 or a.io_scale<=0 or a.prefetch_ms<0:ap.error('invalid positive inputs')
    if a.cost_table:
        saved=json.loads(a.cost_table.read_text());table=saved['layers'];sources=saved['sources'];rows=None
        if saved.get('schema')!=1 or set(table)!=set(map(str,LAYERS)):ap.error('invalid cost table')
        if a.validate:ap.error('--validate requires original traces; omit --cost-table')
    else:
        rows,sources=collect();table=build_table(rows)
    node_min=max(x['nodes_range'][0] for x in table.values());node_max=min(x['nodes_range'][1] for x in table.values())
    if not node_min<=a.nodes<=node_max and not a.allow_extrapolation:ap.error(f'nodes outside observed range {node_min}..{node_max}; explicit --allow-extrapolation required')
    # Sampled RSS is NOT a memory constraint model. Reject instead of silently
    # treating an unsupported requested budget as satisfied.
    if a.memory_budget_gib is not None:ap.error('hard memory-budget planning is not implemented; current goal is balance only, sampled peaks cannot certify a cap')
    counts,pred=optimize(table,a.nodes,a.io_scale,a.prefetch_ms,a.allow_extrapolation)
    ranking_path=ROOT.parent/'q2-llama3-8b-20261021/server-scan/server-results/ranking768.json'
    rank=json.loads(ranking_path.read_text());selected=[]
    if len(rank)!=768 or {(int(r['layer']),int(r['shard'])) for r in rank}!={(l,s) for l in LAYERS for s in range(32)}:
        raise ValueError('invalid ranking joint-shard identities')
    for layer,k in zip(LAYERS,counts):
        selected += sorted([r for r in rank if int(r['layer'])==layer],key=lambda r:-float(r['benefit']))[:k]
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'selection.tsv').write_text(''.join(f"{int(r['layer'])}\t{int(r['shard'])}\n" for r in selected))
    warnings=['Empirical cost model, not a hardware-theoretical bound; candidate count is held fixed, not a tree optimizer.',
       'Selection objective is balance only; importance chooses identities within a fixed layer count. NLL benefits are not additive quality guarantees.',
       'Simulation omits eviction-queue backpressure and shared-resource feedback beyond observed concurrent costs; prediction must be checked against held-out error.',
       'Q4-sized physical carrier and ordinary-DRAM staging retained; no hard memory-budget certification or Page Cache accounting.',
       'Integer counts between measured per-layer bounds are INTERPOLATED; their joint vector is NEW and unmeasured, even without --allow-extrapolation.',
       'Final asynchronous eviction drain excluded from balance interval and separately reported; first-layer prefetch is a special boundary.']
    result=dict(objective='minimize abs(log(load_union / compute_or_evict_union)); local pairing is tie-break only',
        algorithm='deterministic multistart coordinate search, not certified global optimum',nodes=a.nodes,io_scale=a.io_scale,
        q4_count=sum(counts),q2_count=768-sum(counts),per_layer_q4=dict(zip(map(str,LAYERS),counts)),prediction=pred,
        limits=dict(observed_nodes=[node_min,node_max],allow_extrapolation=a.allow_extrapolation,memory_budget_gib=a.memory_budget_gib),
        status='offline prediction only; not yet integrated or tested on phone',
        uncertainty_note='held-out ratio mean absolute error observed about0.025..0.072; predicted1.000 is not a precision guarantee',
        interpolation_layers=[l for l,k in zip(LAYERS,counts) if k not in table[str(l)].get('observed_q4_counts',[])],
        new_joint_selection=True,
        sources=sources,ranking_sha256=digest(ranking_path),warnings=warnings)
    (a.output/'cost-table.json').write_text(json.dumps(dict(schema=1,layers=table,sources=sources,
       units='milliseconds; true compact file bytes, never Q4 carrier bytes',
       model='ridge0.2, centered linear count/32 and nodes/32; conditional observed concurrent costs',
       scope='same native mixed decoder, seven tensor readers, one layer reader, postcompute eviction'),indent=2))
    from tree_profile import export_profile
    initial=export_profile(a.output/'cost-table.json',counts,a.cpu_profile,a.output/'tree-initial',a.io_scale)
    result['tree_initial_profile']=dict(path='tree-initial/initial-profile.tsv',
        sha256=digest(a.output/'tree-initial/initial-profile.tsv'),selection_sha256=digest(a.output/'selection.tsv'),
        verification_scope='full Tail verification, L8 preread assumed ready, not compute/evict union',
        fixed_host_residual_ms=initial['fixed_host_residual_ms'])
    if a.validate:result['leave_one_case_out']=validate(rows)
    (a.output/'plan.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ('nodes','q4_count','q2_count','per_layer_q4')},indent=2))
    print(f"predicted ratio={pred['ratio']:.4f}; load={pred['load_union_ms']:.3f} ms; compute|evict={pred['compute_evict_union_ms']:.3f} ms")
    if a.validate:
        for v in result['leave_one_case_out']:print(f"heldout Top{v['held_out_top']}: ratio abs error={v['mean_ratio_abs_error']:.3f}, load relative error={v['mean_load_relative_error']:.1%}, compute|evict relative error={v['mean_compute_evict_relative_error']:.1%}")

if __name__=='__main__':main()
