"""Normalize generation-only Q2/Q3 direct and Q4 chunked read/copy traces.

Absolute clocks are used for joining. Generation-relative pipeline timestamps
are used ONLY to attach the token count in execute order, never as an origin.
"""
from collections import defaultdict
import hashlib
from pathlib import Path
import re

LAYERS = list(range(8, 32))
WEIGHTS_LAYER = 6815744 * 32


def union_us(intervals):
    total = 0
    end = -1
    for a, b in sorted(intervals):
        if b < a:
            raise ValueError('negative trace interval')
        total += max(0, b - max(a, end))
        end = max(end, b)
    return total


def fields(line):
    return dict(re.findall(r'(\w+)=([^ ]+)', line))


def extract(path, route, bits, deployment):
    path = Path(path)
    text = path.read_text(errors='replace')
    origin_match = re.search(r'^read route generation begin_us=(\d+)', text, re.M)
    nmatch = re.search(r'^n_verify\s*=\s*(\d+)', text, re.M)
    if not origin_match or not nmatch or not re.search(r'^tail (?:budget|HTP preload) failures = 0$', text, re.M):
        raise ValueError(f'incomplete generation trace {path}')
    origin, nrounds = int(origin_match[1]), int(nmatch[1])
    postcompute = route=='direct' or f'q{bits}g32 after-compute eviction ' in text
    events, node_events, reads, evicts = [], [], defaultdict(list), defaultdict(list)
    calls, chunks = {}, defaultdict(lambda: defaultdict(list))
    for line in text.splitlines():
        d = fields(line)
        if line.startswith('cluster trace phase=generation ') and d.get('event') == 'execute_sync':
            if int(d['last']) != int(d['first']) + 1 or d.get('cached') != '0':
                raise ValueError('requires full one-layer-streamed Tail')
            events.append((int(d['first']), int(d['start_us']), int(d['end_us'])))
        elif line.startswith('pipeline interval execute_host_sync ') and d.get('cached') == '0':
            node_events.append((int(d['layer']), int(d['tokens'])))
        elif route == 'direct' and line.startswith(f'q{bits}g32 read tensor='):
            if d.get('ok') != '1':
                raise ValueError('failed read')
            layer = int(d['tensor'].split('.')[1])
            reads[layer].append(dict(begin=int(d['begin_us']), end=int(d['end_us']), bytes=int(d['payload']), name=d['tensor']))
        elif route == 'staged' and line.startswith(f'STAGED_Q{bits} call='):
            if d.get('ok') != '1':
                raise ValueError('failed staged call')
            calls[int(d['call'])] = dict(layer=int(d['tensor'].split('.')[1]), begin=int(d['begin_us']), end=int(d['end_us']), bytes=int(d['bytes']), name=d['tensor'])
        elif route == 'staged' and line.startswith('STAGED_INTERVAL '):
            if d.get('ok') != '1' or d.get('phase') not in ('read', 'copy'):
                raise ValueError('failed/unknown staged interval')
            chunks[int(d['call'])][d['phase']].append((int(d['begin_us']), int(d['end_us']), int(d['bytes'])))
        elif line.startswith('q4 async_evict tensor='):
            if d.get('rc') != '0':
                raise ValueError('failed eviction')
            evicts[int(d['tensor'].split('.')[1])].append((int(d['begin_us']), int(d['end_us'])))
    if len(events) != nrounds * 24 or len(node_events) != len(events):
        raise ValueError(f'execution/token coverage mismatch {path}: {len(events)}/{len(node_events)} expected{nrounds*24}')
    if route == 'staged':
        # Format-specific Q2/Q3 instrumentation can use the same interval
        # contract. A Q4-prefixed call can NEVER become a Q2/Q3 anchor.
        if 'STAGED_POOLS read=7 copy=1 ' not in text or not calls:
            raise ValueError('unsupported staged calibration deployment')
        for cid, item in calls.items():
            item['read'] = chunks[cid]['read']
            item['copy'] = chunks[cid]['copy']
            if not item['read'] or not item['copy']:
                raise ValueError('missing staged read/copy intervals')
            if any(sum(x[2] for x in item[key]) != item['bytes'] for key in ('read', 'copy')):
                raise ValueError('staged chunk byte mismatch')
            reads[item['layer']].append(item)
        for group in reads.values():
            group.sort(key=lambda x: x['begin'])
    rows = []
    prompt_ends = {l: b for l, a, b in _prompt_exec(text)}
    for index, ((layer, eb, ee), (node_layer, nodes)) in enumerate(zip(events, node_events)):
        if layer != LAYERS[index % 24] or node_layer != layer or eb < origin:
            raise ValueError('generation alignment/order mismatch')
        # The layer was loaded at most once between consecutive executions of
        # that same layer. Draft prereads can precede the generation marker.
        previous_end = events[index-24][2] if index >= 24 else prompt_ends.get(layer, 0)
        group = [x for x in reads[layer] if x['begin'] >= previous_end and x['end'] <= eb]
        if len(group) != 7 or len({x['name'] for x in group}) != 7:
            raise ValueError(f'expected7 unique tensor loads layer{layer} round{index//24}: {len(group)}')
        payload = sum(x['bytes'] for x in group)
        if payload != WEIGHTS_LAYER * (bits + .5) / 8:
            raise ValueError('true payload mismatch (not carrier bytes)')
        next_start = events[index+24][1] if index+24 < len(events) else float('inf')
        if route == 'staged' and not postcompute:
            next_read = min((x['begin'] for x in reads[layer] if x['begin'] > eb), default=float('inf'))
            vs = [(a,b) for a,b in evicts[layer] if a >= min(x['begin'] for x in group) and b < next_read]
        else:
            vs = [(a,b) for a,b in evicts[layer] if a >= eb and b <= next_start]
        if len(vs) != 7:
            raise ValueError(f'evict coverage layer{layer} round{index//24}: {len(vs)}')
        # execute_sync's host end timestamp is recorded just after enqueueing
        # post-sync evictions. A worker can start a few us before that stamp.
        slop = max(0, ee-min(a for a,b in vs))
        if postcompute and slop > 1000:
            raise ValueError('eviction appears to precede compute completion')
        if route == 'staged':
            rs = [(a,b) for x in group for a,b,n in x['read']]
            cs = [(a,b) for x in group for a,b,n in x['copy']]
        else:
            rs = [(x['begin'],x['end']) for x in group]
            cs = []
        ready = max(x['end'] for x in group)
        preceding = events[index-1][2] if index % 24 else ready
        row = dict(layer=layer, bits=bits, nodes=nodes, round=index//24, route=route, deployment=deployment,
                   payload_bytes=payload, read_ms=union_us(rs)/1000, copy_ms=union_us(cs)/1000,
                   execute_ms=(ee-eb)/1000, evict_ms=union_us(vs)/1000,
                   prep_ms=max(0, eb-max(ready,preceding))/1000, evict_boundary_slop_us=slop,
                   source_evict_policy='postcompute' if postcompute else 'after-read',
                   intervals=dict(read=rs,copy=cs,execute=[(eb,ee)],evict=vs))
        rows.append(row)
    source=dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), route=route, bits=bits, deployment=deployment, observations=len(rows))
    artifacts=path.parent.parent/(path.stem+'.deployment.sha256')
    if artifacts.exists():source['deployment_manifest_sha256']=hashlib.sha256(artifacts.read_bytes()).hexdigest()
    return rows,source


def _prompt_exec(text):
    for line in text.splitlines():
        if line.startswith('cluster trace phase=prompt ') and 'event=execute_sync ' in line:
            d=fields(line)
            yield int(d['first']),int(d['start_us']),int(d['end_us'])
