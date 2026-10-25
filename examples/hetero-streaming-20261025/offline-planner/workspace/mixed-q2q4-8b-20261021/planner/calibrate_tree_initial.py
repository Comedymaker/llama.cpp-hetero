"""Calibrate initialization from the current mixture's saved full-action logs."""
import itertools,json,re,statistics,hashlib
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
log=root/'phone-results-planner146/logs/planner146_n128.log'
s=log.read_text(errors='replace');groups={}
for b,ms in re.findall(r'^batch gain action=\d+ b=(\d+) device=cpu .*?total-ms=([\d.]+)',s,re.M):groups.setdefault(int(b),[]).append(float(ms))
points={k:statistics.median(v) for k,v in groups.items()}
# This updates only observed action costs. Beyond the last observed batch size,
# retain the old prior shape scaled to its nearest observed endpoint.
oldx=[1,4,8,16];oldy=[27.9727,73.,138.09,267.204]
def old(b):return float(np.interp(b,oldx,oldy))
cpu=[];sources=[]
for b in range(1,17):
    if b in points:v=points[b];src='observed full-action median'
    elif min(points)<b<max(points):v=float(np.interp(b,sorted(points),[points[k] for k in sorted(points)]));src='interpolated observed action medians'
    else:
        near=min(points,key=lambda k:abs(k-b));v=old(b)*points[near]/old(near);src='scaled legacy shape; unmeasured extrapolation'
    cpu.append(v);sources.append(dict(batch=b,ms=v,samples=len(groups.get(b,[])),source=src))
rows=[]
for n,l,ms in re.findall(r'^batch gain round=\d+ actions=\d+ nodes=(\d+) leaves=(\d+).*?verify-ms=([\d.]+)',s,re.M):rows.append([int(n),int(l),float(ms)])
# Actual NPU rows includes pending; use pipeline rows, never tree candidates only.
pipe=[int(x) for x in re.findall(r'^pipeline interval execute_host_sync layer=8 tokens=(\d+)',s,re.M)]
if len(pipe)!=len(rows):raise ValueError('verification row count mismatch')
X=np.array([[1,p,r[1]] for p,r in zip(pipe,rows)],float);y=np.array([r[2] for r in rows])
def nnfit(x,y):
    best=None
    for bits in itertools.product((False,True),repeat=2):
        keep=[0]+[j+1 for j,v in enumerate(bits) if v];z=np.linalg.lstsq(x[:,keep],y,rcond=None)[0]
        c=np.zeros(3);c[keep]=z
        if c[0]<=0 or min(c[1:])<0:continue
        score=float(((x@c-y)**2).sum())
        if best is None or score<best[0]:best=score,c
    return best[1]
c=nnfit(X,y);c[1]=max(.001,c[1])
loo=[]
for i in range(len(y)):
    m=np.arange(len(y))!=i;q=nnfit(X[m],y[m]);loo.append(abs(float(X[i]@q-y[i])))
result=dict(log=str(log),log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),selection_sha256=hashlib.sha256((root/'planner146.tsv').read_bytes()).hexdigest(),
    cpu_batch_priors_ms=cpu,cpu_evidence=sources,
    verification=dict(base_ms=float(c[0]),node_ms=float(c[1]),leaf_ms=float(c[2]),samples=len(y),
      actual_rows_range=[min(pipe),max(pipe)],training_mae_ms=float(abs(X@c-y).mean()),leave_one_round_out_mae_ms=statistics.mean(loo)),
    scope='initialization only; online observe/EMA unchanged; no NPU-draft prior recalibration')
out=root/'tree-initial-146';out.mkdir(exist_ok=True)
(out/'initial-costs.json').write_text(json.dumps(result,indent=2))
(out/'cpu-priors.h').write_text('static const double cpu[] = {'+', '.join(f'{v:.6f}' for v in cpu)+'};\n')
print(json.dumps(result,indent=2))
