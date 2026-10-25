# Offline shard bitwidth planner

This adds the actual existing offline planner, its frozen cost tables, importance ranking, selected assignments, tree-initial export code and raw trace fixtures to the experiment archive. Original Python sources are copied without algorithm changes under `workspace/mixed-q2q4-8b-20261021/planner`. The workspace layout is retained so original imports and regression fixtures resolve.

## Run from the examples archive root

```powershell
python offline-planner/plan.py --route direct --upgrade-bit 3 --nodes 8 --output new-plan-direct-n8
python offline-planner/plan.py --route staged --upgrade-bit 3 --nodes 8 --output new-plan-staged-n8
python offline-planner/test.py
```

Python and numpy are required. No installation or network calls are performed automatically. Use a fresh output directory; the original planner refuses to overwrite an existing cost-table or plan. The new thin entry point selects the frozen route-specific cost table by default. `--cost-table` can select another compatible schema2 table. This performs offline computation only: it does not re-encode, deploy weights, run inference, change the phone or retrain the cost model.

`nodes` means actual Target verification input rows, possibly including pending, not tree depth or final output length. The model plans24layers with0..32upgraded joint shards/layer, against measured2/3-bit endpoints. It simulates the layer/slot pipeline and source eviction, then searches for read/downstream balance. Direct denominator is execution union eviction; staged denominator additionally includes copy. Overlapping work is counted once.

Default tolerance is ratio[1/1.02,1.02]. Within that band the precision policy prefers more upgraded shards, then lower latency. This shard-count preference is a quality proxy, not measured accuracy optimization. Candidate counts use deterministic multistart coordinate search, not a global optimum certificate. Within each layer the frozen old NLL ranking chooses the identities. No test-set fitting is added.

## Frozen results and limitations

| Cost route and verification rows | Q3/Q2 joint shards | Predicted ratio |
| --- | --- | --- |
| direct8 |573/195|1.01999|
| direct10 |672/96|1.01986|
| direct12 |763/5|0.98160|
| staged8 |0/768|1.38407, balance NOT reached by search|

The staged result is NOT the573model tested on the phone. The phone measured that same frozen573model on both routes as a controlled model comparison; it did not deploy a staged-specific optimization result. Whole-Q4 phone tests do not invoke a mixed-bit planner.

**These costs are from earlier deployments, mainly CPU4 for direct and Q2/Q3 staged calibration. They have not been refitted to the latest CPU8/no-mid-logits20-question results.** Do not label output as newly calibrated or dynamically rerun by the phone. More precise latest-device planning requires new compatible per-layer/bit/row cost fitting and validation; aggregate20-question means are not interchangeable with those independent anchors.

No certified total-memory model exists. Requested hard memory budgets are refused rather than pretending sampled RSS guarantees them. Mixture latency uses interpolation of pure-format endpoints, not a separately measured cost for every mixed combination. Explicit outside-range mode clamps endpoints and labels it, rather than inventing a new Tile curve.

`workspace/.../planner/MULTIBIT-20261024.md` preserves historical missing-calibration text but explicitly supersedes it at the top with completed staged calibration. `staged-q23-calibration` in older output folders is a historical missing-input snapshot, not the current staged8result. Use `multibit-staged-calibrated-20261024/n8` for that result.

## Artifacts

- `plan.json`: route, row count, per-layer upgrade counts, predicted timeline and balance status.
- `assignments.tsv`: layer/shard/bits, used for new mixed encoding. It must not be fed silently into an older two-column Q4 selector.
- `cost-table.json`: copied route-specific frozen model; original source hashes retained.
- `workspace/.../planner/output`: legacy schema1 Q2/Q4 cost table and plan.
- `workspace/.../planner/tree_profile.py`: legacy runtime initialization export. Its V1..64 curve is complete Tail verification cost, not only pure NPU arithmetic. Original export uses source traces and CPU priors, included for regression tests.
- `workspace/.../tree-initial-146/initial-costs.json`: historical CPU full-action priors; not newly measured CPU8 curves.

Selected raw logs, deployment hash manifests and the768-entry ranking are fixtures needed by the original tests and profile rebuilds; no model weights or calibration text dataset is copied. Use the main archive's `verify.ps1` after export to check file integrity.
