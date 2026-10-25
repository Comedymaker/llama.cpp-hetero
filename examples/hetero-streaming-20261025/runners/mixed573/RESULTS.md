# Frozen573 CPU8 HumanEval20 without unused intermediate logits

40/40 completed, run.exit=0. Each case confirmed23 boundary vocab-output allocations removed. Same frozen573Q3/195Q2 Tail, Q4 Early/aux, CPU-only batched gain tree, fixed128 tokens. No quantization/kernel correctness fix introduced.

| Metric | Direct pread to HTP | O_DIRECT staging plus copy |
| --- | ---: | ---: |
| Aggregate throughput (2560 / summed seconds) |4.583 t/s|5.276 t/s|
| Total generation seconds,20 questions|558.579|485.178|
| Mean generation seconds|27.929|24.259|
| Mean Draft seconds|8.174|7.006|
| Mean Tail seconds|18.500|16.149|
| Mean pure read interval union seconds|16.446|13.184|
| Mean copy union seconds|0|3.446|
| Mean execute-host-sync union seconds|9.518|8.906|
| Mean eviction union seconds|8.485|0.027|
| Read / (execute union copy union eviction)|0.956|1.231|
| Maximum sampled owned peak GiB|2.142|2.256|
| Accepted / drafted|47.35%|49.71%|
| Verification rounds,total|454|438|
| HumanEval target function passed|0/20|0/20|
| Syntax/truncated|5|6|
| Functional test failed|15|14|

Quality scored from saved texts, no regeneration or repair, original SHA-pinned restricted worker plus existing extraction-v2. Isolated Docker: no network, read-only root/input, nonroot65534, capabilities dropped, no-new-privileges,1GiB,2CPU,128pids,tmpfs256MiB. All scoring completed without infrastructure failures. quality.json preserves exact worker/extractor/input SHA identities and per-question failures. These fixed128 scores are not unrestricted model capability.

Previously CPU8 with intermediate buffers: direct5.350t/s and peak3.552GiB; staged5.910t/s and peak3.658GiB. Current peak reductions1.410/1.402GiB. Current throughput is lower, not a demonstrated speedup. These were separate sequential suites with adaptive tree batch choices and different verification counts (previous416/431 vs current454/438), not a repeated same-tree randomized allocation-only speed comparison; no causal attribution of slowdown to buffer removal. Both previous CPU8 quality groups also0/20.

Independent unions overlap and must not be added as serial stage durations. Execute includes host/backend/sync, not pure HMX arithmetic. Owned peak excludes unmapped file cache and driver-only memory; not an enforced DRAM cap. Inputs,raw40logs,texts,peaks,statuses in phone-results; summary.json contains interval calculations and coverage checks; isolated-grade/quality.json contains functional evaluation.
