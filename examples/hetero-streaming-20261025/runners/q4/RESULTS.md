# AllQ4 CPU8 HumanEval20 completed

40/40 generation runs exit0, suite exit0. Model SHA aea79ab3c536fd4d1c38862117bacf1df9d96fdf09c91c745995b01654a65c33. Canonical shard group32Q4 Early8/Tail24/embedding/head; CPU8-only batched gain top4/depth10/cap63; all24Tailstreamed Q2AB; seven tensor coordinators.23unused intermediate vocab buffers removed in every run. Source cache eviction enqueued only after layer decode synchronization. See README for initial-cost and deployment differences versus573.

| Metric | Direct pread final HTP | O_DIRECT staging plus copy |
| --- | ---: | ---: |
| Aggregate t/s (2560/summed seconds)|3.548|4.150|
| Total generation seconds|721.573|616.824|
| Mean generation seconds|36.079|30.841|
| Mean Draft seconds|10.694|8.451|
| Mean Tail seconds|23.922|21.080|
| Mean pure read union seconds|22.953|19.199|
| Mean copy union seconds|0|5.382|
| Mean execute-host-sync union seconds|6.839|6.650|
| Mean eviction union seconds|12.255|0.003863|
| Mean execution union copy seconds|6.839|11.155|
| Mean execution union copy union eviction seconds|18.898|11.157|
| Read/(execution union copy)|3.356|1.721|
| Read/(execution union copy union eviction)|1.215|1.721|
| Maximum sampled owned peak GiB|2.199|2.308|
| Accepted/drafted|38.11%|40.36%|
| Total verification rounds|475|492|
| Target function passed|5/20|5/20|

Same passed questions in both routes:0,2,7,12,16. Each group has13functional test_failed,1syntax_or_truncated,1worker_terminated. Question10's generated make_palindrome loops while is_palindrome on an empty string, repeatedly slicing empty string to itself; worker returncode-9 under historical CPU resource limit. This is a generated nontermination bug, not a phone inference crash; retained as nonpassed. No generation failures discarded.

Scoring reuses exact historical SHA-pinned restricted worker and extraction-v2, fixed128 saved output, removes exact prompt prefix, no code repair or new generation. Container networknone,read-onlyinput/root,nonroot65534,capdropALL,no-new-privileges,1GiB,2CPU,128pids,tmpfs256MiB. Full outputs/errors/identities in isolated-grade/quality.json. Raw text exact17/20 across routes, not equal to functional correctness.

Performance interval analysis verifies24executions per verification,7tensor loads and7evictions per layer, ordered layer coverage, all loader/advice/eviction successes. Payload per-pass consistent; total actual payload1398.591GB direct and1448.645GB staged. Process storage/actual payload approximately1.018 both, not a warm-cache-only benchmark. Independent interval unions overlap and must not be added serially. Execute includes host dispatch/backend/sync, not pure HMX arithmetic. Peak excludes unmapped file cache and driver-only allocations, not an OS cap.

Previous573Q3/195Q2 with Q4Early/aux and same no-mid-logits setting: direct4.583t/s,peak2.142GiB,passed0/20;staged5.276t/s,peak2.256GiB,passed0/20. AllQ4 gives higher measured20-question correctness but slower throughput. Initial gain cost profiles, resulting tree shapes and rounds differ, and suites were sequential not randomized/repeated; this is whole-system observed comparison, not an isolated quantization-kernel speedup or general model-quality proof.

Artifacts: summary.json and metrics.py/summarize.py; original phone-results; isolated-grade/quality.json.
