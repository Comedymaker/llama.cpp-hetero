# Same-Q4 CPU8 Target20 completed

20/20 completed exit0. Model SHA aea79ab3c536fd4d1c38862117bacf1df9d96fdf09c91c745995b01654a65c33 matches allQ4 hetero20 exactly. CPU8 noaffinity, all32layers autoregressive CPU, noDraft/tree/NPU. Context2048, fixed128 greedy ignoreEOS, same HumanEval000-019 prompts. Existing CPU residency executable/libraries unchanged and hashes saved.

Short probes selected model budget2212MiB with runtime reserve512MiB. Actual13/32cached layers, cached bytes1595310080, fixed weights591020032, active streamed layer122716160. Remaining19layers streamed. Planner total2845917184 is not actual measured RSS. All20 full-run sampled owned peaks2.201..2.219GiB; maximum2.219GiB. No DMA-BUF. This matches approximately hetero direct2.199GiB; hetero staged2.308GiB is slightly higher. Not hardOSmemory cap, excludes unmapped file cache and driver memory.

Aggregate127eval/time throughput0.7123278104t/s (2540evals /3565.77402seconds). Mean eval178.288701s/question; mean prefill12.4831925s excluded. For generation-count convention used by hetero,128/time gives0.7179366908t/s; this includes first output that comes from prefill, not an extra128th decode eval. Report both conventions explicitly.

Same frozen target-function worker plus extraction-v2:5/20, passed questions0,2,7,12,16, identical pass set to bothQ4heteroroutes.13functional failures,1syntax/truncated,1workerterminated (question10). No new generation/code repair in scoring. Scores are fixed128, not unlimited capability or proof all token outputs equal. Isolation networknone/read-onlyinput/nonroot/capdrop/resource limits retained. Perquestion errors and SHA identities in isolated-grade/quality.json.

| Quantity | CPU Target8 | Q4 heterogeneous direct8 | Q4 heterogeneous staged8 |
| --- | ---: | ---: | ---: |
| Generation-count throughput t/s |0.718|3.548|4.150|
| Maximum sampled owned GiB |2.219|2.199|2.308|
| Target-function passed |5/20|5/20|5/20|

Observed hetero/Target ratios using generation-count throughput approximately4.94x direct/5.78x staged. This is observed whole-system comparison with same model/thread/prompt/length and approximate peak, not an isolated hardware speed claim. CPU Target libraries/kernels differ from hetero deployment; runs sequential, no randomized repeats. Target test battery temperature reached44.9C at final start versus earlier hetero suite thermal conditions, so thermal drift may depress CPU baseline. No lower-temperature repeat performed here. CPU Target one-layer mmap residency loader does not implement the hetero staging/eviction pipeline. Source reads shown in residency logs include process scope; not pure per-generation UFS timing.

Artifacts: summary.json, raw phone-results including all probe/full logs and peak samples; isolated-grade/quality.json.
