# Gain-cost tree experiment

## Validated-backend compatibility build

`llama-layerskip-old-direct` is a separate executable target for the deployed October 2 direct-loader library set. It excludes newer prepared-read/commit APIs and rejects two stream slots and HTP overlap options. The regular executable target remains unchanged. Do not deploy newly built libraries with this compatibility executable: use all six libraries from `/data/local/tmp/layerskip-mixed-direct-prefill`, verify their SHA-256 hashes, and use the same DSP library path. Matching symbols is a startup check, not an ABI or performance proof; the linear output and memory regression test must pass before testing gain-cost trees.

The compatibility short test is `/data/local/tmp/llama-exp-gain-cost-old-backends-20261004`. It stops if the old n16 output differs, samples are missing, the sampled owned estimate exceeds 9 GiB, or DMA-BUF allocation exceeds 3.2 GiB. Earlier runs in `llama-exp-gain-cost-tree-smoke*` used a different backend library set and are not valid comparisons with the old deployment.

The optional `--tree-gain-cost` mode builds a parent-linked tree with a greedy frontier. It does not change the legacy linear or fixed-side tree modes, and it does not add intermediate target pruning.

Each selected node gets an Early sequence cloned from its parent. Target rows contain the sequence IDs of all descendants. Verification follows target argmax through matching children and commits only the accepted path in both KV caches. A fallback token is decoded by Early and its hidden state enters the next target batch.

Selection uses calibrated candidate reach probability divided by estimated extra drafting and verification latency. Candidate probabilities are top-k normalized Early probabilities multiplied by an online coverage reliability factor (initial value 0.9). Coverage feedback is collected only on the verified acceptance path. This is a heuristic approximation, not a reproduction of Lever's calibration implementation.

Verification profiles use actual batch rows (including a pending fallback row) and leaf counts. Observed full target decode time includes loading, compute and cleanup; nested preload metrics are not added. Unseen shapes use the nearest observation and configurable per-row/per-leaf penalties. This first version does not learn those marginal penalties automatically. Profiles and reliability reset per process.

Initial parameters should come from measurements with the same model, backend, weight placement and memory budget. Defaults are placeholders, not device measurements. For the 9 GiB cache7 smoke test, the starting verification base is 6000 ms, with 20 ms per row and 5 ms per leaf. Marginal node decoding starts at 100 ms and is updated online from non-first nodes. The first node after switching from Tail can incur page faults; its measured cost is already included in elapsed drafting time and is not used as the marginal cost of subsequent warm expansions.

Example additions to the existing direct-loading command:

```sh
--draft-max 10 \
--tree-gain-cost \
--tree-max-nodes 20 \
--tree-top-k 2 \
--tree-verify-base-ms 6000 \
--tree-node-ms 20 \
--tree-leaf-ms 5
```

Do not combine this mode with `--tree-single-branch`, tree overlap options or target-only flags. The first version supports at most 63 nodes (64 sequence IDs including the committed prefix). `--draft-max` is maximum depth, not a guaranteed chain length. Existing `--ignore-eos` behavior is unchanged and still differs from llama-completion EOS masking; compare against the same layerskip sampling behavior.

More branches can increase KV, graph and shared-buffer allocations. A planning budget is not proof of actual peak memory. Inspect the same-time RSS + DMA-BUF minus overlapping RSS measurements before classifying a budget as satisfied.
