# llama.cpp/examples/speculative

Demonstration of speculative decoding and tree-based speculative decoding techniques

The token tree uses one target-model decode call per verification step. Drafted nodes are reordered in depth-first order before verification. Sequence memberships encode the ancestor relation, so the regular causal attention mask allows each node to attend only to the prompt and its ancestors.

Tree shape can be controlled independently:

```sh
--parallel 4 \
--spec-tree-max-depth 4 \
--spec-tree-max-nodes 12 \
--spec-tree-max-leaves 4
```

`--parallel` sets the allocated sequence capacity. `--spec-tree-max-leaves` must fit in that capacity. If a tree option is omitted, depth and node limits use `--spec-draft-n-max`, while the leaf limit uses `--parallel`.

The implementation uses unified KV metadata for branch sharing. It checks the DFS parent order and the sequence-based ancestor relation before each target verification call. The final log separates target enqueue and synchronization time, draft enqueue and synchronization time, tree layout time, and KV metadata time.

This path batches all tree nodes into one regular target decode call. It does not use a dedicated sparse tree-attention kernel in the Hexagon backend.

For Hexagon operator timing, set `GGML_HEXAGON_PROFILE=1`. The backend then logs each HTP operation, including flash attention and matrix multiplication. The example-level phase counters remain useful for host-side KV, tree construction, enqueue, and synchronization costs.

`--spec-tree-pad-nodes` pads target verification to a constant `--spec-tree-max-nodes` shape. This can improve graph reuse on accelerators, but it also computes unused padding nodes. Keep it disabled unless an A/B test shows a lower total decode time.

Use `--spec-tree-widths 1,1,2,1` to select a bounded static tree. Each value is the maximum number of children created for every active node at that depth. This example permits up to 6 drafted nodes and 2 leaves. Side paths with a cumulative probability below `--spec-draft-p-split` are pruned, while the highest-probability child of the primary path is always retained. The tree derives its depth, node, and leaf limits and batches all nodes at the same depth in one draft decode call.

Use `--spec-tree-p-ratio` to create side children only when their local probability relative to the top-1 child reaches the configured ratio. The cumulative path threshold still applies after this local branch check.

Use `--spec-tree-frontier` with dynamic tree limits to select nodes by calibrated path gain per draft cost. `--spec-tree-frontier-min-gain` keeps candidates whose marginal rate is at least the configured fraction of the best candidate at that depth. Unselected paths remain as leaves instead of being discarded. This mode cannot be combined with `--spec-tree-widths`.

For greedy target sampling, the draft tree reads top-k candidates directly from the draft logits. The candidate count is the configured width at each depth, so `--spec-tree-widths 1,2,1,1` keeps the draft top-2 candidates at the branch while target output remains greedy.

Set `LLAMA_SPEC_PIPELINE=1` to overlap the final draft decode of the current tree with its target verification. The final draft decode is still required to update the draft KV cache, but its logits are not needed to build the current tree. The code joins the draft task before the next speculation round can use that KV state. Leave the variable unset for the serial control.

Set `LLAMA_SPEC_TRACE=/path/to/trace.json` to record a Chrome JSON trace with CPU draft/control and target host-call lanes. The trace includes acceptance, KV selection, draft decode, tree layout, target KV preparation, target verification, and pipeline join intervals. Set `GGML_HEXAGON_PROFILE=1` at the same time to correlate the target host interval with Hexagon submit, receive, and DSP execution times in the backend log.

More info:

- https://github.com/ggml-org/llama.cpp/pull/2926
- https://github.com/ggml-org/llama.cpp/pull/3624
- https://github.com/ggml-org/llama.cpp/pull/5625
