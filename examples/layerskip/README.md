# LayerSkip prototype

This example runs greedy self-speculative decoding for a LayerSkip-trained Llama model.

The model is loaded once. Two contexts reference the same model tensors:

- the early context evaluates `[0, exit_layer)` and owns KV only for those layers;
- the tail context consumes the early residual stream, evaluates `[exit_layer, n_layer)`, and owns KV only for those layers.

The target pass does not evaluate the early layers again. Together, the two contexts hold one logical full-model KV state split at the exit layer.

This first prototype is CPU-only and supports the Llama architecture. It uses greedy sampling and partial KV rollback after rejection.

Example:

```sh
./bin/llama-layerskip \
    -m /data/local/tmp/gguf/layerskip/layerskip-llama2-7B-Q4_0.gguf \
    -p "Explain speculative decoding briefly." \
    -n 32 \
    -c 512 \
    -b 128 \
    -t 4 \
    --exit-layer 8 \
    --draft-max 6 \
    --ignore-eos
```

The log reports both `accept`, which is `n_accept / n_drafted`, and `window accept`, which is `n_accept / (n_verify * draft_max)`.

For streamed Tail weights, `--tail-stream-cluster-sizes` sets the exact sizes of consecutive nonresident groups. For example, with 32 layers, `--exit-layer 8` and `--tail-stream-cache-layers 7`, use `--tail-stream-cluster-layers 4 --tail-stream-cluster-sizes 4,4,3,3,3`. The cached prefix is grouped by `--tail-stream-cluster-layers`; the listed streamed groups cover the remaining 17 layers. The program rejects a list whose sum does not match the remaining Tail layers. The memory plan uses the largest listed group's weight size for each stream slot. This is an application-level plan, not a process memory limit.
