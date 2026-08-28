# Qwen3-4B 与 Qwen3-0.6B 投机采样复测

## 实验目的

复测 Qwen3-4B target + Qwen3-0.6B draft 的投机采样端到端 decode 收益，并记录接受率和 prefill 现象。

## 实验设置

- 日期：待确认；原始记录未记录实验日期，因此不根据文件修改时间推断
- Target：Qwen3-4B-Q4_K_S.gguf
- Draft：Qwen3-0.6B-Q4_K_S.gguf
- 参数：生成 256 token；`ctx=4096`；`ctx-draft=4096`
- Prompt：解释 LLM 投机解码中的 draft model、target verification、接受率和加速原因
- 设备、后端、重复次数和代码版本：原始记录未注明

投机采样指令：

```bash
./build/bin/llama-speculative-simple \
    -m /data/data/com.termux/files/home/models/Qwen3-4B-Q4_K_S.gguf \
    -md /data/data/com.termux/files/home/models/Qwen3-0.6B-Q4_K_S.gguf \
    -p "Please explain speculative decoding in LLM inference. Focus on the draft model, target model verification, acceptance rate, and why it can speed up generation." \
    -n 256 \
    -c 4096 \
    -cd 4096 \
    2>&1 | tee bench-speculative.txt
```

## 实验结果

| 模式 | 主模型 | Draft 模型 | Prefill 吞吐 | Decode 吞吐 | 备注 |
| --- | --- | --- | ---: | ---: | --- |
| 基础模型 | Qwen3-4B-Q4_K_S.gguf | 无 | 47.2 t/s | 14.7 t/s | `llama-cli` Prompt/Generation |
| 投机采样 | Qwen3-4B-Q4_K_S.gguf | Qwen3-0.6B-Q4_K_S.gguf | 29.34 t/s | 15.428 t/s | `llama-speculative-simple` 整体 prompt eval/decoded |

| 模式 | 接受率 | Drafted/Accepted |
| --- | ---: | ---: |
| 投机采样 | 51.556% | 225/116 |

## 实验结论

- Decode 从 14.7 t/s 增至 15.428 t/s，约提升 4.95%；已观察到小幅收益，但距离稳定、显著加速仍有距离。
- Prefill 数值来自不同程序的整体 prompt eval。在统一计时口径前，不能直接认定投机采样使 prefill 下降 37.8%。
- 后续需要补齐日期、设备、后端、代码版本、重复次数和统一评测口径。
