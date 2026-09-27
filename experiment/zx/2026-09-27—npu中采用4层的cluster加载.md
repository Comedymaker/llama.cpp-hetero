# 2026-09-27 NPU 采用四层 Cluster 加载

## 1. 测试配置

| 项目 | 配置 |
| --- | --- |
| 模型 | `facebook/layerskip-llama2-7B`，F16 GGUF |
| 测试 | HumanEval/000 |
| 生成长度 | 128 token，`--ignore-eos` |
| CPU 线程 | 4 |
| Early Exit | 第 8 层 |
| Draft | CPU 前 8 层 |
| Target Tail | HTP 后 24 层 |
| Token 树 | `window=10`，`p-ratio=0.40`，单条完整旁支 |
| 流式单位 | 4 个 Tail 层/cluster，1 个流式 slot |
| 采样 | Greedy |

内存数值是应用层规划的常驻权重、流式缓冲和运行时预留总和，不是 cgroup 硬限制。

## 2. 总体结果

CPU Target 吞吐按 `127 runs / eval time` 计算；异构吞吐采用 `layerskip generation`。

| 内存预算 | CPU Target 常驻层 | HTP 固定常驻 Tail | CPU Target | 异构 | 异构相对 Target |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 GiB | 11/32 | 0 层 | 0.214 t/s | 0.543 t/s | 2.54x |
| 9 GiB | 16/32 | 4 层，1 cluster | 0.243 t/s | 0.558 t/s | 2.29x |
| 11 GiB | 21/32 | 8 层，2 clusters | 0.342 t/s | 0.678 t/s | 1.98x |
| 13 GiB | 27/32 | 12 层，3 clusters | 0.538 t/s | 0.777 t/s | 1.44x |

## 3. 异构结果

| 内存预算 | 规划总量 | Draft 构树 | Tail 验证 | HTP 加载次数 | HTP 加载量 | HTP 加载时间 | 生成时间 | 吞吐 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 GiB | 6.887 GiB | 22.760 s | 211.054 s | 156 | 252.585 GB | 116.169 s | 235.592 s | 0.543 t/s |
| 9 GiB | 8.770 GiB | 36.793 s | 190.852 s | 130 | 210.487 GB | 104.214 s | 229.458 s | 0.558 t/s |
| 11 GiB | 10.279 GiB | 33.198 s | 153.697 s | 104 | 168.390 GB | 82.607 s | 188.689 s | 0.678 t/s |
| 13 GiB | 11.787 GiB | 44.670 s | 117.356 s | 78 | 126.292 GB | 61.603 s | 164.745 s | 0.777 t/s |

四档的树统计完全一致：

| 指标 | 结果 |
| --- | ---: |
| `n_drafted` | 442 |
| `n_accept` | 103 |
| `n_verify` | 26 |
| `n_bonus` | 3 |
| Draft token acceptance | 23.303% |
| Window acceptance | 39.615% |
| Tree nodes | 442 |
| Tree branches | 25 |
| Tree leaves/verify | 1.962 |
| Side branch entries | 5 |
| Side branch accepted tokens | 18 |

四档异构运行的 `generated token ids` 完全一致。

## 4. CPU Target 结果

| 内存预算 | 常驻层 | 规划总量 | Peak RSS | 层加载次数 | 存储读取量 | Eval 时间 | 吞吐 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 GiB | 11/32 | 6.889 GiB | 6.194 GiB | 2709 | 1096.458 GB | 594.710 s | 0.214 t/s |
| 9 GiB | 16/32 | 8.774 GiB | 8.077 GiB | 2064 | 835.455 GB | 521.900 s | 0.243 t/s |
| 11 GiB | 21/32 | 10.659 GiB | 9.961 GiB | 1419 | 574.336 GB | 371.325 s | 0.342 t/s |
| 13 GiB | 27/32 | 12.921 GiB | 12.223 GiB | 645 | 261.167 GB | 236.168 s | 0.538 t/s |

## 5. 四层 Cluster 缓存结果

| 配置 | 吞吐 | 结果 |
| --- | ---: | --- |
| 7 GiB，cache=0，stream cluster=4 | 0.543 t/s | 最快的 7 GiB 配置 |
| 9 GiB，CPU/mmap cache=6，stream cluster=4 | 0.080 t/s | 缓存层与流式图反复切换，严重变慢 |
| 9 GiB，CPU/mmap cache=4，stream cluster=4 | 0.499 t/s | 修复驱逐范围后恢复 |
| 9 GiB，固定 HTP cache=4，stream cluster=4 | 0.558 t/s | 固定 HTP buffer 被验证图直接引用 |
| 11 GiB，固定 HTP cache=8，stream cluster=4 | 0.678 t/s | 两个完整 cluster 常驻 |
| 13 GiB，固定 HTP cache=12，stream cluster=4 | 0.777 t/s | 三个完整 cluster 常驻，本次最快 |

## 6. 结果结论

- 固定 HTP 常驻层每增加 4 层，单次实验的 HTP cluster 加载次数减少 26 次：`156 -> 130 -> 104 -> 78`。
- 7 GiB 到 13 GiB，异构吞吐从 `0.543` 提升到 `0.777 t/s`，提高 43.1%。
- 7 GiB 到 13 GiB，CPU Target 吞吐从 `0.214` 提升到 `0.538 t/s`，提高 151.9%。
- 内存增加时 CPU Target 提升更快，因此异构相对加速从 `2.54x` 降至 `1.44x`。
- 当前异构主要耗时仍是 Tail 验证；13 GiB 下为 117.356 s，占生成时间的 71.2%。
- 13 GiB 是当前四档中异构绝对吞吐最高的配置，但其相对 CPU Target 优势最小。


