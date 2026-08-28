# OpenCL GPU 部署

## 实验目的

验证 llama.cpp 的 OpenCL GPU 后端能否在手机上完成 Llama-3.2-3B 推理。

## 实验设置

- 日期：2026-03-25
- 设备：手机，原始记录未注明具体型号
- 模型：Llama-3.2-3B
- 后端：llama.cpp OpenCL GPU
- 量化格式、Prompt、生成长度、线程数和代码版本：原始记录未注明

## 实验结果

- Llama-3.2-3B 在手机 OpenCL GPU 后端成功运行。
- 原始记录未包含可比较的吞吐、延迟、内存或功耗数据。

## 实验结论

实验只证明了手机 GPU 推理路径可用，不能据此判断 GPU 相对 CPU 或 NPU 的性能优劣。
