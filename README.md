# Load Theory

面向未来昇腾 NPU 的模型负载理论建模库。当前版本用于快速验证建模口径，不承诺 kernel 级精度。

## 当前能力

- Python 定义 Dense / GQA / MLA / MoE 模型。
- 自动解析本地 DeepSeek V4.1 `config.json`。
- 使用 `configs/scenario.json` 统一配置 workload、并行策略、精度和 optimizer。
- 分别计算 prefill、decode 和 training。
- 输出 Cube/Vector 计算需求、访存、KV cache、通信量；对外报告统一使用 GFLOPs。
- 输出：
  - 计算访存比：`total_flops / total_memory_bytes`
  - 算容比：`total_flops / total_weight_bytes`
  - 激活算容比：`total_flops / active_weight_bytes`
  - Cube/Vector 比例。
- 支持 TP / EP / PP / DP 的理论通信字节数。
- 支持模糊模型 `FuzzyModel`，适合没有完整 config 的未来结构。

## 运行本地 DeepSeek 示例

```bash
/opt/anaconda3/envs/torch/bin/python -m load_theory.cli models/deepseek-v4.1/config.json \
  --scenario configs/scenario.json \
  --json reports/deepseek_v41_load_report.json
```

也可以运行：

```bash
/opt/anaconda3/envs/torch/bin/python -m load_theory.examples.deepseek_v41
```

## Scenario 配置

`configs/scenario.json` 是模型无关的统一配置文件，包含：

- `parallel`：TP / EP / PP / DP。
- `precision`：权重、expert 权重、activation、KV cache 和 gradient 的字节数。
- `inference`：prefill batch / length，decode batch / length，context length。
- `training`：global batch size、sequence length、gradient checkpointing。
- `optimizer`：默认 `adamw`。

暂不建模量化 scale 的额外存储开销。

AdamW 的 optimizer state 固定按 12 bytes/param 计算：

```text
FP32 master weight: 4 bytes
FP32 momentum:      4 bytes
FP32 variance:      4 bytes
```

因此配置文件里不需要单独提供 master weight 字节数。

CLI 参数可以在 scenario 基础上覆盖，例如：

```bash
/opt/anaconda3/envs/torch/bin/python -m load_theory.cli models/deepseek-v4.1/config.json \
  --scenario configs/scenario.json \
  --mode inference \
  --context-length 16384
```

## 指标口径

JSON 本身不支持注释，因此报告末尾输出 `field_notes` 字段，逐项说明每个字段的含义和单位。

### Cube GFLOPs

矩阵乘、GEMM、QK^T、V·attention-prob 的理论乘加量，输出单位为 `10^9 FLOPs`。

### Vector GFLOPs

RMSNorm、SwiGLU、softmax 相关非矩阵操作的元素级操作数，输出单位为 `10^9 vector ops`。它不是 Cube 计算量的替代品，而是昇腾 Vector 单元的独立负载。

### 访存

拆分为：

- `weight_bytes`：模型总权重大小。
- `active_weight_bytes`：单 token 实际激活的权重大小。
- `kv_cache_bytes`：上下文 KV 读/写量。
- `activation_bytes`：激活访存。
- `communication_bytes`：TP/EP/PP/DP 理论通信量。

### 算容比

本仓库使用：

```text
compute_capacity_ratio = total_flops / total_weight_bytes
```

MoE 另外输出：

```text
compute_capacity_ratio_active = total_flops / active_weight_bytes
```

前者衡量模型容量效率，后者更接近 decode 阶段的访存压力。

### JSON 计算字段

- `cube_gflops`：Cube 单元计算需求。
- `vector_gflops`：Vector 单元计算需求。
- `total_gflops`：`cube_gflops + vector_gflops`。

内部计算保留 FLOPs 精度，仅在报告输出层统一转换为 GFLOPs。

## 建模假设

- 训练总 FLOPs 近似为 `3 × forward FLOPs`。
- AdamW 状态按 `4 bytes master weight + 8 bytes m/v` 计算。
- `gradient_bytes` 和 `optimizer_bytes` 是训练状态大小，不进入 `total_memory_bytes` 的访存流量口径。
- TP all-reduce 使用环状算法字节量 `2 × (N-1) / N × message_size`。
- MoE EP 通信按 dispatch + combine 估算。
- Activation checkpointing 使用平方根重计算近似。
- DeepSeek V4.1 的 sparse attention、compress_ratios、indexer、engram、DSpark/MTP 和 vision 模块目前不精确建模。

这些假设会在后续版本逐步细化，并在报告 metadata 中显式暴露。

## 后续计划

1. 精确 DeepSeek V4.1 sparse attention / indexer / engram 建模。
2. 内置 GDN、KDA、linear attention、hybrid attention 原型。
3. 给定目标 tokens/s 和 latency，反推未来 NPU 的 Cube、Vector、HBM 和互联需求。
4. 模糊输入的区间估计和敏感性分析。
5. 每层模块级明细和 Markdown 报告。
