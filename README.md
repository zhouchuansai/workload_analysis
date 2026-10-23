# Load Theory

面向未来昇腾 NPU 的模型负载理论建模库。当前版本用于快速验证建模口径，不承诺 kernel 级精度。

## 当前能力

- Python 定义 Dense / GQA / MLA / MoE 模型。
- 自动解析本地 DeepSeek V4.1 `config.json`。
- 使用 `configs/scenario.json` 统一配置 workload、并行策略、精度和 optimizer。
- 分别计算 prefill、decode 和 training。
- 支持多轮串行 Agent 任务的端到端推理负载建模。
- 输出 Cube/Vector 计算需求、访存、KV cache、通信量；对外报告统一使用 GFLOPs。
- 输出：
  - 计算访存比：`total_flops / data_movement_bytes`
  - 算容比：`total_flops / weight_bytes`
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

## Agent 端到端推理

Agent 任务由多轮串行模型交互组成。每轮分别建模当前完整 prompt 的 prefill，以及 `output_length` 个输出 token 的 decode，最后把所有轮次需求累加成任务级端到端结果。

```bash
/opt/anaconda3/envs/torch/bin/python -m load_theory.cli models/deepseek-v4.1/config.json \
  --scenario configs/scenario.json \
  --mode agent \
  --agent-scenario configs/agent_scenario.json \
  --json reports/deepseek_agent_load_report.json
```

也可以运行：

```bash
/opt/anaconda3/envs/torch/bin/python -m load_theory.agent.example
```

`configs/agent_scenario.json` 中每轮包含：

- `context_before_length`：本轮之前已经存在的上下文长度。
- `new_input_length`：本轮新增输入 token 数。
- `output_length`：本轮模型输出 token 数。
- `cached_length`：本轮可复用的已计算前缀长度；可精确描述多轮 Agent 的共享上下文。

每轮的 `prompt_length = context_before_length + new_input_length`。若 `cached_length` 小于该轮 prompt 长度，则仅计算剩余 suffix，同时从 cache 读取命中前缀。

Agent 任务级 `total` 等于：

```text
sum(round_prefill_total + decode_per_output_token × batch_size × output_length)
```

`cache_load_scope` 控制 cache 搬移计费方式：

- `per_round`：每轮需要从 host/peer/remote/disk 搬入 cache 时都计一次。
- `once`：整个 Agent 任务只计一次，适合远端 cache 搬入后常驻本地 HBM 的场景。
- `none`：不计 cache 搬移量，仅保留计算和常规访存。

Agent 中的 `weight` 不是每轮从 host/remote 重新装载一份模型权重，而是 HBM 到 Cube/Vector 的理论读放量。decode 阶段按每个 decode step 读取一次激活权重，同一 batch 内多条序列共享这次读放，不按序列数重复放大。prefill 阶段仍按一次 forward 的权重访问上界估计。若要分析服务冷启动的权重装载量，应单独按模型权重大小计一次，当前 `cache_load` 只描述 KV cache 的额外搬入。

Agent report 单独输出 `capacity`：

- `weight_hbm`：全量模型权重 HBM 常驻容量，不随轮次累计。
- `active_weight`：单 token 激活权重容量下限。

`weight` 仍表示 HBM 到计算单元的理论读放；它是带宽需求，不是容量需求。若只关心模型是否放得下，应查看 `capacity.weight_hbm`，而不是累加每轮 `weight`。

Agent report 的 prefill 归一化指标分为两类：

- `prefill_cube_gflops_per_computed_prompt_token`：平均每个实际参与计算的 prompt token 的 Cube 计算量。
- `prefill_cube_gflops_per_first_token`：平均每次产出 first token 的 prefill 端到端 Cube 计算量。

前者用于分析 prompt 计算强度，后者用于分析一次 prefill 调用的端到端需求。

每轮 Agent 交互还会输出 `dag`：

- `nodes`：`source`、每轮 `prefill`、每轮 `decode`、`sink`。
- `edges`：`task_start`、`first_token`、`context`、`task_end`。
- `depends_on`：可在 `turns[]` 中显式声明上游轮次；未声明时按输入顺序串行连接。

CLI 的 JSON 输出与普通推理/训练报告一致，外层是一个 report 数组；Agent report 位于数组第一个元素。报告末尾的 `field_notes` 会解释 Agent 专属字段和既有需求字段的含义。

## Scenario 配置

`configs/scenario.json` 是模型无关的统一配置文件，包含：

- `parallel`：TP / EP / PP / DP。
- `precision`：权重、expert 权重、activation、KV cache 和 gradient 的字节数。
- `inference`：prefill batch / length，decode batch / length，context length。
- `training`：global batch size、sequence length、gradient checkpointing。
- `optimizer`：默认 `adamw`。

`inference.cache` 表示 prompt 前缀 KV cache 策略。默认不命中、cache 常驻 HBM，因此不产生额外加载量：

```json
{
  "inference": {
    "cache": {
      "hit_rate": 0.0,
      "source": "hbm",
      "load_required": false,
      "entry_count": 8,
      "block_size": 1,
      "compression_ratio": 1.0
    }
  }
}
```

字段含义：

- `hit_rate`：prompt 前缀 KV cache 命中率。
- `source`：cache 所在位置，支持 `hbm`、`host_memory`、`peer_npu`、`remote_node`、`disk`。
- `load_required`：是否需要把 cache 搬入本地 HBM。
- `entry_count`：本 batch 中需要加载的独立 cache 前缀数量；多条请求共享同一个系统提示词时可设为 1。
- `block_size`：cache block 粒度。
- `compression_ratio`：cache 压缩后的字节数比例，`1.0` 表示不压缩。

Agent 场景通常存在稳定系统提示词和多轮共享上下文，可将该值设置为较高水平，例如 `0.6` 到 `0.95`。

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

### Prefill 口径

`prefill` 表示端到端处理一段 prompt 并产出第一个输出 token 的阶段：

1. 对整段 prompt 做一次 forward。
2. 按 causal attention 处理所有 prompt token。
3. 写入整段 prompt 的 KV cache。
4. 只对每个序列的最后一个位置做 LM head，产出 first token。

如果启用 prompt 前缀 KV cache，则命中的前缀不参与 prefill forward 计算，只作为已有 KV 供未命中的 suffix attention 读取。为保证能产出 first token，至少保留最后一个 prompt token 参与计算：

```text
raw_cached_length = floor((prompt_length - 1) × hit_rate)
cached_length = floor(raw_cached_length / block_size) × block_size
computed_length = prompt_length - cached_length
```

当 `cache_hit_rate = 1.0` 时，只计算最后一个 prompt token，并用完整 cached KV 生成 first token。

如果 cache 不在 HBM 中且 `load_required = true`，额外加载量为：

```text
cache_load_bytes =
    entry_count
  × cached_length
  × num_layers
  × kv_bytes_per_token_per_layer
  × compression_ratio
```

`cache_load` 只计入 prefill 的端到端数据搬移量，decode 场景为 0。

prefill JSON 报告只提供端到端口径：

- `total`：整个 prefill batch 的端到端需求。

不输出 `per_prompt_token` 或 `per_first_token` 归一化字段；所有 prefill 指标均基于该端到端阶段计算。

当前 causal attention 计算采用：

```text
F_attention = 2 × L × H × h × S × (S + 1)
```

而不是把每个 prompt token 都按完整上下文 `S` 计算。

### Decode 口径

`decode` 表示输出单个 token 的阶段。JSON 只输出 `per_output_token`：

```text
per_output_token = 平均输出一个 token 的需求
```

当 `decode_length > 1` 时，上下文长度按 autoregressive 过程求平均：

```text
average_context =
    context_length + (decode_length - 1) / 2
```

因此 decode 指标反映的是整段输出过程中的平均单 token 负载，而不是整个 decode batch 的聚合负载。

JSON 本身不支持注释，因此报告末尾输出 `field_notes` 字段，逐项说明每个字段的含义和单位。

### Cube GFLOPs

矩阵乘、GEMM、QK^T、V·attention-prob 的理论乘加量，输出单位为 `10^9 FLOPs`。

### Vector GFLOPs

RMSNorm、SwiGLU、softmax 相关非矩阵操作的元素级操作数，输出单位为 `10^9 vector ops`。它不是 Cube 计算量的替代品，而是昇腾 Vector 单元的独立负载。

### 数据搬移与内存单位

JSON 中的内存/字节类字段统一输出为对象：

```json
{
  "value": 123.456,
  "unit": "GB"
}
```

默认单位为 `GB`；当字节数达到或超过 `1 TB` 时，单位切换为 `TB`。

主要字段包括：

- `weight`：模型总权重大小/读放估计。
- `active_weight`：单 token 实际激活的权重大小。
- `kv_cache`：KV cache 访存量。
- `activation`：激活值访存量。
- `communication`：TP/EP/PP/DP 理论通信量。
- `cache_load`：从 host/peer/remote/disk 等位置搬入 HBM 的 prompt KV cache。
- `optimizer_state`：训练 optimizer state。
- `gradient_state`：训练梯度状态。
- `data_movement`：weight + KV + activation + communication + cache_load 的理论数据搬移量。

`data_movement` 不是 HBM 常驻容量，而是理论数据搬移量。当前版本没有单独输出 HBM 常驻容量；如需评估容量，应另行组合模型权重、KV cache 状态、activation 状态和训练状态。

### 算容比

本仓库使用：

```text
compute_capacity_ratio = total_flops / weight_bytes
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
- `gradient_state` 和 `optimizer_state` 是训练状态大小，不进入 `data_movement` 的数据搬移口径。
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
