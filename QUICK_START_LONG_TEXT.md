# 快速配置指南：处理长数学证明

针对你的情况（output 可能达到 16384 tokens），这里是推荐的配置。

## 📋 推荐配置

打开 `config.py`，设置以下参数：

```python
# ==================== 数据处理配置 ====================
MAX_SEQUENCE_LENGTH = 16384  # 设置为最大值以容纳长证明
TRUNCATE_LONG_SEQUENCES = True  # 自动截断超长部分
SHOW_LENGTH_WARNINGS = True  # 显示哪些样本被截断

# ==================== 数据配置 ====================
DATA_FILES = [
    "data/s1k.json",  # 你的数据文件
]

# ==================== 模型配置 ====================
BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct"  # 使用 Instruct 模型更适合数学推理
LORA_RANK = 32

# ==================== 训练超参数 ====================
LEARNING_RATE = 5e-5  # 对于数学推理，使用较小的学习率
NUM_EPOCHS = 5  # 数学推理通常需要更多轮次

# ==================== 采样配置 ====================
SAMPLING_MAX_TOKENS = 2048  # 测试时生成较长的响应
SAMPLING_TEMPERATURE = 0.3  # 较低温度，让推理更确定
SAMPLING_NUM_SAMPLES = 2
```

## 🎯 为什么这样配置？

### 1. MAX_SEQUENCE_LENGTH = 16384
- 你的数据可能很长，设置为最大值
- 如果训练时内存不足，可以降到 8192

### 2. TRUNCATE_LONG_SEQUENCES = True
- 自动截断过长部分，不会丢失样本
- 对于数学证明，前面的推理步骤通常最重要

### 3. BASE_MODEL 选择 Instruct 版本
- Instruct 模型已经过指令微调，更擅长遵循指令
- 对于数学推理任务，Instruct 版本通常效果更好

### 4. LEARNING_RATE = 5e-5 (较小)
- 数学推理是精确任务，需要更谨慎的学习
- 避免破坏模型原有的推理能力

### 5. NUM_EPOCHS = 5
- 数学推理模式需要更多训练轮次
- 观察损失曲线，如果还在下降可以增加

## 📊 运行后查看统计

运行训练后，注意查看这些信息：

```
Sequence length statistics:
  Min: 245 tokens
  Max: 15234 tokens  ← 如果这个值很大（>10000），说明需要 16384
  Mean: 3456.7 tokens
  Median: 2891.0 tokens  ← 中位数，大部分样本的长度
```

```
Processing summary:
  Successfully processed: 980 examples
  Truncated: 45 examples  ← 如果这个数字很大，考虑增加 MAX_SEQUENCE_LENGTH
  Skipped (too long): 0 examples
  Failed (errors): 0 examples
```

## ⚙️ 根据统计调整

### 如果大部分样本都被截断（Truncated > 30%）

**选项 1：增加长度但减少其他开销**
```python
MAX_SEQUENCE_LENGTH = 16384  # 保持最大
LORA_RANK = 16  # 减少 rank 以节省内存
```

**选项 2：接受截断**
- 如果证明的前半部分已经包含主要推理
- 截断不会严重影响训练效果

### 如果内存不足 (Out of Memory)

**选项 1：减少序列长度**
```python
MAX_SEQUENCE_LENGTH = 8192  # 降到 8K
```

**选项 2：使用更小的模型**
```python
BASE_MODEL = "meta-llama/Llama-3.1-8B"  # 更小的模型
```

**选项 3：减少 LoRA rank**
```python
LORA_RANK = 16  # 从 32 降到 16
```

## 🚀 开始训练

1. 确保配置文件已更新
2. 设置 API 密钥：
   ```bash
   export TINKER_API_KEY=7624312694161539072
   ```
3. 运行训练：
   ```bash
   python train_with_config.py
   ```

## 📈 监控训练

观察这些指标：

1. **损失下降**：应该逐渐降低
2. **截断数量**：如果太多，考虑增加 MAX_SEQUENCE_LENGTH
3. **训练速度**：16384 会比较慢，这是正常的

## 💡 优化建议

### 阶段性训练策略

**第一阶段：快速测试**
```python
MAX_SEQUENCE_LENGTH = 4096  # 较短，训练快
NUM_EPOCHS = 2
```
快速验证数据和流程是否正确

**第二阶段：完整训练**
```python
MAX_SEQUENCE_LENGTH = 16384  # 完整长度
NUM_EPOCHS = 5
```
使用完整长度进行正式训练

这样可以先快速发现问题，再进行耗时的完整训练。
