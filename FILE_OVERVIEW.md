# Tinker 微调完整工具包

完整的 Tinker API 微调工具集，包含训练、导出、评估的全流程支持。

## 📁 文件清单

### 核心训练脚本
1. **train_with_config.py** - 推荐使用的训练脚本
   - 使用配置文件管理参数
   - 支持长文本处理（最大 16384 tokens）
   - 自动统计和截断管理
   - 训练结束后可选导出模型

2. **tinker_finetune.py** - 独立训练脚本
   - 所有配置在代码中
   - 适合快速测试和修改

3. **config.py** - 配置文件
   - 数据文件路径
   - 模型和训练参数
   - 长文本处理设置
   - 采样参数

### 工具脚本
4. **split_dataset.py** - 数据集划分工具
   - 将数据划分为训练集和测试集
   - 支持多个文件合并
   - 可设置测试集比例
   - 显示数据统计信息

5. **evaluate_model.py** - 模型评估脚本
   - 在测试集上评估模型
   - 计算准确率等指标
   - 显示预测示例
   - 保存评估结果

### 文档
6. **README.md** - 主要使用指南
   - 快速开始教程
   - 参数详细说明
   - 常见问题解答
   - 使用示例

7. **QUICK_START_LONG_TEXT.md** - 长文本快速指南
   - 针对 16384 tokens 长输出的配置
   - 内存优化建议
   - 阶段性训练策略
   - 故障排除

8. **MODEL_EXPORT_EVALUATION.md** - 导出和评估详细指南
   - 模型导出方法
   - 与其他框架集成
   - 评估策略和方法
   - 完整示例代码

## 🚀 快速开始

### 1. 基础使用（简单任务）

```bash
# 1. 设置 API 密钥
export TINKER_API_KEY=7624312694161539072

# 2. 编辑 config.py，设置数据路径
# DATA_FILES = ["data/s1k.json"]

# 3. 运行训练
python train_with_config.py
```

### 2. 完整流程（推荐）

```bash
# 1. 设置 API 密钥
export TINKER_API_KEY=7624312694161539072

# 2. 划分数据集
python split_dataset.py
# 生成: data/train_set.json, data/test_set.json

# 3. 编辑 config.py
# DATA_FILES = ["data/train_set.json"]

# 4. 训练模型
python train_with_config.py
# 记录输出的 model_path

# 5. 评估模型
# 编辑 evaluate_model.py，设置 MODEL_PATH
python evaluate_model.py
# 查看 evaluation_results.json
```

### 3. 长文本数据（output > 4096 tokens）

```bash
# 1. 查看长文本配置指南
cat QUICK_START_LONG_TEXT.md

# 2. 编辑 config.py
# MAX_SEQUENCE_LENGTH = 16384
# TRUNCATE_LONG_SEQUENCES = True

# 3. 运行训练（查看截断统计）
python train_with_config.py
```

## 📊 推荐工作流程

### 数学推理任务（如你的数据）

```python
# config.py 推荐配置
MAX_SEQUENCE_LENGTH = 16384  # 容纳长证明
TRUNCATE_LONG_SEQUENCES = True
DATA_FILES = ["data/train_set.json"]
BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct"
LORA_RANK = 32
LEARNING_RATE = 5e-5
NUM_EPOCHS = 5
```

**步骤：**
1. 使用 `split_dataset.py` 划分数据（80% 训练，20% 测试）
2. 使用 `train_with_config.py` 训练模型
3. 观察损失下降和序列长度统计
4. 使用 `evaluate_model.py` 在测试集上评估
5. 如果需要，导出模型权重用于部署

## 🎯 针对不同场景的配置

### 场景 1: 小数据集（< 1000 样本）
```python
NUM_EPOCHS = 10  # 更多轮次
LEARNING_RATE = 1e-4
LORA_RANK = 16  # 较小 rank 避免过拟合
```

### 场景 2: 大数据集（> 10000 样本）
```python
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
LORA_RANK = 64  # 更大容量
```

### 场景 3: 短文本（< 512 tokens）
```python
MAX_SEQUENCE_LENGTH = 2048
NUM_EPOCHS = 5
```

### 场景 4: 超长文本（> 8192 tokens）
```python
MAX_SEQUENCE_LENGTH = 16384
LORA_RANK = 16  # 减少内存使用
TRUNCATE_LONG_SEQUENCES = True
```

## 📈 监控和调试

### 训练时关注的指标

1. **损失值（Loss）**
   - 应该逐渐下降
   - 如果震荡：降低学习率
   - 如果不下降：增加学习率或检查数据

2. **序列长度统计**
   - Max: 最长样本长度
   - Mean/Median: 大部分样本长度
   - Truncated: 被截断的样本数

3. **处理统计**
   - Successfully processed: 成功处理的样本
   - Truncated: 被截断的样本
   - Skipped: 被跳过的样本
   - Failed: 处理失败的样本

### 常见问题排查

**问题：内存不足 (OOM)**
- 降低 MAX_SEQUENCE_LENGTH
- 降低 LORA_RANK
- 使用更小的基础模型

**问题：训练太慢**
- 降低 MAX_SEQUENCE_LENGTH
- 减少数据量先快速测试
- 使用更小的模型

**问题：损失不下降**
- 检查数据格式是否正确
- 增加学习率
- 增加训练轮数
- 检查是否需要更多数据

**问题：很多样本被截断**
- 增加 MAX_SEQUENCE_LENGTH
- 或接受截断（通常前半部分已包含关键信息）

## 🔧 高级功能

### 1. 多阶段训练
```bash
# 阶段1: 快速测试（短序列）
# MAX_SEQUENCE_LENGTH = 4096, NUM_EPOCHS = 2
python train_with_config.py

# 阶段2: 完整训练（长序列）
# MAX_SEQUENCE_LENGTH = 16384, NUM_EPOCHS = 5
python train_with_config.py
```

### 2. 继续训练
```python
# 在训练脚本中使用 load_state()
resume_path = "tinker://previous-model/state/final"
training_client.load_state(resume_path)
```

### 3. 自定义评估
修改 `evaluate_model.py` 中的函数：
- `check_answer_exact_match()` - 完全匹配
- `check_answer_contains()` - 包含检查
- `extract_final_answer()` - 答案提取

### 4. 导出和部署
```python
# 导出权重
rest_client = service_client.create_rest_client()
future = rest_client.download_checkpoint_archive_from_tinker_path(model_path)
archive_data = future.result()

with open("model.tar.gz", "wb") as f:
    f.write(archive_data)

# 使用 HuggingFace 加载
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B-Base")
model = PeftModel.from_pretrained(base_model, "./model_weights")
merged_model = model.merge_and_unload()
```

## 📖 延伸阅读

- [Tinker 官方文档](https://tinker-docs.thinkingmachines.ai/)
- [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook)
- [LoRA 论文](https://arxiv.org/abs/2106.09685)

## 💡 提示和技巧

1. **先小规模测试**：用少量数据和短序列快速验证流程
2. **保存 model_path**：训练完成后立即记录 model_path
3. **定期评估**：每 5-10 epochs 在测试集上评估
4. **监控截断**：如果超过 30% 样本被截断，考虑增加 MAX_SEQUENCE_LENGTH
5. **使用 Instruct 模型**：对于指令遵循任务，Instruct 版本通常效果更好
6. **调整温度**：训练用 temperature=0.7，评估用 temperature=0.0

## 🆘 获取帮助

如有问题：
1. 查看对应的 .md 文档
2. 检查 Tinker 官方文档
3. 查看训练输出的错误信息
4. 联系 Tinker 支持: tinker@thinkingmachines.ai

## ✨ 更新日志

- v1.0: 初始版本，包含训练、导出、评估全流程
- 支持长文本（最大 16384 tokens）
- 提供完整的工具和文档
