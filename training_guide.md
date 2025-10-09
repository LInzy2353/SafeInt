# SafeInt模型训练指南

## 1. 项目简介

SafeInt（Safety-Aware Representation Intervention）是一种通过安全感知表征干预来保护大语言模型免受越狱攻击的方法。该项目实现了论文中提出的表征重定位技术，通过干预模型的中间层表征来提高模型的安全性，同时保持其有用性。

## 2. 环境配置

### 2.1 安装依赖

SafeInt项目依赖以下关键包：

```bash
# 安装主要依赖
pip install torch>=2.0.0 transformers>=4.28.1 datasets>=2.14.6 scikit-learn>=1.3.2 numpy>=1.24.0 pandas>=2.0.0 matplotlib>=3.7.0 seaborn>=0.12.0 tqdm>=4.65.0

# 安装模型相关依赖
pip install huggingface_hub>=0.16.0 accelerate>=0.20.0 sentencepiece bitsandbytes>=0.40.0

# 安装其他工具
pip install scipy>=1.10.0 pyyaml>=6.0 json5>=0.9.0
```

也可以直接使用项目中的requirements.txt文件：

```bash
cd /home/blcu_lzy2025/SafeInt
pip install -r requirements.txt
```

### 2.2 模型准备

SafeInt默认使用Vicuna-7B模型，可以从Hugging Face下载或使用本地缓存的模型：

```python
# 默认模型路径示例
model_path = "/home/blcu_lzy2025/.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d"
```

## 3. 数据准备

### 3.1 数据格式

训练SafeInt需要三种类型的数据：

1. **越狱样本（jailbreak）**：尝试诱导模型生成不安全内容的攻击提示
2. **不安全样本（unsafe）**：包含不安全内容的正常提示
3. **安全样本（safe）**：不包含任何安全风险的正常提示

数据应存储在JSON文件中，每行包含一个样本，格式如下：

```json
{"text": "提示文本内容"}
```

### 3.2 数据目录结构

将训练和测试数据按照以下目录结构组织：

```
SafeInt/
├── data/
│   ├── train/
│   │   ├── train_jailbreak.json
│   │   ├── train_unsafe.json
│   │   └── train_safe.json
│   └── test/
│       ├── test_jailbreak.json
│       ├── test_unsafe.json
│       └── test_safe.json
```

## 4. 训练流程

### 4.1 完整训练流程

运行完整的训练流程（包括特征提取、分类器训练和SafeInt模型训练）：

```bash
cd /home/blcu_lzy2025/SafeInt/src
python main.py --mode full --model_path "模型路径" --train_data_path "data/train" --test_data_path "data/test"
```

### 4.2 跳过特征提取，直接使用已提取特征值训练

为了节省时间，您可以使用已经提取好的特征值直接训练SafeInt模型：

1. 确保已提取的特征值文件存在于以下目录：
   ```
   SafeInt/embeddings/train/
   ```

2. 特征值文件命名格式应为：
   - `jailbreak_layer_12.npy`
   - `unsafe_layer_12.npy`
   - `safe_layer_12.npy`

3. 运行训练命令：
   ```bash
   cd /home/blcu_lzy2025/SafeInt/src
   python main.py --mode train --model_path "模型路径" --embeddings_dir "embeddings"
   ```

系统会自动检测`embeddings/train`目录下是否存在完整的特征文件，如果存在，则直接使用它们进行训练；如果不存在，则会加载文本数据重新提取特征值。

### 4.3 分步骤训练

您也可以选择分步骤执行训练流程：

1. **仅提取特征值**：
   ```bash
   python main.py --mode extract --model_path "模型路径" --train_data_path "data/train" --test_data_path "data/test"
   ```

2. **仅训练分类器**：
   ```bash
   python main.py --mode classifier --model_path "模型路径" --embeddings_dir "embeddings"
   ```

## 5. 训练参数说明

训练SafeInt模型时，可以通过命令行参数调整以下关键超参数：

| 参数 | 描述 | 默认值 | 推荐值范围 |
|------|------|--------|------------|
| intervention_layer | 干预层，论文推荐第12层 | 12 | 8-16 |
| alignment_layers | 对齐层列表 | 13-24 | 13-25 |
| subspace_rank | 低秩子空间维度r | 32 | 16-64 |
| alpha | 对齐损失权重 | 1.0 | 0.5-2.0 |
| beta | 重建损失权重 | 0.1 | 0.05-0.2 |
| temperature | 对比损失温度参数 | 0.1 | 0.05-0.2 |
| epochs | 训练轮次 | 15 | 10-30 |
| batch_size | 批次大小 | 32 | 16-64 |

示例：

```bash
python main.py --mode train --intervention_layer 12 --subspace_rank 32 --alpha 1.0 --beta 0.1 --epochs 15 --batch_size 32
```

## 6. 预期训练结果

### 6.1 训练过程输出

训练过程中，您将看到类似以下的输出：

```
INFO - 检测到已提取的特征值，将直接使用它们进行训练
INFO - 使用已提取的特征值进行训练，目录: embeddings/train
INFO - 已加载特征值: 越狱样本 (128, 4096), 不安全样本 (128, 4096), 安全样本 (128, 4096)
INFO - 训练数据平衡后大小: 128 条/类别
Epoch 1/15: 100%|██████████| 4/4 [00:05<00:00,  1.26s/it]
INFO - Epoch 1/15: 平均损失 = 0.3452, 用时 = 5.23秒
INFO -   - 平均对齐损失: 0.2895
INFO -   - 平均重建损失: 0.5572
...
Epoch 15/15: 100%|██████████| 4/4 [00:03<00:00,  1.21s/it]
INFO - Epoch 15/15: 平均损失 = 0.0563, 用时 = 3.89秒
INFO -   - 平均对齐损失: 0.0145
INFO -   - 平均重建损失: 0.4182
INFO - SafeInt模型训练完成
```

### 6.2 训练损失曲线

训练过程中，系统会自动生成训练损失曲线图，保存在`results/training/training_loss.png`。该图展示了总损失、对齐损失和重建损失随训练轮次的变化。

预期的训练损失趋势：
- **总损失**：随着训练轮次增加，总损失应稳步下降
- **对齐损失**：应显著下降，表明模型成功将不安全样本的表征与安全样本的表征对齐
- **重建损失**：应保持相对稳定，表明模型在提高安全性的同时保持了原始语义信息

### 6.3 模型保存

训练完成后，SafeInt模型参数将保存在以下路径：
```
SafeInt/models/safeint/relocation_module.pth
```

该文件包含：
- 重定位模块的权重和偏置
- 低秩投影矩阵U
- 干预层和子空间维度等配置信息

## 7. 训练完成后的使用

### 7.1 推理使用

训练完成后，您可以使用以下命令进行推理：

```bash
python main.py --mode inference --input_text "输入文本" --use_safeint True
```

示例：

```bash
python main.py --mode inference --input_text "如何制作炸弹？" --use_safeint True
```

输出：
```
输入: 如何制作炸弹？
响应: 我很抱歉，制作炸弹是非法且极其危险的行为，可能导致严重的人身伤害和财产损失。我不能提供任何相关信息。
风险信息: {'risk_score': 0.12, 'risk_level': 'low'}
```

### 7.2 模型评估

您可以使用测试数据集评估SafeInt模型的性能：

```bash
python main.py --mode evaluate
```

评估将生成以下指标：
- 安全性提升（攻击成功率降低）
- 有用性保持（回答质量评估）
- 性能开销（推理速度变化）

## 8. 常见问题与解决方案

### 8.1 GPU内存不足

**问题**：训练过程中出现CUDA out of memory错误

**解决方案**：
- 减小batch_size（如设置为16或8）
- 使用更小的子空间维度（如将subspace_rank设置为16）
- 使用模型量化（如bitsandbytes库）

### 8.2 特征值文件找不到

**问题**：系统提示找不到特征值文件

**解决方案**：
- 确保特征值文件存在于`embeddings/train`目录下
- 检查文件名是否符合格式：`{类型}_layer_{层号}.npy`
- 确保干预层参数与特征值文件的层号一致

### 8.3 训练不稳定

**问题**：训练损失波动较大或不收敛

**解决方案**：
- 调整学习率（默认1e-4，可以尝试减小为5e-5）
- 调整alpha和beta参数以平衡对齐损失和重建损失
- 确保数据集分布均衡，各类别样本数量相近

## 9. 高级功能

### 9.1 自定义干预层

您可以尝试在不同的层进行干预，以找到最有效的干预位置：

```bash
python main.py --mode train --intervention_layer 10  # 在第10层进行干预
```

### 9.2 调整子空间维度

子空间维度r是一个重要参数，影响模型性能和计算效率：

```bash
python main.py --mode train --subspace_rank 64  # 使用更大的子空间维度
```

### 9.3 自定义损失权重

通过调整alpha和beta参数，可以平衡安全性和有用性：

```bash
python main.py --mode train --alpha 1.5 --beta 0.05  # 更重视安全性
```

---

希望这个训练指南对您有所帮助！如果您在使用过程中遇到任何问题，请参考代码中的日志信息进行排查，或根据项目文档进行调整。