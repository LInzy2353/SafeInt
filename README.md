# SafeInt: 论文复现指南

## 项目概述

本指南提供《SafeInt: Shielding Large Language Models from Jailbreak Attacks via Safety-Aware Representation Intervention》(arXiv:2502.15594v2)论文的完整复现步骤。本项目已完成环境搭建、数据集构建、LLM表征提取、逻辑回归分类器实现和端到端训练，您只需按照以下步骤执行命令即可完成复现。

## 环境配置与依赖

### 系统要求
- 操作系统：Linux (Ubuntu 20.04+)
- GPU：≥16GB显存，支持CUDA 11.7+
- Python：3.10
- 虚拟环境：llm-safety

## 快速开始：完整复现流程

### 第一步：激活虚拟环境并安装依赖

```bash
conda activate llm-safety
cd /home/blcu_lzy2025/SafeInt
pip install -r requirements.txt
```

### 第二步：检查数据集

确保已存在以下数据集文件：

```bash
# 查看数据集文件
ls -la /home/blcu_lzy2025/SafeInt/data/
# 确认存在：dunsafe_dataset.csv, jailbreak_dataset.json, safe_dataset.json
```

若数据集缺失，可生成：

```bash
# 生成不安全数据集
src/data_processing/generate_unsafe.py
# 生成安全数据集
src/data_processing/generate_safe_dataset.py
# 生成越狱数据集
src/data_processing/generate_real_jailbreak_dataset.py
```

### 第三步：数据预处理

```bash
python src/data_processing/prepare_datasets.py
# 预处理后会在data/train/和data/test/目录生成训练集和测试集
```

### 第四步：提取LLM表征

```bash
python src/extract_llm_representations.py
# 提取结果将保存在embeddings/目录下，按层号存储为numpy数组
```

### 第五步：运行逻辑回归分类器

```bash
python src/logistic_regression_classifier.py --layers 10-25
# 训练结果将保存在models/logistic_regression/目录
# 可视化结果将保存在figures/目录
```

### 第六步：使用主脚本运行完整流程

我们提供了一个新的主脚本`main.py`，整合了所有模块的功能：

```bash
cd /home/blcu_lzy2025/SafeInt
python src/main.py --model vicuna-7b --run_mode full --layers 10-25
```

主要参数说明：
- `--model`：使用的模型名称（默认：vicuna-7b）
- `--run_mode`：运行模式（full/extract/classifier/train/inference/evaluate）
- `--layers`：处理的层范围（如"10-25"或"10,15,20"）
- `--batch_size`：批处理大小（默认：32）
- `--learning_rate`：学习率（默认：1e-4）
- `--num_epochs`：训练轮次（默认：15）

### 第七步：执行端到端训练（传统方式）

```bash
cd /home/blcu_lzy2025/SafeInt
python src/train_model.py
# 训练日志将输出到控制台
# 模型将保存在models/safety_model/目录
```

## SafeInt核心模块说明

我们实现了论文中描述的5个核心模块：

### 1. 表征重定位（Representation Relocation）
- 实现论文3.1节，公式2
- 将中间层（第12层）表征投影到安全相关低秩子空间
- 通过参数化映射实现"越狱表征→不安全表征"的定向迁移
- 使用命令：`python src/safeint_representation_relocation.py`

### 2. 表征对齐（Representation Alignment）
- 实现论文3.2节，公式3-5
- 通过"分类损失+对比损失"优化分布一致性
- 强制干预后的越狱表征与不安全样本表征分布对齐
- 使用命令：`python src/safeint_representation_alignment.py`

### 3. 表征重建（Representation Reconstruction）
- 实现论文3.3节，公式6-7
- 通过MSE损失约束安全/不安全样本表征不变
- 避免过度干预破坏模型效用
- 使用命令：`python src/safeint_representation_reconstruction.py`

### 4. 训练优化与推理集成
- 优化仅针对干预模块参数（权重W_θ和偏置b_θ）
- 使用AdamW优化器，学习率1e-4，权重衰减1e-5
- 集成到模型推理流程，实现"实时风险检测→动态干预"
- 使用命令：`python src/safeint_training.py`

### 5. 效果评估
- 防御效果：使用ASR（Attack Success Rate）指标评估
- 效用保持：对比MT-Bench和Just-Eval评分
- 鲁棒性：评估自适应攻击下的防御效果
- 使用命令：`python src/safeint_evaluation.py`

## 详细命令行操作指南

### 环境检查与依赖安装

```bash
# 检查CUDA版本
nvidia-smi
# 检查Python版本
python --version
# 激活虚拟环境
conda activate llm-safety
# 安装依赖
pip install torch==2.1.0+cu118 transformers==4.35.2 datasets==2.14.6 scikit-learn==1.3.2
pip install accelerate sentencepiece matplotlib seaborn numpy pandas
# 安装项目特定依赖
pip install -r requirements.txt
```

### 数据处理与准备

```bash
# 查看数据处理脚本
ls -la src/data_processing/
# 运行数据预处理
python src/data_processing/prepare_datasets.py
# 查看生成的训练集和测试集
ls -la data/train/
ls -la data/test/
```

### LLM表征提取

```bash
# 查看提取脚本
cat src/extract_llm_representations.py | grep -A 10 "def main"
# 运行提取脚本
python src/extract_llm_representations.py
# 查看提取结果
ls -la embeddings/
```

### 逻辑回归分类器运行

```bash
# 查看分类器参数
python src/logistic_regression_classifier.py --help
# 运行特定层的分类器
python src/logistic_regression_classifier.py --layers 10,15,20,25
# 查看准确率结果
cat models/logistic_regression/accuracy_results.txt
# 查看可视化结果
ls -la figures/
```

### 端到端训练与评估

```bash
# 检查是否已有模型
ls -la models/safety_model/
# 运行端到端训练
python src/train_model.py
# 或使用nohup后台运行
nohup python src/train_model.py > training_e2e.log 2>&1 &
# 查看训练进度
tail -f training_e2e.log
```

## 常见问题解决

### GPU内存不足

```bash
# 减小batch_size重新运行
python src/train_model.py --batch_size 8
```

### 设备不匹配错误

```bash
# 强制使用CPU运行
CUDA_VISIBLE_DEVICES="" python src/train_model.py
```

### 数据集缺失

```bash
# 重新生成数据集
src/data_processing/generate_unsafe.py
```

## 预期输出与结果

1. **数据集预处理**：生成train_data.json和single_method_test.json文件
2. **LLM表征提取**：生成layer_10到layer_25的训练与测试表征文件
3. **逻辑回归分类器**：生成混淆矩阵和准确率曲线可视化
4. **端到端训练**：在models/safety_model/目录生成完整模型

## 验证论文假设

### 假设1：样本表征可区分性

```bash
# 查看单方法测试集准确率
grep "single method test accuracy" training_e2e.log
# 预期结果：准确率≥95%
```

### 假设2：越狱方法分布一致性

```bash
# 查看多方法测试集准确率
grep "multi method test accuracy" training_e2e.log
# 预期结果：准确率≥90%
```

## 项目结构

```
SafeInt/
├── README.md                    # 项目说明文档
├── requirements.txt             # 项目依赖
├── data/                        # 数据集目录
│   ├── dunsafe_dataset.csv      # 不安全数据集
│   ├── jailbreak_dataset.json   # 越狱数据集
│   ├── safe_dataset.json        # 安全数据集
│   ├── test/                    # 测试集
│   │   └── single_method_test.json  # 单方法测试集
│   └── train/                   # 训练集
│       └── train_data.json      # 训练数据
├── embeddings/                  # LLM表征目录
├── figures/                     # 可视化结果
├── models/                      # 模型保存目录
│   ├── logistic_regression/     # 逻辑回归模型
│   └── safety_model/            # 完整安全模型
└── src/                         # 源代码目录
    ├── data_processing/         # 数据处理模块
    ├── extract_llm_representations.py  # LLM表征提取
    ├── logistic_regression_classifier.py  # 逻辑回归分类器
    ├── safeint_representation_relocation.py  # 表征重定位模块
    ├── safeint_representation_alignment.py  # 表征对齐模块
    ├── safeint_representation_reconstruction.py  # 表征重建模块
    ├── safeint_training.py  # 训练优化与推理集成模块
    ├── safeint_evaluation.py  # 效果评估模块
    ├── main.py  # 完整工作流程主脚本
    └── train_model.py           # 端到端训练脚本
```

## 参考资料

- 论文链接：https://arxiv.org/html/2502.15594
- 原始仓库：https://github.com/SproutNan/AI-Safety_SCAV.git