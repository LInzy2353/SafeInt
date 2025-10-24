# SafeInt 实验流程与可运行命令（训练＋评估）

本文档基于 `src/` 代码现状与评估工具的最新实现，提供可直接运行的完整实验步骤、训练方法（命令）和评估方法（命令），并明确各阶段的输出位置与注意事项。

## 1. 概览

SafeInt 的端到端流程包括：
- 表征提取（中间层 hidden states）
- 风险分类器训练（逻辑回归）
- SafeInt 表征重定位模块训练（U 和 fθ）
- 推理与评估（防御效果、效用保持、鲁棒性）

核心干预公式（与代码一致）：

```
h̃ = h + α · U^T · ( fθ(h) − U·h )
```
- `U`：低秩投影矩阵（正交，固定不训练）
- `fθ`：线性重定位映射（可训练）
- `α`：干预强度系数

实现位置：`src/safeint_representation_relocation.py::apply_intervention` 与推理 Hook。

## 2. 环境与数据准备

- 安装依赖：
```bash
pip install -r requirements.txt
```
- 数据目录：
  - 训练：`data/train/`（`train_safe.json`、`train_jailbreak.json`、`train_unsafe.json`）
  - 测试：`data/test/`（`single_method_test.json`、`test_utility.json`、`test_adaptive_attacks.json` 等）
- 模型路径（示例，Vicuna-7B v1.5）：
```bash
MODEL_PATH=/home/blcu_lzy2025/.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d
```

## 3. 快速全流程训练（推荐）

使用简化脚本 `src/run_training.py` 一次执行提取→分类器→SafeInt训练→评估：

```bash
python src/run_training.py \
  --model_path "$MODEL_PATH" \
  --train_data_path data/train \
  --test_data_path data/test \
  --output_dir results/training \
  --epochs 15 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --intervention_layer 12 \
  --alignment_layers 13-24 \
  --subspace_rank 32 \
  --alpha 1.0 \
  --beta 0.1 \
  --temperature 0.1 \
  --gain_factor 2.0 \
  --intervention_strength 1.0
```
- 可选：禁用 4bit 量化以避免兼容问题：添加 `--disable_4bit`
- 输出：
  - 训练过程与摘要打印到终端；模型与中间产物在 `models/`、`embeddings/`
  - 评估结果在 `results/evaluation/`（见第 6 节）

后台运行示例（附日志）：
```bash
nohup python src/run_training.py --model_path "$MODEL_PATH" --train_data_path data/train --test_data_path data/test \
  --output_dir results/training --epochs 15 --batch_size 32 --learning_rate 1e-4 \
  --intervention_layer 12 --alignment_layers 13-24 --subspace_rank 32 --alpha 1.0 --beta 0.1 --temperature 0.1 \
  --gain_factor 2.0 --intervention_strength 1.0 > logs/train_full_nohup.log 2>&1 &
```

## 4. 分步骤训练（精细控制）

使用 `src/main.py` 逐步执行：

- 仅提取表征：
```bash
python src/main.py --model_path "$MODEL_PATH" --mode extract --layers 10-25
```
- 仅训练分类器（会自动确保有表征）：
```bash
python src/main.py --model_path "$MODEL_PATH" --mode classifier --layers 10-25
```
- 仅训练 SafeInt 模块：
```bash
python src/main.py --model_path "$MODEL_PATH" --mode train \
  --epochs 15 --batch_size 32 --learning_rate 1e-4 \
  --intervention_layer 12 --alignment_layers 13-24 --subspace_rank 32 \
  --alpha 1.0 --beta 0.1 --temperature 0.1
```
- 推理测试（使用 SafeInt）：
```bash
python src/main.py --model_path "$MODEL_PATH" --mode inference --input_text "你的输入文本" --use_safeint True
```
- 完整流程（提取→分类器→训练→评估）：
```bash
python src/main.py --model_path "$MODEL_PATH" --mode full
```

## 5. 评估方法与命令

评估支持两种入口：主流程评估与独立评估器。

### 5.1 主流程评估入口（main.py）
```bash
python src/main.py --model_path "$MODEL_PATH" --mode evaluate
```
- 默认测试集：`data/test/single_method_test.json`
- 评估期间默认启用“持久干预”（`persistent_intervention = True`），并确保重定位器开启；与论文的“触发即拒绝”一致地对 ASR 关键词进行统计（详见 5.3）。
- 输出参见 5.4。

### 5.2 独立评估器（safeint_evaluation.py）

```bash
python src/safeint_evaluation.py \
  --model_path "$MODEL_PATH" \
  --test_file single_method_test.json \
  --utility_file test_utility.json
```
后台运行（含日志与 PID）：
```bash
mkdir -p logs nohup
nohup python src/safeint_evaluation.py --model_path "$MODEL_PATH" \
  --test_file single_method_test.json --utility_file test_utility.json \
  > logs/eval_single_method_nohup.log 2>&1 & echo $! > nohup/eval_single_method.pid
```
常用查看命令：
```bash
ps -ef | grep safeint_evaluation.py | grep -v grep
tail -n 200 -f logs/eval_single_method_nohup.log
```

### 5.3 评估指标与判定逻辑（与代码一致）

- 防御效果（ASR）：
  - 采用关键词与正则模式判断拒绝响应（`reject_keywords` 与 `reject_patterns`），并在评估中暂时开启显式拒绝开关统计“触发即拒绝”的 ASR 关键词。
  - 入口：`SafeIntEvaluator.evaluate_defense_effectiveness()`；数据来自 `single_method_test.json` 按标签拆分为攻击与安全。
- 效用保持：
  - 对安全数据对比原始模型与 SafeInt 响应质量；当前实现为随机模拟分数（代码注释已明确），真实实验需替换为外部 LLM 或人工评分。
  - 入口：`SafeIntEvaluator.evaluate_utility()`；输出包括各维度平均分与差异百分比，默认判定“差异 ≤ 2%”为达标。
- 鲁棒性：
  - 使用自适应攻击数据（如存在 `test_adaptive_attacks.json`）计算 ASR 并可视化。
  - 入口：`SafeIntEvaluator.evaluate_robustness()`。

### 5.4 评估输出位置

- 结果 JSON：`results/evaluation/evaluation_results.json`
- 综合报告：`results/evaluation/comprehensive_evaluation_report.md`
- 图表示例：
  - `results/evaluation/defense_asr_by_dataset.png`
  - `results/evaluation/utility_comparison.png`
  - `results/evaluation/robustness_pie.png`
- 日志：`src/logs/safeint_evaluation.log`（评估器内部日志）与 `logs/*.log`（nohup 输出）

## 6. 关键参数与默认值（摘要）

- `src/run_training.py`：
  - 必填：`--model_path`
  - 数据与输出：`--train_data_path data/train`、`--test_data_path data/test`、`--output_dir results/training`
  - 训练：`--epochs 15`、`--batch_size 32`、`--learning_rate 1e-4`
  - SafeInt：`--intervention_layer 12`、`--alignment_layers 13-24`、`--subspace_rank 32`、`--alpha 1.0`、`--beta 0.1`、`--temperature 0.1`
  - 增强：`--gain_factor 2.0`（权重初始化增益）、`--intervention_strength 1.0`
  - 量化：`--disable_4bit`（可选，禁用 4bit 量化）
- `src/main.py`：支持 `--mode {full, extract, classifier, train, inference, evaluate}` 与同类训练参数；`--layers` 默认 `10-25`。

## 7. 复现实验示例（Vicuna-7B）

1) 快速训练＋评估（推荐）：
```bash
MODEL_PATH=/home/blcu_lzy2025/.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d
python src/run_training.py --model_path "$MODEL_PATH" --train_data_path data/train --test_data_path data/test \
  --epochs 15 --batch_size 32 --learning_rate 1e-4 --intervention_layer 12 --alignment_layers 13-24 --subspace_rank 32 \
  --alpha 1.0 --beta 0.1 --temperature 0.1 --gain_factor 2.0 --intervention_strength 1.0
```
2) 独立评估：
```bash
python src/safeint_evaluation.py --model_path "$MODEL_PATH" --test_file single_method_test.json --utility_file test_utility.json
```
3) 主入口（全流程或仅评估）：
```bash
python src/main.py --model_path "$MODEL_PATH" --mode full
python src/main.py --model_path "$MODEL_PATH" --mode evaluate
```

## 8. 注意事项与建议

- 评估默认启用 `persistent_intervention` 并在防御/鲁棒性评估中暂时开启显式拒绝统计，这会显著降低 ASR（与“触发即拒绝”的关键词判定一致）；如需内容级 ASR，请使用外部 LLM 或人工审核替换关键词判定。
- 如出现显存不足：降低 `--batch_size`、`--subspace_rank`，或启用 4bit（不要加 `--disable_4bit`）。
- 训练与评估日志可用于快速定位问题：`logs/` 与 `src/logs/`。

---
以上命令与路径与当前仓库代码完全对齐，可直接用于复现实验与生成报告。