import os
import json
import torch
import numpy as np
import warnings
import matplotlib
# 设置非交互式后端
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.preprocessing import StandardScaler
from common_font_config import setup_matplotlib_fonts

# 过滤matplotlib字体相关警告
warnings.filterwarnings("ignore", category=UserWarning, message=".*findfont.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Matplotlib is building the font cache.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*font family.*not found.*")

# 配置字体
setup_matplotlib_fonts()

# 全局配置
DEFAULT_MODEL = "lmsys/vicuna-7b-v1.5"
DEFAULT_LAYERS = list(range(10, 26))  # 10-25层
DEFAULT_INTERVENTION_LAYER = 12
DEFAULT_ALIGNMENT_LAYERS = list(range(13, 25))  # 13-24层
DEFAULT_RANK = 32  # 低秩子空间维度
DEFAULT_TEMPERATURE = 0.1  # 对比损失温度参数
DEFAULT_ALPHA = 1.0  # 对齐损失权重
DEFAULT_BETA = 0.1  # 重建损失权重

# 目录路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# 确保目录存在
def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")

# 模型加载函数
def load_model_and_tokenizer(model_name=DEFAULT_MODEL, device="cuda", load_in_8bit=False):
    """
    加载模型和tokenizer
    参数:
        model_name: 模型名称或路径
        device: 运行设备
        load_in_8bit: 是否使用8位量化
    返回:
        model, tokenizer
    """
    print(f"加载模型: {model_name}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # 加载模型
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.float16,
        "device_map": "auto" if device == "cuda" else None
    }
    
    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()
    
    return model, tokenizer

# 数据加载函数
def load_dataset(dataset_path, max_samples=None):
    """
    加载数据集
    参数:
        dataset_path: 数据集路径
        max_samples: 最大样本数
    返回:
        样本列表
    """
    print(f"加载数据集: {dataset_path}")
    
    file_ext = os.path.splitext(dataset_path)[1].lower()
    
    if file_ext == ".json":
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif file_ext == ".csv":
        import pandas as pd
        data = pd.read_csv(dataset_path).to_dict("records")
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}")
    
    if max_samples and len(data) > max_samples:
        data = data[:max_samples]
        print(f"限制样本数为: {max_samples}")
    
    return data

# 表征处理函数
def load_representations(embeddings_dir, layer, split="train"):
    """
    加载指定层和分割的表征
    参数:
        embeddings_dir: 表征存储目录
        layer: 层号
        split: 数据分割 (train/test)
    返回:
        表征数组, 标签数组
    """
    embedding_path = os.path.join(embeddings_dir, f"{split}_layer_{layer}_embeddings.npy")
    labels_path = os.path.join(embeddings_dir, f"{split}_layer_{layer}_labels.npy")
    
    if not os.path.exists(embedding_path) or not os.path.exists(labels_path):
        raise FileNotFoundError(f"未找到层 {layer} 的 {split} 表征文件")
    
    embeddings = np.load(embedding_path)
    labels = np.load(labels_path)
    
    return embeddings, labels

# 数据标准化函数
def normalize_representations(embeddings, scaler=None):
    """
    标准化表征数据
    参数:
        embeddings: 原始表征数组
        scaler: 已存在的scaler对象，如果为None则创建新的
    返回:
        标准化后的表征数组, scaler对象
    """
    if scaler is None:
        scaler = StandardScaler()
        normalized_embeddings = scaler.fit_transform(embeddings)
    else:
        normalized_embeddings = scaler.transform(embeddings)
    
    return normalized_embeddings, scaler

# 余弦相似度计算函数
def cosine_similarity(a, b, dim=-1, eps=1e-8):
    """
    计算余弦相似度
    参数:
        a: 第一个向量或向量集合
        b: 第二个向量或向量集合
        dim: 计算相似度的维度
        eps: 防止除零错误的小值
    返回:
        相似度值
    """
    a_norm = torch.nn.functional.normalize(a, p=2, dim=dim, eps=eps)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=dim, eps=eps)
    
    if len(a.shape) == 2 and len(b.shape) == 2:
        # 矩阵-矩阵相似度
        return torch.matmul(a_norm, b_norm.t())
    elif len(a.shape) == 2 and len(b.shape) == 3:
        # 矩阵-批量矩阵相似度
        a_norm = a_norm.unsqueeze(1)
        return torch.matmul(a_norm, b_norm.transpose(1, 2)).squeeze(1)
    else:
        # 向量-向量相似度
        return torch.sum(a_norm * b_norm, dim=dim)

# 对比损失计算函数
def compute_contrastive_loss(query, positive_samples, negative_samples, temperature=DEFAULT_TEMPERATURE):
    """
    计算对比损失
    参数:
        query: 查询向量 [batch_size, dim]
        positive_samples: 正样本向量 [num_positive, dim]
        negative_samples: 负样本向量 [num_negative, dim]
        temperature: 温度参数
    返回:
        对比损失值
    """
    # 计算查询与所有样本的相似度
    positive_similarity = cosine_similarity(query, positive_samples).mean(dim=-1)  # [batch_size]
    negative_similarity = cosine_similarity(query, negative_samples)  # [batch_size, num_negative]
    
    # 计算对比损失
    logits = torch.cat([positive_similarity.unsqueeze(1), negative_similarity], dim=1)  # [batch_size, 1+num_negative]
    logits = logits / temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=query.device)
    
    loss = torch.nn.functional.cross_entropy(logits, labels)
    
    return loss

# 准确率计算函数
def compute_accuracy(predictions, labels):
    """
    计算准确率
    参数:
        predictions: 预测标签
        labels: 真实标签
    返回:
        准确率值
    """
    correct = (predictions == labels).sum().item()
    total = len(labels)
    return correct / total if total > 0 else 0.0

# 攻击成功率(ASR)计算函数
def compute_asr(results):
    """
    计算攻击成功率
    参数:
        results: 结果字典，包含'成功'和'失败'的计数
    返回:
        ASR值
    """
    success = results.get("成功", 0)
    total = sum(results.values())
    return success / total if total > 0 else 0.0

# 保存结果函数
def save_results(results, save_path):
    """
    保存结果到文件
    参数:
        results: 结果数据
        save_path: 保存路径
    """
    ensure_dir(os.path.dirname(save_path))
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到: {save_path}")

# 可视化函数
def plot_accuracy_by_layer(accuracy_dict, save_path=None):
    """
    绘制每层的准确率曲线
    参数:
        accuracy_dict: 字典，键为层号，值为准确率
        save_path: 保存路径，如果为None则显示
    """
    plt.figure(figsize=(12, 6))
    layers = sorted(accuracy_dict.keys())
    accuracies = [accuracy_dict[layer] for layer in layers]
    
    plt.plot(layers, accuracies, marker="o", linestyle="-", color="blue")
    plt.axhline(y=0.9, color="r", linestyle="--", alpha=0.5, label="90%准确率线")
    plt.xlabel("层号")
    plt.ylabel("准确率")
    plt.title("各层表征分类准确率")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"准确率曲线已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()

# 混淆矩阵绘制函数
def plot_confusion_matrix(cm, classes, save_path=None):
    """
    绘制混淆矩阵
    参数:
        cm: 混淆矩阵
        classes: 类别名称列表
        save_path: 保存路径，如果为None则显示
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.title("混淆矩阵")
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"混淆矩阵已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()

# 损失分布绘制函数
def plot_loss_distribution(loss_values, title="损失分布", save_path=None):
    """
    绘制损失分布直方图
    参数:
        loss_values: 损失值数组
        title: 图表标题
        save_path: 保存路径，如果为None则显示
    """
    plt.figure(figsize=(10, 6))
    plt.hist(loss_values, bins=50, alpha=0.7, color="skyblue")
    plt.xlabel("损失值")
    plt.ylabel("频数")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"{title}已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()

# 解析层参数函数
def parse_layer_argument(layer_arg):
    """
    解析层参数，支持范围格式(如"10-25")和列表格式(如"10,15,20")
    参数:
        layer_arg: 层参数字符串
    返回:
        层列表
    """
    if isinstance(layer_arg, list):
        return layer_arg
    
    if isinstance(layer_arg, int):
        return [layer_arg]
    
    if "-" in layer_arg:
        # 范围格式
        start, end = map(int, layer_arg.split("-"))
        return list(range(start, end + 1))
    elif "," in layer_arg:
        # 列表格式
        return list(map(int, layer_arg.split(",")))
    else:
        # 单个层
        return [int(layer_arg)]

# 初始化正交矩阵函数
def init_orthogonal_matrix(rows, cols):
    """
    初始化行正交矩阵
    参数:
        rows: 行数
        cols: 列数
    返回:
        正交矩阵
    """
    matrix = torch.empty(rows, cols)
    torch.nn.init.orthogonal_(matrix)
    return matrix

# 动态阈值调整函数
def adjust_threshold(scores, false_positive_rate=0.01):
    """
    根据误报率调整动态阈值
    参数:
        scores: 分数数组
        false_positive_rate: 允许的误报率
    返回:
        调整后的阈值
    """
    sorted_scores = sorted(scores)
    threshold_index = int(len(scores) * (1 - false_positive_rate))
    threshold = sorted_scores[threshold_index] if threshold_index < len(scores) else sorted_scores[-1]
    return threshold

# 主函数（用于测试）
if __name__ == "__main__":
    # 测试工具函数
    print("测试工具函数")
    print(f"基础目录: {BASE_DIR}")
    print(f"数据目录: {DATA_DIR}")
    print(f"默认层范围: {DEFAULT_LAYERS}")
    
    # 测试层参数解析
    print(f"解析层参数 '10-15': {parse_layer_argument('10-15')}")
    print(f"解析层参数 '10,15,20': {parse_layer_argument('10,15,20')}")
    
    # 测试正交矩阵初始化
    orthogonal_matrix = init_orthogonal_matrix(3, 5)
    print(f"正交矩阵形状: {orthogonal_matrix.shape}")
    print(f"正交性验证 (第一行与第二行点积): {torch.dot(orthogonal_matrix[0], orthogonal_matrix[1]).item():.6f}")
    
    print("工具函数测试完成")