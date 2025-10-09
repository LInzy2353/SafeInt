import json
import csv
import random
import os
from sklearn.model_selection import train_test_split

# 设置随机种子以确保可复现性
random.seed(42)

# 数据路径
dunsafe_path = '/home/blcu_lzy2025/SafeInt/data/dunsafe_dataset.csv'
jailbreak_path = '/home/blcu_lzy2025/SafeInt/data/jailbreak_dataset.json'
safe_path = '/home/blcu_lzy2025/SafeInt/data/safe_dataset.json'

# 输出路径
train_dir = '/home/blcu_lzy2025/SafeInt/data/train'
test_dir = '/home/blcu_lzy2025/SafeInt/data/test'

# 创建输出目录
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 论文要求的样本数量
train_samples_per_class = 128  # 每类训练样本数量
single_method_test_samples_per_class = 150  # 单方法测试集每类样本数量

# 加载不安全数据集 (D_unsafe)
def load_unsafe_dataset():
    unsafe_data = []
    with open(dunsafe_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 使用instruction字段作为text，标签为1（不安全）
            unsafe_data.append({
                'text': row['instruction'],
                'label': 1
            })
    return unsafe_data

# 加载越狱数据集 (D_jailbreak)
def load_jailbreak_dataset():
    jailbreak_data = []
    with open(jailbreak_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # 检查数据结构，查找jailbreak_examples字段
        if 'jailbreak_examples' in data:
            samples = data['jailbreak_examples']
        else:
            samples = []
        
        for sample in samples:
            # 确保样本是字典类型且有goal字段
            if isinstance(sample, dict) and 'goal' in sample:
                # 使用goal字段作为text，标签为2（越狱）
                jailbreak_data.append({
                    'text': sample['goal'],
                    'label': 2
                })
    return jailbreak_data

# 加载安全数据集 (D_safe)
def load_safe_dataset():
    safe_data = []
    with open(safe_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for sample in data:
            # 使用instruction字段作为text，标签为0（安全）
            safe_data.append({
                'text': sample['instruction'],
                'label': 0
            })
    return safe_data

# 文本清洗函数
def clean_text(text):
    # 去除特殊符号，标准化指令格式
    # 这里可以根据需要添加更复杂的清洗逻辑
    text = text.strip()
    # 统一前缀（如果需要）
    # text = f"请完成以下任务：{text}"
    return text

# 清洗数据集
def clean_dataset(data):
    cleaned_data = []
    for item in data:
        cleaned_item = {
            'text': clean_text(item['text']),
            'label': item['label']
        }
        # 确保文本不为空
        if cleaned_item['text']:
            cleaned_data.append(cleaned_item)
    return cleaned_data

# 主函数
def main():
    # 加载数据集
    print("加载数据集...")
    unsafe_data = load_unsafe_dataset()
    jailbreak_data = load_jailbreak_dataset()
    safe_data = load_safe_dataset()
    
    print(f"不安全样本数量: {len(unsafe_data)}")
    print(f"越狱样本数量: {len(jailbreak_data)}")
    print(f"安全样本数量: {len(safe_data)}")
    
    # 清洗数据集
    print("清洗数据集...")
    unsafe_data = clean_dataset(unsafe_data)
    jailbreak_data = clean_dataset(jailbreak_data)
    safe_data = clean_dataset(safe_data)
    
    # 确保样本数量足够
    def ensure_sample_size(data, required_size):
        if len(data) >= required_size:
            return random.sample(data, required_size)
        else:
            # 如果样本不足，重复采样（仅用于演示，实际应考虑添加更多样本）
            print(f"警告：样本数量不足 {required_size}，使用重复采样")
            return random.choices(data, k=required_size)
    
    # 划分训练集和测试集
    print("划分训练集和测试集...")
    
    # 训练集
    train_unsafe = ensure_sample_size(unsafe_data, train_samples_per_class)
    train_jailbreak = ensure_sample_size(jailbreak_data, train_samples_per_class)
    train_safe = ensure_sample_size(safe_data, train_samples_per_class)
    
    # 合并训练集并打乱顺序
    train_data = train_unsafe + train_jailbreak + train_safe
    random.shuffle(train_data)
    
    # 单方法测试集（仅使用GCG越狱方法的样本）
    # 由于当前数据集没有明确标记越狱方法，我们假设所有越狱样本都是GCG方法
    test_unsafe = ensure_sample_size(unsafe_data, single_method_test_samples_per_class)
    test_jailbreak = ensure_sample_size(jailbreak_data, single_method_test_samples_per_class)
    test_safe = ensure_sample_size(safe_data, single_method_test_samples_per_class)
    
    # 合并测试集并打乱顺序
    test_data = test_unsafe + test_jailbreak + test_safe
    random.shuffle(test_data)
    
    # 保存数据集
    print("保存数据集...")
    with open(os.path.join(train_dir, 'train_data.json'), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(test_dir, 'single_method_test.json'), 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print(f"训练集已保存至 {os.path.join(train_dir, 'train_data.json')}，共 {len(train_data)} 条样本")
    print(f"单方法测试集已保存至 {os.path.join(test_dir, 'single_method_test.json')}，共 {len(test_data)} 条样本")
    
    # 统计各类别样本数量
    def count_labels(data):
        counts = {0: 0, 1: 0, 2: 0}
        for item in data:
            counts[item['label']] += 1
        return counts
    
    print(f"训练集类别分布: {count_labels(train_data)}")
    print(f"测试集类别分布: {count_labels(test_data)}")

if __name__ == "__main__":
    main()