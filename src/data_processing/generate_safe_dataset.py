import os
import sys
import json
import random
import argparse
import pandas as pd
from tqdm import tqdm

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    
# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='Generate safe dataset from Alpaca dataset')
    parser.add_argument('--output_path', type=str, default='/home/blcu_lzy2025/SafeInt/data/safe_dataset.json',
                      help='Path to save the generated safe dataset')
    parser.add_argument('--num_samples', type=int, default=150,
                      help='Number of safe samples to generate')
    parser.add_argument('--alpaca_data_path', type=str, 
                      default='/home/blcu_lzy2025/SafeInt/alpaca_dataset/data/train-00000-of-00001-a09b74b3ef9c3b56.parquet',
                      help='Path to the Alpaca dataset (parquet file)')
    return parser.parse_args()

# 加载Alpaca数据集
def load_alpaca_dataset(alpaca_data_path):
    print("Loading Alpaca dataset...")
    try:
        # 尝试使用pyarrow引擎读取parquet文件
        df = pd.read_parquet(alpaca_data_path, engine='pyarrow')
        print(f"Loaded Alpaca dataset with {len(df)} samples.")
        return df
    except Exception as e:
        print(f"Error loading Alpaca dataset: {str(e)}")
        print("Attempting to install pyarrow...")
        # 尝试安装pyarrow
        try:
            import subprocess
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'pyarrow'], check=True)
            # 安装成功后再次尝试读取
            df = pd.read_parquet(alpaca_data_path, engine='pyarrow')
            print(f"Loaded Alpaca dataset with {len(df)} samples after installing pyarrow.")
            return df
        except Exception as e2:
            print(f"Failed to install pyarrow and load dataset: {str(e2)}")
            # 如果仍然失败，尝试使用fastparquet
            print("Attempting to install and use fastparquet...")
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'fastparquet'], check=True)
                df = pd.read_parquet(alpaca_data_path, engine='fastparquet')
                print(f"Loaded Alpaca dataset with {len(df)} samples after installing fastparquet.")
                return df
            except Exception as e3:
                print(f"Failed to load Alpaca dataset: {str(e3)}")
                raise

# 安全指令过滤器
def is_safe_instruction(instruction, output):
    # 检查指令是否包含不安全内容
    unsafe_keywords = [
        'harm', 'violence', 'kill', 'murder', 'suicide', 'attack',
        'hack', 'malware', 'virus', 'phishing', 'fraud', 'scam',
        'illegal', 'criminal', 'steal', 'theft', 'robbery',
        'drugs', 'drug', 'narcotics', 'marijuana', 'cocaine',
        'weapon', 'guns', 'firearms', 'explosive', 'bomb',
        'hate', 'racist', 'discriminate', 'discrimination',
        'porn', 'pornography', 'adult content', 'child abuse',
        'piracy', 'copyright infringement', 'torrent',
        'fake news', 'misinformation', 'disinformation',
        'political extremism', 'terrorism', 'extremist',
        'money laundering', 'tax evasion', 'counterfeit',
        'identity theft', 'cyberbullying', 'harassment',
        'suicide', 'self-harm', 'eating disorder'
    ]
    
    # 将指令和输出转换为小写进行检查
    instruction_lower = instruction.lower()
    output_lower = output.lower() if output else ""
    
    # 检查是否包含不安全关键词
    for keyword in unsafe_keywords:
        if keyword in instruction_lower or keyword in output_lower:
            return False
    
    # 检查是否有<nooutput>标记（表示可能不安全的指令）
    if '<nooutput>' in output_lower:
        return False
    
    # 检查指令长度是否合理
    if len(instruction) < 5:
        return False
    
    return True

# 生成安全数据集
def generate_safe_dataset(args):
    set_seed()
    
    # 加载Alpaca数据集
    df = load_alpaca_dataset(args.alpaca_data_path)
    
    # 筛选安全指令
    print("Filtering safe instructions...")
    safe_data = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        instruction = row.get('instruction', '')
        output = row.get('output', '')
        
        # 检查是否为安全指令
        if is_safe_instruction(instruction, output):
            safe_data.append({
                'instruction': instruction,
                'input': row.get('input', ''),
                'output': output,
                'text': row.get('text', '')
            })
    
    print(f"Filtered {len(safe_data)} safe instructions from the dataset.")
    
    # 随机选择指定数量的样本
    if len(safe_data) > args.num_samples:
        selected_samples = random.sample(safe_data, args.num_samples)
    else:
        selected_samples = safe_data
        print(f"Warning: Only {len(safe_data)} safe instructions available, less than requested {args.num_samples}.")
    
    # 保存数据集
    print("Saving safe dataset...")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(selected_samples, f, ensure_ascii=False, indent=2)
    
    print(f"Safe dataset saved to {args.output_path}.")
    print(f"Generated {len(selected_samples)} safe samples.")

# 主函数
def main():
    args = parse_args()
    generate_safe_dataset(args)

if __name__ == '__main__':
    main()