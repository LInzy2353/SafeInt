import pandas as pd
import pandas as pd
from datasets import load_dataset
from transformers import pipeline
import numpy as np
import os
import json

# 1. 参数
np.random.seed(42)
SAMPLE_SIZE_DUNSAFE = 200
TOXICITY_THRESHOLD = 0.5

# 2. 加载数据集
print("Loading datasets from Hugging Face mirror...")
malicious_dataset = load_dataset("walledai/MaliciousInstruct", split="train")
jailbreak_dataset = load_dataset("walledai/JailbreakBench", split="train")
print("Datasets loaded successfully.")

# 3. 转换为 DataFrame
malicious_df = pd.DataFrame(malicious_dataset)
jailbreak_df = pd.DataFrame(jailbreak_dataset)

# 4. 统一列名并计算毒性得分
print("Applying toxicity classification...")
toxicity_classifier = pipeline("text-classification", model="facebook/roberta-hate-speech-dynabench-r4-target", trust_remote_code=True)
# MaliciousInstruct 使用 'prompt'
malicious_df = malicious_df.rename(columns={'prompt': 'instruction'})
malicious_df['toxic_score'] = malicious_df['instruction'].apply(lambda x: toxicity_classifier(x)[0]['score'] if x else 0)
# JailbreakBench 使用 'prompt' 或 'target'
jailbreak_df['instruction'] = jailbreak_df.apply(lambda row: row.get('prompt', row.get('target', '')), axis=1)
jailbreak_df['toxic_score'] = jailbreak_df['instruction'].apply(lambda x: toxicity_classifier(x)[0]['score'] if x else 0)

# 5. 过滤和采样
dunsafe_df = pd.concat([
    malicious_df[malicious_df['toxic_score'] > TOXICITY_THRESHOLD],
    jailbreak_df[jailbreak_df['toxic_score'] > TOXICITY_THRESHOLD]
]).drop_duplicates(subset=['instruction'])
if len(dunsafe_df) > SAMPLE_SIZE_DUNSAFE:
    dunsafe_df = dunsafe_df.sample(n=SAMPLE_SIZE_DUNSAFE, random_state=42)
else:
    print(f"Warning: Only {len(dunsafe_df)} harmful samples found, less than requested {SAMPLE_SIZE_DUNSAFE}")

# 6. 保存
dunsafe_df[['instruction', 'toxic_score']].to_csv('dunsafe_dataset.csv', index=False)
print(f"Dunsafe dataset saved to 'dunsafe_dataset.csv' with {len(dunsafe_df)} samples.")

# 7. 验证
print("Sample preview:")
print(dunsafe_df.head())
print(f"Total harmful instructions: {len(dunsafe_df)}")