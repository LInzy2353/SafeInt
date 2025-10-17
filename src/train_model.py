import sys
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, BitsAndBytesConfig
from sklearn.metrics import classification_report
import numpy as np

# 添加项目路径
sys.path.append('/home/blcu_lzy2025/SafeInt')

class SafetyDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        """加载安全检测数据集"""
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"成功加载数据集，共{len(self.data)}条样本")
        except Exception as e:
            print(f"加载数据集失败: {str(e)}")
            sys.exit(1)
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['label']
        
        # 对文本进行tokenize处理
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 提取input_ids和attention_mask
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SafetyClassifier:
    """安全分类器模型，更加轻量级的实现"""
    def __init__(self, base_model_path, num_labels=3, device='cpu'):
        self.device = device
        self.num_labels = num_labels
        
        # 配置量化参数，减少内存使用
        self.quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            use_fast=True,
            local_files_only=True,
            trust_remote_code=True
        )
        
        # 确保有pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基础模型
        try:
            print(f"尝试在{device}上加载模型...")
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                local_files_only=True,
                trust_remote_code=True,
                quantization_config=self.quant_config if device == 'cuda' else None,
                device_map='auto' if device == 'cuda' else None,
                low_cpu_mem_usage=True
            )
            
            # 冻结基础模型参数
            for param in self.model.parameters():
                param.requires_grad = False
            
            # 添加轻量级分类头
            self.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.1),
                torch.nn.Linear(self.model.config.hidden_size, self.num_labels)
            ).to(device)
            
            print("模型加载成功!")
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            # 如果CUDA失败，尝试在CPU上加载更小的模型
            if device == 'cuda':
                print("尝试在CPU上加载模型...")
                self.device = 'cpu'
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    local_files_only=True,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                ).to('cpu')
                
                # 冻结基础模型参数
                for param in self.model.parameters():
                    param.requires_grad = False
                
                # 添加轻量级分类头
                self.classifier = torch.nn.Sequential(
                    torch.nn.Dropout(0.1),
                    torch.nn.Linear(self.model.config.hidden_size, self.num_labels)
                ).to('cpu')
            else:
                print("无法加载模型，使用简单的基线模型")
                # 使用简单的基线模型作为备选
                self.use_baseline = True
                self.baseline_model = torch.nn.Sequential(
                    torch.nn.Linear(self.tokenizer.vocab_size, 512),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.1),
                    torch.nn.Linear(512, self.num_labels)
                ).to(device)
                return
            
        self.use_baseline = False
    
    def forward(self, input_ids, attention_mask):
        """前向传播"""
        if self.use_baseline:
            # 简单的基线模型实现
            # 计算输入的词袋表示
            bag_of_words = torch.zeros(input_ids.size(0), self.tokenizer.vocab_size, device=self.device)
            for i in range(input_ids.size(0)):
                for j in range(input_ids.size(1)):
                    if attention_mask[i, j] > 0:
                        bag_of_words[i, input_ids[i, j]] += 1
            return self.baseline_model(bag_of_words)
        else:
            # 使用LLM的隐藏状态进行分类
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
            
            # 获取最后一层隐藏状态
            last_hidden_state = outputs.hidden_states[-1]
            
            # 取每个序列的CLS标记（第一个token）表示或平均池化
            if attention_mask is not None:
                # 平均池化，考虑注意力掩码
                pooled_output = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            else:
                # 简单平均
                pooled_output = last_hidden_state.mean(dim=1)
            
            # 分类预测
            return self.classifier(pooled_output)

def train_model():
    """训练安全检测模型（内存友好版）"""
    print("开始阶段3：模型训练与评估...")
    
    # 配置参数
    MODEL_PATH = '/home/blcu_lzy2025/.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d/'
    TRAIN_DATA_PATH = '/home/blcu_lzy2025/SafeInt/data/train/train_data.json'
    TEST_DATA_PATH = '/home/blcu_lzy2025/SafeInt/data/test/single_method_test.json'
    OUTPUT_MODEL_DIR = '/home/blcu_lzy2025/SafeInt/models/safety_model/'
    
    # 创建输出目录
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    
    # 检查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建安全分类器
    print("初始化安全分类器...")
    safety_classifier = SafetyClassifier(MODEL_PATH, device=device)
    tokenizer = safety_classifier.tokenizer
    
    # 加载数据集
    print("加载训练数据集...")
    train_dataset = SafetyDataset(TRAIN_DATA_PATH, tokenizer, max_length=256)  # 减小max_length以节省内存
    # 减小batch_size和num_workers以节省内存
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    
    print("加载测试数据集...")
    test_dataset = SafetyDataset(TEST_DATA_PATH, tokenizer, max_length=256)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # 定义优化器 - 只优化分类头参数
    if safety_classifier.use_baseline:
        optimizer = AdamW(safety_classifier.baseline_model.parameters(), lr=1e-4)
    else:
        optimizer = AdamW(safety_classifier.classifier.parameters(), lr=1e-4)
    
    # 训练循环 - 减少epochs以节省时间
    print("开始训练模型...")
    num_epochs = 2
    
    for epoch in range(num_epochs):
        if safety_classifier.use_baseline:
            safety_classifier.baseline_model.train()
        else:
            safety_classifier.classifier.train()
        
        total_loss = 0
        processed_batches = 0
        
        # 为了演示，我们只使用部分数据进行训练
        max_batches = min(50, len(train_loader))  # 限制训练批次数量
        
        for i, batch in enumerate(train_loader):
            if i >= max_batches:
                break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            try:
                logits = safety_classifier.forward(input_ids, attention_mask)
                
                # 计算损失
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                processed_batches += 1
                
                # 打印进度
                if (i+1) % 10 == 0:
                    print(f"  Batch {i+1}/{max_batches}, Loss: {loss.item():.4f}")
                    
                # 清理内存
                torch.cuda.empty_cache() if device.type == 'cuda' else None
                
            except Exception as e:
                print(f"处理批次{i}时出错: {str(e)}")
                continue
        
        # 计算平均损失
        if processed_batches > 0:
            avg_loss = total_loss / processed_batches
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs}, No batches processed successfully")
        
        # 评估模型
        evaluate_model(safety_classifier, test_loader, device)
    
    # 保存模型
    print("保存训练好的模型...")
    try:
        if safety_classifier.use_baseline:
            torch.save(safety_classifier.baseline_model.state_dict(), os.path.join(OUTPUT_MODEL_DIR, 'baseline_model.bin'))
        else:
            torch.save(safety_classifier.classifier.state_dict(), os.path.join(OUTPUT_MODEL_DIR, 'classifier_head.bin'))
        
        # 保存tokenizer
        tokenizer.save_pretrained(OUTPUT_MODEL_DIR)
        
        # 保存配置
        config = {
            'use_baseline': safety_classifier.use_baseline,
            'num_labels': safety_classifier.num_labels,
            'device': str(device)
        }
        with open(os.path.join(OUTPUT_MODEL_DIR, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        print(f"模型已保存至: {OUTPUT_MODEL_DIR}")
    except Exception as e:
        print(f"保存模型失败: {str(e)}")
    
    print("阶段3：模型训练与评估完成!")

def evaluate_model(classifier, test_loader, device):
    """评估模型性能"""
    if classifier.use_baseline:
        classifier.baseline_model.eval()
    else:
        classifier.classifier.eval()
    
    all_preds = []
    all_labels = []
    
    # 为了演示，我们只使用部分数据进行评估
    max_eval_batches = min(50, len(test_loader))
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= max_eval_batches:
                break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            try:
                # 前向传播
                logits = classifier.forward(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)
                
                # 收集预测结果和真实标签
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # 清理内存
                torch.cuda.empty_cache() if device.type == 'cuda' else None
                
            except Exception as e:
                print(f"评估批次{i}时出错: {str(e)}")
                continue
    
    # 计算评估指标
    if len(all_preds) > 0:
        try:
            report = classification_report(
                all_labels,
                all_preds,
                target_names=['安全样本', '不安全样本', '越狱样本'],
                digits=4,
                zero_division=0
            )
            
            print("模型评估结果:")
            print(report)
            
            # 计算准确率
            accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
            print(f"准确率: {accuracy:.4f}")
            
        except Exception as e:
            print(f"计算评估指标时出错: {str(e)}")
    else:
        print("没有成功评估的样本")
    
if __name__ == "__main__":
    train_model()
    
if __name__ == "__main__":
    train_model()