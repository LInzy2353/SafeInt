import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

class LLMRepresentationExtractor:
    def __init__(self, model_name, model_path, layers=None, batch_size=4):
        """初始化LLM表征提取器
        
        Args:
            model_name: 模型名称
            model_path: 模型路径
            layers: 目标层列表
            batch_size: 批处理大小
        """
        self.model_name = model_name
        self.model_path = model_path
        self.batch_size = batch_size
        
        # 默认选择中间层到深层（Llama2-7B的10-25层）
        self.target_layers = layers if layers else list(range(10, 26))
        
        # 存储提取的表征
        self.extracted_representations = {layer: [] for layer in self.target_layers}
        
        # 确保输出目录存在
        os.makedirs("../embeddings", exist_ok=True)
        
        # 加载模型和tokenizer
        self._load_model_and_tokenizer()
        
        # 注册Hook
        self._register_hooks()
    
    def _load_model_and_tokenizer(self):
        """加载LLM模型和tokenizer"""
        print(f"正在加载模型: {self.model_path}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            use_fast=False  # 使用完整tokenizer以支持pad_token设置
        )
        
        # 补充pad_token（用eos_token替代）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型，设置torch_dtype=float16和device_map="auto"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 设置eval模式
        self.model.eval()
        
        print(f"模型加载完成，设备: {self.model.device if hasattr(self.model, 'device') else '多设备'}")
    
    def _register_hooks(self):
        """注册前向Hook来捕获指定层的激活"""
        self.hooks = []
        
        # 遍历模型层，注册Hook到目标层
        for name, module in self.model.named_modules():
            # 处理Llama2的层命名模式
            if 'layers' in name and 'mlp' not in name and 'self_attn' not in name:
                try:
                    # 提取层号
                    layer_idx = int(name.split('.')[-1])
                    if layer_idx in self.target_layers:
                        hook = module.register_forward_hook(self._create_hook(layer_idx))
                        self.hooks.append(hook)
                except ValueError:
                    continue
    
    def _create_hook(self, layer_idx):
        """创建Hook函数来捕获最后一个token的表征"""
        def hook_fn(module, input, output):
            # 提取最后一个token的表征：output[0][:, -1, :]
            last_token_representation = output[0][:, -1, :].cpu().detach().numpy()
            self.extracted_representations[layer_idx].append(last_token_representation)
        return hook_fn
    
    def _process_batch(self, texts):
        """处理一批文本并提取表征"""
        # 使用tokenizer编码文本，设置max_length=512
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # 将输入移动到模型设备
        for k, v in inputs.items():
            if hasattr(self.model, 'device'):
                inputs[k] = v.to(self.model.device)
            # 如果是多设备，transformers会自动处理
        
        # 禁用梯度计算
        with torch.no_grad():
            # 前向传播（会触发Hook）
            self.model(**inputs)
    
    def extract_from_dataset(self, dataset_path, split_name):
        """从数据集中提取表征
        
        Args:
            dataset_path: 数据集路径
            split_name: 数据集分割名称（train或test）
        """
        print(f"正在从{split_name}数据集提取表征: {dataset_path}")
        
        # 加载数据集
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # 提取文本列表
        texts = [item['text'] for item in dataset]
        
        # 批量处理文本
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        for i in range(total_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            
            print(f"处理批次 {i+1}/{total_batches}")
            self._process_batch(batch_texts)
        
        # 合并并保存提取的表征
        self._save_representations(split_name)
        
        # 清空提取的表征，准备下一个数据集
        self.extracted_representations = {layer: [] for layer in self.target_layers}
        
        print(f"{split_name}数据集表征提取完成")
        
    def extract_and_save_embeddings(self, data_file, output_dir, data_type):
        """从数据文件中提取嵌入并保存
        
        Args:
            data_file: 数据文件路径
            output_dir: 输出目录
            data_type: 数据类型（jailbreak/unsafe/safe）
        """
        print(f"正在从文件{data_file}提取{data_type}数据的表征")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载数据集
        with open(data_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # 提取文本列表
        texts = [item['text'] for item in dataset]
        
        # 批量处理文本
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        for i in range(total_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            
            print(f"处理批次 {i+1}/{total_batches}")
            self._process_batch(batch_texts)
        
        # 合并并保存提取的表征
        self._save_representations_by_type(output_dir, data_type)
        
        # 清空提取的表征，准备下一个数据集
        self.extracted_representations = {layer: [] for layer in self.target_layers}
        
        print(f"{data_type}数据表征提取完成")
        
    def _save_representations_by_type(self, output_dir, data_type):
        """根据数据类型保存提取的表征
        
        Args:
            output_dir: 输出目录
            data_type: 数据类型
        """
        for layer_idx, representations in self.extracted_representations.items():
            if representations:
                # 合并所有批次的表征
                all_representations = np.concatenate(representations, axis=0)
                
                # 保存为numpy数组，使用data_type_layer_{layer_idx}.npy格式
                save_path = os.path.join(output_dir, f"{data_type}_layer_{layer_idx}.npy")
                np.save(save_path, all_representations)
                print(f"层 {layer_idx} 的{data_type}表征已保存至: {save_path}")
            else:
                print(f"警告：层 {layer_idx} 没有提取到{data_type}表征")
    
    def _save_representations(self, split_name):
        """保存提取的表征"""
        for layer_idx, representations in self.extracted_representations.items():
            if representations:
                # 合并所有批次的表征
                all_representations = np.concatenate(representations, axis=0)
                
                # 保存为numpy数组
                save_path = f"../embeddings/layer_{layer_idx}_{split_name}.npy"
                np.save(save_path, all_representations)
                print(f"层 {layer_idx} 的{split_name}表征已保存至: {save_path}")
            else:
                print(f"警告：层 {layer_idx} 没有提取到{split_name}表征")
    
    def close(self):
        """关闭提取器，移除Hook"""
        for hook in self.hooks:
            hook.remove()
        
        # 释放内存
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache()
        
        print("提取器已关闭，Hook已移除，内存已释放")


if __name__ == "__main__":
    # 示例使用
    try:
        # 从缓存中加载Vicuna-7B模型
        model_path = "/home/blcu_lzy2025/.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d"
        
        # 初始化提取器，选择中间层到深层（10-25层）
        extractor = LLMRepresentationExtractor(
            model_path=model_path,
            target_layers=list(range(10, 26)),
            batch_size=4
        )
        
        # 提取训练集表征
        extractor.extract_from_dataset("../data/train/train_data.json", "train")
        
        # 提取测试集表征
        extractor.extract_from_dataset("../data/test/single_method_test.json", "test")
        
    except Exception as e:
        print(f"提取过程中发生错误: {str(e)}")
    finally:
        # 确保关闭提取器
        if 'extractor' in locals():
            extractor.close()