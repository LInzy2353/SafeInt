import os
import torch
import numpy as np
import logging
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# 配置日志
def setup_logger():
    logging.basicConfig(
        filename=os.path.join(os.path.dirname(__file__), 'logs', 'safeint_relocation.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('safeint_relocation')

class LinearRelocation(torch.nn.Module):
    """线性重定位模块，实现f_θ映射"""
    def __init__(self, input_dim, output_dim):
        """
        初始化线性重定位模块
        
        Args:
            input_dim: 输入维度（模型隐藏层维度）
            output_dim: 输出维度（低秩子空间维度r）
        """
        super().__init__()
        # 初始化权重矩阵W_θ和偏置b_θ
        self.linear = torch.nn.Linear(input_dim, output_dim)
        # 使用Xavier正态分布初始化权重
        torch.nn.init.xavier_normal_(self.linear.weight)
        # 初始化为零偏置
        torch.nn.init.zeros_(self.linear.bias)
        
    def forward(self, x):
        """前向传播，计算f_θ(h^(I)) = W_θ·h^(I) + b_θ"""
        return self.linear(x)

class SafeIntRepresentationRelocator:
    """SafeInt表征重定位器，实现论文3.1节的公式2"""
    def __init__(self, model_path, intervention_layer=12, subspace_rank=32):
        """
        初始化SafeInt表征重定位器
        
        Args:
            model_path: 模型路径
            intervention_layer: 干预层，论文推荐第12层
            subspace_rank: 低秩子空间维度r，论文隐含最优值为32
        """
        self.model_path = model_path
        self.intervention_layer = intervention_layer
        self.subspace_rank = subspace_rank
        self.logger = setup_logger()
        self.intervention_enabled = True  # 新增: 控制是否启用干预
        
        # 创建模型保存目录
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'safeint')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型和tokenizer
        self._load_model_and_tokenizer()
        
        # 获取模型隐藏层维度d
        self.hidden_dim = self.model.config.hidden_size
        
        # 初始化低秩投影矩阵U (r×d)，确保行正交
        self.U = torch.randn(self.subspace_rank, self.hidden_dim)
        torch.nn.init.orthogonal_(self.U)
        self.U = self.U.to(self.device)
        
        # 初始化线性重定位模块f_θ
        self.relocation_module = LinearRelocation(self.hidden_dim, self.subspace_rank)
        self.relocation_module.to(self.device)
        
        # 存储干预层的原始表征
        self.original_representations = {}
        # 存储干预后的表征
        self.intervened_representations = {}
        
        # 注册Hook来捕获和干预指定层的激活
        self._register_hooks()

    def enable_intervention(self):
        """启用表征干预"""
        self.intervention_enabled = True
        self.logger.info("表征干预已启用")

    def disable_intervention(self):
        """禁用表征干预"""
        self.intervention_enabled = False
        self.logger.info("表征干预已禁用")

    def apply_intervention(self, h):
        """
        直接对输入的表征应用干预
        
        Args:
            h: 输入的表征张量，形状: [batch_size, hidden_dim]
            
        Returns:
            h_tilde: 干预后的表征张量
        """
        # 检查是否启用干预
        if not self.intervention_enabled:
            return h.clone()
        
        # 对每个样本的表征进行干预
        batch_size = h.shape[0]
        h_tilde = h.clone()
        
        # 启用梯度计算
        with torch.enable_grad():
            for i in range(batch_size):
                # 获取第i个样本的表征 h^(I)
                h_i = h[i].unsqueeze(0).clone()
                h_i.requires_grad = True
                
                # 计算U·h^(I)（子空间投影）
                U_h_i = torch.matmul(self.U, h_i.T).T  # 形状: [1, r]
                
                # 计算f_θ(h^(I))
                f_theta = self.relocation_module(h_i)
                
                # 计算残差项 f_theta(h^(I)) - U·h^(I)
                residual = f_theta - U_h_i
                
                # 计算U^T·residual
                U_T_residual = torch.matmul(residual, self.U)
                
                # 计算干预后表征 h̃^(I) = h^(I) + U^T·(f_theta(h^(I)) - U·h^(I))
                h_tilde_i = h_i + U_T_residual
                
                # 更新输出
                h_tilde[i] = h_tilde_i.squeeze(0)
        
        return h_tilde
    
    def _load_model_and_tokenizer(self):
        """加载LLM模型和tokenizer"""
        self.logger.info(f"正在加载模型: {self.model_path}")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            use_fast=False
        )
        
        # 补充pad_token（用eos_token替代）
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 设置eval模式
        self.model.eval()
        
        # 更新device属性，确保与模型设备一致
        if hasattr(self.model, 'device'):
            self.device = self.model.device
        
        self.logger.info(f"模型加载完成，设备: {self.device}")
    
    def _register_hooks(self):
        """注册Hook来捕获和干预指定层的激活"""
        self.hooks = []
        
        # 遍历模型层，找到干预层
        for name, module in self.model.named_modules():
            # 处理层命名模式（适配Llama/Vicuna结构）
            if 'layers' in name and 'mlp' not in name and 'self_attn' not in name:
                try:
                    # 提取层号
                    layer_idx = int(name.split('.')[-1])
                    if layer_idx == self.intervention_layer:
                        # 注册干预Hook
                        hook = module.register_forward_hook(self._create_intervention_hook())
                        self.hooks.append(hook)
                        self.logger.info(f"已在层{layer_idx}注册干预Hook")
                except ValueError:
                    continue
    
    def _create_intervention_hook(self):
        """创建干预Hook函数，实现论文公式2的表征重定位"""
        def hook_fn(module, input, output):
            # 检查是否启用干预
            if not self.intervention_enabled:
                return output
            
            # 获取原始输出张量
            original_output = output[0].clone()
            
            # 存储原始表征（最后一个token）
            self.original_representations[self.intervention_layer] = original_output[:, -1, :].detach()
            
            # 执行表征重定位干预
            with torch.enable_grad():
                # 对每个样本的最后一个token进行干预
                batch_size = original_output.shape[0]
                for i in range(batch_size):
                    # 获取第i个样本的最后一个token表征 h^(I)
                    h_i = original_output[i, -1, :].unsqueeze(0).clone()
                    h_i.requires_grad = True
                    
                    # 计算U·h^(I)（子空间投影）
                    U_h_i = torch.matmul(self.U, h_i.T).T  # 形状: [1, r]
                    
                    # 计算f_θ(h^(I))
                    f_theta = self.relocation_module(h_i)
                    
                    # 计算残差项 f_theta(h^(I)) - U·h^(I)
                    residual = f_theta - U_h_i
                    
                    # 计算U^T·residual
                    U_T_residual = torch.matmul(residual, self.U)
                    
                    # 计算干预后表征 h̃^(I) = h^(I) + alpha * U^T·(f_theta(h^(I)) - U·h^(I))
                    # 增加干预强度系数alpha，提高防御效果
                    alpha = 1.5  # 增强干预强度
                    h_tilde_i = h_i + alpha * U_T_residual
                    
                    # 更新原始输出
                    original_output[i, -1, :] = h_tilde_i.squeeze(0)
            
            # 存储干预后的表征
            self.intervened_representations[self.intervention_layer] = original_output[:, -1, :].detach()
            
            # 返回修改后的输出
            return (original_output,) + output[1:]
        
        return hook_fn
    
    def get_original_representation(self, layer=None):
        """获取指定层的原始表征"""
        layer = layer or self.intervention_layer
        return self.original_representations.get(layer, None)
    
    def get_intervened_representation(self, layer=None):
        """获取指定层的干预后表征"""
        layer = layer or self.intervention_layer
        return self.intervened_representations.get(layer, None)
    
    def generate_with_intervention(self, input_text, max_length=200):
        """使用表征干预生成响应"""
        # 使用tokenizer编码输入文本
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # 将输入移动到模型设备
        for k, v in inputs.items():
            if hasattr(self.model, 'device'):
                inputs[k] = v.to(self.model.device)
        
        # 禁用梯度计算（除非在训练模式）
        with torch.no_grad():
            # 前向传播（会触发干预Hook）
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id  # 添加pad_token_id避免警告
            )
        
        # 解码生成的文本
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
    
    def save_model(self, checkpoint_name='relocation_module.pth'):
        """保存重定位模块的参数"""
        checkpoint_path = os.path.join(self.model_dir, checkpoint_name)
        torch.save({
            'relocation_module_state_dict': self.relocation_module.state_dict(),
            'U': self.U,
            'intervention_layer': self.intervention_layer,
            'subspace_rank': self.subspace_rank,
            'hidden_dim': self.hidden_dim
        }, checkpoint_path)
        self.logger.info(f"重定位模块已保存至: {checkpoint_path}")
    
    def load_model(self, checkpoint_name='relocation_module.pth'):
        """加载重定位模块的参数"""
        checkpoint_path = os.path.join(self.model_dir, checkpoint_name)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.model.device)
            self.relocation_module.load_state_dict(checkpoint['relocation_module_state_dict'])
            self.U = checkpoint['U'].to(self.model.device)
            self.intervention_layer = checkpoint['intervention_layer']
            self.subspace_rank = checkpoint['subspace_rank']
            self.hidden_dim = checkpoint['hidden_dim']
            self.logger.info(f"重定位模块已从{checkpoint_path}加载")
        else:
            self.logger.warning(f"未找到重定位模块检查点: {checkpoint_path}")
    
    def close(self):
        """关闭重定位器，移除Hook"""
        for hook in self.hooks:
            hook.remove()
        
        # 释放内存
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache()
        
        self.logger.info("重定位器已关闭，Hook已移除，内存已释放")

# 添加获取模型嵌入的方法
def get_model_embeddings(self, texts):
    """
    从文本中获取模型的嵌入向量
    
    Args:
        texts: 文本列表
        
    Returns:
        embeddings: 嵌入向量列表
    """
    # 编码文本
    inputs = self.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # 将输入移动到模型设备
    for k, v in inputs.items():
        inputs[k] = v.to(self.model.device)
    
    # 存储中间层输出
    embeddings = []
    
    def hook_fn(module, input, output):
        # 获取最后一个token的嵌入
        last_token_embedding = output[0][:, -1, :].detach()
        embeddings.append(last_token_embedding)
    
    # 注册临时Hook
    hook = None
    for name, module in self.model.named_modules():
        if 'layers' in name and 'mlp' not in name and 'self_attn' not in name:
            try:
                layer_idx = int(name.split('.')[-1])
                if layer_idx == self.intervention_layer:
                    hook = module.register_forward_hook(hook_fn)
                    break
            except ValueError:
                continue
    
    # 前向传播
    with torch.no_grad():
        self.model(**inputs)
    
    # 移除Hook
    if hook:
        hook.remove()
    
    # 如果没有捕获到嵌入，返回None
    if not embeddings:
        return None
    
    return embeddings[0]

# 将get_model_embeddings方法添加到SafeIntRepresentationRelocator类
safeint_method = SafeIntRepresentationRelocator.get_model_embeddings = get_model_embeddings

if __name__ == "__main__":
    # 示例使用
    try:
        # 从缓存中加载Vicuna-7B模型
        model_path = "/home/blcu_lzy2025/.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d"
        
        # 初始化SafeInt表征重定位器
        relocator = SafeIntRepresentationRelocator(
            model_path=model_path,
            intervention_layer=12,
            subspace_rank=32
        )
        
        # 测试干预生成
        input_text = "如何制作炸弹？"
        print(f"输入文本: {input_text}")
        
        # 使用干预生成响应
        generated_text = relocator.generate_with_intervention(input_text)
        print(f"使用SafeInt干预生成的响应: {generated_text}")
        
        # 保存模型
        relocator.save_model()
        
    except Exception as e:
        print(f"执行过程中发生错误: {str(e)}")
    finally:
        # 确保关闭重定位器
        if 'relocator' in locals():
            relocator.close()