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
    def __init__(self, model_path, intervention_layer=12, subspace_rank=32, model=None, tokenizer=None):
        """
        初始化SafeInt表征重定位器
        
        Args:
            model_path: 模型路径
            intervention_layer: 干预层，论文推荐第12层
            subspace_rank: 低秩子空间维度r，论文隐含最优值为32
            model: 可选，已加载的模型实例
            tokenizer: 可选，已加载的tokenizer实例
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
        
        # 加载模型和tokenizer（如果没有提供的话）
        if model is not None and tokenizer is not None:
            self.logger.info("使用已提供的模型和tokenizer实例")
            self.model = model
            self.tokenizer = tokenizer
            # 更新device属性，确保与模型设备一致
            if hasattr(self.model, 'device'):
                self.device = self.model.device
        else:
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
    
    def estimate_U(self, safe_representations, unsafe_representations, method='class_center'):
        """
        基于数据驱动估计安全子空间U
        
        Args:
            safe_representations: 安全样本的表征，形状: [n_safe, hidden_dim]
            unsafe_representations: 不安全样本的表征，形状: [n_unsafe, hidden_dim]
            method: 估计方法，'class_center' 或 'svd'
            
        Returns:
            U: 估计的安全子空间投影矩阵，形状: [subspace_rank, hidden_dim]
        """
        self.logger.info(f"开始使用{method}方法估计安全子空间U")
        
        # 确保输入为tensor并移到正确设备
        if not isinstance(safe_representations, torch.Tensor):
            safe_representations = torch.tensor(safe_representations, dtype=torch.float32)
        if not isinstance(unsafe_representations, torch.Tensor):
            unsafe_representations = torch.tensor(unsafe_representations, dtype=torch.float32)
            
        safe_representations = safe_representations.to(self.device)
        unsafe_representations = unsafe_representations.to(self.device)
        
        if method == 'class_center':
            # 方法1: 基于类中心差异的子空间估计
            safe_center = torch.mean(safe_representations, dim=0)  # [hidden_dim]
            unsafe_center = torch.mean(unsafe_representations, dim=0)  # [hidden_dim]
            
            # 计算类中心差异向量
            center_diff = safe_center - unsafe_center  # [hidden_dim]
            center_diff = center_diff / torch.norm(center_diff)  # 归一化
            
            # 构建以类中心差异为主方向的子空间
            # 使用Gram-Schmidt正交化生成正交基
            U = torch.zeros(self.subspace_rank, self.hidden_dim, device=self.device)
            U[0] = center_diff
            
            # 添加随机正交向量
            for i in range(1, self.subspace_rank):
                # 生成随机向量
                random_vec = torch.randn(self.hidden_dim, device=self.device)
                
                # Gram-Schmidt正交化
                for j in range(i):
                    proj = torch.dot(random_vec, U[j]) * U[j]
                    random_vec = random_vec - proj
                
                # 归一化
                random_vec = random_vec / torch.norm(random_vec)
                U[i] = random_vec
                
        elif method == 'svd':
            # 方法2: 基于SVD的子空间估计
            # 计算安全样本的协方差矩阵
            safe_centered = safe_representations - torch.mean(safe_representations, dim=0)
            unsafe_centered = unsafe_representations - torch.mean(unsafe_representations, dim=0)
            
            # 计算类间散布矩阵 S_b = (μ_safe - μ_unsafe)(μ_safe - μ_unsafe)^T
            safe_mean = torch.mean(safe_representations, dim=0)
            unsafe_mean = torch.mean(unsafe_representations, dim=0)
            mean_diff = (safe_mean - unsafe_mean).unsqueeze(0)  # [1, hidden_dim]
            
            # 计算类内散布矩阵 S_w
            safe_cov = torch.matmul(safe_centered.T, safe_centered) / (safe_centered.shape[0] - 1)
            unsafe_cov = torch.matmul(unsafe_centered.T, unsafe_centered) / (unsafe_centered.shape[0] - 1)
            S_w = safe_cov + unsafe_cov
            
            # 计算广义特征值问题的近似解
            # 使用SVD分解 mean_diff
            try:
                U_svd, S_svd, V_svd = torch.svd(mean_diff)
                # 取前subspace_rank个主成分
                U = V_svd[:self.subspace_rank, :]  # [subspace_rank, hidden_dim]
            except:
                # 如果SVD失败，回退到随机正交矩阵
                self.logger.warning("SVD失败，使用随机正交矩阵")
                U = torch.randn(self.subspace_rank, self.hidden_dim, device=self.device)
                torch.nn.init.orthogonal_(U)
        else:
            raise ValueError(f"不支持的估计方法: {method}")
        
        # 确保U是正交的
        U, _ = torch.qr(U.T)
        U = U.T[:self.subspace_rank, :]
        
        # 更新实例的U矩阵
        self.U = U.detach()
        self.logger.info(f"安全子空间U估计完成，形状: {self.U.shape}")
        
        # 计算子空间质量指标
        self._evaluate_subspace_quality(safe_representations, unsafe_representations)
        
        return self.U
    
    def _evaluate_subspace_quality(self, safe_representations, unsafe_representations):
        """评估估计的子空间质量"""
        try:
            # 计算投影后的类间分离度
            safe_proj = torch.matmul(safe_representations, self.U.T)  # [n_safe, subspace_rank]
            unsafe_proj = torch.matmul(unsafe_representations, self.U.T)  # [n_unsafe, subspace_rank]
            
            safe_center_proj = torch.mean(safe_proj, dim=0)
            unsafe_center_proj = torch.mean(unsafe_proj, dim=0)
            
            # 计算投影空间中的类间距离
            inter_class_dist = torch.norm(safe_center_proj - unsafe_center_proj).item()
            
            # 计算投影空间中的类内方差
            safe_var = torch.var(safe_proj, dim=0).mean().item()
            unsafe_var = torch.var(unsafe_proj, dim=0).mean().item()
            intra_class_var = (safe_var + unsafe_var) / 2
            
            # 计算分离度指标
            separation_ratio = inter_class_dist / (intra_class_var + 1e-8)
            
            self.logger.info(f"子空间质量评估 - 类间距离: {inter_class_dist:.4f}, "
                           f"类内方差: {intra_class_var:.4f}, 分离度: {separation_ratio:.4f}")
            
        except Exception as e:
            self.logger.warning(f"子空间质量评估失败: {str(e)}")

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
        
        # 统一设备与dtype
        target_device = h.device
        target_dtype = h.dtype
        self.U = self.U.to(device=target_device, dtype=target_dtype)
        self.relocation_module = self.relocation_module.to(device=target_device, dtype=target_dtype)
        
        # 对每个样本的表征进行干预
        batch_size = h.shape[0]
        h_tilde = h.clone()
        
        # 自适应强度alpha：根据最近一次风险分数（若可用）或默认值
        alpha = getattr(self, 'adaptive_alpha', 1.0)
        if not isinstance(alpha, (float, int)):
            alpha = 1.0
        alpha = max(0.0, min(2.0, float(alpha)))  # 限制在[0,2]
        
        # 启用梯度计算
        with torch.enable_grad():
            for i in range(batch_size):
                # 获取第i个样本的表征 h^(I)，detach确保为叶子张量
                h_i = h[i].unsqueeze(0).detach().to(target_device, target_dtype)
                h_i.requires_grad_(True)
                
                # 计算U·h^(I)（子空间投影）
                U_h_i = torch.matmul(self.U, h_i.T).T  # 形状: [1, r]
                
                # 计算f_θ(h^(I))
                f_theta = self.relocation_module(h_i)
                
                # 计算残差项 f_theta(h^(I)) - U·h^(I)
                residual = f_theta - U_h_i
                
                # 计算U^T·residual，并应用alpha强度
                U_T_residual = torch.matmul(residual, self.U) * alpha
                
                # 计算干预后表征 h̃^(I) = h^(I) + alpha * U^T·(f_theta(h^(I)) - U·h^(I))
                h_tilde_i = h_i + U_T_residual
                
                # 记录距离变化以便验证
                try:
                    delta = (h_tilde_i - h_i).detach()
                    l2 = torch.norm(delta).item()
                    cos = torch.nn.functional.cosine_similarity(h_tilde_i, h_i).item()
                    if not hasattr(self, 'distance_logs'):
                        self.distance_logs = []
                    self.distance_logs.append({'l2': l2, 'cos': float(cos), 'alpha': float(alpha)})
                except Exception:
                    pass
                
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
            # 兼容Tensor/tuple/BaseModelOutput
            if isinstance(output, (tuple, list)):
                hidden_states = output[0]
            elif hasattr(output, 'last_hidden_state'):
                hidden_states = output.last_hidden_state
            else:
                hidden_states = output
            modified_hidden_states = hidden_states.clone()
            # 统一设备与dtype
            target_device = modified_hidden_states.device
            target_dtype = modified_hidden_states.dtype
            self.U = self.U.to(device=target_device, dtype=target_dtype)
            self.relocation_module = self.relocation_module.to(device=target_device, dtype=target_dtype)
            # 存储原始表征（最后一个token）
            self.original_representations[self.intervention_layer] = modified_hidden_states[:, -1, :].detach()
            # 执行表征重定位干预
            with torch.enable_grad():
                batch_size = modified_hidden_states.shape[0]
                for i in range(batch_size):
                    # 使用detach创建叶子张量，再开启梯度
                    h_i = modified_hidden_states[i, -1, :].unsqueeze(0).detach().to(target_device, target_dtype)
                    h_i.requires_grad_(True)
                    U_h_i = torch.matmul(self.U, h_i.T).T
                    f_theta = self.relocation_module(h_i)
                    residual = f_theta - U_h_i
                    U_T_residual = torch.matmul(residual, self.U)
                    # 使用自适应alpha，限制到[0,2]
                    alpha = getattr(self, 'adaptive_alpha', 1.0)
                    try:
                        alpha = float(alpha)
                    except Exception:
                        alpha = 1.0
                    alpha = max(0.0, min(2.0, alpha))
                    h_tilde_i = h_i + alpha * U_T_residual
                    modified_hidden_states[i, -1, :] = h_tilde_i.squeeze(0)
                    # 记录距离变化，便于验证
                    try:
                        delta = (h_tilde_i - h_i).detach()
                        l2 = torch.norm(delta).item()
                        cos = torch.nn.functional.cosine_similarity(h_tilde_i, h_i).item()
                        if not hasattr(self, 'distance_logs'):
                            self.distance_logs = []
                        self.distance_logs.append({'l2': l2, 'cos': float(cos), 'alpha': float(alpha), 'source': 'hook'})
                    except Exception:
                        pass
            # 存储干预后的表征
            self.intervened_representations[self.intervention_layer] = modified_hidden_states[:, -1, :].detach()
            # 返回修改后的输出，保持原输出结构
            if isinstance(output, (tuple, list)):
                return (modified_hidden_states,) + tuple(output[1:])
            elif hasattr(output, 'last_hidden_state'):
                output.last_hidden_state = modified_hidden_states
                return output
            else:
                return modified_hidden_states
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
    """
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
    embeddings_dict = {}
    def create_hook_fn(layer_idx):
        def hook_fn(module, input, output):
            # 兼容Tensor/tuple/BaseModelOutput
            if isinstance(output, (tuple, list)):
                hidden_states = output[0]
            elif hasattr(output, 'last_hidden_state'):
                hidden_states = output.last_hidden_state
            else:
                hidden_states = output
            last_token_embedding = hidden_states[:, -1, :].detach()
            embeddings_dict[layer_idx] = last_token_embedding
        return hook_fn
    hooks = []
    for name, module in self.model.named_modules():
        if 'layers' in name and 'mlp' not in name and 'self_attn' not in name:
            try:
                layer_idx = int(name.split('.')[-1])
                if layer_idx >= self.intervention_layer and layer_idx < self.intervention_layer + 13:
                    hook = module.register_forward_hook(create_hook_fn(layer_idx))
                    hooks.append(hook)
            except ValueError:
                continue
    with torch.no_grad():
        model_inputs = {k: v for k, v in inputs.items() if k != 'token_type_ids'}
        self.model(**model_inputs)
    for hook in hooks:
        hook.remove()
    return embeddings_dict
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