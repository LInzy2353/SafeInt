import os
import torch
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 导入其他必要的模块
from safeint_representation_relocation import SafeIntRepresentationRelocator
from safeint_representation_alignment import SafeIntRepresentationAligner

# 配置日志
def setup_logger():
    logging.basicConfig(
        filename=os.path.join(os.path.dirname(__file__), 'logs', 'safeint_reconstruction.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('safeint_reconstruction')

class SafeIntRepresentationReconstructor:
    """SafeInt表征重建器，实现论文3.3节的公式6-7"""
    def __init__(self, model_path, intervention_layer=12, alignment_layers=None,
                 subspace_rank=32, alpha=1.0, beta=0.1, temperature=0.1):
        """
        初始化SafeInt表征重建器
        
        Args:
            model_path: 模型路径
            intervention_layer: 干预层
            alignment_layers: 对齐层列表
            subspace_rank: 低秩子空间维度r
            alpha: 对齐损失权重，论文默认1.0
            beta: 重建损失权重，论文隐含最优值为0.1
            temperature: 对比损失的温度参数
        """
        self.model_path = model_path
        self.intervention_layer = intervention_layer
        self.alignment_layers = alignment_layers or list(range(13, 25))  # Vicuna设置
        self.subspace_rank = subspace_rank
        self.alpha = alpha  # 对齐损失权重
        self.beta = beta    # 重建损失权重
        self.temperature = temperature
        self.logger = setup_logger()
        
        # 创建结果保存目录
        self.results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'reconstruction')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 初始化表征重定位器
        self.relocator = SafeIntRepresentationRelocator(
            model_path=model_path,
            intervention_layer=intervention_layer,
            subspace_rank=subspace_rank
        )
        
        # 初始化表征对齐器
        self.aligner = SafeIntRepresentationAligner(
            model_path=model_path,
            intervention_layer=intervention_layer,
            alignment_layers=alignment_layers,
            subspace_rank=subspace_rank,
            temperature=temperature
        )
        
        # MSE损失函数
        self.mse_loss = torch.nn.MSELoss()
    
    def compute_reconstruction_loss(self, original_embeddings, intervened_embeddings):
        """计算重建损失 L_recon（论文公式6）
        
        Args:
            original_embeddings: 原始表征
            intervened_embeddings: 干预后表征
        
        Returns:
            reconstruction_loss: 重建损失值
        """
        try:
            # 计算MSE损失
            loss = self.mse_loss(original_embeddings, intervened_embeddings)
            return loss
        except Exception as e:
            self.logger.error(f"计算重建损失时出错: {str(e)}")
            return torch.tensor(0.0)
    
    def compute_total_loss(self, 
                          jailbreak_embeddings_dict, unsafe_embeddings_dict, 
                          original_jailbreak_embeddings_dict, safe_embeddings_dict,
                          original_safe_embeddings_dict, original_unsafe_embeddings_dict):
        """计算总损失 L_total（论文公式7）
        
        Args:
            jailbreak_embeddings_dict: 干预后越狱样本各层表征
            unsafe_embeddings_dict: 干预后不安全样本各层表征
            original_jailbreak_embeddings_dict: 原始越狱样本各层表征
            safe_embeddings_dict: 干预后安全样本各层表征
            original_safe_embeddings_dict: 原始安全样本各层表征
            original_unsafe_embeddings_dict: 原始不安全样本各层表征
        
        Returns:
            total_loss: 总损失值
            loss_components: 损失各部分的详细值
        """
        try:
            # 1. 计算对齐损失（分类损失 + 对比损失）
            alignment_loss, alignment_loss_details = self.aligner.compute_alignment_loss(
                jailbreak_embeddings_dict, 
                unsafe_embeddings_dict, 
                original_jailbreak_embeddings_dict, 
                safe_embeddings_dict
            )
            
            # 2. 计算重建损失
            # 安全样本的重建损失
            safe_recon_loss = 0.0
            for layer in self.alignment_layers:
                if layer in original_safe_embeddings_dict and layer in safe_embeddings_dict:
                    layer_loss = self.compute_reconstruction_loss(
                        original_safe_embeddings_dict[layer], 
                        safe_embeddings_dict[layer]
                    )
                    safe_recon_loss += layer_loss
            
            # 不安全样本的重建损失
            unsafe_recon_loss = 0.0
            for layer in self.alignment_layers:
                if layer in original_unsafe_embeddings_dict and layer in unsafe_embeddings_dict:
                    layer_loss = self.compute_reconstruction_loss(
                        original_unsafe_embeddings_dict[layer], 
                        unsafe_embeddings_dict[layer]
                    )
                    unsafe_recon_loss += layer_loss
            
            # 总重建损失（论文公式6）
            total_recon_loss = safe_recon_loss + unsafe_recon_loss
            
            # 3. 计算总损失（论文公式7）
            total_loss = self.alpha * alignment_loss + self.beta * total_recon_loss
            
            # 记录损失组件
            loss_components = {
                'alignment_loss': alignment_loss.item(),
                'safe_reconstruction_loss': safe_recon_loss.item(),
                'unsafe_reconstruction_loss': unsafe_recon_loss.item(),
                'total_reconstruction_loss': total_recon_loss.item(),
                'total_loss': total_loss.item()
            }
            
            self.logger.info(f"总损失: {total_loss.item():.4f}")
            self.logger.info(f"  - 对齐损失 (x{self.alpha}): {alignment_loss.item():.4f}")
            self.logger.info(f"  - 重建损失 (x{self.beta}): {total_recon_loss.item():.4f}")
            self.logger.info(f"    - 安全样本重建损失: {safe_recon_loss.item():.4f}")
            self.logger.info(f"    - 不安全样本重建损失: {unsafe_recon_loss.item():.4f}")
            
            # 可视化损失组件
            self._visualize_loss_components(loss_components)
            
            return total_loss, loss_components
        except Exception as e:
            self.logger.error(f"计算总损失时出错: {str(e)}")
            return torch.tensor(0.0), {}
    
    def _visualize_loss_components(self, loss_components):
        """可视化损失组件分布"""
        if not loss_components:
            return
        
        try:
            # 确保使用英文显示
            plt.rcParams["font.family"] = ["Arial", "Helvetica", "Times New Roman", "sans-serif"]
            plt.rcParams["axes.unicode_minus"] = False
            
            # 准备数据
            components = ['Alignment Loss', 'Safe Reconstruction Loss', 'Unsafe Reconstruction Loss', 'Total Loss']
            values = [
                loss_components['alignment_loss'] * self.alpha,
                loss_components['safe_reconstruction_loss'] * self.beta,
                loss_components['unsafe_reconstruction_loss'] * self.beta,
                loss_components['total_loss']
            ]
            
            # 创建堆叠柱状图
            plt.figure(figsize=(10, 6))
            
            # 对齐损失部分
            align_bar = plt.bar(components[0], values[0], color='red', label='Alignment Loss (x{})'.format(self.alpha))
            
            # 重建损失部分
            safe_recon_bar = plt.bar(components[1], values[1], color='blue', label='Safe Reconstruction Loss (x{})'.format(self.beta))
            unsafe_recon_bar = plt.bar(components[2], values[2], color='green', label='Unsafe Reconstruction Loss (x{})'.format(self.beta))
            
            # 总损失部分
            total_bar = plt.bar(components[3], values[3], color='purple', label='Total Loss')
            
            # 添加数值标签
            def add_labels(bars):
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height, 
                             '{:.4f}'.format(height), 
                             ha='center', va='bottom')
            
            add_labels(align_bar)
            add_labels(safe_recon_bar)
            add_labels(unsafe_recon_bar)
            add_labels(total_bar)
            
            plt.title('SafeInt Loss Components')
            plt.ylabel('Loss Value')
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.legend()
            
            # 保存图表
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'loss_components.png'))
            plt.close()
            
            self.logger.info("损失组件图已保存")
        except Exception as e:
            self.logger.error(f"可视化损失组件时出错: {str(e)}")
    
    def get_model_parameters(self):
        """获取需要训练的模型参数"""
        # 只返回重定位模块的参数（W_θ和b_θ），U不参与训练
        return self.relocator.relocation_module.parameters()
    
    def close(self):
        """关闭重建器，释放资源"""
        if hasattr(self, 'relocator'):
            self.relocator.close()
        if hasattr(self, 'aligner'):
            self.aligner.close()
        
        self.logger.info("SafeInt表征重建器已关闭")

if __name__ == "__main__":
    # 示例使用
    try:
        # 从缓存中加载Vicuna-7B模型
        model_path = "/home/blcu_lzy2025/.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d"
        
        # 初始化SafeInt表征重建器
        reconstructor = SafeIntRepresentationReconstructor(
            model_path=model_path,
            intervention_layer=12,
            alignment_layers=list(range(13, 25)),
            subspace_rank=32,
            alpha=1.0,
            beta=0.1,
            temperature=0.1
        )
        
        # 创建模拟数据
        def create_mock_embeddings(batch_size, hidden_dim):
            return torch.randn(batch_size, hidden_dim)
        
        # 模拟数据
        hidden_dim = 4096  # Vicuna-7B的隐藏层维度
        batch_size = 16
        
        # 为每个层创建模拟表征
        jailbreak_embeddings_dict = {}
        unsafe_embeddings_dict = {}
        original_jailbreak_embeddings_dict = {}
        safe_embeddings_dict = {}
        original_safe_embeddings_dict = {}
        original_unsafe_embeddings_dict = {}
        
        for layer in reconstructor.alignment_layers:
            # 原始表征
            original_jailbreak_embeddings_dict[layer] = create_mock_embeddings(batch_size, hidden_dim)
            original_safe_embeddings_dict[layer] = create_mock_embeddings(batch_size, hidden_dim)
            original_unsafe_embeddings_dict[layer] = create_mock_embeddings(batch_size, hidden_dim)
            
            # 干预后表征（添加一些小扰动模拟干预效果）
            jailbreak_embeddings_dict[layer] = original_jailbreak_embeddings_dict[layer] + torch.randn_like(original_jailbreak_embeddings_dict[layer]) * 0.1
            safe_embeddings_dict[layer] = original_safe_embeddings_dict[layer] + torch.randn_like(original_safe_embeddings_dict[layer]) * 0.01
            unsafe_embeddings_dict[layer] = original_unsafe_embeddings_dict[layer] + torch.randn_like(original_unsafe_embeddings_dict[layer]) * 0.01
        
        # 计算总损失
        total_loss, loss_components = reconstructor.compute_total_loss(
            jailbreak_embeddings_dict, 
            unsafe_embeddings_dict, 
            original_jailbreak_embeddings_dict, 
            safe_embeddings_dict,
            original_safe_embeddings_dict, 
            original_unsafe_embeddings_dict
        )
        
        print(f"总损失: {total_loss.item():.4f}")
        print(f"损失组件: {loss_components}")
        
        # 获取可训练参数
        parameters = reconstructor.get_model_parameters()
        print(f"可训练参数数量: {sum(p.numel() for p in parameters)}")
        
    except Exception as e:
        print(f"执行过程中发生错误: {str(e)}")
    finally:
        # 确保关闭重建器
        if 'reconstructor' in locals():
            reconstructor.close()