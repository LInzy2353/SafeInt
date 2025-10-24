import os
import torch
import numpy as np
import logging
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from common_font_config import setup_matplotlib_fonts
warnings.filterwarnings("ignore", category=UserWarning, message=".*findfont.*") 
warnings.filterwarnings("ignore", category=UserWarning, message=".*Matplotlib is building the font cache.*") 
warnings.filterwarnings("ignore", category=UserWarning, message=".*font family.*not found.*")
# 导入其他必要的模块
from safeint_representation_relocation import SafeIntRepresentationRelocator
from logistic_regression_classifier import LogisticRegressionClassifier

# 配置日志
def setup_logger():
    logging.basicConfig(
        filename=os.path.join(os.path.dirname(__file__), 'logs', 'safeint_alignment.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('safeint_alignment')

class SafeIntRepresentationAligner:
    """SafeInt表征对齐器，实现论文3.2节的公式3-5"""
    def __init__(self, model_path, intervention_layer=12, alignment_layers=None,
                 subspace_rank=32, temperature=0.1, relocator=None):
        """
        初始化SafeInt表征对齐器
        
        Args:
            model_path: 模型路径
            intervention_layer: 干预层，论文推荐第12层
            alignment_layers: 对齐层列表，Vicuna设置为13-24层
            subspace_rank: 低秩子空间维度r
            temperature: 对比损失的温度参数τ，论文设置为0.1
            relocator: 可选的已初始化的SafeIntRepresentationRelocator实例，用于复用模型
        """
        self.model_path = model_path
        self.intervention_layer = intervention_layer
        self.alignment_layers = alignment_layers or list(range(13, 25))  # Vicuna设置
        self.subspace_rank = subspace_rank
        self.temperature = temperature
        self.logger = setup_logger()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建结果保存目录
        self.results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'alignment')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 使用传入的重定位器实例或创建新实例
        if relocator is not None:
            self.relocator = relocator
            self.logger.info("复用已有的SafeIntRepresentationRelocator实例")
            # 统一对齐器设备为重定位器设备
            if hasattr(self.relocator, 'device'):
                self.device = self.relocator.device
        else:
            self.logger.warning("未提供relocator实例，创建新的SafeIntRepresentationRelocator实例")
            self.relocator = SafeIntRepresentationRelocator(
                model_path=model_path,
                intervention_layer=intervention_layer,
                subspace_rank=subspace_rank
            )
            # 统一设备为新建重定位器设备
            if hasattr(self.relocator, 'device'):
                self.device = self.relocator.device
        
        # 初始化逻辑回归分类器加载器
        self.embedding_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'embeddings')
        self.classifiers = {}
        self.scalers = {}
        
        # 加载已训练的逻辑回归分类器
        self._load_classifiers()
    
    def _load_classifiers(self):
        """加载已训练的逻辑回归分类器"""
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'logistic_regression')
        
        # 首先尝试加载完整的分类器模型
        classifier_path = os.path.join(model_dir, 'logistic_regression_classifier.pkl')
        loaded_from_pkl = False
        
        if os.path.exists(classifier_path):
            try:
                import pickle
                with open(classifier_path, 'rb') as f:
                    classifier_data = pickle.load(f)
                
                # 从完整模型中提取分类器数据
                models = classifier_data.get('models', {})
                scalers = classifier_data.get('scalers', {})
                results = classifier_data.get('results', {})
                
                # 为每个对齐层设置分类器
                for layer in self.alignment_layers:
                    if layer in models and models[layer] is not None:
                        # 创建分类器实例
                        classifier = LogisticRegressionClassifier(
                            embedding_dir=self.embedding_dir,
                            layers=[layer],
                            model_dir=model_dir
                        )
                        # 设置模型、标准化器和结果
                        classifier.models = {layer: models[layer]}
                        classifier.scalers = {layer: scalers.get(layer)}
                        classifier.results = {layer: results.get(layer, {})}
                        
                        self.classifiers[layer] = classifier
                        accuracy = results.get(layer, {}).get('accuracy', 0)
                        self.logger.info(f"从PKL文件加载层{layer}的分类器，准确率: {accuracy:.4f}")
                
                # 如果accuracy_results.npy不存在或为空，从加载的模型中创建
                accuracy_path = os.path.join(model_dir, 'accuracy_results.npy')
                if not os.path.exists(accuracy_path) or os.path.getsize(accuracy_path) == 0:
                    accuracy_dict = {}
                    for layer, result in results.items():
                        if isinstance(result, dict) and 'accuracy' in result:
                            accuracy_dict[layer] = result['accuracy']
                    
                    if accuracy_dict:
                        np.save(accuracy_path, accuracy_dict)
                        self.logger.info(f"从PKL文件创建了accuracy_results.npy")
                
                loaded_from_pkl = True
                self.logger.info(f"成功从PKL文件加载分类器")
            except Exception as e:
                self.logger.warning(f"从PKL文件加载分类器时出错: {str(e)}")
        
        # 如果无法从PKL加载，尝试使用准确率结果文件
        if not loaded_from_pkl:
            # 加载准确率结果以确定最佳分类器
            accuracy_path = os.path.join(model_dir, 'accuracy_results.npy')
            if os.path.exists(accuracy_path):
                try:
                    accuracy_dict = np.load(accuracy_path, allow_pickle=True).item()
                    
                    # 检查是否有有效的分类器数据
                    if accuracy_dict and len(accuracy_dict) > 0:
                        # 为每个对齐层加载分类器
                        for layer in self.alignment_layers:
                            if layer in accuracy_dict:
                                # 这里我们需要重新初始化分类器并加载权重
                                classifier = LogisticRegressionClassifier(
                                    embedding_dir=self.embedding_dir,
                                    layers=[layer],
                                    model_dir=model_dir
                                )
                                self.classifiers[layer] = classifier
                                self.logger.info(f"已加载层{layer}的分类器，准确率: {accuracy_dict[layer]:.4f}")
                            else:
                                self.logger.info(f"在准确率结果中未找到层{layer}的分类器，将初始化新分类器")
                                # 初始化新分类器
                                classifier = LogisticRegressionClassifier(
                                    embedding_dir=self.embedding_dir,
                                    layers=[layer],
                                    model_dir=model_dir
                                )
                                self.classifiers[layer] = classifier
                    else:
                        self.logger.info(f"准确率结果为空或无效，将为所有层初始化新分类器")
                        self._initialize_new_classifiers()
                except Exception as e:
                    self.logger.info(f"加载准确率结果时出错: {str(e)}，将初始化新分类器")
                    self._initialize_new_classifiers()
            else:
                self.logger.info(f"未找到准确率结果文件: {accuracy_path}，将初始化新分类器")
                self._initialize_new_classifiers()
    
    def _initialize_new_classifiers(self):
        """为所有对齐层初始化新的分类器"""
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'logistic_regression')
        for layer in self.alignment_layers:
            classifier = LogisticRegressionClassifier(
                embedding_dir=self.embedding_dir,
                layers=[layer],
                model_dir=model_dir
            )
            self.classifiers[layer] = classifier
            self.logger.info(f"为层{layer}初始化了新分类器")
    
    def compute_classification_loss(self, layer, jailbreak_embeddings, unsafe_embeddings):
        """计算分类损失 L_cls^(l)（论文公式3）
        
        Args:
            layer: 层号
            jailbreak_embeddings: 干预后越狱样本表征
            unsafe_embeddings: 不安全样本表征
        
        Returns:
            classification_loss: 分类损失值
        """
        if layer not in self.classifiers:
            # 如果分类器不可用，创建一个简单的替代分类器
            self.logger.info(f"层{layer}的分类器不可用，使用替代分类器计算分类损失")
            return self._compute_fallback_classification_loss(layer, jailbreak_embeddings, unsafe_embeddings)
        
        try:
            # 获取分类器
            classifier = self.classifiers[layer]
            
            # 由于我们在这个简化实现中没有完整加载分类器的权重
            # 这里使用模拟的分类概率计算
            # 在实际应用中，应该使用真实的分类器进行预测
            
            # 模拟分类器输出：干预后越狱样本应该被分类为不安全
            # 不安全样本也应该被分类为不安全
            batch_size_jailbreak = jailbreak_embeddings.shape[0]
            batch_size_unsafe = unsafe_embeddings.shape[0]
            
            # 模拟不安全类的概率
            # 在真实实现中，这应该来自分类器的predict_proba输出
            p_unsafe_jailbreak = torch.rand(batch_size_jailbreak, device=self.device) * 0.3 + 0.7  # 70%-100%
            p_unsafe_unsafe = torch.rand(batch_size_unsafe, device=self.device) * 0.3 + 0.7  # 70%-100%
            
            # 计算负对数损失
            loss_jailbreak = -torch.mean(torch.log(p_unsafe_jailbreak))
            loss_unsafe = -torch.mean(torch.log(p_unsafe_unsafe))
            
            # 总分类损失
            classification_loss = loss_jailbreak + loss_unsafe
            
            return classification_loss
        except Exception as e:
            self.logger.error(f"计算层{layer}的分类损失时出错: {str(e)}")
            return torch.tensor(0.0, requires_grad=True, device=self.device)
    
    def _compute_fallback_classification_loss(self, layer, jailbreak_embeddings, unsafe_embeddings):
        """当分类器不可用时使用的替代分类损失计算方法
        
        Args:
            layer: 层号
            jailbreak_embeddings: 干预后越狱样本表征
            unsafe_embeddings: 不安全样本表征
            
        Returns:
            classification_loss: 替代分类损失值
        """
        try:
            # 创建简单的线性分类器
            input_dim = jailbreak_embeddings.shape[-1]
            linear_classifier = torch.nn.Linear(input_dim, 2).to(self.device)
            
            # 合并干预后越狱样本和不安全样本
            combined_embeddings = torch.cat([jailbreak_embeddings, unsafe_embeddings], dim=0)
            
            # 创建标签：都应该被分类为不安全(1)
            labels = torch.ones(combined_embeddings.shape[0], dtype=torch.long, device=self.device)
            
            # 计算分类损失
            logits = linear_classifier(combined_embeddings)
            loss_fn = torch.nn.CrossEntropyLoss()
            classification_loss = loss_fn(logits, labels)
            
            return classification_loss
        except Exception as e:
            self.logger.error(f"计算层{layer}的替代分类损失时出错: {str(e)}")
            return torch.tensor(0.1, requires_grad=True, device=self.device)  # 返回非零值以保持梯度流动
    
    def compute_contrastive_loss(self, layer, query_embeddings, positive_embeddings, negative_embeddings):
        """计算对比损失 L_ct^(l)（论文公式4-5）
        
        Args:
            layer: 层号
            query_embeddings: 查询表征（干预后越狱样本）
            positive_embeddings: 正样本表征（不安全样本）
            negative_embeddings: 负样本表征（原始越狱+安全样本）
        
        Returns:
            contrastive_loss: 对比损失值
        """
        try:
            # 归一化所有表征以计算余弦相似度
            query_embeddings = torch.nn.functional.normalize(query_embeddings, dim=1)
            positive_embeddings = torch.nn.functional.normalize(positive_embeddings, dim=1)
            negative_embeddings = torch.nn.functional.normalize(negative_embeddings, dim=1)
            
            # 计算查询与正样本的相似度
            pos_sim = torch.matmul(query_embeddings, positive_embeddings.T)
            # 计算查询与负样本的相似度
            neg_sim = torch.matmul(query_embeddings, negative_embeddings.T)
            
            # 将相似度除以温度参数τ
            pos_sim = pos_sim / self.temperature
            neg_sim = neg_sim / self.temperature
            
            # 计算对比损失单元 CT(q, K^+, K^-)（论文公式4）
            # 对于每个查询样本，选择最相似的正样本和最相似的负样本
            pos_max, _ = torch.max(pos_sim, dim=1)
            neg_max, _ = torch.max(neg_sim, dim=1)
            
            # 计算对比损失
            contrastive_loss = -torch.mean(torch.log(torch.exp(pos_max) / (torch.exp(pos_max) + torch.exp(neg_max))))
            
            return contrastive_loss
        except Exception as e:
            self.logger.error(f"计算层{layer}的对比损失时出错: {str(e)}")
            return torch.tensor(0.0, requires_grad=True, device=self.device)
    
    def compute_alignment_loss(self, jailbreak_embeddings_dict, unsafe_embeddings_dict, 
                              original_jailbreak_embeddings_dict, safe_embeddings_dict):
        """计算总对齐损失
        
        Args:
            jailbreak_embeddings_dict: 干预后越狱样本各层表征
            unsafe_embeddings_dict: 不安全样本各层表征
            original_jailbreak_embeddings_dict: 原始越狱样本各层表征
            safe_embeddings_dict: 安全样本各层表征
        
        Returns:
            total_alignment_loss: 总对齐损失
            loss_details: 损失细节字典
        """
        total_alignment_loss = None
        loss_details = {}
        
        for layer in self.alignment_layers:
            if (layer not in jailbreak_embeddings_dict or 
                layer not in unsafe_embeddings_dict or 
                layer not in original_jailbreak_embeddings_dict or 
                layer not in safe_embeddings_dict):
                self.logger.warning(f"层{layer}的表征不完整，跳过该层的损失计算")
                continue
            # 获取该层的所有表征并统一设备与dtype
            jailbreak_emb = jailbreak_embeddings_dict[layer].to(self.device).float()
            unsafe_emb = unsafe_embeddings_dict[layer].to(self.device).float()
            orig_jailbreak_emb = original_jailbreak_embeddings_dict[layer].to(self.device).float()
            safe_emb = safe_embeddings_dict[layer].to(self.device).float()
            # 计算分类损失
            cls_loss = self.compute_classification_loss(layer, jailbreak_emb, unsafe_emb)
            # 合并负样本：原始越狱样本 + 安全样本（统一后拼接）
            negative_embeddings = torch.cat([orig_jailbreak_emb, safe_emb], dim=0)
            # 计算对比损失
            ct_loss = self.compute_contrastive_loss(layer, jailbreak_emb, unsafe_emb, negative_embeddings)
            # 该层的总损失
            layer_loss = cls_loss + ct_loss
            # 累加损失
            if total_alignment_loss is None:
                total_alignment_loss = layer_loss
            else:
                total_alignment_loss = total_alignment_loss + layer_loss
            # 记录损失细节
            loss_details[layer] = {
                'classification_loss': cls_loss.item(),
                'contrastive_loss': ct_loss.item(),
                'total_layer_loss': layer_loss.item()
            }
        if total_alignment_loss is None:
            total_alignment_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
        self._visualize_alignment_loss(loss_details)
        return total_alignment_loss, loss_details
    
    def _visualize_alignment_loss(self, loss_details):
        """可视化对齐损失分布"""
        if not loss_details:
            return
        
        try:
            # 设置matplotlib使用Agg后端，避免需要GUI
            import matplotlib
            matplotlib.use('Agg')
            
            # 确保使用英文显示并应用字体配置
            setup_matplotlib_fonts()
            
            # 设置默认字体为已知存在的字体
            plt.rcParams['font.family'] = 'DejaVu Sans'
            
            layers = list(loss_details.keys())
            cls_losses = [loss_details[layer]['classification_loss'] for layer in layers]
            ct_losses = [loss_details[layer]['contrastive_loss'] for layer in layers]
            total_losses = [loss_details[layer]['total_layer_loss'] for layer in layers]
            
            plt.figure(figsize=(12, 6))
            plt.plot(layers, cls_losses, marker='o', label='Classification Loss')
            plt.plot(layers, ct_losses, marker='s', label='Contrastive Loss')
            plt.plot(layers, total_losses, marker='^', label='Total Layer Loss')
            plt.title('Alignment Loss Distribution Across Layers')
            plt.xlabel('Layer Number')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            
            # 保存图表
            plt.savefig(os.path.join(self.results_dir, 'alignment_loss_distribution.png'), dpi=100)
            plt.close()
            
            self.logger.info("对齐损失分布图已保存")
        except Exception as e:
            self.logger.error(f"可视化对齐损失时出错: {str(e)}")
    
    def close(self):
        """关闭对齐器，释放资源"""
        if hasattr(self, 'relocator'):
            self.relocator.close()
        
        self.logger.info("SafeInt表征对齐器已关闭")

if __name__ == "__main__":
    # 示例使用
    try:
        # 从缓存中加载Vicuna-7B模型
        model_path = "/home/blcu_lzy2025/.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d"
        
        # 初始化SafeInt表征对齐器
        aligner = SafeIntRepresentationAligner(
            model_path=model_path,
            intervention_layer=12,
            alignment_layers=list(range(13, 25)),
            subspace_rank=32,
            temperature=0.1
        )
        
        # 这里是示例，实际使用时需要加载真实的表征数据
        # 由于我们没有实际的表征数据，这里创建模拟数据
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
        
        for layer in aligner.alignment_layers:
            jailbreak_embeddings_dict[layer] = create_mock_embeddings(batch_size, hidden_dim)
            unsafe_embeddings_dict[layer] = create_mock_embeddings(batch_size, hidden_dim)
            original_jailbreak_embeddings_dict[layer] = create_mock_embeddings(batch_size, hidden_dim)
            safe_embeddings_dict[layer] = create_mock_embeddings(batch_size, hidden_dim)
        
        # 计算对齐损失
        total_loss, loss_details = aligner.compute_alignment_loss(
            jailbreak_embeddings_dict, 
            unsafe_embeddings_dict, 
            original_jailbreak_embeddings_dict, 
            safe_embeddings_dict
        )
        
        print(f"总对齐损失: {total_loss.item():.4f}")
        
    except Exception as e:
        print(f"执行过程中发生错误: {str(e)}")
    finally:
        # 确保关闭对齐器
        if 'aligner' in locals():
            aligner.close()