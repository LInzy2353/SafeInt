import os
import numpy as np
import logging
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# Configure matplotlib with system available fonts
def setup_matplotlib_fonts():
    # 检查系统可用字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 优先使用这些字体，如果系统中有的话
    preferred_fonts = ["DejaVu Sans", "Liberation Sans", "FreeSans", "sans-serif"]
    
    # 过滤出系统中实际存在的字体
    available_preferred = [f for f in preferred_fonts if f in available_fonts]
    
    if available_preferred:
        plt.rcParams["font.family"] = available_preferred
    else:
        # 如果没有找到首选字体，使用系统默认sans-serif字体
        plt.rcParams["font.family"] = ["sans-serif"]
    
    plt.rcParams["axes.unicode_minus"] = False  # Fix minus sign display

# Set up fonts in advance
setup_matplotlib_fonts()

# 配置日志
def setup_logger():
    logging.basicConfig(
        filename=os.path.join(os.path.dirname(__file__), 'logs', 'logistic_regression.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('logistic_regression')

class LogisticRegressionClassifier:
    def __init__(self, embedding_dir, layers, model_dir=None):
        """初始化逻辑回归分类器
        
        Args:
            embedding_dir: 表征文件目录
            layers: 要处理的层列表
            model_dir: 模型保存目录
        """
        self.embedding_dir = embedding_dir
        self.layers = layers
        self.model_dir = model_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'logistic_regression')
        self.logger = setup_logger()
        self.models = {}
        self.scalers = {}
        self.results = {}
        
        # 创建模型保存目录
        os.makedirs(self.model_dir, exist_ok=True)
        
    def load_embeddings(self, layer, split):
        """加载指定层和数据集分割的表征
        
        Args:
            layer: 层号
            split: 数据集分割('train'或'test')
            
        Returns:
            embeddings: 表征数据
            labels: 标签数据
        """
        try:
            # 根据数据集类型构建不同的文件路径
            if split == 'train':
                # 尝试加载训练集数据
                # 首先尝试直接加载合并的训练数据
                embedding_path = os.path.join(self.embedding_dir, f'layer_{layer}_{split}.npy')
                if os.path.exists(embedding_path):
                    embeddings = np.load(embedding_path)
                else:
                    # 如果合并文件不存在，尝试加载各类型数据并合并
                    safe_path = os.path.join(self.embedding_dir, f'safe_layer_{layer}.npy')
                    unsafe_path = os.path.join(self.embedding_dir, f'unsafe_layer_{layer}.npy')
                    jailbreak_path = os.path.join(self.embedding_dir, f'jailbreak_layer_{layer}.npy')
                    
                    if os.path.exists(safe_path) and os.path.exists(unsafe_path) and os.path.exists(jailbreak_path):
                        safe_emb = np.load(safe_path)
                        unsafe_emb = np.load(unsafe_path)
                        jailbreak_emb = np.load(jailbreak_path)
                        embeddings = np.vstack([safe_emb, unsafe_emb, jailbreak_emb])
                    else:
                        raise FileNotFoundError(f"无法找到层{layer}的训练集表征文件")
            else:
                # 尝试加载测试集数据
                # 首先尝试直接加载合并的测试数据
                embedding_path = os.path.join(self.embedding_dir, f'layer_{layer}_{split}.npy')
                if os.path.exists(embedding_path):
                    embeddings = np.load(embedding_path)
                else:
                    # 如果测试集合并文件不存在，尝试从测试目录加载
                    test_dir = os.path.join(os.path.dirname(self.embedding_dir), 'test')
                    if os.path.exists(test_dir):
                        test_path = os.path.join(test_dir, f'layer_{layer}_{split}.npy')
                        if os.path.exists(test_path):
                            embeddings = np.load(test_path)
                        else:
                            # 尝试加载各类型测试数据并合并
                            safe_path = os.path.join(test_dir, f'safe_layer_{layer}.npy')
                            unsafe_path = os.path.join(test_dir, f'unsafe_layer_{layer}.npy')
                            jailbreak_path = os.path.join(test_dir, f'jailbreak_layer_{layer}.npy')
                            
                            if os.path.exists(safe_path) and os.path.exists(unsafe_path) and os.path.exists(jailbreak_path):
                                safe_emb = np.load(safe_path)
                                unsafe_emb = np.load(unsafe_path)
                                jailbreak_emb = np.load(jailbreak_path)
                                embeddings = np.vstack([safe_emb, unsafe_emb, jailbreak_emb])
                            else:
                                raise FileNotFoundError(f"无法找到层{layer}的测试集表征文件")
                    else:
                        raise FileNotFoundError(f"测试集目录不存在: {test_dir}")
            
            # 加载对应的标签文件
            labels = self.load_labels(split)
            
            self.logger.info(f"成功加载层{layer}的{split}集表征，形状: {embeddings.shape}，标签数: {len(labels)}")
            return embeddings, labels
        except Exception as e:
            self.logger.error(f"加载层{layer}的{split}集表征失败: {str(e)}")
            raise
            
    def load_labels(self, split):
        """加载指定数据集分割的标签
        
        Args:
            split: 数据集分割('train'或'test')
            
        Returns:
            labels: 标签数组
        """
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
        if split == 'train':
            labels_path = os.path.join(data_dir, 'train', 'train_data.json')
        else:
            labels_path = os.path.join(data_dir, 'test', 'single_method_test.json')
        
        try:
            with open(labels_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                labels = np.array([item['label'] for item in data])
                return labels
        except Exception as e:
            self.logger.error(f"加载{split}集标签失败: {str(e)}")
            raise
    
    def train(self):
        """逐层训练逻辑回归模型"""
        self.logger.info("开始训练逻辑回归分类器")
        
        for layer in self.layers:
            try:
                # 加载训练集表征
                X_train, y_train = self.load_embeddings(layer, 'train')
                
                # 数据标准化
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                self.scalers[layer] = scaler
                
                # 训练逻辑回归模型
                model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=42)
                model.fit(X_train_scaled, y_train)
                self.models[layer] = model
                
                # 加载测试集
                X_test, y_test = self.load_embeddings(layer, 'test')
                X_test_scaled = scaler.transform(X_test)
                
                # 预测并评估
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                # 保存结果
                self.results[layer] = {
                    'accuracy': accuracy,
                    'y_true': y_test,
                    'y_pred': y_pred,
                    'status': 'trained'
                }
                
                # 记录结果
                self.logger.info(f"层{layer} - 准确率: {accuracy:.4f}")
                
                # 保存分类报告
                report = classification_report(y_test, y_pred)
                self.logger.info(f"层{layer}分类报告:\n{report}")
                
                # 可视化混淆矩阵
                self.plot_confusion_matrix(layer, y_test, y_pred)
                
            except Exception as e:
                self.logger.error(f"处理层{layer}时出错: {str(e)}")
                # 创建一个空的结果条目，避免后续查找时出错
                self.models[layer] = None
                self.scalers[layer] = None
                self.results[layer] = {
                    'accuracy': 0.0,
                    'y_true': None,
                    'y_pred': None,
                    'status': 'error'
                }
        
        # 保存模型和结果
        self.save_results()
        self.logger.info("逻辑回归分类器训练完成")
        
        # 输出每层的准确率比较
        self.print_accuracy_comparison()
        
    def plot_confusion_matrix(self, layer, y_true, y_pred):
        """Plot confusion matrix"""
        # Ensure font settings
        setup_matplotlib_fonts()
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['safe', 'unsafe', 'jailbreak'], yticklabels=['safe', 'unsafe', 'jailbreak'])
        plt.title(f'Layer {layer} - Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # 保存混淆矩阵图
        figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        plt.savefig(os.path.join(figures_dir, f'confusion_matrix_layer_{layer}.png'))
        plt.close()
        
    def save_results(self):
        """保存训练结果"""
        # 保存准确率结果
        results_path = os.path.join(self.model_dir, 'accuracy_results.npy')
        accuracy_dict = {layer: result['accuracy'] for layer, result in self.results.items() if 'accuracy' in result}
        
        # 确保字典不为空
        if not accuracy_dict:
            self.logger.warning("准确率字典为空，将添加默认值")
            for layer in self.layers:
                if layer in self.results:
                    accuracy_dict[layer] = self.results[layer].get('accuracy', 0.5)  # 使用默认值0.5
                else:
                    accuracy_dict[layer] = 0.5  # 使用默认值0.5
        
        np.save(results_path, accuracy_dict)
        self.logger.info(f"准确率结果已保存到: {results_path}")
        
        # 绘制准确率随层变化的图表
        self.plot_accuracy_by_layer(accuracy_dict)
        
    def plot_accuracy_by_layer(self, accuracy_dict):
        """Plot accuracy by layer"""
        # Ensure font settings
        setup_matplotlib_fonts()
        
        layers = list(accuracy_dict.keys())
        accuracies = list(accuracy_dict.values())
        
        plt.figure(figsize=(10, 6))
        plt.plot(layers, accuracies, marker='o', linestyle='-', color='blue')
        plt.title('Logistic Regression Classifier Accuracy by Layer')
        plt.xlabel('Layer Number')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.xticks(layers)
        plt.ylim(0, 1.0)
        
        # 保存图表
        figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
        plt.savefig(os.path.join(figures_dir, 'accuracy_by_layer.png'))
        plt.close()
        
    def predict_proba(self, embeddings, layer):
        """预测给定层的分类概率
        
        Args:
            embeddings: 输入的表征数据 (numpy array 或 torch tensor)
            layer: 层号
            
        Returns:
            probs: 分类概率数组
        """
        try:
            # 检查层是否存在对应的模型
            if layer not in self.models or self.models[layer] is None:
                self.logger.warning(f"层{layer}的模型不存在，返回默认概率")
                # 返回默认概率分布 [safe, unsafe, jailbreak]
                if hasattr(embeddings, 'shape'):
                    batch_size = embeddings.shape[0] if len(embeddings.shape) > 1 else 1
                else:
                    batch_size = 1
                return np.array([[0.33, 0.33, 0.34]] * batch_size)
            
            # 处理输入数据类型
            if hasattr(embeddings, 'cpu'):  # PyTorch tensor
                embeddings_np = embeddings.cpu().numpy()
            else:  # numpy array
                embeddings_np = embeddings
            
            # 确保是2D数组
            if len(embeddings_np.shape) == 1:
                embeddings_np = embeddings_np.reshape(1, -1)
            
            # 获取对应层的模型和标准化器
            model = self.models[layer]
            scaler = self.scalers[layer]
            
            # 标准化输入数据
            embeddings_scaled = scaler.transform(embeddings_np)
            
            # 预测概率
            probs = model.predict_proba(embeddings_scaled)
            
            return probs
            
        except Exception as e:
            self.logger.error(f"预测层{layer}概率时出错: {str(e)}")
            # 返回默认概率分布
            if hasattr(embeddings, 'shape'):
                batch_size = embeddings.shape[0] if len(embeddings.shape) > 1 else 1
            else:
                batch_size = 1
            return np.array([[0.33, 0.33, 0.34]] * batch_size)

    def print_accuracy_comparison(self):
        """Print accuracy comparison across layers"""
        print("\n=== Logistic Regression Classifier Accuracy Comparison ===")
        for layer in sorted(self.layers):
            if layer in self.results:
                print(f"Layer {layer}: {self.results[layer]['accuracy']:.4f}")
        
        # Find the best performing layer
        if self.results:
            best_layer = max(self.results, key=lambda x: self.results[x]['accuracy'])
            best_accuracy = self.results[best_layer]['accuracy']
            print(f"\nBest Performing Layer: Layer {best_layer}, Accuracy: {best_accuracy:.4f}")
            
    def save_model(self, model_path):
        """保存训练好的模型和标准化器
        
        Args:
            model_path: 模型保存路径
        """
        import pickle
        import os
        
        # 确保目录存在
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # 保存模型和标准化器
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'results': self.results
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        self.logger.info(f"模型已保存到: {model_path}")

import argparse

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Logistic Regression Classifier for LLM Safety')
    parser.add_argument('--model', type=str, default='logistic_regression',
                        help='Specify which model to run (default: logistic_regression)')
    parser.add_argument('--layers', type=str, default='10-25',
                        help='Specify layers range to process (e.g., "10-25" or "10,15,20")')
    args = parser.parse_args()
    
    print(f"Running model: {args.model}")
    
    # 配置参数
    embedding_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'embeddings')
    
    # 解析层参数
    if '-' in args.layers:
        start, end = map(int, args.layers.split('-'))
        layers = list(range(start, end + 1))
    elif ',' in args.layers:
        layers = list(map(int, args.layers.split(',')))
    else:
        layers = [int(args.layers)]
    
    print(f"Processing layers: {layers}")
    
    # 创建并训练分类器
    classifier = LogisticRegressionClassifier(embedding_dir, layers)
    classifier.train()