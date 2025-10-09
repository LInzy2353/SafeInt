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

# Configure matplotlib with English fonts
def setup_matplotlib_fonts():
    # Use common English fonts
    plt.rcParams["font.family"] = ["Arial", "Helvetica", "Times New Roman", "sans-serif"]
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
        # 加载表征文件
        embedding_path = os.path.join(self.embedding_dir, f'layer_{layer}_{split}.npy')
        
        try:
            # 直接加载numpy数组
            embeddings = np.load(embedding_path)
            
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
                    'y_pred': y_pred
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
        accuracy_dict = {layer: result['accuracy'] for layer, result in self.results.items()}
        np.save(results_path, accuracy_dict)
        
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