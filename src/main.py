import sys
import os
import torch
import numpy as np
import logging
import json
import argparse
import time

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入SafeInt模块
from src.safeint_representation_relocation import SafeIntRepresentationRelocator
from src.safeint_representation_alignment import SafeIntRepresentationAligner
from src.safeint_representation_reconstruction import SafeIntRepresentationReconstructor
from src.safeint_training_integration import SafeIntTrainer, SafeIntInference
from src.safeint_evaluation import SafeIntEvaluator
from src.logistic_regression_classifier import LogisticRegressionClassifier
from src.extract_llm_representations import LLMRepresentationExtractor

# 配置日志
def setup_logger(log_file=None):
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    log_level = logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    
    return logging.getLogger('safeint_main')

class SafeIntPipeline:
    """SafeInt完整工作流程管道"""
    def __init__(self, config):
        """
        初始化SafeInt管道
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = setup_logger(config.get('log_file'))
        
        # 创建必要的目录
        self._create_directories()
        
        # 初始化各个模块
        self.extractor = None
        self.classifier = None
        self.relocator = None
        self.aligner = None
        self.reconstructor = None
        self.trainer = None
        self.inference = None
        self.evaluator = None
    
    def _create_directories(self):
        """创建必要的目录"""
        directories = [
            self.config.get('logs_dir'),
            self.config.get('models_dir'),
            self.config.get('embeddings_dir'),
            self.config.get('results_dir')
        ]
        
        for dir_path in directories:
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                self.logger.info(f"创建目录: {dir_path}")
    
    def step_1_extract_embeddings(self):
        """步骤1: 从LLM中提取中间层表征"""
        self.logger.info("===== 步骤1: 提取LLM表征 =====")
        
        try:
            # 初始化表征提取器
            self.extractor = LLMRepresentationExtractor(
                model_name=self.config.get('model_name', 'vicuna-7b'),
                model_path=self.config.get('model_path'),
                layers=self.config.get('layers', list(range(10, 26)))
            )
            
            # 提取训练数据的表征
            train_data_path = self.config.get('train_data_path')
            if train_data_path and os.path.exists(train_data_path):
                self.logger.info(f"从 {train_data_path} 提取训练数据表征")
                
                # 提取各种类型数据的表征
                data_types = ['jailbreak', 'unsafe', 'safe']
                for data_type in data_types:
                    data_file = os.path.join(train_data_path, f'train_{data_type}.json')
                    if os.path.exists(data_file):
                        self.extractor.extract_and_save_embeddings(
                            data_file=data_file,
                            output_dir=os.path.join(self.config.get('embeddings_dir'), 'train'),
                            data_type=data_type
                        )
            
            # 提取测试数据的表征
            test_data_path = self.config.get('test_data_path')
            if test_data_path and os.path.exists(test_data_path):
                self.logger.info(f"从 {test_data_path} 提取测试数据表征")
                
                # 提取各种类型数据的表征
                data_types = ['jailbreak', 'unsafe', 'safe']
                for data_type in data_types:
                    data_file = os.path.join(test_data_path, f'test_{data_type}.json')
                    if os.path.exists(data_file):
                        self.extractor.extract_and_save_embeddings(
                            data_file=data_file,
                            output_dir=os.path.join(self.config.get('embeddings_dir'), 'test'),
                            data_type=data_type
                        )
            
            self.logger.info("表征提取完成")
            return True
        except Exception as e:
            self.logger.error(f"提取表征时出错: {str(e)}")
            return False
    
    def step_2_train_classifier(self):
        """步骤2: 训练分层逻辑回归分类器"""
        self.logger.info("===== 步骤2: 训练分层逻辑回归分类器 =====")
        
        try:
            # 初始化分类器
            self.classifier = LogisticRegressionClassifier(
                model_name=self.config.get('model_name', 'vicuna-7b')
            )
            
            # 加载训练数据的表征
            embeddings_dir = os.path.join(self.config.get('embeddings_dir'), 'train')
            if os.path.exists(embeddings_dir):
                # 加载不同类型数据的表征
                jailbreak_embeddings = {}
                unsafe_embeddings = {}
                safe_embeddings = {}
                
                for layer in self.config.get('layers', list(range(10, 26))):
                    # 越狱样本表征
                    jailbreak_file = os.path.join(embeddings_dir, f'jailbreak_layer_{layer}.npy')
                    if os.path.exists(jailbreak_file):
                        jailbreak_embeddings[layer] = np.load(jailbreak_file)
                    
                    # 不安全样本表征
                    unsafe_file = os.path.join(embeddings_dir, f'unsafe_layer_{layer}.npy')
                    if os.path.exists(unsafe_file):
                        unsafe_embeddings[layer] = np.load(unsafe_file)
                    
                    # 安全样本表征
                    safe_file = os.path.join(embeddings_dir, f'safe_layer_{layer}.npy')
                    if os.path.exists(safe_file):
                        safe_embeddings[layer] = np.load(safe_file)
                
                # 训练分类器
                self.classifier.train(
                    jailbreak_embeddings=jailbreak_embeddings,
                    unsafe_embeddings=unsafe_embeddings,
                    safe_embeddings=safe_embeddings
                )
                
                # 保存分类器
                classifier_path = os.path.join(self.config.get('models_dir'), 'logistic_regression_classifier.pkl')
                self.classifier.save_model(classifier_path)
                self.logger.info(f"分类器已保存到: {classifier_path}")
            
            self.logger.info("分类器训练完成")
            return True
        except Exception as e:
            self.logger.error(f"训练分类器时出错: {str(e)}")
            return False
    
    def step_3_train_safeint(self):
        """步骤3: 训练SafeInt模型"""
        self.logger.info("===== 步骤3: 训练SafeInt模型 =====")
        
        try:
            # 初始化训练器
            self.trainer = SafeIntTrainer(
                model_path=self.config.get('model_path'),
                intervention_layer=self.config.get('intervention_layer', 12),
                alignment_layers=self.config.get('alignment_layers', list(range(13, 25))),
                subspace_rank=self.config.get('subspace_rank', 32),
                alpha=self.config.get('alpha', 1.0),
                beta=self.config.get('beta', 0.1),
                temperature=self.config.get('temperature', 0.1)
            )
            
            # 加载训练数据集
            train_data_path = self.config.get('train_data_path')
            if train_data_path and os.path.exists(train_data_path):
                file_names = {
                    'jailbreak': 'train_jailbreak.json',
                    'unsafe': 'train_unsafe.json',
                    'safe': 'train_safe.json'
                }
                
                datasets = self.trainer.load_dataset(train_data_path, file_names)
                
                if datasets and all(key in datasets for key in ['jailbreak', 'unsafe', 'safe']):
                    # 开始训练
                    self.trainer.train(
                        datasets=datasets,
                        epochs=self.config.get('epochs', 15),
                        batch_size=self.config.get('batch_size', 32)
                    )
                else:
                    self.logger.error("未能加载完整的训练数据集")
                    return False
            
            self.logger.info("SafeInt模型训练完成")
            return True
        except Exception as e:
            self.logger.error(f"训练SafeInt模型时出错: {str(e)}")
            return False
    
    def step_4_inference(self, input_text, use_safeint=True):
        """步骤4: 使用SafeInt模型进行推理"""
        self.logger.info("===== 步骤4: SafeInt模型推理 =====")
        
        try:
            # 初始化推理器（如果尚未初始化）
            if self.inference is None:
                safeint_model_path = os.path.join(self.config.get('models_dir'), 'safeint_relocation_module.pth')
                self.inference = SafeIntInference(
                    model_path=self.config.get('model_path'),
                    safeint_model_path=safeint_model_path if os.path.exists(safeint_model_path) else None
                )
            
            # 生成响应
            response, risk_info = self.inference.generate_response(
                text=input_text,
                max_length=self.config.get('max_length', 256),
                use_safeint=use_safeint
            )
            
            self.logger.info(f"输入: {input_text}")
            self.logger.info(f"响应: {response}")
            self.logger.info(f"风险信息: {risk_info}")
            
            return response, risk_info
        except Exception as e:
            self.logger.error(f"推理过程中出错: {str(e)}")
            return "I'm sorry, but I can't help with that right now.", {}
    
    def step_5_evaluation(self):
        """步骤5: 评估SafeInt模型效果"""
        self.logger.info("===== 步骤5: 评估SafeInt模型效果 =====")
        
        try:
            # 初始化评估器
            safeint_model_path = os.path.join(self.config.get('models_dir'), 'safeint_relocation_module.pth')
            self.evaluator = SafeIntEvaluator(
                model_path=self.config.get('model_path'),
                safeint_model_path=safeint_model_path if os.path.exists(safeint_model_path) else None
            )
            
            # 加载测试数据集
            test_data_path = self.config.get('test_data_path')
            if not test_data_path or not os.path.exists(test_data_path):
                self.logger.warning("测试数据路径不存在，使用模拟数据进行评估")
                return False
            
            # 1. 防御效果评估
            self.logger.info("评估防御效果...")
            defense_datasets = {
                'advbench': self.evaluator.load_benchmark_dataset(test_data_path, 'test_advbench.json'),
                'jailbreakbench': self.evaluator.load_benchmark_dataset(test_data_path, 'test_jailbreakbench.json')
            }
            
            defense_results = self.evaluator.evaluate_defense_effectiveness(defense_datasets)
            self.logger.info(f"防御效果评估完成，总体ASR: {defense_results['overall'].get('asr_keyword', 0.0):.4f}")
            
            # 2. 效用保持评估
            self.logger.info("评估效用保持...")
            utility_dataset = self.evaluator.load_benchmark_dataset(test_data_path, 'test_utility.json')
            utility_results = self.evaluator.evaluate_utility(utility_dataset)
            self.logger.info(f"效用评估完成")
            
            # 3. 鲁棒性评估
            self.logger.info("评估鲁棒性...")
            robustness_dataset = self.evaluator.load_benchmark_dataset(test_data_path, 'test_adaptive_attacks.json')
            robustness_results = self.evaluator.evaluate_robustness(robustness_dataset)
            self.logger.info(f"鲁棒性评估完成，ASR: {robustness_results.get('asr', 0.0):.4f}")
            
            # 4. 生成综合评估报告
            report_path = self.evaluator.generate_comprehensive_report()
            if report_path:
                self.logger.info(f"综合评估报告已生成: {report_path}")
            
            self.logger.info("所有评估完成")
            return True
        except Exception as e:
            self.logger.error(f"评估过程中出错: {str(e)}")
            return False
    
    def run_full_pipeline(self):
        """运行完整的SafeInt工作流程"""
        self.logger.info("开始运行完整的SafeInt工作流程")
        start_time = time.time()
        
        # 运行各个步骤
        success = True
        
        # 步骤1: 提取表征
        if not self.step_1_extract_embeddings():
            success = False
        
        # 步骤2: 训练分类器
        if success and not self.step_2_train_classifier():
            success = False
        
        # 步骤3: 训练SafeInt模型
        if success and not self.step_3_train_safeint():
            success = False
        
        # 步骤5: 评估效果
        if success and not self.step_5_evaluation():
            success = False
        
        # 计算总耗时
        total_time = time.time() - start_time
        
        if success:
            self.logger.info(f"SafeInt工作流程已成功完成，总耗时: {total_time:.2f}秒")
        else:
            self.logger.error(f"SafeInt工作流程未能完全成功，总耗时: {total_time:.2f}秒")
        
        return success
    
    def close(self):
        """关闭管道，释放资源"""
        if hasattr(self, 'extractor') and self.extractor:
            self.extractor.close()
        
        if hasattr(self, 'inference') and self.inference:
            self.inference.close()
        
        if hasattr(self, 'evaluator') and self.evaluator:
            self.evaluator.close()
        
        self.logger.info("SafeInt管道已关闭")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='SafeInt: Shielding Large Language Models from Jailbreak Attacks')
    
    # 模型配置
    parser.add_argument('--model_name', type=str, default='vicuna-7b', help='模型名称')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    
    # 目录配置
    parser.add_argument('--train_data_path', type=str, default='data/train', help='训练数据路径')
    parser.add_argument('--test_data_path', type=str, default='data/test', help='测试数据路径')
    parser.add_argument('--embeddings_dir', type=str, default='embeddings', help='表征保存路径')
    parser.add_argument('--models_dir', type=str, default='models', help='模型保存路径')
    parser.add_argument('--results_dir', type=str, default='results', help='结果保存路径')
    parser.add_argument('--logs_dir', type=str, default='logs', help='日志保存路径')
    
    # 训练配置
    parser.add_argument('--epochs', type=int, default=15, help='训练轮次')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    
    # SafeInt参数配置
    parser.add_argument('--intervention_layer', type=int, default=12, help='干预层')
    parser.add_argument('--alignment_layers', type=str, default='13-24', help='对齐层范围，如"13-24"')
    parser.add_argument('--subspace_rank', type=int, default=32, help='低秩子空间维度r')
    parser.add_argument('--alpha', type=float, default=1.0, help='对齐损失权重')
    parser.add_argument('--beta', type=float, default=0.1, help='重建损失权重')
    parser.add_argument('--temperature', type=float, default=0.1, help='对比损失温度参数')
    parser.add_argument('--layers', type=str, default='10-25', help='处理的层范围，如"10-25"')
    
    # 运行模式
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'extract', 'classifier', 'train', 'inference', 'evaluate'], help='运行模式')
    parser.add_argument('--input_text', type=str, help='推理模式下的输入文本')
    parser.add_argument('--use_safeint', type=bool, default=True, help='推理模式下是否使用SafeInt')
    
    return parser.parse_args()

def create_config_from_args(args):
    """根据命令行参数创建配置字典"""
    # 解析层范围
    def parse_layers(layers_str):
        if '-' in layers_str:
            start, end = map(int, layers_str.split('-'))
            return list(range(start, end + 1))
        elif ',' in layers_str:
            return list(map(int, layers_str.split(',')))
        else:
            return [int(layers_str)]
    
    # 获取项目根目录
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    config = {
        'model_name': args.model_name,
        'model_path': args.model_path,
        'train_data_path': os.path.join(root_dir, args.train_data_path),
        'test_data_path': os.path.join(root_dir, args.test_data_path),
        'embeddings_dir': os.path.join(root_dir, args.embeddings_dir),
        'models_dir': os.path.join(root_dir, args.models_dir),
        'results_dir': os.path.join(root_dir, args.results_dir),
        'logs_dir': os.path.join(root_dir, args.logs_dir),
        'log_file': os.path.join(root_dir, args.logs_dir, 'safeint.log'),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'intervention_layer': args.intervention_layer,
        'alignment_layers': parse_layers(args.alignment_layers),
        'subspace_rank': args.subspace_rank,
        'alpha': args.alpha,
        'beta': args.beta,
        'temperature': args.temperature,
        'layers': parse_layers(args.layers),
        'max_length': 256
    }
    
    return config

def main():
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        # 创建配置
        config = create_config_from_args(args)
        
        # 初始化SafeInt管道
        pipeline = SafeIntPipeline(config)
        
        # 根据运行模式执行不同的操作
        if args.mode == 'full':
            # 运行完整流程
            pipeline.run_full_pipeline()
        elif args.mode == 'extract':
            # 仅提取表征
            pipeline.step_1_extract_embeddings()
        elif args.mode == 'classifier':
            # 仅训练分类器
            pipeline.step_1_extract_embeddings()
            pipeline.step_2_train_classifier()
        elif args.mode == 'train':
            # 仅训练SafeInt模型
            pipeline.step_3_train_safeint()
        elif args.mode == 'inference':
            # 仅进行推理
            if args.input_text:
                response, risk_info = pipeline.step_4_inference(args.input_text, args.use_safeint)
                print(f"\n输入: {args.input_text}")
                print(f"响应: {response}")
                print(f"风险信息: {risk_info}")
            else:
                print("请提供输入文本，使用 --input_text 参数")
        elif args.mode == 'evaluate':
            # 仅进行评估
            pipeline.step_5_evaluation()
        
    except Exception as e:
        print(f"执行过程中发生错误: {str(e)}")
        sys.exit(1)
    finally:
        # 确保关闭管道
        if 'pipeline' in locals():
            pipeline.close()

if __name__ == "__main__":
    main()