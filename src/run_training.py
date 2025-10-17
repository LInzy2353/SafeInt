import json
#!/usr/bin/env python3
import os
import sys
import argparse
import json
import matplotlib.pyplot as plt
from main import SafeIntPipeline

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='SafeInt 训练脚本 - 简化版')
    
    # 基本配置
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--train_data_path', type=str, default='data/train', help='训练数据路径')
    parser.add_argument('--test_data_path', type=str, default='data/test', help='测试数据路径')
    parser.add_argument('--output_dir', type=str, default='results/training', help='结果输出目录')
    
    # 训练配置
    parser.add_argument('--epochs', type=int, default=15, help='训练轮次')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    
    # SafeInt 参数
    parser.add_argument('--intervention_layer', type=int, default=12, help='干预层')
    parser.add_argument('--alignment_layers', type=str, default='13-24', help='对齐层范围，如"13-24"')
    parser.add_argument('--subspace_rank', type=int, default=32, help='低秩子空间维度r')
    parser.add_argument('--alpha', type=float, default=1.0, help='对齐损失权重')
    parser.add_argument('--beta', type=float, default=0.1, help='重建损失权重')
    parser.add_argument('--temperature', type=float, default=0.1, help='对比损失温度参数')
    
    # 表征干预增强参数
    parser.add_argument('--gain_factor', type=float, default=2.0, help='用于增强重定位模块的权重初始化增益因子')
    parser.add_argument('--intervention_strength', type=float, default=1.0, help='用于增强干预效果的强度因子')
    
    # 其他选项
    parser.add_argument('--disable_4bit', action='store_true', help='禁用4位量化')
    
    return parser.parse_args()


def create_config_from_args(args):
    """根据命令行参数创建配置字典"""
    # 获取项目根目录
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 解析层范围
    def parse_layers(layers_str):
        try:
            if '-' in layers_str:
                start, end = map(int, layers_str.split('-'))
                return list(range(start, end + 1))
            elif ',' in layers_str:
                return list(map(int, layers_str.split(',')))
            else:
                return [int(layers_str)]
        except Exception:
            print(f"警告：层参数格式错误（{layers_str}），使用默认值13-24")
            return list(range(13, 25))
    
    config = {
        'model_name': 'vicuna-7b',
        'model_path': args.model_path,
        'train_data_path': os.path.join(root_dir, args.train_data_path),
        'test_data_path': os.path.join(root_dir, args.test_data_path),
        'embeddings_dir': os.path.join(root_dir, 'embeddings'),
        'models_dir': os.path.join(root_dir, 'models'),
        'results_dir': args.output_dir,
        'logs_dir': os.path.join(root_dir, 'logs'),
        'log_file': os.path.join(root_dir, 'output.log'),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': 1e-5,
        'save_interval': 10,
        'intervention_layer': args.intervention_layer,
        'alignment_layers': parse_layers(args.alignment_layers),
        'subspace_rank': args.subspace_rank,
        'alpha': args.alpha,
        'beta': args.beta,
        'temperature': args.temperature,
        'layers': parse_layers('10-25'),  # 默认处理的层范围
        'max_length': 256,
        'load_in_4bit': not args.disable_4bit,  # 如果禁用4位量化，则设置为False
        'gain_factor': args.gain_factor,  # 用于增强重定位模块的权重初始化
        'intervention_strength': args.intervention_strength  # 用于增强干预效果
    }
    
    return config


def print_config_summary(config):
    """打印配置摘要"""
    print("===== SafeInt 训练配置 ======")
    print(f"模型路径: {config['model_path']}")
    print(f"训练数据路径: {config['train_data_path']}")
    print(f"测试数据路径: {config['test_data_path']}")
    print(f"结果输出目录: {config['results_dir']}")
    print(f"干预层: {config['intervention_layer']}")
    print(f"对齐层: {config['alignment_layers']}")
    print(f"子空间维度: {config['subspace_rank']}")
    print(f"训练轮数: {config['epochs']}")
    print(f"批大小: {config['batch_size']}")
    print(f"学习率: {config['learning_rate']}")
    print(f"是否使用4位量化: {config['load_in_4bit']}")
    print(f"增益因子: {config['gain_factor']}")
    print(f"干预强度: {config['intervention_strength']}")
    print("===========================")


if __name__ == "__main__":
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        # 创建配置
        config = create_config_from_args(args)
        
        # 打印配置摘要
        print_config_summary(config)
        
        # 确保输出目录存在
        os.makedirs(config['results_dir'], exist_ok=True)
        os.makedirs(config['embeddings_dir'], exist_ok=True)
        os.makedirs(config['models_dir'], exist_ok=True)
        
        # 初始化SafeInt管道
        pipeline = SafeIntPipeline(config)
        
        # 执行训练流程
        print("开始训练流程...")
        
        # 1. 提取表征
        print("\n步骤1: 提取模型表征...")
        if not pipeline.step_1_extract_embeddings():
            print("表征提取失败，退出程序")
            sys.exit(1)
        
        # 2. 训练分类器
        print("\n步骤2: 训练分类器...")
        if not pipeline.step_2_train_classifier():
            print("分类器训练失败，退出程序")
            sys.exit(1)
        
        # 3. 训练SafeInt模型
        print("\n步骤3: 训练SafeInt模型...")
        if not pipeline.step_3_train_safeint():
            print("SafeInt模型训练失败，退出程序")
            sys.exit(1)
        
        # 4. 评估SafeInt模型效果
        print("\n步骤4: 评估SafeInt模型效果...")
        if not pipeline.step_5_evaluation():
            print("评估失败，但训练过程已完成")
        
        print("\n===== SafeInt 训练流程已完成 ======")
        print(f"所有结果已保存到: {config['results_dir']}")
        print("===============================")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"训练过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        print("请检查错误信息并尝试修复问题")
    finally:
        # 确保关闭管道
        if 'pipeline' in locals():
            pipeline.close()