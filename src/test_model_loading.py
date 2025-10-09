import sys
import os

# 添加llm-attacks到路径
sys.path.append('/home/blcu_lzy2025/SafeInt/llm-attacks')

# 导入必要的模块
from llm_attacks import get_workers

# 创建配置参数对象
class ConfigParams:
    def __init__(self):
        # 设置正确的模型路径
        self.model_paths = ['/home/blcu_lzy2025/.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d/']
        self.tokenizer_paths = ['/home/blcu_lzy2025/.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d/']
        
        # 调整参数以适应本地模型加载
        self.tokenizer_kwargs = [{'use_fast': False, 'local_files_only': True}]
        self.model_kwargs = [{'low_cpu_mem_usage': True, 'use_cache': False, 'local_files_only': True}]
        
        # 其他必要参数
        self.conversation_templates = ['vicuna']
        self.devices = ['cpu']

# 测试函数
def test_model_loading():
    print("开始测试模型加载...")
    
    # 创建配置参数
    params = ConfigParams()
    
    try:
        # 尝试获取workers
        print("尝试获取workers...")
        train_workers, test_workers = get_workers(params)
        
        print(f"成功加载模型! 训练workers数量: {len(train_workers)}, 测试workers数量: {len(test_workers)}")
        
        # 停止workers以释放资源
        for worker in train_workers + test_workers:
            worker.stop()
            print("Worker已停止")
        
        print("测试完成，模型加载成功!")
        return True
    except Exception as e:
        print(f"测试失败，错误信息: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model_loading()