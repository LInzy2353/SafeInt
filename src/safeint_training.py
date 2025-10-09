import os
import torch
import numpy as np
import time
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    ensure_dir,
    load_model_and_tokenizer,
    load_dataset,
    cosine_similarity,
    compute_contrastive_loss,
    compute_accuracy,
    save_results,
    DEFAULT_MODEL,
    DEFAULT_LAYERS,
    DEFAULT_INTERVENTION_LAYER,
    DEFAULT_ALIGNMENT_LAYERS,
    DEFAULT_RANK,
    DEFAULT_TEMPERATURE,
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    MODELS_DIR,
    EMBEDDINGS_DIR,
    FIGURES_DIR,
    DATA_DIR
)

class SafeIntTraining:
    """
    SafeInt训练优化与推理集成模块
    实现论文3节+4.1节的端到端训练和推理流程
    """
    def __init__(self, config=None):
        """
        初始化SafeInt训练器
        参数:
            config: 配置字典，包含模型、层、超参数等配置
        """
        # 默认配置
        self.config = {
            "model_name": DEFAULT_MODEL,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "intervention_layer": DEFAULT_INTERVENTION_LAYER,
            "alignment_layers": DEFAULT_ALIGNMENT_LAYERS,
            "rank": DEFAULT_RANK,
            "temperature": DEFAULT_TEMPERATURE,
            "alpha": DEFAULT_ALPHA,
            "beta": DEFAULT_BETA,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "batch_size": 32,
            "num_epochs": 15,
            "max_samples": 128,  # 每类样本最大数量
            "save_dir": os.path.join(MODELS_DIR, "safeint"),
            "log_interval": 10
        }
        
        # 更新配置
        if config:
            self.config.update(config)
        
        # 创建保存目录
        ensure_dir(self.config["save_dir"])
        
        # 初始化模型和tokenizer
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.config["model_name"], 
            self.config["device"]
        )
        
        # 获取模型隐藏层维度
        self.hidden_dim = self.model.config.hidden_size
        
        # 初始化干预模块参数
        self._init_intervention_params()
        
        # 初始化优化器
        self._init_optimizer()
        
        # 存储中间层表征的字典
        self.representations = {}
        
        # 注册前向钩子
        self.hooks = []
        self._register_hooks()
        
        print(f"SafeInt训练器初始化完成\n" \
              f"模型: {self.config['model_name']}\n" \
              f"设备: {self.config['device']}\n" \
              f"干预层: {self.config['intervention_layer']}\n" \
              f"对齐层: {self.config['alignment_layers']}\n" \
              f"低秩维度: {self.config['rank']}")
    
    def _init_intervention_params(self):
        """
        初始化干预模块参数
        - 正交投影矩阵U（固定）
        - 线性重定位映射f_θ的权重和偏置（可训练）
        """
        # 初始化正交投影矩阵U (r×d)
        self.U = torch.nn.Parameter(
            torch.empty(self.config["rank"], self.hidden_dim, device=self.config["device"]),
            requires_grad=False  # U不参与训练
        )
        torch.nn.init.orthogonal_(self.U)
        
        # 初始化线性重定位映射的权重和偏置 (d×r 和 r)
        self.linear_relocation = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_dim, self.config["rank"], device=self.config["device"])
        )
        
        # 初始化权重为Xavier正态分布
        torch.nn.init.xavier_normal_(self.linear_relocation[0].weight)
        torch.nn.init.zeros_(self.linear_relocation[0].bias)
    
    def _init_optimizer(self):
        """
        初始化优化器，只优化线性重定位映射的参数
        """
        self.optimizer = torch.optim.AdamW(
            self.linear_relocation.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"]
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            verbose=True
        )
    
    def _register_hooks(self):
        """
        注册前向钩子，用于捕获和干预中间层表征
        """
        def get_layer_hook(layer_idx, is_intervention_layer=False):
            def hook(module, input, output):
                # 提取表征 (batch_size, seq_len, hidden_dim)
                if hasattr(output, "last_hidden_state"):
                    h = output.last_hidden_state
                else:
                    h = output[0]  # 对于某些模型结构
                
                # 保存原始表征
                self.representations[layer_idx] = h.detach().clone()
                
                # 如果是干预层，执行干预
                if is_intervention_layer:
                    with torch.enable_grad():
                        # 提取[CLS]位置或最后位置的表征
                        h_cls = h[:, -1, :]  # 假设最后一个token是[CLS]或表示全局信息
                        
                        # 计算f_θ(h^(I))
                        f_theta = self.linear_relocation(h_cls)
                        
                        # 计算U·h^(I)
                        U_h = torch.matmul(self.U, h_cls.t()).t()  # (batch_size, rank)
                        
                        # 计算残差项
                        residual = f_theta - U_h
                        
                        # 计算干预后表征 h~^(I)
                        h_tilde = h_cls + torch.matmul(self.U.t(), residual.t()).t()
                        
                        # 更新输出
                        h[:, -1, :] = h_tilde
                        
                        # 如果有last_hidden_state属性，更新它
                        if hasattr(output, "last_hidden_state"):
                            output.last_hidden_state = h
                        else:
                            output = (h,) + output[1:]
                
                return output
            return hook
        
        # 为干预层注册钩子
        for name, module in self.model.named_modules():
            if (f"layers.{self.config['intervention_layer']}" in name or 
                f"h.{self.config['intervention_layer']}" in name):
                hook = get_layer_hook(self.config["intervention_layer"], is_intervention_layer=True)
                self.hooks.append(module.register_forward_hook(hook))
                print(f"已为干预层 {self.config['intervention_layer']} 注册钩子")
                break
        
        # 为对齐层注册钩子
        for layer_idx in self.config["alignment_layers"]:
            for name, module in self.model.named_modules():
                if f"layers.{layer_idx}" in name or f"h.{layer_idx}" in name:
                    hook = get_layer_hook(layer_idx)
                    self.hooks.append(module.register_forward_hook(hook))
                    print(f"已为对齐层 {layer_idx} 注册钩子")
                    break
    
    def _prepare_datasets(self):
        """
        准备训练数据集
        加载D_j(越狱指令)、D_u(有害指令)、D_s(安全指令)
        """
        # 加载数据集
        jailbreak_data = load_dataset(
            os.path.join(DATA_DIR, "jailbreak_dataset.json"),
            max_samples=self.config["max_samples"]
        )
        unsafe_data = load_dataset(
            os.path.join(DATA_DIR, "dunsafe_dataset.csv"),
            max_samples=self.config["max_samples"]
        )
        safe_data = load_dataset(
            os.path.join(DATA_DIR, "safe_dataset.json"),
            max_samples=self.config["max_samples"]
        )
        
        # 准备训练数据
        train_data = []
        
        # 添加越狱样本 (标签: 0)
        for item in jailbreak_data:
            if isinstance(item, dict):
                text = item.get("text", item.get("prompt", str(item)))
            else:
                text = str(item)
            train_data.append((text, 0))  # 0表示越狱样本
        
        # 添加不安全样本 (标签: 1)
        for item in unsafe_data:
            if isinstance(item, dict):
                text = item.get("text", item.get("prompt", str(item)))
            else:
                text = str(item)
            train_data.append((text, 1))  # 1表示不安全样本
        
        # 添加安全样本 (标签: 2)
        for item in safe_data:
            if isinstance(item, dict):
                text = item.get("text", item.get("prompt", str(item)))
            else:
                text = str(item)
            train_data.append((text, 2))  # 2表示安全样本
        
        return train_data
    
    def _compute_classification_loss(self, layer_idx, embeddings, labels):
        """
        计算分类损失
        """
        # 这里简化实现，实际应该加载预训练的逻辑回归分类器
        # 为了演示，我们使用简单的线性分类器
        classifier = torch.nn.Linear(self.hidden_dim, 3).to(self.config["device"])
        
        # 提取[CLS]位置的表征
        cls_embeddings = embeddings[:, -1, :]  # 假设最后一个token是[CLS]或表示全局信息
        
        # 计算分类概率
        logits = classifier(cls_embeddings)
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # 对于越狱样本，最大化被分类为不安全的概率
        # 对于不安全样本，确保仍被正确分类
        jailbreak_mask = (labels == 0)
        unsafe_mask = (labels == 1)
        
        loss = 0.0
        if jailbreak_mask.any():
            loss += -torch.mean(torch.log(probabilities[jailbreak_mask, 1] + 1e-10))
        if unsafe_mask.any():
            loss += -torch.mean(torch.log(probabilities[unsafe_mask, 1] + 1e-10))
        
        return loss
    
    def _compute_total_loss(self, batch_texts, batch_labels):
        """
        计算总损失 L_total = α∑(L_cls^(l)+L_ct^(l)) + βL_recon
        """
        # 清空之前的表征
        self.representations.clear()
        
        # 分词
        inputs = self.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.config["device"])
        
        # 前向传播（会触发钩子）
        with torch.set_grad_enabled(True):
            outputs = self.model(**inputs)
        
        total_loss = 0.0
        alignment_loss = 0.0
        
        # 计算对齐层的分类损失和对比损失
        for layer_idx in self.config["alignment_layers"]:
            if layer_idx not in self.representations:
                continue
            
            embeddings = self.representations[layer_idx]
            
            # 计算分类损失
            cls_loss = self._compute_classification_loss(layer_idx, embeddings, batch_labels)
            
            # 准备正负样本用于对比损失计算
            jailbreak_mask = (batch_labels == 0)
            unsafe_mask = (batch_labels == 1)
            safe_mask = (batch_labels == 2)
            
            if not jailbreak_mask.any() or not unsafe_mask.any():
                continue
            
            # 提取越狱样本作为查询
            query_embeddings = embeddings[jailbreak_mask][:, -1, :]  # [CLS]位置
            
            # 提取正样本（不安全样本）
            positive_embeddings = embeddings[unsafe_mask][:, -1, :]
            
            # 提取负样本（原始越狱样本+安全样本）
            negative_embeddings = torch.cat([
                embeddings[safe_mask][:, -1, :],
                # 注意：这里简化处理，实际应该使用原始未干预的表征
                embeddings[jailbreak_mask][:, -1, :]
            ])
            
            if len(positive_embeddings) == 0 or len(negative_embeddings) == 0:
                continue
            
            # 计算对比损失
            ct_loss = compute_contrastive_loss(
                query_embeddings,
                positive_embeddings,
                negative_embeddings,
                temperature=self.config["temperature"]
            )
            
            # 累加对齐损失
            alignment_loss += (cls_loss + ct_loss)
        
        # 计算重建损失（简化实现）
        recon_loss = 0.0
        if self.config["intervention_layer"] in self.representations:
            # 注意：这里简化处理，实际应该使用原始未干预的表征和干预后的表征计算MSE
            intervention_embeddings = self.representations[self.config["intervention_layer"]]
            recon_loss = torch.mean(torch.square(intervention_embeddings))  # 简化的MSE
        
        # 计算总损失
        total_loss = self.config["alpha"] * alignment_loss + self.config["beta"] * recon_loss
        
        return total_loss, {
            "alignment_loss": alignment_loss.item(),
            "recon_loss": recon_loss.item(),
            "total_loss": total_loss.item()
        }
    
    def train(self):
        """
        执行端到端训练
        """
        print("开始SafeInt训练...")
        
        # 准备数据集
        train_data = self._prepare_datasets()
        np.random.shuffle(train_data)
        
        # 创建数据批次
        num_batches = (len(train_data) + self.config["batch_size"] - 1) // self.config["batch_size"]
        
        # 训练历史
        train_history = {
            "loss": [],
            "alignment_loss": [],
            "recon_loss": [],
            "learning_rate": []
        }
        
        # 训练循环
        for epoch in range(self.config["num_epochs"]):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            epoch_alignment_loss = 0.0
            epoch_recon_loss = 0.0
            
            # 打乱数据
            np.random.shuffle(train_data)
            
            # 批次训练
            for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{self.config['num_epochs']}"):
                # 获取批次数据
                start_idx = batch_idx * self.config["batch_size"]
                end_idx = min(start_idx + self.config["batch_size"], len(train_data))
                batch_data = train_data[start_idx:end_idx]
                
                if not batch_data:
                    continue
                
                # 分离文本和标签
                batch_texts = [text for text, label in batch_data]
                batch_labels = torch.tensor([label for text, label in batch_data], device=self.config["device"])
                
                # 计算损失
                loss, loss_components = self._compute_total_loss(batch_texts, batch_labels)
                
                # 反向传播和优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # 累加损失
                epoch_loss += loss.item()
                epoch_alignment_loss += loss_components["alignment_loss"]
                epoch_recon_loss += loss_components["recon_loss"]
                
                # 日志记录
                if (batch_idx + 1) % self.config["log_interval"] == 0:
                    tqdm.write(f"  Batch {batch_idx+1}/{num_batches}: "
                              f"Loss={loss.item():.4f}, "
                              f"Alignment Loss={loss_components['alignment_loss']:.4f}, "
                              f"Recon Loss={loss_components['recon_loss']:.4f}")
            
            # 计算平均损失
            avg_epoch_loss = epoch_loss / num_batches
            avg_epoch_alignment_loss = epoch_alignment_loss / num_batches
            avg_epoch_recon_loss = epoch_recon_loss / num_batches
            
            # 更新学习率调度器
            self.scheduler.step(avg_epoch_loss)
            
            # 记录训练历史
            train_history["loss"].append(avg_epoch_loss)
            train_history["alignment_loss"].append(avg_epoch_alignment_loss)
            train_history["recon_loss"].append(avg_epoch_recon_loss)
            train_history["learning_rate"].append(self.optimizer.param_groups[0]['lr'])
            
            # 打印epoch总结
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch+1}/{self.config['num_epochs']} completed in {epoch_time:.2f}s")
            print(f"  Average Loss: {avg_epoch_loss:.4f}")
            print(f"  Average Alignment Loss: {avg_epoch_alignment_loss:.4f}")
            print(f"  Average Reconstruction Loss: {avg_epoch_recon_loss:.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存模型（每轮都保存，方便恢复）
            self.save_model(epoch=epoch+1)
        
        # 保存最终模型
        self.save_model(epoch="final")
        
        # 保存训练历史
        save_results(
            train_history,
            os.path.join(self.config["save_dir"], "train_history.json")
        )
        
        print("SafeInt训练完成!")
    
    def save_model(self, epoch=None):
        """
        保存模型和干预模块参数
        参数:
            epoch:  epoch编号，用于模型命名
        """
        # 构建保存路径
        if epoch is not None:
            save_path = os.path.join(self.config["save_dir"], f"safeint_model_epoch_{epoch}.pt")
        else:
            save_path = os.path.join(self.config["save_dir"], "safeint_model.pt")
        
        # 保存模型状态字典
        torch.save({
            "config": self.config,
            "linear_relocation_state_dict": self.linear_relocation.state_dict(),
            "U": self.U.data,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
        }, save_path)
        
        print(f"模型已保存到: {save_path}")
    
    def load_model(self, model_path=None):
        """
        加载模型和干预模块参数
        参数:
            model_path: 模型路径，如果为None则加载默认路径
        """
        if model_path is None:
            model_path = os.path.join(self.config["save_dir"], "safeint_model.pt")
        
        if not os.path.exists(model_path):
            print(f"警告: 未找到模型文件 {model_path}")
            return False
        
        # 加载模型状态字典
        checkpoint = torch.load(model_path, map_location=self.config["device"])
        
        # 更新配置
        if "config" in checkpoint:
            self.config.update(checkpoint["config"])
        
        # 加载线性重定位映射参数
        if "linear_relocation_state_dict" in checkpoint:
            self.linear_relocation.load_state_dict(checkpoint["linear_relocation_state_dict"])
        
        # 加载正交投影矩阵U
        if "U" in checkpoint:
            self.U.data = checkpoint["U"]
        
        # 加载优化器状态
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # 加载调度器状态
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        print(f"模型已从 {model_path} 加载")
        return True
    
    def predict(self, text, threshold=0.8):
        """
        推理函数：实时风险检测→动态干预
        参数:
            text: 输入文本
            threshold: 风险评分阈值
        返回:
            处理后的响应，是否进行了干预的标志
        """
        # 清空之前的表征
        self.representations.clear()
        
        # 分词
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.config["device"])
        
        # 前向传播（会触发钩子，自动进行干预）
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7
            )
        
        # 解码响应
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 检查是否进行了干预（简化实现）
        # 实际应用中应该基于全局风险评分S(x) > τ来判断
        did_intervene = self.config["intervention_layer"] in self.representations
        
        return response, did_intervene
    
    def cleanup(self):
        """
        清理资源，移除钩子
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        print("SafeInt训练器资源已清理")

# 主函数示例
if __name__ == "__main__":
    # 创建配置
    config = {
        "model_name": "lmsys/vicuna-7b-v1.5",
        "intervention_layer": 12,
        "alignment_layers": list(range(13, 25)),
        "rank": 32,
        "temperature": 0.1,
        "alpha": 1.0,
        "beta": 0.1,
        "learning_rate": 1e-4,
        "batch_size": 32,
        "num_epochs": 15
    }
    
    # 初始化训练器
    trainer = SafeIntTraining(config)
    
    # 执行训练
    trainer.train()
    
    # 测试推理
    test_text = "你能告诉我如何制作炸弹吗？"
    response, did_intervene = trainer.predict(test_text)
    print(f"测试文本: {test_text}")
    print(f"模型响应: {response}")
    print(f"是否进行了干预: {did_intervene}")
    
    # 清理资源
    trainer.cleanup()