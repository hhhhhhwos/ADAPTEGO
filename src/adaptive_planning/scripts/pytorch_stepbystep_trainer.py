#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 PyTorchStepByStep 的可解释训练流程
提供清晰的步骤化训练过程
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import os
import json
from datetime import datetime
import sys

# 添加PyTorchStepByStep路径
THIRD_PARTY_PATH = os.path.join(os.path.dirname(__file__), '../../third_party')
PYTORCH_STEP_PATH = os.path.join(THIRD_PARTY_PATH, 'PyTorchStepByStep')
sys.path.append(PYTORCH_STEP_PATH)

# 导入DRL-Nav网络
from drl_nav_network import DRLNavNetwork, DRLNavRewardCalculator

class StepByStepTrainer:
    """
    基于PyTorchStepByStep理念的步骤化训练器
    提供清晰、可调试的训练过程
    """
    def __init__(self, model, device='cpu'):
        self.device = device
        self.model = model.to(device)
        self.losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # 训练组件
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        
        # 数据
        self.train_loader = None
        self.val_loader = None
        
        # 训练状态
        self.epoch = 0
        self.best_loss = float('inf')
        
        print("📚 StepByStep训练器初始化完成")
    
    def set_loaders(self, train_loader, val_loader=None):
        """Step 1: 设置数据加载器"""
        self.train_loader = train_loader
        self.val_loader = val_loader
        print(f"✅ Step 1: 数据加载器设置完成")
        print(f"   训练样本: {len(train_loader.dataset)}")
        if val_loader:
            print(f"   验证样本: {len(val_loader.dataset)}")
    
    def set_optimizer(self, optimizer_class=torch.optim.Adam, **kwargs):
        """Step 2: 设置优化器"""
        default_kwargs = {'lr': 1e-3, 'weight_decay': 1e-5}
        default_kwargs.update(kwargs)
        
        self.optimizer = optimizer_class(self.model.parameters(), **default_kwargs)
        print(f"✅ Step 2: 优化器设置完成 - {optimizer_class.__name__}")
        print(f"   参数: {default_kwargs}")
        
        return self
    
    def set_lr_scheduler(self, scheduler_class=torch.optim.lr_scheduler.StepLR, **kwargs):
        """Step 3: 设置学习率调度器"""
        if self.optimizer is None:
            raise ValueError("请先设置优化器")
        
        default_kwargs = {'step_size': 50, 'gamma': 0.95}
        default_kwargs.update(kwargs)
        
        self.scheduler = scheduler_class(self.optimizer, **default_kwargs)
        print(f"✅ Step 3: 学习率调度器设置完成 - {scheduler_class.__name__}")
        print(f"   参数: {default_kwargs}")
        
        return self
    
    def set_loss_function(self, loss_fn=nn.MSELoss()):
        """Step 4: 设置损失函数"""
        self.loss_fn = loss_fn
        print(f"✅ Step 4: 损失函数设置完成 - {type(loss_fn).__name__}")
        
        return self
    
    def train_step(self):
        """Step 5: 单步训练"""
        if not all([self.train_loader, self.optimizer, self.loss_fn]):
            raise ValueError("请先完成Steps 1-4的设置")
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (states, targets) in enumerate(self.train_loader):
            states = states.to(self.device)
            targets = targets.to(self.device)
            
            # 前向传播
            predictions = self.model(states)
            loss = self.loss_fn(predictions, targets)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # 详细日志（每50个batch打印一次）
            if batch_idx % 50 == 0:
                print(f"   Batch {batch_idx}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.6f}, "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        avg_loss = total_loss / num_batches
        self.losses.append(avg_loss)
        
        if self.scheduler:
            self.scheduler.step()
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        
        return avg_loss
    
    def validation_step(self):
        """Step 6: 验证步骤"""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for states, targets in self.val_loader:
                states = states.to(self.device)
                targets = targets.to(self.device)
                
                predictions = self.model(states)
                loss = self.loss_fn(predictions, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_val_loss = total_loss / num_batches
        self.val_losses.append(avg_val_loss)
        
        # 保存最佳模型
        if avg_val_loss < self.best_loss:
            self.best_loss = avg_val_loss
            self.save_checkpoint('best_model.pth')
        
        return avg_val_loss
    
    def train(self, epochs):
        """Step 7: 完整训练循环"""
        print(f"🚀 开始训练 {epochs} 个epoch...")
        
        for epoch in range(epochs):
            self.epoch = epoch
            
            print(f"\n📈 Epoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # 训练步骤
            train_loss = self.train_step()
            
            # 验证步骤
            val_loss = self.validation_step()
            
            # 打印epoch结果
            print(f"🔸 Epoch {epoch+1} 结果:")
            print(f"   训练损失: {train_loss:.6f}")
            if val_loss is not None:
                print(f"   验证损失: {val_loss:.6f}")
            if self.scheduler:
                print(f"   学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 每10个epoch保存一次检查点
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
        
        print(f"\n✅ 训练完成！最佳验证损失: {self.best_loss:.6f}")
        return self
    
    def plot_training_curves(self, save_path=None):
        """Step 8: 可视化训练过程"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Training Progress', fontsize=16)
        
        # 损失曲线
        axes[0, 0].plot(self.losses, label='Training Loss', color='blue')
        if self.val_losses:
            axes[0, 0].plot(self.val_losses, label='Validation Loss', color='red')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 学习率曲线
        if self.learning_rates:
            axes[0, 1].plot(self.learning_rates, color='green')
            axes[0, 1].set_title('Learning Rate Schedule')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].grid(True)
        
        # 损失分布
        axes[1, 0].hist(self.losses, bins=20, alpha=0.7, color='blue', label='Training')
        if self.val_losses:
            axes[1, 0].hist(self.val_losses, bins=20, alpha=0.7, color='red', label='Validation')
        axes[1, 0].set_title('Loss Distribution')
        axes[1, 0].set_xlabel('Loss Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # 训练趋势
        if len(self.losses) > 10:
            window = min(10, len(self.losses) // 5)
            smoothed_train = np.convolve(self.losses, np.ones(window)/window, mode='valid')
            axes[1, 1].plot(smoothed_train, label=f'Smoothed Training (window={window})', color='blue')
            if self.val_losses and len(self.val_losses) > window:
                smoothed_val = np.convolve(self.val_losses, np.ones(window)/window, mode='valid')
                axes[1, 1].plot(smoothed_val, label=f'Smoothed Validation (window={window})', color='red')
            axes[1, 1].set_title('Smoothed Training Curves')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Smoothed Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 训练曲线已保存: {save_path}")
        
        plt.show()
    
    def save_checkpoint(self, filename):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'losses': self.losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'best_loss': self.best_loss
        }
        
        torch.save(checkpoint, filename)
        print(f"💾 检查点已保存: {filename}")
    
    def load_checkpoint(self, filename):
        """加载检查点"""
        checkpoint = torch.load(filename, map_location=self.device)
        
        self.epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.losses = checkpoint.get('losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.learning_rates = checkpoint.get('learning_rates', [])
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        
        print(f"📂 检查点已加载: {filename}")
        print(f"   当前epoch: {self.epoch}")
        print(f"   最佳验证损失: {self.best_loss:.6f}")

def create_synthetic_dataset(num_samples=10000):
    """创建合成数据集用于演示"""
    print("🔧 创建合成数据集...")
    
    # 生成随机状态
    # 状态：[pos(3), goal(3), obs_density, avg_clear, speed, goal_dist, complexity, battery]
    states = []
    weights = []
    
    for _ in range(num_samples):
        # 随机位置和目标
        pos = np.random.uniform(-10, 10, 3)
        goal = np.random.uniform(-10, 10, 3)
        
        # 环境特征
        obs_density = np.random.uniform(0, 1)
        avg_clear = np.random.uniform(0.5, 3.0)
        speed = np.random.uniform(0, 2.0)
        goal_dist = np.linalg.norm(goal - pos)
        complexity = np.random.uniform(0, 1)
        battery = np.random.uniform(0.5, 1.0)
        
        state = np.concatenate([pos, goal, [obs_density, avg_clear, speed, goal_dist, complexity, battery]])
        
        # 基于规则生成"专家"权重
        w_smooth = 1.0 + np.random.normal(0, 0.1)
        w_collision = 2.0 + obs_density * 5.0 + np.random.normal(0, 0.2)
        w_time = 0.3 + (1.0 / max(goal_dist, 0.1)) * 0.5 + np.random.normal(0, 0.05)
        corridor = 1.0 - obs_density * 0.4 + np.random.normal(0, 0.05)
        max_vel = 2.0 - obs_density * 0.8 + np.random.normal(0, 0.1)
        freq = 15.0 + complexity * 10.0 + np.random.normal(0, 1.0)
        
        weight = np.array([w_smooth, w_collision, w_time, corridor, max_vel, freq])
        
        states.append(state)
        weights.append(weight)
    
    # 转换为tensor
    states_tensor = torch.FloatTensor(np.array(states))
    weights_tensor = torch.FloatTensor(np.array(weights))
    
    print(f"✅ 数据集创建完成: {num_samples} 样本")
    print(f"   状态维度: {states_tensor.shape[1]}")
    print(f"   权重维度: {weights_tensor.shape[1]}")
    
    return TensorDataset(states_tensor, weights_tensor)

def main_training_pipeline():
    """主训练流程 - PyTorchStepByStep风格"""
    print("=" * 60)
    print("    🎯 自适应权重学习 - PyTorchStepByStep训练")
    print("=" * 60)
    
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    
    # Step 0: 准备数据
    print("\n📊 Step 0: 准备数据...")
    dataset = create_synthetic_dataset(num_samples=10000)
    
    # 数据分割
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 创建模型
    model = DRLNavNetwork(state_dim=12, action_dim=6)
    
    # 创建训练器
    trainer = StepByStepTrainer(model, device)
    
    # Step by Step 设置
    trainer.set_loaders(train_loader, val_loader)
    trainer.set_optimizer(torch.optim.Adam, lr=3e-4, weight_decay=1e-5)
    trainer.set_lr_scheduler(torch.optim.lr_scheduler.StepLR, step_size=30, gamma=0.9)
    trainer.set_loss_function(nn.MSELoss())
    
    # 开始训练
    trainer.train(epochs=100)
    
    # 可视化结果
    trainer.plot_training_curves('training_curves.png')
    
    # 导出最终模型
    model_export_path = 'adaptive_weights_stepbystep.ts'
    example_input = torch.randn(1, 12).to(device)
    traced_model = torch.jit.trace(model.eval(), example_input)
    traced_model.save(model_export_path)
    
    print(f"🎉 训练完成！模型已导出: {model_export_path}")
    
    return trainer

if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # 运行训练
    trainer = main_training_pipeline()
