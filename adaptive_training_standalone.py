#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AdaptEgo 自适应权重预测训练脚本 - 独立运行版本
适配ROS2 Humble环境，支持纯Python训练
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ROS2兼容导入 - 仅在ROS环境下使用
try:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import LaserScan
    from std_msgs.msg import Float32MultiArray
    ROS_AVAILABLE = True
    print("✅ ROS2环境可用")
except ImportError:
    ROS_AVAILABLE = False
    print("⚠️  ROS2环境不可用，使用纯Python模式")

class WeightPredictorNetwork(nn.Module):
    """自适应权重预测网络"""
    
    def __init__(self, input_dim=12, output_dim=6, hidden_dims=[128, 128, 64]):
        super().__init__()
        
        # 构建多层网络
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
            
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())  # 输出[0,1]，后续会缩放
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class AdaptiveWeightTrainer:
    """自适应权重训练器"""
    
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer  
        self.criterion = criterion
        self.device = device
        
    def train_epoch(self, dataloader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)
    
    def validate_epoch(self, dataloader):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                
        return total_loss / len(dataloader)

def generate_synthetic_data(save_path, n_samples=2000):
    """生成合成专家演示数据用于测试"""
    print(f"🔄 生成 {n_samples} 个合成训练样本...")
    
    # 生成随机环境状态
    states = np.random.randn(n_samples, 12)
    
    # 规范化到合理范围
    # 位置 [0, 10]
    states[:, 0:3] = states[:, 0:3] * 2 + 5
    # 目标相对位置 [-5, 5]  
    states[:, 3:6] = states[:, 3:6] * 2
    # 环境特征 [0, 5]
    states[:, 6:12] = np.abs(states[:, 6:12]) * 2
    
    # 基于启发式规则生成专家权重
    weights = np.zeros((n_samples, 6))
    
    for i in range(n_samples):
        # 提取关键特征
        goal_dist = np.linalg.norm(states[i, 3:6])
        obstacle_density = states[i, 6]
        
        # 专家策略
        # w_smooth: 距离越远越需要平滑
        weights[i, 0] = 0.5 + 0.5 * min(goal_dist / 10.0, 1.0)
        
        # w_collision: 障碍越多权重越高
        weights[i, 1] = 3.0 + 5.0 * min(obstacle_density / 5.0, 1.0)
        
        # w_time: 基础时间权重
        weights[i, 2] = 0.3 + 0.4 * np.random.rand()
        
        # corridor_width: 障碍多时走廊要宽
        weights[i, 3] = 0.8 + 0.5 * (1 - min(obstacle_density / 5.0, 1.0))
        
        # max_velocity: 根据距离和障碍调整
        weights[i, 4] = 1.0 + 1.5 * min(goal_dist / 10.0, 1.0) * (1 - min(obstacle_density / 5.0, 1.0))
        
        # replan_freq: 障碍多时频率高
        weights[i, 5] = 10.0 + 15.0 * min(obstacle_density / 5.0, 1.0)
    
    # 添加噪声
    weights += np.random.normal(0, 0.1, weights.shape)
    
    # 模拟成功率（基于权重合理性）
    success = np.random.choice([0, 1], n_samples, p=[0.15, 0.85])
    
    # 保存数据
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, states=states, weights=weights, success=success)
    print(f"✅ 合成数据已保存到: {save_path}")

def plot_training_results(train_losses, val_losses):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失', linewidth=2)
    plt.plot(val_losses, label='验证损失', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('🚀 训练过程')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(np.log10(train_losses), label='Log训练损失', linewidth=2)
    plt.plot(np.log10(val_losses), label='Log验证损失', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Log10(MSE Loss)')
    plt.title('🔍 对数损失曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/training_curves.png', dpi=300, bbox_inches='tight')
    print("📊 训练曲线已保存: results/training_curves.png")
    
    # 如果是交互环境就显示
    try:
        plt.show()
    except:
        pass

def generate_performance_report(model, X_test, y_test, weight_ranges, device):
    """生成详细的性能报告"""
    model.eval()
    
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_pred = model(X_test_tensor).cpu().numpy()
    
    # 计算误差统计
    mse = np.mean((y_pred - y_test) ** 2)
    mae = np.mean(np.abs(y_pred - y_test))
    
    # 各维度误差
    dim_errors = np.mean((y_pred - y_test) ** 2, axis=0)
    
    weight_names = ['平滑度', '避障', '时间', '走廊宽度', '最大速度', '重规划频率']
    
    # 生成报告
    report = f"""
🎯 AdaptEgo 自适应权重预测性能报告
{'=' * 50}

📊 整体性能指标:
  • 测试样本数量: {len(X_test):,}
  • 均方误差 (MSE): {mse:.6f}
  • 平均绝对误差 (MAE): {mae:.6f}
  • 相关系数: {np.corrcoef(y_pred.flatten(), y_test.flatten())[0,1]:.4f}

🎛️ 各维度性能 (归一化空间):
"""
    
    for i, name in enumerate(weight_names):
        report += f"  • {name}: MSE={dim_errors[i]:.6f}, MAE={np.mean(np.abs(y_pred[:,i] - y_test[:,i])):.6f}\n"
    
    report += f"""
🎯 实际权重空间性能估计:
"""
    
    for i, name in enumerate(weight_names):
        min_val, max_val = weight_ranges[i]
        actual_mae = np.mean(np.abs(y_pred[:,i] - y_test[:,i])) * (max_val - min_val)
        report += f"  • {name}: 平均误差 ≈ {actual_mae:.4f} (范围: [{min_val}, {max_val}])\n"
    
    report += f"""
🏆 模型特点:
  • 输入维度: 12 (环境状态特征)
  • 输出维度: 6 (EGO-Planner权重参数)
  • 网络结构: 12 → 128 → 128 → 64 → 6
  • 激活函数: ReLU + Sigmoid输出
  • 正则化: Dropout(0.2) + L2权重衰减

📈 推荐使用场景:
  • 动态环境下的无人机路径规划
  • 需要实时权重调整的场景
  • EGO-Planner参数自动化调优

⚠️  注意事项:
  • 模型基于仿真数据训练，实际部署时需要域适应
  • 建议结合安全机制使用
  • 定期重新训练以适应新环境

生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # 保存报告
    with open('results/performance_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("📄 性能报告已保存: results/performance_report.txt")
    print(f"📊 测试集MSE: {mse:.6f}, MAE: {mae:.6f}")

def main():
    """主函数 - 完整的训练管道"""
    print("🚀 AdaptEgo 自适应权重预测训练系统")
    print("=" * 50)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    
    # 创建输出目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # 1. 检查数据
    data_path = 'data/expert_demonstrations.npz'
    if not os.path.exists(data_path):
        print("📁 未找到专家数据，生成合成训练数据...")
        generate_synthetic_data(data_path, n_samples=2000)
    
    # 2. 加载数据
    print("📊 加载训练数据...")
    try:
        data = np.load(data_path)
        states = data['states']
        weights = data['weights']
        success = data.get('success', np.ones(len(states)))
        
        print(f"   状态数据: {states.shape}")
        print(f"   权重数据: {weights.shape}")
        print(f"   成功率: {success.mean():.2%}")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 3. 过滤成功的数据
    if 'success' in data:
        success_mask = success == 1
        states = states[success_mask]
        weights = weights[success_mask]
        print(f"✅ 使用成功样本: {len(states)} 个")
    
    # 4. 数据标准化
    print("🔄 数据预处理...")
    state_mean = states.mean(axis=0)
    state_std = states.std(axis=0) + 1e-8
    states_norm = (states - state_mean) / state_std
    
    # 权重归一化到[0,1]
    weight_ranges = np.array([
        [0.1, 2.0], [1.0, 10.0], [0.1, 1.0],
        [0.3, 1.5], [0.5, 3.0], [5.0, 30.0]
    ])
    weights_norm = np.zeros_like(weights)
    for i in range(6):
        min_val, max_val = weight_ranges[i]
        weights_norm[:, i] = (weights[:, i] - min_val) / (max_val - min_val)
        weights_norm[:, i] = np.clip(weights_norm[:, i], 0, 1)
    
    # 5. 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        states_norm, weights_norm, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42
    )
    
    print(f"   训练集: {X_train.shape[0]} 样本")
    print(f"   验证集: {X_val.shape[0]} 样本")
    print(f"   测试集: {X_test.shape[0]} 样本")
    
    # 6. 创建数据加载器
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), 
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val), 
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 7. 创建网络
    print("🧠 创建神经网络...")
    model = WeightPredictorNetwork(
        input_dim=12,
        output_dim=6,
        hidden_dims=[128, 128, 64]
    ).to(device)
    
    print(f"   网络参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 8. 训练配置
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5
    )
    criterion = nn.MSELoss()
    
    # 9. 训练过程
    print("\n🚀 开始训练...")
    trainer = AdaptiveWeightTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device
    )
    
    # TensorBoard记录
    writer = SummaryWriter('logs/adaptive_training')
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    train_losses = []
    val_losses = []
    
    for epoch in range(100):  # 减少到100轮用于快速演示
        # 训练
        train_loss = trainer.train_epoch(train_loader)
        
        # 验证
        val_loss = trainer.validate_epoch(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # TensorBoard记录
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'state_mean': state_mean,
                'state_std': state_std,
                'weight_ranges': weight_ranges
            }, 'models/best_adaptive_model.pth')
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Train={train_loss:.6f}, Val={val_loss:.6f}, "
                  f"LR={optimizer.param_groups[0]['lr']:.2e}")
        
        if patience_counter >= patience:
            print(f"早停于epoch {epoch}")
            break
    
    writer.close()
    
    # 10. 最终评估
    print("\n📊 最终评估...")
    
    # 加载最佳模型
    checkpoint = torch.load('models/best_adaptive_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 测试集评估
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test), 
        torch.FloatTensor(y_test)
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    test_loss = trainer.validate_epoch(test_loader)
    print(f"🎯 测试集损失: {test_loss:.6f}")
    
    # 11. 导出TorchScript模型
    print("💾 导出部署模型...")
    model.eval()
    dummy_input = torch.randn(1, 12).to(device)
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save('models/adaptive_weights.ts')
    print("   TorchScript模型已保存: models/adaptive_weights.ts")
    
    # 12. 可视化结果
    plot_training_results(train_losses, val_losses)
    
    # 13. 生成性能报告
    generate_performance_report(model, X_test, y_test, weight_ranges, device)
    
    print("\n✅ 训练完成！")
    print("📁 输出文件:")
    print("   • models/best_adaptive_model.pth - PyTorch模型")
    print("   • models/adaptive_weights.ts - TorchScript模型")
    print("   • results/training_curves.png - 训练曲线")
    print("   • results/performance_report.txt - 性能报告")

if __name__ == '__main__':
    main()
