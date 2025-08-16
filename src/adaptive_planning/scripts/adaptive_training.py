#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
完整的自适应规划训练管道
整合行为克隆、强化学习和专家数据生成
基于你的网络架构和训练流程设计
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import pickle
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
    """
    自适应权重预测网络 - 结合你的架构设计
    """
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
        
        # 权重范围定义（对应EGO-Planner参数）
        self.weight_ranges = np.array([
            [0.1, 2.0],    # w_smooth
            [1.0, 10.0],   # w_collision  
            [0.1, 1.0],    # w_time
            [0.3, 1.5],    # corridor_width
            [0.5, 3.0],    # max_velocity
            [5.0, 30.0]    # replan_freq
        ])
        
    def forward(self, state_features):
        """前向传播"""
        normalized_weights = self.network(state_features)
        
        # 反归一化到实际权重范围
        min_vals = torch.tensor(self.weight_ranges[:, 0], dtype=torch.float32)
        max_vals = torch.tensor(self.weight_ranges[:, 1], dtype=torch.float32)
        
        # 移到相同设备
        if normalized_weights.is_cuda:
            min_vals = min_vals.cuda()
            max_vals = max_vals.cuda()
        
        # 广播到正确的形状
        if normalized_weights.dim() == 2:  # 批处理
            min_vals = min_vals.unsqueeze(0).expand_as(normalized_weights)
            max_vals = max_vals.unsqueeze(0).expand_as(normalized_weights)
            
        weights = min_vals + normalized_weights * (max_vals - min_vals)
        return weights

class ExpertDataCollector:
    """
    专家演示数据收集器 - 结合你的数据收集逻辑
    """
    def __init__(self, output_dir='data'):
        rospy.init_node('expert_data_collector', anonymous=True)
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 数据缓存
        self.states = []
        self.weights = []
        self.success_flags = []
        
        # 当前状态 - 扩展状态变量
        self.current_pos = np.zeros(3)
        self.current_vel = np.zeros(3)
        self.goal_pos = np.array([2.0, 0.0, 1.0])
        self.obstacle_density = 0.0
        self.avg_clearance = 5.0
        self.battery_level = 1.0
        self.current_weights = np.array([1.5, 5.0, 0.5, 0.8, 2.0, 15.0])  # 默认权重
        
        # 订阅者 - 兼容你的话题设计
        self.odom_sub = rospy.Subscriber('/mavros/local_position/odom', 
                                       Odometry, self.odom_callback, queue_size=10)
        self.goal_sub = rospy.Subscriber('/move_base_simple/goal', 
                                       PoseStamped, self.goal_callback, queue_size=10)
        self.scan_sub = rospy.Subscriber('/scan', 
                                       LaserScan, self.scan_callback, queue_size=10)
        self.weights_sub = rospy.Subscriber('/planner/adaptive_weights',
                                          Float32MultiArray, self.weights_callback, queue_size=10)
        
        rospy.loginfo("专家数据收集器已启动")
        
    def odom_callback(self, msg):
        """里程计回调"""
        self.current_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        self.current_vel = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])
        
    def goal_callback(self, msg):
        """目标回调"""
        self.goal_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        
    def scan_callback(self, msg):
        """激光扫描回调"""
        ranges = np.array(msg.ranges)
        valid_ranges = ranges[np.isfinite(ranges) & (ranges > 0.1)]
        
        if len(valid_ranges) > 0:
            self.avg_clearance = np.mean(valid_ranges)
            close_obstacles = np.sum(valid_ranges < 1.0)
            self.obstacle_density = close_obstacles / len(valid_ranges)
        else:
            self.avg_clearance = 5.0
            self.obstacle_density = 0.0
            
    def weights_callback(self, msg):
        """权重回调"""
        if len(msg.data) == 6:
            self.current_weights = np.array(msg.data)
        
    def compute_state_features(self):
        """计算当前状态特征 - 与网络输入保持一致"""
        goal_distance = np.linalg.norm(self.goal_pos - self.current_pos)
        current_speed = np.linalg.norm(self.current_vel)
        path_complexity = self.obstacle_density * np.log(goal_distance + 1.0)
        
        state = np.array([
            # 当前位置 (3)
            self.current_pos[0], self.current_pos[1], self.current_pos[2],
            # 目标位置 (3)  
            self.goal_pos[0], self.goal_pos[1], self.goal_pos[2],
            # 环境特征 (6)
            self.obstacle_density, self.avg_clearance, current_speed,
            goal_distance, path_complexity, self.battery_level
        ], dtype=np.float32)
        
        return state
        
    def record_sample(self, success=True):
        """记录一个样本"""
        state = self.compute_state_features()
        self.states.append(state)
        self.weights.append(self.current_weights.copy())
        self.success_flags.append(1.0 if success else 0.0)
        
        rospy.loginfo(f"记录样本 #{len(self.states)}, 成功: {success}")
        
    def save_data(self, filename='expert_demonstrations.npz'):
        """保存收集的数据"""
        if len(self.states) == 0:
            rospy.logwarn("没有数据可保存")
            return
            
        filepath = os.path.join(self.output_dir, filename)
        np.savez_compressed(
            filepath,
            states=np.array(self.states),
            weights=np.array(self.weights),
            success=np.array(self.success_flags)
        )
        
        rospy.loginfo(f"已保存 {len(self.states)} 个样本到 {filepath}")
        
        # 打印统计信息
        success_rate = np.mean(self.success_flags)
        print(f"成功率: {success_rate:.2%}")
        print(f"总样本数: {len(self.states)}")

class AdaptivePlanningTrainer:
    """
    自适应规划训练器 - 结合你的训练架构
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 创建输出目录
        os.makedirs(config['output_dir'], exist_ok=True)
        
        # 初始化网络 - 使用你的网络架构
        self.model = WeightPredictorNetwork(
            input_dim=config['input_dim'],
            output_dim=config['output_dim'],
            hidden_dims=config['hidden_dims']
        ).to(self.device)
        
        # 优化器和损失函数
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        self.criterion = nn.MSELoss()
        
        # TensorBoard
        self.writer = SummaryWriter(f"runs/adaptive_planning_{config['experiment_name']}")
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        
    def load_expert_data(self, data_path):
        """加载专家演示数据"""
        print(f"正在加载专家数据: {data_path}")
        
        if data_path.endswith('.npz'):
            data = np.load(data_path)
            states = data['states']  # [N, 12]
            weights = data['weights']  # [N, 6] 
            success_flags = data.get('success', np.ones(len(states)))
            
        else:
            raise ValueError(f"不支持的数据格式: {data_path}")
            
        # 只使用成功的演示
        success_mask = success_flags > 0.5
        states = states[success_mask]
        weights = weights[success_mask]
        
        print(f"加载了 {len(states)} 条有效演示数据")
        return states, weights
        
    def preprocess_data(self, states, weights):
        """数据预处理"""
        # 状态归一化
        states_mean = np.mean(states, axis=0)
        states_std = np.std(states, axis=0) + 1e-8
        states_norm = (states - states_mean) / states_std
        
        # 权重归一化到[0,1] - 使用网络定义的范围
        weight_ranges = self.model.weight_ranges
        weights_norm = np.zeros_like(weights)
        
        for i in range(weights.shape[1]):
            min_val, max_val = weight_ranges[i]
            weights_norm[:, i] = (weights[:, i] - min_val) / (max_val - min_val)
            weights_norm[:, i] = np.clip(weights_norm[:, i], 0.0, 1.0)
            
        # 保存归一化参数
        self.norm_params = {
            'states_mean': states_mean,
            'states_std': states_std,
            'weight_ranges': weight_ranges
        }
        
        return states_norm, weights_norm
        
    def create_data_loaders(self, states, weights):
        """创建数据加载器"""
        # 划分训练/验证集
        X_train, X_val, y_train, y_val = train_test_split(
            states, weights, test_size=0.2, random_state=42
        )
        
        # 转换为张量
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False
        )
        
        return train_loader, val_loader
        
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (states, weights) in enumerate(train_loader):
            self.optimizer.zero_grad()
            
            # 前向传播
            pred_weights = self.model(states)
            loss = self.criterion(pred_weights, weights)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
                
        return total_loss / len(train_loader)
        
    def validate(self, val_loader):
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for states, weights in val_loader:
                pred_weights = self.model(states)
                loss = self.criterion(pred_weights, weights)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
        
    def train_bc(self, data_path):
        """行为克隆训练"""
        print("开始行为克隆训练...")
        
        # 加载和预处理数据
        states, weights = self.load_expert_data(data_path)
        states_norm, weights_norm = self.preprocess_data(states, weights)
        train_loader, val_loader = self.create_data_loaders(states_norm, weights_norm)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['bc_epochs']):
            # 训练
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            # 记录损失
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # 学习率调整
            self.scheduler.step(val_loss)
            
            print(f'Epoch {epoch+1}/{self.config["bc_epochs"]}, '
                  f'Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            
            # TensorBoard记录
            self.writer.add_scalar('Training/EpochLoss', train_loss, epoch)
            self.writer.add_scalar('Validation/EpochLoss', val_loss, epoch)
            
            # 早停和模型保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model(f"{self.config['output_dir']}/best_bc_model.pth")
            else:
                patience_counter += 1
                
            if patience_counter >= self.config['patience']:
                print(f"早停于epoch {epoch+1}")
                break
                
        print("行为克隆训练完成")
        self.writer.close()
        
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'norm_params': self.norm_params,
            'config': self.config
        }, path)
        print(f"模型已保存: {path}")
        
    def export_torchscript(self, output_path):
        """导出TorchScript模型用于ROS部署"""
        self.model.eval()
        
        # 创建示例输入
        example_input = torch.randn(1, self.config['input_dim']).to(self.device)
        
        # 追踪模型
        traced_model = torch.jit.trace(self.model, example_input)
        
        # 保存TorchScript模型
        traced_model.save(output_path)
        print(f"TorchScript模型已导出: {output_path}")
        
    def plot_training_curves(self):
        """绘制训练曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.config['output_dir']}/training_curves.png")
        print(f"训练曲线已保存: {self.config['output_dir']}/training_curves.png")

def main():
    """主训练流程 - 整合所有训练组件"""
    
    print("=== Fast-Drone-250 自适应规划训练系统 ===")
    
    # 训练配置
    config = {
        'input_dim': 12,           # 状态特征维数
        'output_dim': 6,           # 权重参数数量
        'hidden_dims': [128, 128, 64],  # 隐藏层结构
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'batch_size': 64,
        'bc_epochs': 200,
        'patience': 20,
        'output_dir': 'runs/adaptive_planning',
        'experiment_name': 'fast_drone_250'
    }
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 检查专家数据
    expert_data_path = 'data/expert_demonstrations.npz'
    
    if not os.path.exists(expert_data_path):
        print(f"\n⚠️  专家数据文件不存在: {expert_data_path}")
        print("请先运行数据收集脚本生成专家演示数据:")
        print("  python3 -c \"from adaptive_training import ExpertDataCollector; collector = ExpertDataCollector(); import rospy; rospy.spin()\"")
        print("然后在RViz中手动飞行收集专家演示数据")
        return
    
    # 创建训练器
    trainer = AdaptivePlanningTrainer(config)
    
    # 执行行为克隆训练
    try:
        trainer.train_bc(expert_data_path)
        
        # 导出模型
        trainer.export_torchscript(f"{config['output_dir']}/adaptive_weights.ts")
        
        # 绘制训练曲线
        trainer.plot_training_curves()
        
        print(f"\n✅ 训练完成!")
        print(f"最佳模型: {config['output_dir']}/best_bc_model.pth")
        print(f"ROS部署模型: {config['output_dir']}/adaptive_weights.ts")
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
