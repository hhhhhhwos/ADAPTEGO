#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 DRL-Nav 的网络架构和奖励设计
直接使用 DRL-Nav 项目的核心代码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

# 添加DRL-Nav路径
THIRD_PARTY_PATH = os.path.join(os.path.dirname(__file__), '../../third_party')
DRL_NAV_PATH = os.path.join(THIRD_PARTY_PATH, 'DRL-Nav')
sys.path.append(DRL_NAV_PATH)

class DRLNavNetwork(nn.Module):
    """
    基于 DRL-Nav 的网络架构
    适配到我们的自适应权重预测任务
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DRLNavNetwork, self).__init__()
        
        # 特征提取器（基于DRL-Nav的设计）
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 位置编码（DRL-Nav的关键特征）
        self.position_encoder = nn.Sequential(
            nn.Linear(3, 64),  # x, y, z position
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # 目标编码
        self.goal_encoder = nn.Sequential(
            nn.Linear(3, 64),  # goal x, y, z
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # 环境特征编码
        self.env_encoder = nn.Sequential(
            nn.Linear(6, 64),  # obstacle_density, clearance, etc.
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # 注意力机制（DRL-Nav的核心）
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim + 64*3, 
            num_heads=8, 
            dropout=0.1
        )
        
        # 权重预测头
        self.weight_predictor = nn.Sequential(
            nn.Linear(hidden_dim + 64*3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()  # 权重在[0,1]之间，后续会缩放
        )
        
        # 权重缩放参数
        self.weight_scales = nn.Parameter(torch.tensor([
            2.0,   # w_smooth: [0.1, 2.0] 
            10.0,  # w_collision: [1.0, 10.0]
            1.0,   # w_time: [0.1, 1.0]
            1.5,   # corridor_width: [0.3, 1.5]
            3.0,   # max_velocity: [0.5, 3.0] 
            30.0   # replan_freq: [5, 30]
        ]))
        
        self.weight_offsets = nn.Parameter(torch.tensor([
            0.1,   # w_smooth offset
            1.0,   # w_collision offset
            0.1,   # w_time offset
            0.3,   # corridor_width offset
            0.5,   # max_velocity offset
            5.0    # replan_freq offset
        ]))
    
    def forward(self, state):
        batch_size = state.shape[0]
        
        # 分离不同类型的状态信息
        position = state[:, :3]      # 当前位置
        goal = state[:, 3:6]         # 目标位置
        env_features = state[:, 6:]  # 环境特征
        
        # 编码不同特征
        pos_encoded = self.position_encoder(position)
        goal_encoded = self.goal_encoder(goal)
        env_encoded = self.env_encoder(env_features)
        
        # 全局特征提取
        global_features = self.feature_extractor(state)
        
        # 拼接所有特征
        combined_features = torch.cat([
            global_features, pos_encoded, goal_encoded, env_encoded
        ], dim=-1)
        
        # 自注意力（DRL-Nav的关键机制）
        # reshape为 (seq_len, batch, feature)
        combined_features_reshaped = combined_features.unsqueeze(0)
        attended_features, _ = self.attention(
            combined_features_reshaped,
            combined_features_reshaped, 
            combined_features_reshaped
        )
        attended_features = attended_features.squeeze(0)
        
        # 预测权重（0-1之间）
        raw_weights = self.weight_predictor(attended_features)
        
        # 缩放到实际权重范围
        scaled_weights = raw_weights * self.weight_scales + self.weight_offsets
        
        return scaled_weights

class DRLNavRewardCalculator:
    """
    基于 DRL-Nav 的奖励函数设计
    """
    def __init__(self):
        self.prev_distance_to_goal = None
        self.collision_penalty = -100.0
        self.reach_reward = 100.0
        self.progress_scale = 10.0
        self.smoothness_scale = 0.1
        self.efficiency_scale = 0.01
    
    def calculate_reward(self, state, action, next_state, done, info=None):
        """
        DRL-Nav风格的多目标奖励函数
        """
        reward = 0.0
        
        # 提取状态信息
        pos = next_state[:3]
        goal = next_state[3:6]
        obstacle_density = next_state[6] if len(next_state) > 6 else 0.0
        
        # 1. 到达目标奖励
        distance_to_goal = np.linalg.norm(goal - pos)
        if distance_to_goal < 0.3:  # 到达目标
            reward += self.reach_reward
        
        # 2. 进度奖励（DRL-Nav的核心）
        if self.prev_distance_to_goal is not None:
            progress = self.prev_distance_to_goal - distance_to_goal
            reward += progress * self.progress_scale
        self.prev_distance_to_goal = distance_to_goal
        
        # 3. 碰撞惩罚
        if done and distance_to_goal > 0.5:  # 非正常结束
            reward += self.collision_penalty
        
        # 4. 权重合理性奖励
        if len(action) >= 6:
            w_smooth, w_collision, w_time = action[0], action[1], action[2]
            corridor_width, max_velocity = action[3], action[4]
            
            # 权重应该适应环境
            # 障碍密集时，避障权重应该高，速度应该低
            expected_collision_weight = 2.0 + obstacle_density * 6.0
            expected_max_velocity = 2.5 - obstacle_density * 1.5
            
            weight_appropriateness = (
                -abs(w_collision - expected_collision_weight) * 0.1 +
                -abs(max_velocity - expected_max_velocity) * 0.1
            )
            reward += weight_appropriateness
        
        # 5. 平滑度奖励
        if len(action) >= 6:
            smoothness_penalty = np.var(action) * self.smoothness_scale
            reward -= smoothness_penalty
        
        # 6. 效率奖励（时间惩罚）
        reward -= self.efficiency_scale
        
        return reward

class DRLNavTrainer:
    """
    基于 DRL-Nav 训练流程的包装器
    """
    def __init__(self, state_dim, action_dim, device='cpu'):
        self.device = device
        self.network = DRLNavNetwork(state_dim, action_dim).to(device)
        self.reward_calculator = DRLNavRewardCalculator()
        
        # 优化器（DRL-Nav使用Adam）
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), 
            lr=3e-4, 
            weight_decay=1e-5
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=1000, 
            gamma=0.95
        )
    
    def train_step(self, batch_states, batch_actions, batch_rewards, batch_next_states):
        """
        单步训练（模仿学习 + 强化学习）
        """
        self.network.train()
        
        # 转换为tensor
        states = torch.FloatTensor(batch_states).to(self.device)
        target_actions = torch.FloatTensor(batch_actions).to(self.device)
        rewards = torch.FloatTensor(batch_rewards).to(self.device)
        
        # 前向传播
        predicted_actions = self.network(states)
        
        # 计算损失（结合模仿学习和价值学习）
        # 1. 模仿学习损失（MSE）
        imitation_loss = F.mse_loss(predicted_actions, target_actions)
        
        # 2. 奖励加权损失（强化学习思想）
        reward_weights = torch.softmax(rewards, dim=0)
        weighted_loss = torch.mean(reward_weights.unsqueeze(1) * 
                                  (predicted_actions - target_actions) ** 2)
        
        # 3. 正则化损失
        reg_loss = torch.mean(torch.abs(predicted_actions))
        
        # 总损失
        total_loss = imitation_loss + 0.1 * weighted_loss + 0.01 * reg_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            'total_loss': total_loss.item(),
            'imitation_loss': imitation_loss.item(),
            'weighted_loss': weighted_loss.item(),
            'reg_loss': reg_loss.item()
        }
    
    def predict(self, state):
        """预测权重"""
        self.network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            weights = self.network(state_tensor)
            return weights.cpu().numpy()[0]
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"模型已保存: {path}")
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"模型已加载: {path}")

def export_drl_nav_model(trainer, output_path):
    """
    导出DRL-Nav模型为TorchScript
    """
    trainer.network.eval()
    
    # 创建示例输入（12维状态）
    example_input = torch.randn(1, 12)
    
    # 追踪模型
    traced_model = torch.jit.trace(trainer.network, example_input)
    
    # 保存TorchScript模型
    traced_model.save(output_path)
    print(f"DRL-Nav模型已导出为TorchScript: {output_path}")

# 使用示例
if __name__ == "__main__":
    # 测试网络
    state_dim = 12  # [pos(3), goal(3), env_features(6)]
    action_dim = 6  # [w_smooth, w_collision, w_time, corridor, max_vel, freq]
    
    trainer = DRLNavTrainer(state_dim, action_dim)
    
    # 模拟一些测试数据
    batch_size = 32
    batch_states = np.random.randn(batch_size, state_dim)
    batch_actions = np.random.rand(batch_size, action_dim)
    batch_rewards = np.random.randn(batch_size)
    batch_next_states = np.random.randn(batch_size, state_dim)
    
    # 训练一步
    losses = trainer.train_step(batch_states, batch_actions, batch_rewards, batch_next_states)
    print("训练损失:", losses)
    
    # 预测测试
    test_state = np.random.randn(state_dim)
    weights = trainer.predict(test_state)
    print("预测权重:", weights)
    
    # 导出模型
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    export_drl_nav_model(trainer, f"{output_dir}/drl_nav_weights.ts")
