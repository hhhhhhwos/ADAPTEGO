#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 gym-pybullet-drones 的自适应权重学习环境
直接使用 gym-pybullet-drones 的代码，添加权重预测任务
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import argparse

# 添加第三方路径
THIRD_PARTY_PATH = os.path.join(os.path.dirname(__file__), '../../third_party')
sys.path.append(os.path.join(THIRD_PARTY_PATH, 'gym-pybullet-drones'))
sys.path.append(os.path.join(THIRD_PARTY_PATH, 'DRL-Nav'))

try:
    from gym_pybullet_drones.envs.HoverAviary import HoverAviary
    from gym_pybullet_drones.envs.FlyThruGateAviary import FlyThruGateAviary
    from gym_pybullet_drones.utils.enums import DroneModel, Physics
    print("✓ gym-pybullet-drones 导入成功")
except ImportError as e:
    print(f"❌ 无法导入 gym-pybullet-drones: {e}")
    print("请先运行: pip install gym-pybullet-drones")
    sys.exit(1)

class AdaptiveWeightsPredictionEnv(gym.Wrapper):
    """
    包装 gym-pybullet-drones 环境，用于学习自适应权重预测
    """
    def __init__(self, base_env):
        super().__init__(base_env)
        
        # 扩展观测空间：原始obs + 环境特征
        original_obs_dim = base_env.observation_space.shape[0]
        # 添加环境特征：障碍密度、距离目标、复杂度等
        env_features_dim = 6  
        new_obs_dim = original_obs_dim + env_features_dim
        
        # 修改观测空间
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(new_obs_dim,), dtype=np.float32
        )
        
        # 动作空间：预测6个权重参数
        # [w_smooth, w_collision, w_time, corridor_width, max_velocity, replan_freq]
        self.action_space = gym.spaces.Box(
            low=np.array([0.1, 1.0, 0.1, 0.3, 0.5, 5.0]),
            high=np.array([2.0, 10.0, 1.0, 1.5, 3.0, 30.0]),
            dtype=np.float32
        )
        
        # 内部状态
        self.step_count = 0
        self.start_pos = None
        self.goal_pos = np.array([2.0, 0.0, 1.0])  # 默认目标
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.step_count = 0
        self.start_pos = obs[:3] if len(obs) >= 3 else np.zeros(3)
        
        # 随机化目标点
        self.goal_pos = np.random.uniform([-5, -5, 0.5], [5, 5, 2.0])
        
        # 扩展观测
        extended_obs = self._extend_observation(obs)
        return extended_obs, info
        
    def step(self, action):
        # action是预测的权重，这里我们需要转换为实际的控制命令
        # 简化处理：权重影响控制策略
        weights = action
        
        # 基于权重生成控制指令（这里简化为朝向目标的速度）
        current_pos = self.get_current_position()
        direction = self.goal_pos - current_pos
        distance = np.linalg.norm(direction)
        
        if distance > 0.1:
            # 根据权重调整速度
            max_speed = weights[4] if len(weights) > 4 else 1.0  # max_velocity weight
            speed = min(max_speed, distance * 0.5)
            control_action = (direction / distance) * speed
            # 补齐到环境需要的动作维度
            if hasattr(self.env, 'action_space'):
                action_dim = self.env.action_space.shape[0]
                if action_dim == 4:  # RPM控制
                    control_action = np.array([0.5, 0.5, 0.5, 0.5]) + np.random.normal(0, 0.1, 4)
                    control_action = np.clip(control_action, 0, 1)
        else:
            control_action = np.zeros(self.env.action_space.shape[0])
            
        # 执行原始环境步进
        obs, reward, done, truncated, info = self.env.step(control_action)
        
        # 重新计算奖励（基于权重预测质量）
        custom_reward = self._calculate_weight_prediction_reward(weights, obs)
        
        # 扩展观测
        extended_obs = self._extend_observation(obs)
        
        self.step_count += 1
        return extended_obs, custom_reward, done, truncated, info
    
    def get_current_position(self):
        """获取当前位置（从环境状态）"""
        # 这里需要根据具体环境调整
        try:
            if hasattr(self.env, '_getDroneStateVector'):
                state = self.env._getDroneStateVector(0)
                return state[:3]  # 位置
        except:
            pass
        return self.start_pos if self.start_pos is not None else np.zeros(3)
    
    def _extend_observation(self, obs):
        """扩展观测空间，添加环境特征"""
        current_pos = obs[:3] if len(obs) >= 3 else np.zeros(3)
        
        # 计算环境特征
        goal_distance = np.linalg.norm(self.goal_pos - current_pos)
        obstacle_density = self._estimate_obstacle_density(current_pos)
        path_complexity = min(1.0, goal_distance / 10.0 + obstacle_density)
        current_speed = np.linalg.norm(obs[10:13]) if len(obs) >= 13 else 0.0
        avg_clearance = max(0.5, 2.0 - obstacle_density)
        battery_level = max(0.5, 1.0 - self.step_count / 1000.0)  # 简化的电量模型
        
        env_features = np.array([
            goal_distance, obstacle_density, path_complexity,
            current_speed, avg_clearance, battery_level
        ], dtype=np.float32)
        
        # 拼接原始观测和环境特征
        extended_obs = np.concatenate([obs, env_features])
        return extended_obs.astype(np.float32)
    
    def _estimate_obstacle_density(self, position):
        """估计当前位置的障碍密度"""
        # 简化的障碍密度估计
        # 可以基于环境的实际障碍信息
        x, y, z = position
        density = 0.0
        
        # 简单的基于位置的密度模型
        if abs(x) > 3 or abs(y) > 3:  # 边界区域密度高
            density += 0.3
        if x > 0 and y > 0:  # 某个象限密度高
            density += 0.4
            
        return min(1.0, density + np.random.normal(0, 0.1))
    
    def _calculate_weight_prediction_reward(self, weights, obs):
        """计算权重预测的奖励"""
        current_pos = obs[:3] if len(obs) >= 3 else np.zeros(3)
        
        # 基础奖励：到达目标
        goal_distance = np.linalg.norm(self.goal_pos - current_pos)
        reach_reward = 10.0 if goal_distance < 0.3 else -goal_distance * 0.1
        
        # 权重合理性奖励
        w_smooth, w_collision, w_time = weights[0], weights[1], weights[2]
        corridor_width, max_velocity = weights[3], weights[4]
        
        # 权重应该适应环境
        obstacle_density = self._estimate_obstacle_density(current_pos)
        
        # 障碍密集时，避障权重应该高
        collision_appropriateness = w_collision * obstacle_density
        # 速度权重应该合理
        speed_appropriateness = 1.0 - abs(max_velocity - (2.0 - obstacle_density))
        
        weight_reward = (collision_appropriateness + speed_appropriateness) * 0.1
        
        # 平滑度奖励（避免权重剧烈变化）
        smoothness_reward = -np.var(weights) * 0.01
        
        total_reward = reach_reward + weight_reward + smoothness_reward
        return total_reward

def create_training_env(env_id="hover", gui=False):
    """创建训练环境"""
    if env_id == "hover":
        base_env = HoverAviary(
            drone_model=DroneModel.CF2X,
            initial_xyzs=np.array([[0, 0, 1]]),
            initial_rpys=np.array([[0, 0, 0]]),
            physics=Physics.PYB_GND_DRAG_DW,
            neighbourhood_radius=10,
            freq=240,
            aggregate_phy_steps=1,
            gui=gui,
            record=False,
            obstacles=True,  # 添加障碍物
            user_debug_gui=False
        )
    else:
        # 可以添加其他环境类型
        base_env = HoverAviary(gui=gui)
    
    # 包装成自适应权重预测环境
    env = AdaptiveWeightsPredictionEnv(base_env)
    return env

def train_adaptive_weights_model():
    """训练自适应权重预测模型"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="hover", help="Environment type")
    parser.add_argument("--algo", default="PPO", choices=["PPO", "SAC"])
    parser.add_argument("--steps", type=int, default=100000, help="Training steps")
    parser.add_argument("--gui", action="store_true", help="Show GUI")
    parser.add_argument("--output_dir", default="models", help="Output directory")
    args = parser.parse_args()
    
    print(f"🚀 开始训练自适应权重模型...")
    print(f"   环境: {args.env}")
    print(f"   算法: {args.algo}")
    print(f"   训练步数: {args.steps}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建环境
    env = create_training_env(args.env, gui=args.gui)
    
    # 创建评估环境
    eval_env = create_training_env(args.env, gui=False)
    
    # 选择算法
    if args.algo == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log=f"{args.output_dir}/tensorboard/"
        )
    else:  # SAC
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=100000,
            batch_size=256,
            gamma=0.99,
            tau=0.02,
            tensorboard_log=f"{args.output_dir}/tensorboard/"
        )
    
    # 评估回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{args.output_dir}/best_model",
        log_path=f"{args.output_dir}/eval_logs",
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    # 开始训练
    model.learn(
        total_timesteps=args.steps,
        callback=eval_callback,
        progress_bar=True
    )
    
    # 保存最终模型
    final_model_path = f"{args.output_dir}/adaptive_weights_final"
    model.save(final_model_path)
    
    # 导出为TorchScript（用于ROS部署）
    try:
        export_torchscript(model, env.observation_space.shape[0], 
                          env.action_space.shape[0], 
                          f"{args.output_dir}/adaptive_weights.ts")
        print(f"✅ 模型已导出为 TorchScript: {args.output_dir}/adaptive_weights.ts")
    except Exception as e:
        print(f"⚠️ TorchScript导出失败: {e}")
    
    print(f"✅ 训练完成！模型保存至: {final_model_path}")
    return final_model_path

def export_torchscript(model, obs_dim, act_dim, output_path):
    """导出TorchScript模型用于ROS部署"""
    class TorchScriptPolicy(torch.nn.Module):
        def __init__(self, sb3_model):
            super().__init__()
            # 提取SB3模型的网络
            if hasattr(sb3_model.policy, 'mlp_extractor'):
                # PPO
                self.features_extractor = sb3_model.policy.features_extractor
                self.mlp_extractor = sb3_model.policy.mlp_extractor
                self.action_net = sb3_model.policy.action_net
            else:
                # SAC
                self.actor = sb3_model.policy.actor
            
            self.obs_dim = obs_dim
            self.act_dim = act_dim
            
        def forward(self, obs):
            obs = obs.float()
            
            if hasattr(self, 'actor'):
                # SAC
                mean_actions = self.actor.mu(obs)
                return mean_actions
            else:
                # PPO  
                features = self.features_extractor(obs)
                latent_pi, _ = self.mlp_extractor(features)
                actions = self.action_net(latent_pi)
                return actions
    
    # 创建包装器
    ts_policy = TorchScriptPolicy(model)
    ts_policy.eval()
    
    # 创建示例输入
    dummy_obs = torch.randn(1, obs_dim)
    
    # 追踪并保存
    traced_model = torch.jit.trace(ts_policy, dummy_obs)
    traced_model.save(output_path)

if __name__ == "__main__":
    train_adaptive_weights_model()
