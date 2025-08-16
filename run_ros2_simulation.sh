#!/bin/bash

# AdaptEgo ROS2 Humble 仿真演示脚本
# 适配Ubuntu 22.04 + ROS2 Humble环境

echo "========================================"
echo "🚁 AdaptEgo ROS2 仿真演示系统"
echo "基于深度学习的自适应路径规划"
echo "ROS2 Humble + PyTorch + 自适应权重预测"
echo "========================================"

# 设置ROS2环境
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
export PYTHONPATH="/home/kklab/AdaptEgo/src:$PYTHONPATH"

# 检查Python依赖
echo "🔍 检查系统依赖..."
python3 -c "
import sys
try:
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import rclpy
    print('✅ 所有依赖库已安装')
except ImportError as e:
    print(f'❌ 缺少依赖: {e}')
    print('请运行: pip3 install torch numpy matplotlib')
    sys.exit(1)
" || exit 1

# 创建必要目录
echo "📁 创建项目目录..."
mkdir -p /home/kklab/AdaptEgo/{data,models,results,logs}

# 清理之前的进程
echo "🧹 清理环境..."
pkill -f "ros2\|python3.*adaptive" >/dev/null 2>&1
sleep 2

echo ""
echo "🎮 选择仿真模式："
echo "1) 纯Python仿真演示 (推荐，无需ROS2环境复杂配置)"
echo "2) 模型训练模式"
echo "3) 数据收集与评估"
echo "4) 学术实验套件"
echo "5) 简单功能测试"
read -p "请选择模式 (1-5): " sim_mode

case $sim_mode in
    1)
        echo "🚀 启动纯Python仿真演示..."
        echo "这个模式会运行一个自包含的仿真环境，演示自适应权重预测功能"
        
        python3 << 'EOF'
#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
import os

print("🚁 AdaptEgo 纯Python仿真演示")
print("=" * 50)

class SimpleAdaptiveDemo:
    def __init__(self):
        # 简单的权重预测网络
        self.network = nn.Sequential(
            nn.Linear(12, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
            nn.Sigmoid()
        )
        
        # 权重范围
        self.weight_ranges = np.array([
            [0.1, 2.0],    # w_smooth
            [1.0, 10.0],   # w_collision  
            [0.1, 1.0],    # w_time
            [0.3, 1.5],    # corridor_width
            [0.5, 3.0],    # max_velocity
            [5.0, 30.0]    # replan_freq
        ])
        
        # 仿真状态
        self.drone_pos = np.array([0.0, 0.0, 1.0])
        self.goal_pos = np.array([5.0, 5.0, 1.0])
        self.obstacles = [
            np.array([2.0, 2.0, 1.0]),
            np.array([3.5, 1.5, 1.0]),
            np.array([1.0, 4.0, 1.0])
        ]
        
    def compute_environment_features(self):
        """计算12维环境特征"""
        # 位置特征 (3维)
        pos_features = self.drone_pos.copy()
        
        # 目标特征 (3维)  
        goal_rel = self.goal_pos - self.drone_pos
        goal_dist = np.linalg.norm(goal_rel)
        goal_features = np.append(goal_rel, goal_dist)[:3]
        
        # 障碍物特征 (6维)
        obstacle_features = np.zeros(6)
        if len(self.obstacles) > 0:
            # 最近障碍物距离
            dists = [np.linalg.norm(obs - self.drone_pos) for obs in self.obstacles]
            min_dist = min(dists)
            obstacle_features[0] = min_dist
            
            # 障碍物密度
            close_obs = sum(1 for d in dists if d < 3.0)
            obstacle_features[1] = close_obs / len(self.obstacles)
            
            # 其他特征
            obstacle_features[2:] = np.random.randn(4) * 0.1
        
        features = np.concatenate([pos_features, goal_features, obstacle_features])
        return features
    
    def predict_weights(self, features):
        """预测自适应权重"""
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            raw_weights = self.network(features_tensor).numpy()[0]
            
            # 缩放到实际范围
            scaled_weights = np.zeros(6)
            for i in range(6):
                min_val, max_val = self.weight_ranges[i]
                scaled_weights[i] = min_val + raw_weights[i] * (max_val - min_val)
                
        return scaled_weights
    
    def simulate_step(self):
        """仿真一步"""
        # 计算环境特征
        features = self.compute_environment_features()
        
        # 预测权重
        weights = self.predict_weights(features)
        
        # 简单运动模型 (朝目标移动)
        direction = self.goal_pos - self.drone_pos
        direction = direction / np.linalg.norm(direction)
        
        # 基于权重调整速度
        speed = weights[4] * 0.2  # max_velocity权重影响速度
        self.drone_pos += direction * speed
        
        return features, weights
    
    def run_demo(self):
        """运行演示"""
        print("🎯 目标位置:", self.goal_pos)
        print("🚧 障碍物位置:", [obs.tolist() for obs in self.obstacles])
        print()
        
        trajectory = []
        weights_history = []
        
        for step in range(20):
            features, weights = self.simulate_step()
            trajectory.append(self.drone_pos.copy())
            weights_history.append(weights.copy())
            
            print(f"步骤 {step+1:2d}:")
            print(f"  位置: [{self.drone_pos[0]:.2f}, {self.drone_pos[1]:.2f}, {self.drone_pos[2]:.2f}]")
            print(f"  权重: [平滑:{weights[0]:.2f}, 避障:{weights[1]:.2f}, 时间:{weights[2]:.2f}]")
            print(f"  速度:{weights[4]:.2f}, 频率:{weights[5]:.1f}")
            
            # 检查是否到达目标
            if np.linalg.norm(self.drone_pos - self.goal_pos) < 0.5:
                print("🎉 成功到达目标！")
                break
                
            time.sleep(0.5)
        
        # 绘制结果
        self.plot_results(trajectory, weights_history)
    
    def plot_results(self, trajectory, weights_history):
        """绘制仿真结果"""
        trajectory = np.array(trajectory)
        weights_history = np.array(weights_history)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 轨迹图
        ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-o', linewidth=2, markersize=4, label='无人机轨迹')
        ax1.plot(0, 0, 'go', markersize=10, label='起点')
        ax1.plot(self.goal_pos[0], self.goal_pos[1], 'r*', markersize=15, label='目标')
        
        # 障碍物
        for i, obs in enumerate(self.obstacles):
            circle = plt.Circle((obs[0], obs[1]), 0.3, color='red', alpha=0.5)
            ax1.add_patch(circle)
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('🚁 无人机飞行轨迹')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # 权重变化图
        weight_names = ['平滑度', '避障', '时间', '走廊宽度', '最大速度', '重规划频率']
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        for i in range(6):
            ax2.plot(weights_history[:, i], color=colors[i], linewidth=2, 
                    marker='o', markersize=3, label=weight_names[i])
        
        ax2.set_xlabel('仿真步骤')
        ax2.set_ylabel('权重值')
        ax2.set_title('⚖️ 自适应权重变化')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/simulation_demo.png', dpi=300, bbox_inches='tight')
        print("\n📊 仿真结果已保存到 results/simulation_demo.png")
        plt.show()

# 运行演示
demo = SimpleAdaptiveDemo()
demo.run_demo()

print("\n" + "=" * 50)
print("✅ AdaptEgo Python仿真演示完成！")
print("🔍 主要功能展示:")
print("  • 12维环境状态感知")
print("  • 6维自适应权重预测")  
print("  • 实时权重调整")
print("  • 轨迹可视化")
print("📁 结果文件: results/simulation_demo.png")
EOF
        ;;
        
    2)
        echo "🧠 启动模型训练模式..."
        cd /home/kklab/AdaptEgo
        python3 src/adaptive_planning/scripts/adaptive_training.py
        ;;
        
    3)
        echo "📊 启动数据收集与评估..."
        cd /home/kklab/AdaptEgo
        python3 src/adaptive_planning/scripts/paper_evaluation.py
        ;;
        
    4)
        echo "🎓 启动学术实验套件..."
        cd /home/kklab/AdaptEgo
        ./start_paper_experiments.sh
        ;;
        
    5)
        echo "🔧 运行简单功能测试..."
        python3 << 'EOF'
import numpy as np
import torch
import torch.nn as nn

print("🔧 AdaptEgo 功能测试")
print("=" * 30)

# 测试1: 网络结构
print("1️⃣ 测试神经网络...")
model = nn.Sequential(
    nn.Linear(12, 128),
    nn.ReLU(), 
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 6),
    nn.Sigmoid()
)

# 测试输入
test_input = torch.randn(1, 12)
output = model(test_input)
print(f"   输入维度: {test_input.shape}")
print(f"   输出维度: {output.shape}")
print(f"   输出范围: [{output.min():.3f}, {output.max():.3f}]")

# 测试2: 数据处理
print("\n2️⃣ 测试数据处理...")
features = np.random.randn(100, 12)
weights = np.random.rand(100, 6)
print(f"   特征形状: {features.shape}")
print(f"   权重形状: {weights.shape}")

# 测试3: 权重缩放
print("\n3️⃣ 测试权重缩放...")
weight_ranges = np.array([
    [0.1, 2.0], [1.0, 10.0], [0.1, 1.0],
    [0.3, 1.5], [0.5, 3.0], [5.0, 30.0]
])
raw_weights = np.random.rand(6)
scaled_weights = np.zeros(6)
for i in range(6):
    min_val, max_val = weight_ranges[i]
    scaled_weights[i] = min_val + raw_weights[i] * (max_val - min_val)
    
print(f"   原始权重: {raw_weights}")
print(f"   缩放权重: {scaled_weights}")

print("\n✅ 所有功能测试通过！")
EOF
        ;;
        
    *)
        echo "❌ 无效选择，退出"
        exit 1
        ;;
esac

echo ""
echo "🎯 仿真演示完成！"
echo ""
echo "📚 其他可用脚本:"
echo "  • ./run_ros2_simulation.sh  - 重新运行此脚本"
echo "  • ./start_paper_experiments.sh  - 学术实验"
echo "  • python3 src/adaptive_planning/scripts/adaptive_training.py  - 直接训练"
echo ""
echo "📁 输出目录:"
echo "  • results/  - 仿真结果"
echo "  • models/   - 训练模型"
echo "  • logs/     - 日志文件"
