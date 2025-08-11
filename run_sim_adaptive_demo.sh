#!/bin/bash

# AdaptEgo 自适应EGO-Planner仿真演示
# 集成深度学习权重预测的智能无人机导航系统
# 基于EGO-Planner + 自适应权重预测

echo "========================================"
echo "AdaptEgo 自适应EGO-Planner仿真演示"
echo "基于深度学习的智能权重自适应方案"
echo "版本: Enhanced with AI-Powered Adaptive Planning"
echo "========================================"

# 设置环境变量
export DRONE_ID=0
export USE_ADAPTIVE=true
export COLLECT_DATA=true
export ROS_PYTHON_VERSION=3

# 检查依赖
echo "检查系统依赖..."
python3 -c "import torch, numpy, pandas, matplotlib, sklearn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ 缺少Python依赖，请安装："
    echo "pip3 install torch numpy pandas matplotlib scikit-learn"
    exit 1
fi

# 检查模型文件
MODEL_DIR="/home/kklab/fast_drone_ws/AdaptEgo/src/adaptive_planning/models"
if [ ! -d "$MODEL_DIR" ]; then
    mkdir -p "$MODEL_DIR"
    echo "📁 已创建模型目录: $MODEL_DIR"
fi

# 清理之前的进程
echo "清理环境..."
pkill -f "roslaunch\|roscore\|rosrun" >/dev/null 2>&1
sleep 2

# 启动roscore
echo "启动ROS核心..."
roscore &
ROSCORE_PID=$!
sleep 3

# 选择启动模式
echo ""
echo "选择启动模式："
echo "1) 完整仿真 + 自适应权重 (推荐)"
echo "2) 仅数据收集模式"
echo "3) 性能评估模式"  
echo "4) 训练模式"
read -p "请选择模式 (1-4): " startup_mode

case $startup_mode in
    1)
        echo "启动完整仿真环境..."
        
        # 启动仿真环境
        roslaunch adaptive_planning sim_adaptive_planning.launch \
            drone_id:=$DRONE_ID \
            use_adaptive_weights:=$USE_ADAPTIVE \
            collect_data:=$COLLECT_DATA \
            init_x:=0.0 \
            init_y:=0.0 \
            init_z:=1.0 &
        
        LAUNCH_PID=$!
        sleep 8
        
        echo "========================================"
        echo "仿真环境已启动！"
        echo "- 无人机ID: $DRONE_ID"
        echo "- 自适应权重: $USE_ADAPTIVE"  
        echo "- 数据收集: $COLLECT_DATA"
        echo "========================================"
        
        # 选择任务模式
        echo "选择任务模式："
        echo "1) 手动目标点 (RViz 2D Nav Goal)"
        echo "2) 自动waypoint巡航"
        echo "3) 随机目标挑战"
        read -p "请选择 (1-3): " task_choice
        
        case $task_choice in
            1)
                echo "手动模式 - 请在RViz中设置目标点"
                echo "使用 '2D Nav Goal' 工具点击设置目标"
                ;;
            2)
                echo "启动自动waypoint任务..."
                sleep 2
                rosrun adaptive_planning sim_auto_mission.py _mission_type:=waypoint_tour _auto_start:=true &
                ;;
            3)
                echo "启动随机目标挑战..."
                sleep 2
                rosrun adaptive_planning sim_auto_mission.py _mission_type:=random_goals _auto_start:=true &
                ;;
        esac
        ;;
        
    2)
        echo "启动数据收集模式..."
        
        # 启动基础仿真（无自适应权重）
        roslaunch adaptive_planning sim_adaptive_planning.launch \
            drone_id:=$DRONE_ID \
            use_adaptive_weights:=false \
            collect_data:=true &
        
        sleep 5
        
        # 启动交互式数据收集
        echo "启动交互式专家演示收集..."
        python3 /home/kklab/fast_drone_ws/AdaptEgo/src/adaptive_planning/scripts/data_collection_eval.py collect &
        ;;
        
    3)
        echo "启动性能评估模式..."
        
        # 启动仿真环境
        roslaunch adaptive_planning sim_adaptive_planning.launch \
            drone_id:=$DRONE_ID \
            use_adaptive_weights:=$USE_ADAPTIVE \
            collect_data:=false &
            
        sleep 5
        
        # 启动性能评估
        python3 /home/kklab/fast_drone_ws/AdaptEgo/src/adaptive_planning/scripts/data_collection_eval.py evaluate &
        ;;
        
    4)
        echo "启动训练模式..."
        
        # 检查是否有专家数据
        EXPERT_DATA="/home/kklab/fast_drone_ws/AdaptEgo/data/expert_demonstrations.npz"
        if [ ! -f "$EXPERT_DATA" ]; then
            echo "❌ 没有找到专家演示数据!"
            echo "请先运行数据收集模式收集专家演示数据"
            echo "或者生成合成训练数据"
            
            read -p "是否生成合成训练数据? (y/n): " gen_synthetic
            if [ "$gen_synthetic" = "y" ]; then
                echo "生成合成训练数据..."
                python3 -c "
import numpy as np
import os

# 生成合成专家数据用于测试
print('生成合成专家演示数据...')
n_samples = 1000

# 生成随机状态
states = np.random.randn(n_samples, 12) * 2.0
# 生成对应的权重（基于简单规则）
weights = np.zeros((n_samples, 6))
for i in range(n_samples):
    obs_density = abs(states[i, 6])  # 障碍密度
    goal_dist = abs(states[i, 9])    # 目标距离
    
    # 简单的专家规则
    weights[i, 0] = 1.0 + obs_density  # w_smooth
    weights[i, 1] = 3.0 + obs_density * 5.0  # w_collision
    weights[i, 2] = max(0.2, 1.0 - goal_dist * 0.1)  # w_time
    weights[i, 3] = max(0.3, 1.0 - obs_density * 0.5)  # corridor_width
    weights[i, 4] = min(3.0, 1.0 + goal_dist * 0.3)  # max_velocity  
    weights[i, 5] = 10.0 + obs_density * 10.0  # replan_freq

success = np.random.choice([0, 1], n_samples, p=[0.1, 0.9])  # 90%成功率

os.makedirs('data', exist_ok=True)
np.savez_compressed('data/expert_demonstrations.npz',
                   states=states, weights=weights, success=success)
print('合成数据已保存到 data/expert_demonstrations.npz')
"
            else
                echo "请先收集专家演示数据后再运行训练"
                kill $ROSCORE_PID
                exit 1
            fi
        fi
        
        echo "开始训练自适应权重预测器..."
        cd /home/kklab/fast_drone_ws/AdaptEgo/src/adaptive_planning/scripts/
        python3 adaptive_training.py
        
        echo "训练完成，模型已保存"
        kill $ROSCORE_PID
        exit 0
        ;;
        
    *)
        echo "无效选择，使用默认完整模式"
        startup_mode=1
        ;;
esac

echo ""
echo "监控信息："
echo "- 查看权重变化: rostopic echo /drone_${DRONE_ID}_adaptive_planning/weights"
echo "- 查看位置状态: rostopic echo /drone_${DRONE_ID}_visual_slam/odom"
echo "- 专家数据目录: ~/sim_expert_data/"
echo "- 模型目录: $MODEL_DIR"
echo ""
echo "有用的命令："
echo "- 重新训练模型: ./run_sim_adaptive_demo.sh (选择模式4)"
echo "- 收集专家数据: ./run_sim_adaptive_demo.sh (选择模式2)" 
echo "- 性能评估: ./run_sim_adaptive_demo.sh (选择模式3)"
echo ""

# 等待用户终止
echo "演示运行中... 按 Ctrl+C 停止"
trap 'echo "正在停止..."; kill $LAUNCH_PID $ROSCORE_PID 2>/dev/null; pkill -f "roslaunch\|roscore\|rosrun" >/dev/null 2>&1; exit 0' SIGINT

wait
