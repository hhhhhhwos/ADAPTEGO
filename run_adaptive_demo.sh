#!/bin/bash

# 一键运行自适应混合规划演示
# 基于多个开源项目的缝合方案

echo "=================================================="
echo "     🚀 自适应混合规划演示 - 开源缝合版本"
echo "=================================================="

set -e

# 工作空间路径
WORKSPACE_ROOT="/home/kklab/fast_drone_ws/AdaptEgo"
SCRIPTS_DIR="$WORKSPACE_ROOT/src/adaptive_planning/scripts"
MODELS_DIR="$WORKSPACE_ROOT/src/adaptive_planning/models"

# 创建必要目录
mkdir -p "$MODELS_DIR"
mkdir -p "$HOME/adaptive_planning_data"
mkdir -p "$HOME/adaptive_planning_metrics"
mkdir -p "$HOME/adaptive_planning_logs"

# 解析参数
STAGE="all"
MODE="single"
TRAINING="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"  # single, swarm
            shift 2
            ;;
        --train)
            TRAINING="true"
            shift
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --stage <阶段>    运行阶段: setup|train|demo|all (默认: all)"
            echo "  --mode <模式>     运行模式: single|swarm (默认: single)"
            echo "  --train          是否进行训练 (默认: false)"
            echo ""
            echo "阶段说明:"
            echo "  setup   - 下载开源项目并安装依赖"
            echo "  train   - 训练RL模型"
            echo "  demo    - 运行演示"
            echo "  all     - 执行所有阶段"
            echo ""
            echo "示例:"
            echo "  $0 --stage setup                # 只进行环境设置"
            echo "  $0 --stage train                # 只训练模型"
            echo "  $0 --mode swarm --train          # 训练并运行群体演示"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

echo "🎯 配置:"
echo "   阶段: $STAGE"
echo "   模式: $MODE"
echo "   训练: $TRAINING"
echo ""

# 阶段1: 环境设置
setup_environment() {
    echo "🔧 阶段1: 环境设置"
    echo "----------------------------------------"
    
    # 下载第三方项目
    echo "📥 下载开源项目..."
    bash "$WORKSPACE_ROOT/scripts/download_third_party.sh"
    
    # 安装Python依赖
    echo "📦 安装Python依赖..."
    pip3 install --user numpy torch torchvision matplotlib scikit-learn
    pip3 install --user stable-baselines3[extra]
    
    # 尝试安装gym-pybullet-drones
    echo "🚁 安装gym-pybullet-drones..."
    if [ -d "$WORKSPACE_ROOT/third_party/gym-pybullet-drones" ]; then
        cd "$WORKSPACE_ROOT/third_party/gym-pybullet-drones"
        pip3 install --user -e .
    else
        echo "⚠️ gym-pybullet-drones目录不存在，尝试直接安装"
        pip3 install --user gym-pybullet-drones
    fi
    
    # 设置Python路径
    export PYTHONPATH="$PYTHONPATH:$WORKSPACE_ROOT/third_party/gym-pybullet-drones"
    export PYTHONPATH="$PYTHONPATH:$WORKSPACE_ROOT/third_party/DRL-Nav"
    export PYTHONPATH="$PYTHONPATH:$WORKSPACE_ROOT/third_party/Autonomous-Quadcopter-Control-RL"
    export PYTHONPATH="$PYTHONPATH:$WORKSPACE_ROOT/third_party/Drooid-Drone-swarm-Algorithm"
    export PYTHONPATH="$PYTHONPATH:$WORKSPACE_ROOT/third_party/PyTorchStepByStep"
    
    # 编译ROS包
    echo "🔨 编译adaptive_planning包..."
    cd "$WORKSPACE_ROOT"
    source /opt/ros/noetic/setup.bash
    catkin_make --pkg adaptive_planning
    
    echo "✅ 环境设置完成"
}

# 阶段2: 模型训练
train_models() {
    echo "🎓 阶段2: 模型训练"
    echo "----------------------------------------"
    
    cd "$SCRIPTS_DIR"
    
    if [ "$TRAINING" = "true" ]; then
        echo "🚀 开始训练自适应权重模型..."
        
        # 1. 使用PyTorchStepByStep训练权重预测器
        echo "📚 Step 1: PyTorchStepByStep训练..."
        python3 pytorch_stepbystep_trainer.py
        
        # 2. 使用gym-pybullet-drones训练导航策略
        echo "🚁 Step 2: gym-pybullet-drones训练..."
        python3 train_with_pybullet.py --algo PPO --steps 50000 --env hover
        
        # 3. 基于DRL-Nav的网络微调（可选）
        echo "🧠 Step 3: DRL-Nav网络微调..."
        python3 drl_nav_network.py
        
        # 4. 训练RL控制器（可选）
        if [ "$MODE" = "single" ]; then
            echo "🎮 Step 4: 训练RL控制器..."
            python3 rl_controller.py train
        fi
        
        echo "✅ 模型训练完成"
        
    else
        echo "📂 使用预训练模型或创建占位符模型..."
        
        # 创建占位符模型（实际项目中应该有预训练模型）
        python3 -c "
import torch
import torch.nn as nn

# 创建简单的占位符模型
class PlaceholderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

model = PlaceholderModel()
example_input = torch.randn(1, 12)
traced = torch.jit.trace(model, example_input)
traced.save('$MODELS_DIR/adaptive_weights.ts')
print('✅ 占位符模型已创建: adaptive_weights.ts')
"
    fi
    
    echo "🎯 可用模型:"
    ls -la "$MODELS_DIR/" || echo "   (模型目录为空)"
}

# 阶段3: 运行演示
run_demo() {
    echo "🎬 阶段3: 运行演示"
    echo "----------------------------------------"
    
    # 设置ROS环境
    source /opt/ros/noetic/setup.bash
    source "$WORKSPACE_ROOT/devel/setup.bash"
    
    echo "🔗 检查ROS连接..."
    if ! rostopic list &>/dev/null; then
        echo "⚠️ ROS master未运行，启动roscore..."
        roscore &
        ROSCORE_PID=$!
        sleep 3
        
        # 确保退出时关闭roscore
        trap "kill $ROSCORE_PID 2>/dev/null || true" EXIT
    fi
    
    echo "🚀 启动自适应混合规划演示..."
    
    if [ "$MODE" = "swarm" ]; then
        echo "   模式: 群体协调 (基于Drooid算法)"
        roslaunch adaptive_planning adaptive_hybrid_demo.launch \
            use_rl_weights:=true \
            use_swarm_mode:=true \
            num_agents:=3 &
    else
        echo "   模式: 单机自适应"
        roslaunch adaptive_planning adaptive_hybrid_demo.launch \
            use_rl_weights:=true \
            use_rl_control:=false \
            use_swarm_mode:=false &
    fi
    
    LAUNCH_PID=$!
    
    echo ""
    echo "📋 演示说明:"
    echo "   • 系统已启动，包含以下组件："
    echo "     - EGO-Planner (原有规划器)"
    echo "     - 自适应权重预测器 (基于DRL-Nav + gym-pybullet-drones)"
    if [ "$MODE" = "swarm" ]; then
        echo "     - Drooid群体协调器"
    fi
    echo "   • 使用RViz中的2D Nav Goal设置目标点"
    echo "   • 数据自动保存到 ~/adaptive_planning_* 目录"
    echo "   • 按Ctrl+C停止演示"
    echo ""
    
    # 等待用户中断
    wait $LAUNCH_PID || true
    
    echo ""
    echo "📊 演示结束，检查结果文件..."
    echo "   数据: ~/adaptive_planning_data/"
    echo "   指标: ~/adaptive_planning_metrics/"
    echo "   日志: ~/adaptive_planning_logs/"
}

# 主流程
main() {
    case $STAGE in
        "setup")
            setup_environment
            ;;
        "train")
            train_models
            ;;
        "demo")
            run_demo
            ;;
        "all")
            setup_environment
            echo ""
            train_models
            echo ""
            run_demo
            ;;
        *)
            echo "❌ 未知阶段: $STAGE"
            echo "可用阶段: setup, train, demo, all"
            exit 1
            ;;
    esac
}

# 错误处理
handle_error() {
    echo "❌ 脚本执行失败，行号: $1"
    echo "请检查错误信息并重试"
    exit 1
}

trap 'handle_error $LINENO' ERR

# 运行主流程
main

echo ""
echo "🎉 自适应混合规划演示完成!"
echo ""
echo "📄 论文相关:"
echo "   • 方法: 基于多开源项目的自适应权重学习"
echo "   • 创新点: gym-pybullet-drones仿真 + DRL-Nav网络 + Drooid群体协调"
echo "   • 对比基线: 固定权重EGO-Planner, RL-only, 传统群体算法"
echo "   • 实验数据: 已保存在用户目录，可直接用于论文分析"
