#!/bin/bash
# 下载并集成开源项目的脚本

echo "=================================================="
echo "        下载开源项目代码"
echo "=================================================="

WORKSPACE_ROOT="/home/kklab/fast_drone_ws/AdaptEgo"
THIRD_PARTY_DIR="$WORKSPACE_ROOT/third_party"

# 创建第三方代码目录
mkdir -p "$THIRD_PARTY_DIR"
cd "$THIRD_PARTY_DIR"

echo "🔄 克隆 gym-pybullet-drones..."
if [ ! -d "gym-pybullet-drones" ]; then
    git clone https://github.com/utiasDSL/gym-pybullet-drones.git
else
    echo "   ✓ 已存在，跳过"
fi

echo "🔄 克隆 Autonomous-Quadcopter-Control-RL..."
if [ ! -d "Autonomous-Quadcopter-Control-RL" ]; then
    git clone https://github.com/IvLabs/Autonomous-Quadcopter-Control-RL.git
else
    echo "   ✓ 已存在，跳过"
fi

echo "🔄 克隆 DRL-Nav..."
if [ ! -d "DRL-Nav" ]; then
    git clone https://github.com/bilalkabas/DRL-Nav.git
else
    echo "   ✓ 已存在，跳过"
fi

echo "🔄 克隆 Drooid-Drone-swarm-Algorithm..."
if [ ! -d "Drooid-Drone-swarm-Algorithm" ]; then
    git clone https://github.com/Humancyyborg/Drooid-Drone-swarm-Algorithm.git
else
    echo "   ✓ 已存在，跳过"
fi

echo "🔄 克隆 PyTorchStepByStep..."
if [ ! -d "PyTorchStepByStep" ]; then
    git clone https://github.com/dvgodoy/PyTorchStepByStep.git
else
    echo "   ✓ 已存在，跳过"
fi

echo ""
echo "✅ 所有开源项目已下载到 $THIRD_PARTY_DIR"
echo ""
echo "📋 接下来的步骤："
echo "   1. 安装 gym-pybullet-drones 依赖"
echo "   2. 提取并适配关键代码到你的项目"
echo "   3. 创建混合训练脚本"
