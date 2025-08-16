#!/bin/bash

echo "========================================"
echo "🚁 AdaptEgo 仿真平台启动器"
echo "基于ROS + RViz + SO3仿真器"
echo "========================================"

# 检查环境
if [ -z "$ROS_DISTRO" ]; then
    echo "❌ ROS环境未设置！"
    echo "请运行: source /opt/ros/noetic/setup.bash"
    exit 1
fi

# 设置工作目录
cd /home/kklab/AdaptEgo

# 清理环境
echo "🧹 清理旧进程..."
pkill -f "roslaunch\|roscore\|rosrun" 2>/dev/null
sleep 2

echo ""
echo "🎮 AdaptEgo 仿真模式选择："
echo ""
echo "1️⃣  基础仿真 - 适合初学者"
echo "   ├─ SO3无人机仿真器" 
echo "   ├─ 随机森林地图"
echo "   ├─ RViz 3D可视化"
echo "   └─ 手动目标点设置"
echo ""
echo "2️⃣  智能仿真 - 展示AI能力"
echo "   ├─ 上述基础功能"
echo "   ├─ 自适应权重预测"
echo "   ├─ 实时环境感知"
echo "   └─ 智能路径优化"
echo ""
echo "3️⃣  研究模式 - 数据收集"
echo "   ├─ 完整功能"
echo "   ├─ 训练数据收集"
echo "   ├─ 性能评估"
echo "   └─ 模型训练支持"
echo ""
echo "4️⃣  退出"

read -p "请选择模式 [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "🚀 启动基础仿真平台..."
        echo "📝 操作指南："
        echo "  1. 等待RViz启动完成"
        echo "  2. 在RViz中点击 '2D Nav Goal'"
        echo "  3. 在地图上点击设置目标点"
        echo "  4. 观察无人机自主导航"
        echo ""
        roslaunch adaptive_planning sim_adaptive_planning.launch \
            use_adaptive_weights:=false \
            collect_data:=false
        ;;
    2)
        echo ""
        echo "🧠 启动智能自适应仿真..."
        echo "📊 将显示："
        echo "  ✓ 实时权重调整过程"
        echo "  ✓ 环境复杂度分析"
        echo "  ✓ AI决策可视化"
        echo ""
        roslaunch adaptive_planning sim_adaptive_planning.launch \
            use_adaptive_weights:=true \
            collect_data:=false
        ;;
    3)
        echo ""
        echo "🔬 启动研究数据收集模式..."
        echo "💾 将记录："
        echo "  ✓ 轨迹数据"
        echo "  ✓ 环境状态"
        echo "  ✓ 权重变化"
        echo "  ✓ 性能指标"
        echo ""
        roslaunch adaptive_planning sim_adaptive_planning.launch \
            use_adaptive_weights:=true \
            collect_data:=true
        ;;
    4)
        echo "👋 再见！"
        exit 0
        ;;
    *)
        echo "❌ 无效选择，请重新运行"
        exit 1
        ;;
esac
