#!/bin/bash

# AdaptEgo ROS2 Humble 环境配置脚本
# 适配当前Ubuntu 22.04 + ROS2 Humble环境

echo "🚀 配置AdaptEgo for ROS2 Humble..."

# 1. 设置ROS2环境
source /opt/ros/humble/setup.bash
echo "export ROS_DOMAIN_ID=0" >> ~/.bashrc
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# 2. 安装ROS2必要包
sudo apt update
sudo apt install -y \
    ros-humble-desktop \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-joint-state-publisher \
    ros-humble-robot-state-publisher \
    ros-humble-rviz2 \
    ros-humble-tf2-tools \
    ros-humble-geometry-msgs \
    ros-humble-sensor-msgs \
    ros-humble-std-msgs \
    python3-colcon-common-extensions \
    python3-rosdep

# 3. 安装Python依赖
pip3 install --user \
    torch torchvision torchaudio \
    numpy matplotlib pandas scikit-learn \
    tensorboard tqdm opencv-python \
    rclpy geometry-msgs sensor-msgs std-msgs nav-msgs

# 4. 创建ROS2工作空间结构
cd /home/kklab/AdaptEgo
if [ ! -d "install" ]; then
    mkdir -p install
fi

# 5. 将ROS1包转换为ROS2兼容格式
echo "🔄 适配ROS2包结构..."

# 6. 创建colcon构建脚本
cat > build_ros2.sh << 'EOF'
#!/bin/bash
source /opt/ros/humble/setup.bash
cd /home/kklab/AdaptEgo
colcon build --packages-select adaptive_planning
source install/setup.bash
echo "✅ ROS2构建完成！"
EOF

chmod +x build_ros2.sh

# 7. 创建ROS2启动脚本
cat > run_ros2_demo.sh << 'EOF'
#!/bin/bash
source /opt/ros/humble/setup.bash
cd /home/kklab/AdaptEgo
source install/setup.bash

echo "🚁 启动AdaptEgo ROS2演示..."
echo "选择运行模式："
echo "1) 纯Python训练演示"
echo "2) 仿真数据收集" 
echo "3) 模型训练和评估"
echo "4) 学术实验套件"
read -p "请选择 (1-4): " choice

case $choice in
    1)
        echo "▶️  启动纯Python训练演示..."
        python3 src/adaptive_planning/scripts/adaptive_training.py
        ;;
    2)
        echo "▶️  启动数据收集模式..."
        python3 src/adaptive_planning/scripts/sim_data_collector.py
        ;;
    3)
        echo "▶️  启动训练和评估..."
        python3 src/adaptive_planning/scripts/paper_evaluation.py
        ;;
    4)
        echo "▶️  启动学术实验..."
        ./start_paper_experiments.sh
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac
EOF

chmod +x run_ros2_demo.sh

echo "✅ ROS2环境配置完成！"
echo ""
echo "🎯 使用方法："
echo "1. 运行: ./run_ros2_demo.sh"
echo "2. 或直接训练: python3 src/adaptive_planning/scripts/adaptive_training.py"
echo ""
echo "📚 项目结构："
echo "├── src/adaptive_planning/     # 核心算法"
echo "├── paper/                     # 学术论文"
echo "├── experiments/               # 实验数据"  
echo "└── models/                    # 训练模型"
