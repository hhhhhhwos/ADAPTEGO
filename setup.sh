#!/bin/bash

# AdaptEgo 快速设置脚本
# 自动化配置环境依赖和初始化系统

set -e  # 遇到错误立即停止

echo "=================================================="
echo "🚁 AdaptEgo 自适应EGO-Planner系统"
echo "🔧 快速设置和环境检查"
echo "========================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查ROS环境
check_ros() {
    log_info "检查ROS环境..."
    
    if [ -z "$ROS_DISTRO" ]; then
        log_error "ROS未正确设置，请安装ROS Melodic或Noetic"
        exit 1
    fi
    
    log_success "ROS $ROS_DISTRO 环境正常"
    
    # 检查catkin工具
    if ! command -v catkin &> /dev/null; then
        log_warning "catkin工具未找到，尝试安装..."
        sudo apt update
        sudo apt install -y python3-catkin-tools
    fi
}

# 检查Python环境
check_python() {
    log_info "检查Python环境..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python3未安装"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log_success "Python版本: $PYTHON_VERSION"
    
    # 检查pip3
    if ! command -v pip3 &> /dev/null; then
        log_warning "pip3未找到，尝试安装..."
        sudo apt update
        sudo apt install -y python3-pip
    fi
}

# 安装Python依赖
install_python_deps() {
    log_info "检查并安装Python依赖..."
    
    REQUIRED_PACKAGES=(
        "torch>=1.8.0"
        "numpy>=1.19.0" 
        "pandas>=1.3.0"
        "matplotlib>=3.3.0"
        "scikit-learn>=0.24.0"
        "tensorboard>=2.7.0"
        "tqdm>=4.60.0"
    )
    
    MISSING_PACKAGES=()
    
    for package in "${REQUIRED_PACKAGES[@]}"; do
        package_name=$(echo $package | cut -d'>' -f1 | cut -d'=' -f1)
        
        if ! python3 -c "import $package_name" &> /dev/null; then
            MISSING_PACKAGES+=($package)
        else
            log_success "$package_name 已安装"
        fi
    done
    
    if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
        log_info "安装缺失的Python包: ${MISSING_PACKAGES[*]}"
        pip3 install --user "${MISSING_PACKAGES[@]}"
        
        # 检查安装是否成功
        for package in "${MISSING_PACKAGES[@]}"; do
            package_name=$(echo $package | cut -d'>' -f1 | cut -d'=' -f1)
            if python3 -c "import $package_name" &> /dev/null; then
                log_success "$package_name 安装成功"
            else
                log_error "$package_name 安装失败"
                exit 1
            fi
        done
    else
        log_success "所有Python依赖已安装"
    fi
}

# 检查ROS依赖
check_ros_deps() {
    log_info "检查ROS依赖包..."
    
    REQUIRED_ROS_PACKAGES=(
        "geometry_msgs"
        "nav_msgs" 
        "sensor_msgs"
        "std_msgs"
        "tf"
        "dynamic_reconfigure"
    )
    
    for package in "${REQUIRED_ROS_PACKAGES[@]}"; do
        if rospack find $package &> /dev/null; then
            log_success "$package 已安装"
        else
            log_warning "$package 未找到"
        fi
    done
}

# 创建目录结构
setup_directories() {
    log_info "创建必要的目录结构..."
    
    BASE_DIR="/home/$USER/fast_drone_ws/AdaptEgo"
    
    DIRECTORIES=(
        "$BASE_DIR/data"
        "$BASE_DIR/models" 
        "$BASE_DIR/results"
        "$BASE_DIR/logs"
        "$BASE_DIR/plots"
        "$BASE_DIR/src/adaptive_planning/models"
        "$BASE_DIR/src/adaptive_planning/config"
    )
    
    for dir in "${DIRECTORIES[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_success "创建目录: $dir"
        else
            log_info "目录已存在: $dir"
        fi
    done
}

# 设置权限
set_permissions() {
    log_info "设置脚本执行权限..."
    
    SCRIPT_FILES=(
        "run_sim_adaptive_demo.sh"
        "run_adaptive_demo.sh" 
        "run_simple_demo.sh"
        "shfiles/*.sh"
        "scripts/*.py"
    )
    
    cd "/home/$USER/fast_drone_ws/AdaptEgo"
    
    for pattern in "${SCRIPT_FILES[@]}"; do
        if ls $pattern 1> /dev/null 2>&1; then
            chmod +x $pattern
            log_success "设置执行权限: $pattern"
        fi
    done
}

# 编译ROS工作空间
build_workspace() {
    log_info "编译ROS工作空间..."
    
    cd "/home/$USER/fast_drone_ws"
    
    if [ -f "devel/setup.bash" ]; then
        source devel/setup.bash
    fi
    
    log_info "开始编译..."
    if catkin build -DCMAKE_BUILD_TYPE=Release; then
        log_success "编译完成"
    else
        log_error "编译失败，请检查错误信息"
        exit 1
    fi
    
    # 设置环境变量
    if ! grep -q "source ~/fast_drone_ws/devel/setup.bash" ~/.bashrc; then
        echo "source ~/fast_drone_ws/devel/setup.bash" >> ~/.bashrc
        log_success "已添加环境变量到 ~/.bashrc"
    fi
}

# 生成测试数据
generate_test_data() {
    log_info "生成测试用的合成专家数据..."
    
    cd "/home/$USER/fast_drone_ws/AdaptEgo"
    
    python3 -c "
import numpy as np
import os

print('生成合成专家演示数据...')
n_samples = 500

# 生成随机状态 [pos(3), goal(3), obs_density, avg_clearance, speed, goal_dist, complexity, battery]
np.random.seed(42)
states = np.random.randn(n_samples, 12) 

# 归一化部分特征到合理范围
states[:, 0:3] *= 5.0  # 位置 [-5, 5]
states[:, 3:6] *= 5.0  # 目标位置 [-5, 5]
states[:, 6] = np.abs(states[:, 6]) * 0.3  # 障碍密度 [0, 0.3]
states[:, 7] = np.abs(states[:, 7]) * 2.0 + 1.0  # 平均间隙 [1, 3]
states[:, 8] = np.abs(states[:, 8]) * 1.0  # 速度 [0, 1]
states[:, 9] = np.abs(states[:, 9]) * 8.0  # 目标距离 [0, 8]
states[:, 10] = np.abs(states[:, 10]) * 2.0  # 复杂度 [0, 2]
states[:, 11] = 0.8 + np.random.random(n_samples) * 0.2  # 电池 [0.8, 1.0]

# 生成对应的权重（基于启发式规则）
weights = np.zeros((n_samples, 6))
for i in range(n_samples):
    obs_density = states[i, 6]
    goal_dist = states[i, 9]
    current_speed = states[i, 8]
    
    # w_smooth: 障碍越多越需要平滑
    weights[i, 0] = 0.8 + obs_density * 1.2
    
    # w_collision: 障碍越多权重越大
    weights[i, 1] = 3.0 + obs_density * 6.0
    
    # w_time: 距离远时间权重小，距离近时间权重大
    weights[i, 2] = max(0.2, 1.0 - goal_dist * 0.08)
    
    # corridor_width: 障碍多时走廊窄
    weights[i, 3] = max(0.4, 1.2 - obs_density * 0.8)
    
    # max_velocity: 距离远速度大，障碍多速度小
    weights[i, 4] = min(2.8, 1.0 + goal_dist * 0.25 - obs_density * 1.5)
    weights[i, 4] = max(0.6, weights[i, 4])
    
    # replan_freq: 障碍多重规划频繁
    weights[i, 5] = 12.0 + obs_density * 15.0

# 90%成功率
success = np.random.choice([0, 1], n_samples, p=[0.1, 0.9])

os.makedirs('data', exist_ok=True)
np.savez_compressed('data/expert_demonstrations.npz',
                   states=states, weights=weights, success=success)

print(f'合成数据已保存: {n_samples} 样本')
print(f'成功率: {np.mean(success):.2%}')
print('数据保存位置: data/expert_demonstrations.npz')
"
    
    if [ $? -eq 0 ]; then
        log_success "测试数据生成完成"
    else
        log_warning "测试数据生成失败，但不影响主要功能"
    fi
}

# 运行系统测试
run_system_test() {
    log_info "运行系统集成测试..."
    
    cd "/home/$USER/fast_drone_ws/AdaptEgo"
    
    # 测试模块导入
    log_info "测试Python模块导入..."
    python3 -c "
import sys
sys.path.append('src/adaptive_planning/scripts')

try:
    from adaptive_training import WeightPredictorNetwork
    print('✓ adaptive_training 模块正常')
    
    import torch
    model = WeightPredictorNetwork(input_dim=12, output_dim=6)
    test_input = torch.randn(1, 12)
    output = model(test_input)
    assert output.shape == (1, 6)
    print('✓ 神经网络前向传播正常')
    
    import numpy as np
    if os.path.exists('data/expert_demonstrations.npz'):
        data = np.load('data/expert_demonstrations.npz')
        print(f'✓ 测试数据加载正常: {len(data[\"states\"])} 样本')
    
    print('✓ 所有模块测试通过')
    
except Exception as e:
    print(f'✗ 模块测试失败: {e}')
    sys.exit(1)
" 
    
    if [ $? -eq 0 ]; then
        log_success "系统测试通过"
    else
        log_error "系统测试失败"
        exit 1
    fi
}

# 显示使用指南
show_usage() {
    echo ""
    echo "========================================"
    echo "🎉 设置完成！接下来你可以："
    echo "========================================"
    echo ""
    echo "📖 查看文档："
    echo "   • 使用手册: cat USAGE_MANUAL.md"
    echo "   • 开发指南: cat DEVELOPER_GUIDE.md" 
    echo "   • 项目结构: cat PROJECT_STRUCTURE.md"
    echo ""
    echo "🚀 快速开始："
    echo "   • 仿真演示: ./run_sim_adaptive_demo.sh"
    echo "   • 数据收集: ./run_sim_adaptive_demo.sh (选择模式2)"
    echo "   • 模型训练: ./run_sim_adaptive_demo.sh (选择模式4)"
    echo ""
    echo "🔧 调试命令："
    echo "   • 检查ROS话题: rostopic list | grep adaptive"
    echo "   • 查看权重变化: rostopic echo /drone_0_adaptive_planning/weights"
    echo "   • 监控训练进度: tensorboard --logdir=runs/"
    echo ""
    echo "📁 重要目录："
    echo "   • 数据存储: ~/fast_drone_ws/AdaptEgo/data/"
    echo "   • 模型保存: ~/fast_drone_ws/AdaptEgo/models/" 
    echo "   • 结果输出: ~/fast_drone_ws/AdaptEgo/results/"
    echo ""
    echo "💡 提示: 记得重新打开终端或运行 'source ~/.bashrc' 来加载环境变量"
    echo "========================================"
}

# 主函数
main() {
    log_info "开始AdaptEgo系统设置..."
    
    # 基础环境检查
    check_ros
    check_python
    
    # 安装依赖
    install_python_deps
    check_ros_deps
    
    # 设置项目
    setup_directories
    set_permissions
    build_workspace
    
    # 生成测试数据
    generate_test_data
    
    # 系统测试
    run_system_test
    
    # 显示使用指南
    show_usage
    
    log_success "🎉 AdaptEgo 自适应EGO-Planner系统设置完成！"
}

# 脚本入口
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
