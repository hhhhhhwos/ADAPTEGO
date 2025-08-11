# AdaptEgo 自适应EGO-Planner系统使用手册

## 🚀 快速开始

### 系统要求
- Ubuntu 18.04/20.04
- ROS Melodic/Noetic  
- Python 3.6+
- CUDA支持的GPU (可选，用于训练加速)

### 安装步骤
```bash
# 1. 克隆项目 (如果还没有)
cd ~/catkin_ws/src
git clone https://github.com/ZJU-FAST-Lab/AdaptEgo.git

# 2. 安装Python依赖
pip3 install torch numpy pandas matplotlib scikit-learn tensorboard tqdm

# 3. 编译ROS工作空间
cd ~/catkin_ws
catkin build

# 4. 设置环境变量
source devel/setup.bash
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
```

## 📋 使用流程

### 1. 🎮 仿真演示 (推荐新手)

启动完整仿真系统：
```bash
cd AdaptEgo
./run_sim_adaptive_demo.sh
# 选择: 1) 完整仿真 + 自适应权重
```

**操作步骤：**
1. 等待仿真环境启动 (约10秒)
2. 在RViz中使用 `2D Nav Goal` 设置目标点
3. 观察无人机自主导航过程
4. 监控自适应权重实时变化

**关键话题监控：**
```bash
# 查看自适应权重变化
rostopic echo /drone_0_adaptive_planning/weights

# 查看无人机位置
rostopic echo /drone_0_visual_slam/odom

# 查看目标位置  
rostopic echo /move_base_simple/goal
```

### 2. 📊 专家数据收集

收集专家演示数据用于训练：
```bash
./run_sim_adaptive_demo.sh
# 选择: 2) 仅数据收集模式
```

**交互命令：**
- `s` - 开始新的episode
- `e` - 成功结束当前episode
- `f` - 失败结束当前episode  
- `save` - 保存所有收集的数据
- `quit` - 退出收集系统

**数据收集建议：**
- 收集多样化场景: 空旷区域、密集障碍、狭窄通道
- 每个episode尽量完成完整的导航任务
- 标记成功/失败状态以提高数据质量
- 建议收集至少100个成功episode

### 3. 🧠 模型训练

训练自适应权重预测模型：
```bash
./run_sim_adaptive_demo.sh  
# 选择: 4) 训练模式
```

**训练流程：**
1. 自动检查专家数据是否存在
2. 如无数据，可选择生成合成数据进行测试
3. 执行行为克隆训练 (约15-30分钟)
4. 导出TorchScript模型用于部署
5. 生成训练曲线和性能报告

**训练参数配置：**
```python
config = {
    'input_dim': 12,           # 状态特征维度
    'output_dim': 6,           # 权重参数数量  
    'hidden_dims': [128, 128, 64],  # 网络层结构
    'learning_rate': 1e-3,     # 学习率
    'batch_size': 64,          # 批大小
    'epochs': 200,             # 训练轮数
    'patience': 20             # 早停耐心值
}
```

### 4. 📈 性能评估

评估系统性能表现：
```bash
./run_sim_adaptive_demo.sh
# 选择: 3) 性能评估模式  
```

**评估指标：**
- **成功率**: 到达目标的比例
- **完成时间**: 平均导航用时
- **路径长度**: 实际飞行距离
- **轨迹平滑度**: 基于jerk计算
- **平均速度**: 飞行效率指标

## 🛠️ 高级配置

### 网络架构自定义

修改 `scripts/adaptive_training.py` 中的网络结构：
```python
class WeightPredictorNetwork(nn.Module):
    def __init__(self, input_dim=12, output_dim=6, hidden_dims=[128, 128, 64]):
        # 自定义网络层
        # 可以添加更多层、改变激活函数、增加批归一化等
```

### 状态特征扩展

在 `scripts/sim_weight_adapter.py` 中添加新特征：
```python
def compute_state_features(self):
    # 现有特征: 位置(3) + 目标(3) + 环境信息(6)
    # 可添加: 姿态信息、历史轨迹、传感器数据等
    new_feature = self.calculate_custom_feature()
    state = np.concatenate([existing_state, [new_feature]])
    return state
```

### 权重映射调整

修改权重参数范围和映射关系：
```python
self.weight_ranges = np.array([
    [0.1, 2.0],    # w_smooth - 平滑度权重
    [1.0, 10.0],   # w_collision - 碰撞权重
    [0.1, 1.0],    # w_time - 时间权重  
    [0.3, 1.5],    # corridor_width - 走廊宽度
    [0.5, 3.0],    # max_velocity - 最大速度
    [5.0, 30.0]    # replan_freq - 重规划频率
])
```

## 🔧 故障排除

### 常见问题

**1. 启动脚本权限错误**
```bash
chmod +x run_sim_adaptive_demo.sh
```

**2. Python模块导入失败**  
```bash
pip3 install --upgrade torch numpy pandas matplotlib
```

**3. ROS话题无数据**
```bash
# 检查节点状态
rosnode list | grep adaptive

# 检查话题连接
rostopic info /drone_0_adaptive_planning/weights
```

**4. 训练过程中内存不足**
```python
# 减小批大小
config['batch_size'] = 32

# 减少网络层数
config['hidden_dims'] = [64, 64]
```

**5. 仿真环境启动失败**
```bash
# 清理之前的进程
pkill -f "roslaunch\|roscore\|rosrun"

# 重新启动
./run_sim_adaptive_demo.sh
```

### 调试模式

启用详细日志输出：
```bash
export ROS_LOG_LEVEL=DEBUG
./run_sim_adaptive_demo.sh
```

查看实时权重变化：
```bash
rostopic echo /drone_0_adaptive_planning/weights | grep -A6 "data:"
```

## 📁 文件结构说明

### 重要脚本文件
- `run_sim_adaptive_demo.sh` - 主启动脚本
- `scripts/adaptive_training.py` - 训练管道
- `scripts/data_collection_eval.py` - 数据收集与评估
- `scripts/sim_weight_adapter.py` - 权重适配器
- `launch/sim_adaptive_planning.launch` - ROS启动配置

### 数据文件位置
- `data/expert_demonstrations.npz` - 专家演示数据
- `models/adaptive_weights.ts` - 训练好的模型
- `results/evaluation_results.csv` - 评估结果

### 日志和输出
- `runs/` - TensorBoard日志
- `~/sim_expert_data/` - 仿真数据输出
- `plots/` - 训练曲线和可视化

## 🎯 使用建议

### 论文写作角度
1. **对比实验**: 运行固定权重 vs 自适应权重的对比
2. **消融实验**: 测试不同网络结构和特征组合的效果
3. **泛化能力**: 在不同障碍环境中测试模型表现
4. **实时性分析**: 测量权重预测的计算延迟

### 项目扩展方向
1. **多机协同**: 扩展到多无人机自适应规划
2. **真机验证**: 移植到实际无人机平台
3. **强化学习**: 加入在线学习能力
4. **传感器融合**: 集成更多传感器信息

## 📚 参考资料

- [EGO-Planner论文](https://arxiv.org/abs/2008.08835)
- [Fast-Drone-250原项目](https://github.com/ZJU-FAST-Lab/Fast-Drone-250)
- [PyTorch官方文档](https://pytorch.org/docs/)
- [ROS教程](http://wiki.ros.org/ROS/Tutorials)

---
如有问题，请检查 `PROJECT_STRUCTURE.md` 或创建 GitHub Issue。
