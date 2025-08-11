#!/usr/bin/env python3
"""
仿真环境权重适配器 - 基于DRL-Nav + gym-pybullet-drones架构
与SO3四旋翼仿真器集成的权重自适应系统
"""

import rospy
import numpy as np
import torch
import torch.nn as nn
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
import tf.transformations as tf_trans

# 基于DRL-Nav的网络架构（适配仿真环境）
class WeightPredictorNetwork(nn.Module):
    """
    自适应权重预测网络 - 结合你的架构设计
    输入：环境状态特征（位置、速度、目标、障碍信息等）
    输出：EGO-Planner的自适应权重参数
    """
    def __init__(self, input_dim=12, output_dim=6, hidden_dims=[128, 128, 64]):
        super().__init__()
        
        # 构建多层网络
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
            
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())  # 输出[0,1]，后续会缩放
        
        self.network = nn.Sequential(*layers)
        
        # 权重范围定义（对应EGO-Planner参数）
        self.weight_ranges = torch.tensor([
            [0.1, 2.0],    # w_smooth
            [1.0, 10.0],   # w_collision  
            [0.1, 1.0],    # w_time
            [0.3, 1.5],    # corridor_width
            [0.5, 3.0],    # max_velocity
            [5.0, 30.0]    # replan_freq
        ])
        
    def forward(self, state_features):
        """前向传播"""
        normalized_weights = self.network(state_features)
        
        # 反归一化到实际权重范围
        min_vals = self.weight_ranges[:, 0]
        max_vals = self.weight_ranges[:, 1]
        
        # 广播到正确的形状
        if normalized_weights.dim() == 2:  # 批处理
            min_vals = min_vals.unsqueeze(0).expand_as(normalized_weights)
            max_vals = max_vals.unsqueeze(0).expand_as(normalized_weights)
            
        weights = min_vals + normalized_weights * (max_vals - min_vals)
        return weights

class SimWeightAdapter:
    """
    仿真环境权重适配器
    结合gym-pybullet-drones训练数据 + DRL-Nav架构
    """
    def __init__(self):
        rospy.init_node('sim_weight_adapter', anonymous=True)
        
        # 参数配置 - 结合你的参数设计
        self.drone_id = rospy.get_param('~drone_id', 0)
        self.model_path = rospy.get_param('~model_path', '')
        self.update_rate = rospy.get_param('~update_rate', 10.0)  # Hz
        self.sensing_range = rospy.get_param('~sensing_range', 5.0)
        self.goal_tolerance = rospy.get_param('~goal_tolerance', 0.5)  # m
        self.use_learned_weights = rospy.get_param('~use_learned_weights', True)
        
        # 状态变量 - 扩展你的状态定义
        self.current_pos = np.zeros(3)
        self.current_vel = np.zeros(3)
        self.goal_pos = np.array([2.0, 0.0, 1.0])  # 默认目标
        self.obstacle_density = 0.0
        self.avg_clearance = 5.0
        self.battery_level = 1.0
        self.have_odom = False
        self.local_pointcloud = None
        
        # 网络模型 - 使用你的网络架构
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = WeightPredictorNetwork(
            input_dim=12,  # 扩展的状态特征
            output_dim=6,  # EGO-Planner权重参数
            hidden_dims=[128, 128, 64]
        ).to(self.device)
        
        # 加载预训练模型
        self.load_model()
        
        # ROS接口 - 适配仿真环境话题
        self.setup_ros_interface()
        
        # 默认权重（基于你的设计和EGO-Planner原始设置）
        self.default_weights = {
            'w_smooth': 1.5,      # 平滑度权重
            'w_collision': 5.0,   # 碰撞权重
            'w_time': 0.5,        # 时间权重
            'corridor_width': 0.8,  # 走廊宽度
            'max_velocity': 2.0,    # 最大速度
            'replan_freq': 15.0     # 重规划频率
        }
        
        rospy.loginfo(f"[SimWeightAdapter] 仿真权重适配器已启动 - 无人机{self.drone_id}")
        rospy.loginfo(f"[SimWeightAdapter] 使用设备: {self.device}")
        rospy.loginfo(f"[SimWeightAdapter] 模型路径: {self.model_path}")
        
    def setup_ros_interface(self):
        """设置ROS话题接口 - 适配仿真环境"""
        # 订阅者 - 兼容多种话题格式
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.goal_sub = rospy.Subscriber('/goal', PoseStamped, self.goal_callback)
        
        # 激光扫描（如果有的话）
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback, queue_size=1)
        
        # 点云（如果有的话）
        self.pointcloud_sub = rospy.Subscriber('/local_pointcloud', PointCloud2, self.pointcloud_callback)
        
        # 发布者
        self.weights_pub = rospy.Publisher('/adaptive_weights', Float32MultiArray, queue_size=1)
        self.weights_viz_pub = rospy.Publisher('/adaptive_weights_viz', MarkerArray, queue_size=1)
        
        # 定时器
        self.update_timer = rospy.Timer(rospy.Duration(1.0 / self.update_rate), self.update_weights)
        
    def load_model(self):
        """加载预训练模型"""
        if self.model_path and rospy.os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.network.load_state_dict(checkpoint['model_state_dict'])
                self.network.eval()
                rospy.loginfo(f"[SimWeightAdapter] 成功加载模型: {self.model_path}")
            except Exception as e:
                rospy.logwarn(f"[SimWeightAdapter] 模型加载失败: {e}，使用默认权重")
                self.use_learned_weights = False
        else:
            rospy.logwarn(f"[SimWeightAdapter] 模型文件不存在: {self.model_path}，使用默认权重")
            self.use_learned_weights = False
    
    def odom_callback(self, msg):
        """里程计回调 - 适配你的状态管理"""
        # 提取位置
        self.current_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        
        # 提取速度
        self.current_vel = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])
        
        self.have_odom = True
        
    def goal_callback(self, msg):
        """目标点回调 - 适配你的目标管理"""
        self.goal_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        rospy.loginfo(f"[SimWeightAdapter] 新目标设定: {self.goal_pos}")
        
    def scan_callback(self, msg):
        """激光扫描回调 - 处理障碍物信息"""
        try:
            ranges = np.array(msg.ranges)
            valid_ranges = ranges[np.isfinite(ranges) & (ranges > 0.1)]
            
            if len(valid_ranges) > 0:
                self.avg_clearance = np.mean(valid_ranges)
                # 计算障碍密度（近距离障碍的比例）
                close_obstacles = np.sum(valid_ranges < 1.0)
                self.obstacle_density = close_obstacles / len(valid_ranges)
            else:
                self.avg_clearance = 5.0
                self.obstacle_density = 0.0
        except Exception as e:
            rospy.logwarn(f"[SimWeightAdapter] 激光数据处理失败: {e}")
            
    def pointcloud_callback(self, msg):
        """点云回调（简化处理）"""
        self.local_pointcloud = msg
        
    def compute_state_features(self):
        """
        计算环境状态特征 - 结合你的特征工程设计
        返回12维状态向量用于权重预测
        """
        if not self.have_odom:
            return None
            
        # 计算基础特征
        goal_distance = np.linalg.norm(self.goal_pos - self.current_pos)
        current_speed = np.linalg.norm(self.current_vel)
        
        # 路径复杂度评估（基于障碍密度和目标距离）
        path_complexity = self.obstacle_density * np.log(goal_distance + 1.0)
        
        # 构建状态向量 [12维] - 与你的设计保持一致
        state = np.array([
            # 当前位置 (3)
            self.current_pos[0], self.current_pos[1], self.current_pos[2],
            # 目标位置 (3)  
            self.goal_pos[0], self.goal_pos[1], self.goal_pos[2],
            # 环境特征 (6)
            self.obstacle_density,      # 障碍物密度
            self.avg_clearance,         # 平均间隙
            current_speed,              # 当前速度
            goal_distance,              # 目标距离
            path_complexity,            # 路径复杂度
            self.battery_level          # 电池电量
        ], dtype=np.float32)
        
        return state
        
    def predict_adaptive_weights(self):
        """使用神经网络预测自适应权重"""
        if not self.use_learned_weights:
            return self.default_weights
            
        try:
            # 计算状态特征
            state = self.compute_state_features()
            if state is None:
                return self.default_weights
                
            # 转换为张量
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # 预测权重
            with torch.no_grad():
                predicted_weights = self.network(state_tensor)
                weights_np = predicted_weights.cpu().numpy().flatten()
                
            # 转换为字典格式（对应EGO-Planner参数）
            adaptive_weights = {
                'w_smooth': float(weights_np[0]),        # 平滑度权重
                'w_collision': float(weights_np[1]),     # 碰撞权重  
                'w_time': float(weights_np[2]),          # 时间权重
                'corridor_width': float(weights_np[3]),  # 走廊宽度
                'max_velocity': float(weights_np[4]),    # 最大速度
                'replan_freq': float(weights_np[5])      # 重规划频率
            }
            
            return adaptive_weights
            
        except Exception as e:
            rospy.logwarn(f"[SimWeightAdapter] 权重预测失败: {e}，使用默认权重")
            return self.default_weights
            
    def update_weights(self, event):
        """定时更新权重 - 结合你的更新逻辑"""
        if not self.have_odom:
            return
            
        # 预测自适应权重
        adaptive_weights = self.predict_adaptive_weights()
        
        # 发布权重消息 - 按照你的权重顺序
        weights_msg = Float32MultiArray()
        weights_msg.data = [
            adaptive_weights['w_smooth'],
            adaptive_weights['w_collision'], 
            adaptive_weights['w_time'],
            adaptive_weights['corridor_width'],
            adaptive_weights['max_velocity'],
            adaptive_weights['replan_freq']
        ]
        self.weights_pub.publish(weights_msg)
        
        # 可视化
        self.publish_weights_visualization(adaptive_weights)
        
        # 日志输出（降低频率）
        if rospy.get_time() % 2.0 < 0.1:  # 每2秒输出一次
            rospy.loginfo(f"[SimWeightAdapter] 当前权重: smooth={adaptive_weights['w_smooth']:.2f}, "
                        f"collision={adaptive_weights['w_collision']:.2f}, "
                        f"速度上限={adaptive_weights['max_velocity']:.2f}m/s")
        
    def publish_weights_visualization(self, weights):
        """发布权重可视化"""
        marker_array = MarkerArray()
        
        # 创建文本标记显示权重值
        for i, (name, value) in enumerate(weights.items()):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = rospy.Time.now()
            marker.id = i
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            
            # 位置
            marker.pose.position.x = -5.0
            marker.pose.position.y = 5.0 - i * 0.3
            marker.pose.position.z = 2.0
            marker.pose.orientation.w = 1.0
            
            # 内容
            marker.text = f"{name}: {value:.2f}"
            marker.scale.z = 0.2
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            
            marker_array.markers.append(marker)
            
        self.weights_viz_pub.publish(marker_array)

if __name__ == '__main__':
    try:
        adapter = SimWeightAdapter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
