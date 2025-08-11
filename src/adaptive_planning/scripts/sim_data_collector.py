#!/usr/bin/env python3
"""
仿真数据收集器 - 基于PyTorchStepByStep训练范式
收集专家演示数据用于训练自适应权重预测器
"""

import rospy
import os
import pickle
import numpy as np
from datetime import datetime
from collections import deque

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray
import tf.transformations as tf_trans

class SimDataCollector:
    """
    仿真环境专家数据收集器
    收集状态-动作对用于监督学习训练
    """
    def __init__(self):
        rospy.init_node('sim_data_collector', anonymous=True)
        
        # 参数
        self.drone_id = rospy.get_param('~drone_id', 0)
        self.output_dir = rospy.get_param('~output_dir', f'{os.environ["HOME"]}/sim_expert_data')
        self.collection_rate = rospy.get_param('~collection_rate', 10.0)
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 数据缓存
        self.data_buffer = []
        self.current_episode = 0
        self.episode_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'metadata': {}
        }
        
        # 状态变量
        self.current_pose = None
        self.current_vel = None
        self.current_goal = None
        self.planned_path = None
        self.current_weights = None
        self.local_pointcloud = None
        
        # 历史记录（用于计算奖励）
        self.pose_history = deque(maxlen=50)  # 5秒历史
        self.goal_history = deque(maxlen=10)
        
        # ROS接口
        self.setup_ros_interface()
        
        # 定时保存
        self.save_timer = rospy.Timer(rospy.Duration(30.0), self.save_episode_data)
        
        rospy.loginfo(f"[SimDataCollector] 仿真数据收集器已启动 - 输出目录: {self.output_dir}")
        
    def setup_ros_interface(self):
        """设置ROS话题接口"""
        # 订阅者
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.goal_sub = rospy.Subscriber('/goal', PoseStamped, self.goal_callback)  
        self.weights_sub = rospy.Subscriber('/adaptive_weights', Float32MultiArray, self.weights_callback)
        self.path_sub = rospy.Subscriber('/planned_traj', Path, self.path_callback)
        self.pointcloud_sub = rospy.Subscriber('/local_map', PointCloud2, self.pointcloud_callback)
        
        # 数据收集定时器
        self.collect_timer = rospy.Timer(rospy.Duration(1.0 / self.collection_rate), self.collect_data_point)
        
    def odom_callback(self, msg):
        """里程计回调"""
        self.current_pose = msg.pose.pose
        self.current_vel = msg.twist.twist
        
        # 记录位置历史
        pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        self.pose_history.append({
            'position': pos,
            'timestamp': rospy.Time.now().to_sec()
        })
        
    def goal_callback(self, msg):
        """目标回调"""
        if self.current_goal != msg.pose:
            # 新目标 - 开始新的episode
            self.start_new_episode()
            
        self.current_goal = msg.pose
        
        # 记录目标历史
        goal_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        self.goal_history.append({
            'position': goal_pos,
            'timestamp': rospy.Time.now().to_sec()
        })
        
    def weights_callback(self, msg):
        """权重回调"""
        if len(msg.data) == 6:
            self.current_weights = np.array(msg.data)
            
    def path_callback(self, msg):
        """规划路径回调"""
        self.planned_path = msg
        
    def pointcloud_callback(self, msg):
        """点云回调"""
        self.local_pointcloud = msg
        
    def start_new_episode(self):
        """开始新的数据收集episode"""
        if len(self.episode_data['states']) > 0:
            # 保存当前episode
            self.save_current_episode()
            
        # 重置episode数据
        self.current_episode += 1
        self.episode_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'metadata': {
                'episode_id': self.current_episode,
                'start_time': rospy.Time.now().to_sec(),
                'drone_id': self.drone_id
            }
        }
        
        rospy.loginfo(f"[SimDataCollector] 开始新episode: {self.current_episode}")
        
    def extract_state_vector(self):
        """提取状态向量"""
        if not all([self.current_pose, self.current_vel, self.current_goal]):
            return None
            
        # 当前位置和姿态
        pos = np.array([
            self.current_pose.position.x,
            self.current_pose.position.y,
            self.current_pose.position.z
        ])
        
        quat = [
            self.current_pose.orientation.x,
            self.current_pose.orientation.y,
            self.current_pose.orientation.z,
            self.current_pose.orientation.w
        ]
        euler = tf_trans.euler_from_quaternion(quat)
        
        # 速度
        vel = np.array([
            self.current_vel.linear.x,
            self.current_vel.linear.y,
            self.current_vel.linear.z
        ])
        
        ang_vel = np.array([
            self.current_vel.angular.x,
            self.current_vel.angular.y,
            self.current_vel.angular.z
        ])
        
        # 目标相对位置
        goal_pos = np.array([
            self.current_goal.position.x,
            self.current_goal.position.y,
            self.current_goal.position.z
        ])
        
        rel_goal = goal_pos - pos
        goal_distance = np.linalg.norm(rel_goal)
        goal_direction = rel_goal / (goal_distance + 1e-6)
        
        # 局部环境特征（简化）
        obstacle_density = self.estimate_obstacle_density()
        path_complexity = self.estimate_path_complexity()
        
        # 组合状态向量
        state = np.concatenate([
            pos,                    # 3
            euler,                  # 3
            vel,                    # 3
            ang_vel,                # 3
            goal_direction,         # 3
            [goal_distance],        # 1
            [obstacle_density],     # 1
            [path_complexity]       # 1
        ])  # 总共18维
        
        return state
        
    def estimate_obstacle_density(self):
        """估计局部障碍物密度"""
        # 简化实现 - 基于点云密度
        if self.local_pointcloud is None:
            return 0.0
            
        # 这里应该解析PointCloud2，简化为随机值
        return np.random.uniform(0.0, 1.0)
        
    def estimate_path_complexity(self):
        """估计路径复杂度"""
        if self.planned_path is None or len(self.planned_path.poses) < 2:
            return 0.0
            
        # 计算路径曲率作为复杂度指标
        poses = self.planned_path.poses
        total_curvature = 0.0
        
        for i in range(1, len(poses) - 1):
            p1 = np.array([poses[i-1].pose.position.x, poses[i-1].pose.position.y])
            p2 = np.array([poses[i].pose.position.x, poses[i].pose.position.y])
            p3 = np.array([poses[i+1].pose.position.x, poses[i+1].pose.position.y])
            
            # 计算角度变化
            v1 = p2 - p1
            v2 = p3 - p2
            
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle_change = np.arccos(cos_angle)
                total_curvature += angle_change
                
        return total_curvature / max(len(poses) - 2, 1)
        
    def calculate_reward(self):
        """计算专家奖励信号"""
        if len(self.pose_history) < 2:
            return 0.0
            
        # 基于多个指标的复合奖励
        reward = 0.0
        
        # 1. 目标接近奖励
        if self.current_goal:
            current_pos = self.pose_history[-1]['position']
            goal_pos = np.array([
                self.current_goal.position.x,
                self.current_goal.position.y,
                self.current_goal.position.z
            ])
            
            goal_dist = np.linalg.norm(goal_pos - current_pos)
            reward += -goal_dist * 0.1  # 距离越远奖励越低
            
            # 到达目标奖励
            if goal_dist < 0.5:
                reward += 10.0
                
        # 2. 平滑性奖励
        if len(self.pose_history) >= 3:
            positions = [entry['position'] for entry in list(self.pose_history)[-3:]]
            
            # 计算加速度（平滑性指标）
            acc = positions[2] - 2*positions[1] + positions[0]
            acc_magnitude = np.linalg.norm(acc)
            reward += -acc_magnitude * 0.5  # 加速度越小越好
            
        # 3. 碰撞惩罚（基于点云）
        collision_risk = self.estimate_collision_risk()
        reward += -collision_risk * 5.0
        
        # 4. 效率奖励
        if len(self.pose_history) >= 2:
            dt = self.pose_history[-1]['timestamp'] - self.pose_history[-2]['timestamp']
            if dt > 0:
                velocity_magnitude = np.linalg.norm(
                    self.pose_history[-1]['position'] - self.pose_history[-2]['position']
                ) / dt
                reward += velocity_magnitude * 0.1  # 鼓励合理速度
                
        return reward
        
    def estimate_collision_risk(self):
        """估计碰撞风险"""
        # 简化实现
        obstacle_density = self.estimate_obstacle_density()
        return obstacle_density  # 障碍物密度越高，碰撞风险越大
        
    def collect_data_point(self, event):
        """收集单个数据点"""
        if not all([self.current_pose, self.current_goal, self.current_weights is not None]):
            return
            
        # 提取状态
        state = self.extract_state_vector()
        if state is None:
            return
            
        # 动作（权重）
        action = self.current_weights.copy()
        
        # 奖励
        reward = self.calculate_reward()
        
        # 添加到episode数据
        self.episode_data['states'].append(state)
        self.episode_data['actions'].append(action)
        self.episode_data['rewards'].append(reward)
        
        # 更新元数据
        self.episode_data['metadata']['last_update'] = rospy.Time.now().to_sec()
        self.episode_data['metadata']['data_points'] = len(self.episode_data['states'])
        
    def save_current_episode(self):
        """保存当前episode数据"""
        if len(self.episode_data['states']) == 0:
            return
            
        # 转换为numpy数组
        episode_data_np = {
            'states': np.array(self.episode_data['states']),
            'actions': np.array(self.episode_data['actions']),
            'rewards': np.array(self.episode_data['rewards']),
            'metadata': self.episode_data['metadata']
        }
        
        # 保存文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"episode_{self.current_episode:04d}_{timestamp}.pkl"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(episode_data_np, f)
                
            rospy.loginfo(f"[SimDataCollector] 已保存episode {self.current_episode}: {filename}")
            rospy.loginfo(f"  - 数据点数: {len(episode_data_np['states'])}")
            rospy.loginfo(f"  - 平均奖励: {np.mean(episode_data_np['rewards']):.3f}")
            
        except Exception as e:
            rospy.logerr(f"[SimDataCollector] 保存episode失败: {e}")
            
    def save_episode_data(self, event):
        """定时保存episode数据"""
        if len(self.episode_data['states']) > 0:
            self.save_current_episode()
            
        # 输出统计信息
        total_files = len([f for f in os.listdir(self.output_dir) if f.endswith('.pkl')])
        rospy.loginfo(f"[SimDataCollector] 统计 - 总episode数: {total_files}")

if __name__ == '__main__':
    try:
        collector = SimDataCollector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
