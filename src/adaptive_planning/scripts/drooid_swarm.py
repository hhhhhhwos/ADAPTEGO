#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 Drooid-Drone-swarm-Algorithm 的多机编队控制
直接使用Drooid的核心算法
"""

import rospy
import numpy as np
from geometry_msgs.msg import TwistStamped, PoseStamped, Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray, Int32MultiArray
import threading
import os
import sys

# 添加Drooid路径
THIRD_PARTY_PATH = os.path.join(os.path.dirname(__file__), '../../third_party')
DROOID_PATH = os.path.join(THIRD_PARTY_PATH, 'Drooid-Drone-swarm-Algorithm')
sys.path.append(DROOID_PATH)

class DrooidSwarmAgent:
    """
    基于Drooid算法的群体智能代理
    实现分布式编队和避碰
    """
    def __init__(self, agent_id, total_agents, namespace=""):
        self.agent_id = agent_id
        self.total_agents = total_agents
        self.namespace = namespace
        
        # 代理状态
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.goal_position = np.zeros(3)
        self.has_odom = False
        
        # 邻居信息
        self.neighbors = {}  # {agent_id: {'pos': np.array, 'vel': np.array, 'time': float}}
        self.neighbor_lock = threading.Lock()
        
        # Drooid算法参数
        self.sensing_radius = rospy.get_param('~sensing_radius', 3.0)
        self.safe_distance = rospy.get_param('~safe_distance', 1.0)
        self.formation_gain = rospy.get_param('~formation_gain', 1.0)
        self.cohesion_gain = rospy.get_param('~cohesion_gain', 0.5)
        self.separation_gain = rospy.get_param('~separation_gain', 2.0)
        self.alignment_gain = rospy.get_param('~alignment_gain', 0.3)
        self.goal_gain = rospy.get_param('~goal_gain', 1.5)
        self.max_speed = rospy.get_param('~max_speed', 2.0)
        
        # 编队参数
        self.formation_type = rospy.get_param('~formation_type', 'line')  # line, circle, triangle
        self.formation_size = rospy.get_param('~formation_size', 2.0)
        
        # ROS接口
        self.setup_ros_interface()
        
        rospy.loginfo(f"[Drooid] Agent {agent_id} initialized")
    
    def setup_ros_interface(self):
        """设置ROS接口"""
        # 订阅自己的位置信息
        self.odom_sub = rospy.Subscriber(
            f'{self.namespace}/mavros/local_position/odom',
            Odometry,
            self.odom_callback,
            queue_size=10
        )
        
        # 订阅目标位置
        self.goal_sub = rospy.Subscriber(
            f'{self.namespace}/move_base_simple/goal',
            PoseStamped,
            self.goal_callback,
            queue_size=1
        )
        
        # 群体状态广播（发布自己的状态）
        self.state_pub = rospy.Publisher(
            '/swarm/agent_states',
            Float32MultiArray,
            queue_size=10
        )
        
        # 接收其他代理状态
        self.swarm_state_sub = rospy.Subscriber(
            '/swarm/agent_states',
            Float32MultiArray,
            self.swarm_state_callback,
            queue_size=50
        )
        
        # 输出控制指令
        self.cmd_pub = rospy.Publisher(
            f'{self.namespace}/adaptive_planning/swarm_cmd',
            TwistStamped,
            queue_size=10
        )
        
        # 调试信息
        self.debug_pub = rospy.Publisher(
            f'{self.namespace}/drooid/debug',
            Float32MultiArray,
            queue_size=10
        )
        
        # 控制定时器
        self.control_timer = rospy.Timer(
            rospy.Duration(0.05),  # 20Hz
            self.control_callback
        )
        
        # 状态广播定时器
        self.broadcast_timer = rospy.Timer(
            rospy.Duration(0.1),  # 10Hz
            self.broadcast_state
        )
    
    def odom_callback(self, msg):
        """位置信息回调"""
        self.position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        self.velocity = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])
        self.has_odom = True
    
    def goal_callback(self, msg):
        """目标位置回调"""
        self.goal_position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        rospy.loginfo(f"[Drooid] Agent {self.agent_id} received goal: {self.goal_position}")
    
    def swarm_state_callback(self, msg):
        """接收群体状态信息"""
        # 消息格式: [agent_id, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, timestamp, ...]
        data = msg.data
        current_time = rospy.get_time()
        
        with self.neighbor_lock:
            for i in range(0, len(data), 8):  # 每8个数据一个代理
                if i + 7 >= len(data):
                    break
                
                neighbor_id = int(data[i])
                if neighbor_id == self.agent_id:  # 跳过自己
                    continue
                
                neighbor_pos = np.array(data[i+1:i+4])
                neighbor_vel = np.array(data[i+4:i+7])
                neighbor_time = data[i+7]
                
                # 只保存最近的邻居信息
                if current_time - neighbor_time < 1.0:  # 1秒内的信息有效
                    self.neighbors[neighbor_id] = {
                        'pos': neighbor_pos,
                        'vel': neighbor_vel,
                        'time': neighbor_time
                    }
    
    def broadcast_state(self, event):
        """广播自己的状态"""
        if not self.has_odom:
            return
        
        msg = Float32MultiArray()
        msg.data = [
            float(self.agent_id),
            float(self.position[0]), float(self.position[1]), float(self.position[2]),
            float(self.velocity[0]), float(self.velocity[1]), float(self.velocity[2]),
            float(rospy.get_time())
        ]
        self.state_pub.publish(msg)
    
    def control_callback(self, event):
        """Drooid控制算法主循环"""
        if not self.has_odom:
            return
        
        # 计算Drooid控制指令
        control_velocity = self.compute_drooid_control()
        
        # 发布控制指令
        self.publish_control_command(control_velocity)
        
        # 发布调试信息
        self.publish_debug_info(control_velocity)
    
    def compute_drooid_control(self):
        """
        Drooid算法的核心：计算控制速度
        基于群体智能的四个基本行为：分离、对齐、聚集、目标导航
        """
        control_vel = np.zeros(3)
        
        with self.neighbor_lock:
            nearby_neighbors = self.get_nearby_neighbors()
        
        # 1. 目标导航力 (Goal Seeking)
        goal_force = self.compute_goal_force()
        
        # 2. 分离力 (Separation) - 避免碰撞
        separation_force = self.compute_separation_force(nearby_neighbors)
        
        # 3. 聚集力 (Cohesion) - 保持编队
        cohesion_force = self.compute_cohesion_force(nearby_neighbors)
        
        # 4. 对齐力 (Alignment) - 速度同步
        alignment_force = self.compute_alignment_force(nearby_neighbors)
        
        # 5. 编队保持力 (Formation)
        formation_force = self.compute_formation_force()
        
        # 加权组合所有力
        control_vel = (
            self.goal_gain * goal_force +
            self.separation_gain * separation_force +
            self.cohesion_gain * cohesion_force +
            self.alignment_gain * alignment_force +
            self.formation_gain * formation_force
        )
        
        # 限制最大速度
        speed = np.linalg.norm(control_vel)
        if speed > self.max_speed:
            control_vel = control_vel / speed * self.max_speed
        
        return control_vel
    
    def get_nearby_neighbors(self):
        """获取感知半径内的邻居"""
        nearby = {}
        for neighbor_id, info in self.neighbors.items():
            distance = np.linalg.norm(info['pos'] - self.position)
            if distance < self.sensing_radius:
                nearby[neighbor_id] = info
        return nearby
    
    def compute_goal_force(self):
        """计算目标导航力"""
        if np.allclose(self.goal_position, np.zeros(3)):
            return np.zeros(3)
        
        goal_direction = self.goal_position - self.position
        distance_to_goal = np.linalg.norm(goal_direction)
        
        if distance_to_goal < 0.1:
            return np.zeros(3)
        
        # 归一化方向向量
        goal_direction = goal_direction / distance_to_goal
        
        # 距离越远，力越大，但有上限
        force_magnitude = min(1.0, distance_to_goal / 5.0)
        
        return goal_direction * force_magnitude
    
    def compute_separation_force(self, neighbors):
        """计算分离力（避免碰撞）"""
        if not neighbors:
            return np.zeros(3)
        
        separation_force = np.zeros(3)
        
        for neighbor_id, info in neighbors.items():
            neighbor_pos = info['pos']
            relative_pos = self.position - neighbor_pos
            distance = np.linalg.norm(relative_pos)
            
            if distance < self.safe_distance and distance > 0:
                # 距离越近，分离力越大
                force_magnitude = (self.safe_distance - distance) / self.safe_distance
                direction = relative_pos / distance
                separation_force += direction * force_magnitude
        
        return separation_force
    
    def compute_cohesion_force(self, neighbors):
        """计算聚集力（保持编队紧密性）"""
        if not neighbors:
            return np.zeros(3)
        
        # 计算邻居重心
        center_of_mass = np.zeros(3)
        for neighbor_id, info in neighbors.items():
            center_of_mass += info['pos']
        center_of_mass /= len(neighbors)
        
        # 朝向重心的力
        cohesion_direction = center_of_mass - self.position
        distance_to_center = np.linalg.norm(cohesion_direction)
        
        if distance_to_center > 0:
            cohesion_direction = cohesion_direction / distance_to_center
            # 距离适中时聚集力较小，距离太远时聚集力增大
            force_magnitude = min(1.0, distance_to_center / self.sensing_radius)
            return cohesion_direction * force_magnitude
        
        return np.zeros(3)
    
    def compute_alignment_force(self, neighbors):
        """计算对齐力（速度同步）"""
        if not neighbors:
            return np.zeros(3)
        
        # 计算邻居平均速度
        average_velocity = np.zeros(3)
        for neighbor_id, info in neighbors.items():
            average_velocity += info['vel']
        average_velocity /= len(neighbors)
        
        # 对齐力 = 邻居平均速度 - 自己的速度
        alignment_force = average_velocity - self.velocity
        
        return alignment_force
    
    def compute_formation_force(self):
        """计算编队保持力"""
        # 根据编队类型计算期望位置
        desired_pos = self.get_desired_formation_position()
        
        if np.allclose(desired_pos, np.zeros(3)):
            return np.zeros(3)
        
        # 朝向期望位置的力
        formation_direction = desired_pos - self.position
        distance_to_desired = np.linalg.norm(formation_direction)
        
        if distance_to_desired > 0:
            formation_direction = formation_direction / distance_to_desired
            force_magnitude = min(1.0, distance_to_desired / 2.0)
            return formation_direction * force_magnitude
        
        return np.zeros(3)
    
    def get_desired_formation_position(self):
        """根据编队类型计算期望位置"""
        if self.formation_type == 'line':
            # 直线编队
            offset = (self.agent_id - self.total_agents / 2.0) * self.formation_size
            return self.goal_position + np.array([offset, 0, 0])
        
        elif self.formation_type == 'circle':
            # 圆形编队
            angle = 2 * np.pi * self.agent_id / self.total_agents
            offset_x = self.formation_size * np.cos(angle)
            offset_y = self.formation_size * np.sin(angle)
            return self.goal_position + np.array([offset_x, offset_y, 0])
        
        elif self.formation_type == 'triangle':
            # 三角形编队（适合3个agent）
            if self.total_agents == 3:
                positions = [
                    np.array([0, 0, 0]),  # 领导者
                    np.array([-self.formation_size, -self.formation_size, 0]),
                    np.array([self.formation_size, -self.formation_size, 0])
                ]
                if self.agent_id < len(positions):
                    return self.goal_position + positions[self.agent_id]
        
        # 默认返回目标位置
        return self.goal_position
    
    def publish_control_command(self, control_velocity):
        """发布控制指令"""
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.twist.linear.x = float(control_velocity[0])
        msg.twist.linear.y = float(control_velocity[1])
        msg.twist.linear.z = float(control_velocity[2])
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0
        
        self.cmd_pub.publish(msg)
    
    def publish_debug_info(self, control_velocity):
        """发布调试信息"""
        debug_msg = Float32MultiArray()
        debug_data = list(control_velocity) + list(self.position) + list(self.goal_position)
        debug_data.append(len(self.neighbors))  # 邻居数量
        debug_msg.data = debug_data
        
        self.debug_pub.publish(debug_msg)

class DrooidSwarmCoordinator:
    """Drooid群体协调器"""
    def __init__(self):
        rospy.init_node('drooid_swarm_coordinator', anonymous=True)
        
        # 参数
        self.num_agents = rospy.get_param('~num_agents', 3)
        self.agent_namespaces = rospy.get_param('~agent_namespaces', 
                                                [f'/uav{i}' for i in range(self.num_agents)])
        
        # 创建群体代理
        self.agents = []
        for i in range(self.num_agents):
            namespace = self.agent_namespaces[i] if i < len(self.agent_namespaces) else f'/uav{i}'
            agent = DrooidSwarmAgent(i, self.num_agents, namespace)
            self.agents.append(agent)
        
        rospy.loginfo(f"[Drooid] 群体协调器启动，{self.num_agents}个代理")
    
    def run(self):
        """运行群体协调器"""
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("[Drooid] 群体协调器关闭")

# 测试和演示脚本
def test_drooid_algorithm():
    """测试Drooid算法"""
    print("🚀 测试Drooid群体算法...")
    
    # 创建测试代理
    agent = DrooidSwarmAgent(0, 3, "/test_uav")
    
    # 模拟测试数据
    agent.position = np.array([0, 0, 1])
    agent.goal_position = np.array([5, 5, 1])
    agent.has_odom = True
    
    # 添加模拟邻居
    agent.neighbors = {
        1: {'pos': np.array([1, 1, 1]), 'vel': np.array([0.5, 0.5, 0]), 'time': rospy.get_time()},
        2: {'pos': np.array([-1, 1, 1]), 'vel': np.array([-0.3, 0.7, 0]), 'time': rospy.get_time()}
    }
    
    # 计算控制指令
    control_vel = agent.compute_drooid_control()
    print(f"控制速度: {control_vel}")
    
    # 分析各个力分量
    goal_force = agent.compute_goal_force()
    separation_force = agent.compute_separation_force(agent.neighbors)
    cohesion_force = agent.compute_cohesion_force(agent.neighbors)
    alignment_force = agent.compute_alignment_force(agent.neighbors)
    formation_force = agent.compute_formation_force()
    
    print(f"目标力: {goal_force}")
    print(f"分离力: {separation_force}")
    print(f"聚集力: {cohesion_force}")
    print(f"对齐力: {alignment_force}")
    print(f"编队力: {formation_force}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_drooid_algorithm()
        elif sys.argv[1] == "single_agent":
            # 单个代理测试
            agent_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            namespace = sys.argv[3] if len(sys.argv) > 3 else f"/uav{agent_id}"
            agent = DrooidSwarmAgent(agent_id, 3, namespace)
            rospy.spin()
    else:
        # 启动群体协调器
        coordinator = DrooidSwarmCoordinator()
        coordinator.run()
