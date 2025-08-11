#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 Autonomous-Quadcopter-Control-RL 的低层控制
提供RL控制器作为px4ctrl的替代方案
"""

import numpy as np
import torch
import torch.nn as nn
import rospy
from geometry_msgs.msg import TwistStamped, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
import os
import sys

# 添加Autonomous-Quadcopter-Control-RL路径
THIRD_PARTY_PATH = os.path.join(os.path.dirname(__file__), '../../third_party')
QUAD_RL_PATH = os.path.join(THIRD_PARTY_PATH, 'Autonomous-Quadcopter-Control-RL')
sys.path.append(QUAD_RL_PATH)

class QuadcopterRLController(nn.Module):
    """
    基于 Autonomous-Quadcopter-Control-RL 的四旋翼RL控制器
    直接从位置/速度指令生成电机转速
    """
    def __init__(self, state_dim=12, action_dim=4):
        super(QuadcopterRLController, self).__init__()
        
        # 状态：位置(3) + 姿态(3) + 线速度(3) + 角速度(3) = 12
        # 动作：四个电机的推力 [motor1, motor2, motor3, motor4]
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 位置控制分支
        self.position_branch = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # 姿态控制分支  
        self.attitude_branch = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, action_dim),
            nn.Tanh()  # 输出[-1, 1]，后续映射到实际推力
        )
        
        # 推力映射参数
        self.thrust_min = 0.0
        self.thrust_max = 1.0
        
    def forward(self, state, target_pose=None):
        # 特征提取
        features = self.feature_extractor(state)
        
        # 分支处理
        pos_features = self.position_branch(features)
        att_features = self.attitude_branch(features)
        
        # 融合
        combined = torch.cat([pos_features, att_features], dim=-1)
        raw_thrust = self.fusion_layer(combined)
        
        # 映射到推力范围 [thrust_min, thrust_max]
        thrust = (raw_thrust + 1.0) * 0.5 * (self.thrust_max - self.thrust_min) + self.thrust_min
        
        return thrust

class RLControllerNode:
    """
    RL控制器ROS节点
    可以替代px4ctrl进行低层控制实验
    """
    def __init__(self):
        rospy.init_node('rl_controller_node', anonymous=True)
        
        # 参数
        self.model_path = rospy.get_param('~model_path', '')
        self.control_rate = rospy.get_param('~control_rate', 50.0)  # 50Hz
        self.use_rl_control = rospy.get_param('~use_rl_control', False)
        
        # 状态
        self.current_odom = None
        self.target_pose = None
        self.target_twist = None
        self.has_odom = False
        
        # 加载RL控制器
        if self.use_rl_control and os.path.exists(self.model_path):
            self.rl_controller = QuadcopterRLController()
            self.load_rl_model(self.model_path)
            rospy.loginfo(f"[RLController] RL模型加载成功: {self.model_path}")
        else:
            self.rl_controller = None
            rospy.logwarn("[RLController] 未启用RL控制或模型文件不存在")
        
        # 订阅器
        self.odom_sub = rospy.Subscriber(
            '/drone_0_visual_slam/odom', 
            Odometry, 
            self.odom_callback, 
            queue_size=10
        )
        
        # 可以接收位置指令或速度指令
        self.pose_cmd_sub = rospy.Subscriber(
            '/rl_controller/pose_cmd',
            PoseStamped,
            self.pose_cmd_callback,
            queue_size=10
        )
        
        self.twist_cmd_sub = rospy.Subscriber(
            '/rl_controller/twist_cmd', 
            TwistStamped,
            self.twist_cmd_callback,
            queue_size=10
        )
        
        # 发布器
        # 输出到mavros（替代px4ctrl的输出）
        self.cmd_vel_pub = rospy.Publisher(
            '/mavros/setpoint_velocity/cmd_vel',
            TwistStamped,
            queue_size=10
        )
        
        # 调试信息发布
        self.debug_pub = rospy.Publisher(
            '/rl_controller/debug',
            Float32MultiArray,
            queue_size=10
        )
        
        # 控制定时器
        self.control_timer = rospy.Timer(
            rospy.Duration(1.0 / self.control_rate),
            self.control_callback
        )
        
        rospy.loginfo("[RLController] RL控制器节点启动完成")
    
    def odom_callback(self, msg):
        self.current_odom = msg
        self.has_odom = True
    
    def pose_cmd_callback(self, msg):
        self.target_pose = msg
        self.target_twist = None  # 清除速度指令
    
    def twist_cmd_callback(self, msg):
        self.target_twist = msg
        self.target_pose = None  # 清除位置指令
    
    def control_callback(self, event):
        if not self.has_odom:
            return
        
        if self.use_rl_control and self.rl_controller is not None:
            self.rl_control_step()
        else:
            self.pid_control_step()
    
    def rl_control_step(self):
        """使用RL控制器计算控制指令"""
        if not (self.target_pose or self.target_twist):
            return
        
        # 构造状态向量
        state = self.build_state_vector()
        
        # RL推理
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            thrust_commands = self.rl_controller(state_tensor)
            thrust_commands = thrust_commands.squeeze(0).numpy()
        
        # 转换推力指令为速度指令（简化处理）
        cmd_vel = self.thrust_to_velocity(thrust_commands)
        
        # 发布控制指令
        self.publish_velocity_command(cmd_vel)
        
        # 发布调试信息
        debug_msg = Float32MultiArray()
        debug_msg.data = list(thrust_commands) + list(cmd_vel)
        self.debug_pub.publish(debug_msg)
    
    def pid_control_step(self):
        """传统PID控制（作为对比基线）"""
        if not (self.target_pose or self.target_twist):
            return
        
        # 简单PID控制实现
        current_pos = np.array([
            self.current_odom.pose.pose.position.x,
            self.current_odom.pose.pose.position.y,
            self.current_odom.pose.pose.position.z
        ])
        
        current_vel = np.array([
            self.current_odom.twist.twist.linear.x,
            self.current_odom.twist.twist.linear.y,
            self.current_odom.twist.twist.linear.z
        ])
        
        if self.target_pose:
            # 位置控制模式
            target_pos = np.array([
                self.target_pose.pose.position.x,
                self.target_pose.pose.position.y,
                self.target_pose.pose.position.z
            ])
            
            # 简单PD控制
            pos_error = target_pos - current_pos
            kp = 1.0
            kd = 0.5
            cmd_vel = kp * pos_error - kd * current_vel
            
        elif self.target_twist:
            # 速度控制模式（直接传递）
            cmd_vel = np.array([
                self.target_twist.twist.linear.x,
                self.target_twist.twist.linear.y,
                self.target_twist.twist.linear.z
            ])
        
        # 限制速度
        cmd_vel = np.clip(cmd_vel, -2.0, 2.0)
        
        # 发布控制指令
        self.publish_velocity_command(cmd_vel)
    
    def build_state_vector(self):
        """构建RL控制器的状态向量"""
        # 当前状态
        pos = np.array([
            self.current_odom.pose.pose.position.x,
            self.current_odom.pose.pose.position.y,
            self.current_odom.pose.pose.position.z
        ])
        
        # 简化姿态表示（欧拉角）
        q = self.current_odom.pose.pose.orientation
        # 这里应该转换为欧拉角，简化处理
        att = np.array([q.x, q.y, q.z])
        
        vel = np.array([
            self.current_odom.twist.twist.linear.x,
            self.current_odom.twist.twist.linear.y,
            self.current_odom.twist.twist.linear.z
        ])
        
        ang_vel = np.array([
            self.current_odom.twist.twist.angular.x,
            self.current_odom.twist.twist.angular.y,
            self.current_odom.twist.twist.angular.z
        ])
        
        # 拼接状态：pos(3) + att(3) + vel(3) + ang_vel(3) = 12
        state = np.concatenate([pos, att, vel, ang_vel])
        return state
    
    def thrust_to_velocity(self, thrust_commands):
        """将推力指令转换为速度指令（简化映射）"""
        # 这是一个简化的映射，实际应该基于动力学模型
        # thrust_commands: [motor1, motor2, motor3, motor4]
        
        # 简化：推力差异映射为速度分量
        total_thrust = np.mean(thrust_commands)
        
        # 基础垂直速度
        vz = (total_thrust - 0.5) * 2.0  # 悬停在0.5，范围[-1, 1]
        
        # 水平速度（基于推力不对称性）
        vx = (thrust_commands[0] + thrust_commands[1] - thrust_commands[2] - thrust_commands[3]) * 0.5
        vy = (thrust_commands[0] + thrust_commands[3] - thrust_commands[1] - thrust_commands[2]) * 0.5
        
        cmd_vel = np.array([vx, vy, vz])
        cmd_vel = np.clip(cmd_vel, -2.0, 2.0)
        
        return cmd_vel
    
    def publish_velocity_command(self, cmd_vel):
        """发布速度指令"""
        msg = TwistStamped()
        msg.header.stamp = rospy.Time.now()
        msg.twist.linear.x = float(cmd_vel[0])
        msg.twist.linear.y = float(cmd_vel[1])
        msg.twist.linear.z = float(cmd_vel[2])
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0
        
        self.cmd_vel_pub.publish(msg)
    
    def load_rl_model(self, model_path):
        """加载RL控制器模型"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            self.rl_controller.load_state_dict(checkpoint['model_state_dict'])
            self.rl_controller.eval()
            rospy.loginfo(f"[RLController] 模型加载成功: {model_path}")
        except Exception as e:
            rospy.logerr(f"[RLController] 模型加载失败: {e}")
            self.rl_controller = None

def train_quadcopter_rl_controller():
    """
    训练四旋翼RL控制器
    基于 Autonomous-Quadcopter-Control-RL 的方法
    """
    print("🚀 开始训练四旋翼RL控制器...")
    
    # 这里可以集成Autonomous-Quadcopter-Control-RL的训练代码
    # 现在提供一个简化的训练框架
    
    controller = QuadcopterRLController()
    optimizer = torch.optim.Adam(controller.parameters(), lr=3e-4)
    
    # 模拟训练数据
    num_episodes = 1000
    for episode in range(num_episodes):
        # 模拟状态
        state = torch.randn(1, 12)
        
        # 前向传播
        thrust = controller(state)
        
        # 这里应该有环境交互和奖励计算
        # 简化为随机奖励
        reward = torch.randn(1)
        loss = -reward * torch.sum(thrust)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Loss: {loss.item():.4f}")
    
    # 保存模型
    model_save_path = "models/quadcopter_rl_controller.pth"
    os.makedirs("models", exist_ok=True)
    torch.save({
        'model_state_dict': controller.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_save_path)
    
    print(f"✅ 模型保存至: {model_save_path}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_quadcopter_rl_controller()
    else:
        # 启动ROS节点
        try:
            controller_node = RLControllerNode()
            rospy.spin()
        except rospy.ROSInterruptException:
            pass
