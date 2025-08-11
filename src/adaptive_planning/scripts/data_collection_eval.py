#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据收集与性能评估系统
结合你的评估框架和数据收集逻辑
支持专家演示收集、性能评估和基准测试
"""

import os
import csv
import time
import rospy
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import LaserScan
import tf.transformations as tf_trans
import pickle
from datetime import datetime

class ExpertDataCollector:
    """
    专家演示数据收集器 - 增强版
    支持实时数据收集和专家演示标注
    """
    def __init__(self, output_dir='data'):
        rospy.init_node('expert_data_collector', anonymous=True)
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 数据缓存
        self.states = []
        self.weights = []
        self.success_flags = []
        self.trajectory_data = []
        
        # 当前状态
        self.current_pos = np.zeros(3)
        self.current_vel = np.zeros(3)
        self.goal_pos = np.array([2.0, 0.0, 1.0])
        self.obstacle_density = 0.0
        self.avg_clearance = 5.0
        self.battery_level = 1.0
        self.current_weights = np.array([1.5, 5.0, 0.5, 0.8, 2.0, 15.0])
        
        # 专家标注状态
        self.collecting = False
        self.current_episode = 0
        self.episode_start_time = None
        self.last_goal_pos = None
        
        # 历史记录
        self.pose_history = deque(maxlen=100)
        
        # ROS接口
        self.setup_subscribers()
        
        rospy.loginfo("专家数据收集器已启动 - 按 's' 开始收集, 'e' 结束episode")
        
    def setup_subscribers(self):
        """设置ROS订阅者"""
        # 兼容多种话题格式
        self.odom_sub = rospy.Subscriber('/mavros/local_position/odom', 
                                       Odometry, self.odom_callback, queue_size=10)
        self.goal_sub = rospy.Subscriber('/move_base_simple/goal', 
                                       PoseStamped, self.goal_callback, queue_size=10)
        self.scan_sub = rospy.Subscriber('/scan', 
                                       LaserScan, self.scan_callback, queue_size=10)
        self.weights_sub = rospy.Subscriber('/planner/adaptive_weights',
                                          Float32MultiArray, self.weights_callback, queue_size=10)
        
        # 仿真环境话题（可选）
        self.sim_odom_sub = rospy.Subscriber('/drone_0_visual_slam/odom',
                                           Odometry, self.odom_callback, queue_size=10)
        
    def odom_callback(self, msg):
        """里程计回调"""
        self.current_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        self.current_vel = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])
        
        # 记录轨迹
        if self.collecting:
            self.pose_history.append({
                'time': rospy.Time.now().to_sec(),
                'pos': self.current_pos.copy(),
                'vel': self.current_vel.copy()
            })
        
    def goal_callback(self, msg):
        """目标回调"""
        new_goal = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        
        # 检测新目标
        if self.last_goal_pos is None or np.linalg.norm(new_goal - self.last_goal_pos) > 0.5:
            self.goal_pos = new_goal
            self.last_goal_pos = new_goal.copy()
            rospy.loginfo(f"新目标: {self.goal_pos}")
            
            # 自动开始新episode
            if not self.collecting:
                self.start_episode()
        
    def scan_callback(self, msg):
        """激光扫描回调"""
        try:
            ranges = np.array(msg.ranges)
            valid_ranges = ranges[np.isfinite(ranges) & (ranges > 0.1)]
            
            if len(valid_ranges) > 0:
                self.avg_clearance = np.mean(valid_ranges)
                close_obstacles = np.sum(valid_ranges < 1.0)
                self.obstacle_density = close_obstacles / len(valid_ranges)
            else:
                self.avg_clearance = 5.0
                self.obstacle_density = 0.0
        except Exception as e:
            rospy.logwarn(f"激光数据处理失败: {e}")
            
    def weights_callback(self, msg):
        """权重回调"""
        if len(msg.data) == 6:
            self.current_weights = np.array(msg.data)
        
    def compute_state_features(self):
        """计算状态特征"""
        goal_distance = np.linalg.norm(self.goal_pos - self.current_pos)
        current_speed = np.linalg.norm(self.current_vel)
        path_complexity = self.obstacle_density * np.log(goal_distance + 1.0)
        
        state = np.array([
            # 位置和目标 (6)
            self.current_pos[0], self.current_pos[1], self.current_pos[2],
            self.goal_pos[0], self.goal_pos[1], self.goal_pos[2],
            # 环境特征 (6)
            self.obstacle_density, self.avg_clearance, current_speed,
            goal_distance, path_complexity, self.battery_level
        ], dtype=np.float32)
        
        return state
        
    def start_episode(self):
        """开始新的数据收集episode"""
        self.collecting = True
        self.current_episode += 1
        self.episode_start_time = rospy.Time.now().to_sec()
        self.pose_history.clear()
        
        rospy.loginfo(f"开始收集Episode {self.current_episode}")
        
    def end_episode(self, success=None):
        """结束episode"""
        if not self.collecting:
            return
            
        self.collecting = False
        
        # 如果没有指定成功状态，自动判断
        if success is None:
            goal_distance = np.linalg.norm(self.goal_pos - self.current_pos)
            success = goal_distance < 1.0  # 1米内算成功
            
        episode_duration = rospy.Time.now().to_sec() - self.episode_start_time
        
        # 保存episode数据
        episode_data = {
            'episode_id': self.current_episode,
            'success': success,
            'duration': episode_duration,
            'trajectory': list(self.pose_history),
            'final_goal_distance': np.linalg.norm(self.goal_pos - self.current_pos),
            'start_pos': self.pose_history[0]['pos'] if self.pose_history else None,
            'goal_pos': self.goal_pos.copy()
        }
        
        # 提取样本
        samples_added = 0
        for i, pose_data in enumerate(self.pose_history):
            if i % 5 == 0:  # 降采样，每5个点取1个
                # 重建当时的状态
                temp_pos = self.current_pos
                temp_goal = self.goal_pos
                self.current_pos = pose_data['pos']
                
                state = self.compute_state_features()
                self.states.append(state)
                self.weights.append(self.current_weights.copy())
                self.success_flags.append(1.0 if success else 0.0)
                samples_added += 1
                
                # 恢复当前位置
                self.current_pos = temp_pos
        
        rospy.loginfo(f"Episode {self.current_episode} 完成: "
                      f"成功={success}, 用时={episode_duration:.1f}s, 样本={samples_added}")
        
        # 保存episode详情
        episode_file = os.path.join(self.output_dir, f'episode_{self.current_episode:04d}.pkl')
        with open(episode_file, 'wb') as f:
            pickle.dump(episode_data, f)
            
    def save_data(self, filename='expert_demonstrations.npz'):
        """保存所有收集的数据"""
        if len(self.states) == 0:
            rospy.logwarn("没有数据可保存")
            return
            
        filepath = os.path.join(self.output_dir, filename)
        np.savez_compressed(
            filepath,
            states=np.array(self.states),
            weights=np.array(self.weights),
            success=np.array(self.success_flags)
        )
        
        # 保存汇总统计
        stats = {
            'total_samples': len(self.states),
            'success_samples': np.sum(self.success_flags),
            'success_rate': np.mean(self.success_flags),
            'total_episodes': self.current_episode,
            'collection_date': datetime.now().isoformat()
        }
        
        stats_file = os.path.join(self.output_dir, 'collection_stats.json')
        import json
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        rospy.loginfo(f"已保存 {len(self.states)} 个样本到 {filepath}")
        print(f"成功率: {stats['success_rate']:.2%}")
        print(f"总episodes: {stats['total_episodes']}")

class PerformanceEvaluator:
    """
    性能评估器 - 增强版
    支持多种评估指标和实时监控
    """
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 评估指标
        self.metrics = defaultdict(list)
        self.current_experiment = None
        self.experiment_start_time = None
        
        # 轨迹记录
        self.trajectory = []
        self.goal_reached = False
        self.collision_count = 0
        
        # ROS订阅者（用于实时监控）
        self.odom_sub = rospy.Subscriber('/mavros/local_position/odom',
                                       Odometry, self.trajectory_callback, queue_size=10)
        
    def trajectory_callback(self, msg):
        """轨迹记录回调"""
        if self.current_experiment is not None:
            pos = np.array([
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ])
            vel = np.array([
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z
            ])
            
            self.trajectory.append({
                'time': rospy.Time.now().to_sec(),
                'pos': pos,
                'vel': vel
            })
        
    def start_experiment(self, experiment_name):
        """开始实验"""
        self.current_experiment = experiment_name
        self.experiment_start_time = time.time()
        self.trajectory = []
        self.goal_reached = False
        self.collision_count = 0
        
        rospy.loginfo(f"开始实验: {experiment_name}")
        
    def end_experiment(self, success=True):
        """结束实验并计算指标"""
        if self.experiment_start_time is None:
            return
            
        experiment_time = time.time() - self.experiment_start_time
        
        # 计算轨迹指标
        metrics = self.calculate_trajectory_metrics()
        
        # 添加基本指标
        metrics.update({
            'experiment': self.current_experiment,
            'success': 1 if success else 0,
            'completion_time': experiment_time,
            'collision_count': self.collision_count,
            'trajectory_points': len(self.trajectory)
        })
        
        # 记录指标
        for key, value in metrics.items():
            self.metrics[key].append(value)
            
        rospy.loginfo(f"实验完成: {self.current_experiment}")
        rospy.loginfo(f"成功: {success}, 用时: {experiment_time:.2f}s")
        
        # 重置状态
        self.current_experiment = None
        
    def calculate_trajectory_metrics(self):
        """计算轨迹指标"""
        if len(self.trajectory) < 2:
            return {'path_length': 0.0, 'smoothness_jerk': 0.0, 'avg_speed': 0.0}
            
        positions = np.array([p['pos'] for p in self.trajectory])
        velocities = np.array([p['vel'] for p in self.trajectory])
        times = np.array([p['time'] for p in self.trajectory])
        
        # 路径长度
        path_diffs = np.diff(positions, axis=0)
        path_length = np.sum(np.linalg.norm(path_diffs, axis=1))
        
        # 平滑度 (jerk)
        dt = np.diff(times)
        if len(dt) > 0 and np.all(dt > 0):
            accelerations = np.diff(velocities, axis=0) / dt[:, np.newaxis]
            if len(accelerations) > 1:
                dt_acc = dt[1:]
                jerk = np.diff(accelerations, axis=0) / dt_acc[:, np.newaxis]
                smoothness = np.mean(np.linalg.norm(jerk, axis=1))
            else:
                smoothness = 0.0
        else:
            smoothness = 0.0
            
        # 平均速度
        speeds = np.linalg.norm(velocities, axis=1)
        avg_speed = np.mean(speeds)
        
        return {
            'path_length': path_length,
            'smoothness_jerk': smoothness,
            'avg_speed': avg_speed
        }
        
    def save_results(self, filename='evaluation_results.csv'):
        """保存评估结果"""
        if not self.metrics:
            rospy.logwarn("没有评估结果可保存")
            return
            
        filepath = os.path.join(self.output_dir, filename)
        df = pd.DataFrame(self.metrics)
        df.to_csv(filepath, index=False)
        
        rospy.loginfo(f"评估结果已保存到 {filepath}")
        
        # 打印汇总统计
        print("\n=== 评估结果汇总 ===")
        if 'experiment' in df.columns:
            grouped = df.groupby('experiment')
            
            for exp_name, group in grouped:
                success_rate = group['success'].mean()
                avg_time = group['completion_time'].mean()
                avg_path_length = group['path_length'].mean()
                avg_smoothness = group['smoothness_jerk'].mean()
                
                print(f"\n{exp_name}:")
                print(f"  成功率: {success_rate:.2%}")
                print(f"  平均用时: {avg_time:.2f}s")
                print(f"  平均路径长: {avg_path_length:.2f}m")
                print(f"  平均平滑度: {avg_smoothness:.4f}")

class DataCollectionNode:
    """
    数据收集节点 - 交互式收集专家演示
    """
    def __init__(self):
        self.collector = ExpertDataCollector()
        
    def interactive_collection(self):
        """交互式数据收集"""
        print("\n=== 专家数据收集系统 ===")
        print("命令:")
        print("  's' - 开始新episode")
        print("  'e' - 结束当前episode (成功)")
        print("  'f' - 结束当前episode (失败)")
        print("  'save' - 保存所有数据")
        print("  'quit' - 退出")
        print("===================\n")
        
        try:
            while not rospy.is_shutdown():
                command = input("请输入命令: ").strip().lower()
                
                if command == 's':
                    self.collector.start_episode()
                elif command == 'e':
                    self.collector.end_episode(success=True)
                elif command == 'f':
                    self.collector.end_episode(success=False)
                elif command == 'save':
                    self.collector.save_data()
                elif command == 'quit':
                    break
                else:
                    print("无效命令")
                    
        except KeyboardInterrupt:
            print("\n中断信号接收")
            
        # 最终保存
        if len(self.collector.states) > 0:
            self.collector.save_data()
            print("数据已自动保存")

def main():
    """主函数 - 选择运行模式"""
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  python data_collection_eval.py collect  - 交互式数据收集")
        print("  python data_collection_eval.py evaluate - 性能评估")
        return
        
    mode = sys.argv[1]
    
    if mode == 'collect':
        # 交互式数据收集
        rospy.init_node('data_collection_node', anonymous=True)
        node = DataCollectionNode()
        node.interactive_collection()
        
    elif mode == 'evaluate':
        # 性能评估模式
        rospy.init_node('performance_evaluator', anonymous=True)
        evaluator = PerformanceEvaluator()
        
        # 示例评估
        evaluator.start_experiment('test_run')
        
        print("评估运行中，按Ctrl+C结束...")
        try:
            rospy.sleep(30.0)  # 运行30秒
            evaluator.end_experiment(success=True)
        except KeyboardInterrupt:
            evaluator.end_experiment(success=False)
            
        evaluator.save_results()
        
    else:
        print(f"未知模式: {mode}")

if __name__ == '__main__':
    main()
