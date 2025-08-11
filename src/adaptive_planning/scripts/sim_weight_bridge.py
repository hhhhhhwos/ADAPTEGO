#!/usr/bin/env python3
"""
仿真权重桥接器 - 将自适应权重应用到EGO-Planner
与SO3仿真器环境集成
"""

import rospy
import dynamic_reconfigure.client
from std_msgs.msg import Float32MultiArray

class SimWeightBridge:
    """
    权重桥接器：将预测的自适应权重实时应用到EGO-Planner参数
    """
    def __init__(self):
        rospy.init_node('sim_weight_bridge', anonymous=True)
        
        # 参数
        self.drone_id = rospy.get_param('~drone_id', 0)
        self.planner_namespace = rospy.get_param('~planner_namespace', f'/drone_{self.drone_id}_ego_planner')
        
        # 动态参数客户端
        try:
            # 等待规划器节点启动
            rospy.loginfo("[SimWeightBridge] 等待EGO-Planner节点启动...")
            rospy.wait_for_service(f'{self.planner_namespace}/set_parameters', timeout=10.0)
            
            # 创建动态重配置客户端
            self.config_client = dynamic_reconfigure.client.Client(
                f'{self.planner_namespace}/EGOPlannerNodeConfig', 
                timeout=30
            )
            rospy.loginfo("[SimWeightBridge] 成功连接到EGO-Planner动态重配置")
            
        except Exception as e:
            rospy.logwarn(f"[SimWeightBridge] 动态重配置连接失败: {e}")
            self.config_client = None
            
        # ROS接口
        self.weights_sub = rospy.Subscriber('/adaptive_weights', Float32MultiArray, self.weights_callback)
        
        # 权重名称映射
        self.weight_names = [
            'w_time', 'w_smooth', 'w_dist', 'w_feasibility', 'w_end', 'w_guide'
        ]
        
        rospy.loginfo(f"[SimWeightBridge] 仿真权重桥接器已启动 - 无人机{self.drone_id}")
        
    def weights_callback(self, msg):
        """权重回调：更新EGO-Planner参数"""
        if self.config_client is None:
            return
            
        if len(msg.data) != 6:
            rospy.logwarn(f"[SimWeightBridge] 权重维度错误: {len(msg.data)}, 期望6个")
            return
            
        try:
            # 构建参数更新字典
            params_update = {}
            for i, weight_name in enumerate(self.weight_names):
                params_update[weight_name] = float(msg.data[i])
                
            # 应用参数更新
            self.config_client.update_configuration(params_update)
            
            # 日志记录（降低频率）
            if rospy.get_time() % 2.0 < 0.1:  # 每2秒打印一次
                weight_str = ', '.join([f'{name}: {val:.2f}' for name, val in params_update.items()])
                rospy.loginfo(f"[SimWeightBridge] 已更新权重: {weight_str}")
                
        except Exception as e:
            rospy.logwarn(f"[SimWeightBridge] 权重更新失败: {e}")

if __name__ == '__main__':
    try:
        bridge = SimWeightBridge()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
