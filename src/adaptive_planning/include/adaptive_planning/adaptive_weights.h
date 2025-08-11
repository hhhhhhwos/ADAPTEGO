#ifndef _ADAPTIVE_WEIGHTS_H_
#define _ADAPTIVE_WEIGHTS_H_

#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <Eigen/Eigen>

namespace adaptive_planning {

// 自适应权重参数结构体
struct AdaptiveWeights {
    double w_smooth;        // 平滑度权重 [0.1, 2.0]
    double w_collision;     // 避障权重 [1.0, 10.0] 
    double w_time;          // 时间权重 [0.1, 1.0]
    double corridor_width;  // 安全走廊宽度 [0.3, 1.5]m
    double max_velocity;    // 最大速度 [0.5, 3.0]m/s
    double replan_freq;     // 重规划频率 [5, 30]Hz
    
    AdaptiveWeights() {
        // 默认值 (现有ego-planner的参数)
        w_smooth = 1.0;
        w_collision = 5.0;
        w_time = 0.5;
        corridor_width = 0.8;
        max_velocity = 2.0;
        replan_freq = 20.0;
    }
};

// 环境状态特征
struct EnvironmentState {
    Eigen::Vector3d position;
    Eigen::Vector3d velocity;
    Eigen::Vector3d goal;
    double obstacle_density;    // 障碍密度
    double avg_clearance;      // 平均通行空间
    double current_speed;      // 当前速度
    double goal_distance;      // 目标距离
    double path_complexity;    // 路径复杂度
    double battery_level;      // 电量水平
    
    Eigen::VectorXd toVector() const {
        Eigen::VectorXd state(12);
        state << position, goal, obstacle_density, avg_clearance, 
                 current_speed, goal_distance, path_complexity, battery_level;
        return state;
    }
};

class WeightAdapter {
public:
    WeightAdapter(ros::NodeHandle& nh);
    ~WeightAdapter() = default;
    
    void initialize();
    void odomCallback(const nav_msgs::Odometry::ConstPtr& msg);
    void goalCallback(const geometry_msgs::PoseStamped::ConstPtr& msg);
    void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg);
    
    AdaptiveWeights predictWeights(const EnvironmentState& state);
    void publishWeights(const AdaptiveWeights& weights);
    
private:
    ros::NodeHandle nh_;
    
    // Subscribers
    ros::Subscriber odom_sub_;
    ros::Subscriber goal_sub_;
    ros::Subscriber pointcloud_sub_;
    
    // Publishers
    ros::Publisher weights_pub_;
    
    // 当前状态
    EnvironmentState current_state_;
    bool has_odom_;
    bool has_goal_;
    
    // 参数
    std::string model_path_;
    double update_rate_;
    
    // 模型推理相关 (预留，后续用PyTorch C++或Python桥接)
    bool use_learned_weights_;
    
    // 环境感知
    void updateEnvironmentFeatures();
    double calculateObstacleDensity(const sensor_msgs::PointCloud2::ConstPtr& cloud);
    double calculateAverageClearance(const sensor_msgs::PointCloud2::ConstPtr& cloud);
    double calculatePathComplexity();
};

} // namespace adaptive_planning

#endif
