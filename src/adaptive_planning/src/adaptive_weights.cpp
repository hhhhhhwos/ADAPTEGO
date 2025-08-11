#include "adaptive_planning/adaptive_weights.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <cmath>

namespace adaptive_planning {

WeightAdapter::WeightAdapter(ros::NodeHandle& nh) : nh_(nh), has_odom_(false), has_goal_(false) {
    initialize();
}

void WeightAdapter::initialize() {
    // 读取参数
    nh_.param<std::string>("model_path", model_path_, "");
    nh_.param<double>("update_rate", update_rate_, 10.0);
    nh_.param<bool>("use_learned_weights", use_learned_weights_, false);
    
    // 订阅器
    odom_sub_ = nh_.subscribe("/drone_0_visual_slam/odom", 10, 
                              &WeightAdapter::odomCallback, this);
    goal_sub_ = nh_.subscribe("/move_base_simple/goal", 1, 
                              &WeightAdapter::goalCallback, this);
    pointcloud_sub_ = nh_.subscribe("/drone_0_pcl_render_node/cloud", 10,
                                    &WeightAdapter::pointCloudCallback, this);
    
    // 发布器
    weights_pub_ = nh_.advertise<std_msgs::Float32MultiArray>("/adaptive_planning/weights", 10);
    
    // 初始化状态
    current_state_.position = Eigen::Vector3d::Zero();
    current_state_.velocity = Eigen::Vector3d::Zero();
    current_state_.goal = Eigen::Vector3d(15.0, 0.0, 1.0); // 默认目标
    current_state_.battery_level = 1.0; // 满电
    
    ROS_INFO("[AdaptivePlanning] Weight adapter initialized");
    ROS_INFO("[AdaptivePlanning] Model path: %s", model_path_.c_str());
    ROS_INFO("[AdaptivePlanning] Use learned weights: %s", use_learned_weights_ ? "true" : "false");
}

void WeightAdapter::odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    current_state_.position << msg->pose.pose.position.x,
                               msg->pose.pose.position.y,
                               msg->pose.pose.position.z;
    
    current_state_.velocity << msg->twist.twist.linear.x,
                               msg->twist.twist.linear.y,
                               msg->twist.twist.linear.z;
    
    current_state_.current_speed = current_state_.velocity.norm();
    current_state_.goal_distance = (current_state_.goal - current_state_.position).norm();
    
    has_odom_ = true;
    
    // 更新环境特征并发布权重
    if (has_goal_) {
        updateEnvironmentFeatures();
        AdaptiveWeights weights = predictWeights(current_state_);
        publishWeights(weights);
    }
}

void WeightAdapter::goalCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
    current_state_.goal << msg->pose.position.x,
                           msg->pose.position.y,
                           msg->pose.position.z;
    
    has_goal_ = true;
    ROS_INFO("[AdaptivePlanning] New goal received: (%.2f, %.2f, %.2f)", 
             current_state_.goal.x(), current_state_.goal.y(), current_state_.goal.z());
}

void WeightAdapter::pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg) {
    if (!has_odom_) return;
    
    // 更新障碍感知特征
    current_state_.obstacle_density = calculateObstacleDensity(msg);
    current_state_.avg_clearance = calculateAverageClearance(msg);
}

void WeightAdapter::updateEnvironmentFeatures() {
    // 计算路径复杂度 (简单的启发式)
    double dist_to_goal = current_state_.goal_distance;
    double speed_ratio = current_state_.current_speed / 2.0; // 归一化到最大速度2m/s
    
    // 路径复杂度: 结合距离、障碍密度和当前速度
    current_state_.path_complexity = 
        std::min(1.0, current_state_.obstacle_density * 0.4 + 
                      (1.0 / std::max(0.1, dist_to_goal)) * 0.3 +
                      speed_ratio * 0.3);
}

AdaptiveWeights WeightAdapter::predictWeights(const EnvironmentState& state) {
    AdaptiveWeights weights;
    
    if (use_learned_weights_ && !model_path_.empty()) {
        // TODO: 这里后续集成PyTorch模型推理
        // 现在使用基于规则的自适应策略
        ROS_WARN_ONCE("[AdaptivePlanning] Learned model not implemented, using rule-based adaptation");
    }
    
    // 基于规则的自适应权重策略 (作为baseline和fallback)
    double obs_factor = std::max(0.1, std::min(2.0, state.obstacle_density * 2.0));
    double dist_factor = std::max(0.5, std::min(1.5, 2.0 / std::max(0.5, state.goal_distance)));
    double speed_factor = std::max(0.8, std::min(1.2, state.current_speed / 1.5));
    
    // 自适应调整权重
    weights.w_collision = 5.0 * obs_factor;     // 障碍越密集，避障权重越高
    weights.w_smooth = 1.0 / speed_factor;      // 速度越快，越需要平滑
    weights.w_time = 0.5 * dist_factor;         // 距离目标越近，时间权重越高
    
    // 安全走廊和速度限制
    weights.corridor_width = std::max(0.4, std::min(1.2, 1.0 - state.obstacle_density * 0.5));
    weights.max_velocity = std::max(1.0, std::min(2.5, 2.0 - state.obstacle_density * 0.8));
    weights.replan_freq = std::max(10.0, std::min(25.0, 15.0 + state.path_complexity * 10.0));
    
    return weights;
}

void WeightAdapter::publishWeights(const AdaptiveWeights& weights) {
    std_msgs::Float32MultiArray msg;
    msg.data.resize(6);
    msg.data[0] = static_cast<float>(weights.w_smooth);
    msg.data[1] = static_cast<float>(weights.w_collision);
    msg.data[2] = static_cast<float>(weights.w_time);
    msg.data[3] = static_cast<float>(weights.corridor_width);
    msg.data[4] = static_cast<float>(weights.max_velocity);
    msg.data[5] = static_cast<float>(weights.replan_freq);
    
    weights_pub_.publish(msg);
    
    // 调试信息
    static int count = 0;
    if (++count % 50 == 0) { // 每5秒打印一次 (10Hz * 50)
        ROS_INFO("[AdaptivePlanning] Weights: smooth=%.2f, collision=%.2f, time=%.2f, "
                 "corridor=%.2f, max_vel=%.2f, freq=%.1f", 
                 weights.w_smooth, weights.w_collision, weights.w_time,
                 weights.corridor_width, weights.max_velocity, weights.replan_freq);
    }
}

double WeightAdapter::calculateObstacleDensity(const sensor_msgs::PointCloud2::ConstPtr& cloud) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*cloud, *pcl_cloud);
    
    if (pcl_cloud->empty()) return 0.0;
    
    // 计算局部区域内的点密度 (半径2m内)
    int nearby_points = 0;
    double radius = 2.0;
    
    for (const auto& point : pcl_cloud->points) {
        double dist = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
        if (dist < radius) {
            nearby_points++;
        }
    }
    
    // 归一化到 [0, 1]
    double volume = (4.0/3.0) * M_PI * radius * radius * radius;
    double density = nearby_points / (volume * 100.0); // 假设100点/m³为高密度
    return std::min(1.0, density);
}

double WeightAdapter::calculateAverageClearance(const sensor_msgs::PointCloud2::ConstPtr& cloud) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*cloud, *pcl_cloud);
    
    if (pcl_cloud->empty()) return 2.0; // 假设开阔空间
    
    // 找到最近障碍距离
    double min_dist = std::numeric_limits<double>::max();
    for (const auto& point : pcl_cloud->points) {
        double dist = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
        min_dist = std::min(min_dist, dist);
    }
    
    return std::min(3.0, min_dist); // 限制在3米内
}

} // namespace adaptive_planning
