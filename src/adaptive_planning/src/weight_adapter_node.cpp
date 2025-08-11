#include "adaptive_planning/adaptive_weights.h"
#include <ros/ros.h>

int main(int argc, char** argv) {
    ros::init(argc, argv, "weight_adapter_node");
    ros::NodeHandle nh("~");
    
    ROS_INFO("[AdaptivePlanning] Starting weight adapter node...");
    
    adaptive_planning::WeightAdapter adapter(nh);
    
    ros::Rate rate(10.0); // 10Hz
    while (ros::ok()) {
        ros::spinOnce();
        rate.sleep();
    }
    
    return 0;
}
