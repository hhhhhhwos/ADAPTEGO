/**
 * @file keyboard_control.cpp
 * @brief 键盘控制节点，用于控制无人机移动
 * 
 * 控制说明：
 * W/S - 前进/后退
 * A/D - 左移/右移  
 * R/F - 上升/下降
 * Q/E - 左转/右转
 * L   - 降落
 * ESC - 退出
 */

#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <termios.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>

class KeyboardControl
{
private:
    ros::NodeHandle nh;
    ros::Publisher twist_pub;
    struct termios old_tio;
    
    void setupTerminal()
    {
        tcgetattr(STDIN_FILENO, &old_tio);
        struct termios new_tio = old_tio;
        new_tio.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &new_tio);
        fcntl(STDIN_FILENO, F_SETFL, O_NONBLOCK);
    }
    
    void restoreTerminal()
    {
        tcsetattr(STDIN_FILENO, TCSANOW, &old_tio);
    }
    
    int getch()
    {
        int ch = getchar();
        if (ch != EOF) {
            return ch;
        }
        return -1;
    }

public:
    KeyboardControl()
    {
        twist_pub = nh.advertise<geometry_msgs::Twist>("/px4ctrl/keyboard_cmd", 10);
        setupTerminal();
        
        ROS_INFO("=== 键盘控制启动 ===");
        ROS_INFO("控制说明：");
        ROS_INFO("W/S - 前进/后退");
        ROS_INFO("A/D - 左移/右移");
        ROS_INFO("R/F - 上升/下降");
        ROS_INFO("Q/E - 左转/右转");
        ROS_INFO("L   - 降落");
        ROS_INFO("ESC - 退出");
        ROS_INFO("==================");
    }
    
    ~KeyboardControl()
    {
        restoreTerminal();
    }
    
    void run()
    {
        ros::Rate rate(50); // 50Hz
        
        while (ros::ok())
        {
            int key = getch();
            
            if (key != -1)
            {
                geometry_msgs::Twist twist;
                bool send_msg = true;
                
                switch (key)
                {
                    case 'w':
                    case 'W':
                        twist.linear.x = 1.0; // 前进
                        ROS_INFO("前进");
                        break;
                        
                    case 's':
                    case 'S':
                        twist.linear.x = -1.0; // 后退
                        ROS_INFO("后退");
                        break;
                        
                    case 'a':
                    case 'A':
                        twist.linear.y = 1.0; // 左移
                        ROS_INFO("左移");
                        break;
                        
                    case 'd':
                    case 'D':
                        twist.linear.y = -1.0; // 右移
                        ROS_INFO("右移");
                        break;
                        
                    case 'r':
                    case 'R':
                        twist.linear.z = 1.0; // 上升
                        ROS_INFO("上升");
                        break;
                        
                    case 'f':
                    case 'F':
                        twist.linear.z = -1.0; // 下降
                        ROS_INFO("下降");
                        break;
                        
                    case 'q':
                    case 'Q':
                        twist.angular.z = 1.0; // 左转
                        ROS_INFO("左转");
                        break;
                        
                    case 'e':
                    case 'E':
                        twist.angular.z = -1.0; // 右转
                        ROS_INFO("右转");
                        break;
                        
                    case 'l':
                    case 'L':
                        twist.linear.x = 999.0; // 降落指令的特殊标识
                        ROS_INFO("降落指令");
                        break;
                        
                    case 27: // ESC键
                        ROS_INFO("退出键盘控制");
                        return;
                        
                    default:
                        send_msg = false;
                        break;
                }
                
                if (send_msg)
                {
                    twist_pub.publish(twist);
                }
            }
            
            ros::spinOnce();
            rate.sleep();
        }
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "keyboard_control");
    
    try {
        KeyboardControl control;
        control.run();
    }
    catch (const std::exception& e) {
        ROS_ERROR("键盘控制异常: %s", e.what());
    }
    
    return 0;
}
