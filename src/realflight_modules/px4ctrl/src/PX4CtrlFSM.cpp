// PX4无人机控制状态机实现文件
// 实现了基于状态机的无人机控制逻辑，支持手动、悬停、命令、起飞、降落五种状态
// 新增键盘控制功能，支持WASD实时控制

/**
 * ============================================================================
 *                     PX4 无人机飞行控制状态机系统
 * ============================================================================
 * 
 * 文件：PX4CtrlFSM.cpp
 * 作者：原始框架 + 用户自定义修改
 * 功能：基于有限状态机的PX4无人机智能飞行控制系统
 * 
 * 核心特性：
 * ├── 五状态有限状态机设计
 * │   ├── MANUAL_CTRL   : 手动控制模式（遥控器直接控制）
 * │   ├── AUTO_HOVER    : 自动悬停模式（支持键盘控制）
 * │   ├── CMD_CTRL      : 指令跟踪模式（外部轨迹跟踪）
 * │   ├── AUTO_TAKEOFF  : 自动起飞模式（三阶段起飞）
 * │   └── AUTO_LAND     : 自动降落模式（安全着陆）
 * 
 * ├── 多层安全保护机制
 * │   ├── 遥控器优先级保护（一键切换到手动模式）
 * │   ├── 高度限制保护（最低-0.3m，最高2.0m）
 * │   ├── 通信超时保护（自动切换到安全状态）
 * │   ├── 着陆检测保护（智能着陆状态识别）
 * │   └── 状态转换保护（合法状态转换检查）
 * 
 * ├── 先进控制算法
 * │   ├── 几何控制器（高精度姿态控制）
 * │   ├── 坐标系转换（机体-世界坐标系）
 * │   ├── 平滑轨迹生成（起飞降落曲线）
 * │   └── 实时位置估计（多传感器融合）
 * 
 * └── 用户增强功能
 *     ├── WASD键盘实时控制（直观操作界面）
 *     ├── 一键起飞降落（简化操作流程）
 *     ├── 参数自定义调整（移动步长、高度限制）
 *     └── 详细状态反馈（实时监控信息）
 * 
 * 技术架构：
 * ├── ROS/MAVROS通信框架
 * ├── Eigen数学库（矩阵运算）
 * ├── PX4飞控固件接口
 * └── 多线程安全设计
 * 
 * 修改历史：
 * v1.0 - 原始框架实现
 * v2.0 - 增加键盘控制功能
 * v2.1 - 优化安全机制和状态转换
 * v2.2 - 参数调优（移动距离0.3m，高度限制2m）
 * v2.3 - 添加详细代码注释和文档
 * 
 * 编译依赖：
 * - ROS Noetic/Melodic
 * - MAVROS
 * - Eigen3
 * - geometry_msgs
 * - quadrotor_msgs
 * 
 * 使用说明：
 * 1. 确保PX4飞控连接正常
 * 2. 启动MAVROS通信节点
 * 3. 运行px4ctrl节点
 * 4. 通过遥控器或键盘进行控制
 * 
 * 安全提醒：
 * ⚠️  首次使用请在空旷安全区域测试
 * ⚠️  务必保持遥控器开机作为紧急停止手段  
 * ⚠️  检查所有参数设置是否符合实际飞行环境
 * ⚠️  建议先进行仿真测试再进行实物飞行
 * 
 * ============================================================================
 */

#include "PX4CtrlFSM.h"
#include <uav_utils/converters.h>  // UAV工具库，用于坐标转换等
#include <std_msgs/String.h>       // ROS标准字符串消息类型
#include <geometry_msgs/Twist.h>   // ROS几何消息类型，用于键盘控制输入

using namespace std;
using namespace uav_utils;

/**
 * @brief PX4CtrlFSM构造函数
 * @param param_ 系统参数引用
 * @param controller_ 线性控制器引用
 * 
 * 功能：初始化状态机，设置初始状态为手动控制模式
 */
PX4CtrlFSM::PX4CtrlFSM(Parameter_t &param_, LinearControl &controller_) : param(param_), controller(controller_) /*, thrust_curve(thrust_curve_)*/
{
	// 系统启动时默认为手动控制模式，由遥控器完全控制
	state = MANUAL_CTRL;
	
	// 初始化悬停目标位置为零点(0,0,0,0) -> (x,y,z,yaw)
	hover_pose.setZero();
	
	// 【新增】移动控制变量初始化
	move_locked = false;              // 移动锁定状态：false=未锁定，true=正在移动中
	land_pending = false;             // 降落挂起状态：false=无降落请求，true=等待降落
	move_start_time = ros::Time::now(); // 记录移动开始时间，用于超时保护
}

/* 
 * ============================================================================
 *                             五状态有限状态机设计
 * ============================================================================
 * 
 * 状态转换图：
 *                    system start (系统启动)
 *                         |
 *                         |
 *                         v
 *              ----- > MANUAL_CTRL <-----------------  (手动控制 - 遥控器完全控制)
 *              |         ^   |    \                 |
 *              |         |   |     \                |
 *              |         |   |      > AUTO_TAKEOFF  |  (自动起飞)
 *              |         |   |        /             |
 *              |         |   |       /              |
 *              |         |   |      /               |
 *              |         |   v     /                |
 *              |       AUTO_HOVER <                 |  (自动悬停 - 支持遥控器和键盘控制)
 *              |         ^   |  \  \                |
 *              |         |   |   \  \               |
 *              |         |   |    > AUTO_LAND -------  (自动降落)
 *              |         |   |
 *              |         |   v
 *              -------- CMD_CTRL                       (命令控制 - 轨迹跟踪)
 *
 * 状态说明：
 * - MANUAL_CTRL (L1)：手动控制，PX4飞控由遥控器直接控制
 * - AUTO_HOVER (L2)：自动悬停，支持遥控器位置控制和键盘控制
 * - CMD_CTRL (L3)：命令控制，执行外部轨迹规划指令
 * - AUTO_TAKEOFF：自动起飞流程，包含电机预热、上升、到达目标高度
 * - AUTO_LAND：自动降落流程，包含下降、着陆检测、电机上锁
 * ============================================================================
 */

/**
 * @brief 主控制循环函数 - 状态机核心处理函数
 * 
 * 功能概述：
 * 1. 执行状态机逻辑，处理状态转换
 * 2. 推力模型估计和控制律计算
 * 3. 发布控制指令到PX4飞控
 * 4. 执行着陆检测和安全检查
 * 5. 清理临时标志位
 * 
 * 调用频率：高频调用（通常50-100Hz），确保实时控制性能
 */
void PX4CtrlFSM::process()
{
	// 获取当前时间戳，用于超时检测和时间相关计算
	ros::Time now_time = ros::Time::now();
	
	// 控制输出结构体：包含期望姿态四元数、角速度、推力等
	Controller_Output_t u;
	
	// 期望状态结构体：包含期望位置、速度、加速度、jerk、偏航角等
	Desired_State_t des(odom_data);
	
	// 降落过程中电机低速运行标志
	bool rotor_low_speed_during_land = false;

	// ============================================================================
	// STEP1: 状态机核心逻辑 - 根据当前状态执行相应的控制逻辑
	// ============================================================================
	switch (state)
	{
	// ========================================================================
	// 状态1：MANUAL_CTRL - 手动控制模式
	// 功能：PX4飞控完全由遥控器控制，px4ctrl程序监控状态转换条件
	// ========================================================================
	case MANUAL_CTRL:
	{
		// 检查是否请求进入悬停模式（遥控器5通道拨到内侧）
		if (rc_data.enter_hover_mode) // Try to jump to AUTO_HOVER
		{
			// 安全检查1：必须有里程计数据（位置定位）
			if (!odom_is_received(now_time))
			{
				ROS_ERROR("[px4ctrl] Reject AUTO_HOVER(L2). No odom!");
				break;
			}
			
			// 安全检查2：进入悬停前不能有外部轨迹命令
			if (cmd_is_received(now_time))
			{
				ROS_ERROR("[px4ctrl] Reject AUTO_HOVER(L2). You are sending commands before toggling into AUTO_HOVER, which is not allowed. Stop sending commands now!");
				break;
			}
			
			// 安全检查3：速度不能过大（可能是定位模块异常）
			if (odom_data.v.norm() > 3.0)
			{
				ROS_ERROR("[px4ctrl] Reject AUTO_HOVER(L2). Odom_Vel=%fm/s, which seems that the locolization module goes wrong!", odom_data.v.norm());
				break;
			}

			// 安全检查通过，执行状态转换
			state = AUTO_HOVER;                    // 切换到悬停状态
			controller.resetThrustMapping();       // 重置推力映射模型
			set_hov_with_odom();                  // 设置当前位置为悬停目标点
			toggle_offboard_mode(true);           // 切换PX4到OFFBOARD模式

			ROS_INFO("\033[32m[px4ctrl] MANUAL_CTRL(L1) --> AUTO_HOVER(L2)\033[32m");
		}
		// 检查是否请求自动起飞（通过takeoff.sh脚本触发）
		else if (param.takeoff_land.enable && takeoff_land_data.triggered && takeoff_land_data.takeoff_land_cmd == quadrotor_msgs::TakeoffLand::TAKEOFF) // Try to jump to AUTO_TAKEOFF
		{
			// 起飞安全检查1：必须有位置信息
			if (!odom_is_received(now_time))
			{
				ROS_ERROR("[px4ctrl] Reject AUTO_TAKEOFF. No odom!");
				break;
			}
			
			// 起飞安全检查2：起飞前不能有外部命令
			if (cmd_is_received(now_time))
			{
				ROS_ERROR("[px4ctrl] Reject AUTO_TAKEOFF. You are sending commands before toggling into AUTO_TAKEOFF, which is not allowed. Stop sending commands now!");
				break;
			}
			
			// 起飞安全检查3：必须静止起飞，速度不能超过0.1m/s
			if (odom_data.v.norm() > 0.1)
			{
				ROS_ERROR("[px4ctrl] Reject AUTO_TAKEOFF. Odom_Vel=%fm/s, non-static takeoff is not allowed!", odom_data.v.norm());
				break;
			}
			
			// 起飞安全检查4：着陆检测器必须确认飞机在地面
			if (!get_landed())
			{
				ROS_ERROR("[px4ctrl] Reject AUTO_TAKEOFF. land detector says that the drone is not landed now!");
				break;
			}
			
			// 起飞安全检查5：如果有遥控器连接，检查遥控器状态
			if (rc_is_received(now_time)) // Check this only if RC is connected.
			{
				// 遥控器必须在悬停模式、命令模式，且所有摇杆居中
				if (!rc_data.is_hover_mode || !rc_data.is_command_mode || !rc_data.check_centered())
				{
					ROS_ERROR("[px4ctrl] Reject AUTO_TAKEOFF. If you have your RC connected, keep its switches at \"auto hover\" and \"command control\" states, and all sticks at the center, then takeoff again.");
					
					// 等待遥控器调整到正确状态
					while (ros::ok())
					{
						ros::Duration(0.01).sleep();  // 10ms延时
						ros::spinOnce();              // 处理ROS消息
						if (rc_data.is_hover_mode && rc_data.is_command_mode && rc_data.check_centered())
						{
							ROS_INFO("\033[32m[px4ctrl] OK, you can takeoff again.\033[32m");
							break;
						}
					}
					break;
				}
			}

			// 所有安全检查通过，开始起飞流程
			state = AUTO_TAKEOFF;                           // 切换到起飞状态
			controller.resetThrustMapping();                // 重置推力映射
			set_start_pose_for_takeoff_land(odom_data);    // 记录起飞起始位置
			toggle_offboard_mode(true);                    // 进入OFFBOARD模式（在解锁前）
			
			// 等待0.1秒让PX4模式切换完成
			for (int i = 0; i < 10 && ros::ok(); ++i) // wait for 0.1 seconds to allow mode change by FMU // mark
			{
				ros::Duration(0.01).sleep();
				ros::spinOnce();
			}
			
			// 如果启用自动解锁，则解锁电机
			if (param.takeoff_land.enable_auto_arm)
			{
				toggle_arm_disarm(true);
			}
			
			// 记录起飞开始时间
			takeoff_land.toggle_takeoff_land_time = now_time;

			ROS_INFO("\033[32m[px4ctrl] MANUAL_CTRL(L1) --> AUTO_TAKEOFF\033[32m");
		}

		// 检查是否请求重启飞控（当EKF2状态估计器异常时）
		if (rc_data.toggle_reboot) // Try to reboot. EKF2 based PX4 FCU requires reboot when its state estimator goes wrong.
		{
			// 安全检查：只有在上锁状态下才能重启
			if (state_data.current_state.armed)
			{
				ROS_ERROR("[px4ctrl] Reject reboot! Disarm the drone first!");
				break;
			}
			reboot_FCU();  // 执行飞控重启
		}

		break;  // MANUAL_CTRL状态处理结束
	}

	// ========================================================================
	// 状态2：AUTO_HOVER - 自动悬停模式
	// 功能：保持在指定位置悬停，支持遥控器位置调整和键盘控制
	// ========================================================================
	case AUTO_HOVER:
	{
		// 退出条件1：遥控器切换到非悬停模式 或 失去位置信息
		if (!rc_data.is_hover_mode || !odom_is_received(now_time))
		{
			state = MANUAL_CTRL;                    // 退回手动控制
			toggle_offboard_mode(false);           // 退出OFFBOARD模式
			
			// 【新增】清除移动控制相关状态
			move_locked = false;                   // 解除移动锁定
			land_pending = false;                  // 清除降落挂起

			ROS_WARN("[px4ctrl] AUTO_HOVER(L2) --> MANUAL_CTRL(L1)");
		}
		
		// 状态转换条件1：进入轨迹跟踪模式
		else if (rc_data.is_command_mode && cmd_is_received(now_time))
		{
			// 必须在OFFBOARD模式下才能进入CMD_CTRL
			if (state_data.current_state.mode == "OFFBOARD")
			{
				state = CMD_CTRL;                  // 切换到命令控制状态
				des = get_cmd_des();              // 获取轨迹命令期望状态
				
				// 【新增】清除移动控制相关状态
				move_locked = false;
				land_pending = false;
				
				ROS_INFO("\033[32m[px4ctrl] AUTO_HOVER(L2) --> CMD_CTRL(L3)\033[32m");
			}
		}
		
		// 状态转换条件2：请求降落
		else if (takeoff_land_data.triggered && takeoff_land_data.takeoff_land_cmd == quadrotor_msgs::TakeoffLand::LAND)
		{
			state = AUTO_LAND;                           // 切换到降落状态
			set_start_pose_for_takeoff_land(odom_data); // 记录降落开始位置
			
			// 【新增】清除移动控制相关状态
			move_locked = false;
			land_pending = false;

			ROS_INFO("\033[32m[px4ctrl] AUTO_HOVER(L2) --> AUTO_LAND\033[32m");
		}
		// 正常悬停控制逻辑
		else
		{
			// 【新增功能】遥控器优先级控制机制（紧急避险设计）
			bool rc_has_input = false;
			
			if (rc_is_received(now_time)) {
				// 检查遥控器摇杆是否有显著偏移（超过5%阈值）
				double rc_threshold = 0.05; // 5%的摇杆偏移阈值
				if (fabs(rc_data.ch[0]) > rc_threshold || fabs(rc_data.ch[1]) > rc_threshold || 
					fabs(rc_data.ch[2]) > rc_threshold || fabs(rc_data.ch[3]) > rc_threshold) {
					rc_has_input = true;
				}
			}
			
			// 控制优先级处理
			if (rc_has_input) {
				// 优先级1：遥控器输入（紧急避险）
				// 当遥控器有输入时，立即接管控制，用于紧急情况
				set_hov_with_rc();                 // 根据遥控器输入调整悬停位置
				move_locked = false;               // 中断自动移动控制
				land_pending = false;              // 取消降落挂起状态
				ROS_WARN_THROTTLE(2.0, "[px4ctrl] RC override - Move command interrupted");
			} else {
				// 优先级2：键盘/自动移动控制
				// 只有在遥控器无输入时才处理其他控制指令
				processMoveCmd();                  // 处理移动命令（键盘控制等）
			}
			
			// 生成悬停期望状态
			des = get_hover_des();
			
			// 轨迹触发机制：允许外部轨迹规划器开始工作
			if ((rc_data.enter_command_mode) ||
				(takeoff_land.delay_trigger.first && now_time > takeoff_land.delay_trigger.second))
			{
				takeoff_land.delay_trigger.first = false;
				publish_trigger(odom_data.msg);    // 发布触发信号给轨迹规划器
				ROS_INFO("\033[32m[px4ctrl] TRIGGER sent, allow user command.\033[32m");
			}

			// cout << "des.p=" << des.p.transpose() << endl;  // 调试输出（已注释）
		}

		break;  // AUTO_HOVER状态处理结束
	}

	// ========================================================================
	// 状态3：CMD_CTRL - 命令控制模式（轨迹跟踪）
	// 功能：执行外部轨迹规划器发送的轨迹指令
	// ========================================================================
	case CMD_CTRL:
	{
		// 退出条件1：遥控器切换到非悬停模式 或 失去位置信息
		if (!rc_data.is_hover_mode || !odom_is_received(now_time))
		{
			state = MANUAL_CTRL;           // 紧急退回手动控制
			toggle_offboard_mode(false);   // 退出OFFBOARD模式

			ROS_WARN("[px4ctrl] From CMD_CTRL(L3) to MANUAL_CTRL(L1)!");
		}
		
		// 退出条件2：遥控器切换到非命令模式 或 轨迹命令超时
		else if (!rc_data.is_command_mode || !cmd_is_received(now_time))
		{
			state = AUTO_HOVER;            // 退回悬停模式
			set_hov_with_odom();          // 设置当前位置为悬停目标
			des = get_hover_des();        // 生成悬停期望状态
			ROS_INFO("[px4ctrl] From CMD_CTRL(L3) to AUTO_HOVER(L2)!");
		}
		
		// 正常轨迹跟踪
		else
		{
			des = get_cmd_des();          // 获取外部轨迹命令的期望状态
		}

		// 安全限制：在CMD_CTRL模式下不允许直接降落
		// 必须先回到AUTO_HOVER模式才能降落
		if (takeoff_land_data.triggered && takeoff_land_data.takeoff_land_cmd == quadrotor_msgs::TakeoffLand::LAND)
		{
			ROS_ERROR("[px4ctrl] Reject AUTO_LAND, which must be triggered in AUTO_HOVER. \
					Stop sending control commands for longer than %fs to let px4ctrl return to AUTO_HOVER first.",
					  param.msg_timeout.cmd);
		}

		break;  // CMD_CTRL状态处理结束
	}

	// ========================================================================
	// 状态4：AUTO_TAKEOFF - 自动起飞模式
	// 功能：安全的自动起飞流程，包含电机预热、上升、到达目标高度
	// ========================================================================
	case AUTO_TAKEOFF:
	{
		// 起飞阶段1：电机预热阶段（前3秒）
		// 让螺旋桨缓慢转动但不产生足够升力，给操作员反应时间
		if ((now_time - takeoff_land.toggle_takeoff_land_time).toSec() < AutoTakeoffLand_t::MOTORS_SPEEDUP_TIME) // Wait for several seconds to warn prople.
		{
			des = get_rotor_speed_up_des(now_time);  // 生成电机预热期望状态
		}
		
		// 起飞阶段3：到达目标高度，起飞完成
		else if (odom_data.p(2) >= (takeoff_land.start_pose(2) + param.takeoff_land.height)) // reach the desired height
		{
			state = AUTO_HOVER;                      // 切换到悬停模式
			set_hov_with_odom();                    // 设置当前位置为悬停目标
			ROS_INFO("\033[32m[px4ctrl] AUTO_TAKEOFF --> AUTO_HOVER(L2)\033[32m");

			// 设置延迟触发，2秒后允许外部轨迹规划器工作
			takeoff_land.delay_trigger.first = true;
			takeoff_land.delay_trigger.second = now_time + ros::Duration(AutoTakeoffLand_t::DELAY_TRIGGER_TIME);
		}
		
		// 起飞阶段2：正常上升阶段
		else
		{
			des = get_takeoff_land_des(param.takeoff_land.speed);  // 以设定速度上升
		}

		break;  // AUTO_TAKEOFF状态处理结束
	}

	// ========================================================================
	// 状态5：AUTO_LAND - 自动降落模式
	// 功能：安全的自动降落流程，包含下降、着陆检测、电机上锁
	// ========================================================================
	case AUTO_LAND:
	{
		// 紧急退出条件1：遥控器切换到非悬停模式 或 失去位置信息
		if (!rc_data.is_hover_mode || !odom_is_received(now_time))
		{
			state = MANUAL_CTRL;           // 紧急退回手动控制
			toggle_offboard_mode(false);   // 退出OFFBOARD模式

			ROS_WARN("[px4ctrl] From AUTO_LAND to MANUAL_CTRL(L1)!");
		}
		
		// 退出条件2：遥控器切换到非命令模式（用户取消降落）
		else if (!rc_data.is_command_mode)
		{
			state = AUTO_HOVER;            // 退回悬停模式
			set_hov_with_odom();          // 设置当前位置为悬停目标
			des = get_hover_des();        // 生成悬停期望状态
			ROS_INFO("[px4ctrl] From AUTO_LAND to AUTO_HOVER(L2)!");
		}
		
		// 降落阶段1：还未着陆，继续下降
		else if (!get_landed())
		{
			des = get_takeoff_land_des(-param.takeoff_land.speed);  // 负速度表示下降
		}
		
		// 降落阶段2：已着陆，准备上锁
		else
		{
			rotor_low_speed_during_land = true;  // 设置电机低速运行标志

			// 提示信息（只打印一次）
			static bool print_once_flag = true;
			if (print_once_flag)
			{
				ROS_INFO("\033[32m[px4ctrl] Wait for abount 10s to let the drone arm.\033[32m");
				print_once_flag = false;
			}

			// 等待PX4确认着陆状态，然后上锁电机
			if (extended_state_data.current_extended_state.landed_state == mavros_msgs::ExtendedState::LANDED_STATE_ON_GROUND) // PX4 allows disarm after this
			{
				// 避免频繁调用上锁服务
				static double last_trial_time = 0; // Avoid too frequent calls
				if (now_time.toSec() - last_trial_time > 1.0)
				{
					if (toggle_arm_disarm(false)) // disarm - 上锁电机
					{
						print_once_flag = true;
						state = MANUAL_CTRL;           // 回到手动控制模式
						toggle_offboard_mode(false);   // 退出OFFBOARD模式
						ROS_INFO("\033[32m[px4ctrl] AUTO_LAND --> MANUAL_CTRL(L1)\033[32m");
					}

					last_trial_time = now_time.toSec();
				}
			}
		}

		break;  // AUTO_LAND状态处理结束
	}

	default:
		break;
	}

	// ============================================================================
	// STEP2: 推力模型估计
	// 功能：在悬停和命令控制模式下，根据IMU加速度估计推力模型参数
	// ============================================================================
	if (state == AUTO_HOVER || state == CMD_CTRL)
	{
		// 推力模型估计，用于补偿电池电压变化、螺旋桨老化等因素
		// controller.estimateThrustModel(imu_data.a, bat_data.volt, param);  // 旧版本（包含电池电压）
		controller.estimateThrustModel(imu_data.a,param);  // 当前版本
	}

	// ============================================================================
	// STEP3: 控制律计算和期望状态处理
	// 功能：根据期望状态计算所需的姿态、推力等控制量
	// ============================================================================
	if (rotor_low_speed_during_land) // used at the start of auto takeoff
	{
		// 特殊情况：降落过程中电机低速运行
		motors_idling(imu_data, u);   // 设置电机怠速参数
	}
	else
	{
		// 正常情况：通过PID控制器计算控制输出
		debug_msg = controller.calculateControl(des, odom_data, imu_data, u);
		debug_msg.header.stamp = now_time;
		debug_pub.publish(debug_msg);  // 发布调试信息
	}

	// ============================================================================
	// STEP4: 发布控制指令到PX4飞控
	// 功能：将计算得到的控制量通过MAVROS发送给PX4
	// ============================================================================
	if (param.use_bodyrate_ctrl)
	{
		// 方式1：发布角速度控制指令（更直接的控制方式）
		publish_bodyrate_ctrl(u, now_time);
	}
	else
	{
		// 方式2：发布姿态控制指令（让PX4内部控制器处理角速度）
		publish_attitude_ctrl(u, now_time);
	}

	// ============================================================================
	// STEP5: 着陆检测
	// 功能：检测无人机是否已经着陆，用于自动降落流程
	// ============================================================================
	land_detector(state, des, odom_data);
	// cout << takeoff_land.landed << " ";  // 调试输出（已注释）
	// fflush(stdout);

	// ============================================================================
	// STEP6: 清理临时标志位
	// 功能：清除本次控制周期的临时标志，避免重复触发
	// ============================================================================
	rc_data.enter_hover_mode = false;      // 清除进入悬停模式标志
	rc_data.enter_command_mode = false;    // 清除进入命令模式标志
	rc_data.toggle_reboot = false;         // 清除重启请求标志
	takeoff_land_data.triggered = false;   // 清除起飞降落触发标志
}

// ============================================================================
//                            实用工具函数实现
// ============================================================================

/**
 * @brief 设置电机怠速状态的控制输出
 * @param imu IMU数据，提供当前四元数姿态
 * @param u 控制输出结构体，将被设置为怠速状态
 * 
 * 功能：配置电机怠速时的控制参数
 * 应用场景：
 *   - 刚解锁后等待起飞指令时
 *   - 降落后保持最小推力维持解锁状态
 *   - 紧急情况下快速切换到安全状态
 * 
 * 参数设置：
 *   - 姿态保持：使用当前IMU姿态，防止突变
 *   - 角速度：清零，停止所有旋转
 *   - 推力：4%最小推力，保持解锁但不起飞
 */
void PX4CtrlFSM::motors_idling(const Imu_Data_t &imu, Controller_Output_t &u)
{
	u.q = imu.q;                              // 保持当前姿态，避免姿态跳变
	u.bodyrates = Eigen::Vector3d::Zero();    // 角速度置零，停止旋转
	u.thrust = 0.04;                          // 4%推力，维持解锁状态但不起飞
}

/**
 * @brief 着陆检测器 - 智能判断无人机是否已着陆
 * @param state 当前FSM状态
 * @param des 期望状态（目标位置、速度等）
 * @param odom 里程计数据（实际位置、速度等）
 * 
 * 功能：通过多重约束条件判断无人机着陆状态
 * 
 * 着陆判断逻辑：
 *   约束1：位置偏差约束 - 目标高度比实际高度低0.5m以上
 *   约束2：速度约束 - 整体速度低于0.1m/s
 *   约束3：时间约束 - 前两个约束需持续满足3秒以上
 * 
 * 特殊情况处理：
 *   - 手动模式且未解锁：直接判定为已着陆
 *   - 从手动模式切换到自动模式：重置着陆状态为false
 * 
 * 应用场景：
 *   - 自动降落完成检测
 *   - 安全状态确认
 *   - 下一步操作的前提条件判断
 */
void PX4CtrlFSM::land_detector(const State_t state, const Desired_State_t &des, const Odom_Data_t &odom)
{
	// 状态切换检测：从手动模式进入自动模式时重置着陆状态
	static State_t last_state = State_t::MANUAL_CTRL;
	if (last_state == State_t::MANUAL_CTRL && (state == State_t::AUTO_HOVER || state == State_t::AUTO_TAKEOFF))
	{
		takeoff_land.landed = false; // 进入自动模式时总是重置为未着陆状态
	}
	last_state = state;

	// 特殊情况：手动模式且未解锁，直接判定为已着陆
	if (state == State_t::MANUAL_CTRL && !state_data.current_state.armed)
	{
		takeoff_land.landed = true;
		return; // 无需进行其他判断
	}

	// ========================================================================
	// 着陆检测参数定义
	// ========================================================================
	constexpr double POSITION_DEVIATION_C = -0.5; // 约束1：目标位置需比实际位置低0.5米以上
	constexpr double VELOCITY_THR_C = 0.1;		  // 约束2：速度阈值 0.1m/s
	constexpr double TIME_KEEP_C = 3.0;			  // 约束3：约束1&2需持续满足的时间 3秒

	// 时间记录和状态追踪静态变量
	static ros::Time time_C12_reached; // 约束1&2同时满足的起始时间
	static bool is_last_C12_satisfy;   // 上一次约束1&2是否满足

	// 如果已经着陆，重置时间记录
	if (takeoff_land.landed)
	{
		time_C12_reached = ros::Time::now();
		is_last_C12_satisfy = false;
	}
	else
	{
		// ========================================================================
		// 核心着陆判断逻辑
		// ========================================================================
		
		// 检查约束1&2是否同时满足
		bool C12_satisfy = (des.p(2) - odom.p(2)) < POSITION_DEVIATION_C && // 约束1：高度偏差
						   odom.v.norm() < VELOCITY_THR_C;                   // 约束2：速度阈值
		
		// 约束1&2刚开始满足：记录起始时间
		if (C12_satisfy && !is_last_C12_satisfy)
		{
			time_C12_reached = ros::Time::now();
		}
		// 约束1&2持续满足：检查时间约束
		else if (C12_satisfy && is_last_C12_satisfy)
		{
			// 约束3检查：持续时间是否超过阈值
			if ((ros::Time::now() - time_C12_reached).toSec() > TIME_KEEP_C) 
			{
				takeoff_land.landed = true; // 所有约束满足，判定为已着陆
			}
		}

		// 更新约束1&2的满足状态
		is_last_C12_satisfy = C12_satisfy;
	}
}

/**
 * @brief 生成悬停期望状态
 * @return Desired_State_t 悬停模式的期望状态
 * 
 * 功能：将悬停目标转换为控制器可用的期望状态
 * 应用场景：AUTO_HOVER状态下的目标状态生成
 * 
 * 状态设置：
 *   - 位置：使用hover_pose前3个分量 (x,y,z)
 *   - 速度/加速度/加加速度：全部置零（悬停特征）
 *   - 偏航角：使用hover_pose第4个分量
 *   - 偏航角速度：置零（保持偏航角稳定）
 */
Desired_State_t PX4CtrlFSM::get_hover_des()
{
	Desired_State_t des;
	des.p = hover_pose.head<3>();               // 目标位置：取hover_pose的前3个元素 (x,y,z)
	des.v = Eigen::Vector3d::Zero();            // 目标速度：0（悬停状态）
	des.a = Eigen::Vector3d::Zero();            // 目标加速度：0（匀速状态）
	des.j = Eigen::Vector3d::Zero();            // 目标加加速度：0（平滑运动）
	des.yaw = hover_pose(3);                    // 目标偏航角：hover_pose第4个元素
	des.yaw_rate = 0.0;                         // 目标偏航角速度：0（保持偏航稳定）

	return des;
}

/**
 * @brief 生成指令跟踪期望状态
 * @return Desired_State_t 指令跟踪模式的期望状态
 * 
 * 功能：将外部轨迹指令转换为控制器期望状态
 * 应用场景：CMD_CTRL状态下跟踪外部轨迹规划
 * 
 * 数据来源：cmd_data结构体（通过ROS话题接收）
 * 特点：直接传递所有轨迹信息，支持复杂机动飞行
 */
Desired_State_t PX4CtrlFSM::get_cmd_des()
{
	Desired_State_t des;
	des.p = cmd_data.p;                         // 目标位置：直接使用指令位置
	des.v = cmd_data.v;                         // 目标速度：直接使用指令速度
	des.a = cmd_data.a;                         // 目标加速度：直接使用指令加速度
	des.j = cmd_data.j;                         // 目标加加速度：直接使用指令加加速度
	des.yaw = cmd_data.yaw;                     // 目标偏航角：直接使用指令偏航角
	des.yaw_rate = cmd_data.yaw_rate;           // 目标偏航角速度：直接使用指令偏航角速度

	return des;
}

/**
 * @brief 生成电机加速期望状态
 * @param now 当前时间
 * @return Desired_State_t 电机加速阶段的期望状态
 * 
 * 功能：为起飞时的电机预加速阶段生成平滑的期望状态
 * 应用场景：AUTO_TAKEOFF状态的初始阶段，电机转速逐渐提升
 * 
 * 算法特点：
 *   - 使用指数函数生成平滑的垂直加速度曲线
 *   - 保持起始位置和偏航角不变
 *   - 安全限制：垂直加速度不超过0.1m/s²
 */
Desired_State_t PX4CtrlFSM::get_rotor_speed_up_des(const ros::Time now)
{
	// 计算自起飞开始的时间差
	double delta_t = (now - takeoff_land.toggle_takeoff_land_time).toSec();
	
	// 使用指数函数生成垂直加速度曲线
	// 参数6.0和7.0是经验值，产生满意的加速度曲线
	double des_a_z = exp((delta_t - AutoTakeoffLand_t::MOTORS_SPEEDUP_TIME) * 6.0) * 7.0 - 7.0;
	
	// 安全检查：垂直加速度不应过大
	if (des_a_z > 0.1)
	{
		ROS_ERROR("des_a_z > 0.1!, des_a_z=%f", des_a_z);
		des_a_z = 0.0;
	}

	Desired_State_t des;
	des.p = takeoff_land.start_pose.head<3>();      // 位置：保持起飞起始位置
	des.v = Eigen::Vector3d::Zero();                // 速度：零（还未开始移动）
	des.a = Eigen::Vector3d(0, 0, des_a_z);         // 加速度：垂直方向平滑增加
	des.j = Eigen::Vector3d::Zero();                // 加加速度：零
	des.yaw = takeoff_land.start_pose(3);           // 偏航角：保持起飞时的偏航角
	des.yaw_rate = 0.0;                             // 偏航角速度：零

	return des;
}

/**
 * @brief 生成起飞/降落期望状态
 * @param speed 垂直运动速度（正值=起飞，负值=降落）
 * @return Desired_State_t 起飞或降落过程的期望状态
 * 
 * 功能：为起飞/降落阶段生成匀速垂直运动的期望状态
 * 应用场景：
 *   - speed > 0：起飞阶段，向上匀速运动
 *   - speed < 0：降落阶段，向下匀速运动
 * 
 * 算法特点：
 *   - 基于起始位置和恒定垂直速度计算目标位置
 *   - 保持水平位置和偏航角不变
 *   - 起飞时需要额外考虑电机预加速时间
 */
Desired_State_t PX4CtrlFSM::get_takeoff_land_des(const double speed)
{
	ros::Time now = ros::Time::now();
	
	// 计算有效运动时间
	// 起飞时(speed>0)需要减去电机预加速时间，降落时不需要
	double delta_t = (now - takeoff_land.toggle_takeoff_land_time).toSec() - 
					 (speed > 0 ? AutoTakeoffLand_t::MOTORS_SPEEDUP_TIME : 0);

	Desired_State_t des;
	// 位置：起始位置 + 垂直位移
	des.p = takeoff_land.start_pose.head<3>() + Eigen::Vector3d(0, 0, speed * delta_t);
	des.v = Eigen::Vector3d(0, 0, speed);           // 速度：垂直方向恒定速度
	des.a = Eigen::Vector3d::Zero();                // 加速度：零（匀速运动）
	des.j = Eigen::Vector3d::Zero();                // 加加速度：零
	des.yaw = takeoff_land.start_pose(3);           // 偏航角：保持起始偏航角
	des.yaw_rate = 0.0;                             // 偏航角速度：零

	return des;
}

/**
 * @brief 使用里程计数据设置悬停位置
 * 
 * 功能：将当前实际位置设为新的悬停目标
 * 应用场景：
 *   - 状态切换时同步悬停目标与当前位置
 *   - 避免位置跳变导致的剧烈运动
 *   - 为后续键盘控制建立基准位置
 * 
 * 数据来源：odom_data（里程计/定位系统）
 */
void PX4CtrlFSM::set_hov_with_odom()
{
	hover_pose.head<3>() = odom_data.p;                            // 位置：使用当前实际位置
	hover_pose(3) = get_yaw_from_quaternion(odom_data.q);          // 偏航角：从四元数提取当前偏航角

	last_set_hover_pose_time = ros::Time::now();                   // 记录设置时间
}

/**
 * @brief 使用遥控器数据更新悬停位置
 * 
 * 功能：根据遥控器摇杆输入实时调整悬停目标位置
 * 应用场景：手动飞行模式下的位置控制
 * 
 * 控制映射：
 *   - ch[0] (右摇杆左右) -> Y轴位移 (横滚控制)
 *   - ch[1] (右摇杆前后) -> X轴位移 (俯仰控制)  
 *   - ch[2] (左摇杆上下) -> Z轴位移 (油门控制)
 *   - ch[3] (左摇杆左右) -> 偏航角调整 (偏航控制)
 * 
 * 安全特性：
 *   - 速度限制：max_manual_vel参数限制最大手动速度
 *   - 高度保护：最低飞行高度限制为-0.3m
 *   - 通道反向：支持遥控器通道反向配置
 */
void PX4CtrlFSM::set_hov_with_rc()
{
	ros::Time now = ros::Time::now();
	double delta_t = (now - last_set_hover_pose_time).toSec();     // 计算时间间隔
	last_set_hover_pose_time = now;

	// 根据遥控器输入更新悬停位置
	// 位移 = 摇杆量 × 最大速度 × 时间间隔 × 反向系数
	hover_pose(0) += rc_data.ch[1] * param.max_manual_vel * delta_t * (param.rc_reverse.pitch ? 1 : -1);     // X轴(俯仰)
	hover_pose(1) += rc_data.ch[0] * param.max_manual_vel * delta_t * (param.rc_reverse.roll ? 1 : -1);      // Y轴(横滚)
	hover_pose(2) += rc_data.ch[2] * param.max_manual_vel * delta_t * (param.rc_reverse.throttle ? 1 : -1);  // Z轴(油门)
	hover_pose(3) += rc_data.ch[3] * param.max_manual_vel * delta_t * (param.rc_reverse.yaw ? 1 : -1);       // 偏航角

	// 高度安全限制：最低-0.3米
	if (hover_pose(2) < -0.3)
		hover_pose(2) = -0.3;

	// 调试信息输出（已注释）
	// if (param.print_dbg)
	// {
	// 	static unsigned int count = 0;
	// 	if (count++ % 100 == 0)
	// 	{
	// 		cout << "hover_pose=" << hover_pose.transpose() << endl;
	// 		cout << "ch[0~3]=" << rc_data.ch[0] << " " << rc_data.ch[1] << " " << rc_data.ch[2] << " " << rc_data.ch[3] << endl;
	// 	}
	// }
}

void PX4CtrlFSM::set_start_pose_for_takeoff_land(const Odom_Data_t &odom)
{
	takeoff_land.start_pose.head<3>() = odom_data.p;
	takeoff_land.start_pose(3) = get_yaw_from_quaternion(odom_data.q);

	takeoff_land.toggle_takeoff_land_time = ros::Time::now();
}

bool PX4CtrlFSM::rc_is_received(const ros::Time &now_time)
{
	return (now_time - rc_data.rcv_stamp).toSec() < param.msg_timeout.rc;
}

bool PX4CtrlFSM::cmd_is_received(const ros::Time &now_time)
{
	return (now_time - cmd_data.rcv_stamp).toSec() < param.msg_timeout.cmd;
}

bool PX4CtrlFSM::odom_is_received(const ros::Time &now_time)
{
	return (now_time - odom_data.rcv_stamp).toSec() < param.msg_timeout.odom;
}

bool PX4CtrlFSM::imu_is_received(const ros::Time &now_time)
{
	return (now_time - imu_data.rcv_stamp).toSec() < param.msg_timeout.imu;
}

bool PX4CtrlFSM::bat_is_received(const ros::Time &now_time)
{
	return (now_time - bat_data.rcv_stamp).toSec() < param.msg_timeout.bat;
}

bool PX4CtrlFSM::recv_new_odom()
{
	if (odom_data.recv_new_msg)
	{
		odom_data.recv_new_msg = false;
		return true;
	}

	return false;
}

void PX4CtrlFSM::publish_bodyrate_ctrl(const Controller_Output_t &u, const ros::Time &stamp)
{
	mavros_msgs::AttitudeTarget msg;

	msg.header.stamp = stamp;
	msg.header.frame_id = std::string("FCU");

	msg.type_mask = mavros_msgs::AttitudeTarget::IGNORE_ATTITUDE;

	msg.body_rate.x = u.bodyrates.x();
	msg.body_rate.y = u.bodyrates.y();
	msg.body_rate.z = u.bodyrates.z();

	msg.thrust = u.thrust;

	ctrl_FCU_pub.publish(msg);
}

void PX4CtrlFSM::publish_attitude_ctrl(const Controller_Output_t &u, const ros::Time &stamp)
{
	mavros_msgs::AttitudeTarget msg;

	msg.header.stamp = stamp;
	msg.header.frame_id = std::string("FCU");

	msg.type_mask = mavros_msgs::AttitudeTarget::IGNORE_ROLL_RATE |
					mavros_msgs::AttitudeTarget::IGNORE_PITCH_RATE |
					mavros_msgs::AttitudeTarget::IGNORE_YAW_RATE;

	msg.orientation.x = u.q.x();
	msg.orientation.y = u.q.y();
	msg.orientation.z = u.q.z();
	msg.orientation.w = u.q.w();

	msg.thrust = u.thrust;

	ctrl_FCU_pub.publish(msg);
}

void PX4CtrlFSM::publish_trigger(const nav_msgs::Odometry &odom_msg)
{
	geometry_msgs::PoseStamped msg;
	msg.header.frame_id = "world";
	msg.pose = odom_msg.pose.pose;

	traj_start_trigger_pub.publish(msg);
}

bool PX4CtrlFSM::toggle_offboard_mode(bool on_off)
{
	mavros_msgs::SetMode offb_set_mode;

	if (on_off)
	{
		state_data.state_before_offboard = state_data.current_state;
		if (state_data.state_before_offboard.mode == "OFFBOARD") // Not allowed
			state_data.state_before_offboard.mode = "MANUAL";

		offb_set_mode.request.custom_mode = "OFFBOARD";
		if (!(set_FCU_mode_srv.call(offb_set_mode) && offb_set_mode.response.mode_sent))
		{
			ROS_ERROR("Enter OFFBOARD rejected by PX4!");
			return false;
		}
	}
	else
	{
		offb_set_mode.request.custom_mode = state_data.state_before_offboard.mode;
		if (!(set_FCU_mode_srv.call(offb_set_mode) && offb_set_mode.response.mode_sent))
		{
			ROS_ERROR("Exit OFFBOARD rejected by PX4!");
			return false;
		}
	}

	return true;

	// if (param.print_dbg)
	// 	printf("offb_set_mode mode_sent=%d(uint8_t)\n", offb_set_mode.response.mode_sent);
}

bool PX4CtrlFSM::toggle_arm_disarm(bool arm)
{
	mavros_msgs::CommandBool arm_cmd;
	arm_cmd.request.value = arm;
	if (!(arming_client_srv.call(arm_cmd) && arm_cmd.response.success))
	{
		if (arm)
			ROS_ERROR("ARM rejected by PX4!");
		else
			ROS_ERROR("DISARM rejected by PX4!");

		return false;
	}

	return true;
}

void PX4CtrlFSM::reboot_FCU()
{
	// https://mavlink.io/en/messages/common.html, MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN(#246)
	mavros_msgs::CommandLong reboot_srv;
	reboot_srv.request.broadcast = false;
	reboot_srv.request.command = 246; // MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN
	reboot_srv.request.param1 = 1;	  // Reboot autopilot
	reboot_srv.request.param2 = 0;	  // Do nothing for onboard computer
	reboot_srv.request.confirmation = true;

	reboot_FCU_srv.call(reboot_srv);

	ROS_INFO("Reboot FCU");

	// if (param.print_dbg)
	// 	printf("reboot result=%d(uint8_t), success=%d(uint8_t)\n", reboot_srv.response.result, reboot_srv.response.success);
}

// ============================================================================
//                           【新增】键盘控制功能实现
// ============================================================================

/**
 * @brief 键盘控制指令回调函数
 * @param msg Twist消息指针，包含线速度和角速度信息
 * 
 * 功能：处理键盘输入，实现WASD实时控制
 * 控制映射：
 *   W/S键 -> linear.x  -> 前进/后退 (±0.3m)
 *   A/D键 -> linear.y  -> 左移/右移 (±0.3m)
 *   R/F键 -> linear.z  -> 上升/下降 (±0.15m)
 *   Q/E键 -> angular.z -> 左转/右转 (±0.2rad ≈ ±11.5°)
 *   L键   -> linear.x=999 -> 降落指令
 * 
 * 安全机制：
 *   - 仅在AUTO_HOVER状态下生效
 *   - 坐标系转换：机体坐标 -> 世界坐标
 *   - 高度限制：-0.3m ~ 2.0m
 *   - 角度归一化：±π范围内
 */
void PX4CtrlFSM::keyboardCmdCallback(const geometry_msgs::Twist::ConstPtr& msg)
{
	// 安全检查：只有在AUTO_HOVER状态下才响应键盘控制
	if (state != AUTO_HOVER) {
		ROS_WARN_THROTTLE(2.0, "[px4ctrl] Keyboard control only available in AUTO_HOVER mode");
		return;
	}

	// 控制参数定义
	double step = 0.3;              // 每次水平移动的距离 (米)
	double yaw_step = 0.2;          // 每次转向的角度 (弧度，约11.5度)
	double yaw = get_yaw_from_quaternion(odom_data.q); // 获取当前偏航角 (弧度)
	
	// 运动增量变量初始化
	double dx = 0, dy = 0, dz = 0, dyaw = 0;
	
	// ========================================================================
	// 键盘输入解析：将Twist消息转换为运动增量
	// ========================================================================
	
	// 水平移动控制 (机体坐标系)
	if (msg->linear.x > 0) dx = step;        // W键：前进
	else if (msg->linear.x < 0) dx = -step;  // S键：后退
	
	if (msg->linear.y > 0) dy = step;        // A键：左移
	else if (msg->linear.y < 0) dy = -step;  // D键：右移
	
	// 垂直移动控制 (世界坐标系Z轴)
	if (msg->linear.z > 0) dz = step * 0.5;        // R键：上升 (0.15m)
	else if (msg->linear.z < 0) dz = -step * 0.5;  // F键：下降 (0.15m)
	
	// 偏航控制 (世界坐标系Z轴旋转)
	if (msg->angular.z > 0) dyaw = yaw_step;       // Q键：左转
	else if (msg->angular.z < 0) dyaw = -yaw_step; // E键：右转
	
	// 特殊指令：降落 (通过特殊数值999.0标识)
	if (fabs(msg->linear.x - 999.0) < 0.1) {
		takeoff_land_data.triggered = true;
		takeoff_land_data.takeoff_land_cmd = quadrotor_msgs::TakeoffLand::LAND;
		ROS_INFO("[px4ctrl] Landing command received via keyboard");
		return;
	}
	
	// 无效输入过滤：如果所有增量都很小，则忽略本次输入
	if (fabs(dx) < 0.01 && fabs(dy) < 0.01 && fabs(dz) < 0.01 && fabs(dyaw) < 0.01) {
		return;
	}
	
	// ========================================================================
	// 坐标系转换：机体坐标系 -> 世界坐标系
	// ========================================================================
	// 水平移动需要考虑当前偏航角，垂直移动和偏航旋转不需要转换
	double wx = dx * cos(yaw) - dy * sin(yaw);  // 世界坐标系X方向增量
	double wy = dx * sin(yaw) + dy * cos(yaw);  // 世界坐标系Y方向增量
	
	// ========================================================================
	// 更新悬停目标位置
	// ========================================================================
	hover_pose(0) += wx;    // X位置增量
	hover_pose(1) += wy;    // Y位置增量
	hover_pose(2) += dz;    // Z位置增量 (高度)
	hover_pose(3) += dyaw;  // 偏航角增量
	
	// ========================================================================
	// 安全限制和数值处理
	// ========================================================================
	
	// 高度限制保护
	if (hover_pose(2) < -0.3) {
		hover_pose(2) = -0.3;
		ROS_WARN("[px4ctrl] Height limited to minimum -0.3m");
	}
	if (hover_pose(2) > 2.0) {
		hover_pose(2) = 2.0;
		ROS_WARN("[px4ctrl] Height limited to maximum 2.0m");
	}
	
	// 偏航角归一化到 [-π, π] 范围
	while (hover_pose(3) > M_PI) hover_pose(3) -= 2 * M_PI;
	while (hover_pose(3) < -M_PI) hover_pose(3) += 2 * M_PI;
	
	// 实时反馈：每0.5秒打印一次目标位置 (防止日志刷屏)
	ROS_INFO_THROTTLE(0.5, "[px4ctrl] Keyboard control: target(%.2f, %.2f, %.2f, %.2f°)", 
			 hover_pose(0), hover_pose(1), hover_pose(2), hover_pose(3) * 180.0 / M_PI);
}

void PX4CtrlFSM::processMoveCmd()
{
	// 键盘控制模式下，这个函数主要用于状态监控和保护
	if (move_locked) {
		// 检查是否到达目标位置
		double err = sqrt(pow(odom_data.p(0) - hover_pose(0), 2) +
						  pow(odom_data.p(1) - hover_pose(1), 2));
		if (err < 0.05) { // 精度放宽到±5cm，键盘控制更适合
			move_locked = false;
			ROS_INFO("[px4ctrl] Move completed, hovering at target position (error: %.3fm)", err);
		}
		
		// 稳高保护：如果高度偏差过大，保持当前高度
		if (fabs(odom_data.p(2) - hover_pose(2)) > 0.1) {
			hover_pose(2) = odom_data.p(2); // 保持当前高度
		}
		
		// 超时保护：如果移动时间过长，强制解锁
		if ((ros::Time::now() - move_start_time).toSec() > 5.0) { // 5秒超时
			move_locked = false;
			ROS_ERROR("[px4ctrl] Move timeout - force unlock (error: %.3fm)", err);
		}
	}
}