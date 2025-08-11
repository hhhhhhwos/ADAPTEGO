#!/bin/bash

# 项目重命名脚本
# 将 Fast-Drone-250 重命名为 AdaptEgo

echo "========================================="
echo "项目重命名: Fast-Drone-250 → AdaptEgo"
echo "========================================="

WORKSPACE_DIR="/home/kklab/fast_drone_ws"
OLD_NAME="Fast-Drone-250"
NEW_NAME="AdaptEgo"

# 检查当前目录
if [ ! -d "$WORKSPACE_DIR/$OLD_NAME" ]; then
    echo "❌ 错误: 找不到项目目录 $WORKSPACE_DIR/$OLD_NAME"
    exit 1
fi

# 检查新目录是否已存在
if [ -d "$WORKSPACE_DIR/$NEW_NAME" ]; then
    echo "❌ 错误: 目标目录 $WORKSPACE_DIR/$NEW_NAME 已存在"
    read -p "是否删除现有目录并继续? (y/N): " confirm
    if [ "$confirm" != "y" ]; then
        echo "取消重命名操作"
        exit 1
    fi
    rm -rf "$WORKSPACE_DIR/$NEW_NAME"
fi

echo "🔄 正在重命名项目文件夹..."
cd "$WORKSPACE_DIR"
mv "$OLD_NAME" "$NEW_NAME"

if [ $? -eq 0 ]; then
    echo "✅ 项目文件夹重命名成功!"
    echo ""
    echo "📋 后续操作建议:"
    echo "1. 更新您的工作空间路径:"
    echo "   cd $WORKSPACE_DIR/$NEW_NAME"
    echo ""
    echo "2. 重新编译ROS工作空间:"
    echo "   cd $WORKSPACE_DIR && catkin build"
    echo ""
    echo "3. 更新环境变量:"
    echo "   source devel/setup.bash"
    echo ""
    echo "🎉 AdaptEgo 项目重命名完成!"
else
    echo "❌ 重命名失败"
    exit 1
fi
