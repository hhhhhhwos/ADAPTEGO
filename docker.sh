#!/bin/bash

echo "🐳 AdaptEgo 完全容器化环境"
echo "============================="
echo "项目和依赖都在容器内，完全隔离"
echo ""

# 检查Docker是否运行
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker未运行，请先启动Docker"
    exit 1
fi

case ${1:-menu} in
    "build")
        echo "🔨 构建完全容器化镜像..."
        echo "注意：这会将整个项目打包到镜像中"
        docker compose build
        ;;
    "start")
        echo "🚀 启动完全容器化环境..."
        docker compose up -d
        echo ""
        echo "✅ 容器已启动！"
        echo "进入容器: $0 shell"
        echo "开始训练: $0 train"
        echo "查看日志: $0 logs"
        ;;
    "train")
        echo "🧠 在容器内开始训练..."
        docker compose exec adaptego-full python3 train.py
        ;;
    "train-bg")
        echo "🧠 后台训练模式..."
        docker compose exec -d adaptego-full python3 train.py
        echo "查看训练日志: $0 logs"
        ;;
    "shell")
        echo "💻 进入容器Shell..."
        docker compose exec adaptego-full bash
        ;;
    "logs")
        echo "📋 查看容器日志..."
        docker compose logs -f adaptego-full
        ;;
    "tensorboard")
        echo "📊 启动TensorBoard..."
        docker compose exec -d adaptego-full tensorboard --logdir=logs --host=0.0.0.0 --port=6006
        echo "TensorBoard访问: http://localhost:6006"
        ;;
    "stop")
        echo "🛑 停止完全容器化环境..."
        docker compose down
        ;;
    "clean")
        echo "🧹 清理完全容器化环境..."
        docker compose down --rmi all --volumes
        ;;
    "install")
        echo "📦 在容器内安装额外依赖..."
        if [ -z "$2" ]; then
            echo "用法: $0 install <package_name>"
            echo "例如: $0 install opencv-python"
            exit 1
        fi
        docker compose exec adaptego-full pip3 install $2
        ;;
    "update-code")
        echo "🔄 更新容器内的代码..."
        echo "重新构建镜像以更新代码..."
        docker compose build --no-cache
        ;;
    *)
        echo "AdaptEgo 完全容器化管理工具"
        echo ""
        echo "🏗️  构建和启动:"
        echo "  $0 build        - 构建完全容器化镜像"
        echo "  $0 start        - 启动容器环境"
        echo ""
        echo "🚀 训练相关:"
        echo "  $0 train        - 交互式训练"
        echo "  $0 train-bg     - 后台训练"
        echo "  $0 tensorboard  - 启动TensorBoard"
        echo ""
        echo "💻 开发调试:"
        echo "  $0 shell        - 进入容器Shell"
        echo "  $0 logs         - 查看运行日志"
        echo "  $0 install <pkg> - 安装Python包"
        echo ""
        echo "🔧 维护:"
        echo "  $0 update-code  - 更新容器内代码"
        echo "  $0 stop         - 停止环境"
        echo "  $0 clean        - 完全清理"
        echo ""
        echo "📁 数据目录 (自动同步到主机):"
        echo "  - models/   : 训练模型"
        echo "  - data/     : 训练数据"
        echo "  - results/  : 结果文件"
        echo "  - logs/     : 训练日志"
        ;;
esac
