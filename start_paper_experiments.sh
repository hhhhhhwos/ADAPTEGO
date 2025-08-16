#!/bin/bash
# AdaptEgo 论文实验一键启动脚本

echo "=== AdaptEgo 论文实验系统 ==="
echo "请选择要运行的实验:"
echo "1) 数据收集 - 收集专家演示数据"
echo "2) 模型训练 - 训练自适应权重网络"  
echo "3) 快速评估 - 运行小规模性能测试"
echo "4) 完整实验 - 运行完整论文实验"
echo "5) 分析结果 - 生成论文图表和统计"
echo "0) 退出"

read -p "请输入选择 (0-5): " choice

case $choice in
    1)
        echo "🔬 启动数据收集系统..."
        echo "请在另一个终端运行仿真环境:"
        echo "  ./run_sim_adaptive_demo.sh"
        echo ""
        echo "然后在RViz中手动飞行收集数据"
        cd experiments
        python3 collect_expert_data.py
        ;;
    2)
        echo "🧠 开始模型训练..."
        cd src/adaptive_planning/scripts
        python3 adaptive_training.py
        ;;
    3)
        echo "⚡ 运行快速评估..."
        cd experiments
        python3 quick_evaluation.py
        ;;
    4)
        echo "🔬 运行完整论文实验..."
        echo "⚠️  预计耗时: 2-4小时"
        read -p "确认继续? (y/N): " confirm
        if [ "$confirm" = "y" ]; then
            cd experiments
            python3 full_paper_experiments.py
        fi
        ;;
    5)
        echo "📊 生成分析报告..."
        cd experiments
        if [ -f "paper_results/experiment_results.csv" ]; then
            python3 -c "
import sys, os
sys.path.append('../src/adaptive_planning/scripts')
from paper_evaluation import PaperExperimentSystem
analyzer = PaperExperimentSystem()
analyzer._generate_analysis_report()
"
        else
            echo "❌ 未找到实验结果文件"
            echo "请先运行实验 (选项3或4)"
        fi
        ;;
    0)
        echo "👋 再见!"
        exit 0
        ;;
    *)
        echo "❌ 无效选择"
        ;;
esac
