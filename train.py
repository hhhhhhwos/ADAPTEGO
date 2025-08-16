#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AdaptEgo 统一训练脚本
整合行为克隆和强化学习训练
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# 设置项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT / "src" / "adaptive_planning" / "scripts"))

def main():
    parser = argparse.ArgumentParser(description='AdaptEgo 自适应权重训练')
    parser.add_argument('--mode', choices=['bc', 'rl', 'hybrid'], default='bc',
                       help='训练模式: bc(行为克隆), rl(强化学习), hybrid(混合)')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.002, help='学习率')
    parser.add_argument('--device', default='auto', help='设备选择: auto, cpu, cuda')
    parser.add_argument('--save_dir', default='models', help='模型保存目录')
    
    args = parser.parse_args()
    
    # 设备选择
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("=" * 50)
    print("🚁 AdaptEgo 自适应权重训练")
    print("=" * 50)
    print(f"训练模式: {args.mode}")
    print(f"设备: {device}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"训练轮数: {args.epochs}")
    print("=" * 50)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.mode == 'bc':
        print("🧠 开始行为克隆训练...")
        from adaptive_training import main as bc_train
        bc_train()
        
    elif args.mode == 'rl':
        print("🎮 开始强化学习训练...")
        from train_with_pybullet import main as rl_train
        rl_train()
        
    elif args.mode == 'hybrid':
        print("🔄 开始混合训练 (先行为克隆，后强化学习)...")
        print("第1阶段: 行为克隆预训练...")
        from adaptive_training import main as bc_train
        bc_train()
        
        print("第2阶段: 强化学习微调...")
        from train_with_pybullet import main as rl_train
        rl_train()
    
    print("🎉 训练完成！")
    print(f"模型保存在: {args.save_dir}/")

if __name__ == '__main__':
    main()
