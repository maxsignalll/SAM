#!/usr/bin/env python3
"""
分析B7策略的EMA衰减过程，理解为什么Need Score会大幅下降
"""

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import numpy as np

def simulate_ema_decay():
    """模拟burst scan实验中的EMA衰减过程"""
    
    # 配置参数
    alpha = 0.1  # EMA alpha值
    reporting_interval = 4  # 每4秒一个周期
    
    # Phase时间配置（秒）
    phases = {
        'Phase1_Baseline': (0, 30),
        'Phase2A_Normal': (30, 90),
        'Phase2B_Burst_Scan': (90, 105),
        'Phase2C_Recovery': (105, 165),
        'Phase2D_Second_Burst': (165, 195),
        'Phase2E_Final_Normal': (195, 240),
        'Phase3_Competition': (240, 300)
    }
    
    # 各Phase的TPS和miss_rate配置
    phase_configs = {
        'Phase1_Baseline': {'tps': 30, 'miss_rate': 0.01},  # zipf_alpha=0.99
        'Phase2A_Normal': {'tps': 70, 'miss_rate': 0.30},   # zipf_alpha=0.72
        'Phase2B_Burst_Scan': {'tps': 400, 'miss_rate': 0.55},  # zipf_alpha=0.45
        'Phase2C_Recovery': {'tps': 70, 'miss_rate': 0.30},     # zipf_alpha=0.72
        'Phase2D_Second_Burst': {'tps': 75, 'miss_rate': 0.30}, # zipf_alpha=0.72
        'Phase2E_Final_Normal': {'tps': 80, 'miss_rate': 0.05}, # zipf_alpha=0.95
        'Phase3_Competition': {'tps': 90, 'miss_rate': 0.01}    # zipf_alpha=0.99
    }
    
    # 初始化
    time_points = []
    actual_tps = []
    actual_miss_rate = []
    actual_need_score = []
    ema_tps = []
    ema_miss_rate = []
    ema_need_score = []
    phase_labels = []
    
    # 初始EMA值
    current_ema_tps = 0
    current_ema_miss_rate = 0
    
    # 模拟整个实验过程
    for t in range(0, 300, reporting_interval):
        # 确定当前所在Phase
        current_phase = None
        for phase_name, (start, end) in phases.items():
            if start <= t < end:
                current_phase = phase_name
                break
        
        if current_phase:
            config = phase_configs[current_phase]
            current_tps = config['tps']
            current_miss_rate = config['miss_rate']
            
            # 更新EMA
            if t == 0:
                # 初始化
                current_ema_tps = current_tps
                current_ema_miss_rate = current_miss_rate
            else:
                # EMA更新公式: new_ema = (1-alpha) * old_ema + alpha * current
                current_ema_tps = (1 - alpha) * current_ema_tps + alpha * current_tps
                current_ema_miss_rate = (1 - alpha) * current_ema_miss_rate + alpha * current_miss_rate
            
            # 计算need score
            current_need = current_tps * current_miss_rate
            ema_need = current_ema_tps * current_ema_miss_rate
            
            # 记录数据
            time_points.append(t)
            actual_tps.append(current_tps)
            actual_miss_rate.append(current_miss_rate)
            actual_need_score.append(current_need)
            ema_tps.append(current_ema_tps)
            ema_miss_rate.append(current_ema_miss_rate)
            ema_need_score.append(ema_need)
            phase_labels.append(current_phase)
    
    # 打印关键时刻的数据
    print("=== Burst Scan EMA衰减分析 ===\n")
    
    # Phase2A末期（基线）
    idx_2a_end = time_points.index(88)  # 接近90秒
    print(f"Phase2A末期 (t=88s):")
    print(f"  实际: TPS={actual_tps[idx_2a_end]}, MissRate={actual_miss_rate[idx_2a_end]:.3f}, Need={actual_need_score[idx_2a_end]:.1f}")
    print(f"  EMA:  TPS={ema_tps[idx_2a_end]:.1f}, MissRate={ema_miss_rate[idx_2a_end]:.3f}, Need={ema_need_score[idx_2a_end]:.1f}")
    
    # Phase2B末期（峰值）
    idx_2b_end = time_points.index(104)  # 接近105秒
    print(f"\nPhase2B末期 (t=104s):")
    print(f"  实际: TPS={actual_tps[idx_2b_end]}, MissRate={actual_miss_rate[idx_2b_end]:.3f}, Need={actual_need_score[idx_2b_end]:.1f}")
    print(f"  EMA:  TPS={ema_tps[idx_2b_end]:.1f}, MissRate={ema_miss_rate[idx_2b_end]:.3f}, Need={ema_need_score[idx_2b_end]:.1f}")
    
    # Phase2D开始（触发时刻）
    idx_2d_start = time_points.index(168)  # 接近165秒
    print(f"\nPhase2D开始 (t=168s):")
    print(f"  实际: TPS={actual_tps[idx_2d_start]}, MissRate={actual_miss_rate[idx_2d_start]:.3f}, Need={actual_need_score[idx_2d_start]:.1f}")
    print(f"  EMA:  TPS={ema_tps[idx_2d_start]:.1f}, MissRate={ema_miss_rate[idx_2d_start]:.3f}, Need={ema_need_score[idx_2d_start]:.1f}")
    
    # Phase2D中期
    idx_2d_mid = time_points.index(180)  # 180秒
    print(f"\nPhase2D中期 (t=180s):")
    print(f"  实际: TPS={actual_tps[idx_2d_mid]}, MissRate={actual_miss_rate[idx_2d_mid]:.3f}, Need={actual_need_score[idx_2d_mid]:.1f}")
    print(f"  EMA:  TPS={ema_tps[idx_2d_mid]:.1f}, MissRate={ema_miss_rate[idx_2d_mid]:.3f}, Need={ema_need_score[idx_2d_mid]:.1f}")
    
    # 计算衰减
    cycles_2c = 15  # Phase2C有60秒 = 15个周期
    decay_factor = (1 - alpha) ** cycles_2c
    print(f"\n=== 衰减分析 ===")
    print(f"Phase2C持续60秒 = {cycles_2c}个周期")
    print(f"每周期衰减系数: {1-alpha:.1f}")
    print(f"{cycles_2c}个周期后的残留影响: {decay_factor:.3f} ({decay_factor*100:.1f}%)")
    
    # 理论计算
    peak_need = 400 * 0.55  # 220
    recovery_need = 70 * 0.30  # 21
    print(f"\n理论EMA Need Score在Phase2D开始时:")
    print(f"  峰值贡献: {peak_need} * {decay_factor:.3f} = {peak_need * decay_factor:.1f}")
    print(f"  恢复期贡献: {recovery_need} * (1 - {decay_factor:.3f}) = {recovery_need * (1 - decay_factor):.1f}")
    print(f"  总计: {peak_need * decay_factor + recovery_need * (1 - decay_factor):.1f}")
    
    # 绘图
    plt.figure(figsize=(14, 10))
    
    # 子图1: TPS对比
    plt.subplot(3, 1, 1)
    plt.plot(time_points, actual_tps, 'b-', label='Actual TPS', linewidth=2)
    plt.plot(time_points, ema_tps, 'r--', label='EMA TPS', linewidth=2)
    plt.ylabel('TPS')
    plt.title('B7策略 Burst Scan实验 - EMA衰减分析')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 标记Phase边界
    for phase, (start, end) in phases.items():
        if 'Phase2' in phase:
            plt.axvline(x=start, color='gray', linestyle=':', alpha=0.5)
            plt.text(start + 2, 350, phase.split('_')[1], rotation=90, va='bottom', fontsize=8)
    
    # 子图2: Miss Rate对比
    plt.subplot(3, 1, 2)
    plt.plot(time_points, actual_miss_rate, 'b-', label='Actual Miss Rate', linewidth=2)
    plt.plot(time_points, ema_miss_rate, 'r--', label='EMA Miss Rate', linewidth=2)
    plt.ylabel('Miss Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图3: Need Score对比
    plt.subplot(3, 1, 3)
    plt.plot(time_points, actual_need_score, 'b-', label='Actual Need Score', linewidth=2)
    plt.plot(time_points, ema_need_score, 'r--', label='EMA Need Score', linewidth=2)
    plt.ylabel('Need Score')
    plt.xlabel('Time (seconds)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 标记关键点
    plt.axvline(x=90, color='green', linestyle='--', alpha=0.5, label='Phase2A End')
    plt.axvline(x=105, color='red', linestyle='--', alpha=0.5, label='Phase2B End')
    plt.axvline(x=165, color='orange', linestyle='--', alpha=0.5, label='Phase2D Start')
    
    plt.tight_layout()
    
    # 创建figures目录（如果不存在）
    import os
    figures_dir = 'figures'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    
    plt.savefig('figures/ema_decay_analysis.png', dpi=150)
    plt.close()
    
    print(f"\n图表已保存到: figures/ema_decay_analysis.png")
    
    # 分析问题
    print("\n=== 问题分析 ===")
    print("1. EMA alpha=0.1导致衰减速度较快")
    print("2. Phase2C（恢复期）持续60秒，足够让EMA大幅衰减")
    print(f"3. 到Phase2D时，burst的影响只剩{decay_factor*100:.1f}%")
    print("4. 这解释了为什么Need Score增加不明显")
    
    print("\n=== 可能的解决方案 ===")
    print("1. 增大EMA alpha值（如0.2或0.3）使其更敏感")
    print("2. 缩短Phase2C的持续时间")
    print("3. 在Phase2D使用更高的TPS或miss_rate")
    print("4. 考虑使用不同的EMA策略（如双速EMA）")

if __name__ == "__main__":
    simulate_ema_decay()