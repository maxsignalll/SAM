#!/usr/bin/env python3
"""
Generate publication-ready figures for the paper with accessibility improvements.
Based on accessibility_improvements.md guidelines.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Professional plot settings for publication
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (8, 6),
    'font.family': 'sans-serif',
    'pdf.fonttype': 42,  # TrueType fonts for better compatibility
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.2,
    'lines.markersize': 8,
    'font.weight': 'normal',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
})

# Color-blind friendly palettes - more muted and elegant
COLOR_PALETTES = {
    'elegant': {  # Muted colorblind-friendly palette
        'S0_EMG_AS': '#CC6677',      # Muted rose
        'B2_NoElasticFixedByPriority': '#4477AA',  # Muted blue
        'B7_DYNAMIC_NEED': '#228833',  # Muted green
        'B12_MT_LRU_Inspired': '#AA3377',  # Muted purple
        'B1_StaticAverage': '#EE7733',  # Muted orange
        'B4_IndividualOptimizedCache': '#009988',  # Muted teal
        'B6_DataSizeProportionalStatic': '#BBBBBB',  # Light gray
    }
}

# Hatching patterns for black and white printing
HATCH_PATTERNS = {
    'S0_EMG_AS': None,        # Solid
    'B1_StaticAverage': '///',  # Forward diagonal
    'B2_NoElasticFixedByPriority': '\\\\\\',  # Backward diagonal
    'B4_IndividualOptimizedCache': 'xxx',  # Cross
    'B6_DataSizeProportionalStatic': '...',  # Dots
    'B7_DYNAMIC_NEED': '|||',  # Vertical
    'B12_MT_LRU_Inspired': '---',  # Horizontal
}

# Line styles for time series
LINE_STYLES = {
    'S0_EMG_AS': '-',         # Solid
    'B2_NoElasticFixedByPriority': '--',  # Dashed
    'B7_DYNAMIC_NEED': '-.',  # Dash-dot
    'B12_MT_LRU_Inspired': ':',  # Dotted
}

# Markers for time series
MARKERS = {
    'S0_EMG_AS': 'o',         # Circle
    'B2_NoElasticFixedByPriority': 's',  # Square
    'B7_DYNAMIC_NEED': '^',   # Triangle
    'B12_MT_LRU_Inspired': 'D',  # Diamond
}

# Strategy display names - simplified
STRATEGY_NAMES = {
    'S0_EMG_AS': 'S0',
    'B1_StaticAverage': 'B1',
    'B2_NoElasticFixedByPriority': 'B2',
    'B4_IndividualOptimizedCache': 'B4',
    'B6_DataSizeProportionalStatic': 'B6',
    'B7_DYNAMIC_NEED': 'B7',
    'B12_MT_LRU_Inspired': 'B12',
}

def load_experiment_data(experiment_paths: List[str]) -> Dict[str, pd.DataFrame]:
    """Load data from all experiments."""
    all_data = defaultdict(list)
    
    for exp_path in experiment_paths:
        logger.info(f"Loading experiment: {exp_path}")
        
        results_dir = os.path.join(exp_path, 'individual_results')
        if not os.path.exists(results_dir):
            logger.warning(f"Results directory not found: {results_dir}")
            continue
            
        for strategy_dir in os.listdir(results_dir):
            csv_path = os.path.join(results_dir, strategy_dir, 'data.csv')
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    df['experiment'] = os.path.basename(exp_path)
                    all_data[strategy_dir].append(df)
                    logger.info(f"  Loaded {strategy_dir}: {len(df)} rows")
                except Exception as e:
                    logger.error(f"  Error loading {strategy_dir}: {e}")
    
    # Combine data for each strategy
    strategy_data = {}
    for strategy, dfs in all_data.items():
        if dfs:
            strategy_data[strategy] = pd.concat(dfs, ignore_index=True)
            logger.info(f"Combined {strategy}: {len(strategy_data[strategy])} total rows from {len(dfs)} experiments")
    
    return strategy_data

def plot_figure1_performance_comparison(strategy_data: Dict[str, pd.DataFrame], output_dir: str, palette: str = 'accessible'):
    """
    Figure 1: Full-page width, three subplots side-by-side
    (a) Effective throughput comparison (cost-based)
    (b) Average hit rate comparison
    (c) Effective throughput by phase (Phase 2 and Phase 3)
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3.5))
    
    # Select color palette
    colors = COLOR_PALETTES[palette]
    
    # Cost model parameters
    COST_HIT = 1    # Cost of a cache hit
    COST_MISS = 50  # Cost of a cache miss (network request)
    
    # Calculate metrics for each strategy
    strategies = []
    effective_throughputs = []
    hit_rates = []
    
    for strategy, df in sorted(strategy_data.items()):
        # Calculate effective throughput per experiment, then overall average
        experiment_effective_throughputs = []
        experiment_hit_rates = []
        
        for exp in df['experiment'].unique():
            exp_df = df[df['experiment'] == exp]
            # Filter to only include Phase 2 and Phase 3
            exp_df = exp_df[exp_df['phase_name'].isin(['Phase2_HighPriorityHotspot', 'Phase3_HotspotShiftToMedium'])]
            
            if exp_df.empty:
                continue
                
            # Calculate effective throughput based on cost model
            total_hits = exp_df['cache_hits'].sum()
            total_misses = exp_df['cache_misses'].sum()
            total_ops = total_hits + total_misses
            
            # Total effective cost = (hits * 1) + (misses * 50)
            total_cost = (total_hits * COST_HIT) + (total_misses * COST_MISS)
            
            # Effective throughput = total operations / total cost
            if total_cost > 0:
                effective_throughput = total_ops / total_cost
                experiment_effective_throughputs.append(effective_throughput)
            
            # Hit rate per experiment
            total_hits = exp_df['cache_hits'].sum()
            total_accesses = (exp_df['cache_hits'] + exp_df['cache_misses']).sum()
            if total_accesses > 0:
                experiment_hit_rates.append(total_hits / total_accesses)
        
        # Overall average/median across experiments
        strategies.append(STRATEGY_NAMES.get(strategy, strategy))
        effective_throughputs.append(np.mean(experiment_effective_throughputs) if experiment_effective_throughputs else 0)  # Use mean for effective throughput
        hit_rates.append(np.median(experiment_hit_rates) if experiment_hit_rates else 0)  # Keep median for hit rate
    
    # Plot effective throughput (subplot a)
    x = np.arange(len(strategies))
    bars1 = ax1.bar(x, effective_throughputs, alpha=0.8)
    
    # Apply colors and patterns
    for i, (bar, strategy) in enumerate(zip(bars1, sorted(strategy_data.keys()))):
        bar.set_color(colors.get(strategy, '#333333'))
        bar.set_hatch(HATCH_PATTERNS.get(strategy, None))
        bar.set_edgecolor('black')
        bar.set_linewidth(0.8)
    
    ax1.set_xlabel('Strategy', fontweight='bold')
    ax1.set_ylabel('Effective Throughput', fontweight='bold')
    ax1.set_title('(a) Effective Throughput Comparison', fontweight='bold', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, rotation=45, ha='right')
    ax1.grid(True, axis='y', alpha=0.3)
    # Add some padding to the top for value labels
    ax1.set_ylim(0, max(effective_throughputs) * 1.15) if effective_throughputs else 1
    
    # Add value labels
    for bar, val in zip(bars1, effective_throughputs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot hit rate (subplot b)
    bars2 = ax2.bar(x, hit_rates, alpha=0.8)
    
    # Apply colors and patterns
    for i, (bar, strategy) in enumerate(zip(bars2, sorted(strategy_data.keys()))):
        bar.set_color(colors.get(strategy, '#333333'))
        bar.set_hatch(HATCH_PATTERNS.get(strategy, None))
        bar.set_edgecolor('black')
        bar.set_linewidth(0.8)
    
    ax2.set_xlabel('Strategy', fontweight='bold')
    ax2.set_ylabel('Median Hit Rate', fontweight='bold')
    ax2.set_title('(b) Hit Rate Comparison', fontweight='bold', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, rotation=45, ha='right')
    ax2.set_ylim(0, 1)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars2, hit_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot effective throughput by phase (subplot c)
    # Calculate phase-specific effective throughputs
    phase2_effective_throughputs = []
    phase3_effective_throughputs = []
    
    for strategy, df in sorted(strategy_data.items()):
        # Phase 2 and 3 effective throughput
        phase2_eff_tps = []
        phase3_eff_tps = []
        
        for exp in df['experiment'].unique():
            exp_df = df[df['experiment'] == exp]
            
            # Phase 2
            phase2_df = exp_df[exp_df['phase_name'] == 'Phase2_HighPriorityHotspot']
            if not phase2_df.empty:
                total_hits = phase2_df['cache_hits'].sum()
                total_misses = phase2_df['cache_misses'].sum()
                total_ops = total_hits + total_misses
                total_cost = (total_hits * COST_HIT) + (total_misses * COST_MISS)
                if total_cost > 0:
                    phase2_eff_tps.append(total_ops / total_cost)
            
            # Phase 3
            phase3_df = exp_df[exp_df['phase_name'] == 'Phase3_HotspotShiftToMedium']
            if not phase3_df.empty:
                total_hits = phase3_df['cache_hits'].sum()
                total_misses = phase3_df['cache_misses'].sum()
                total_ops = total_hits + total_misses
                total_cost = (total_hits * COST_HIT) + (total_misses * COST_MISS)
                if total_cost > 0:
                    phase3_eff_tps.append(total_ops / total_cost)
        
        phase2_effective_throughputs.append(np.mean(phase2_eff_tps) if phase2_eff_tps else 0)
        phase3_effective_throughputs.append(np.mean(phase3_eff_tps) if phase3_eff_tps else 0)
    
    # Plot grouped bars
    x = np.arange(len(strategies))
    width = 0.4
    
    # Phase 2 bars with strategy colors
    bars3_1 = ax3.bar(x - width/2, phase2_effective_throughputs, width, alpha=0.9)
    # Phase 3 bars with strategy colors but lighter/with pattern
    bars3_2 = ax3.bar(x + width/2, phase3_effective_throughputs, width, alpha=0.6)
    
    # Apply colors and patterns for Phase 2 bars (solid)
    for i, (bar, strategy) in enumerate(zip(bars3_1, sorted(strategy_data.keys()))):
        bar.set_color(colors.get(strategy, '#333333'))
        bar.set_edgecolor('black')
        bar.set_linewidth(0.8)
    
    # Apply colors and patterns for Phase 3 bars (with hatching)
    for i, (bar, strategy) in enumerate(zip(bars3_2, sorted(strategy_data.keys()))):
        bar.set_color(colors.get(strategy, '#333333'))
        bar.set_hatch('///')  # Diagonal hatching for Phase 3
        bar.set_edgecolor('black')
        bar.set_linewidth(0.8)
    
    ax3.set_xlabel('Strategy', fontweight='bold')
    ax3.set_ylabel('Effective Throughput', fontweight='bold')
    ax3.set_title('(c) Effective Throughput by Phase', fontweight='bold', fontsize=11)
    ax3.set_xticks(x)
    ax3.set_xticklabels(strategies, rotation=45, ha='right')
    ax3.grid(True, axis='y', alpha=0.3)
    # Add some padding to the top for value labels
    max_val = max(max(phase2_effective_throughputs), max(phase3_effective_throughputs)) if phase2_effective_throughputs else 1
    ax3.set_ylim(0, max_val * 1.15)
    
    # Add value labels with rotation to avoid overlap
    for bar, val in zip(bars3_1, phase2_effective_throughputs):
        if val > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=6, rotation=45)
    
    for bar, val in zip(bars3_2, phase3_effective_throughputs):
        if val > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=6, rotation=45)
    
    # Add legend inside the plot area, at the bottom right corner
    ax3.text(0.98, 0.02, 'Solid bars: Phase 2\nHatched bars: Phase 3', 
             transform=ax3.transAxes, ha='right', va='bottom', fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                      edgecolor='gray', alpha=0.9))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'figure1_performance_comparison.pdf')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"Saved Figure 1 to {output_path}")

def plot_figure2_timeseries(strategy_data: Dict[str, pd.DataFrame], output_dir: str, palette: str = 'accessible'):
    """
    Figure 2: Full-page width, two subplots stacked vertically
    Cache allocation timeseries for S0, B7, B12, B2
    (a) High priority database
    (b) Medium priority database
    """
    # Filter strategies
    selected_strategies = ['S0_EMG_AS', 'B7_DYNAMIC_NEED', 'B12_MT_LRU_Inspired', 'B2_NoElasticFixedByPriority']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5, 4.5), gridspec_kw={'hspace': 0.35})
    
    # Adjust the left margin to give more space to the plot
    plt.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.08)
    
    # Select color palette
    colors = COLOR_PALETTES[palette]
    
    # Phase offsets for continuous timeline
    phase_offsets = {
        'Phase1_Baseline': 0,
        'Phase2_HighPriorityHotspot': 60,
        'Phase3_HotspotShiftToMedium': 360
    }
    
    # Plot for both databases
    for db_id, ax, title in [('db_high_priority', ax1, '(a) High Priority Database'),
                              ('db_medium_priority', ax2, '(b) Medium Priority Database')]:
        
        for strategy in selected_strategies:
            if strategy not in strategy_data:
                continue
                
            df = strategy_data[strategy]
            # Focus on specific database
            df_db = df[df['db_id'] == db_id].copy()
            
            if df_db.empty:
                continue
            
            # Add continuous time
            df_db['continuous_time'] = df_db.apply(
                lambda row: row['elapsed_seconds'] + phase_offsets.get(row['phase_name'], 0),
                axis=1
            )
            
            # Align to 4-second buckets
            df_db['time_bucket'] = (df_db['continuous_time'] / 4.0).round() * 4.0
            
            # Calculate median across experiments
            grouped = df_db.groupby(['experiment', 'time_bucket'])['current_cache_pages'].median().reset_index()
            grouped = grouped.groupby('time_bucket')['current_cache_pages'].median().reset_index()
            
            # Plot with line style and markers
            ax.plot(grouped['time_bucket'], grouped['current_cache_pages'],
                    label=STRATEGY_NAMES.get(strategy, strategy),
                    color=colors.get(strategy, '#333333'),
                    linestyle=LINE_STYLES.get(strategy, '-'),
                    marker=MARKERS.get(strategy, 'o'),
                    markevery=8,  # Show marker every 8 points for cleaner look
                    linewidth=1.8,
                    markersize=3,
                    alpha=0.9)
        
        # Add phase boundaries
        ax.axvline(x=60, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=360, color='gray', linestyle='--', alpha=0.5)
        
        # Get Y-axis limits and place phase labels slightly lower
        y_min, y_max = ax.get_ylim()
        label_y = y_max - (y_max - y_min) * 0.08  # 8% from top
        
        ax.text(25, label_y, 'Phase 1', ha='center', fontsize=8, weight='bold')
        ax.text(210, label_y, 'Phase 2', ha='center', fontsize=8, weight='bold')
        ax.text(510, label_y, 'Phase 3', ha='center', fontsize=8, weight='bold')
        
        ax.set_xlabel('Time (s)' if ax == ax2 else '', fontweight='bold', fontsize=9)
        ax.set_ylabel('Pages', fontweight='bold', fontsize=9)
        ax.set_title(title, fontweight='bold', fontsize=10, pad=4)
        ax.grid(True, alpha=0.3)
        
        # Optimize tick formatting
        ax.tick_params(axis='y', labelsize=8, pad=1)
        ax.tick_params(axis='x', labelsize=8, pad=2)
        
        # Rotate y-axis tick labels for compactness
        for tick in ax.get_yticklabels():
            tick.set_rotation(45)
            tick.set_horizontalalignment('right')
        
        # Format y-axis ticks to be more compact (show as integers)
        from matplotlib.ticker import FormatStrFormatter
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        
        # Only show legend on first subplot
        if ax == ax1:
            ax.legend(loc='upper right', framealpha=0.9, fontsize=7, ncol=2)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'figure2_timeseries.pdf')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    logger.info(f"Saved Figure 2 to {output_path}")

def main():
    """Main execution function."""
    # Find the most recent experiment results
    results_base = Path('./results/comparison')
    
    if not results_base.exists():
        logger.error("No results directory found!")
        return
    
    # Find all experiment directories and use the most recent one
    experiment_dirs = [d for d in results_base.iterdir() if d.is_dir() and d.name.startswith('experiment_')]
    if not experiment_dirs:
        logger.error("No experiment directories found!")
        return
    
    # Use the most recent experiment
    latest_experiment = max(experiment_dirs, key=lambda x: x.name)
    experiment_paths = [str(latest_experiment)]
    
    logger.info(f"Using latest experiment: {latest_experiment.name}")
    
    # Output directory
    output_dir = './figures/paper'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load experiment data
    logger.info("Loading experiment data...")
    strategy_data = load_experiment_data(experiment_paths)
    
    if not strategy_data:
        logger.error("No data loaded!")
        return
    
    logger.info(f"Loaded data for {len(strategy_data)} strategies")
    
    # Generate figures with elegant color palette
    palette = 'elegant'
    logger.info(f"Generating figures with {palette} palette...")
    plot_figure1_performance_comparison(strategy_data, output_dir, palette)
    plot_figure2_timeseries(strategy_data, output_dir, palette)
    
    logger.info("All figures generated successfully!")
    logger.info(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()