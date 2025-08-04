#!/usr/bin/env python3
"""
Generate publication-ready figures for VLDB paper from experimental results.
"""

import os
import sys
import glob
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# VLDB-compliant settings with improved readability
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (8, 6),
    'font.family': 'sans-serif',  # Use sans-serif instead
    'pdf.fonttype': 42,  # TrueType fonts for PDF
    'ps.fonttype': 42,   # TrueType fonts for PostScript
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.2,
})

# Color palette - professional and colorblind-friendly with better contrast
COLORS = {
    'S0_EMG_AS': '#2E86AB',  # Strong Blue
    'B1_StaticAverage': '#F77F00',  # Vivid Orange
    'B2_NoElasticFixedByPriority': '#06D6A0',  # Teal
    'B4_IndividualOptimizedCache': '#EF476F',  # Coral Red
    'B6_DataSizeProportionalStatic': '#7209B7',  # Deep Purple
    'B7_DYNAMIC_NEED': '#A0522D',  # Sienna Brown
    'B12_MT_LRU_Inspired': '#F72585',  # Magenta
}

# Strategy display names
STRATEGY_NAMES = {
    'S0_EMG_AS': 'S0-EMG-AS',
    'B1_StaticAverage': 'B1-Static-Avg',
    'B2_NoElasticFixedByPriority': 'B2-Fixed-Priority',
    'B4_IndividualOptimizedCache': 'B4-Individual-Opt',
    'B6_DataSizeProportionalStatic': 'B6-Data-Proportional',
    'B7_DYNAMIC_NEED': 'B7-Dynamic-Need',
    'B12_MT_LRU_Inspired': 'B12-MT-LRU',
}

def load_experiment_data(experiment_paths: List[str]) -> Dict[str, List[pd.DataFrame]]:
    """Load data from multiple experiments."""
    strategy_data = defaultdict(list)
    
    for exp_path in experiment_paths:
        logger.info(f"Loading experiment: {exp_path}")
        
        # Find all strategy result directories
        result_dir = os.path.join(exp_path, 'individual_results')
        if not os.path.exists(result_dir):
            logger.warning(f"Result directory not found: {result_dir}")
            continue
            
        for strategy_dir in os.listdir(result_dir):
            csv_path = os.path.join(result_dir, strategy_dir, 'data.csv')
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    strategy_data[strategy_dir].append(df)
                    logger.info(f"  Loaded {strategy_dir}: {len(df)} rows")
                except Exception as e:
                    logger.error(f"  Failed to load {csv_path}: {e}")
    
    return dict(strategy_data)

def compute_throughput_and_hit_rate_by_db(df: pd.DataFrame, db_id: str, phase_filter: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute throughput (ops/sec) and hit rate for a specific database."""
    # Filter for specific database
    df_db = df[df['db_id'] == db_id].copy()
    
    # Apply phase filter if specified
    if phase_filter:
        df_db = df_db[df_db['phase_name'] == phase_filter].copy()
    
    # Sort by timestamp to maintain chronological order
    df_db = df_db.sort_values(['timestamp', 'elapsed_seconds'])
    
    # If phase filter is applied, use elapsed_seconds directly
    if phase_filter:
        df_db['cumulative_seconds'] = df_db['elapsed_seconds']
    else:
        # Calculate phase start times based on actual max elapsed_seconds from previous phases
        phase_start_times = {}
        cumulative_time = 0
        
        for phase in ['Phase1_Baseline', 'Phase2_HighPriorityHotspot', 'Phase3_HotspotShiftToMedium']:
            phase_start_times[phase] = cumulative_time
            phase_data = df[df['phase_name'] == phase]
            if not phase_data.empty:
                # Get the maximum elapsed_seconds for this phase
                max_elapsed = phase_data['elapsed_seconds'].max()
                cumulative_time += max_elapsed
        
        # Add cumulative seconds based on actual phase progression
        df_db['cumulative_seconds'] = df_db.apply(
            lambda row: row['elapsed_seconds'] + phase_start_times.get(row['phase_name'], 0),
            axis=1
        )
    
    # Sort by cumulative time
    df_db = df_db.sort_values('cumulative_seconds')
    
    # Calculate instantaneous throughput using ops_count (operations per window)
    throughput_data = []
    hit_rate_data = []
    
    # Sort by cumulative_seconds to ensure proper ordering
    df_db = df_db.sort_values('cumulative_seconds').reset_index(drop=True)
    
    for i in range(len(df_db)):
        # ops_count is the number of operations in this time window
        # elapsed_seconds is the time since phase start
        if i > 0:
            time_diff = df_db.iloc[i]['cumulative_seconds'] - df_db.iloc[i-1]['cumulative_seconds']
            if time_diff > 0:
                # ops_count is operations in this window
                throughput = df_db.iloc[i]['ops_count'] / time_diff
            else:
                # If no time difference, use a default window size
                throughput = df_db.iloc[i]['ops_count'] / 4.0  # Assuming ~4 second windows
        else:
            # For first data point, estimate based on elapsed time
            if df_db.iloc[i]['elapsed_seconds'] > 0:
                throughput = df_db.iloc[i]['ops_count'] / df_db.iloc[i]['elapsed_seconds']
            else:
                throughput = 0
        
        throughput_data.append({
            'cumulative_seconds': df_db.iloc[i]['cumulative_seconds'],
            'throughput': throughput
        })
        
        # Calculate hit rate
        total_ops = df_db.iloc[i]['cache_hits'] + df_db.iloc[i]['cache_misses']
        if total_ops > 0:
            hit_rate = df_db.iloc[i]['cache_hits'] / total_ops
            hit_rate_data.append({
                'cumulative_seconds': df_db.iloc[i]['cumulative_seconds'],
                'hit_rate': hit_rate
            })
    
    throughput_df = pd.DataFrame(throughput_data)
    hit_rate_df = pd.DataFrame(hit_rate_data)
    
    return throughput_df, hit_rate_df

def plot_cache_allocation_timeseries_by_priority(strategy_data: Dict[str, List[pd.DataFrame]], output_dir: str):
    """Plot average cache allocation time series for high and medium priority databases separately."""
    # Create two subplots - one for high priority, one for medium priority
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    priority_dbs = {
        'high': 'db_high_priority',
        'medium': 'db_medium_priority'
    }
    
    axes = {'high': ax1, 'medium': ax2}
    titles = {'high': 'High Priority Database', 'medium': 'Medium Priority Database'}
    
    for priority, db_name in priority_dbs.items():
        ax = axes[priority]
        
        for strategy, dfs in strategy_data.items():
            if strategy not in COLORS:
                continue
                
            # Combine data from all experiments for specific database
            all_allocations = []
            
            for df in dfs:
                # Filter for specific database
                db_data = df[df['db_id'] == db_name].copy()
                
                if not db_data.empty:
                    # Sort by timestamp to maintain chronological order
                    db_data = db_data.sort_values('timestamp')
                    
                    # Calculate phase start times based on actual max elapsed_seconds from previous phases
                    phase_start_times = {}
                    cumulative_time = 0
                    
                    for phase in ['Phase1_Baseline', 'Phase2_HighPriorityHotspot', 'Phase3_HotspotShiftToMedium']:
                        phase_start_times[phase] = cumulative_time
                        phase_data = df[df['phase_name'] == phase]
                        if not phase_data.empty:
                            # Get the maximum elapsed_seconds for this phase
                            max_elapsed = phase_data['elapsed_seconds'].max()
                            cumulative_time += max_elapsed
                    
                    # Create cumulative elapsed_seconds
                    db_data['cumulative_seconds'] = db_data.apply(
                        lambda row: row['elapsed_seconds'] + phase_start_times.get(row['phase_name'], 0),
                        axis=1
                    )
                    
                    time_series = db_data[['cumulative_seconds', 'current_cache_pages']].sort_values('cumulative_seconds')
                    time_series.columns = ['elapsed_seconds', 'current_cache_pages']  # Rename for consistency
                    all_allocations.append(time_series)
            
            # Calculate mean and standard error across experiments
            if all_allocations:
                # Align time series by interpolating to common time points
                # Get actual max time from the data
                max_time = max(ts['elapsed_seconds'].max() for ts in all_allocations)
                time_points = np.linspace(0, max_time, 200)  # More points for smoother curves
                
                interpolated_data = []
                for time_series in all_allocations:
                    interp_values = np.interp(time_points, 
                                            time_series['elapsed_seconds'], 
                                            time_series['current_cache_pages'])
                    interpolated_data.append(interp_values)
                
                interpolated_data = np.array(interpolated_data)
                mean_allocation = np.mean(interpolated_data, axis=0)
                se_allocation = np.std(interpolated_data, axis=0) / np.sqrt(len(interpolated_data))
                
                # Plot with confidence interval (removed shading)
                ax.plot(time_points, mean_allocation, 
                       color=COLORS[strategy], 
                       label=STRATEGY_NAMES.get(strategy, strategy),
                       linewidth=2.5,
                       alpha=0.9)
                # Plot error bars as lines instead of shaded area
                ax.plot(time_points, mean_allocation - se_allocation,
                       color=COLORS[strategy], alpha=0.3, linewidth=0.5, linestyle='--')
                ax.plot(time_points, mean_allocation + se_allocation,
                       color=COLORS[strategy], alpha=0.3, linewidth=0.5, linestyle='--')
        
        ax.set_ylabel('Cache Pages Allocated', fontsize=11)
        ax.set_title(f'Cache Allocation - {titles[priority]}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', frameon=True, fancybox=False, shadow=False)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    ax2.set_xlabel('Time (seconds)', fontsize=11)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'cache_allocation_by_priority.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved cache allocation by priority to {output_path}")

def plot_throughput_timeseries(strategy_data: Dict[str, List[pd.DataFrame]], output_dir: str):
    """Plot throughput and hit rate time series for high priority (Phase 2) and medium priority (Phase 3) databases."""
    # Create figure with 4 subplots (2x2)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10), sharex=False)
    
    databases = {
        'db_high_priority': {
            'axes': (ax1, ax3), 
            'title': 'High Priority Database', 
            'phase': 'Phase2_HighPriorityHotspot',
            'phase_name': 'Phase 2'
        },
        'db_medium_priority': {
            'axes': (ax2, ax4), 
            'title': 'Medium Priority Database',
            'phase': 'Phase3_HotspotShiftToMedium',
            'phase_name': 'Phase 3'
        }
    }
    
    for db_id, db_info in databases.items():
        throughput_ax, hit_rate_ax = db_info['axes']
        
        for strategy, dfs in strategy_data.items():
            if strategy not in COLORS:
                continue
                
            # Calculate throughput and hit rate for each experiment
            all_throughputs = []
            all_hit_rates = []
            
            for df in dfs:
                throughput_df, hit_rate_df = compute_throughput_and_hit_rate_by_db(df, db_id, db_info['phase'])
                if not throughput_df.empty:
                    all_throughputs.append(throughput_df)
                if not hit_rate_df.empty:
                    all_hit_rates.append(hit_rate_df)
            
            # Plot throughput
            if all_throughputs:
                # Align time series by interpolating to common time points
                max_time = max(tp['cumulative_seconds'].max() for tp in all_throughputs)
                time_points = np.linspace(0, max_time, 300)  # More points for smoother curves
                
                interpolated_data = []
                for throughput_df in all_throughputs:
                    # Apply smoothing to individual experiments first
                    if len(throughput_df) > 10:
                        throughput_df['throughput_smooth'] = throughput_df['throughput'].rolling(
                            window=5, center=True, min_periods=1
                        ).mean()
                    else:
                        throughput_df['throughput_smooth'] = throughput_df['throughput']
                        
                    interp_values = np.interp(time_points,
                                            throughput_df['cumulative_seconds'],
                                            throughput_df['throughput_smooth'])
                    interpolated_data.append(interp_values)
                
                interpolated_data = np.array(interpolated_data)
                mean_throughput = np.mean(interpolated_data, axis=0)
                se_throughput = np.std(interpolated_data, axis=0) / np.sqrt(len(interpolated_data))
                
                # Plot with confidence interval
                throughput_ax.plot(time_points, mean_throughput,
                                 color=COLORS[strategy],
                                 label=STRATEGY_NAMES.get(strategy, strategy),
                                 linewidth=2.5,
                                 alpha=0.9)
                throughput_ax.fill_between(time_points,
                                         mean_throughput - se_throughput,
                                         mean_throughput + se_throughput,
                                         color=COLORS[strategy],
                                         alpha=0.15)
            
            # Plot hit rate
            if all_hit_rates:
                # Align time series by interpolating to common time points
                max_time = max(hr['cumulative_seconds'].max() for hr in all_hit_rates)
                time_points = np.linspace(0, max_time, 300)
                
                interpolated_data = []
                for hit_rate_df in all_hit_rates:
                    interp_values = np.interp(time_points,
                                            hit_rate_df['cumulative_seconds'],
                                            hit_rate_df['hit_rate'])
                    interpolated_data.append(interp_values)
                
                interpolated_data = np.array(interpolated_data)
                mean_hit_rate = np.mean(interpolated_data, axis=0)
                se_hit_rate = np.std(interpolated_data, axis=0) / np.sqrt(len(interpolated_data))
                
                # Plot with confidence interval
                hit_rate_ax.plot(time_points, mean_hit_rate,
                               color=COLORS[strategy],
                               label=STRATEGY_NAMES.get(strategy, strategy),
                               linewidth=2.5,
                               alpha=0.9)
                hit_rate_ax.fill_between(time_points,
                                       mean_hit_rate - se_hit_rate,
                                       mean_hit_rate + se_hit_rate,
                                       color=COLORS[strategy],
                                       alpha=0.15)
        
        # Configure throughput subplot
        throughput_ax.set_ylabel('Throughput (ops/sec)', fontsize=11)
        throughput_ax.set_title(f'{db_info["title"]} - Throughput ({db_info["phase_name"]})', fontsize=12, fontweight='bold')
        throughput_ax.legend(loc='best', frameon=True, fancybox=False, shadow=False, fontsize=8)
        throughput_ax.grid(True, alpha=0.3, linestyle='--')
        
        # Configure hit rate subplot
        hit_rate_ax.set_ylabel('Hit Rate', fontsize=11)
        hit_rate_ax.set_title(f'{db_info["title"]} - Hit Rate ({db_info["phase_name"]})', fontsize=12, fontweight='bold')
        hit_rate_ax.legend(loc='best', frameon=True, fancybox=False, shadow=False, fontsize=8)
        hit_rate_ax.grid(True, alpha=0.3, linestyle='--')
        hit_rate_ax.set_ylim(0, 1.0)
    
    # Set common x labels for bottom plots
    ax3.set_xlabel('Time (seconds)', fontsize=11)
    ax4.set_xlabel('Time (seconds)', fontsize=11)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'throughput_timeseries.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved throughput and hit rate timeseries by database to {output_path}")

def plot_total_throughput_comparison(strategy_data: Dict[str, List[pd.DataFrame]], output_dir: str):
    """Plot throughput and hit rate comparison for high priority (Phase 2) and medium priority (Phase 3) databases."""
    # Create figure with four subplots (2x2)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    databases = {
        'db_high_priority': {'title': 'High Priority Database', 'phase': 'Phase2_HighPriorityHotspot'},
        'db_medium_priority': {'title': 'Medium Priority Database', 'phase': 'Phase3_HotspotShiftToMedium'}
    }
    
    # Process data for each database separately
    for db_idx, (db_id, db_info) in enumerate(databases.items()):
        strategies = []
        mean_throughputs = []
        throughput_errors = []
        mean_hit_rates = []
        hit_rate_errors = []
        
        for strategy, dfs in strategy_data.items():
            if strategy not in COLORS:
                continue
                
            # Calculate throughput and hit rate for each experiment for this specific database
            experiment_throughputs = []
            experiment_hit_rates = []
            
            for df in dfs:
                # Filter to include only the specific phase AND specific database
                df_filtered = df[(df['phase_name'] == db_info['phase']) & 
                               (df['db_id'] == db_id)]
                
                # Calculate total operations and total time for filtered data
                total_ops = df_filtered['ops_count'].sum()
                total_time = df_filtered.groupby('phase_name')['elapsed_seconds'].max().sum()
                
                if total_time > 0:
                    avg_throughput = total_ops / total_time
                    experiment_throughputs.append(avg_throughput)
                
                # Calculate average hit rate
                total_hits = df_filtered['cache_hits'].sum()
                total_misses = df_filtered['cache_misses'].sum()
                total_accesses = total_hits + total_misses
                
                if total_accesses > 0:
                    avg_hit_rate = total_hits / total_accesses
                    experiment_hit_rates.append(avg_hit_rate)
            
            if experiment_throughputs and experiment_hit_rates:
                strategies.append(STRATEGY_NAMES.get(strategy, strategy))
                
                # Throughput statistics
                mean_throughputs.append(np.mean(experiment_throughputs))
                throughput_errors.append(np.std(experiment_throughputs) / np.sqrt(len(experiment_throughputs)))
                
                # Hit rate statistics
                mean_hit_rates.append(np.mean(experiment_hit_rates))
                hit_rate_errors.append(np.std(experiment_hit_rates) / np.sqrt(len(experiment_hit_rates)))
        
        # Select correct axes based on database index
        if db_idx == 0:  # High priority
            throughput_ax = ax1
            hit_rate_ax = ax2
        else:  # Medium priority
            throughput_ax = ax3
            hit_rate_ax = ax4
        
        # Create throughput bar plot
        x_pos = np.arange(len(strategies))
        # Get colors for each strategy
        bar_colors = []
        for strategy_name in strategies:
            # Find the original strategy key that maps to this display name
            for key, display_name in STRATEGY_NAMES.items():
                if display_name == strategy_name and key in COLORS:
                    bar_colors.append(COLORS[key])
                    break
        
        bars1 = throughput_ax.bar(x_pos, mean_throughputs, yerr=throughput_errors,
                                 capsize=6,
                                 color=bar_colors,
                                 edgecolor='black', linewidth=1.2,
                                 alpha=0.85)
        
        # Customize throughput plot
        throughput_ax.set_xlabel('Strategy', fontsize=11)
        throughput_ax.set_ylabel('Average Throughput (ops/sec)', fontsize=11)
        phase_name = 'Phase 2' if db_id == 'db_high_priority' else 'Phase 3'
        throughput_ax.set_title(f'{db_info["title"]} - Throughput ({phase_name})', fontsize=12, fontweight='bold')
        throughput_ax.set_xticks(x_pos)
        throughput_ax.set_xticklabels(strategies, rotation=45, ha='right')
        throughput_ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on throughput bars
        for i, (bar, mean_val, err) in enumerate(zip(bars1, mean_throughputs, throughput_errors)):
            height = bar.get_height()
            throughput_ax.text(bar.get_x() + bar.get_width()/2., height + err + max(mean_throughputs)*0.02,
                             f'{mean_val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Create hit rate bar plot
        bars2 = hit_rate_ax.bar(x_pos, mean_hit_rates, yerr=hit_rate_errors,
                               capsize=6,
                               color=bar_colors,  # Use the same colors as throughput plot
                               edgecolor='black', linewidth=1.2,
                               alpha=0.85)
        
        # Customize hit rate plot
        hit_rate_ax.set_xlabel('Strategy', fontsize=11)
        hit_rate_ax.set_ylabel('Average Hit Rate', fontsize=11)
        phase_name = 'Phase 2' if db_id == 'db_high_priority' else 'Phase 3'
        hit_rate_ax.set_title(f'{db_info["title"]} - Hit Rate ({phase_name})', fontsize=12, fontweight='bold')
        hit_rate_ax.set_xticks(x_pos)
        hit_rate_ax.set_xticklabels(strategies, rotation=45, ha='right')
        hit_rate_ax.grid(True, axis='y', alpha=0.3, linestyle='--')
        hit_rate_ax.set_ylim(0, 1.0)
        
        # Add value labels on hit rate bars
        for i, (bar, mean_val, err) in enumerate(zip(bars2, mean_hit_rates, hit_rate_errors)):
            height = bar.get_height()
            hit_rate_ax.text(bar.get_x() + bar.get_width()/2., height + err + 0.015,
                           f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'total_throughput_comparison.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved total throughput and hit rate comparison to {output_path}")

def main():
    """Main execution function."""
    # Experiment paths
    experiment_paths = [
        'results/comparison/experiment_20250716_020906',
        'results/comparison/experiment_20250717_204641',
        'results/comparison/experiment_20250717_221925',
        'results/comparison/experiment_20250717_234041',
        'results/comparison/experiment_20250718_021747'
    ]
    
    # Output directory
    output_dir = './figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all experiment data
    logger.info("Loading experiment data...")
    strategy_data = load_experiment_data(experiment_paths)
    
    if not strategy_data:
        logger.error("No data loaded!")
        return
    
    logger.info(f"Loaded data for {len(strategy_data)} strategies")
    
    # Generate plots
    logger.info("Generating plots...")
    plot_cache_allocation_timeseries_by_priority(strategy_data, output_dir)
    plot_throughput_timeseries(strategy_data, output_dir)
    plot_total_throughput_comparison(strategy_data, output_dir)
    
    # Import and run the stability/fairness analysis
    logger.info("Generating stability and fairness analysis...")
    try:
        from plot_stability_fairness_metrics import plot_stability_fairness_comparison
        plot_stability_fairness_comparison(strategy_data, output_dir)
    except ImportError:
        logger.warning("Could not import stability/fairness analysis module")
    
    logger.info("All plots generated successfully!")

if __name__ == "__main__":
    main()