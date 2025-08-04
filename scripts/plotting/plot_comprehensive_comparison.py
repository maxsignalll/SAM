#!/usr/bin/env python3
"""
Comprehensive comparison plotting script that handles multiple experiments
with different sets of strategies.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Professional plot settings
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (12, 8),
    'font.family': 'sans-serif',
    'pdf.fonttype': 42,
    'axes.linewidth': 1.2,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2.2,
})

# Complete color palette for all strategies
COLORS = {
    'S0_EMG_AS': '#2E86AB',  # Strong Blue
    'B1_StaticAverage': '#F77F00',  # Vivid Orange
    'B2_NoElasticFixedByPriority': '#06D6A0',  # Teal
    'B3_NoFixedElasticByPriority': '#F79E02',  # Amber
    'B4_IndividualOptimizedCache': '#EF476F',  # Coral Red
    'B5_GlobalLRU': '#118AB2',  # Ocean Blue
    'B6_DataSizeProportionalStatic': '#7209B7',  # Deep Purple
    'B7_DYNAMIC_NEED': '#A0522D',  # Sienna Brown
    'B8_EFFICIENCY_ONLY': '#2A9D8F',  # Sea Green
    'B9_EMG_AS_SINGLE_EMA': '#E76F51',  # Burnt Orange
    'B10_Pure_V_Factor': '#9B5DE5',  # Lavender
    'B11_ML_Driven': '#00BBF9',  # Sky Blue
    'B12_MT_LRU_Inspired': '#F72585',  # Magenta
    'B13_Active_Sampling': '#00F5FF',  # Cyan
}

# Strategy display names
STRATEGY_NAMES = {
    'S0_EMG_AS': 'S0 (EMG-AS)',
    'B1_StaticAverage': 'B1 (Static Avg)',
    'B2_NoElasticFixedByPriority': 'B2 (Fixed Priority)',
    'B3_NoFixedElasticByPriority': 'B3 (No Fixed)',
    'B4_IndividualOptimizedCache': 'B4 (Individual Opt)',
    'B5_GlobalLRU': 'B5 (Global LRU)',
    'B6_DataSizeProportionalStatic': 'B6 (Data Size)',
    'B7_DYNAMIC_NEED': 'B7 (Dynamic Need)',
    'B8_EFFICIENCY_ONLY': 'B8 (Efficiency Only)',
    'B9_EMG_AS_SINGLE_EMA': 'B9 (Single EMA)',
    'B10_Pure_V_Factor': 'B10 (Pure V)',
    'B11_ML_Driven': 'B11 (ML-Driven)',
    'B12_MT_LRU_Inspired': 'B12 (MT-LRU)',
    'B13_Active_Sampling': 'B13 (Active Sampling)',
}

def load_all_experiments(experiment_paths: List[str]) -> Dict[str, pd.DataFrame]:
    """Load data from all experiments, handling different strategy sets."""
    all_data = defaultdict(list)
    
    for exp_path in experiment_paths:
        logger.info(f"Loading experiment: {exp_path}")
        
        # List all strategy directories in this experiment
        results_dir = os.path.join(exp_path, 'individual_results')
        if not os.path.exists(results_dir):
            logger.warning(f"Results directory not found: {results_dir}")
            continue
            
        for strategy_dir in os.listdir(results_dir):
            csv_path = os.path.join(results_dir, strategy_dir, 'data.csv')
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    # Add experiment identifier
                    df['experiment'] = os.path.basename(exp_path)
                    all_data[strategy_dir].append(df)
                    logger.info(f"  Loaded {strategy_dir}: {len(df)} rows")
                    
                    # DEBUG: Print column names and data types for first file
                    if len(all_data[strategy_dir]) == 1:
                        logger.debug(f"  Columns in {strategy_dir}: {df.columns.tolist()}")
                        logger.debug(f"  Data types:\n{df.dtypes}")
                        logger.debug(f"  Sample data:\n{df.head(3)}")
                        
                except Exception as e:
                    logger.error(f"  Error loading {strategy_dir}: {e}")
    
    # Combine data for each strategy
    strategy_data = {}
    for strategy, dfs in all_data.items():
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            strategy_data[strategy] = combined_df
            logger.info(f"Combined {strategy}: {len(combined_df)} total rows from {len(dfs)} experiments")
            
            # DEBUG: Check for any data type issues after concatenation
            logger.debug(f"\nCombined data for {strategy}:")
            logger.debug(f"elapsed_seconds dtype: {combined_df['elapsed_seconds'].dtype}")
            logger.debug(f"Unique elapsed_seconds values: {combined_df['elapsed_seconds'].nunique()}")
            logger.debug(f"Min elapsed_seconds: {combined_df['elapsed_seconds'].min()}")
            logger.debug(f"Max elapsed_seconds: {combined_df['elapsed_seconds'].max()}")
            
            # Check if elapsed_seconds has any non-numeric values
            if combined_df['elapsed_seconds'].dtype == 'object':
                logger.warning(f"WARNING: elapsed_seconds is object type for {strategy}")
                non_numeric = combined_df[pd.to_numeric(combined_df['elapsed_seconds'], errors='coerce').isna()]
                if not non_numeric.empty:
                    logger.warning(f"Non-numeric elapsed_seconds values found:\n{non_numeric[['experiment', 'elapsed_seconds']].head()}")
    
    return strategy_data

def plot_cache_allocation_timeseries(strategy_data: Dict[str, pd.DataFrame], output_dir: str):
    """Plot cache allocation over time for all strategies."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    # Database configurations
    dbs = [
        ('db_high_priority', 'High Priority DB', axes[0]),
        ('db_medium_priority', 'Medium Priority DB', axes[1]),
        ('db_low_priority', 'Low Priority DB', axes[2]),
        ('db_bg_1', 'Background DB 1', axes[3])
    ]
    
    for db_id, db_name, ax in dbs:
        # Track which strategies have been plotted to avoid duplicate legend entries
        plotted_strategies = set()
        
        for strategy, df in sorted(strategy_data.items()):
            # Filter for this database
            df_db = df[df['db_id'] == db_id].copy()
            if df_db.empty:
                continue
            
            # Skip if already plotted (shouldn't happen with proper grouping)
            if strategy in plotted_strategies:
                logger.warning(f"Strategy {strategy} already plotted for {db_id}, skipping duplicate")
                continue
            plotted_strategies.add(strategy)
            
            # Create continuous timeline by adding phase offsets
            phase_offsets = {
                'Phase1_Baseline': 0,
                'Phase2_HighPriorityHotspot': 60,   # Phase1 lasts ~60s
                'Phase3_HotspotShiftToMedium': 360  # Phase2 lasts ~300s
            }
            
            # Add continuous time column
            df_db['continuous_time'] = df_db.apply(
                lambda row: row['elapsed_seconds'] + phase_offsets.get(row['phase_name'], 0),
                axis=1
            )
            
            # DEBUG: Check data before grouping
            logger.debug(f"\n{'='*80}")
            logger.debug(f"Cache Allocation - Strategy: {strategy}, DB: {db_id}")
            logger.debug(f"Data shape before grouping: {df_db.shape}")
            logger.debug(f"Unique experiments: {df_db['experiment'].nunique()}")
            logger.debug(f"Unique continuous_time values: {df_db['continuous_time'].nunique()}")
            
            # Check for duplicate time values
            time_counts = df_db.groupby('continuous_time').size()
            time_dups = time_counts[time_counts > 1]
            if not time_dups.empty:
                logger.debug(f"Duplicate continuous_time values before grouping:\n{time_dups.head(10)}")
            
            # FIXED: Align time values to 4-second buckets before grouping
            df_db['time_bucket'] = (df_db['continuous_time'] / 4.0).round() * 4.0
            
            # Group by time bucket and calculate mean cache allocation
            # First group by experiment and time to get per-experiment medians
            # Then group by time to get overall median
            grouped = df_db.groupby(['experiment', 'time_bucket'])['current_cache_pages'].median().reset_index()
            
            # DEBUG: After first grouping
            logger.debug(f"\nAfter first groupby (experiment, time_bucket):")
            logger.debug(f"Shape: {grouped.shape}")
            logger.debug(f"Sample:\n{grouped[['experiment', 'time_bucket', 'current_cache_pages']].head(10)}")
            
            grouped = grouped.groupby('time_bucket')['current_cache_pages'].median().reset_index()
            
            # Rename time_bucket back to continuous_time
            grouped = grouped.rename(columns={'time_bucket': 'continuous_time'})
            
            # DEBUG: After final grouping
            logger.debug(f"\nAfter second groupby (continuous_time only):")
            logger.debug(f"Final shape: {grouped.shape}")
            logger.debug(f"Sample:\n{grouped.head(10)}")
            
            # Check for duplicate time values in final data
            final_time_counts = grouped['continuous_time'].value_counts()
            final_dups = final_time_counts[final_time_counts > 1]
            if not final_dups.empty:
                logger.error(f"ERROR: Duplicate continuous_time in final data:\n{final_dups}")
                logger.debug(f"Full final grouped data:\n{grouped}")
            else:
                logger.debug(f"No duplicates in final data - shape: {grouped.shape}")
            
            # IMPORTANT: Sort by continuous_time to ensure proper plotting
            grouped = grouped.sort_values('continuous_time')
            logger.debug(f"After sorting - first few values: {grouped['continuous_time'].head(10).tolist()}")
            logger.debug(f"After sorting - last few values: {grouped['continuous_time'].tail(10).tolist()}")
            
            # Check data types
            logger.debug(f"Data type of continuous_time: {grouped['continuous_time'].dtype}")
            logger.debug(f"Data type of current_cache_pages: {grouped['current_cache_pages'].dtype}")
            
            # Plot with appropriate color and label
            color = COLORS.get(strategy, '#333333')
            label = STRATEGY_NAMES.get(strategy, strategy)
            
            # Convert to numpy arrays to ensure clean plotting
            x_values = grouped['continuous_time'].values
            y_values = grouped['current_cache_pages'].values
            
            logger.debug(f"Plotting {len(x_values)} points for {strategy}")
            logger.debug(f"X range: [{x_values.min()}, {x_values.max()}]")
            logger.debug(f"Y range: [{y_values.min()}, {y_values.max()}]")
            
            ax.plot(x_values, y_values, 
                   label=label, color=color, linewidth=2, alpha=0.8)
            
        # Add phase boundaries
        ax.axvline(x=60, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=360, color='gray', linestyle='--', alpha=0.5)
        ax.text(30, ax.get_ylim()[1]*0.95, 'Phase 1', ha='center', fontsize=9)
        ax.text(210, ax.get_ylim()[1]*0.95, 'Phase 2', ha='center', fontsize=9)
        ax.text(510, ax.get_ylim()[1]*0.95, 'Phase 3', ha='center', fontsize=9)
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Cache Pages')
        ax.set_title(f'{db_name} Cache Allocation')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'cache_allocation_timeseries_all.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved cache allocation timeseries to {output_path}")

def plot_throughput_timeseries(strategy_data: Dict[str, pd.DataFrame], output_dir: str):
    """Plot throughput over time for high and medium priority databases."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Configurations for subplots
    configs = [
        ('db_high_priority', 'Phase2_HighPriorityHotspot', 'High Priority DB - Phase 2', axes[0, 0], axes[0, 1]),
        ('db_medium_priority', 'Phase3_HotspotShiftToMedium', 'Medium Priority DB - Phase 3', axes[1, 0], axes[1, 1])
    ]
    
    for db_id, phase_filter, title_prefix, ax_tps, ax_hr in configs:
        # Track which strategies have been plotted
        plotted_strategies = set()
        
        for strategy, df in sorted(strategy_data.items()):
            # Filter for database and phase
            df_filtered = df[(df['db_id'] == db_id) & (df['phase_name'] == phase_filter)].copy()
            if df_filtered.empty:
                continue
            
            # Skip if already plotted (shouldn't happen with proper grouping)
            if strategy in plotted_strategies:
                logger.warning(f"Strategy {strategy} already plotted for {db_id}/{phase_filter}, skipping duplicate")
                continue
            plotted_strategies.add(strategy)
            
            # DEBUG: Print initial data shape and sample
            logger.debug(f"\n{'='*80}")
            logger.debug(f"Strategy: {strategy}, DB: {db_id}, Phase: {phase_filter}")
            logger.debug(f"Initial filtered data shape: {df_filtered.shape}")
            logger.debug(f"Unique experiments: {df_filtered['experiment'].nunique()}")
            logger.debug(f"Unique elapsed_seconds: {df_filtered['elapsed_seconds'].nunique()}")
            logger.debug(f"Sample of filtered data:\n{df_filtered[['experiment', 'elapsed_seconds', 'ops_count']].head(10)}")
            
            # Calculate TPS (ops_count is per 4-second window)
            df_filtered['tps'] = df_filtered['ops_count'] / 4.0
            
            # DEBUG: Check for duplicate time values before grouping
            time_value_counts = df_filtered.groupby('elapsed_seconds').size()
            duplicates = time_value_counts[time_value_counts > 1]
            if not duplicates.empty:
                logger.debug(f"Duplicate time values before grouping:\n{duplicates.head(10)}")
            
            # FIXED: Align time values to 4-second buckets before grouping
            df_filtered['time_bucket'] = (df_filtered['elapsed_seconds'] / 4.0).round() * 4.0
            
            # Group by time bucket - first by experiment then overall
            grouped = df_filtered.groupby(['experiment', 'time_bucket']).agg({
                'tps': 'median',
                'cache_hit_rate': 'median'
            }).reset_index()
            
            # DEBUG: Print shape after first grouping
            logger.debug(f"\nAfter first groupby (experiment, time_bucket):")
            logger.debug(f"Shape: {grouped.shape}")
            logger.debug(f"Sample:\n{grouped.head(10)}")
            
            # Check for duplicate experiment-time combinations
            dup_check = grouped.groupby(['experiment', 'time_bucket']).size()
            dups = dup_check[dup_check > 1]
            if not dups.empty:
                logger.debug(f"UNEXPECTED: Duplicate (experiment, time_bucket) after groupby:\n{dups}")
            
            grouped = grouped.groupby('time_bucket').agg({
                'tps': 'median',
                'cache_hit_rate': 'median'
            }).reset_index()
            
            # Rename time_bucket back to elapsed_seconds
            grouped = grouped.rename(columns={'time_bucket': 'elapsed_seconds'})
            
            # DEBUG: Print final shape and check for duplicates
            logger.debug(f"\nAfter second groupby (elapsed_seconds only):")
            logger.debug(f"Final shape: {grouped.shape}")
            logger.debug(f"Sample of final data:\n{grouped.head(10)}")
            
            # Check for duplicate time values in final data
            final_time_counts = grouped['elapsed_seconds'].value_counts()
            final_dups = final_time_counts[final_time_counts > 1]
            if not final_dups.empty:
                logger.error(f"ERROR: Duplicate elapsed_seconds in final data:\n{final_dups}")
                logger.debug(f"Full final data:\n{grouped}")
            else:
                logger.debug(f"No duplicates in final data - shape: {grouped.shape}")
            
            # IMPORTANT: Sort by elapsed_seconds to ensure proper plotting
            grouped = grouped.sort_values('elapsed_seconds')
            logger.debug(f"After sorting - first few values: {grouped['elapsed_seconds'].head(10).tolist()}")
            logger.debug(f"After sorting - last few values: {grouped['elapsed_seconds'].tail(10).tolist()}")
            
            # Check data types
            logger.debug(f"Data type of elapsed_seconds: {grouped['elapsed_seconds'].dtype}")
            logger.debug(f"Data type of tps: {grouped['tps'].dtype}")
            
            # Plot
            color = COLORS.get(strategy, '#333333')
            label = STRATEGY_NAMES.get(strategy, strategy)
            
            # Convert to numpy arrays to ensure clean plotting
            x_values = grouped['elapsed_seconds'].values
            y_tps = grouped['tps'].values
            y_hr = grouped['cache_hit_rate'].values
            
            logger.debug(f"Plotting {len(x_values)} points for {strategy}")
            logger.debug(f"X range: [{x_values.min()}, {x_values.max()}]")
            logger.debug(f"Y (TPS) range: [{y_tps.min()}, {y_tps.max()}]")
            
            ax_tps.plot(x_values, y_tps, 
                       label=label, color=color, linewidth=2, alpha=0.8)
            ax_hr.plot(x_values, y_hr, 
                      label=label, color=color, linewidth=2, alpha=0.8)
        
        # Format TPS subplot
        ax_tps.set_xlabel('Time (seconds)')
        ax_tps.set_ylabel('TPS (ops/sec)')
        ax_tps.set_title(f'{title_prefix} - Throughput')
        ax_tps.grid(True, alpha=0.3)
        ax_tps.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        
        # Format Hit Rate subplot
        ax_hr.set_xlabel('Time (seconds)')
        ax_hr.set_ylabel('Cache Hit Rate')
        ax_hr.set_title(f'{title_prefix} - Hit Rate')
        ax_hr.grid(True, alpha=0.3)
        ax_hr.set_ylim(0, 1)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'throughput_timeseries_all.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved throughput timeseries to {output_path}")

def plot_total_throughput_comparison(strategy_data: Dict[str, pd.DataFrame], output_dir: str):
    """Plot total throughput and hit rate comparison across all strategies."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Calculate metrics for each strategy
    strategies = []
    total_throughputs = []
    total_hit_rates = []
    
    for strategy, df in sorted(strategy_data.items()):
        # Calculate total throughput (sum of all databases)
        total_ops = df.groupby(['experiment', 'elapsed_seconds'])['ops_count'].sum()
        total_time = df.groupby('experiment')['elapsed_seconds'].max() - df.groupby('experiment')['elapsed_seconds'].min()
        
        # Median across experiments
        avg_throughput = (total_ops / 4.0).median()  # ops_count is per 4-second window
        
        # Calculate median hit rate across experiments
        experiment_hit_rates = []
        for exp in df['experiment'].unique():
            exp_df = df[df['experiment'] == exp]
            exp_hits = exp_df['cache_hits'].sum()
            exp_accesses = (exp_df['cache_hits'] + exp_df['cache_misses']).sum()
            if exp_accesses > 0:
                experiment_hit_rates.append(exp_hits / exp_accesses)
        avg_hit_rate = np.median(experiment_hit_rates) if experiment_hit_rates else 0
        
        strategies.append(STRATEGY_NAMES.get(strategy, strategy))
        total_throughputs.append(avg_throughput)
        total_hit_rates.append(avg_hit_rate)
    
    # Plot throughput
    x = np.arange(len(strategies))
    colors = [COLORS.get(s, '#333333') for s in sorted(strategy_data.keys())]
    
    bars1 = ax1.bar(x, total_throughputs, color=colors, alpha=0.8)
    ax1.set_xlabel('Strategy')
    ax1.set_ylabel('Median Total TPS')
    ax1.set_title('Total Throughput Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, rotation=45, ha='right')
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars1, total_throughputs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Plot hit rate
    bars2 = ax2.bar(x, total_hit_rates, color=colors, alpha=0.8)
    ax2.set_xlabel('Strategy')
    ax2.set_ylabel('Median Hit Rate')
    ax2.set_title('Cache Hit Rate Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, rotation=45, ha='right')
    ax2.set_ylim(0, 1)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars2, total_hit_rates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'total_performance_comparison_all.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved total performance comparison to {output_path}")

def plot_hit_rate_timeseries(strategy_data: Dict[str, pd.DataFrame], output_dir: str):
    """Plot cache hit rate over time for all phases."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Phase offsets for continuous timeline
    phase_offsets = {
        'Phase1_Baseline': 0,
        'Phase2_HighPriorityHotspot': 60,
        'Phase3_HotspotShiftToMedium': 360
    }
    
    # Configurations
    configs = [
        ('db_high_priority', 'High Priority DB', axes[0, 0]),
        ('db_medium_priority', 'Medium Priority DB', axes[0, 1]),
        ('db_low_priority', 'Low Priority DB', axes[1, 0]),
        ('All DBs', 'System-wide Median', axes[1, 1])
    ]
    
    for db_id, title, ax in configs:
        for strategy, df in sorted(strategy_data.items()):
            df_copy = df.copy()
            
            # Add continuous time
            df_copy['continuous_time'] = df_copy.apply(
                lambda row: row['elapsed_seconds'] + phase_offsets.get(row['phase_name'], 0),
                axis=1
            )
            
            if db_id == 'All DBs':
                # Calculate system-wide median
                grouped = df_copy.groupby(['experiment', 'continuous_time', 'phase_name']).agg({
                    'cache_hits': 'sum',
                    'cache_misses': 'sum'
                }).reset_index()
                grouped['hit_rate'] = grouped['cache_hits'] / (grouped['cache_hits'] + grouped['cache_misses'])
                # Median across experiments
                final_grouped = grouped.groupby('continuous_time')['hit_rate'].median().reset_index()
            else:
                # Filter for specific database
                df_db = df_copy[df_copy['db_id'] == db_id]
                if df_db.empty:
                    continue
                # First group by experiment and time, then median across experiments
                final_grouped = df_db.groupby(['experiment', 'continuous_time'])['cache_hit_rate'].median().reset_index()
                final_grouped = final_grouped.groupby('continuous_time')['cache_hit_rate'].median().reset_index()
                final_grouped.rename(columns={'cache_hit_rate': 'hit_rate'}, inplace=True)
            
            # Plot
            color = COLORS.get(strategy, '#333333')
            label = STRATEGY_NAMES.get(strategy, strategy)
            ax.plot(final_grouped['continuous_time'], final_grouped['hit_rate'],
                   label=label, color=color, linewidth=2, alpha=0.8)
        
        # Add phase boundaries
        ax.axvline(x=60, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=360, color='gray', linestyle='--', alpha=0.5)
        ax.text(30, 0.95, 'Phase 1', ha='center', fontsize=9, transform=ax.get_xaxis_transform())
        ax.text(210, 0.95, 'Phase 2', ha='center', fontsize=9, transform=ax.get_xaxis_transform())
        ax.text(510, 0.95, 'Phase 3', ha='center', fontsize=9, transform=ax.get_xaxis_transform())
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Cache Hit Rate')
        ax.set_title(f'{title} - Hit Rate Over Time')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'hit_rate_timeseries_all.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved hit rate timeseries to {output_path}")

def main():
    """Main execution function."""
    # Experiment paths
    experiment_paths = [
        'results/comparison/experiment_20250716_051030',
        'results/comparison/experiment_20250716_215121',
        'results/comparison/experiment_20250717_004756',
        'results/comparison/experiment_20250717_023713'
    ]
    
    # Output directory
    output_dir = './figures/comprehensive_additional'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all experiment data
    logger.info("Loading all experiments...")
    strategy_data = load_all_experiments(experiment_paths)
    
    if not strategy_data:
        logger.error("No data loaded!")
        return
    
    logger.info(f"Loaded data for {len(strategy_data)} strategies:")
    for strategy in sorted(strategy_data.keys()):
        logger.info(f"  - {strategy}")
    
    # Generate plots
    logger.info("Generating plots...")
    plot_cache_allocation_timeseries(strategy_data, output_dir)
    plot_throughput_timeseries(strategy_data, output_dir)
    plot_hit_rate_timeseries(strategy_data, output_dir)
    plot_total_throughput_comparison(strategy_data, output_dir)
    
    logger.info("All plots generated successfully!")
    logger.info(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()