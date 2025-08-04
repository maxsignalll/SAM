#!/usr/bin/env python3
"""
Generate CPU performance comparison plot for S0 and B7 strategies.
Optimized for single-column display in double-column paper.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Professional plot settings for single-column display
plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.figsize': (3.5, 2.0),  # Single column width, further reduced height
    'font.family': 'sans-serif',
    'pdf.fonttype': 42,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'font.weight': 'normal',
    'axes.labelweight': 'normal',
})

# Load data from actual experiments or use defaults
import os
import json
import sys
from pathlib import Path

# Add src to path for scalability utilities
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def load_scalability_data():
    """Load scalability test results from latest experiment."""
    # Try to load latest results
    latest_path = 'results/scalability/type3_summary_latest.json'
    
    if os.path.exists(latest_path):
        print(f"Loading data from: {latest_path}")
        with open(latest_path, 'r') as f:
            data = json.load(f)
        
        db_counts = data['metadata']['database_counts']
        s0_times = []
        b7_times = []
        s0_std = []
        b7_std = []
        
        for db_count in db_counts:
            db_str = str(db_count)
            if db_str in data['results']:
                result = data['results'][db_str]
                
                # S0 data
                if 'S0_EMG_AS' in result:
                    s0_times.append(result['S0_EMG_AS']['mean_ms'])
                    s0_std.append(result['S0_EMG_AS']['std_ms'])
                else:
                    s0_times.append(0)
                    s0_std.append(0)
                
                # B7 data
                if 'B7_DYNAMIC_NEED' in result:
                    b7_times.append(result['B7_DYNAMIC_NEED']['mean_ms'])
                    b7_std.append(result['B7_DYNAMIC_NEED']['std_ms'])
                else:
                    b7_times.append(0)
                    b7_std.append(0)
        
        print(f"Loaded data for {len(db_counts)} database counts")
        return db_counts, s0_times, b7_times, s0_std, b7_std
    else:
        print("No experimental data found, using default values")
        # Default data from cpu_performance_analysis_updated.md
        db_counts = [20, 40, 80, 120]
        s0_times = [0.215, 0.232, 0.299, 0.294]
        b7_times = [0.109, 0.173, 0.288, 0.302]
        s0_std = [0.094, 0.065, 0.104, 0.172]
        b7_std = [0.026, 0.032, 0.040, 0.082]
        return db_counts, s0_times, b7_times, s0_std, b7_std

# Load the data
db_counts, s0_times, b7_times, s0_std, b7_std = load_scalability_data()

# Create figure
fig, ax = plt.subplots(1, 1)

# Define colors - simple and clean
color_s0 = '#E74C3C'  # Soft red
color_b7 = '#3498DB'  # Soft blue

# Plot lines with error bars and different line styles
ax.errorbar(db_counts, s0_times, yerr=s0_std, 
            label='S0', 
            color=color_s0,
            marker='o',
            markersize=6,
            capsize=3,
            capthick=1,
            linewidth=1.5,
            linestyle='-',  # Solid line
            alpha=0.9)

ax.errorbar(db_counts, b7_times, yerr=b7_std,
            label='B7',
            color=color_b7,
            marker='s',
            markersize=6,
            capsize=3,
            capthick=1,
            linewidth=1.5,
            linestyle='--',  # Dashed line
            alpha=0.9)

# Add subtle grid
ax.grid(True, linestyle=':', alpha=0.3, linewidth=0.5)

# Set labels
ax.set_xlabel('Number of Databases', fontsize=9)
ax.set_ylabel('Average CPU Time (ms)', fontsize=9)

# Set x-axis ticks
ax.set_xticks(db_counts)
ax.set_xticklabels(db_counts)

# Set y-axis range (auto-scale based on data, with some margin)
max_y = max(max(s0_times) + max(s0_std), max(b7_times) + max(b7_std)) * 1.2
# Ensure minimum range for visibility
max_y = max(max_y, 0.5)
ax.set_ylim(0, max_y)

# Add legend with frame
legend = ax.legend(loc='upper left', 
                   frameon=True,
                   fancybox=False,
                   edgecolor='gray',
                   facecolor='white',
                   framealpha=0.9,
                   borderpad=0.3,
                   columnspacing=0.5,
                   handlelength=1.5)

# Fine-tune layout
plt.tight_layout(pad=0.5)

# Save figures
output_dir = 'figures/cpu_performance/'
import os
os.makedirs(output_dir, exist_ok=True)

# Save as PDF (for paper) and PNG (for preview)
plt.savefig(f'{output_dir}cpu_performance_comparison.pdf', 
            dpi=300, 
            bbox_inches='tight',
            pad_inches=0.05)

plt.savefig(f'{output_dir}cpu_performance_comparison.png', 
            dpi=150, 
            bbox_inches='tight',
            pad_inches=0.05)

print(f"✅ CPU performance comparison saved to {output_dir}")
plt.close()

if __name__ == '__main__':
    print("CPU performance comparison plots generated successfully!")

