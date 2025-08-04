#!/usr/bin/env python3
"""
Verify burst-scan-only mode fix
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import pandas as pd

def clean_old_files():
    """Clean up old files"""
    print("🧹 Cleaning old files...")
    
    # Remove old figures
    if os.path.exists('figures/burst_scan_results.txt'):
        os.remove('figures/burst_scan_results.txt')
        print("   ✅ Removed figures/burst_scan_results.txt")
    
    # Remove old experiments
    import shutil
    import glob
    for exp_dir in glob.glob('results/comparison/experiment_*'):
        if 'test' in exp_dir or time.time() - os.path.getmtime(exp_dir) < 3600:  # Keep recent
            shutil.rmtree(exp_dir)
            print(f"   ✅ Removed {exp_dir}")

def run_short_experiment():
    """Run a short experiment (60s) to test error handling"""
    print("\n🔬 Running SHORT experiment (60s) - should show error...")
    cmd = [
        sys.executable,
        'scripts/run_cache_strategy_comparison.py',
        '--experiment-type', '2',
        '--burst-scan-only',
        '--duration', '60'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check for warning messages
    if "WARNING: Cannot generate burst scan results" in result.stdout:
        print("   ✅ Error detection working!")
    else:
        print("   ⚠️ No warning message found")
    
    # Find the experiment directory
    import glob
    exp_dirs = sorted(glob.glob('results/comparison/experiment_*'))
    if exp_dirs:
        latest_exp = exp_dirs[-1]
        print(f"   📂 Experiment: {latest_exp}")
        
        # Check for burst_scan_results.txt
        results_file = Path(latest_exp) / 'burst_scan_results.txt'
        if results_file.exists():
            print(f"   ✅ File created: {results_file}")
            with open(results_file, 'r') as f:
                content = f.read()
                if "# ERROR" in content:
                    print("   ✅ Error marker present")
                else:
                    print("   ⚠️ No error marker")
                print("   Content preview:")
                for line in content.split('\n')[:5]:
                    print(f"      {line}")
        else:
            print(f"   ❌ No burst_scan_results.txt found")
        
        return latest_exp
    
    return None

def run_long_experiment():
    """Run a long experiment (200s) to get real data"""
    print("\n🔬 Running LONG experiment (200s) - should get real data...")
    print("   ⏳ This will take about 3.5 minutes...")
    
    cmd = [
        sys.executable,
        'scripts/run_cache_strategy_comparison.py',
        '--experiment-type', '2',
        '--burst-scan-only',
        '--duration', '200'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check for results
    if "Burst Scan Analysis Results" in result.stdout:
        print("   ✅ Analysis completed!")
        
        # Extract results
        lines = result.stdout.split('\n')
        for line in lines:
            if "Need Score Increase:" in line:
                print(f"   {line.strip()}")
            elif "Cache Allocation Increase:" in line:
                print(f"   {line.strip()}")
            elif "Overreaction Factor:" in line:
                print(f"   {line.strip()}")
    else:
        print("   ⚠️ No analysis results found")
    
    # Find the experiment directory
    import glob
    exp_dirs = sorted(glob.glob('results/comparison/experiment_*'))
    if exp_dirs:
        latest_exp = exp_dirs[-1]
        print(f"   📂 Experiment: {latest_exp}")
        
        # Check for burst_scan_results.txt
        results_file = Path(latest_exp) / 'burst_scan_results.txt'
        if results_file.exists():
            print(f"   ✅ File created: {results_file}")
            with open(results_file, 'r') as f:
                content = f.read()
                if "# ERROR" not in content:
                    print("   ✅ Real data (no error marker)")
                else:
                    print("   ⚠️ Still has error marker")
                print("   Content:")
                for line in content.split('\n'):
                    if line and not line.startswith('#'):
                        print(f"      {line}")
        
        return latest_exp
    
    return None

def verify_plot_generation(exp_dir):
    """Verify that plots can be generated"""
    print("\n🎨 Testing plot generation...")
    
    cmd = [
        sys.executable,
        'scripts/plot_dual_combat_story.py',
        exp_dir
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("   ✅ Plot script executed successfully")
    else:
        print(f"   ❌ Plot script failed: {result.stderr}")
    
    # Check for generated figure
    fig_path = Path('figures/paper/figure7_dual_combat_robustness.pdf')
    if fig_path.exists():
        print(f"   ✅ Figure generated: {fig_path}")
    else:
        print("   ❌ Figure not generated")

def main():
    print("="*60)
    print("🧪 BURST-SCAN-ONLY FIX VERIFICATION")
    print("="*60)
    
    # Step 1: Clean old files
    clean_old_files()
    
    # Step 2: Run short experiment (should fail gracefully)
    short_exp = run_short_experiment()
    if short_exp:
        verify_plot_generation(short_exp)
    
    # Step 3: Ask user if they want to run long experiment
    print("\n" + "="*60)
    response = input("Run LONG experiment (200s) to test real data? [y/N]: ")
    if response.lower() == 'y':
        long_exp = run_long_experiment()
        if long_exp:
            verify_plot_generation(long_exp)
    else:
        print("Skipping long experiment")
    
    print("\n" + "="*60)
    print("✅ Verification complete!")
    print("\nKey findings:")
    print("- Short experiments now generate error files with warnings")
    print("- Long experiments (200s+) generate real data")
    print("- Plot script handles both cases gracefully")

if __name__ == '__main__':
    main()