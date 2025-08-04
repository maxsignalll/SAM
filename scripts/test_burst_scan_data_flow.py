#!/usr/bin/env python3
"""
Test script to verify burst-scan-only data flow
"""

import os
import glob
from pathlib import Path

def find_burst_scan_results():
    """Find all burst_scan_results.txt files"""
    print("🔍 Searching for burst_scan_results.txt files...")
    
    patterns = [
        'results/comparison/experiment_*/burst_scan_results.txt',
        'results/comparison/experiment_*/burst_scan/burst_scan_results.txt',
        'figures/burst_scan_results.txt',
        'SAM-reproducible/figures/burst_scan_results.txt'
    ]
    
    found_files = []
    for pattern in patterns:
        files = glob.glob(pattern)
        found_files.extend(files)
    
    if not found_files:
        print("❌ No burst_scan_results.txt files found")
        return []
    
    print(f"\n✅ Found {len(found_files)} file(s):")
    for f in found_files:
        # Get file modification time
        mtime = Path(f).stat().st_mtime
        from datetime import datetime
        mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"\n📄 {f}")
        print(f"   Modified: {mtime_str}")
        
        # Read and display content
        with open(f, 'r') as file:
            content = file.read()
            print("   Content:")
            for line in content.strip().split('\n'):
                print(f"     {line}")
    
    return found_files

def find_latest_experiment():
    """Find the latest experiment directory"""
    pattern = 'results/comparison/experiment_*'
    all_exps = glob.glob(pattern)
    
    if not all_exps:
        print("❌ No experiment directories found")
        return None
    
    latest = max(all_exps, key=lambda x: Path(x).stat().st_mtime)
    print(f"\n📂 Latest experiment: {latest}")
    
    # Check for burst_scan subdirectory
    burst_scan_dir = Path(latest) / 'burst_scan'
    if burst_scan_dir.exists():
        print(f"   ✅ Has burst_scan subdirectory")
    else:
        print(f"   ⚠️ No burst_scan subdirectory (likely burst-scan-only mode)")
    
    # List all files in experiment directory
    print(f"   Files in experiment root:")
    for f in Path(latest).iterdir():
        if f.is_file():
            print(f"     - {f.name}")
    
    return latest

def simulate_plot_search(exp_path):
    """Simulate how plot_dual_combat_story.py searches for files"""
    print(f"\n🎯 Simulating plot script search with exp_path: {exp_path}")
    
    search_paths = []
    
    # Priority 1: Check experiment path if provided
    if exp_path:
        search_paths.extend([
            os.path.join(exp_path, 'burst_scan_results.txt'),
            os.path.join(exp_path, 'burst_scan', 'burst_scan_results.txt'),
            os.path.join(os.path.dirname(exp_path), 'burst_scan_results.txt')
        ])
    
    # Priority 2: Check figures directory (only as fallback if no exp_path)
    # NOTE: After our fix, this should only happen if exp_path is None
    if not exp_path:
        search_paths.extend([
            'figures/burst_scan_results.txt',
            'SAM-reproducible/figures/burst_scan_results.txt'
        ])
    
    print("   Search order:")
    for i, path in enumerate(search_paths, 1):
        exists = os.path.exists(path)
        status = "✅ EXISTS" if exists else "❌ not found"
        print(f"     {i}. {path} ... {status}")
        if exists:
            print(f"        → Would use this file!")
            with open(path, 'r') as f:
                for line in f:
                    key, value = line.strip().split('=')
                    print(f"          {key}: {value}")
            return path
    
    print("   ⚠️ No file found, would use hardcoded defaults!")
    return None

def main():
    print("="*60)
    print("🧪 BURST-SCAN-ONLY DATA FLOW TEST")
    print("="*60)
    
    # Step 1: Find all burst_scan_results.txt files
    found_files = find_burst_scan_results()
    
    # Step 2: Find latest experiment
    latest_exp = find_latest_experiment()
    
    # Step 3: Simulate plot search
    if latest_exp:
        simulate_plot_search(latest_exp)
    
    # Step 4: Check for potential issues
    print("\n" + "="*60)
    print("🔍 POTENTIAL ISSUES:")
    
    # Check if figures/burst_scan_results.txt exists
    if os.path.exists('figures/burst_scan_results.txt'):
        print("⚠️ Global figures/burst_scan_results.txt exists - may cause stale data issues")
        print("   Recommendation: Delete it with 'rm figures/burst_scan_results.txt'")
    
    # Check if latest experiment has the file in expected location
    if latest_exp:
        expected_file = Path(latest_exp) / 'burst_scan_results.txt'
        if not expected_file.exists():
            print(f"⚠️ Latest experiment missing burst_scan_results.txt at expected location")
            print(f"   Expected: {expected_file}")
    
    print("\n" + "="*60)

if __name__ == '__main__':
    main()