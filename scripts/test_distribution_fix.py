#!/usr/bin/env python3
"""Quick test to verify the 'latest' distribution fix"""

import json
from pathlib import Path

def check_distribution_types(config_file):
    """Check all distribution types in a config file"""
    print(f"\nChecking: {config_file}")
    print("-" * 60)
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    issues_found = False
    
    # Check dynamic workload phases
    if 'dynamic_workload_phases' in config:
        for phase in config['dynamic_workload_phases']:
            phase_name = phase.get('name', 'Unknown')
            
            if 'ycsb_config_overrides' in phase and 'access_pattern_per_db' in phase['ycsb_config_overrides']:
                patterns = phase['ycsb_config_overrides']['access_pattern_per_db']
                
                for db_id, pattern_config in patterns.items():
                    if 'distribution' in pattern_config:
                        dist_type = pattern_config['distribution']
                        
                        if dist_type not in ['zipfian', 'uniform']:
                            print(f"  ❌ ISSUE: Phase '{phase_name}', DB '{db_id}' uses unsupported distribution: '{dist_type}'")
                            issues_found = True
                        else:
                            if phase_name == "Phase2D_Second_Wave" and db_id == "db_medium_priority":
                                print(f"  ✅ FIXED: Phase '{phase_name}', DB '{db_id}' now uses: '{dist_type}'")
                                if dist_type == 'zipfian':
                                    alpha = pattern_config.get('zipf_alpha', 'N/A')
                                    print(f"     zipf_alpha: {alpha}")
    
    if not issues_found:
        print("  ✅ All distribution types are valid!")
    
    return not issues_found

def main():
    print("="*60)
    print("Distribution Type Validation Test")
    print("="*60)
    
    config_files = [
        'configs/config_tps_controlled_comparison.json',
        'configs/config_dual_combat.json',
        'configs/config_burst_scan_optimized.json'
    ]
    
    all_valid = True
    for config_file in config_files:
        if Path(config_file).exists():
            valid = check_distribution_types(config_file)
            all_valid = all_valid and valid
        else:
            print(f"\n⚠️ Config file not found: {config_file}")
    
    print("\n" + "="*60)
    if all_valid:
        print("✅ All configuration files have valid distribution types!")
    else:
        print("❌ Some configuration files have invalid distribution types")
    print("="*60)

if __name__ == "__main__":
    main()