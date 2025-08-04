#!/usr/bin/env python3
"""Verify Type 3 scalability configuration is correct"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from scalability_utils import generate_scalability_config

print("="*60)
print("Type 3 Configuration Verification")
print("="*60)

# Test each database count
for db_count in [20, 40, 80, 120]:
    print(f"\n📊 Testing {db_count} databases configuration:")
    print("-" * 40)
    
    config = generate_scalability_config(db_count, 'S0_EMG_AS', 60)
    
    # Check database instances
    db_instances = config['database_instances']
    main_dbs = [db for db in db_instances if 'bg' not in db['id']]
    bg_dbs = [db for db in db_instances if 'bg' in db['id']]
    
    print(f"  Total databases: {len(db_instances)}")
    print(f"  Main databases: {len(main_dbs)}")
    print(f"  Background databases: {len(bg_dbs)}")
    
    # Verify main databases
    main_ids = [db['id'] for db in main_dbs]
    expected_main = ['db_high_priority', 'db_medium_priority', 'db_low_priority']
    
    if sorted(main_ids) == sorted(expected_main):
        print(f"  ✅ Main databases correct: {', '.join(main_ids)}")
    else:
        print(f"  ❌ Main databases incorrect!")
        print(f"     Expected: {expected_main}")
        print(f"     Got: {main_ids}")
    
    # Check scalability experiment flag
    scalability_config = config['ycsb_general_config']['scalability_experiment']
    if scalability_config['enabled']:
        print(f"  ✅ Active set strategy: ENABLED")
    else:
        print(f"  ❌ Active set strategy: DISABLED (should be enabled!)")
    
    # Check TPS distribution
    phase = config['dynamic_workload_phases'][0]
    tps_dist = phase['ycsb_config_overrides']['tps_distribution_per_db']
    
    main_tps = [tps_dist.get(db_id, 0) for db_id in expected_main]
    bg_tps_sample = [tps_dist.get(f'db_bg_{i}', 0) for i in range(1, min(4, len(bg_dbs)+1))]
    
    print(f"  TPS for main DBs: {main_tps}")
    print(f"  TPS for bg DBs (sample): {bg_tps_sample}")

print("\n" + "="*60)
print("Verification complete!")
print("="*60)