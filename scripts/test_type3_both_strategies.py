#!/usr/bin/env python3
"""Quick test to verify both S0 and B7 strategies run in Type 3 experiment"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from scalability_utils import generate_scalability_config, save_scalability_summary
from run_cache_strategy_comparison import CacheStrategyComparison

print("="*60)
print("Type 3 Dual Strategy Test - S0 and B7")
print("="*60)

# Test with 20 databases, 10 seconds each
db_count = 20
duration = 10
results_by_scale = {}

for strategy in ['S0_EMG_AS', 'B7_DYNAMIC_NEED']:
    print(f"\n🚀 Testing {strategy} with {db_count} databases for {duration}s...")
    print("-" * 40)
    
    try:
        # Generate configuration
        config = generate_scalability_config(db_count, strategy, duration)
        
        # Verify active set is enabled for S0
        if strategy == 'S0_EMG_AS':
            scalability_enabled = config['ycsb_general_config']['scalability_experiment']['enabled']
            print(f"  Active set strategy: {'ENABLED' if scalability_enabled else 'DISABLED'}")
        
        # Create comparison instance
        comparison = CacheStrategyComparison(
            base_output_dir=f"results/scalability/test_{db_count}db_{strategy}"
        )
        
        # Save config
        config_path = Path(comparison.base_output_dir) / "config_generated.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Update config
        comparison.config = config
        comparison.config['general_experiment_setup']['output_directory'] = str(comparison.base_output_dir)
        
        print("  Preparing environment...")
        comparison.prepare_experiment_environment(force_clean_db=False)
        
        print("  Running experiment...")
        comparison.run_all_strategies(strategies_to_run_from_cli=[strategy])
        
        # Check for CPU timing data
        if hasattr(comparison, 'orchestrator') and hasattr(comparison.orchestrator, 'decision_cpu_times'):
            cpu_times = comparison.orchestrator.decision_cpu_times
            if cpu_times:
                import numpy as np
                mean_time = np.mean(cpu_times)
                std_time = np.std(cpu_times)
                
                results_by_scale[f"{db_count}_{strategy}"] = {
                    'mean': mean_time,
                    'std': std_time,
                    'samples': cpu_times[:10]
                }
                
                print(f"\n  ✅ {strategy} Success!")
                print(f"     Samples: {len(cpu_times)}")
                print(f"     Mean: {mean_time*1000:.3f}ms")
                print(f"     Std: {std_time*1000:.3f}ms")
            else:
                print(f"\n  ⚠️ {strategy}: No CPU timing data")
        else:
            print(f"\n  ⚠️ {strategy}: Orchestrator not available")
            
    except Exception as e:
        print(f"\n  ❌ {strategy} failed: {e}")

# Check if both strategies ran
print("\n" + "="*60)
print("Test Summary:")
print("="*60)

if len(results_by_scale) == 2:
    print("✅ Both strategies ran successfully!")
    
    # Compare performance
    s0_mean = results_by_scale.get(f"{db_count}_S0_EMG_AS", {}).get('mean', 0) * 1000
    b7_mean = results_by_scale.get(f"{db_count}_B7_DYNAMIC_NEED", {}).get('mean', 0) * 1000
    
    print(f"\nPerformance Comparison:")
    print(f"  S0 (Active Set): {s0_mean:.3f}ms")
    print(f"  B7 (Baseline):   {b7_mean:.3f}ms")
    
    if s0_mean > 0 and b7_mean > 0:
        ratio = s0_mean / b7_mean
        print(f"  Ratio (S0/B7):   {ratio:.2f}x")
else:
    print(f"❌ Only {len(results_by_scale)} strategy ran successfully")
    
print("\nTest complete!")