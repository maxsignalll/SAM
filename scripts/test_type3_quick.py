#!/usr/bin/env python3
"""Quick test of Type 3 experiment with CPU timing collection fix"""

import sys
import json
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from scalability_utils import generate_scalability_config, save_scalability_summary
from run_cache_strategy_comparison import CacheStrategyComparison

print("="*60)
print("Quick Type 3 Test - CPU Timing Collection Fix")
print("="*60)

# Test with one scale, both strategies, short duration
db_counts = [20]
strategies = ['S0_EMG_AS', 'B7_DYNAMIC_NEED']
duration = 10  # 10 seconds for quick test

results_by_scale = {}

for db_count in db_counts:
    print(f"\n🔬 Testing with {db_count} databases")
    print("-" * 40)
    
    for strategy in strategies:
        print(f"\n📊 Running {strategy}...")
        
        try:
            # Generate configuration
            config = generate_scalability_config(db_count, strategy, duration)
            
            # Verify scalability is enabled
            is_enabled = config['ycsb_general_config']['scalability_experiment']['enabled']
            print(f"  Scalability enabled: {is_enabled}")
            
            # Create comparison instance
            test_comparison = CacheStrategyComparison(
                base_output_dir=f"results/scalability/quick_test_{db_count}db_{strategy}"
            )
            
            # Save config
            config_path = Path(test_comparison.base_output_dir) / "config_generated.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Update config
            test_comparison.config = config
            test_comparison.config['general_experiment_setup']['output_directory'] = str(test_comparison.base_output_dir)
            
            # Prepare and run
            print(f"  Preparing environment...")
            test_comparison.prepare_experiment_environment(force_clean_db=False)
            
            print(f"  Running experiment...")
            test_comparison.run_all_strategies(strategies_to_run_from_cli=[strategy])
            
            # Extract CPU timing data
            print(f"  Checking for CPU timing data...")
            
            if hasattr(test_comparison, 'orchestrator'):
                print(f"  ✓ Orchestrator available")
                
                if hasattr(test_comparison.orchestrator, 'decision_cpu_times'):
                    cpu_times = test_comparison.orchestrator.decision_cpu_times
                    print(f"  ✓ decision_cpu_times found")
                    
                    if cpu_times:
                        mean_time = np.mean(cpu_times)
                        std_time = np.std(cpu_times)
                        
                        results_by_scale[f"{db_count}_{strategy}"] = {
                            'mean': mean_time,
                            'std': std_time,
                            'samples': cpu_times[:10]
                        }
                        
                        print(f"  ✅ SUCCESS! Collected {len(cpu_times)} samples")
                        print(f"     Mean: {mean_time*1000:.3f}ms")
                        print(f"     Std: {std_time*1000:.3f}ms")
                        print(f"     First 3: {[f'{t*1000:.3f}ms' for t in cpu_times[:3]]}")
                    else:
                        print(f"  ⚠️ decision_cpu_times is empty")
                else:
                    print(f"  ❌ decision_cpu_times attribute not found")
            else:
                print(f"  ❌ Orchestrator not available")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()

# Summary
print("\n" + "="*60)
print("Test Summary:")
print("="*60)

if results_by_scale:
    print("✅ CPU timing collection is working!")
    print("\nResults:")
    for key, data in results_by_scale.items():
        db_count, strategy = key.rsplit('_', 1)
        strategy_parts = strategy.split('_')
        strategy_name = strategy_parts[0] if strategy_parts else strategy
        mean_ms = data['mean'] * 1000
        std_ms = data['std'] * 1000
        print(f"  {strategy_name:<5} ({db_count} DBs): {mean_ms:.3f} ± {std_ms:.3f} ms")
else:
    print("❌ No CPU timing data collected - fix still needed")

print("\nTest complete!")