#!/usr/bin/env python3
"""Test script to verify CPU timing collection is working"""

import sys
import json
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from scalability_utils import generate_scalability_config
from experiment_manager import ExperimentOrchestrator

print("="*60)
print("Testing CPU Timing Collection")
print("="*60)

# Test with minimal configuration (20 databases, 10 seconds)
db_count = 20
duration = 10
strategy = 'B7_DYNAMIC_NEED'

print(f"\n🔧 Generating configuration for {db_count} databases...")
config = generate_scalability_config(db_count, strategy, duration)

# Verify scalability experiment is enabled
scalability_config = config['ycsb_general_config'].get('scalability_experiment', {})
is_enabled = scalability_config.get('enabled', False)
print(f"✓ Scalability experiment enabled: {is_enabled}")

# Set output directory
output_dir = Path("results/test_cpu_timing")
output_dir.mkdir(parents=True, exist_ok=True)
config['general_experiment_setup']['output_directory'] = str(output_dir)

print(f"\n🚀 Creating orchestrator and running experiment...")
try:
    orchestrator = ExperimentOrchestrator(final_config=config)
    
    # Check if scalability flag is set
    print(f"✓ Orchestrator scalability flag: {orchestrator.is_scalability_experiment}")
    print(f"✓ CPU times list initialized: {hasattr(orchestrator, 'decision_cpu_times')}")
    
    # Run the experiment
    output_csv = output_dir / "data.csv"
    orchestrator.run(output_csv_path=output_csv)
    
    # Check CPU timing data
    if hasattr(orchestrator, 'decision_cpu_times'):
        cpu_times = orchestrator.decision_cpu_times
        print(f"\n📊 CPU Timing Results:")
        print(f"  Samples collected: {len(cpu_times)}")
        
        if cpu_times:
            mean_time = np.mean(cpu_times)
            std_time = np.std(cpu_times)
            print(f"  Mean: {mean_time*1000:.3f}ms")
            print(f"  Std: {std_time*1000:.3f}ms")
            print(f"  Min: {min(cpu_times)*1000:.3f}ms")
            print(f"  Max: {max(cpu_times)*1000:.3f}ms")
            print(f"  First 5 samples (ms): {[f'{t*1000:.3f}' for t in cpu_times[:5]]}")
        else:
            print("  ⚠️ No CPU timing data collected!")
    else:
        print("\n❌ decision_cpu_times attribute not found in orchestrator")
        
    # Cleanup
    orchestrator.cleanup()
    
    print("\n✅ Test complete!")
    
except Exception as e:
    print(f"\n❌ Test failed: {e}")
    import traceback
    traceback.print_exc()