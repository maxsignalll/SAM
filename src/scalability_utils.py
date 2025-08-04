#!/usr/bin/env python3
"""
Utility functions for Type 3 Scalability Experiments
"""

import json
import os
from typing import Dict, List, Any
from pathlib import Path
import copy


def generate_scalability_config(db_count: int, strategy: str, duration: int = 60) -> Dict[str, Any]:
    """
    Generate a scalability test configuration for a given database count and strategy.
    
    Args:
        db_count: Number of databases to test with
        strategy: Strategy name (e.g., 'S0_EMG_AS', 'B7_DYNAMIC_NEED')
        duration: Test duration in seconds
    
    Returns:
        Complete configuration dictionary
    """
    # Load base template
    template_path = Path(__file__).parent.parent / 'configs' / 'config_scalability_template.json'
    
    if not template_path.exists():
        # Try alternative path
        template_path = Path(__file__).parent.parent / 'configs' / 'config_scalability_test.json'
    
    if not template_path.exists():
        # Create a minimal template if it doesn't exist
        base_config = create_minimal_scalability_template()
    else:
        with open(template_path, 'r') as f:
            base_config = json.load(f)
    
    # Deep copy to avoid modifying the original
    config = copy.deepcopy(base_config)
    
    # Update general experiment setup
    config['general_experiment_setup']['experiment_name'] = f"Type3_Scalability_{strategy}_{db_count}db"
    config['general_experiment_setup']['active_strategy'] = strategy
    config['general_experiment_setup']['total_duration_seconds'] = duration
    
    # Generate database instances
    db_instances = generate_database_instances(db_count)
    config['database_instances'] = db_instances
    
    # Update workload phases for new databases
    update_workload_phases(config, db_instances)
    
    # Enable scalability experiment flag
    if 'ycsb_general_config' not in config:
        config['ycsb_general_config'] = {}
    
    config['ycsb_general_config']['scalability_experiment'] = {
        'enabled': True,  # Use active set version for scalability testing
        'database_count': db_count,
        'background_db_path': 'data/type3/',
        'background_db_record_count': 10000
    }
    
    return config


def generate_database_instances(db_count: int) -> List[Dict[str, Any]]:
    """
    Generate database instance configurations for scalability testing.
    
    Fixed configuration:
    - 3 main databases: high, medium, low priority (same as default config)
    - Remaining are background databases with low priority
    """
    instances = []
    
    # Always generate exactly 3 main databases (matching default configuration)
    # High priority database
    instances.append({
        "id": "db_high_priority",
        "base_priority": 1000,
        "priority": 1000,
        "ycsb_initial_record_count": 100000,
        "database_path": "data/type3/",
        "db_filename": "data/type3/db_high_priority.sqlite"
    })
    
    # Medium priority database
    instances.append({
        "id": "db_medium_priority",
        "base_priority": 100,
        "priority": 100,
        "ycsb_initial_record_count": 100000,
        "database_path": "data/type3/",
        "db_filename": "data/type3/db_medium_priority.sqlite"
    })
    
    # Low priority database
    instances.append({
        "id": "db_low_priority",
        "base_priority": 10,
        "priority": 10,
        "ycsb_initial_record_count": 10000,
        "database_path": "data/type3/",
        "db_filename": "data/type3/db_low_priority.sqlite"
    })
    
    # Generate background databases for the remainder
    bg_count = db_count - 3
    for i in range(1, bg_count + 1):
        db_id = f"db_bg_{i}"
        instances.append({
            "id": db_id,
            "base_priority": 1,
            "priority": 1,
            "ycsb_initial_record_count": 10000,
            "database_path": "data/type3/",
            "db_filename": f"data/type3/{db_id}.sqlite"
        })
    
    return instances


def update_workload_phases(config: Dict[str, Any], db_instances: List[Dict[str, Any]]) -> None:
    """
    Update workload phases configuration for new database instances.
    """
    if 'dynamic_workload_phases' not in config:
        config['dynamic_workload_phases'] = [create_default_phase()]
    
    for phase in config['dynamic_workload_phases']:
        if 'ycsb_config_overrides' not in phase:
            phase['ycsb_config_overrides'] = {}
        
        overrides = phase['ycsb_config_overrides']
        
        # Update TPS distribution
        if 'tps_distribution_per_db' not in overrides:
            overrides['tps_distribution_per_db'] = {}
        
        tps_dist = overrides['tps_distribution_per_db']
        
        # Set TPS for each database based on type
        for db in db_instances:
            db_id = db['id']
            if db_id not in tps_dist:
                # Main databases get normal TPS (matching default config)
                if db_id == 'db_high_priority':
                    tps_dist[db_id] = 1.0  # Main high priority
                elif db_id == 'db_medium_priority':
                    tps_dist[db_id] = 1.0  # Main medium priority
                elif db_id == 'db_low_priority':
                    tps_dist[db_id] = 1.0  # Main low priority
                else:
                    # Background databases get low TPS
                    tps_dist[db_id] = 1.0  # Background databases
        
        # Update access patterns
        if 'access_pattern_per_db' not in overrides:
            overrides['access_pattern_per_db'] = {}
        
        access_patterns = overrides['access_pattern_per_db']
        
        for db in db_instances:
            db_id = db['id']
            if db_id not in access_patterns:
                # All databases use zipfian distribution for consistency
                access_patterns[db_id] = {
                    "distribution": "zipfian",
                    "zipf_alpha": 0.9
                }


def create_default_phase() -> Dict[str, Any]:
    """Create a default workload phase for scalability testing."""
    return {
        "name": "Scalability_Test_Phase",
        "duration_seconds": 60,
        "ycsb_config_overrides": {
            "threadcount": 8,
            "tps_distribution_per_db": {},
            "access_pattern_per_db": {}
        }
    }


def create_minimal_scalability_template() -> Dict[str, Any]:
    """Create a minimal scalability test configuration template."""
    return {
        "general_experiment_setup": {
            "experiment_name": "Type3_Scalability_Test",
            "workload_mode": "static",
            "workload_type": "ycsb",
            "random_seed": 42,
            "output_directory": "results",
            "log_level": "INFO",
            "active_strategy": "S0_EMG_AS",
            "reporting_interval_seconds": 4,
            "measurement_interval_seconds": 4,
            "data_collection_timeout_seconds": 1.0,
            "save_logs": True,
            "total_duration_seconds": 60
        },
        "system_sqlite_config": {
            "page_size_bytes": 4096,
            "total_ram_for_cache_mb": 20
        },
        "dynamic_workload_phases": [create_default_phase()],
        "database_instances": [],
        "ycsb_general_config": {
            "ycsb_row_size_bytes": 2048,
            "operation_distribution": {
                "read": 0.8,
                "update": 0.2
            },
            "hot_rate": 0.2
        },
        "strategy_configurations": {
            "S0_EMG_AS": {
                "adjustment_interval_seconds": 5,
                "fixed_pool_percentage_of_total_ram": 0.3
            },
            "B7_DYNAMIC_NEED": {
                "tuning_interval": 5.0,
                "ema_alpha": 0.1
            }
        }
    }


def aggregate_scalability_results(results_by_scale: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Aggregate scalability test results into a unified format.
    
    Args:
        results_by_scale: Dictionary with keys like "20_S0_EMG_AS" containing CPU time data
    
    Returns:
        Unified results dictionary
    """
    import datetime
    
    # Extract unique database counts and strategies
    db_counts = sorted(list(set(int(key.split('_')[0]) for key in results_by_scale.keys())))
    strategies = sorted(list(set('_'.join(key.split('_')[1:]) for key in results_by_scale.keys())))
    
    summary = {
        'metadata': {
            'timestamp': datetime.datetime.now().isoformat(),
            'experiment_type': 3,
            'database_counts': db_counts,
            'strategies': strategies
        },
        'results': {}
    }
    
    # Organize results by database count
    for db_count in db_counts:
        summary['results'][str(db_count)] = {}
        
        for strategy in strategies:
            key = f"{db_count}_{strategy}"
            if key in results_by_scale:
                data = results_by_scale[key]
                summary['results'][str(db_count)][strategy] = {
                    'mean_ms': data.get('mean', 0) * 1000,  # Convert to milliseconds
                    'std_ms': data.get('std', 0) * 1000,
                    'samples': data.get('samples', []),
                    'sample_count': len(data.get('samples', []))
                }
    
    return summary


def save_scalability_summary(results: Dict[str, Any], output_dir: str = None) -> str:
    """
    Save scalability test summary to JSON file.
    
    Args:
        results: Aggregated results dictionary
        output_dir: Output directory (default: results/scalability/)
    
    Returns:
        Path to saved file
    """
    if output_dir is None:
        output_dir = 'results/scalability'
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'type3_summary_{timestamp}.json'
    filepath = os.path.join(output_dir, filename)
    
    # Save JSON
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Also save as latest for easy access
    latest_path = os.path.join(output_dir, 'type3_summary_latest.json')
    with open(latest_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return filepath


def load_latest_scalability_results() -> Dict[str, Any]:
    """
    Load the latest scalability test results.
    
    Returns:
        Results dictionary or None if not found
    """
    latest_path = 'results/scalability/type3_summary_latest.json'
    
    if os.path.exists(latest_path):
        with open(latest_path, 'r') as f:
            return json.load(f)
    
    # Try to find any summary file
    scalability_dir = 'results/scalability'
    if os.path.exists(scalability_dir):
        files = [f for f in os.listdir(scalability_dir) if f.startswith('type3_summary_') and f.endswith('.json')]
        if files:
            # Sort by filename (which includes timestamp) and get the latest
            files.sort()
            with open(os.path.join(scalability_dir, files[-1]), 'r') as f:
                return json.load(f)
    
    return None