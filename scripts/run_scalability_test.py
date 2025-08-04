#!/usr/bin/env python3
"""
Type 3 Experiment: Scalability Test Runner

This script implements a comprehensive scalability testing framework for the S0 strategy.
It automatically runs experiments across different database counts and collects CPU performance data.
"""

import os
import sys
import json
import time
import shutil
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_cache_strategy_comparison import CacheStrategyComparison

class ScalabilityTestOrchestrator:
    """Orchestrator for running scalability tests across multiple database counts"""
    
    def __init__(self, base_output_dir: str = None):
        self.base_output_dir = base_output_dir or "results/scalability"
        self.test_results = {}
        self.config_template_path = "configs/config_scalability_test.json"
        
        # Default scalability test parameters
        self.default_db_counts = [20, 40, 80, 120, 160]
        self.default_test_duration = 180  # seconds per scale
        self.default_warmup_duration = 30
        self.default_runs_per_scale = 3
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
    def load_config_template(self) -> Dict[str, Any]:
        """Load the scalability test configuration template"""
        try:
            with open(self.config_template_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Configuration template not found: {self.config_template_path}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in configuration template: {e}")
            raise
    
    def generate_config_for_scale(self, db_count: int, run_id: int = 1) -> Dict[str, Any]:
        """Generate configuration for specific database count"""
        config = self.load_config_template()
        
        # Update experiment name
        config['general_experiment_setup']['experiment_name'] = f"Scalability_Test_{db_count}DB_Run{run_id}"
        
        # Update database instances
        config['database_instances'] = {}
        
        # Generate database instances dynamically
        for i in range(1, db_count + 1):
            db_id = f"scale_db_{i:03d}"
            config['database_instances'][db_id] = {
                "priority": 5,
                "record_count": 10000,
                "database_path": "data/",
                "description": f"Scalability test database {i}/{db_count}"
            }
        
        # Update workload phases with proper TPS distribution
        phase_config = config['dynamic_workload_phases'][0]
        phase_config['duration_seconds'] = self.default_test_duration
        
        # Generate TPS distribution (2 TPS per database)
        tps_distribution = {}
        access_pattern = {}
        for i in range(1, db_count + 1):
            db_id = f"scale_db_{i:03d}"
            tps_distribution[db_id] = 2
            access_pattern[db_id] = {"distribution": "zipfian", "zipf_alpha": 0.9}
        
        phase_config['ycsb_config_overrides']['tps_distribution_per_db'] = tps_distribution
        phase_config['ycsb_config_overrides']['access_pattern_per_db'] = access_pattern
        
        # Update scalability experiment config
        config['ycsb_general_config']['scalability_experiment']['total_database_count'] = db_count
        
        return config
    
    def run_single_scale_test(self, db_count: int, run_id: int = 1) -> Dict[str, Any]:
        """Run a single scalability test for specific database count"""
        self.logger.info(f"Starting scalability test: {db_count} databases, run {run_id}")
        
        # Generate configuration
        config = self.generate_config_for_scale(db_count, run_id)
        
        # Create output directory for this scale
        scale_output_dir = Path(self.base_output_dir) / f"scale_{db_count:03d}db" / f"run_{run_id}"
        scale_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = scale_output_dir / "config_generated.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Initialize strategy comparison with this configuration
        comparison = CacheStrategyComparison(base_output_dir=str(scale_output_dir))
        
        # Update config file to our generated one
        comparison.update_config_file(str(config_path))
        
        # Prepare environment (clean databases for fresh start)
        comparison.prepare_experiment_environment(force_clean_db=True)
        
        # Run S0 strategy test
        comparison.run_all_strategies(strategies_to_run_from_cli=['S0_EMG_AS'])
        
        # Collect results
        results = {
            'db_count': db_count,
            'run_id': run_id,
            'experiment_dir': str(comparison.experiment_dir),
            'config_path': str(config_path),
            'results': comparison.experiment_results.copy(),
            'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Look for CPU timing data
        timing_file = comparison.experiment_dir / "individual_results" / "S0_EMG_AS" / "scalability_cpu_timings.json"
        if timing_file.exists():
            with open(timing_file, 'r') as f:
                timing_data = json.load(f)
                results['cpu_timing_data'] = timing_data
                self.logger.info(f"CPU timing data collected: {len(timing_data.get('raw_cpu_times_seconds', []))} samples")
        else:
            self.logger.warning(f"CPU timing data not found: {timing_file}")
            results['cpu_timing_data'] = None
        
        return results
    
    def run_full_scalability_test(self, db_counts: List[int] = None, runs_per_scale: int = None) -> Dict[str, Any]:
        """Run full scalability test across all database counts"""
        db_counts = db_counts or self.default_db_counts
        runs_per_scale = runs_per_scale or self.default_runs_per_scale
        
        self.logger.info(f"Starting full scalability test")
        self.logger.info(f"Database counts: {db_counts}")
        self.logger.info(f"Runs per scale: {runs_per_scale}")
        self.logger.info(f"Output directory: {self.base_output_dir}")
        
        all_results = {
            'test_summary': {
                'db_counts': db_counts,
                'runs_per_scale': runs_per_scale,
                'total_tests': len(db_counts) * runs_per_scale,
                'start_time': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'scale_results': {}
        }
        
        # Run tests for each scale
        for db_count in db_counts:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Testing scale: {db_count} databases")
            self.logger.info(f"{'='*60}")
            
            scale_results = []
            
            for run_id in range(1, runs_per_scale + 1):
                try:
                    result = self.run_single_scale_test(db_count, run_id)
                    scale_results.append(result)
                    
                    # Log basic statistics if timing data available
                    if result.get('cpu_timing_data') and result['cpu_timing_data'].get('statistics'):
                        stats = result['cpu_timing_data']['statistics']
                        self.logger.info(f"  Run {run_id}: Mean CPU time = {stats['mean_seconds']*1000:.3f} ms")
                    
                except Exception as e:
                    self.logger.error(f"Failed to run scale test {db_count}DB run {run_id}: {e}")
                    continue
            
            all_results['scale_results'][db_count] = scale_results
            
            # Calculate aggregate statistics for this scale
            if scale_results:
                cpu_times = []
                for result in scale_results:
                    if result.get('cpu_timing_data') and result['cpu_timing_data'].get('raw_cpu_times_seconds'):
                        cpu_times.extend(result['cpu_timing_data']['raw_cpu_times_seconds'])
                
                if cpu_times:
                    aggregate_stats = {
                        'db_count': db_count,
                        'total_samples': len(cpu_times),
                        'mean_cpu_time_ms': sum(cpu_times) / len(cpu_times) * 1000,
                        'min_cpu_time_ms': min(cpu_times) * 1000,
                        'max_cpu_time_ms': max(cpu_times) * 1000,
                        'std_cpu_time_ms': self._calculate_std(cpu_times) * 1000
                    }
                    all_results['scale_results'][f'{db_count}_aggregate'] = aggregate_stats
                    
                    self.logger.info(f"Scale {db_count} aggregate: {aggregate_stats['mean_cpu_time_ms']:.3f} ± {aggregate_stats['std_cpu_time_ms']:.3f} ms")
        
        # Save comprehensive results
        all_results['test_summary']['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
        results_file = Path(self.base_output_dir) / "scalability_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        self.logger.info(f"\nScalability test completed. Results saved to: {results_file}")
        
        return all_results
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def generate_summary_report(self, results: Dict[str, Any] = None):
        """Generate a summary report of scalability test results"""
        if results is None:
            # Load results from file
            results_file = Path(self.base_output_dir) / "scalability_test_results.json"
            if not results_file.exists():
                self.logger.error("No results file found for summary report")
                return
            
            with open(results_file, 'r') as f:
                results = json.load(f)
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("SCALABILITY TEST SUMMARY REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Test summary
        summary = results['test_summary']
        report_lines.append(f"Test Period: {summary['start_time']} to {summary['end_time']}")
        report_lines.append(f"Database Counts Tested: {summary['db_counts']}")
        report_lines.append(f"Runs per Scale: {summary['runs_per_scale']}")
        report_lines.append(f"Total Tests: {summary['total_tests']}")
        report_lines.append("")
        
        # Performance summary
        report_lines.append("PERFORMANCE SUMMARY:")
        report_lines.append("-" * 40)
        report_lines.append(f"{'DB Count':<10} {'Mean CPU (ms)':<15} {'Std Dev (ms)':<15} {'Samples':<10}")
        report_lines.append("-" * 40)
        
        for db_count in summary['db_counts']:
            agg_key = f'{db_count}_aggregate'
            if agg_key in results['scale_results']:
                agg = results['scale_results'][agg_key]
                report_lines.append(f"{agg['db_count']:<10} {agg['mean_cpu_time_ms']:<15.3f} {agg['std_cpu_time_ms']:<15.3f} {agg['total_samples']:<10}")
        
        report_lines.append("")
        
        # Complexity analysis
        db_counts = []
        cpu_times = []
        for db_count in summary['db_counts']:
            agg_key = f'{db_count}_aggregate'
            if agg_key in results['scale_results']:
                agg = results['scale_results'][agg_key]
                db_counts.append(agg['db_count'])
                cpu_times.append(agg['mean_cpu_time_ms'])
        
        if len(db_counts) >= 2:
            # Calculate growth rate
            growth_rates = []
            for i in range(1, len(db_counts)):
                rate = (cpu_times[i] - cpu_times[i-1]) / (db_counts[i] - db_counts[i-1])
                growth_rates.append(rate)
            
            avg_growth_rate = sum(growth_rates) / len(growth_rates)
            
            report_lines.append("COMPLEXITY ANALYSIS:")
            report_lines.append("-" * 40)
            report_lines.append(f"Average growth rate: {avg_growth_rate:.4f} ms per additional database")
            
            # Linear complexity verification
            if avg_growth_rate < 0.1:  # Less than 0.1ms per DB
                complexity_assessment = "Excellent - Close to O(1)"
            elif avg_growth_rate < 0.5:  # Less than 0.5ms per DB  
                complexity_assessment = "Good - Close to O(N)"
            else:
                complexity_assessment = "Concerning - Higher than O(N)"
            
            report_lines.append(f"Complexity assessment: {complexity_assessment}")
        
        # Save and display report
        report_content = "\n".join(report_lines)
        report_file = Path(self.base_output_dir) / "scalability_summary_report.txt"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        print(report_content)
        self.logger.info(f"Summary report saved to: {report_file}")


def main():
    """Main function for scalability test runner"""
    parser = argparse.ArgumentParser(
        description="Run Type 3 Scalability Test for S0 Strategy",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--db-counts',
        nargs='*',
        type=int,
        default=[20, 40, 80, 120],
        help="List of database counts to test"
    )
    
    parser.add_argument(
        '--runs-per-scale',
        type=int,
        default=3,
        help="Number of runs per database count"
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default="results/scalability",
        help="Output directory for scalability test results"
    )
    
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help="Only generate summary report from existing results"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{args.output_dir}/scalability_test.log")
        ]
    )
    
    # Create orchestrator
    orchestrator = ScalabilityTestOrchestrator(args.output_dir)
    
    if args.summary_only:
        # Generate summary report only
        orchestrator.generate_summary_report()
    else:
        # Run full scalability test
        results = orchestrator.run_full_scalability_test(
            db_counts=args.db_counts,
            runs_per_scale=args.runs_per_scale
        )
        
        # Generate summary report
        orchestrator.generate_summary_report(results)


if __name__ == "__main__":
    main()