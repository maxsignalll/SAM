"""
Enhanced metrics analyzer for detailed performance analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path
import json


class EnhancedMetricsAnalyzer:
    """Enhanced metrics analyzer for performance evaluation"""
    
    def __init__(self):
        self.metrics_cache = {}
        
    def analyze_strategy_results(self, data_file: str, strategy_id: str) -> Dict[str, Any]:
        """Analyze strategy results and calculate enhanced metrics"""
        # Read raw data
        df = pd.read_csv(data_file)
        
        # Filter warmup phase (first 30 seconds)
        df_main = df[df['elapsed_seconds'] >= 30].copy()
        if df_main.empty:
            df_main = df.copy()
        
        # Basic metrics
        basic_metrics = self._calculate_basic_metrics(df_main)
        
        # Percentile latency metrics
        percentile_metrics = self._calculate_percentile_latencies(df_main)
        
        # Dynamic adjustment metrics
        dynamic_metrics = self._calculate_dynamic_adjustment_metrics(df_main, strategy_id)
        
        # Fairness metrics
        fairness_metrics = self._calculate_detailed_fairness_metrics(df_main)
        
        # Merge all metrics
        all_metrics = {
            **basic_metrics,
            **percentile_metrics,
            **dynamic_metrics,
            **fairness_metrics,
            "strategy_id": strategy_id
        }
        
        return all_metrics
    
    def _calculate_basic_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate basic performance metrics"""
        total_operations = df['ops_count'].sum()
        
        # Weight by operation count
        weighted_latency_sum = (df['avg_latency_ms'] * df['ops_count']).sum()
        avg_latency = weighted_latency_sum / total_operations if total_operations > 0 else 0
        
        # Cache metrics
        total_hits = df['cache_hits'].sum()
        total_misses = df['cache_misses'].sum()
        cache_hit_rate = (total_hits / (total_hits + total_misses)) * 100 if (total_hits + total_misses) > 0 else 0
        
        # Throughput metrics
        total_time_seconds = df['elapsed_seconds'].max() - df['elapsed_seconds'].min()
        throughput_per_second = total_operations / total_time_seconds if total_time_seconds > 0 else 0
        
        # Stability metrics
        latency_std = df['avg_latency_ms'].std()
        latency_cv = latency_std / df['avg_latency_ms'].mean() if df['avg_latency_ms'].mean() > 0 else 0
        
        # Calculate per-database average latency
        avg_latency_per_db = df.groupby('db_instance').apply(
            lambda x: np.average(x['avg_latency_ms'], weights=x['ops_count']) if x['ops_count'].sum() > 0 else 0.0
        ).to_dict()

        # Calculate per-DB throughput
        ops_per_db = df.groupby('db_instance')['ops_count'].sum()
        db_durations = df.groupby('db_instance')['elapsed_seconds'].apply(lambda x: x.max() - x.min())
        # Avoid division by zero
        db_durations[db_durations == 0] = 1 
        throughput_per_db = (ops_per_db / db_durations).to_dict()

        # Calculate per-database cache hit rate
        df['total_accesses'] = df['cache_hits'] + df['cache_misses']
        
        db_hit_rates = df.groupby('db_instance').apply(
            lambda x: x['cache_hits'].sum() / x['total_accesses'].sum() if x['total_accesses'].sum() > 0 else 0
        ).to_dict()
        
        return {
            "total_operations": total_operations,
            "avg_latency": avg_latency,
            "latency_std": float(latency_std),
            "latency_cv": float(latency_cv),
            "cache_hit_rate": float(cache_hit_rate),
            "throughput_per_second": float(throughput_per_second),
            "avg_latency_per_db": avg_latency_per_db,
            "throughput_per_db": throughput_per_db,
            "hit_rate_per_db": db_hit_rates
        }
    
    def _calculate_percentile_latencies(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate percentile latencies"""
        # Check if percentile columns exist
        if 'p50_latency_ms' in df.columns:
            total_ops = df['ops_count'].sum()
            if total_ops == 0:
                return {
                    "latency_p50": 0.0,
                    "latency_p90": 0.0,
                    "latency_p95": 0.0,
                    "latency_p99": 0.0,
                    "latency_p999": 0.0
                }
            
            # For low percentiles, use weighted average
            weighted_p50 = (df['p50_latency_ms'] * df['ops_count']).sum() / total_ops
            weighted_p90 = (df['p90_latency_ms'] * df['ops_count']).sum() / total_ops
            
            # For high percentiles, use max value
            max_p95 = df['p95_latency_ms'].max()
            max_p99 = df['p99_latency_ms'].max()
            max_p999 = df['p999_latency_ms'].max()
            
            return {
                "latency_p50": float(weighted_p50),
                "latency_p90": float(weighted_p90),
                "latency_p95": float(max_p95),
                "latency_p99": float(max_p99),
                "latency_p999": float(max_p999)
            }
        else:
            # Fallback method for compatibility
            latencies = df['avg_latency_ms'].values
            
            # Expand latencies by operation count
            weighted_latencies = []
            for _, row in df.iterrows():
                # Repeat avg latency by ops count
                weighted_latencies.extend([row['avg_latency_ms']] * int(row['ops_count']))
            
            if not weighted_latencies:
                return {
                    "latency_p50": 0.0,
                    "latency_p90": 0.0,
                    "latency_p95": 0.0,
                    "latency_p99": 0.0,
                    "latency_p999": 0.0
                }
            
            weighted_latencies = np.array(weighted_latencies)
            
            return {
                "latency_p50": float(np.percentile(weighted_latencies, 50)),
                "latency_p90": float(np.percentile(weighted_latencies, 90)),
                "latency_p95": float(np.percentile(weighted_latencies, 95)),
                "latency_p99": float(np.percentile(weighted_latencies, 99)),
                "latency_p999": float(np.percentile(weighted_latencies, 99.9))
            }
    
    def _calculate_dynamic_adjustment_metrics(self, df: pd.DataFrame, strategy_id: str) -> Dict[str, Any]:
        """Calculate dynamic adjustment metrics for elastic strategies"""
        # Only calculate for elastic strategies
        if strategy_id not in ["S0", "B3"]:
            return {
                "cache_adjustments_count": 0,
                "avg_pages_transferred_per_adjustment": 0.0,
                "max_pages_transferred": 0,
                "cache_size_volatility": 0.0,
                "adjustment_effectiveness": 0.0
            }
        
        # Analyze cache size changes by database
        cache_changes = {}
        adjustment_points = []
        
        for db_id in df['db_instance'].unique():
            db_data = df[df['db_instance'] == db_id].sort_values('elapsed_seconds')
            cache_sizes = db_data['current_cache_pages'].values # Use page count directly
            
            # Detect cache size changes
            cache_diffs = np.diff(cache_sizes)
            significant_changes = np.where(np.abs(cache_diffs) > 10)[0]  # Changes over 10 pages considered as adjustment
            
            cache_changes[db_id] = {
                "changes": len(significant_changes),
                "total_pages_transferred": np.sum(np.abs(cache_diffs[significant_changes])),
                "cache_size_std": np.std(cache_sizes)
            }
            
            # Record adjustment time points
            if len(significant_changes) > 0:
                adjustment_times = db_data.iloc[significant_changes + 1]['elapsed_seconds'].values
                adjustment_points.extend(adjustment_times)
        
        # Calculate overall metrics
        total_adjustments = sum(stats["changes"] for stats in cache_changes.values())
        total_pages_transferred = sum(stats["total_pages_transferred"] for stats in cache_changes.values())
        total_adjustments = sum(stats["changes"] for stats in cache_changes.values())
        total_pages_transferred = sum(stats["total_pages_transferred"] for stats in cache_changes.values())
        
        # Cache size volatility
        cache_volatility = np.mean([stats["cache_size_std"] for stats in cache_changes.values()])
        
        # Adjustment effectiveness
        adjustment_effectiveness = self._calculate_adjustment_effectiveness(df, adjustment_points)
        
        return {
            "cache_adjustments_count": int(total_adjustments),
            "avg_pages_transferred_per_adjustment": float(total_pages_transferred / total_adjustments) if total_adjustments > 0 else 0.0,
            "max_pages_transferred": float(max(stats["total_pages_transferred"] for stats in cache_changes.values())) if cache_changes else 0.0,
            "cache_size_volatility": float(cache_volatility),
            "adjustment_effectiveness": float(adjustment_effectiveness),
            "per_db_adjustment_stats": cache_changes
        }
    
    def _calculate_adjustment_effectiveness(self, df: pd.DataFrame, adjustment_points: List[float]) -> float:
        """
        Calculate adjustment effectiveness: compare performance before and after adjustment
        """
        if not adjustment_points:
            return 0.0
        
        effectiveness_scores = []
        
        for adj_time in adjustment_points[:10]:  # Analyze at most first 10 adjustments
            # Performance 5s before
            before_data = df[(df['elapsed_seconds'] >= adj_time - 5) & 
                           (df['elapsed_seconds'] < adj_time)]
            
            # Performance 5s after
            after_data = df[(df['elapsed_seconds'] > adj_time) & 
                          (df['elapsed_seconds'] <= adj_time + 5)]
            
            if not before_data.empty and not after_data.empty:
                # Compare cache hit rates
                before_hit_rate = before_data['cache_hits'].sum() / (
                    before_data['cache_hits'].sum() + before_data['cache_misses'].sum())
                after_hit_rate = after_data['cache_hits'].sum() / (
                    after_data['cache_hits'].sum() + after_data['cache_misses'].sum())
                
                # Compare latencies
                before_latency = before_data['avg_latency_ms'].mean()
                after_latency = after_data['avg_latency_ms'].mean()
                
                # Combined score: hit rate improvement + latency reduction
                hit_rate_improvement = (after_hit_rate - before_hit_rate) / before_hit_rate if before_hit_rate > 0 else 0
                latency_improvement = (before_latency - after_latency) / before_latency if before_latency > 0 else 0
                
                effectiveness = (hit_rate_improvement + latency_improvement) / 2
                effectiveness_scores.append(effectiveness)
        
        return np.mean(effectiveness_scores) if effectiveness_scores else 0.0
    
    def _calculate_detailed_fairness_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate detailed fairness metrics"""
        # Group statistics by database
        db_stats = {}
        
        for db_id in df['db_instance'].unique():
            db_data = df[df['db_instance'] == db_id]
            
            # Calculate per-database metrics
            total_ops = db_data['ops_count'].sum()
            avg_latency = (db_data['ops_count'] * db_data['avg_latency_ms']).sum() / total_ops if total_ops > 0 else 0
            total_hits = db_data['cache_hits'].sum()
            total_misses = db_data['cache_misses'].sum()
            hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0
            avg_cache_pages = db_data['current_cache_pages'].mean()
            # Assume 4KB page size
            avg_cache_mb = avg_cache_pages * 4 / 1024
            
            db_stats[db_id] = {
                "total_operations": int(total_ops),
                "avg_latency": float(avg_latency),
                "cache_hit_rate": float(hit_rate),
                "avg_cache_allocation_mb": float(avg_cache_mb),
                "throughput_share": 0.0  # Calculated later
            }
        
        # Calculate throughput share
        total_system_ops = sum(stats["total_operations"] for stats in db_stats.values())
        for db_id, stats in db_stats.items():
            stats["throughput_share"] = stats["total_operations"] / total_system_ops if total_system_ops > 0 else 0
        
        # Calculate Jain's Fairness Index
        throughputs = [stats["total_operations"] for stats in db_stats.values()]
        jains_index = self._calculate_jains_fairness_index(throughputs)
        
        # Calculate resource efficiency
        resource_efficiency = self._calculate_resource_efficiency(db_stats)
        
        # Calculate priority satisfaction
        priority_satisfaction = self._calculate_priority_satisfaction(db_stats, df)
        
        return {
            "fairness_index": float(jains_index),
            "resource_efficiency": float(resource_efficiency),
            "priority_satisfaction": float(priority_satisfaction),
            "per_database_stats": db_stats,
            "throughput_variance": float(np.var(throughputs)),
            "latency_variance": float(np.var([stats["avg_latency"] for stats in db_stats.values()]))
        }
    
    def _calculate_jains_fairness_index(self, values: List[float]) -> float:
        """Calculate Jain's fairness index"""
        if not values or len(values) <= 1:
            return 1.0
        
        n = len(values)
        sum_x = sum(values)
        sum_x_squared = sum(x**2 for x in values)
        
        if sum_x_squared == 0:
            return 1.0
        
        return (sum_x ** 2) / (n * sum_x_squared)
    
    def _calculate_resource_efficiency(self, db_stats: Dict[str, Dict]) -> float:
        """Calculate resource efficiency = hit_rate * throughput / cache"""
        total_hits = sum(stats["cache_hit_rate"] * stats["total_operations"] for stats in db_stats.values())
        total_ops = sum(stats["total_operations"] for stats in db_stats.values())
        total_cache = sum(stats["avg_cache_allocation_mb"] for stats in db_stats.values())
        
        if total_ops == 0 or total_cache == 0:
            return 0.0
        
        avg_hit_rate = total_hits / total_ops
        ops_per_mb = total_ops / total_cache
        
        # Normalize to 0-1 range
        efficiency = avg_hit_rate * min(ops_per_mb / 1000, 1.0)  # 1000 ops/MB as ideal
        
        return efficiency
    
    def _calculate_priority_satisfaction(self, db_stats: Dict[str, Dict], df: pd.DataFrame) -> float:
        """Calculate priority satisfaction based on resource allocation and performance"""
        # Infer priority from database ID
        priority_map = {
            "db_high_priority": 10,
            "db_medium_priority": 5,
            "db_low_priority": 1,
            "db_unified": 5  # B5 strategy unified DB
        }
        
        satisfaction_scores = []
        
        for db_id, stats in db_stats.items():
            priority = priority_map.get(db_id, 5)  # Default medium priority
            
            # Calculate relative performance score
            perf_score = (stats["cache_hit_rate"] * 
                         (1 / (stats["avg_latency"] + 1)) * 
                         stats["throughput_share"])
            
            # Expected score based on priority
            total_priority = sum(priority_map.get(db, 5) for db in db_stats.keys())
            expected_score = priority / total_priority
            
            # Satisfaction = actual / expected
            satisfaction = min(perf_score / expected_score, 2.0) if expected_score > 0 else 1.0
            satisfaction_scores.append(satisfaction)
        
        return np.mean(satisfaction_scores) if satisfaction_scores else 1.0
    
    def generate_enhanced_report(self, all_metrics: List[Dict[str, Any]], output_file: str):
        """Generate enhanced comparison report grouped by strategy and database"""
        
        report_data = []
        for metrics in all_metrics:
            strategy_id = metrics.get('strategy_id', 'Unknown')
            
            # Process per-database detailed metrics
            for db_id in metrics.get("avg_latency_per_db", {}).keys():
                report_data.append({
                    "Strategy": strategy_id,
                    "Database": db_id,
                    "Avg Latency (ms)": metrics["avg_latency_per_db"].get(db_id, 0),
                    "P99 Latency (ms)": metrics.get("latency_p99", 0), # Note: P99 is strategy-level
                    "Throughput (ops/sec)": metrics["throughput_per_db"].get(db_id, 0),
                    "Cache Hit Rate (%)": metrics["hit_rate_per_db"].get(db_id, 0) * 100,
                })
            
            # Add strategy overall data as special row
            report_data.append({
                "Strategy": strategy_id,
                "Database": "--- OVERALL ---",
                "Avg Latency (ms)": metrics.get("avg_latency", 0),
                "P99 Latency (ms)": metrics.get("latency_p99", 0),
                "Throughput (ops/sec)": metrics.get("throughput_per_second", 0),
                "Cache Hit Rate (%)": metrics.get("cache_hit_rate", 0),
            })

        if not report_data:
            print("No data available for report generation.")
            return

        # Create and beautify DataFrame
        report_df = pd.DataFrame(report_data)
        report_df = report_df.round(2)
        
        # Use multi-level index for grouping
        report_df.set_index(['Strategy', 'Database'], inplace=True)
        report_df.sort_index(inplace=True)

        # Print to console
        print("\n" + "="*20 + " Strategy Performance Comparison (Per-Database) " + "="*20)
        print(report_df.to_string())
        print("="*80)

        # Save to CSV file
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            report_df.to_csv(output_path)
            print(f"✅ Detailed comparison saved to: {output_path}")
        except Exception as e:
            print(f"❌ Failed to save summary to {output_file}: {e}")


def create_enhanced_analyzer() -> EnhancedMetricsAnalyzer:
    """Create enhanced metrics analyzer instance"""
    return EnhancedMetricsAnalyzer() 