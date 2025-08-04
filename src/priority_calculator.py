import math
import time
from typing import Dict, Any, Optional, Tuple

class PriorityCalculator:
    """
    Dynamic priority calculator
    Implements base priority + dynamic adjustment factors
    """
    
    def __init__(self, config: dict):
        """Initialize priority calculator"""
        self.config = config.get("priority_calculation_config", {})
        
        # Base priority settings
        self.base_priority_scheme = self.config.get("base_priority_scheme", "large_interval")
        self.priority_mappings = {
            "large_interval": {"P0": 1000, "P1": 500, "P2": 100},
            "small_interval": {"P0": 40, "P1": 30, "P2": 20}
        }
        
        # Dynamic adjustment weights
        self.dynamic_weights = self.config.get("dynamic_adjustment_weights", {
            "transaction_count_weight": 0.2,
            "io_wait_weight": 0.3,
            "cache_hit_gap_weight": 0.3,
            "response_time_weight": 0.2
        })
        
        # Normalization parameters
        self.normalization_params = self.config.get("normalization_params", {
            "max_transaction_count_per_interval": 1000,
            "max_io_wait_ms": 50.0,
            "max_response_time_ms": 100.0,
            "reference_hit_rate": 0.8
        })
        
        # Priority bounds
        self.priority_bounds = self.config.get("priority_bounds", {
            "min_final_priority": 10,
            "max_final_priority": 2000
        })
        
        print(f"[PriorityCalculator] Initialized with scheme: {self.base_priority_scheme}")
    
    def get_base_priority_score(self, priority_level: str) -> int:
        """Get base priority score for level (P0/P1/P2)"""
        mapping = self.priority_mappings[self.base_priority_scheme]
        return mapping.get(priority_level, mapping["P2"])  # Default to lowest priority
    
    def calculate_transaction_factor(self, transaction_count: int) -> float:
        """Calculate transaction count factor (0-1)"""
        max_count = self.normalization_params["max_transaction_count_per_interval"]
        normalized = min(transaction_count / max_count, 1.0)
        return normalized
    
    def calculate_io_wait_factor(self, avg_io_wait_ms: float) -> float:
        """Calculate I/O wait factor (0-1, higher means more cache needed)"""
        max_wait = self.normalization_params["max_io_wait_ms"]
        normalized = min(avg_io_wait_ms / max_wait, 1.0)
        return normalized
    
    def calculate_cache_hit_gap_factor(self, current_hit_rate: float, 
                                     target_hit_rate: Optional[float] = None) -> float:
        """Calculate cache hit rate gap factor (0-1)"""
        if target_hit_rate is None:
            target_hit_rate = self.normalization_params["reference_hit_rate"]
        
        # Calculate gap from target
        gap = max(0, target_hit_rate - current_hit_rate)
        # Normalize to 0-1 range
        max_possible_gap = target_hit_rate
        normalized_gap = min(gap / max_possible_gap, 1.0) if max_possible_gap > 0 else 0.0
        
        return normalized_gap
    
    def calculate_response_time_factor(self, avg_response_time_ms: float) -> float:
        """Calculate response time factor (0-1, higher = worse performance)"""
        max_response_time = self.normalization_params["max_response_time_ms"]
        normalized = min(avg_response_time_ms / max_response_time, 1.0)
        return normalized
    
    def calculate_dynamic_adjustment_score(self, runtime_stats: Dict[str, Any]) -> float:
        """Calculate dynamic adjustment score (weighted sum)"""
        # Extract stats
        transaction_count = runtime_stats.get("transaction_count", 0)
        avg_io_wait = runtime_stats.get("avg_io_wait_ms", 0.0)
        cache_hit_rate = runtime_stats.get("cache_hit_rate", 0.0)
        avg_response_time = runtime_stats.get("avg_response_time_ms", 0.0)
        
        # Calculate factors
        transaction_factor = self.calculate_transaction_factor(transaction_count)
        io_wait_factor = self.calculate_io_wait_factor(avg_io_wait)
        cache_gap_factor = self.calculate_cache_hit_gap_factor(cache_hit_rate)
        response_time_factor = self.calculate_response_time_factor(avg_response_time)
        
        # Weighted sum
        weighted_sum = (
            transaction_factor * self.dynamic_weights["transaction_count_weight"] +
            io_wait_factor * self.dynamic_weights["io_wait_weight"] +
            cache_gap_factor * self.dynamic_weights["cache_hit_gap_weight"] +
            response_time_factor * self.dynamic_weights["response_time_weight"]
        )
        
        return weighted_sum
    
    def calculate_final_priority(self, base_priority_level: str, 
                               runtime_stats: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Calculate final priority score and breakdown"""
        # Get base score
        base_score = self.get_base_priority_score(base_priority_level)
        
        # Calculate dynamic score
        dynamic_score = self.calculate_dynamic_adjustment_score(runtime_stats)
        
        # Convert dynamic score to same scale as base score
        # Max dynamic adjustment is 50% of base score
        max_dynamic_adjustment = base_score * 0.5
        dynamic_adjustment = dynamic_score * max_dynamic_adjustment
        
        # Calculate final score
        final_score = base_score + dynamic_adjustment
        
        # Apply bounds
        final_score = max(self.priority_bounds["min_final_priority"], 
                         min(final_score, self.priority_bounds["max_final_priority"]))
        
        # Create breakdown
        breakdown = {
            "base_score": base_score,
            "dynamic_adjustment": dynamic_adjustment,
            "dynamic_factors": {
                "transaction_factor": self.calculate_transaction_factor(
                    runtime_stats.get("transaction_count", 0)),
                "io_wait_factor": self.calculate_io_wait_factor(
                    runtime_stats.get("avg_io_wait_ms", 0.0)),
                "cache_gap_factor": self.calculate_cache_hit_gap_factor(
                    runtime_stats.get("cache_hit_rate", 0.0)),
                "response_time_factor": self.calculate_response_time_factor(
                    runtime_stats.get("avg_response_time_ms", 0.0))
            },
            "final_score": final_score
        }
        
        return final_score, breakdown
    
    def get_priority_level_from_base_priority(self, base_priority: int) -> str:
        """Convert numeric base priority to level string (P0/P1/P2)"""
        if base_priority >= 8:
            return "P0"  # High priority
        elif base_priority >= 4:
            return "P1"  # Medium priority
        else:
            return "P2"  # Low priority
    
    def calculate_priority_for_database(self, db_config: dict, 
                                      runtime_stats: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Calculate priority for specific database"""
        base_priority_num = db_config.get("base_priority", 1)
        priority_level = self.get_priority_level_from_base_priority(base_priority_num)
        
        final_score, breakdown = self.calculate_final_priority(priority_level, runtime_stats)
        
        # Create detailed report
        report = {
            "db_id": db_config.get("id", "unknown"),
            "base_priority_num": base_priority_num,
            "priority_level": priority_level,
            "final_priority_score": final_score,
            "calculation_breakdown": breakdown,
            "timestamp": time.time()
        }
        
        return final_score, report


def create_priority_calculator(config: dict) -> PriorityCalculator:
    """Factory function to create priority calculator"""
    return PriorityCalculator(config) 