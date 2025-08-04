import math
import os
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

@dataclass
class DatabaseProfile:
    """Database profile"""
    id: str
    base_priority: int
    data_size_mb: float
    estimated_working_set_mb: float
    min_cache_requirement_mb: float
    access_pattern: str  # "sequential", "random", "mixed"
    
@dataclass
class AllocationResult:
    """Allocation result"""
    db_id: str
    fixed_quota_pages: int
    fixed_quota_mb: float
    weight_breakdown: Dict[str, float]
    allocation_reasoning: str

class CacheAllocator:
    """
    Cache allocator using weight-based fixed quota allocation
    """
    
    def __init__(self, config: dict):
        """Initialize cache allocator"""
        self.config = config
        self.system_cfg = config.get("system_sqlite_config", {})
        self.allocation_cfg = config.get("cache_allocation_config", {})
        
        self.page_size_bytes = self.system_cfg.get("page_size_bytes", 4096)
        self.total_ram_mb = self.system_cfg.get("total_simulated_ram_mb_for_dbs", 16)
        self.total_pages = (self.total_ram_mb * 1024 * 1024) // self.page_size_bytes
        
        strategy_cfg = config.get("priority_fixed_elastic_strategy_config", {})
        self.fixed_pool_percentage = strategy_cfg.get("fixed_pool_percentage_of_total_ram", 0.6)
        self.fixed_pool_pages = int(self.total_pages * self.fixed_pool_percentage)
        
        self.weight_params = self.allocation_cfg.get("weight_parameters", {
            "priority_weight_alpha": 0.5,
            "data_size_weight_beta": 0.3,
            "min_cache_weight_gamma": 0.2
        })
        
        elastic_cfg = strategy_cfg.get("elastic_pool_manager_config", {})
        self.min_pages_per_db = elastic_cfg.get("min_db_cache_pages_absolute", 50)
        self.min_mb_per_db = (self.min_pages_per_db * self.page_size_bytes) / (1024 * 1024)
        
        print(f"[CacheAllocator] Initialized")
        print(f"  Total memory: {self.total_ram_mb}MB ({self.total_pages} pages)")
        print(f"  Fixed pool: {self.fixed_pool_pages} pages ({self.fixed_pool_pages * self.page_size_bytes / 1024 / 1024:.1f}MB)")
    
    def analyze_database_profile(self, db_config: dict) -> DatabaseProfile:
        """Analyze database configuration and create profile"""
        db_id = db_config["id"]
        base_priority = db_config.get("base_priority", 1)
        
        db_filename = db_config.get("db_filename", "")
        data_size_mb = 0.0
        if os.path.exists(db_filename):
            data_size_mb = os.path.getsize(db_filename) / (1024 * 1024)
        
        record_count = db_config.get("ycsb_initial_record_count", 500000)
        estimated_working_set_mb = self._estimate_working_set_size(record_count, data_size_mb)
        
        min_cache_requirement_mb = max(
            self.min_mb_per_db,
            estimated_working_set_mb * 0.1
        )
        
        access_pattern = self._infer_access_pattern(db_config)
        
        return DatabaseProfile(
            id=db_id,
            base_priority=base_priority,
            data_size_mb=data_size_mb,
            estimated_working_set_mb=estimated_working_set_mb,
            min_cache_requirement_mb=min_cache_requirement_mb,
            access_pattern=access_pattern
        )
    
    def _estimate_working_set_size(self, record_count: int, total_data_size_mb: float) -> float:
        """Estimate working set size based on 80-20 rule"""
        # 80-20 rule: assume 80% of accesses target 20% of data
        pareto_factor = 0.2
        
        avg_record_size = (total_data_size_mb * 1024 * 1024) / record_count if record_count > 0 else 1024
        hot_records = int(record_count * pareto_factor)
        working_set_mb = (hot_records * avg_record_size) / (1024 * 1024)
        
        working_set_mb = max(1.0, min(working_set_mb, total_data_size_mb))
        
        return working_set_mb
    
    def _infer_access_pattern(self, db_config: dict) -> str:
        """Infer access pattern from database configuration"""
        db_id = db_config.get("id", "").lower()
        
        if "oltp" in db_id or "high" in db_id:
            return "random"
        elif "analytics" in db_id or "scan" in db_id:
            return "sequential"
        else:
            return "mixed"
    
    def calculate_priority_factor(self, profile: DatabaseProfile, all_profiles: List[DatabaseProfile]) -> float:
        """Calculate normalized priority factor (0-1)"""
        all_priorities = [p.base_priority for p in all_profiles]
        max_priority = max(all_priorities) if all_priorities else 1
        min_priority = min(all_priorities) if all_priorities else 1
        
        if max_priority == min_priority:
            return 1.0 / len(all_profiles)
        
        normalized = (profile.base_priority - min_priority) / (max_priority - min_priority)
        
        return normalized
    
    def calculate_data_size_factor(self, profile: DatabaseProfile, all_profiles: List[DatabaseProfile]) -> float:
        """Calculate normalized data size factor (0-1)"""
        all_sizes = [p.estimated_working_set_mb for p in all_profiles]
        total_size = sum(all_sizes)
        
        if total_size == 0:
            return 1.0 / len(all_profiles)
        
        size_ratio = profile.estimated_working_set_mb / total_size
        
        return size_ratio
    
    def calculate_min_cache_factor(self, profile: DatabaseProfile, all_profiles: List[DatabaseProfile]) -> float:
        """Calculate normalized minimum cache factor (0-1)"""
        all_min_requirements = [p.min_cache_requirement_mb for p in all_profiles]
        total_min_requirement = sum(all_min_requirements)
        
        if total_min_requirement == 0:
            return 1.0 / len(all_profiles)
        
        min_ratio = profile.min_cache_requirement_mb / total_min_requirement
        
        return min_ratio
    
    def calculate_composite_weight(self, profile: DatabaseProfile, all_profiles: List[DatabaseProfile]) -> Tuple[float, Dict[str, float]]:
        """Calculate composite weight and breakdown"""
        priority_factor = self.calculate_priority_factor(profile, all_profiles)
        data_size_factor = self.calculate_data_size_factor(profile, all_profiles)
        min_cache_factor = self.calculate_min_cache_factor(profile, all_profiles)
        
        alpha = self.weight_params["priority_weight_alpha"]
        beta = self.weight_params["data_size_weight_beta"]
        gamma = self.weight_params["min_cache_weight_gamma"]
        
        composite_weight = (
            priority_factor * alpha +
            data_size_factor * beta +
            min_cache_factor * gamma
        )
        
        breakdown = {
            "priority_factor": priority_factor,
            "data_size_factor": data_size_factor,
            "min_cache_factor": min_cache_factor,
            "priority_contribution": priority_factor * alpha,
            "data_size_contribution": data_size_factor * beta,
            "min_cache_contribution": min_cache_factor * gamma,
            "composite_weight": composite_weight
        }
        
        return composite_weight, breakdown
    
    def allocate_fixed_quotas(self, db_configs: List[dict]) -> Dict[str, AllocationResult]:
        """Allocate fixed quotas to databases"""
        print(f"[CacheAllocator] Starting fixed quota allocation, target pool: {self.fixed_pool_pages} pages")
        
        profiles = [self.analyze_database_profile(config) for config in db_configs]
        
        weights = {}
        weight_breakdowns = {}
        total_weight = 0.0
        
        for profile in profiles:
            weight, breakdown = self.calculate_composite_weight(profile, profiles)
            weights[profile.id] = weight
            weight_breakdowns[profile.id] = breakdown
            total_weight += weight
        
        print(f"[CacheAllocator] Weight calculation completed")
        
        results = {}
        allocated_pages_sum = 0
        
        for profile in profiles:
            db_id = profile.id
            
            if total_weight > 0:
                proportional_pages = int((weights[db_id] / total_weight) * self.fixed_pool_pages)
            else:
                proportional_pages = self.fixed_pool_pages // len(profiles)
            
            min_pages_required = max(
                self.min_pages_per_db,
                int((profile.min_cache_requirement_mb * 1024 * 1024) / self.page_size_bytes)
            )
            
            final_pages = max(proportional_pages, min_pages_required)
            final_mb = (final_pages * self.page_size_bytes) / (1024 * 1024)
            
            reasoning = self._create_allocation_reasoning(
                profile, proportional_pages, min_pages_required, final_pages, weight_breakdowns[db_id]
            )
            
            results[db_id] = AllocationResult(
                db_id=db_id,
                fixed_quota_pages=final_pages,
                fixed_quota_mb=final_mb,
                weight_breakdown=weight_breakdowns[db_id],
                allocation_reasoning=reasoning
            )
            
            allocated_pages_sum += final_pages
            
            print(f"[CacheAllocator] {db_id}: {final_pages} pages ({final_mb:.1f}MB)")
        
        if allocated_pages_sum > self.fixed_pool_pages:
            print(f"[CacheAllocator] Warning: Total allocation exceeds budget")
            results = self._rebalance_allocations(results, self.fixed_pool_pages)
        
        print(f"[CacheAllocator] Fixed quota allocation completed")
        
        return results
    
    def _create_allocation_reasoning(self, profile: DatabaseProfile, proportional_pages: int, 
                                   min_pages: int, final_pages: int, 
                                   weight_breakdown: Dict[str, float]) -> str:
        """Create allocation reasoning"""
        reasoning_parts = []
        
        reasoning_parts.append(f"Database '{profile.id}' allocation:")
        reasoning_parts.append(f"  Priority: {profile.base_priority}")
        reasoning_parts.append(f"  Working set: {profile.estimated_working_set_mb:.1f}MB")
        reasoning_parts.append(f"  Min requirement: {profile.min_cache_requirement_mb:.1f}MB")
        reasoning_parts.append(f"  Weight: {weight_breakdown['composite_weight']:.4f}")
        reasoning_parts.append(f"  Final: {final_pages} pages")
        
        if final_pages > proportional_pages:
            reasoning_parts.append(f"  Adjustment: Meet minimum requirement")
        
        return " | ".join(reasoning_parts)
    
    def _rebalance_allocations(self, results: Dict[str, AllocationResult], 
                             target_pages: int) -> Dict[str, AllocationResult]:
        """Rebalance allocations to meet budget constraints"""
        print(f"[CacheAllocator] Starting rebalance")
        
        current_total = sum(r.fixed_quota_pages for r in results.values())
        excess_pages = current_total - target_pages
        
        if excess_pages <= 0:
            return results
        
        sorted_results = sorted(results.items(), 
                              key=lambda x: x[1].weight_breakdown['priority_factor'])
        
        for db_id, result in sorted_results:
            if excess_pages <= 0:
                break
            
            min_pages = self.min_pages_per_db
            max_reduction = max(0, result.fixed_quota_pages - min_pages)
            
            reduction = min(excess_pages, max_reduction)
            
            if reduction > 0:
                new_pages = result.fixed_quota_pages - reduction
                new_mb = (new_pages * self.page_size_bytes) / (1024 * 1024)
                
                result.fixed_quota_pages = new_pages
                result.fixed_quota_mb = new_mb
                result.allocation_reasoning += f" | Budget adjustment: -{reduction} pages"
                
                excess_pages -= reduction
                print(f"[CacheAllocator] Adjust {db_id}: -{reduction} pages")
        
        if excess_pages > 0:
            print(f"[CacheAllocator] Warning: Still {excess_pages} pages over budget")
        
        return results
    
    def get_allocation_summary(self, results: Dict[str, AllocationResult]) -> str:
        """Generate allocation summary report"""
        lines = []
        lines.append("=== Fixed Quota Allocation Summary ===")
        lines.append(f"Total memory pool: {self.fixed_pool_pages} pages ({self.fixed_pool_pages * self.page_size_bytes / 1024 / 1024:.1f}MB)")
        
        total_allocated_pages = sum(r.fixed_quota_pages for r in results.values())
        total_allocated_mb = sum(r.fixed_quota_mb for r in results.values())
        
        lines.append(f"Total allocated: {total_allocated_pages} pages ({total_allocated_mb:.1f}MB)")
        lines.append(f"Utilization: {(total_allocated_pages / self.fixed_pool_pages) * 100:.1f}%")
        lines.append("")
        
        for db_id, result in sorted(results.items()):
            lines.append(f"Database: {db_id}")
            lines.append(f"  Allocation: {result.fixed_quota_pages} pages ({result.fixed_quota_mb:.1f}MB)")
            lines.append(f"  Weight: {result.weight_breakdown['composite_weight']:.4f}")
            lines.append("")
        
        return "\n".join(lines)


def create_cache_allocator(config: dict) -> CacheAllocator:
    """Factory function to create cache allocator instance"""
    return CacheAllocator(config) 