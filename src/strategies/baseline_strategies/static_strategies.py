import os
from typing import Any, Dict
from ..base_strategy import BaseStrategy


def _get_min_pages(strategy_config: dict, main_config: dict) -> int:
    """Helper to get the minimum pages, falling back from strategy to main config."""
    # Prioritize getting from strategy-specific configuration
    min_mb = strategy_config.get("min_cache_per_db_mb")
    if min_mb is not None:
        page_size_bytes = main_config.get("system_sqlite_config", {}).get("page_size_bytes", 4096)
        return (min_mb * 1024 * 1024) // page_size_bytes
    
    # If not in strategy config, fallback to main config
    return main_config.get("cache_allocation_config", {}).get("min_pages_per_db", 10)


class B1_StaticAverageStrategy(BaseStrategy):
    """
    B1: Static Average Baseline Strategy
    At experiment start, equally distribute total available RAM among all database instances.
    This is a very simple, static strategy used to measure relative benefits of more complex strategies.
    - Initial phase: Equally distribute cache.
    - Dynamic phase: Maintain static cache size.
    - End phase: Maintain static cache size.
    """
    is_dynamic = False
    def __init__(self, orchestrator: Any, strategy_name: str, strategy_specific_config: Dict = None):
        super().__init__(orchestrator, strategy_name, strategy_specific_config)

    def calculate_initial_allocations(self):
        """Calculates an equal allocation for all database instances."""
        num_databases = len(self.db_instance_configs)
        if num_databases == 0:
            return {}
            
        pages_per_db = self.total_pages // num_databases
        min_pages_per_db = _get_min_pages(self.strategy_specific_config, self.orchestrator.config)
        
        allocations = {}
        for db_conf in self.db_instance_configs:
            db_id = db_conf["id"]
            allocations[db_id] = max(min_pages_per_db, pages_per_db)
            
        self.logger.info(f"B1 Static Average Allocation: {allocations}")
        return allocations

    def cleanup(self):
        """B1 strategy has no resources that need explicit cleanup."""
        self.logger.debug("B1_StaticAverageStrategy cleanup called.")
        pass


class B2_NoElasticFixedByPriorityStrategy(BaseStrategy):
    """
    Baseline B2 (Ablation): Fixed pool only, priority-based allocation
    This strategy tests the effect of "fixed pool". It treats all available cache as fixed pool and performs one-time static allocation based on database base_priority.
    It has no elastic pool and makes no runtime adjustments.
    """
    is_dynamic = False
    def __init__(self, orchestrator: Any, strategy_name: str, strategy_specific_config: Dict = None):
        super().__init__(orchestrator, strategy_name, strategy_specific_config)

    def calculate_initial_allocations(self):
        o = self.orchestrator
        total_allocs = {}
        num_databases = len(self.db_instance_configs)
        min_pages_per_db = _get_min_pages(self.strategy_specific_config, o.config)
        
        priority_sum = sum(db.get("base_priority", 1) for db in self.db_instance_configs if db.get("base_priority", 1) > 0)
        use_equal_fallback = priority_sum == 0
        
        for db_conf in self.db_instance_configs:
            db_id = db_conf["id"]
            priority = db_conf.get("base_priority", 1)
            
            if use_equal_fallback or priority <= 0:
                pages_for_db = self.total_pages // num_databases if num_databases > 0 else min_pages_per_db
            else:
                pages_for_db = int((priority / priority_sum) * self.total_pages)
            
            final_pages = max(min_pages_per_db, pages_for_db)
            total_allocs[db_id] = final_pages
            
        self.logger.info(f"B2 Fixed-by-Priority Allocation: {total_allocs}")
        return total_allocs

    def cleanup(self):
        """B2 strategy has no resources that need explicit cleanup."""
        self.logger.debug("B2_NoElasticFixedByPriorityStrategy cleanup called.")
        pass


class B6_DataSizeProportionalStaticStrategy(BaseStrategy):
    """
    Baseline B6: Static allocation proportional to data size
    This strategy statically allocates total cache one-time based on each database's initial data size (record count).
    Larger databases get more cache.
    """
    is_dynamic = False
    def __init__(self, orchestrator: Any, strategy_name: str, strategy_specific_config: Dict = None):
        super().__init__(orchestrator, strategy_name, strategy_specific_config)

    def calculate_initial_allocations(self):
        o = self.orchestrator
        allocations = {}
        db_file_sizes = {}
        total_db_file_size = 0
        num_databases = len(self.db_instance_configs)
        if num_databases == 0: return {}

        total_pages = self.total_pages
        min_pages_per_db = _get_min_pages(self.strategy_specific_config, o.config)
        
        for db_conf in self.db_instance_configs:
            db_id = db_conf["id"]
            db_filename = db_conf["db_filename"]
            size_bytes = 1 # Nominal size to avoid division by zero
            if os.path.exists(db_filename):
                size_bytes = max(1, os.path.getsize(db_filename))
            else:
                self.logger.error(f"DB file not found for B6 size calculation: {db_filename}")
            db_file_sizes[db_id] = size_bytes
            total_db_file_size += size_bytes
        
        if total_db_file_size <= num_databases: # All files were missing or empty
             self.logger.warning("All DB files missing/empty for B6. Falling back to equal distribution.")
             return B1_StaticAverageStrategy(o, "B1_StaticAverageStrategy").calculate_initial_allocations()

        # Proportional allocation
        for db_id, size in db_file_sizes.items():
            share = size / total_db_file_size
            pages = int(share * total_pages)
            allocations[db_id] = max(min_pages_per_db, pages)

        # Note: This simplified allocation may not sum up perfectly to total_pages.
        # The original implementation had more complex logic to distribute remainder pages.
        # For this refactoring, this is a reasonable simplification.
        
        self.logger.info(f"B6 (DataSizeProportional): TotalRAMPgs={total_pages}, FinalAllocations={allocations}")
        o.db_fixed_page_allocations = allocations.copy()
        return allocations 

    def cleanup(self):
        """B6 strategy has no resources that need explicit cleanup."""
        pass