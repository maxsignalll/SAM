import random
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


class B4_IndividualOptimizedCacheStrategy(BaseStrategy):
    """
    Baseline B4 (Anarchic): Greedy requests with random ordering
    This is a "selfish" or "anarchic" baseline strategy. It simulates the following scenario:
    - In each adjustment period, all database instances are randomly ordered.
    - Each instance calculates demand based on miss rate and access frequency (demand = miss_rate * sqrt(total_accesses)).
    - Following random order, system satisfies these demands until resource pool is exhausted.
    - Remaining cache is allocated weighted by miss rate, not simple average.
    This model tests a resource allocation scenario without central coordination, where order and opportunity play decisive roles.
    
    Improvements:
    1. Demand calculation considers miss rate and access frequency, not just cache_misses count
    2. Remaining cache allocated by miss rate weighting, avoiding homogenization from simple averaging
    """
    is_dynamic = True
    def __init__(self, orchestrator: Any, strategy_name: str, strategy_specific_config: Dict = None):
        """Initializes the B4 strategy."""
        super().__init__(orchestrator, strategy_name, strategy_specific_config)
        self.random = random.Random()
        self.logger.info(f"{self.strategy_name} initialized.")

    def cleanup(self):
        """Cleans up resources used by the B4 strategy."""
        self.logger.info("Cleaning up B4_IndividualOptimizedCacheStrategy.")

    def calculate_initial_allocations(self):
        """Initially, allocates cache equally among all DBs. This provides a fair starting point."""
        allocations = {}
        num_databases = len(self.db_instance_configs)
        if num_databases == 0:
            return {}

        pages_per_db = self.total_pages // num_databases
        min_pages_per_db = _get_min_pages(self.strategy_specific_config, self.orchestrator.config)

        for db_conf in self.db_instance_configs:
            db_id = db_conf["id"]
            allocations[db_id] = max(min_pages_per_db, pages_per_db)
        
        self.logger.info(f"B4 Initial Equal Allocation: {allocations}")
        return allocations

    def update_allocations(self, current_metrics: Dict[str, Any], elapsed_in_phase: float):
        """
        Implements the "Greedy request with random order" logic.
        Improved version: Use demand calculation based on miss rate and access frequency, not direct cache_misses count
        """
        o = self.orchestrator
        
        db_ids = list(o.db_current_page_allocations.keys())
        if not db_ids: return {}
        
        self.random.shuffle(db_ids)
        self.logger.info(f"B4 'Greedy Anarchy' adjustment. Shuffled Order: {db_ids}")

        min_pages_per_db = _get_min_pages(self.strategy_specific_config, o.config)
        new_allocations = {db_id: min_pages_per_db for db_id in db_ids}
        
        remaining_pool = self.total_pages - sum(new_allocations.values())
        if remaining_pool < 0:
            self.logger.warning(f"B4: Total pages ({self.total_pages}) is less than total minimums. "
                                f"Allocations will be capped at minimums: {new_allocations}")
            return new_allocations

        self.logger.debug(f"B4: Starting with a distributable pool of {remaining_pool} pages.")

        # Round 1: Calculate demand for each database
        demands = {}
        for db_id in db_ids:
            stats = current_metrics.get(db_id, {})
            cache_hits = stats.get('cache_hits', 0)
            cache_misses = stats.get('cache_misses', 0)
            total_accesses = cache_hits + cache_misses
            
            if total_accesses > 0:
                miss_rate = cache_misses / total_accesses
                # Calculate demand based on miss rate and access frequency
                # Demand = miss_rate * sqrt(total_accesses) * scaling factor
                # Use sqrt to avoid high-access databases dominating too much
                demand_factor = miss_rate * (total_accesses ** 0.5)
                # Convert demand to pages (using proportion of total pool)
                demand_pages = int(demand_factor * remaining_pool / 100)  # Scaling factor
            else:
                demand_pages = 0
            
            demands[db_id] = demand_pages
            self.logger.debug(f"B4: DB[{db_id}] miss_rate={miss_rate:.3f}, accesses={total_accesses}, demand={demand_pages}")

        # Round 2: Satisfy demands in random order
        for db_id in db_ids:
            if remaining_pool <= 0:
                self.logger.debug("B4: Resource pool depleted. Ending allocation round.")
                break

            demand_pages = demands[db_id]
            granted_pages = min(demand_pages, remaining_pool)
            
            if granted_pages > 0:
                new_allocations[db_id] += granted_pages
                remaining_pool -= granted_pages
                self.logger.info(f"B4: DB[{db_id}] demanded {demand_pages}, was granted {granted_pages}. Pool left: {remaining_pool}")
            else:
                self.logger.info(f"B4: DB[{db_id}] demanded {demand_pages}, was granted 0.")

        # Round 3: Allocate remaining cache weighted by miss rate (not simple average)
        if remaining_pool > 0:
            self.logger.info(f"B4: {remaining_pool} pages remain after demand satisfaction. Distributing based on miss rates.")
            
            # Calculate miss rate weights
            total_weight = 0.0
            weights = {}
            for db_id in db_ids:
                stats = current_metrics.get(db_id, {})
                cache_hits = stats.get('cache_hits', 0)
                cache_misses = stats.get('cache_misses', 0)
                total_accesses = cache_hits + cache_misses
                
                if total_accesses > 0:
                    miss_rate = cache_misses / total_accesses
                    # Give higher weight to high miss rate databases
                    weight = miss_rate + 0.1  # Add base weight to avoid 0
                else:
                    weight = 0.1  # Databases with no access get minimum weight
                
                weights[db_id] = weight
                total_weight += weight
            
            # Allocate remaining cache by weight
            allocated = 0
            for i, db_id in enumerate(db_ids):
                if i == len(db_ids) - 1:  # Last database gets all remaining
                    extra_pages = remaining_pool - allocated
                else:
                    weight_ratio = weights[db_id] / total_weight
                    extra_pages = int(remaining_pool * weight_ratio)
                
                new_allocations[db_id] += extra_pages
                allocated += extra_pages
        
        self.logger.info(f"B4 Final Allocations for this cycle: {new_allocations} (Total: {sum(new_allocations.values())})")
        return new_allocations


class B5_SimulatedGlobalLruStrategy(BaseStrategy):
    """
    Baseline B5: Simulate global LRU (multi-table implementation)
    
    Implementation:
    - Use single database file, but contains multiple independent tables internally
    - Each original database corresponds to a set of tables, maintaining data isolation
    - Workload knows original database ID and routes to corresponding tables
    - All tables share same cache pool, simulating global LRU effect
    
    This implementation balances implementation complexity and experimental fairness:
    - Maintains original data access patterns and locality
    - Achieves cache sharing effect
    - Avoids unfair advantages from data consolidation
    
    Note: Current dynamic allocation mechanism is not fully implemented.
    """
    is_dynamic = True
    def __init__(self, orchestrator: Any, strategy_name: str, strategy_specific_config: Dict = None):
        super().__init__(orchestrator, strategy_name, strategy_specific_config)
        b5_config = self.strategy_specific_config or self.orchestrator.config.get("strategy_configurations", {}).get("B5_SimulatedGlobalLru", {})
        self.global_lru_size = b5_config.get("global_lru_size", 100000)
        self.global_lru = {} 
        self.next_timestamp = 0

    def calculate_initial_allocations(self):
        """
        In a true global LRU, individual DBs don't have fixed allocations.
        However, for our simulation, we must assign pages. We'll start with
        an equal distribution. The `update_allocations` will then adjust
        based on the simulated global hit rate.
        """
        num_databases = len(self.db_instance_configs)
        if num_databases == 0: return {}

        pages_per_db = self.total_pages // num_databases
        min_pages_per_db = _get_min_pages(self.strategy_specific_config, self.orchestrator.config)
        
        allocations = {db["id"]: max(min_pages_per_db, pages_per_db) for db in self.db_instance_configs}
        self.logger.info(f"B5 Initial Equal Allocation: {allocations}")
        return allocations

    def update_allocations(self, current_metrics: Dict[str, Any], elapsed_in_phase: float):
        # This is a complex simulation. For now, we'll just log.
        # A real implementation would need to intercept page accesses.
        self.logger.info("B5 update_allocations (Simulated Global LRU) not fully implemented. Allocations remain static.")
        return self.orchestrator.db_current_page_allocations

    def cleanup(self):
        """B5 strategy cleanup."""
        self.logger.debug("B5_SimulatedGlobalLruStrategy cleanup called.")
        self.global_lru.clear()
        self.next_timestamp = 0


class B7_Dynamic_Need_Strategy(BaseStrategy):
    """
    Baseline B7: Dynamic strategy based on "need" (ops * miss_rate)
    Uses simple single-speed EMA to smooth ops and miss_rate metrics, then allocates based on ops * miss_rate.
    This score measures database "pain level" - both busy and low cache hit rate.
    
    Key features:
    - Single-speed EMA: Uses traditional exponential moving average to smooth metrics
    - No vertical factor: Does not consider marginal gains, purely based on current observed need
    - No cooldown period: Executes calculation and allocation every tuning cycle
    """
    is_dynamic = True
    
    def __init__(self, orchestrator, strategy_name: str, strategy_specific_config: Dict = None):
        super().__init__(orchestrator, strategy_name, strategy_specific_config)
        
        # EMA parameters
        self.ema_alpha = self.strategy_specific_config.get('ema_alpha', 0.1)
        
        # Database state tracking
        self.db_states = {}
        
        self.logger.info(f"B7 strategy initialized with EMA alpha={self.ema_alpha}")
    
    def calculate_initial_allocations(self) -> Dict[str, int]:
        """Initial allocation: equally distribute all available pages"""
        # Use database configuration instead of db_instances
        db_configs = self.db_instance_configs
        if not db_configs:
            raise RuntimeError("B7 strategy: Database configuration list is empty, cannot calculate initial allocation")
        
        num_dbs = len(db_configs)
        pages_per_db = max(1, self.total_pages // num_dbs)
        allocations = {}
        
        for db_config in db_configs:
            db_id = db_config.get("id")
            if not db_id:
                raise RuntimeError(f"B7 strategy: Database configuration missing id field: {db_config}")
            
            allocations[db_id] = pages_per_db
            # Initialize database state
            self.db_states[db_id] = {
                'ema_ops': 0.0,
                'ema_miss_rate': 0.0,
                'initialized': False
            }
        
        self.logger.info(f"B7 initial allocation: {pages_per_db} pages per database, allocations={allocations}")
        return allocations
    
    def update_allocations(self, current_metrics: Dict[str, Any], elapsed_in_phase: float) -> Dict[str, int]:
        """Update cache allocation"""
        
        # Update EMA state for all databases
        self._update_ema_states(current_metrics)
        
        # Calculate need scores for all databases
        scores = {}
        total_score = 0.0
        
        # Get database ID list
        db_ids = [db_config.get("id") for db_config in self.db_instance_configs]
        
        for db_id in db_ids:
            if db_id in self.db_states and self.db_states[db_id]['initialized']:
                score = self._calculate_need_score(db_id)
                scores[db_id] = score
                total_score += score
            else:
                scores[db_id] = 0.0
        
        # Reallocate cache based on scores
        new_allocations = {}
        allocated_pages = 0
        
        if total_score > 0:
            for db_id in db_ids:
                if db_id in scores:
                    proportion = scores[db_id] / total_score
                    pages = max(1, int(self.total_pages * proportion))
                    new_allocations[db_id] = pages
                    allocated_pages += pages
        else:
            # If no valid scores, allocate equally
            pages_per_db = max(1, self.total_pages // len(db_ids))
            for db_id in db_ids:
                new_allocations[db_id] = pages_per_db
                allocated_pages += pages_per_db
        
        # Adjust to ensure total pages don't exceed limit
        if allocated_pages > self.total_pages:
            scale_factor = self.total_pages / allocated_pages
            for db_id in new_allocations:
                new_allocations[db_id] = max(1, int(new_allocations[db_id] * scale_factor))
        
        self.logger.info(f"B7 allocation update at {elapsed_in_phase:.1f}s: {new_allocations}")
        self.logger.info(f"B7 scores: {scores}, total_score: {total_score}")
        return new_allocations
    
    def _update_ema_states(self, current_metrics: Dict[str, Any]):
        """Update EMA state for all databases"""
        db_ids = [db_config.get("id") for db_config in self.db_instance_configs]
        for db_id in db_ids:
            if db_id not in current_metrics:
                continue
                
            db_metrics = current_metrics[db_id]
            
            # Calculate ops_per_second: use ops_count and reporting_interval
            if 'ops_count' not in db_metrics:
                raise RuntimeError(f"B7 strategy: DB[{db_id}] missing required ops_count metric")
            ops_count = db_metrics['ops_count']
            
            # Get reporting_interval from config
            reporting_interval = self.orchestrator.general_setup.get("reporting_interval_seconds")
            if reporting_interval is None:
                raise RuntimeError(f"B7 strategy: Missing reporting_interval_seconds in config")
            current_ops = ops_count / reporting_interval if reporting_interval > 0 else 0.0
            
            # Calculate hit_rate
            if 'cache_hits' not in db_metrics:
                raise RuntimeError(f"B7 strategy: DB[{db_id}] missing required cache_hits metric")
            if 'cache_misses' not in db_metrics:
                raise RuntimeError(f"B7 strategy: DB[{db_id}] missing required cache_misses metric")
            
            cache_hits = db_metrics['cache_hits']
            cache_misses = db_metrics['cache_misses']
            total_accesses = cache_hits + cache_misses
            
            if total_accesses == 0:
                # When no cache access, set default values
                current_hit_rate = 0.0
                current_miss_rate = 1.0  # Assume complete miss
                self.logger.warning(f"B7 strategy: DB[{db_id}] has no cache access record, using default miss_rate=1.0")
            else:
                current_hit_rate = cache_hits / total_accesses
                current_miss_rate = 1.0 - current_hit_rate
            
            # Debug info
            self.logger.debug(f"DB[{db_id}] B7 Raw Metrics: ops_count={ops_count}, hits={cache_hits}, misses={cache_misses}, interval={reporting_interval:.1f}s -> ops_per_sec={current_ops:.2f}, hit_rate={current_hit_rate:.3f}")
            
            if db_id not in self.db_states:
                self.db_states[db_id] = {
                    'ema_ops': current_ops,
                    'ema_miss_rate': current_miss_rate,
                    'initialized': True
                }
            else:
                state = self.db_states[db_id]
                if not state['initialized']:
                    # First update, directly use current values
                    state['ema_ops'] = current_ops
                    state['ema_miss_rate'] = current_miss_rate
                    state['initialized'] = True
                else:
                    # Update EMA
                    state['ema_ops'] = (1 - self.ema_alpha) * state['ema_ops'] + self.ema_alpha * current_ops
                    state['ema_miss_rate'] = (1 - self.ema_alpha) * state['ema_miss_rate'] + self.ema_alpha * current_miss_rate
    
    def _calculate_need_score(self, db_id: str) -> float:
        """Calculate database need score: ema_ops * ema_miss_rate"""
        if db_id not in self.db_states or not self.db_states[db_id]['initialized']:
            return 0.0
        
        state = self.db_states[db_id]
        score = state['ema_ops'] * state['ema_miss_rate']
        
        self.logger.info(f"DB[{db_id}] B7 Need Score: EMA_OPS={state['ema_ops']:.2f}, EMA_MissRate={state['ema_miss_rate']:.3f} -> Score={score:.2f}")
        return score