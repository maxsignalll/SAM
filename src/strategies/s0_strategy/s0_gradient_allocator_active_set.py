"""
Gradient Descent Cache Allocator for S0 Strategy - Active Set Optimization Version

Core Optimizations:
1. Use knee point detection to dynamically determine active set size k
2. Perform gradient descent only on k databases that need optimization most
3. Periodically perform global checks and active set updates
"""

import logging
import math
import numpy as np
from typing import Dict, Any, Tuple, Set, Optional, List


class S0GradientAllocatorActiveSet:
    """Active Set Based Gradient Descent Cache Allocator"""
    
    def __init__(self, strategy: Any, orchestrator: Any, hyperparams: Dict[str, Any], logger: logging.Logger):
        self.strategy = strategy
        self.orchestrator = orchestrator
        self.hyperparams = hyperparams
        self.logger = logger
        
        # Create dedicated active set log file
        self._setup_active_set_logger()
        
        # Gradient descent parameters
        self.learning_rate = hyperparams.get('gradient_learning_rate', 0.1)
        self.momentum = hyperparams.get('gradient_momentum', 0.7)
        self.min_step_pages = hyperparams.get('gradient_min_step_pages', 5)
        self.max_step_pages = hyperparams.get('gradient_max_step_pages', 100)
        self.convergence_threshold = hyperparams.get('convergence_threshold', 0.001)
        self.max_iterations = hyperparams.get('max_iterations', 50)
        
        # Active set parameters
        self.min_active_set_size = hyperparams.get('min_active_set_size', 3)  # Reduce minimum active set size
        self.max_active_set_size = hyperparams.get('max_active_set_size', 20)  # Reduce maximum active set size
        self.max_iterations_per_active_set = hyperparams.get('max_iterations_per_active_set', 20)  # Maximum lifecycle for single active set
        self.knee_sensitivity = hyperparams.get('knee_sensitivity', 1.0)
        self.instability_weight_gradient = hyperparams.get('instability_weight_gradient', 0.5)
        self.instability_weight_score_change = hyperparams.get('instability_weight_score_change', 0.3)
        self.instability_weight_recent_change = hyperparams.get('instability_weight_recent_change', 0.2)
        
        # Enhanced convergence detection parameters - adjusted for more aggressive optimization
        self.relative_improvement_threshold = hyperparams.get('relative_improvement_threshold', 0.0005)  # 0.05% relative improvement, more sensitive
        self.stagnation_threshold = hyperparams.get('stagnation_threshold', 6)  # Stagnation count threshold increased from 3 to 6, more tolerant
        
        # Internal state
        self.velocities = {}
        self.active_set = set()
        self.iterations_since_global_check = 0
        self.iterations_on_current_set = 0  # Number of iterations on current active set
        self.last_global_scores = {}
        self.last_allocations = {}
        self.score_history = {}  # db_id -> list of recent scores
        self.allocation_history = {}  # db_id -> list of recent allocations
        self.history_size = 5
        
        # Supply-demand matching related state
        self.demand_set = set()
        self.supply_set = set()
        self.candidate_pool = []
        self.sorted_db_list = []
        
        # Active set convergence detection
        self.active_set_score_sum_history = []  # History of FinalScore sum within active set
        self.convergence_window = hyperparams.get('convergence_window', 3)  # Convergence detection window
        self.score_improvement_threshold = hyperparams.get('score_improvement_threshold', 0.01)  # Score improvement threshold (absolute value, kept as backup)
        
        # Stagnation detector state
        self.high_water_mark = 0.0  # Historical highest score
        self.stagnation_counter = 0  # Stagnation counter
        
        # Epoch manager reference
        self.epoch_manager = None
        
        # Statistics
        self.total_iterations = 0
        self.active_set_updates = 0
        self.global_checks = 0
    
    def _setup_active_set_logger(self):
        """Setup dedicated active set logging"""
        import os
        from datetime import datetime
        
        # Create log directory
        log_dir = "results/active_set_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Create dedicated active set logger
        strategy_name = getattr(self.strategy, 'strategy_name', 'Unknown')
        self.as_logger = logging.getLogger(f"ActiveSet_{strategy_name}")
        # Set to DEBUG level to record all detailed information
        self.as_logger.setLevel(logging.INFO)  # Scalability test: changed to INFO level
        
        # Remove existing handlers if any
        for handler in self.as_logger.handlers[:]:
            self.as_logger.removeHandler(handler)
        
        # Create file handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"active_set_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        # Set to DEBUG level to record all detailed information
        file_handler.setLevel(logging.DEBUG)
        
        # Set format
        formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler
        self.as_logger.addHandler(file_handler)
        self.as_logger.propagate = False  # Prevent log propagation to parent logger
        
        # Record log file location
        # self.as_logger.info(f"Active set log file created: {log_file}")  # Scalability test comment
        # self.logger.info(f"Active set dedicated log file: {log_file}")  # Scalability test comment
    
    def _log(self, level: str, message: str):
        """Log to both main log and active set dedicated log"""
        # Log to main log
        getattr(self.logger, level)(message)
        # Log to active set dedicated log
        if hasattr(self, 'as_logger'):
            getattr(self.as_logger, level)(message)
        
    def calculate_allocations(self, 
                            db_ids: list,
                            scores: Dict[str, float], 
                            current_allocations: Dict[str, int],
                            fixed_allocations: Dict[str, int],
                            total_pages: int) -> Dict[str, int]:
        """
        Active set based gradient descent calculation of new cache allocations
        """
        # Debug log: examine input scores
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # self._log('info', f"\n=== [{timestamp}] Active set allocator received scores ===")  # Scalability test comment
        # Scalability test comment - detailed logs
        # self._log('info', f"DB IDs: {db_ids}")
        # self._log('info', f"Scores: {scores}")
        # self._log('info', f"Total pages: {total_pages}")
        
        # New: record detailed information about current and fixed allocations
        # self._log('info', f"Current allocation total: {sum(current_allocations.values())} pages")
        # self._log('info', f"Fixed allocation total: {sum(fixed_allocations.values())} pages")
        # self._log('info', f"Elastic pool size: {total_pages - sum(fixed_allocations.values())} pages")
        
        # New: analyze scores distribution - optimization: calculate statistics only when needed
        cv_score = None
        if scores:
            # Optimization: avoid repeated sorting, sort once use multiple times
            score_values = list(scores.values())
            if score_values:
                max_score = max(score_values)
                min_score = min(score_values)
                avg_score = sum(score_values) / len(score_values)
                # Calculate CV only when needed
                if avg_score > 1e-6:
                    std_score = math.sqrt(sum((s - avg_score) ** 2 for s in score_values) / len(score_values))
                    cv_score = std_score / avg_score
                else:
                    cv_score = 0
            
            # self._log('info', f"Scores distribution: max={max_score:.4f}, min={min_score:.4f}, "
            #                 f"avg={avg_score:.4f}, std={std_score:.4f}, CV={cv_score:.2%}")  # Scalability test comment
            
            # Record top 5 high score databases - commented out during scalability test
            # top_dbs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
            # self._log('info', f"Top 5 high score DBs: {[(db, f'{score:.4f}') for db, score in top_dbs]}")  # Scalability test comment
        
        # Check if all scores are 0
        all_zero = all(score == 0 for score in scores.values())
        if all_zero:
            self._log('warning', "Warning: All database scores are 0!")
            # self._log('info', f"Current allocation status: {current_allocations}")  # Scalability test comment
            # When all scores are 0, provide initial allocation based on priority
            # This helps system have reasonable initial allocation at startup
            if hasattr(self.strategy, 'db_states'):
                priorities = {db_id: self.strategy.db_states.get(db_id, {}).get('priority', 1) 
                            for db_id in db_ids}
                # self._log('info', f"Database priorities: {priorities}")  # Scalability test comment
                # Create priority-based virtual scores
                virtual_scores = {db_id: priority for db_id, priority in priorities.items()}
                # self._log('info', f"Using priority-based virtual scores: {virtual_scores}")  # Scalability test comment
                scores = virtual_scores
        
        # Ensure current_allocations is not empty
        if not current_allocations:
            # self.logger.warning("Current allocation is empty, initializing with average allocation")  # Scalability test comment
            avg_pages = max(1, total_pages // len(db_ids))
            current_allocations = {db_id: avg_pages for db_id in db_ids}
        
        # Update history records
        self._update_history(db_ids, scores, current_allocations)
        
        # Determine if global check is needed
        if self._should_perform_global_check() or not self.active_set:
            # self._log('info', "Triggering global check")  # Scalability test comment
            self._perform_global_check(db_ids, scores, current_allocations, fixed_allocations, total_pages)
            self.global_checks += 1
        else:
            # self.logger.info(f"Continue optimizing on current active set (iteration {self.iterations_on_current_set})")  # Scalability test comment
            pass
        
        # Execute optimization on active set
        # self._log('info', f"=== Starting active set optimization ===")  # Scalability test comment
        # self._log('info', f"Active set size: {len(self.active_set)}")  # Scalability test comment
        # self._log('info', f"Demand side: {sorted(list(self.demand_set))}")  # Scalability test comment
        # self._log('info', f"Supply side: {sorted(list(self.supply_set))}")  # Scalability test comment
        
        new_allocations = self._optimize_active_set(
            scores, current_allocations, fixed_allocations, total_pages
        )
        
        # Record score sum within active set (for convergence detection)
        if self.active_set:
            active_set_score_sum = sum(scores.get(db_id, 0) for db_id in self.active_set)
            self.active_set_score_sum_history.append(active_set_score_sum)
            
            # self.logger.debug(f"Active set score sum: {active_set_score_sum:.4f}")  # Scalability test comment
            
            # Limit history record length
            if len(self.active_set_score_sum_history) > self.convergence_window * 2:
                self.active_set_score_sum_history = self.active_set_score_sum_history[-self.convergence_window * 2:]
        
        self.iterations_since_global_check += 1
        self.iterations_on_current_set += 1
        
        return new_allocations
    
    def _should_perform_global_check(self) -> bool:
        """Determine if global check is needed - based on active set convergence state
        
        Uses three mechanisms:
        1. Relative improvement detection
        2. Stagnation counter
        3. Lifecycle upper limit
        """
        # If active set is empty, need global check
        if not self.active_set:
            # self.logger.info("Active set is empty, need global check")  # Scalability test comment
            return True
        
        # If not enough historical data, continue optimization
        if len(self.active_set_score_sum_history) < self.convergence_window:
            # self.logger.debug(f"Insufficient historical data: {len(self.active_set_score_sum_history)} < {self.convergence_window}, continue optimization")  # Scalability test comment
            return False
        
        # Get latest score sum
        current_score = self.active_set_score_sum_history[-1]
        recent_scores = self.active_set_score_sum_history[-self.convergence_window:]
        # self._log('info', f"Recent {self.convergence_window} score sums: {[f'{s:.4f}' for s in recent_scores]}")  # Scalability test comment
        
        # Method 1: Convergence detection based on relative improvement
        improvements = []
        for i in range(1, len(recent_scores)):
            improvement = recent_scores[i] - recent_scores[i-1]
            improvements.append(improvement)
        
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0
        avg_score = sum(recent_scores) / len(recent_scores)
        
        # Calculate relative improvement rate
        if avg_score > 0:
            relative_improvement = abs(avg_improvement) / avg_score
            # self._log('info', f"Average improvement: {avg_improvement:.4f}, average score: {avg_score:.4f}, "  # Scalability test comment
            #                 f"relative improvement rate: {relative_improvement:.2%}")  # Scalability test comment
            
            if relative_improvement < self.relative_improvement_threshold:
                # self.logger.info(f"Active set convergence detection (relative improvement): "  # Scalability test comment
                #                f"{relative_improvement:.2%} < {self.relative_improvement_threshold:.2%}")  # Scalability test comment
                return True
        
        # Method 2: Stagnation detector
        # Update historical highest score and stagnation counter
        if current_score > self.high_water_mark:
            # self.logger.debug(f"Update historical highest score: {self.high_water_mark:.4f} -> {current_score:.4f}")  # Scalability test comment
            self.high_water_mark = current_score
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
            # self._log('info', f"Did not exceed historical highest score, stagnation count: {self.stagnation_counter}")  # Scalability test comment
        
        if self.stagnation_counter >= self.stagnation_threshold:
            # self.logger.info(f"Active set convergence detection (stagnation detection): "  # Scalability test comment
            #                f"Consecutive {self.stagnation_counter} times did not exceed historical highest score")  # Scalability test comment
            return True
        
        # Method 3: Lifecycle protection
        if self.iterations_on_current_set >= self.max_iterations_per_active_set:
            # self.logger.info(f"Active set reached maximum lifecycle {self.max_iterations_per_active_set}, triggering global check")  # Scalability test comment
            return True
        # else:
        #     self.logger.debug(f"Active set lifecycle: {self.iterations_on_current_set}/{self.max_iterations_per_active_set}")  # Scalability test comment
        
        return False
    
    def _perform_global_check(self, db_ids: list, scores: Dict[str, float],
                             current_allocations: Dict[str, int],
                             fixed_allocations: Dict[str, int],
                             total_pages: int):
        """Execute global check and active set identification - using supply-demand matching principle"""
        # self._log('info', "=== Executing global check (supply-demand matching mode) ===")  # Scalability test comment
        # self._log('info', f"Input scores: {scores}")  # Scalability test comment
        # self._log('info', f"Current allocations: {current_allocations}")  # Scalability test comment
        
        # Use knee point detection to determine demand side quantity (this method sets self.sorted_db_list)
        k_demand = self._determine_active_set_size_by_score(scores)
        
        # Demand side: top k_demand databases with highest scores
        demand_set = set(self.sorted_db_list[:k_demand])
        # self.logger.info(f"Demand side set: {sorted(list(demand_set))}")  # Scalability test comment
        
        # Calculate total cache demand of demand side
        demand_total_current = sum(current_allocations.get(db, 0) for db in demand_set)
        total_score_sum = sum(scores.values())
        
        if total_score_sum > 0:
            demand_total_target = sum(scores[db] * total_pages / total_score_sum for db in demand_set)
        else:
            demand_total_target = 0
            
        # self.logger.info(f"Demand side current total cache: {demand_total_current} pages")  # Scalability test comment
        # self.logger.info(f"Demand side target total cache: {demand_total_target:.0f} pages")  # Scalability test comment
        
        # Estimate cache amount needed from supply side
        cache_to_transfer = max(0, demand_total_target - demand_total_current)
        # self._log('info', f"Expected cache transfer needed: {cache_to_transfer:.0f} pages")  # Scalability test comment
        
        # Supply side: select from databases with lowest scores
        supply_candidates = self.sorted_db_list[k_demand:]  # Sorted by score descending, so later ones are low score
        supply_set = set()
        accumulated_supply = 0
        
        # self.logger.info(f"Supply candidate count: {len(supply_candidates)}")  # Scalability test comment
        # self._log('info', f"Supply candidate list: {supply_candidates}")  # Scalability test comment
        
        # Added: Record supply-demand difference details
        if cache_to_transfer > 0:
            # self._log('info', f"=== supply_demand_analysis_details ===")  # Scalability test comment
            # self._log('info', f"Demand side needs additional cache: {cache_to_transfer:.0f}  pages")  # Scalability test comment
            # Record the gap for each demand side
            for db in demand_set:
                current = current_allocations.get(db, 0)
                if total_score_sum > 0:
                    target = scores[db] * total_pages / total_score_sum
                    gap = target - current
                    if gap > 0:
                        self._log('debug', f"Demand side {db}: Current={current}, Target={target:.0f}, Gap={gap:.0f}")
        
        # Select sufficient suppliers to ensure adequate resources for flow
        # Core logic: Select suppliers based on demand, not proportion of candidates
        for i in range(len(supply_candidates) - 1, -1, -1):  # Start from lowest score
            db_id = supply_candidates[i]
            db_allocation = current_allocations.get(db_id, 0)
            db_fixed = fixed_allocations.get(db_id, 0)
            
            # Calculate elastic cache amount this database can provide
            available_supply = max(0, db_allocation - db_fixed)
            
            if available_supply > 0:
                supply_set.add(db_id)
                accumulated_supply += available_supply
                
                # self.logger.info(f"Add supplier: {db_id}, Currentallocation={db_allocation}, "  # Scalability test comment
                #                f"Fixed allocation={db_fixed}, Available supply={available_supply}, "  # Scalability test comment
                #                f"Accumulated supply={accumulated_supply}")  # Scalability test comment
                
                # Main stop condition: Accumulated supply meets demand (20% margin)
                if accumulated_supply >= cache_to_transfer * 1.2:
                    # self.logger.info(f"Accumulated supply {accumulated_supply} >= demand*1.2 {cache_to_transfer*1.2:.0f}")  # Scalability test comment
                    # self.logger.info(f"Selected {len(supply_set)} suppliers, meeting cache transfer demand")  # Scalability test comment
                    break
                
                # Safety protection: Prevent selecting too many suppliers in extreme cases
                # This is just a safety net, not the main stop condition
                if len(supply_set) >= len(supply_candidates) * 0.8:  # Increase to 80%
                    # self.logger.warning(f"Number of suppliers reaches 80% of candidates ({len(supply_set)}/{len(supply_candidates)}), "  # Scalability test comment
                    #                   f"This may indicate excessive demand or suppliers are too poor")  # Scalability test comment
                    # self.logger.info(f"CurrentAccumulated supply: {accumulated_supply}, demand: {cache_to_transfer:.0f}")  # Scalability test comment
                    break
                
                # Additional protection: Total active set size limit
                if len(demand_set) + len(supply_set) >= self.max_active_set_size:
                    # self.logger.info(f"Active set size reaches upper limit {self.max_active_set_size}, Stop adding suppliers")  # Scalability test comment
                    break
            # else:
            #     self.logger.debug(f"Skip supplier {db_id}：No available elastic cache（Current={db_allocation}, fixed={db_fixed}）")  # Scalability test comment
        
        # Check if supply is sufficient
        if accumulated_supply < cache_to_transfer * 0.8:  # insufficient_supply80%demand
            # self.logger.warning(f"Supply may be insufficient: Accumulated supply {accumulated_supply} < demand*0.8 {cache_to_transfer*0.8:.0f}")  # Scalability test comment
            # self.logger.info(f"Selected all available suppliers:{len(supply_set)}  items")  # Scalability test comment
            pass
        
        # Special handling: If k_demand is 0 (equilibrium plateau state), enable exploratory optimization
        if k_demand == 0:
            # self.logger.info("\nDetected equilibrium plateau state, enabling exploratory optimization mode")  # Scalability test comment
            pass
            
            # Force selection of highest and lowest scoring databases for fine-tuning experiments
            min_active_set = max(3, min(len(db_ids) // 4, 8))  # at_least3 items, at_most8 items
            
            # Select highest and lowest scoring halves
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_dbs = [db_id for db_id, _ in sorted_scores[:min_active_set//2]]
            bottom_dbs = [db_id for db_id, _ in sorted_scores[-min_active_set//2:]]
            
            # Create exploratory active set
            exploration_set = set(top_dbs + bottom_dbs)
            
            self.active_set = exploration_set
            self.demand_set = set(top_dbs)  # High scorers as demand side
            self.supply_set = set(bottom_dbs)  # Low scorers as suppliers
            
            # self.logger.info(f"Exploratory active set: {len(self.active_set)} items (Demand side:{len(self.demand_set)}, supplier:{len(self.supply_set)})")  # Scalability test comment
            # self.logger.info(f"Demand side: {sorted(self.demand_set)}")  # Scalability test comment
            # self.logger.info(f"supplier: {sorted(self.supply_set)}")  # Scalability test comment
            
            # set_iteration_count
            self.iterations_since_global_check = 0
            self.iterations_on_current_set = 0
            return  # continue_optimization
        
        # Merge demand side and suppliers as active set
        self.active_set = demand_set.union(supply_set)
        
        # Save supplier list and candidate pool
        self.supply_set = supply_set
        self.demand_set = demand_set
        self.candidate_pool = [db for db in supply_candidates if db not in supply_set]
        
        # Reset counters and history
        self.iterations_since_global_check = 0
        self.iterations_on_current_set = 0  # Iterations for current active set
        self.active_set_updates += 1
        self.active_set_score_sum_history = []  # Clear active set score history
        # Reset stagnation detector
        self.high_water_mark = 0.0
        self.stagnation_counter = 0
        
        # Log information
        # self.logger.info(f"Active set update (supply-demand matching mode):")  # Scalability test comment
        # self.logger.info(f"  - Demand side: {len(self.demand_set)} itemshigh-scoring databases")  # Scalability test comment
        # self.logger.info(f"  - supplier: {len(self.supply_set)} itemslow-scoring databases")  # Scalability test comment
        # self.logger.info(f"  - candidate_pool: {len(self.candidate_pool)} itemscandidate databases")  # Scalability test comment
        # self.logger.info(f"  - Active set total: {len(self.active_set)} itemsdatabase")  # Scalability test comment
        # self._log('info', f"Demand side: {sorted(list(self.demand_set))}")  # Scalability test comment
        # self._log('info', f"supplier: {sorted(list(self.supply_set))}")  # Scalability test comment
        
        # Record initial score sum of active set
        if self.active_set:
            initial_score_sum = sum(scores.get(db_id, 0) for db_id in self.active_set)
            self.active_set_score_sum_history.append(initial_score_sum)
            # self._log('info', f"Active set initial score sum: {initial_score_sum:.4f}")  # Scalability test comment
            
            # Record cache amount to transfer
            if cache_to_transfer > 0:
                # self._log('info', f"Expected cache transfer: {cache_to_transfer:.0f}  pages")  # Scalability test comment
                pass
        
        # Save global state
        self.last_global_scores = scores.copy()
    
    def _calculate_instability_scores(self, db_ids: list, scores: Dict[str, float],
                                     current_allocations: Dict[str, int],
                                     target_allocations: Dict[str, int]) -> Dict[str, float]:
        """Calculate instability score for each database"""
        # Debug information
        self.logger.debug(f"Instability calculation - db_ids: {db_ids}")
        self.logger.debug(f"Instability calculation - target_allocations keys: {list(target_allocations.keys())}")
        self.logger.debug(f"Instability calculation - current_allocations keys: {list(current_allocations.keys())}")
        
        instability = {}
        
        for db_id in db_ids:
            # 1. Gradient magnitude (gap between current and target allocation)
            current_alloc = current_allocations.get(db_id, 0)
            target_alloc = target_allocations.get(db_id, 0)
            gradient = abs(target_alloc - current_alloc)
            gradient_normalized = gradient / max(1, current_alloc)
            
            # 2. Score change rate
            score_volatility = self._get_score_volatility(db_id)
            
            # 3. Recent allocation changes
            recent_allocation_change = self._get_recent_allocation_change(db_id)
            
            # Comprehensive instability score
            instability[db_id] = (
                gradient_normalized * self.instability_weight_gradient +
                score_volatility * self.instability_weight_score_change +
                recent_allocation_change * self.instability_weight_recent_change
            )
        
        return instability
    
    def _dual_heap_filter(self, scores: Dict[str, float], k_max: int) -> Tuple[List[Tuple[float, str]], List[Tuple[float, str]], Set[str]]:
        """Bidirectional candidate set filtering - O(N log k_max) complexity
        
        Returns:
            top_candidates: [(score, db_id), ...] sorted in descending order
            bottom_candidates: [(score, db_id), ...] sorted in ascending order  
            middle_set: set of database IDs not in top or bottom
        """
        import heapq
        
        # min_heap_storagetop k（use_negative_scores_for_max_heap_effect）
        top_heap = []
        # max_heap_storagebottom k（use_scores_directly）
        bottom_heap = []
        
        # record all database IDs
        all_db_ids = set(scores.keys())
        
        for db_id, score in scores.items():
            # update top heap
            if len(top_heap) < k_max:
                heapq.heappush(top_heap, (score, db_id))
            elif score > top_heap[0][0]:
                heapq.heapreplace(top_heap, (score, db_id))
            
            # update bottom heap (use negative scores for max heap)
            if len(bottom_heap) < k_max:
                heapq.heappush(bottom_heap, (-score, db_id))
            elif score < -bottom_heap[0][0]:
                heapq.heapreplace(bottom_heap, (-score, db_id))
        
        # extract and sort top candidates (descending)
        top_candidates = sorted([(s, db) for s, db in top_heap], reverse=True)
        # extract and sort bottom candidates (ascending)
        bottom_candidates = sorted([(-s, db) for s, db in bottom_heap])
        
        # calculate middle part
        top_db_set = {db for _, db in top_candidates}
        bottom_db_set = {db for _, db in bottom_candidates}
        middle_set = all_db_ids - top_db_set - bottom_db_set
        
        return top_candidates, bottom_candidates, middle_set
    
    def _determine_active_set_size_by_score(self, scores: Dict[str, float]) -> int:
        """Optimized knee point detection - using bidirectional candidate set filtering"""
        # self._log('info', "=== Start knee point detection ===")  # Scalability test comment
        
        if not scores:
            # self._log('warning', "Score dictionary empty, return minimum active set size")  # Scalability test comment
            return self.min_active_set_size
        
        n = len(scores)
        
        # If database count is small, use original algorithm
        if n <= 10:  # Use original algorithm only for very small scales
            # Need sorting to set sorted_db_list
            sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            self.sorted_db_list = [db_id for db_id, _ in sorted_items]
            sorted_scores = [score for _, score in sorted_items]
            
            if n <= self.min_active_set_size:
                return n
                
            # check if all scores are 0
            if sorted_scores[0] <= 0:
                result = min(max(1, int(n * 0.2)), self.min_active_set_size, n)
                return result
            
            # simplified CV detection
            max_score = sorted_scores[0]
            min_score = sorted_scores[-1]
            score_range = max_score - min_score
            
            if score_range < max_score * 0.02:
                return 0  # balanced state, skip optimization
                
            # use simplified knee detection - fixed ratio
            k_demand = max(self.min_active_set_size, int(n * 0.3))
            return min(k_demand, self.max_active_set_size, n)
        
        # use bidirectional candidate set filtering for optimization
        # for 20-80 databases, k_max is set smaller for performance gain
        if n <= 30:
            k_max = 10  # small scale uses smaller k_max
        elif n <= 80:
            k_max = 20  # medium scale
        else:
            k_max = max(30, int(n * 0.1))  # large scale at most 10%
        top_candidates, bottom_candidates, middle_set = self._dual_heap_filter(scores, k_max)
        
        # construct sorted_db_list (approximate)
        # top part is precisely sorted, middle part order not important, bottom part is precisely sorted
        self.sorted_db_list = [db for _, db in top_candidates]
        self.sorted_db_list.extend(middle_set)  # middle part order not important
        self.sorted_db_list.extend([db for _, db in bottom_candidates])
        
        # based on top candidates set for balanced detection
        if top_candidates:
            top_scores = [score for score, _ in top_candidates]
            max_score = top_scores[0]
            min_score_in_top = top_scores[-1]
            
            # if top k_max are all close, might be balanced state
            if max_score > 0 and (max_score - min_score_in_top) < max_score * 0.05:
                # check gap between top and bottom again
                if bottom_candidates:
                    max_bottom_score = bottom_candidates[-1][0]  # bottom is ascending
                    if (max_score - max_bottom_score) < max_score * 0.1:
                        # overall more balanced, skip optimization
                        return 0
        
        # simplified knee detection - on top candidates set
        # since top k_max already filtered, can use fixed ratio
        if len(top_candidates) >= 20:
            k_demand = max(self.min_active_set_size, len(top_candidates) // 2)
        else:
            k_demand = max(self.min_active_set_size, int(n * 0.3))
            
        return min(k_demand, self.max_active_set_size, n)
    
    def _determine_active_set_size(self, instability_scores: Dict[str, float]) -> int:
        """Use bucket-based method to determine active set size (O(N) complexity)"""
        if not instability_scores:
            return self.min_active_set_size
        
        # if database quantity too small, directly return
        if len(instability_scores) <= self.min_active_set_size:
            return len(instability_scores)
        
        # use bucket-based method for knee detection
        return self._bucket_based_knee_detection(instability_scores)
    
    def _bucket_based_knee_detection(self, instability_scores: Dict[str, float]) -> int:
        """Bucket sort based knee point detection (O(N) complexity)"""
        # configuration parameters
        num_buckets = self.hyperparams.get('knee_detection_buckets', 100)  # number of buckets
        min_bucket_drop_ratio = self.hyperparams.get('min_bucket_drop_ratio', 0.5)  # minimum drop ratio
        
        # find score range
        scores = list(instability_scores.values())
        min_score = min(scores)
        max_score = max(scores)
        
        # if all scores are same, return default value
        if max_score - min_score < 1e-10:
            return min(self.min_active_set_size * 2, len(scores))
        
        # create buckets
        buckets = [[] for _ in range(num_buckets)]
        bucket_counts = [0] * num_buckets
        
        # allocate databases to buckets (O(N))
        for db_id, score in instability_scores.items():
            # normalize score to [0, 1]
            normalized_score = (score - min_score) / (max_score - min_score)
            # calculate bucket index (ensure highest score enters last bucket)
            bucket_idx = min(int(normalized_score * num_buckets), num_buckets - 1)
            buckets[bucket_idx].append(db_id)
            bucket_counts[bucket_idx] += 1
        
        # from high score buckets start accumulated, findknee point
        accumulated_count = 0
        prev_non_empty_count = None
        knee_found = False
        k = 0
        
        # from highest score buckets traverse downward
        for i in range(num_buckets - 1, -1, -1):
            if bucket_counts[i] > 0:
                accumulated_count += bucket_counts[i]
                
                # detect sudden drop (knee point)
                if prev_non_empty_count is not None:
                    drop_ratio = bucket_counts[i] / prev_non_empty_count
                    
                    # if current bucket count relative to previous non-empty bucket has significant drop
                    if drop_ratio < min_bucket_drop_ratio and not knee_found:
                        # found_knee_point
                        k = accumulated_count - bucket_counts[i]  # not include current buckets
                        knee_found = True
                        self.logger.debug(
                            f"Bucket method knee detection: at bucket{i}Knee point detected, "
                            f"Drop ratio = {drop_ratio:.2f}, Active set size = {k}"
                        )
                        break
                
                prev_non_empty_count = bucket_counts[i]
                
                # if accumulated quantity already exceeded maximum active set size, stop
                if accumulated_count >= self.max_active_set_size:
                    k = self.max_active_set_size
                    break
        
        # if_no_obvious_knee_point_found
        if not knee_found:
            # usegradientmethodfindknee point
            k = self._gradient_based_knee_detection(bucket_counts, num_buckets)
            
            if k == 0:
                # if gradient method also not found, use default strategy
                # select buckets containing top 20% databases
                target_count = int(len(instability_scores) * 0.2)
                accumulated_count = 0
                
                for i in range(num_buckets - 1, -1, -1):
                    accumulated_count += bucket_counts[i]
                    if accumulated_count >= target_count:
                        k = accumulated_count
                        break
        
        # ensurekwithin reasonablerangewithin
        k = max(self.min_active_set_size, min(k, self.max_active_set_size))
        
        # recordbucket_distributioninfo（fordebug）
        if self.logger.isEnabledFor(logging.DEBUG):
            non_empty_buckets = sum(1 for count in bucket_counts if count > 0)
            self.logger.debug(
                f"bucket_distribution: total_buckets={num_buckets}, non_empty_buckets={non_empty_buckets}, "
                f"finalActive set size = {k}"
            )
        
        return k
    
    def _gradient_based_knee_detection(self, bucket_counts: list, num_buckets: int) -> int:
        """Gradient-based knee point detection (backup method)"""
        # calculateaccumulatedquantity
        accumulated_counts = []
        accumulated = 0
        
        for i in range(num_buckets - 1, -1, -1):
            accumulated += bucket_counts[i]
            if bucket_counts[i] > 0:  # only record non-empty buckets
                accumulated_counts.append(accumulated)
        
        if len(accumulated_counts) < 3:
            return 0  # too few data points, cannot detect
        
        # calculate second order gradient (acceleration)
        max_acceleration = 0
        knee_index = 0
        
        for i in range(1, len(accumulated_counts) - 1):
            # calculate second-order difference (approximate second derivative)
            acceleration = (accumulated_counts[i+1] - accumulated_counts[i]) - \
                         (accumulated_counts[i] - accumulated_counts[i-1])
            
            if acceleration > max_acceleration:
                max_acceleration = acceleration
                knee_index = i
        
        # if found obvious acceleration point
        if max_acceleration > len(accumulated_counts) * 0.1:  # threshold
            return accumulated_counts[knee_index]
        
        return 0
    
    def _optimize_active_set(self, scores: Dict[str, float],
                            current_allocations: Dict[str, int],
                            fixed_allocations: Dict[str, int],
                            total_pages: int) -> Dict[str, int]:
        """Execute gradient descent optimization only on active set"""
        # self._log('info', "=== startactive_setoptimization ===")  # Scalability test comment
        
        # If active set is empty, return current allocation
        if not self.active_set:
            # self.logger.warning("Active set empty, return current allocation")  # Scalability test comment
            return current_allocations.copy()
        
        # Extract active set related data
        active_scores = {db: scores.get(db, 0) for db in self.active_set}
        active_current = {db: current_allocations.get(db, 0) for db in self.active_set}
        active_fixed = {db: fixed_allocations.get(db, 0) for db in self.active_set}
        
        # self._log('info', f"active_setsize: {len(self.active_set)}")  # Scalability test comment
        # self._log('info', f"Demand side: {sorted(list(self.demand_set))}")  # Scalability test comment
        # self._log('info', f"supplier: {sorted(list(self.supply_set))}")  # Scalability test comment
        
        # Calculate total pages of active set (maintain non-active set allocations unchanged)
        inactive_pages = sum(
            current_allocations.get(db, 0) for db in current_allocations 
            if db not in self.active_set
        )
        active_total_pages = total_pages - inactive_pages
        
        self.logger.debug(f"total pagescount: {total_pages}, Non-active set usage: {inactive_pages}, Active set available: {active_total_pages}")
        
        # Added: record state before gradient descent
        # self._log('info', f"=== Gradient descent optimization starts ===")  # Scalability test comment
        # self._log('info', f"Total score within active set: {sum(active_scores.values()):.4f}")  # Scalability test comment
        # self._log('info', f"Active set current cache: {sum(active_current.values())}  pages")  # Scalability test comment
        # self._log('info', f"Active set fixed cache: {sum(active_fixed.values())}  pages")  # Scalability test comment
        
        # Run gradient descent on active set
        active_new_allocations = self._gradient_descent_on_subset(
            list(self.active_set),
            active_scores,
            active_current,
            active_total_pages  # This is the active set budget
        )
        
        # Merge results: maintain non-active set allocations unchanged
        new_allocations = current_allocations.copy()
        new_allocations.update(active_new_allocations)
        
        # Record major changes
        for db_id in self.active_set:
            old_alloc = current_allocations.get(db_id, 0)
            new_alloc = new_allocations.get(db_id, 0)
            if abs(new_alloc - old_alloc) > 10:
                # self._log('info', f"Cache adjustment: {db_id}: {old_alloc} -> {new_alloc} ({new_alloc - old_alloc:+d})")  # Scalability test comment
                pass
        
        # Verify sum
        total_allocated = sum(new_allocations.values())
        if abs(total_allocated - total_pages) > 1:
            # self.logger.warning(f"Allocation sum mismatch: {total_allocated} vs {total_pages}")  # Scalability test comment
            # Fine-tune to ensure correct sum
            self._adjust_total(new_allocations, total_pages)
        
        return new_allocations
    
    def _gradient_descent_on_subset(self, db_ids: list, scores: Dict[str, float],
                                   current_allocations: Dict[str, int],
                                   budget: int) -> Dict[str, int]:
        """Execute gradient descent on database subset
        
        Parameters:
            db_ids: List of database IDs in active set
            scores: Scores of each database
            current_allocations: Current allocation status
            budget: Total budget for active set (already calculated by caller)
        
        Returns:
            New allocation scheme, guaranteed sum equals budget
        """
        # initialize velocity for unseen databases
        for db_id in db_ids:
            if db_id not in self.velocities:
                self.velocities[db_id] = 0.0
        
        # from current allocation start, only include databases in active set
        allocations = {db_id: current_allocations.get(db_id, 0) for db_id in db_ids}
        
        # calculate target allocation - directly use budget
        total_score = sum(scores.values())
        
        target_allocations = {}
        for db_id in db_ids:
            if total_score > 0:
                # allocate budget by score proportion
                target_allocations[db_id] = budget * scores.get(db_id, 0) / total_score
            else:
                # Average allocation
                target_allocations[db_id] = budget / len(db_ids)
        
        # gradient_descentiteration
        converged = False
        iteration = 0
        
        # self._log('info', f"startgradient_descent: budget={budget}")  # Scalability test comment
        
        # recordinitial_state
        initial_total_score = sum(scores.values())
        # self._log('info', f"initialtotal_score: {initial_total_score:.4f}")  # Scalability test comment
        
        while iteration < self.max_iterations and not converged:
            # calculategradient
            gradients = self._calculate_gradients(allocations, target_allocations)
            
            # Update velocity (momentum)
            for db_id in db_ids:
                gradient = gradients.get(db_id, 0.0)
                self.velocities[db_id] = (
                    self.momentum * self.velocities.get(db_id, 0.0) + 
                    (1 - self.momentum) * gradient
                )
            
            # calculate adjustments
            adjustments = {}
            for db_id in db_ids:
                velocity = self.velocities.get(db_id, 0.0)
                # not use elastic pool again, directly use velocity as adjustment
                adjustment = self.learning_rate * velocity
                
                # limit adjustment magnitude
                if abs(adjustment) < 0.1:
                    adjustment = 0
                else:
                    adjustment = math.copysign(
                        min(abs(adjustment), self.max_step_pages),
                        adjustment
                    )
                
                adjustments[db_id] = int(adjustment)
            
            # apply adjustments
            new_allocations = {}
            exhausted_victims = []  # record resource exhausted suppliers
            
            for db_id in db_ids:
                current = allocations.get(db_id, 0)
                adjustment = adjustments.get(db_id, 0)
                
                # new allocation cannot be negative
                new_alloc = max(1, current + adjustment)  # at least keep1 pages
                new_allocations[db_id] = new_alloc
                
                # detect resource exhausted (supplier cache reduced to minimum 1 pages)
                if db_id in self.supply_set and new_alloc <= 1 and current > 1:
                    exhausted_victims.append(db_id)
                    # self._log('info', f"detectiontoresourceexhausted: {db_id} cache reduced to minimum 1")  # Scalability test comment
            
            # ensure total correct - use new budget-aware method
            self._rebalance_allocations_with_budget(db_ids, new_allocations, budget)
            
            # handle resource exhausted member replacement
            if exhausted_victims and self.candidate_pool:
                for victim_db in exhausted_victims:
                    # executeMember replacement
                    new_member = self._replace_exhausted_member(
                        victim_db, new_allocations, scores
                    )
                    if new_member:
                        # update db_ids list
                        db_ids.remove(victim_db)
                        db_ids.append(new_member)
                        # continue optimization
            
            # check converged
            max_change = max(
                abs(new_allocations[db_id] - allocations.get(db_id, 0)) 
                for db_id in db_ids
            )
            
            if iteration % 5 == 0:  # every 5 iterations record
                # Calculate current target function value (active set share of total)
                current_score_sum = 0
                for db_id in db_ids:
                    alloc_ratio = new_allocations[db_id] / budget if budget > 0 else 0
                    current_score_sum += scores.get(db_id, 0) * alloc_ratio
                
                self._log('debug', f"iteration {iteration}: max_change={max_change:.1f}, "
                                 f"Targetfunction_value={current_score_sum:.4f}")
                
                # Record major allocation changes
                major_changes = []
                for db_id in db_ids:
                    change = new_allocations[db_id] - allocations.get(db_id, 0)
                    if abs(change) > 10:
                        major_changes.append(f"{db_id}: {change:+d}")
                if major_changes:
                    self._log('debug', f"major_changes: {', '.join(major_changes[:5])}")
            
            if max_change < self.convergence_threshold:
                converged = True
                # self._log('info', f"gradient_descentconverged: max_change {max_change:.1f} < threshold {self.convergence_threshold}")  # Scalability test comment
            
            allocations = new_allocations
            iteration += 1
            self.total_iterations += 1
        
        if converged:
            # self.logger.info(f"active_setgradient_descentconverged: {iteration}iterationsiteration")  # Scalability test comment
            pass
        else:
            # self.logger.info(f"active_setgradient_descentreached_maximumiteration_count: {iteration}")  # Scalability test comment
            pass
        
        return allocations
    
    def _calculate_gradients(self, current: Dict[str, int], 
                            target: Dict[str, float]) -> Dict[str, float]:
        """Calculate gradients"""
        gradients = {}
        
        # only calculate gradient for databases in target (i.e. databases in active set)
        for db_id in target:
            # directly calculate difference, not involve fixed allocation again
            current_alloc = current.get(db_id, 0)
            target_alloc = target[db_id]
            
            # simple gradient: difference between target and current
            gradient = (target_alloc - current_alloc)
                
            gradients[db_id] = gradient
            
        return gradients
    
    def _calculate_target_allocations(self, db_ids: list, scores: Dict[str, float],
                                     fixed_allocations: Dict[str, int],
                                     total_pages: int) -> Dict[str, int]:
        """Calculate target allocations"""
        total_score = sum(scores.values())
        total_fixed = sum(fixed_allocations.get(db_id, 0) for db_id in db_ids)
        elastic_pool = total_pages - total_fixed
        
        target_allocations = {}
        for db_id in db_ids:
            fixed = fixed_allocations.get(db_id, 0)
            if total_score > 0:
                elastic = int(elastic_pool * scores.get(db_id, 0) / total_score)
            else:
                elastic = elastic_pool // len(db_ids)
            target_allocations[db_id] = fixed + elastic
        
        return target_allocations
    
    def _rebalance_allocations(self, db_ids: list, allocations: Dict[str, int],
                              fixed_allocations: Dict[str, int], total_pages: int):
        """Rebalance allocations to satisfy sum constraints"""
        total_allocated = sum(allocations.values())
        diff = total_allocated - total_pages
        
        if abs(diff) <= 1:
            # tiny difference, adjust largest database
            if diff != 0:
                max_db = max(db_ids, key=lambda x: allocations[x])
                allocations[max_db] -= diff
            return
        
        if diff > 0:
            # need reduce allocation
            reducible = {
                db_id: allocations[db_id] - fixed_allocations.get(db_id, 0)
                for db_id in db_ids
                if allocations[db_id] > fixed_allocations.get(db_id, 0)
            }
            
            total_reducible = sum(reducible.values())
            if total_reducible > 0:
                for db_id, reducible_amount in reducible.items():
                    reduction = int(diff * reducible_amount / total_reducible)
                    allocations[db_id] -= reduction
        else:
            # needincreaseallocation
            for db_id in db_ids:
                increase = int(-diff / len(db_ids))
                allocations[db_id] += increase
        
        # Final fine-tuning
        final_diff = sum(allocations.values()) - total_pages
        if final_diff != 0:
            max_db = max(db_ids, key=lambda x: allocations[x])
            allocations[max_db] -= final_diff
    
    def _rebalance_allocations_with_budget(self, db_ids: list, allocations: Dict[str, int], budget: int):
        """Rebalance allocations to ensure sum equals budget
        
        Use fairer method, prioritize adjusting from low-score databases
        """
        total = sum(allocations[db_id] for db_id in db_ids)
        diff = total - budget
        
        if abs(diff) <= 1:
            # tiny difference, adjust from largest database
            if diff != 0:
                max_db = max(db_ids, key=lambda x: allocations[x])
                allocations[max_db] -= diff
            return
        
        if diff > 0:
            # Need to reduce allocation - reduce by allocation proportion
            for db_id in db_ids:
                reduction = int(diff * allocations[db_id] / total)
                allocations[db_id] -= reduction
        else:
            # needincreaseallocation - averageincrease
            increase_per_db = int(-diff / len(db_ids))
            for db_id in db_ids:
                allocations[db_id] += increase_per_db
        
        # Final fine-tuning to ensure exact equality
        final_diff = sum(allocations[db_id] for db_id in db_ids) - budget
        if final_diff != 0:
            max_db = max(db_ids, key=lambda x: allocations[x])
            allocations[max_db] -= final_diff
    
    def _adjust_total(self, allocations: Dict[str, int], total_pages: int):
        """Adjust total allocation to match target - improved version, start adjusting from low-score databases"""
        total = sum(allocations.values())
        diff = total - total_pages
        
        if diff == 0:
            return
        
        # if has score info, prioritize adjusting from low score databases
        if hasattr(self, 'last_global_scores') and self.last_global_scores:
            # sort by score, from low to high
            sorted_dbs = sorted(allocations.keys(), 
                              key=lambda x: self.last_global_scores.get(x, 0))
            
            remaining_diff = diff
            for db_id in sorted_dbs:
                if remaining_diff == 0:
                    break
                    
                if diff > 0:
                    # need reduce, start from low score
                    can_reduce = min(allocations[db_id] - 1, remaining_diff)  # at least keep1 pages
                    if can_reduce > 0:
                        allocations[db_id] -= can_reduce
                        remaining_diff -= can_reduce
                else:
                    # need increase (this is rare)
                    allocations[db_id] += 1
                    remaining_diff += 1
                    
            # if still remaining, adjust from largest database
            if remaining_diff != 0:
                max_db = max(allocations.keys(), key=lambda x: allocations[x])
                allocations[max_db] -= remaining_diff
        else:
            # degrade to original method
            max_db = max(allocations.keys(), key=lambda x: allocations[x])
            allocations[max_db] -= diff
    
    def _update_history(self, db_ids: list, scores: Dict[str, float], 
                       allocations: Dict[str, int]):
        """Update history records"""
        for db_id in db_ids:
            # update score history
            if db_id not in self.score_history:
                self.score_history[db_id] = []
            self.score_history[db_id].append(scores.get(db_id, 0))
            if len(self.score_history[db_id]) > self.history_size:
                self.score_history[db_id].pop(0)
            
            # update allocation history
            if db_id not in self.allocation_history:
                self.allocation_history[db_id] = []
            self.allocation_history[db_id].append(allocations.get(db_id, 0))
            if len(self.allocation_history[db_id]) > self.history_size:
                self.allocation_history[db_id].pop(0)
        
        self.last_allocations = allocations.copy()
    
    def _get_score_volatility(self, db_id: str) -> float:
        """Get score volatility"""
        if db_id not in self.score_history or len(self.score_history[db_id]) < 2:
            return 0.0
        
        scores = self.score_history[db_id]
        if len(scores) < 2:
            return 0.0
        
        # Calculate standard deviation
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std_dev = math.sqrt(variance)
        
        # Normalize (relative to mean)
        if mean > 0:
            return std_dev / mean
        return 0.0
    
    def _get_recent_allocation_change(self, db_id: str) -> float:
        """Get recent allocation changes"""
        if db_id not in self.allocation_history or len(self.allocation_history[db_id]) < 2:
            return 0.0
        
        history = self.allocation_history[db_id]
        if len(history) < 2:
            return 0.0
        
        # Calculate recent change rate
        recent_change = abs(history[-1] - history[-2])
        if history[-2] > 0:
            return recent_change / history[-2]
        return 0.0
    
    def _replace_exhausted_member(self, victim_db: str, allocations: Dict[str, int],
                                 scores: Dict[str, float]) -> Optional[str]:
        """Replace exhausted members"""
        if not self.candidate_pool:
            # self.logger.warning(f"candidate_pool is empty, cannot replace {victim_db}")  # Scalability test comment
            return None
        
        # select database with lowest score from candidate_pool (candidate_pool already sorted by score descending, so take last item)
        new_member = self.candidate_pool.pop()
        
        # Update sets
        self.active_set.remove(victim_db)
        self.active_set.add(new_member)
        self.supply_set.remove(victim_db)
        self.supply_set.add(new_member)
        
        # Initialize new member allocation (use current allocation or minimum value)
        allocations[new_member] = allocations.get(new_member, 1)  # at_least1 pages
        
        # Initialize new member velocity
        if new_member not in self.velocities:
            self.velocities[new_member] = 0.0
        
        # self.logger.info(f"Member replacement: {victim_db} (exhausted) → {new_member} (new supplier)")  # Scalability test comment
        
        return new_member
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics"""
        return {
            'total_iterations': self.total_iterations,
            'global_checks': self.global_checks,
            'active_set_updates': self.active_set_updates,
            'current_active_set_size': len(self.active_set),
            'iterations_since_global_check': self.iterations_since_global_check,
            'avg_iterations_per_call': self.total_iterations / max(1, self.global_checks)
        }
    
    def reset_state(self):
        """Reset internal state"""
        self.velocities.clear()
        self.active_set.clear()
        self.iterations_since_global_check = 0
        self.last_global_scores.clear()
        self.last_allocations.clear()
        self.score_history.clear()
        self.allocation_history.clear()
        # self.logger.info("Active set gradient allocator state reset"  # Scalability test comment
