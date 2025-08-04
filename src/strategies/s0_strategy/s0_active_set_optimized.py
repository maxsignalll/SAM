"""
S0 Active Set Management Module (Performance Optimized Version)

Optimization focus:
1. Simplify knee point detection algorithm, avoid NumPy operations
2. Reduce sorting operation overhead
3. Use more efficient data structures
"""

import logging
import heapq
import time
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass


@dataclass
class DatabaseState:
    """Database state data class"""
    db_id: str
    h_factor: float
    v_factor: float
    final_score: float
    saturation_confidence: float
    current_allocation: int
    min_allocation: int
    current_ops: float = 0.0  # Current operations, used for Activity-Weighted Convergence
    
    def get_available_pages(self) -> int:
        """Get allocatable pages (portion exceeding minimum value)"""
        return max(0, self.current_allocation - self.min_allocation)


class S0ActiveSetManagerOptimized:
    """S0 active set manager (optimized version)"""
    
    def __init__(self, hyperparams: Dict[str, Any], logger: logging.Logger):
        self.hyperparams = hyperparams
        self.logger = logger
        
        # Active set parameters
        self.active_set_params = {
            # FinalScore calculation parameters
            'positive_v_threshold': hyperparams.get('positive_v_threshold', 0.001),
            'hr_ratio_strict': hyperparams.get('hr_ratio_strict', 0.9),
            'base_alpha': hyperparams.get('base_alpha', 0.5),
            'saturation_alpha': hyperparams.get('saturation_alpha', 0.8),
            
            # Active set selection parameters
            'k_max': hyperparams.get('k_max', 10),  # Maximum candidate set size
            'cv_threshold': hyperparams.get('cv_threshold', 0.1),  # CV threshold, system is balanced if below this
            'min_demand_supply_ratio': hyperparams.get('min_demand_supply_ratio', 0.8),  # Demand-supply matching ratio
            
            # Convergence detection parameters
            'convergence_threshold': hyperparams.get('convergence_threshold', 0.01),  # Relative improvement threshold
            'stagnation_cycles': hyperparams.get('stagnation_cycles', 3),  # Stagnation detection cycle count
            'max_active_set_lifetime': hyperparams.get('max_active_set_lifetime', 20),  # Maximum active set lifetime
        }
        
        # State management
        self.saturation_confidences: Dict[str, float] = {}  # Saturation confidence
        self.active_set: Set[str] = set()  # Current active set
        self.active_set_age: int = 0  # Active set age
        self.convergence_tracker = {
            'best_score': 0.0,
            'stagnation_count': 0,
            'last_improvement_cycle': 0
        }
        self.db_states_cache: Dict[str, DatabaseState] = {}  # Database state cache
        
        # Global scan cache (for dynamic member flow)
        self.last_bottom_candidates: List[Tuple[float, str]] = []  # (score, db_id)
        
        # Performance optimization: pre-allocated list
        self._score_buffer = []
        
        # Performance timers
        self.perf_timers = {
            'fixed_allocation_sum': 0.0,
            'candidate_selection': 0.0,
            'cv_calculation': 0.0,
            'knee_detection': 0.0,
            'supply_calculation': 0.0,
            'supply_sorting': 0.0,
            'convergence_check': 0.0,
            'total_calls': 0
        }
        
        # Create performance log file for active set manager
        # Note: path handling simplified here, correct path will be set later during strategy initialization
        self.perf_log_path = None
        
    def calculate_final_score(self, db_id: str, h_factor: float, v_factor: float,
                             hr_current: float, hr_base: float, hr_max: float) -> float:
        """
        Calculate final score (including saturation confidence mechanism)
        
        Args:
            db_id: Database ID
            h_factor: horizontal factor (current efficiency)
            v_factor: vertical factor (marginal gain)
            hr_current: current hit rate
            hr_base: baseline hit rate
            hr_max: maximum hit rate
            
        Returns:
            final score
        """
        # Get or initialize saturation confidence
        if db_id not in self.saturation_confidences:
            self.saturation_confidences[db_id] = 0.0
        
        # Calculate hit rate ratio
        hr_ratio = (hr_current - hr_base) / (hr_max - hr_base) if hr_max > hr_base else 0.0
        
        # Update saturation confidence
        if abs(v_factor) < self.active_set_params['positive_v_threshold'] and \
           hr_ratio >= self.active_set_params['hr_ratio_strict']:
            # Meets saturation condition, confidence increases
            self.saturation_confidences[db_id] = min(1.0, 
                self.saturation_confidences[db_id] + 0.34)
        else:
            # Does not meet saturation condition, instant reset
            self.saturation_confidences[db_id] = 0.0
        
        # Calculate dynamic weight alpha_t
        base_alpha = self.active_set_params['base_alpha']
        saturation_alpha = self.active_set_params['saturation_alpha']
        confidence = self.saturation_confidences[db_id]
        
        alpha_t = base_alpha + (saturation_alpha - base_alpha) * confidence
        
        # Calculate final score
        final_score = alpha_t * h_factor + (1 - alpha_t) * v_factor
        
        return final_score
    
    def select_active_set(self, db_states: Dict[str, DatabaseState], 
                         total_pages: int = 5120) -> Set[str]:
        """
        Global scan to select active set (O(N) time complexity)
        
        Args:
            db_states: State information for all databases
            total_pages: System total pages
            
        Returns:
            New active set (database ID set)
        """
        self.perf_timers['total_calls'] += 1
        n = len(db_states)
        k_max = min(self.active_set_params['k_max'], n // 2)  # Not exceeding half of the total
        
        if n == 0:
            return set()
        
        # Calculate total fixed allocations
        t_start = time.perf_counter()
        total_fixed = sum(state.min_allocation for state in db_states.values())
        actual_elastic_pages = max(0, total_pages - total_fixed)
        self.perf_timers['fixed_allocation_sum'] += time.perf_counter() - t_start
        
        # If elastic pool is too small, return empty set directly
        if actual_elastic_pages < total_pages * 0.1:
            return set()
        
        # 3.2.1 Global score calculation - completed externally, use directly here
        
        # 3.2.2 Bidirectional candidate set filtering O(N log k_max)
        t_start = time.perf_counter()
        top_candidates = []  # Min heap, stores top k_max highest scores
        bottom_candidates = []  # Max heap, stores bottom k_max lowest scores (using negative scores)
        
        for db_id, state in db_states.items():
            score = state.final_score
            
            # Maintain top k_max
            if len(top_candidates) < k_max:
                heapq.heappush(top_candidates, (score, db_id))
            elif score > top_candidates[0][0]:
                heapq.heapreplace(top_candidates, (score, db_id))
            
            # Maintain bottom k_max (using negative scores)
            if len(bottom_candidates) < k_max:
                heapq.heappush(bottom_candidates, (-score, db_id))
            elif -score > bottom_candidates[0][0]:
                heapq.heapreplace(bottom_candidates, (-score, db_id))
        
        # Convert to list and sort
        top_list = sorted([(s, id) for s, id in top_candidates], reverse=True)
        bottom_list = sorted([(-s, id) for s, id in bottom_candidates])
        self.perf_timers['candidate_selection'] += time.perf_counter() - t_start
        
        # Cache bottom candidates for dynamic member flow
        self.last_bottom_candidates = bottom_list.copy()
        
        # 3.2.3 System equilibrium diagnosis
        t_start = time.perf_counter()
        all_candidates_scores = [s for s, _ in top_list] + [s for s, _ in bottom_list]
        if self._calculate_cv_fast(all_candidates_scores) < self.active_set_params['cv_threshold']:
            self.perf_timers['cv_calculation'] += time.perf_counter() - t_start
            # self.logger.info("System is in equilibrium state, skip this round of optimization")
            return set()
        self.perf_timers['cv_calculation'] += time.perf_counter() - t_start
        
        # 3.2.4 Dynamic k value determination (simplified knee point detection)
        t_start = time.perf_counter()
        k_demand = self._find_knee_point_fast([s for s, _ in top_list])
        if k_demand == 0:
            k_demand = 1  # Select at least one demand side
        self.perf_timers['knee_detection'] += time.perf_counter() - t_start
        
        # 3.2.5 Supply-demand matching active set construction
        sinks = [db_id for _, db_id in top_list[:k_demand]]
        
        # Calculate number of suppliers
        t_start = time.perf_counter()
        # Ensure active set has sufficient diversity
        if k_demand == 1:
            # Even with only one demand side, need sufficient suppliers
            k_supply = min(k_demand + 5, len(bottom_list))  # At least 6 members
        else:
            # Multi-hotspot scenario: estimate required number of suppliers
            # Assume each supplier can contribute 30% of current allocation on average
            avg_supply_ratio = 0.3
            estimated_supply_per_source = 0
            supply_candidates = []
            
            # Calculate average available supply
            for score, db_id in bottom_list[:min(10, len(bottom_list))]:
                if db_id not in sinks:
                    state = db_states[db_id]
                    available = state.get_available_pages()
                    if available > 0:
                        estimated_supply_per_source += available
                        supply_candidates.append((score, db_id, available))
            
            if len(supply_candidates) > 0:
                estimated_supply_per_source /= len(supply_candidates)
                # Calculate how many suppliers are needed
                # Use actual elastic pool instead of theoretical value
                k_supply = min(
                    int(actual_elastic_pages / max(estimated_supply_per_source, 10)),
                    k_demand * 3,  # Increase to 3x to ensure sufficient suppliers
                    len(supply_candidates)
                )
            else:
                k_supply = 0
        self.perf_timers['supply_calculation'] += time.perf_counter() - t_start
        
        # Select suppliers - prioritize databases with large available cache
        t_start = time.perf_counter()
        sources = []
        supply_count = 0
        
        # First sort candidates in bottom_list by available cache amount
        supply_candidates_sorted = []
        for score, db_id in bottom_list:
            if db_id not in sinks:
                state = db_states[db_id]
                available = state.get_available_pages()
                if available > 0:
                    supply_candidates_sorted.append((available, score, db_id))
        
        # Sort by available cache in descending order
        supply_candidates_sorted.sort(reverse=True)
        
        # Select suppliers
        for available, score, db_id in supply_candidates_sorted:
            sources.append(db_id)
            supply_count += 1
            
            if supply_count >= k_supply:
                break
        self.perf_timers['supply_sorting'] += time.perf_counter() - t_start
        
        # Build active set
        active_set = set(sinks + sources)
        
        # Write performance data
        self._write_performance_data(n)
        
        return active_set
    
    def optimize_in_active_set(self, active_set: Set[str], 
                              db_states: Dict[str, DatabaseState],
                              gradient_allocator: Any,
                              system_total_pages: int = None) -> Tuple[Dict[str, int], bool]:
        """
        Perform local optimization within active set
        
        Args:
            active_set: active set
            db_states: database states
            gradient_allocator: gradient allocator
            
        Returns:
            (new allocation, whether converged)
        """
        # Update active set
        if active_set != self.active_set:
            self.active_set = active_set
            self.active_set_age = 0
            self.convergence_tracker['stagnation_count'] = 0
        else:
            self.active_set_age += 1
        
        # Prepare scores within active set
        active_scores = {}
        for db_id in active_set:
            if db_id in db_states:
                active_scores[db_id] = db_states[db_id].final_score
        
        # 3.3.1 Gradient descent (only within active set)
        current_allocations = {db_id: db_states[db_id].current_allocation 
                              for db_id in active_set}
        fixed_allocations = {db_id: db_states[db_id].min_allocation 
                            for db_id in active_set}
        # If system total pages not provided, use sum of current allocations in active set
        if system_total_pages is not None:
            total_pages = system_total_pages
        else:
            total_pages = sum(db_states[db_id].current_allocation for db_id in active_set)
        
        new_allocations = gradient_allocator.calculate_allocations(
            list(active_set),
            active_scores,
            current_allocations,
            fixed_allocations,
            total_pages
        )
        
        # 3.3.2 Dynamic member flow (replacement mechanism)
        # Check if any suppliers are exhausted
        for db_id in list(active_set):
            if db_id in db_states:
                state = db_states[db_id]
                new_alloc = new_allocations.get(db_id, state.current_allocation)
                
                # If cache drops near minimum value
                if new_alloc <= state.min_allocation * 1.1:  # Leave 10% margin
                    # Remove from active set
                    active_set.remove(db_id)
                    # self.logger.info(f"Database{db_id}cache exhausted，Remove from active set")
                    
                    # Try to recruit new member
                    recruited = self._recruit_new_member(active_set, db_states)
                    if recruited:
                        pass  # self.logger.info(f"Recruit new member{recruited}joinedactive set")
        
        # 3.3.3 Convergence detection
        t_start = time.perf_counter()
        total_score = sum(active_scores.values())
        
        # Update db_states_cache reference (for convergence detection)
        self.db_states_cache = db_states
        
        is_converged = self._check_convergence(total_score)
        self.perf_timers['convergence_check'] += time.perf_counter() - t_start
        
        # Check if maximum lifetime is reached
        if self.active_set_age >= self.active_set_params['max_active_set_lifetime']:
            # self.logger.info(f"Active set reached maximum lifetime{self.active_set_age}, marked as converged")
            is_converged = True
        
        return new_allocations, is_converged
    
    def intelligent_adjust_total(self, allocations: Dict[str, int], 
                               db_states: Dict[str, DatabaseState],
                               target_total: int) -> Dict[str, int]:
        """
        Intelligently adjust total to ensure no constraint violations
        
        Args:
            allocations: Current allocation
            db_states: database states
            target_total: Target total pages
            
        Returns:
            Adjusted allocation
        """
        current_total = sum(allocations.values())
        delta = current_total - target_total
        
        if abs(delta) <= 1:  # Tolerate 1 page error
            return allocations
        
        adjusted = allocations.copy()
        
        if delta > 0:  # Need to reduce
            # First priority: reclaim from suppliers within active set
            active_sources = []
            for db_id in self.active_set:
                if db_id in db_states and db_states[db_id].final_score < 0.5:  # Simplified judgment
                    available = adjusted[db_id] - db_states[db_id].min_allocation
                    if available > 0:
                        active_sources.append((db_states[db_id].final_score, db_id, available))
            
            # Sort by score in ascending order
            active_sources.sort()
            
            # Start reclaiming from lowest scores
            for score, db_id, available in active_sources:
                if delta <= 0:
                    break
                reclaim = min(delta, available)
                adjusted[db_id] -= reclaim
                delta -= reclaim
            
            # Second priority: reclaim from global bottom candidates
            if delta > 0 and self.last_bottom_candidates:
                for score, db_id in self.last_bottom_candidates:
                    if delta <= 0:
                        break
                    if db_id not in self.active_set and db_id in adjusted:
                        available = adjusted[db_id] - db_states[db_id].min_allocation
                        if available > 0:
                            reclaim = min(delta, available)
                            adjusted[db_id] -= reclaim
                            delta -= reclaim
        
        else:  # Need to increase (delta < 0)
            # Allocate to demand side within active set
            active_demands = []
            for db_id in self.active_set:
                if db_id in db_states and db_states[db_id].final_score > 0.5:
                    active_demands.append((db_states[db_id].final_score, db_id))
            
            # Sort by score in descending order
            active_demands.sort(reverse=True)
            
            # Average allocation
            if active_demands:
                per_db = abs(delta) // len(active_demands)
                remainder = abs(delta) % len(active_demands)
                
                for i, (score, db_id) in enumerate(active_demands):
                    addition = per_db + (1 if i < remainder else 0)
                    adjusted[db_id] += addition
        
        return adjusted
    
    def _calculate_cv_fast(self, values: List[float]) -> float:
        """Fast calculation of coefficient of variation (avoid NumPy)"""
        if len(values) < 2:
            return 0.0
        
        # Calculate mean
        mean_val = sum(values) / len(values)
        if abs(mean_val) < 1e-9:
            return 0.0
        
        # Calculate standard deviation
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        std_val = variance ** 0.5
        
        return std_val / abs(mean_val)
    
    def _find_knee_point_fast(self, scores: List[float]) -> int:
        """
        Simplified knee point detection (avoid NumPy operations)
        
        Args:
            scores: Score list in descending order
            
        Returns:
            Knee point position (number of demand sides)
        """
        n = len(scores)
        if n <= 2:
            return n
        
        # Use simplified method: find maximum slope change point
        max_slope_change = 0
        knee_index = 0
        
        # Calculate slope changes between adjacent points
        for i in range(1, n - 1):
            # Before and after slopes
            slope_before = scores[i-1] - scores[i]
            slope_after = scores[i] - scores[i+1]
            
            # Slope change rate
            slope_change = abs(slope_after - slope_before)
            
            if slope_change > max_slope_change:
                max_slope_change = slope_change
                knee_index = i
        
        # Return knee point position (select at least 1)
        return max(1, knee_index + 1)
    
    def _recruit_new_member(self, active_set: Set[str], 
                           db_states: Dict[str, DatabaseState]) -> Optional[str]:
        """
        Recruit new member from global bottom candidate set
        
        Args:
            active_set: Current active set
            db_states: database states
            
        Returns:
            New member ID, return None if no suitable candidate
        """
        for score, db_id in self.last_bottom_candidates:
            if db_id not in active_set and db_id in db_states:
                state = db_states[db_id]
                if state.get_available_pages() > 0:
                    active_set.add(db_id)
                    return db_id
        
        return None
    
    def _check_convergence(self, total_score: float) -> bool:
        """
        Check if active set converged - using Activity-Weighted Convergence
        
        Args:
            total_score: Active set total score
            
        Returns:
            Whether converged
        """
        tracker = self.convergence_tracker
        
        # Calculate relative improvement
        if tracker['best_score'] > 0:
            relative_improvement = (total_score - tracker['best_score']) / tracker['best_score']
        else:
            relative_improvement = 1.0  # First time always considered as improvement
        
        # Calculate total operations in active set
        total_ops = sum(state.current_ops for state in self.db_states_cache.values() 
                       if state.db_id in self.active_set)
        
        # Calculate weighted improvement (Activity-Weighted Improvement)
        weighted_improvement = relative_improvement * total_ops
        
        # Update best score
        if total_score > tracker['best_score']:
            tracker['best_score'] = total_score
            tracker['last_improvement_cycle'] = self.active_set_age
            tracker['stagnation_count'] = 0
        else:
            tracker['stagnation_count'] += 1
        
        # Use weighted threshold for convergence judgment
        # Low-traffic active set needs smaller weighted improvement to converge
        # High-traffic active set needs larger weighted improvement to converge
        # Dynamic threshold: based on percentage of total OPS
        weighted_threshold = self.active_set_params['convergence_threshold'] * max(10.0, total_ops * 0.1)
        
        # DetermineWhether converged
        if abs(weighted_improvement) < weighted_threshold and \
           tracker['stagnation_count'] >= self.active_set_params['stagnation_cycles']:
            return True
        
        return False
    
    def _write_performance_data(self, db_count: int):
        """Write performance data to CSV file"""
        if not self.perf_log_path:
            return  # If log path not set, skip writing
        
        try:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            call_num = self.perf_timers['total_calls']
            
            # Calculate total time
            total_time = sum(v for k, v in self.perf_timers.items() if k != 'total_calls')
            total_ms = total_time * 1000
            
            # Calculate for each operationAveragetime spent（milliseconds）
            if call_num > 0:
                fixed_alloc_ms = self.perf_timers['fixed_allocation_sum'] * 1000 / call_num
                candidate_ms = self.perf_timers['candidate_selection'] * 1000 / call_num
                cv_calc_ms = self.perf_timers['cv_calculation'] * 1000 / call_num
                knee_ms = self.perf_timers['knee_detection'] * 1000 / call_num
                supply_calc_ms = self.perf_timers['supply_calculation'] * 1000 / call_num
                supply_sort_ms = self.perf_timers['supply_sorting'] * 1000 / call_num
                convergence_ms = self.perf_timers['convergence_check'] * 1000 / call_num
            else:
                fixed_alloc_ms = candidate_ms = cv_calc_ms = knee_ms = 0
                supply_calc_ms = supply_sort_ms = convergence_ms = 0
            
            # Write to CSV
            with open(self.perf_log_path, 'a') as f:
                f.write(f"{timestamp},{call_num},{total_ms:.3f},{fixed_alloc_ms:.3f},"
                       f"{candidate_ms:.3f},{cv_calc_ms:.3f},{knee_ms:.3f},"
                       f"{supply_calc_ms:.3f},{supply_sort_ms:.3f},{convergence_ms:.3f},{db_count}\n")
                f.flush()  # Ensure immediate disk write
        except Exception as e:
            self.logger.error(f"Active set manager failed to write performance data: {e}")
    
    def print_performance_report(self):
        """Print performance analysis report"""
        if self.perf_timers['total_calls'] == 0:
            return
            
        print("\n=== Active set manager performance analysis report ===")
        print(f"Total call count: {self.perf_timers['total_calls']}")
        
        total_time = sum(v for k, v in self.perf_timers.items() if k != 'total_calls')
        
        print("\nOperation time cost (milliseconds):")
        for op, time_sec in sorted(self.perf_timers.items(), key=lambda x: x[1], reverse=True):
            if op == 'total_calls':
                continue
            time_ms = time_sec * 1000
            avg_ms = time_ms / self.perf_timers['total_calls']
            pct = (time_sec / total_time * 100) if total_time > 0 else 0
            print(f"  {op:20}: {time_ms:8.2f}ms Total, {avg_ms:6.3f}ms Average ({pct:5.1f}%)")
        
        print(f"\nTotal time: {total_time * 1000:.2f}ms")
        print(f"Averageper call: {total_time * 1000 / self.perf_timers['total_calls']:.3f}ms")
    
    def reset_for_new_epoch(self):
        """Reset state (for new epoch start)"""
        self.active_set.clear()
        self.active_set_age = 0
        self.convergence_tracker = {
            'best_score': 0.0,
            'stagnation_count': 0,
            'last_improvement_cycle': 0
        }