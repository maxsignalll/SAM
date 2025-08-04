"""
S0 Active Set Management Module

Implements the core mechanisms of S0 adaptive active set strategy, including:
1. FinalScore calculation (including saturation confidence)
2. Global scan to select active set
3. Local optimization
4. Intelligent total constraint

Based on docs/S0_Active_Set_Final_Design_V3.md implementation
"""

import logging
import heapq
import numpy as np
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


class S0ActiveSetManager:
    """S0 active set manager"""
    
    def __init__(self, hyperparams: Dict[str, Any], logger: logging.Logger):
        self.hyperparams = hyperparams
        self.logger = logger
        
        # Active set parameters
        self.active_set_params = {
            # FinalScorecalculation parameters
            'positive_v_threshold': hyperparams.get('positive_v_threshold', 0.001),
            'hr_ratio_strict': hyperparams.get('hr_ratio_strict', 0.9),
            'base_alpha': hyperparams.get('base_alpha', 0.5),
            'saturation_alpha': hyperparams.get('saturation_alpha', 0.8),
            
            # Active set selection parameters
            'k_max': hyperparams.get('k_max', 10),  # Maximum candidate set size
            'cv_threshold': hyperparams.get('cv_threshold', 0.1),  # CVThreshold, below which system is considered balanced
            'min_demand_supply_ratio': hyperparams.get('min_demand_supply_ratio', 0.8),  # Supply-demand matching ratio
            
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
        
    def calculate_final_score(self, db_id: str, h_factor: float, v_factor: float,
                             hr_current: float, hr_base: float, hr_max: float) -> float:
        """
        Calculate final score (including saturation confidence mechanism)
        
        Args:
            db_id: DatabaseID
            h_factor: Horizontal factor (current efficiency)
            v_factor: Vertical factor (marginal gain)
            hr_current: Current hit rate
            hr_base: Baseline hit rate
            hr_max: Maximum hit rate
            
        Returns:
            Final score
        """
        # Get or initializeSaturation confidence
        if db_id not in self.saturation_confidences:
            self.saturation_confidences[db_id] = 0.0
        
        # Calculate hit rate ratio
        hr_ratio = (hr_current - hr_base) / (hr_max - hr_base) if hr_max > hr_base else 0.0
        
        # UpdateSaturation confidence
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
        
        # CalculateFinal score
        final_score = alpha_t * h_factor + (1 - alpha_t) * v_factor
        
        # Log debug information
        if db_id in ['db_high_priority', 'db_medium_priority']:
            self.logger.debug(f"FinalScore[{db_id}]: H={h_factor:.3f}, V={v_factor:.3f}, "
                            f"HR_ratio={hr_ratio:.3f}, SatConf={confidence:.2f}, "
                            f"alpha_t={alpha_t:.3f}, Score={final_score:.3f}")
        
        return final_score
    
    def select_active_set(self, db_states: Dict[str, DatabaseState], 
                         total_pages: int = 1280) -> Set[str]:
        """
        Global scan to select active set (O(N) time complexity)
        
        Args:
            db_states: State information of all databases
            total_pages: Total system pages
            
        Returns:
            New active set (set of database IDs)
        """
        n = len(db_states)
        k_max = min(self.active_set_params['k_max'], n // 2)  # Not more than half of total
        
        if n == 0:
            return set()
        
        # 3.2.1 Global score calculation - already done externally, used directly here
        
        # 3.2.2 Bidirectional candidate set filtering O(N log k_max)
        top_candidates = []  # Min heap, stores top k_max highest scores
        bottom_candidates = []  # Max heap, stores top k_max lowest scores (using negative scores)
        
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
        
        # Cache bottom candidates for dynamic member flow
        self.last_bottom_candidates = bottom_list.copy()
        
        # 3.2.3 System balance diagnosis
        all_candidates_scores = [s for s, _ in top_list] + [s for s, _ in bottom_list]
        if self._calculate_cv(all_candidates_scores) < self.active_set_params['cv_threshold']:
            self.logger.info("System is in balanced state, skipping this optimization round")
            return set()
        
        # 3.2.4 Dynamic k value determination (knee point detection)
        k_demand = self._find_knee_point([s for s, _ in top_list])
        if k_demand == 0:
            k_demand = 1  # Select at least one demander
        
        # 3.2.5 Supply-demand matching active set construction
        sinks = [db_id for _, db_id in top_list[:k_demand]]
        
        # Adaptive supply strategy: dynamically adjust based on number of demanders
        # Important fix: need to consider actual available elastic pool, not theoretical value
        # Actual elastic pool = total pages - fixed allocations of all databases
        total_fixed = sum(state.min_allocation for state in db_states.values())
        actual_elastic_pages = max(0, total_pages - total_fixed)
        
        self.logger.info(f"Active set selection: total_pages={total_pages}, fixed_allocation={total_fixed}, "
                        f"actual_elastic_pool={actual_elastic_pages} pages")
        
        # Calculate number of suppliers based on number of demanders
        # Ensure active set has sufficient diversity
        if k_demand == 1:
            # Even with only one demander, need sufficient suppliers
            k_supply = min(k_demand + 5, len(bottom_list))  # At least 6 members
        else:
            # Multi-hotspot scenario: estimate required number of suppliers
            # Assume each supplier can contribute 30% of its current allocation on average
            avg_supply_ratio = 0.3
            estimated_supply_per_source = 0
            supply_candidates = []
            
            # Calculate average supply amount
            for score, db_id in bottom_list[:min(10, len(bottom_list))]:
                if db_id not in sinks:
                    state = db_states[db_id]
                    available = state.get_available_pages()
                    if available > 0:
                        estimated_supply_per_source += available
                        supply_candidates.append((score, db_id, available))
            
            if len(supply_candidates) > 0:
                estimated_supply_per_source /= len(supply_candidates)
                # Calculate how many suppliers needed
                # Use actual elastic pool rather than theoretical value
                k_supply = min(
                    int(actual_elastic_pages / max(estimated_supply_per_source, 10)),
                    k_demand * 3,  # Increase to 3x to ensure sufficient suppliers
                    len(supply_candidates)
                )
            else:
                k_supply = 0
        
        # Select suppliers - prioritize databases with large available cache
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
        
        # If insufficient suppliers, try to expand selection range
        if supply_count < k_supply:
            self.logger.warning(f"Insufficient suppliers: found only {supply_count}, target is {k_supply}")
        
        # Build active set
        active_set = set(sinks + sources)
        
        # Calculate total available cache of active set
        active_set_available = sum(db_states[db_id].get_available_pages() 
                                 for db_id in sources if db_id in db_states)
        
        self.logger.info(f"Active set selected: {len(sinks)} demanders, {len(sources)} suppliers, "
                        f"total {len(active_set)} databases")
        self.logger.info(f"Demanders: {sinks[:3]}...")
        self.logger.info(f"Suppliers: {sources[:5]}... (total available cache: {active_set_available} pages)")
        
        return active_set
    
    def optimize_in_active_set(self, active_set: Set[str], 
                              db_states: Dict[str, DatabaseState],
                              gradient_allocator: Any,
                              system_total_pages: int = None) -> Tuple[Dict[str, int], bool]:
        """
        Perform local optimization within the active set
        
        Args:
            active_set: Active set
            db_states: Database states
            gradient_allocator: Gradient allocator
            
        Returns:
            (new allocations, whether converged)
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
        # If system total pages not provided, use current allocation sum of active set
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
        # Check if any supplier is exhausted
        for db_id in list(active_set):
            if db_id in db_states:
                state = db_states[db_id]
                new_alloc = new_allocations.get(db_id, state.current_allocation)
                
                # If cache drops near minimum value
                if new_alloc <= state.min_allocation * 1.1:  # Keep 10% margin
                    # Remove from active set
                    active_set.remove(db_id)
                    self.logger.info(f"Database {db_id} cache exhausted, removed from active set")
                    
                    # Try to recruit new member
                    recruited = self._recruit_new_member(active_set, db_states)
                    if recruited:
                        self.logger.info(f"Recruited new member {recruited} to active set")
        
        # 3.3.3 Convergence detection
        total_score = sum(active_scores.values())
        
        # Update db_states_cache reference (for convergence detection)
        self.db_states_cache = db_states
        
        is_converged = self._check_convergence(total_score)
        
        # Check if reached maximum lifetime
        if self.active_set_age >= self.active_set_params['max_active_set_lifetime']:
            self.logger.info(f"Active set reached maximum lifetime {self.active_set_age}, marked as converged")
            is_converged = True
        
        return new_allocations, is_converged
    
    def intelligent_adjust_total(self, allocations: Dict[str, int], 
                               db_states: Dict[str, DatabaseState],
                               target_total: int) -> Dict[str, int]:
        """
        Intelligently adjust total to ensure constraints are not violated
        
        Args:
            allocations: Current allocations
            db_states: Database states
            target_total: Target total pages
            
        Returns:
            Adjusted allocations
        """
        current_total = sum(allocations.values())
        delta = current_total - target_total
        
        if abs(delta) <= 1:  # Tolerate 1 page error
            return allocations
        
        adjusted = allocations.copy()
        
        if delta > 0:  # Need to reduce
            # First priority: reclaim from suppliers in active set
            active_sources = []
            for db_id in self.active_set:
                if db_id in db_states and db_states[db_id].final_score < 0.5:  # Simplified check
                    available = adjusted[db_id] - db_states[db_id].min_allocation
                    if available > 0:
                        active_sources.append((db_states[db_id].final_score, db_id, available))
            
            # Sort by score in ascending order
            active_sources.sort()
            
            # Start reclaiming from lowest score
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
            # Allocate to demanders in active set
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
    
    def _calculate_cv(self, values: List[float]) -> float:
        """Calculate coefficient of variation"""
        if len(values) < 2:
            return 0.0
        
        mean_val = np.mean(values)
        if abs(mean_val) < 1e-9:
            return 0.0
        
        std_val = np.std(values)
        return std_val / abs(mean_val)
    
    def _find_knee_point(self, scores: List[float]) -> int:
        """
        Find knee point using Menger curvature method
        
        Args:
            scores: List of scores in descending order
            
        Returns:
            Knee point position (number of demanders)
        """
        n = len(scores)
        if n <= 2:
            return n
        
        # Remove single hotspot detection, let knee point detection work naturally
        # This avoids the problem of active set being too small
        
        # Normalize coordinates
        x = np.arange(n) / (n - 1) if n > 1 else np.array([0])
        y = (np.array(scores) - min(scores)) / (max(scores) - min(scores) + 1e-9)
        
        # Calculate curvature for each point
        max_curvature = 0
        knee_index = 0
        
        for i in range(1, n - 1):
            # Calculate Menger curvature using three points
            p1 = np.array([x[i-1], y[i-1]])
            p2 = np.array([x[i], y[i]])
            p3 = np.array([x[i+1], y[i+1]])
            
            # Calculate triangle area
            area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - 
                           (p3[0] - p1[0]) * (p2[1] - p1[1]))
            
            # Calculate three side lengths
            a = np.linalg.norm(p2 - p1)
            b = np.linalg.norm(p3 - p2)
            c = np.linalg.norm(p3 - p1)
            
            # Menger curvature
            if a * b * c > 1e-9:
                curvature = 4 * area / (a * b * c)
                
                if curvature > max_curvature:
                    max_curvature = curvature
                    knee_index = i
        
        # Return knee point position (at least 1)
        return max(1, knee_index + 1)
    
    def _recruit_new_member(self, active_set: Set[str], 
                           db_states: Dict[str, DatabaseState]) -> Optional[str]:
        """
        Recruit new member from global bottom candidates
        
        Args:
            active_set: Current active set
            db_states: Database states
            
        Returns:
            New member ID, None if no suitable candidate
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
        Check if active set has converged - using Activity-Weighted Convergence
        
        Args:
            total_score: Total score of active set
            
        Returns:
            Whether converged
        """
        tracker = self.convergence_tracker
        
        # Calculate relative improvement
        if tracker['best_score'] > 0:
            relative_improvement = (total_score - tracker['best_score']) / tracker['best_score']
        else:
            relative_improvement = 1.0  # First time always considered as improvement
        
        # Calculate total operations of active set
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
        # Low traffic active set needs smaller weighted improvement to converge
        # High traffic active set needs larger weighted improvement to converge
        # Dynamic threshold: based on percentage of total OPS
        weighted_threshold = self.active_set_params['convergence_threshold'] * max(10.0, total_ops * 0.1)
        
        # Determine if converged
        if abs(weighted_improvement) < weighted_threshold and \
           tracker['stagnation_count'] >= self.active_set_params['stagnation_cycles']:
            self.logger.info(f"Active set converged: relative_improvement={relative_improvement:.3f}, "
                           f"weighted_improvement={weighted_improvement:.1f}, total_ops={total_ops:.0f}, "
                           f"stagnation_count={tracker['stagnation_count']} cycles")
            return True
        
        # Log debug information
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Convergence check: total_score={total_score:.3f}, relative_improvement={relative_improvement:.3f}, "
                            f"weighted_improvement={weighted_improvement:.1f}, total_ops={total_ops:.0f}, "
                            f"stagnation_count={tracker['stagnation_count']}")
        
        return False
    
    def reset_for_new_epoch(self):
        """Reset state (for new epoch start)"""
        self.active_set.clear()
        self.active_set_age = 0
        self.convergence_tracker = {
            'best_score': 0.0,
            'stagnation_count': 0,
            'last_improvement_cycle': 0
        }
        self.logger.info("Active set manager reset")