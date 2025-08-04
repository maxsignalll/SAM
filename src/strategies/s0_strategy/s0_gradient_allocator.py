"""
S0 Strategy Gradient Descent Cache Allocator

Core Ideas:
1. Treat cache allocation as optimization problem: maximize overall performance
2. Use scores as gradient estimates of performance
3. Small step adjustments to avoid oscillation
4. Momentum mechanism for smooth updates
"""

import logging
from typing import Dict, Any, Tuple
import math


class S0GradientAllocator:
    """Gradient descent based cache allocator"""
    
    def __init__(self, strategy: Any, orchestrator: Any, hyperparams: Dict[str, Any], logger: logging.Logger):
        self.strategy = strategy  # Save reference to strategy object
        self.orchestrator = orchestrator
        self.hyperparams = hyperparams
        self.logger = logger
        
        # Gradient descent parameters
        self.learning_rate = hyperparams.get('gradient_learning_rate', 0.1)
        self.momentum = hyperparams.get('gradient_momentum', 0.7)
        self.min_step_pages = hyperparams.get('gradient_min_step_pages', 5)
        self.max_step_pages = hyperparams.get('gradient_max_step_pages', 100)
        
        # Simplified parameters - only keep basic learning rate decay
        
        # Initialize velocity vectors (adjustment speed for each database)
        self.velocities = {}
        
        # Epoch manager reference (will be passed from strategy)
        self.epoch_manager = None
        
    def calculate_allocations(self, 
                            db_ids: list,
                            scores: Dict[str, float], 
                            current_allocations: Dict[str, int],
                            fixed_allocations: Dict[str, int],
                            total_pages: int) -> Dict[str, int]:
        """
        Calculate new cache allocations based on gradient descent
        
        Args:
            db_ids: List of database IDs
            scores: Composite scores for each database (H*α + V*(1-α))
            current_allocations: Current cache allocations
            fixed_allocations: Fixed cache allocations (lower bounds)
            total_pages: Total cache pages
            
        Returns:
            New cache allocation scheme
        """
        # Initialize velocities for new databases
        for db_id in db_ids:
            if db_id not in self.velocities:
                self.velocities[db_id] = 0.0
        
        # Calculate gradients (based on relative score differences)
        gradients = self._calculate_gradients(db_ids, scores, current_allocations)
        
        # Output gradient information
        self.logger.debug(f"Gradient calculation results:")
        for db_id in db_ids:
            self.logger.debug(f"  {db_id}: gradient={gradients.get(db_id, 0):.4f}")
        
        # Calculate learning rate
        effective_lr = self._calculate_learning_rate()
        
        # Update velocities (momentum) - with direction-aware asymmetric handling
        new_velocities = {}
        for db_id in db_ids:
            old_velocity = self.velocities.get(db_id, 0.0)
            gradient = gradients.get(db_id, 0.0)
            
            # Check if this is a cache reduction case
            current_alloc = current_allocations.get(db_id, 0)
            fixed_alloc = fixed_allocations.get(db_id, 0)
            
            if gradient < 0 and current_alloc > fixed_alloc:
                # Negative gradient, need to reduce cache
                ratio_above_fixed = current_alloc / fixed_alloc if fixed_alloc > 0 else 2.0
                
                if ratio_above_fixed < 1.5:  # Close to fixed allocation
                    # Reduce momentum to avoid over-reduction
                    effective_momentum = self.momentum * 0.3
                    self.logger.debug(f"DB[{db_id}] near minimum allocation (ratio={ratio_above_fixed:.2f}), reducing momentum")
                else:
                    # Normal momentum
                    effective_momentum = self.momentum
            else:
                # Positive gradient or already at fixed allocation, use normal momentum
                effective_momentum = self.momentum
            
            # Momentum update: v_t = β * v_{t-1} + (1-β) * g_t
            new_velocity = effective_momentum * old_velocity + (1 - effective_momentum) * gradient
            new_velocities[db_id] = new_velocity
        
        # Calculate elastic pool size
        elastic_pool_size = sum([
            self.orchestrator.db_current_page_allocations.get(db_id, 0) - 
            self.strategy.s0_fixed_allocations.get(db_id, 0)
            for db_id in db_ids
        ])
        
        # Calculate adjustments
        adjustments = self._calculate_adjustments(db_ids, new_velocities, effective_lr, elastic_pool_size)
        
        # Apply adjustments and ensure constraints
        new_allocations = self._apply_adjustments(
            db_ids, current_allocations, adjustments, fixed_allocations, total_pages
        )
        
        # Update internal state
        self.velocities = new_velocities
        
        # Log adjustment information
        self._log_allocation_changes(current_allocations, new_allocations, gradients, new_velocities, scores, adjustments)
        
        return new_allocations
    
    def _calculate_gradients(self, db_ids: list, scores: Dict[str, float], 
                           current_allocations: Dict[str, int]) -> Dict[str, float]:
        """
        Calculate gradients: based on elastic pool allocation ratio differences
        
        Gradient = (target ratio - current elastic pool ratio)
        """
        total_score = sum(scores.values())
        
        # Calculate current allocations in elastic pool
        elastic_allocations = {}
        for db_id in db_ids:
            current = current_allocations.get(db_id, 0)
            fixed = self.strategy.s0_fixed_allocations.get(db_id, 0)
            elastic_allocations[db_id] = max(0, current - fixed)
        
        total_elastic = sum(elastic_allocations.values())
        
        if total_score <= 0 or total_elastic <= 0:
            # If all scores are 0 or no elastic pool, return 0 gradients (maintain status quo)
            self.logger.debug("Gradient calculation: all scores are 0 or elastic pool is empty, returning 0 gradients")
            return {db_id: 0.0 for db_id in db_ids}
        
        gradients = {}
        for db_id in db_ids:
            # Target ratio (based on scores)
            target_ratio = scores.get(db_id, 0) / total_score
            
            # Current elastic pool ratio
            current_ratio = elastic_allocations[db_id] / total_elastic
            
            # Gradient = target - current
            gradient = target_ratio - current_ratio
            gradients[db_id] = gradient
        
        # Normalize large gradients to avoid aggressive adjustments
        max_gradient = max(abs(g) for g in gradients.values()) if gradients else 1.0
        if max_gradient > 0.75:  # Normalize when gradient is too large
            factor = 0.75 / max_gradient
            gradients = {k: v * factor for k, v in gradients.items()}
            self.logger.debug(f"Gradient normalization: max_gradient={max_gradient:.3f} > 0.75, scaling factor={factor:.3f}")
            
        return gradients
    
    def _calculate_learning_rate(self) -> float:
        """
        Calculate standard time-decayed learning rate
        """
        base_lr = self.learning_rate
        
        # Standard time decay
        if self.epoch_manager:
            steps = self.epoch_manager.steps_in_epoch
            # Use standard learning rate decay formula
            effective_lr = base_lr / (1.0 + 0.1 * steps)
            
            if steps > 0 and steps % 10 == 0:  # Output every 10 steps
                self.logger.debug(f"Learning rate decay: {base_lr:.3f} -> {effective_lr:.3f} (steps={steps})")
        else:
            effective_lr = base_lr
            
        return effective_lr
    
    def _calculate_adjustments(self, db_ids: list, velocities: Dict[str, float], 
                             learning_rate: float, elastic_pool_size: int) -> Dict[str, int]:
        """
        Calculate actual page adjustments based on velocity and learning rate
        """
        adjustments = {}
        
        for db_id in db_ids:
            velocity = velocities.get(db_id, 0.0)
            
            # Dynamic step size: based on velocity magnitude, 2%-20% of elastic pool
            velocity_magnitude = abs(velocity)
            dynamic_step_ratio = min(0.2, max(0.02, velocity_magnitude))
            
            # Adjustment = learning rate * velocity direction * dynamic step * elastic pool size
            raw_adjustment = learning_rate * math.copysign(dynamic_step_ratio, velocity) * elastic_pool_size
            
            # Limit adjustment range
            if abs(raw_adjustment) < self.min_step_pages:
                page_adjustment = 0
            else:
                page_adjustment = int(math.copysign(
                    min(abs(raw_adjustment), self.max_step_pages),
                    raw_adjustment
                ))
            
            adjustments[db_id] = page_adjustment
            
            # Debug information
            if abs(page_adjustment) > 0:
                self.logger.debug(f"DB[{db_id}] adjustment: velocity={velocity:.3f}, "
                                f"step_ratio={dynamic_step_ratio:.3f}, lr={learning_rate:.3f}, "
                                f"adjustment={page_adjustment}")
            
        return adjustments
    
    def _apply_adjustments(self, db_ids: list, current_allocations: Dict[str, int],
                          adjustments: Dict[str, int], fixed_allocations: Dict[str, int],
                          total_pages: int) -> Dict[str, int]:
        """
        Apply adjustments and ensure constraints are satisfied (with asymmetric protection)
        """
        new_allocations = current_allocations.copy()
        
        # First round: apply adjustments (with ratchet protection)
        for db_id in db_ids:
            current = current_allocations.get(db_id, 0)
            adjustment = adjustments.get(db_id, 0)
            min_alloc = fixed_allocations.get(db_id, 0)
            
            # Apply adjustment but ensure not below minimum
            proposed = current + adjustment
            
            # Ratchet protection: if reducing cache, check if reasonable
            if adjustment < 0 and current > min_alloc:
                # Check recent performance trend
                recent_velocity = self.velocities.get(db_id, 0)
                
                # If velocity history shows sustained improvement, delay reduction
                if recent_velocity > 0.1:
                    # Performance is improving, limit reduction amount
                    max_decrease = max(5, int(current * 0.05))  # Maximum 5% or 5 pages
                    actual_adjustment = max(adjustment, -max_decrease)
                    proposed = current + actual_adjustment
                    if actual_adjustment != adjustment:
                        self.logger.debug(f"DB[{db_id}] ratchet protection: limited reduction from {adjustment} to {actual_adjustment}")
            
            new_allocations[db_id] = max(min_alloc, proposed)
        
        # Second round: ensure correct total
        total_allocated = sum(new_allocations.values())
        diff = total_allocated - total_pages
        
        if abs(diff) > 0:
            # Need to adjust to satisfy total constraint
            self._rebalance_allocations(db_ids, new_allocations, fixed_allocations, diff, total_pages)
        
        return new_allocations
    
    def _rebalance_allocations(self, db_ids: list, allocations: Dict[str, int],
                              fixed_allocations: Dict[str, int], excess: int, total_pages: int):
        """
        Rebalance allocations to satisfy total constraint
        """
        # Protection check: if no databases, return directly
        if not db_ids:
            return
        if excess > 0:
            # Need to reduce allocations
            # Reduce proportionally to excess over fixed allocations
            reducible = {
                db_id: allocations[db_id] - fixed_allocations.get(db_id, 0)
                for db_id in db_ids
                if allocations[db_id] > fixed_allocations.get(db_id, 0)
            }
            
            total_reducible = sum(reducible.values())
            if total_reducible > 0:
                for db_id, reducible_amount in reducible.items():
                    reduction = int(excess * reducible_amount / total_reducible)
                    allocations[db_id] -= reduction
                    
        else:
            # Need to increase allocations
            # Increase proportionally to current allocations
            total_current = sum(allocations.values())
            if total_current > 0:
                for db_id in db_ids:
                    increase = int(-excess * allocations[db_id] / total_current)
                    allocations[db_id] += increase
        
        # Final fine-tuning
        final_total = sum(allocations.values())
        if final_total != total_pages:
            # Find the largest database for fine-tuning
            max_db = max(db_ids, key=lambda x: allocations[x])
            allocations[max_db] += total_pages - final_total
    
    def _log_allocation_changes(self, old_allocations: Dict[str, int], 
                               new_allocations: Dict[str, int],
                               gradients: Dict[str, float],
                               velocities: Dict[str, float],
                               scores: Dict[str, float],
                               adjustments: Dict[str, int]):
        """
        Log allocation change information
        """
        significant_changes = []
        
        for db_id, new_alloc in new_allocations.items():
            old_alloc = old_allocations.get(db_id, 0)
            change = new_alloc - old_alloc
            
            if abs(change) >= 5:  # Significant changes
                gradient = gradients.get(db_id, 0)
                velocity = velocities.get(db_id, 0)
                
                change_info = (
                    f"{db_id}: {old_alloc}→{new_alloc} ({change:+d} pages) "
                    f"[gradient={gradient:+.3f}, velocity={velocity:+.3f}]"
                )
                significant_changes.append(change_info)
        
        if significant_changes:
            self.logger.info(f"🎯 Gradient allocation update: {len(significant_changes)} significant changes")
            for change in significant_changes:
                self.logger.info(f"  {change}")
        else:
            self.logger.info("📊 Gradient allocation: no significant changes")
            
        # Always output some debug information
        self.logger.debug(f"Gradient allocation summary: total_score={sum(scores.values()):.3f}, "
                         f"max_gradient={max(abs(g) for g in gradients.values()) if gradients else 0:.3f}, "
                         f"adjustments={len([a for a in adjustments.values() if a != 0])}")
        # Output detailed information for debugging
        for db_id in new_allocations:
            if db_id in old_allocations:
                change = new_allocations[db_id] - old_allocations.get(db_id, 0)
                if abs(change) > 0:
                    self.logger.debug(f"  {db_id}: {old_allocations[db_id]}→{new_allocations[db_id]} ({change:+d} pages) [below significance threshold 5 pages]")
            
    
    def reset_state(self):
        """Reset internal state"""
        self.velocities.clear()
        self.logger.info("Gradient allocator state has been reset")