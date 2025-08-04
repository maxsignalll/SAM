"""
Optimized version of S0 Epoch Manager
Ensures identical behavior with significant performance improvements
Main optimizations:
1. Use numpy instead of statistics library
2. Cache computation results
3. Reduce redundant calculations
"""

import time
import numpy as np
from typing import Dict, Tuple, Optional


class S0EpochManagerOptimized:
    """Epoch Manager - Optimized Version"""
    
    def __init__(self, hyperparams: Dict, logger):
        self.hyperparams = hyperparams
        self.logger = logger
        
        # Epoch parameters
        self.champion_change_threshold = hyperparams.get('epoch_champion_change_threshold', 3)
        self.h_variance_threshold = hyperparams.get('epoch_h_variance_threshold', 0.3)
        self.h_sum_threshold = hyperparams.get('epoch_h_sum_threshold', 0.2)
        self.decay_rate = hyperparams.get('epoch_decay_rate', 0.05)
        self.min_decay_factor = hyperparams.get('epoch_min_decay_factor', 0.1)
        
        # State
        self.current_champion_id = None
        self.pending_champion_id = None
        self.champion_consecutive_count = 0
        self.epoch_start_time = time.time()
        self.steps_in_epoch = 0
        self.epoch_count = 0
        
        # Optimization: cache previous computation results
        self._last_h_factors = None
        self._last_h_sum = None
        self._last_h_variance = None
        self._last_champion = None
        self._h_factors_array = None
        self._initialized = False
        
    def check_epoch_transition(self, h_factors: Dict[str, float], current_time: float) -> Tuple[bool, Optional[str]]:
        """
        Check if new epoch needs to start (optimized version)
        """
        # If H-factors unchanged, use cached results
        if self._last_h_factors == h_factors and self._initialized:
            return False, None
        
        # Update cache
        self._last_h_factors = h_factors.copy()
        
        # Convert to numpy array (one-time allocation)
        if self._h_factors_array is None or len(h_factors) != len(self._h_factors_array):
            self._h_factors_array = np.zeros(len(h_factors), dtype=np.float64)
        
        # Fill array and find champion
        max_h = -1.0
        new_champion = None
        for i, (db_id, h_value) in enumerate(h_factors.items()):
            self._h_factors_array[i] = h_value
            if h_value > max_h:
                max_h = h_value
                new_champion = db_id
        
        self._last_champion = new_champion
        
        # Use numpy to compute statistics
        if len(self._h_factors_array) > 0:
            self._last_h_sum = np.sum(self._h_factors_array)
            if len(self._h_factors_array) > 1:
                # Use numpy variance calculation (much faster than statistics library)
                self._last_h_variance = np.var(self._h_factors_array, ddof=1)  # ddof=1 for sample variance
            else:
                self._last_h_variance = 0.0
        else:
            self._last_h_sum = 0.0
            self._last_h_variance = 0.0
        
        self._initialized = True
        
        # Check champion change
        if new_champion != self.current_champion_id:
            if new_champion == self.pending_champion_id:
                self.champion_consecutive_count += 1
                if self.champion_consecutive_count >= self.champion_change_threshold:
                    return True, f"champion_change_from_{self.current_champion_id}_to_{new_champion}"
            else:
                self.pending_champion_id = new_champion
                self.champion_consecutive_count = 1
        else:
            self.pending_champion_id = None
            self.champion_consecutive_count = 0
        
        # Check variance change (using cached values)
        if hasattr(self, 'last_epoch_h_variance'):
            if self.last_epoch_h_variance > 0:
                variance_change_ratio = abs(self._last_h_variance - self.last_epoch_h_variance) / self.last_epoch_h_variance
                if variance_change_ratio > self.h_variance_threshold:
                    return True, f"variance_change_ratio_{variance_change_ratio:.2f}"
        
        # Check sum change (using cached values)
        if hasattr(self, 'last_epoch_h_sum'):
            if self.last_epoch_h_sum > 0:
                sum_change_ratio = abs(self._last_h_sum - self.last_epoch_h_sum) / self.last_epoch_h_sum
                if sum_change_ratio > self.h_sum_threshold:
                    return True, f"sum_change_ratio_{sum_change_ratio:.2f}"
        
        return False, None
    
    def start_new_epoch(self, reason: str, current_time: float):
        """Start new epoch"""
        self.epoch_count += 1
        self.steps_in_epoch = 0
        self.epoch_start_time = current_time
        
        # Update champion
        self.current_champion_id = self.pending_champion_id or self._last_champion
        self.pending_champion_id = None
        self.champion_consecutive_count = 0
        
        # Save epoch statistics
        self.last_epoch_h_variance = self._last_h_variance
        self.last_epoch_h_sum = self._last_h_sum
    
    def increment_step(self):
        """Increment steps in epoch"""
        self.steps_in_epoch += 1
    
    def get_decay_factor(self) -> float:
        """Get decay factor"""
        if self.steps_in_epoch == 0:
            return 1.0
        
        decay = 1.0 - self.decay_rate * self.steps_in_epoch
        return max(decay, self.min_decay_factor)
    
    def _check_champion_change(self, h_factors: Dict[str, float]) -> Tuple[bool, Optional[str]]:
        """Check champion change (integrated into main method)"""
        # This method has been integrated into check_epoch_transition
        pass
    
    def _check_variance_change(self, h_factors: Dict[str, float]) -> Tuple[bool, Optional[str]]:
        """Check variance change (integrated into main method)"""
        # This method has been integrated into check_epoch_transition
        pass
    
    def _check_sum_change(self, h_factors: Dict[str, float]) -> Tuple[bool, Optional[str]]:
        """Check sum change (integrated into main method)"""
        # This method has been integrated into check_epoch_transition
        pass