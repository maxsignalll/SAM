"""
S0 Strategy Epoch Manager

Responsible for detecting significant H-factor changes and managing epoch transitions
"""

import logging
from typing import Dict, List, Optional, Tuple
import statistics


class S0EpochManager:
    """Epoch Manager - Detects and manages epochs based on H-factor changes"""
    
    def __init__(self, hyperparams: Dict, logger: logging.Logger):
        self.hyperparams = hyperparams
        self.logger = logger
        
        # Epoch detection parameters
        self.champion_change_threshold = hyperparams.get('epoch_champion_change_threshold', 3)  # Champion retention period
        self.h_variance_threshold = hyperparams.get('epoch_h_variance_threshold', 0.3)  # H variance change threshold
        self.h_sum_threshold = hyperparams.get('epoch_h_sum_threshold', 0.2)  # H sum change threshold
        
        # Multi-champion handling parameters
        self.multi_champion_mode = hyperparams.get('multi_champion_mode', 'stable')  # Multi-champion handling mode
        self.champion_tie_threshold = hyperparams.get('champion_tie_threshold', 0.01)  # Champion tie threshold（1%）
        
        # Epoch state
        self.current_epoch = 0
        self.steps_in_epoch = 0
        self.epoch_start_time = None
        
        # H-factor history tracking
        self.h_champion_history = []  # History of database IDs with highest H-factor（or set history）
        self.h_variance_history = []  # H-factor variance history
        self.h_sum_history = []  # H-factor sum history
        self.last_h_factors = {}  # Previous H-factor values
        
        # Additional state for multi-champion mode
        if self.multi_champion_mode == 'collective':
            self.champion_set_history = []  # Champion set history
            self.last_champion_set = set()  # Previous champion set
        
        # Epoch transition records
        self.epoch_transitions = []  # (epoch_id, reason, timestamp)
        
    def check_epoch_transition(self, h_factors: Dict[str, float], 
                             current_time: float) -> Tuple[bool, Optional[str]]:
        """
        Check if new epoch needs to start
        
        Args:
            h_factors: Current H-factors of all databases
            current_time: Current timestamp
            
        Returns:
            (Whether to transition, Transition reason)
        """
        if not h_factors:
            return False, None
            
        # First call, initialize
        if not self.last_h_factors:
            self._initialize_tracking(h_factors, current_time)
            return False, None
            
        # Detection 1: Champion handover
        champion_changed, champion_reason = self._check_champion_change(h_factors)
        if champion_changed:
            return True, champion_reason
            
        # Detection 2: H variance significant change
        variance_changed, variance_reason = self._check_variance_change(h_factors)
        if variance_changed:
            return True, variance_reason
            
        # Detection 3: H sum significant change (load change)
        sum_changed, sum_reason = self._check_sum_change(h_factors)
        if sum_changed:
            return True, sum_reason
            
        return False, None
    
    def _initialize_tracking(self, h_factors: Dict[str, float], current_time: float):
        """Initialize tracking state"""
        self.last_h_factors = h_factors.copy()
        self.epoch_start_time = current_time
        
        if self.multi_champion_mode == 'collective':
            # Collective mode：Find all champions within threshold
            champions = self._find_champion_set(h_factors)
            self.last_champion_set = champions
            self.champion_set_history = [champions]
            if len(champions) > 1:
                self.logger.info(f"Initial multi-champion set: {sorted(champions)}")
            elif champions:
                self.logger.info(f"Initial champion: {next(iter(champions))}")
        else:
            # Stable mode：Use deterministic champion selection
            champion_id = self._find_stable_champion(h_factors)
            self.h_champion_history = [champion_id]
            if champion_id:
                self.logger.info(f"Initial champion: {champion_id}")
        
        # Calculate initial statistics
        h_values = list(h_factors.values())
        self.h_variance_history = [statistics.variance(h_values) if len(h_values) > 1 else 0.0]
        self.h_sum_history = [sum(h_values)]
        
    def _check_champion_change(self, h_factors: Dict[str, float]) -> Tuple[bool, Optional[str]]:
        """Check if H-factor champion has sustained change"""
        if self.multi_champion_mode == 'collective':
            return self._check_champion_change_collective(h_factors)
        else:
            return self._check_champion_change_stable(h_factors)
    
    def _check_champion_change_stable(self, h_factors: Dict[str, float]) -> Tuple[bool, Optional[str]]:
        """Stable mode：Use deterministic champion selection"""
        # FindCurrent champion（stable sort）
        current_champion = self._find_stable_champion(h_factors)
        
        # Record champion history
        self.h_champion_history.append(current_champion)
        if len(self.h_champion_history) > self.champion_change_threshold + 1:
            self.h_champion_history.pop(0)
        
        # Check if new champion consistently leads
        if len(self.h_champion_history) >= self.champion_change_threshold:
            # Champions of recent N periods
            recent_champions = self.h_champion_history[-self.champion_change_threshold:]
            
            # If all are the same new champion，and different from previous champion
            if (len(set(recent_champions)) == 1 and 
                recent_champions[0] != self.h_champion_history[0]):
                reason = f"Champion changed from {self.h_champion_history[0]} to {recent_champions[0]}"
                self.logger.info(f"Epoch transition detected: {reason}")
                return True, reason
                
        return False, None
    
    def _check_champion_change_collective(self, h_factors: Dict[str, float]) -> Tuple[bool, Optional[str]]:
        """Collective mode：Track champion set changes"""
        # FindCurrent championset
        current_champions = self._find_champion_set(h_factors)
        
        # RecordChampion set history
        self.champion_set_history.append(current_champions)
        if len(self.champion_set_history) > self.champion_change_threshold + 1:
            self.champion_set_history.pop(0)
        
        # Check if champion set has sustained change
        if len(self.champion_set_history) >= self.champion_change_threshold:
            # Check if champion sets of recent N periods are all same
            recent_sets = self.champion_set_history[-self.champion_change_threshold:]
            
            # Check if all sets are same
            all_same = all(s == recent_sets[0] for s in recent_sets)
            
            if all_same and recent_sets[0] != self.champion_set_history[0]:
                # Champion set has sustained change
                old_set = sorted(self.champion_set_history[0])
                new_set = sorted(recent_sets[0])
                
                if len(new_set) > 1:
                    reason = f"Champion set changed from {old_set} to {new_set} (multi-champion)"
                else:
                    reason = f"Champion set changed from {old_set} to {new_set}"
                
                self.logger.info(f"Epoch transition detected: {reason}")
                self.last_champion_set = current_champions
                return True, reason
        
        return False, None
    
    def _check_variance_change(self, h_factors: Dict[str, float]) -> Tuple[bool, Optional[str]]:
        """Check if H-factor variance hasSignificant change"""
        h_values = list(h_factors.values())
        current_variance = statistics.variance(h_values) if len(h_values) > 1 else 0.0
        
        # Recordvariance history
        self.h_variance_history.append(current_variance)
        if len(self.h_variance_history) > 10:  # Keep last 10 values
            self.h_variance_history.pop(0)
        
        # Need sufficient historical data
        if len(self.h_variance_history) < 3:
            return False, None
            
        # Compare current variance with historical average
        historical_avg = statistics.mean(self.h_variance_history[:-1])
        if historical_avg > 0:
            variance_change = abs(current_variance - historical_avg) / historical_avg
            if variance_change > self.h_variance_threshold:
                reason = f"H variance changed by {variance_change:.1%} (from {historical_avg:.3f} to {current_variance:.3f})"
                self.logger.info(f"Epoch transition detected: {reason}")
                return True, reason
                
        return False, None
    
    def _check_sum_change(self, h_factors: Dict[str, float]) -> Tuple[bool, Optional[str]]:
        """Check if H-factor sum hasSignificant change（Load change）"""
        current_sum = sum(h_factors.values())
        
        # Recordsum history
        self.h_sum_history.append(current_sum)
        if len(self.h_sum_history) > 10:  # Keep last 10 values
            self.h_sum_history.pop(0)
        
        # Need sufficient historical data
        if len(self.h_sum_history) < 3:
            return False, None
            
        # Compare current sum with historical average
        historical_avg = statistics.mean(self.h_sum_history[:-1])
        if historical_avg > 0:
            sum_change = abs(current_sum - historical_avg) / historical_avg
            if sum_change > self.h_sum_threshold:
                reason = f"H sum changed by {sum_change:.1%} (from {historical_avg:.3f} to {current_sum:.3f})"
                self.logger.info(f"Epoch transition detected: {reason}")
                return True, reason
                
        return False, None
    
    def start_new_epoch(self, reason: str, current_time: float):
        """Start new epoch"""
        self.current_epoch += 1
        self.steps_in_epoch = 0
        self.epoch_start_time = current_time
        
        # Recordtransition
        self.epoch_transitions.append((self.current_epoch, reason, current_time))
        
        # Reset history tracking（Keep last value as starting point for new epoch）
        if self.multi_champion_mode == 'collective':
            if self.champion_set_history:
                self.champion_set_history = [self.champion_set_history[-1]]
        else:
            if self.h_champion_history:
                self.h_champion_history = [self.h_champion_history[-1]]
        
        if self.h_variance_history:
            self.h_variance_history = [self.h_variance_history[-1]]
        if self.h_sum_history:
            self.h_sum_history = [self.h_sum_history[-1]]
            
        self.logger.info(f"=== Start new epoch #{self.current_epoch} ===")
        self.logger.info(f"Reason: {reason}")
    
    def increment_step(self):
        """Increment steps in epoch"""
        self.steps_in_epoch += 1
    
    def get_epoch_info(self) -> Dict:
        """Get current epoch info"""
        return {
            'current_epoch': self.current_epoch,
            'steps_in_epoch': self.steps_in_epoch,
            'epoch_start_time': self.epoch_start_time,
            'recent_transitions': self.epoch_transitions[-5:] if self.epoch_transitions else []
        }
    
    def calculate_time_decay_factor(self) -> float:
        """
        Calculate time decay factor
        
        Returns:
            Decay factor (0, 1]，Closer to 0 means more decay
        """
        # Use smooth decay function
        decay_rate = self.hyperparams.get('epoch_decay_rate', 0.05)  # Decay per step5%
        decay_factor = 1.0 / (1.0 + decay_rate * self.steps_in_epoch)
        
        # Set minimumDecay factor，Avoid stopping learning completely
        min_decay = self.hyperparams.get('epoch_min_decay_factor', 0.1)
        decay_factor = max(min_decay, decay_factor)
        
        return decay_factor
    
    def _find_stable_champion(self, h_factors: Dict[str, float]) -> Optional[str]:
        """
        Findchampion（Stable mode）
        When multiple databases have same H value，Use alphabetical order for determinism
        """
        if not h_factors:
            return None
        
        # First sort by H value descending，Then sort by db_id alphabetically
        sorted_items = sorted(h_factors.items(), key=lambda x: (-x[1], x[0]))
        champion = sorted_items[0][0]
        
        # Detect and record close competitors
        max_h = sorted_items[0][1]
        near_ties = [(db, h) for db, h in sorted_items if abs(h - max_h) < 0.001]
        if len(near_ties) > 1:
            self.logger.debug(f"Detected databases with close H values: {near_ties}")
        
        return champion
    
    def _find_champion_set(self, h_factors: Dict[str, float]) -> set:
        """
        Find all champions within thresholdset（Collective mode）
        """
        if not h_factors:
            return set()
        
        # Find highest H value
        max_h = max(h_factors.values())
        
        # Find all databases within threshold
        champions = set()
        for db_id, h_value in h_factors.items():
            if abs(h_value - max_h) <= self.champion_tie_threshold * max_h:
                champions.add(db_id)
        
        return champions