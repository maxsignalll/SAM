"""
S0 Strategy FinalScore Calculation Module

Implements saturation confidence mechanism and dynamic weight calculation according to documentation
"""

from typing import Dict, Tuple
import logging


class S0FinalScoreCalculator:
    """S0 Strategy Final Score Calculator"""
    
    def __init__(self, hyperparams: Dict, logger: logging.Logger):
        self.logger = logger
        
        # Saturation detection parameters
        self.positive_v_threshold = hyperparams.get('positive_v_threshold', 0.001)
        self.hr_ratio_strict = hyperparams.get('hr_ratio_strict', 0.95)
        self.confidence_increment = hyperparams.get('confidence_increment', 0.34)
        
        # Dynamic weight parameters
        self.base_alpha = hyperparams.get('base_alpha', 0.5)
        self.saturation_alpha = hyperparams.get('saturation_alpha', 0.9)
        
        # Saturation confidence state for each database
        self.saturation_confidence = {}
    
    def calculate_final_score(self, db_id: str, h_factor: float, v_factor: float, 
                            hr_current: float, hr_base: float, hr_max: float) -> Tuple[float, float]:
        """
        Calculate final score for a single database
        
        Args:
            db_id: Database ID
            h_factor: H-factor (horizontal factor)
            v_factor: V-factor (vertical factor)
            hr_current: Current hit rate
            hr_base: Baseline hit rate
            hr_max: Maximum hit rate
            
        Returns:
            (final_score, alpha_t): Final score and dynamic weight used
        """
        # Initialize saturation confidence
        if db_id not in self.saturation_confidence:
            self.saturation_confidence[db_id] = 0.0
        
        # Update saturation confidence
        self._update_saturation_confidence(db_id, v_factor, hr_current, hr_base, hr_max)
        
        # Calculate dynamic weightalpha_t
        alpha_t = self._calculate_dynamic_alpha(db_id)
        
        # Calculate final score
        final_score = alpha_t * h_factor + (1 - alpha_t) * v_factor
        
        return final_score, alpha_t
    
    def _update_saturation_confidence(self, db_id: str, v_factor: float, 
                                    hr_current: float, hr_base: float, hr_max: float):
        """Update saturation confidence"""
        # Calculate HR ratio
        if hr_max > hr_base:
            hr_ratio = (hr_current - hr_base) / (hr_max - hr_base)
        else:
            hr_ratio = 0.0
        
        # Check saturation condition
        if abs(v_factor) < self.positive_v_threshold and hr_ratio > self.hr_ratio_strict:
            # Meets saturation condition, increase confidence
            old_confidence = self.saturation_confidence[db_id]
            new_confidence = min(1.0, old_confidence + self.confidence_increment)
            self.saturation_confidence[db_id] = new_confidence
            
            self.logger.debug(f"DB[{db_id}] Saturation confidence increases: {old_confidence:.2f} -> {new_confidence:.2f}, "
                            f"|V|={abs(v_factor):.6f} < {self.positive_v_threshold}, "
                            f"HR_ratio={hr_ratio:.3f} > {self.hr_ratio_strict}")
        else:
            # Does not meet saturation condition, reset immediately
            if self.saturation_confidence[db_id] > 0:
                self.logger.debug(f"DB[{db_id}] Saturation confidence reset: {self.saturation_confidence[db_id]:.2f} -> 0.0, "
                                f"|V|={abs(v_factor):.6f}, HR_ratio={hr_ratio:.3f}")
            self.saturation_confidence[db_id] = 0.0
    
    def _calculate_dynamic_alpha(self, db_id: str) -> float:
        """Calculate dynamic weightalpha_t"""
        confidence = self.saturation_confidence[db_id]
        alpha_t = self.base_alpha + (self.saturation_alpha - self.base_alpha) * confidence
        return alpha_t
    
    def get_saturation_status(self) -> Dict[str, float]:
        """Get saturation confidence status for all databases"""
        return self.saturation_confidence.copy()
    
    def reset_saturation_confidence(self, db_id: str = None):
        """Reset saturation confidence"""
        if db_id:
            self.saturation_confidence[db_id] = 0.0
        else:
            self.saturation_confidence.clear()