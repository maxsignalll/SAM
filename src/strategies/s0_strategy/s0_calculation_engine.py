"""
S0 Strategy - H/V Factor Calculation Engine

Responsible for calculating horizontal factor (H) and vertical factor (V) in S0 strategy, including:
- EMA processing and normalization
- Dynamic weight calculation
- Hit rate normalization
- Batch score calculation
- New V-factor experimental system integration
"""

import math
import statistics
from typing import Any, Dict, List, Tuple, Optional

from .knee_detector import KneeDetector


class S0CalculationEngine:
    """S0 strategy calculation engine, responsible for H/V factor and dynamic weight calculation"""
    
    def __init__(self, hyperparams: Dict[str, Any], logger, state_manager=None):
        self.hyperparams = hyperparams
        self.logger = logger
        self.state_manager = state_manager
        
        # Cache calculation results to improve performance
        self._h_factors_cache = {}
        self._v_factors_cache = {}
        
        # V-factor manager removed, using simplified historical snapshot method
        
        # Initialize knee point detector
        knee_sensitivity = self.hyperparams.get('knee_sensitivity', 1.0)
        min_knee_gain = self.hyperparams.get('min_knee_gain', 0.1)
        self.knee_detector = KneeDetector(sensitivity=knee_sensitivity, min_knee_gain=min_knee_gain)
        
        # Core member related parameters
        self.min_core_members = self.hyperparams.get('min_core_members', 1)
        self.max_core_members_ratio = self.hyperparams.get('max_core_members_ratio', 0.5)
        
        # Cache recent core member set
        self._last_core_members = set()
        self._core_member_history = []  # Used to smooth core member changes
        
        # Simplified version of historical data (for variance calculation)
        self._h_history = {}  # {db_id: [Recent 8 H values]}
        self._v_history = {}  # {db_id: [Recent 8 V values]}
        self._allocation_history = {}  # {db_id: Previous allocation}
        self._adjustment_history = {}  # {db_id: [Recent adjustment amounts]}
        self._time_step = 0  # Time step counter
        
        # New algorithm related state
        self.use_simplified_algorithm = self.hyperparams.get('use_simplified_algorithm', False)
        self.history_window = self.hyperparams.get('history_window', 8)
        self.noise_levels = {}  # EWMA noise level estimation
        
    def _calculate_coefficient_of_variation(self, values: List[float]) -> float:
        """Calculate coefficient of variation (CV = σ/μ)"""
        if not values or len(values) < 2:
            # Return default low CV when insufficient data points, triggering more balanced weight allocation
            return 0.0
        
        mean_val = statistics.mean(values)
        if abs(mean_val) < 1e-9:
            return 0.0
        
        std_val = statistics.stdev(values)
        return std_val / abs(mean_val)
    
    def _calculate_dual_track_v_factor(self, db_id: str, state: Dict[str, Any], 
                                      current_allocation: int, current_hr_norm: float,
                                      current_cycles: int) -> Dict[str, float]:
        """
        Dual-track V-factor calculation
        
        Returns:
            {'v_fast': float, 'v_slow': float, 'v_combined': float}
        """
        dual_track = state['dual_v_track']
        
        # Update data window
        dual_track['allocation_window'].append((current_cycles, current_allocation))
        dual_track['hr_norm_window'].append((current_cycles, current_hr_norm))
        
        # Maintain window size
        max_window_size = 10  # Keep at most 10 points
        if len(dual_track['allocation_window']) > max_window_size:
            dual_track['allocation_window'].pop(0)
            dual_track['hr_norm_window'].pop(0)
        
        # Calculate V_fast (based on 2-3 points)
        v_fast = self._calculate_v_from_window(
            dual_track['allocation_window'][-3:],  # Recent 3 points
            dual_track['hr_norm_window'][-3:],
            method='diff'  # Difference method
        )
        
        # Calculate V_slow (based on 8-10 points)
        v_slow = 0.0
        if len(dual_track['allocation_window']) >= 5:  # Calculate slow V only with at least 5 points
            v_slow = self._calculate_v_from_window(
                dual_track['allocation_window'],  # All points
                dual_track['hr_norm_window'],
                method='regression'  # Regression method
            )
        
        # Update state
        dual_track['v_fast'] = v_fast
        dual_track['v_slow'] = v_slow
        dual_track['last_v_fast_update'] = current_cycles
        dual_track['last_v_slow_update'] = current_cycles
        
        # Combine V-factor (adjust weights based on consistency)
        if abs(v_fast) < 1e-6 and abs(v_slow) < 1e-6:
            v_combined = 0.0
        elif abs(v_slow) < 1e-6:  # Early stage with only v_fast
            v_combined = v_fast * 0.5  # Reduce weight due to uncertainty
        elif v_fast * v_slow > 0:  # Same direction
            v_combined = 0.3 * v_fast + 0.7 * v_slow
        else:  # Opposite direction, possibly noise
            v_combined = v_slow  # Trust slow speed
        
        if db_id in ['db_high_priority', 'db_medium_priority']:
            self.logger.info(f"📊 DUAL_V[{db_id}]: V_fast={v_fast:.4f}, V_slow={v_slow:.4f}, V_combined={v_combined:.4f}")
        
        return {
            'v_fast': v_fast,
            'v_slow': v_slow,
            'v_combined': v_combined
        }
    
    def _calculate_v_from_window(self, alloc_window: list, hr_window: list, method: str = 'diff') -> float:
        """
        Calculate V-factor from data window
        
        Args:
            alloc_window: [(cycles, allocation), ...]
            hr_window: [(cycles, hr_norm), ...]
            method: 'diff' for simple difference, 'regression' for linear regression
        """
        if len(alloc_window) < 2:
            return 0.0
        
        if method == 'diff':
            # SimpleDifference method
            if len(alloc_window) >= 2:
                delta_alloc = alloc_window[-1][1] - alloc_window[0][1]
                delta_hr = hr_window[-1][1] - hr_window[0][1]
                
                if abs(delta_alloc) < 1:  # Allocation barely changed
                    return 0.0
                
                return delta_hr / delta_alloc
        
        elif method == 'regression':
            # LinearRegression method
            if len(alloc_window) < 3:  # Need at least 3 points
                return 0.0
            
            # Extract data
            allocations = [point[1] for point in alloc_window]
            hr_norms = [point[1] for point in hr_window]
            
            # Calculate linear regression slope
            n = len(allocations)
            sum_x = sum(allocations)
            sum_y = sum(hr_norms)
            sum_xy = sum(x * y for x, y in zip(allocations, hr_norms))
            sum_x2 = sum(x * x for x in allocations)
            
            denominator = n * sum_x2 - sum_x * sum_x
            if abs(denominator) < 1e-6:
                return 0.0
            
            slope = (n * sum_xy - sum_x * sum_y) / denominator
            return slope
        
        return 0.0
    
    def _calculate_importance_urgency_matrix(self, db_id: str, state: Dict[str, Any], 
                                           priority: int, current_hr_norm: float) -> Dict[str, Any]:
        """
        Calculate importance-urgency matrix position
        
        Returns:
            {
                'importance': float,  # Importance score [0, 1]
                'urgency': float,     # Urgency score [0, 1]
                'quadrant': str,      # Q1-Q4
                'donor_score': float  # Score as donor
            }
        """
        iu_state = state['importance_urgency']
        
        # Calculate importance (relatively stable)
        # 40% from fixed priority, 60% from historical average H-factor
        priority_weight = 0.4
        historical_weight = 0.6
        
        # Normalize priority (assuming priority range 1-10)
        normalized_priority = (priority - 1) / 9.0  # Map to [0, 1]
        
        # Update historical average H-factor (using slow EMA)
        current_h = state['ema_ops_slow'] * current_hr_norm  # Simplified H-factor
        if iu_state['historical_avg_h'] == 0:
            iu_state['historical_avg_h'] = current_h
        else:
            iu_state['historical_avg_h'] = 0.1 * current_h + 0.9 * iu_state['historical_avg_h']
        
        # Normalize historical H-factor (assuming H-factor range [0, 1])
        normalized_hist_h = min(1.0, iu_state['historical_avg_h'])
        
        importance = priority_weight * normalized_priority + historical_weight * normalized_hist_h
        iu_state['importance_score'] = importance
        
        # Calculate urgency (dynamic change)
        # Based on dual-speed EMA difference
        ema_diff = abs(state['ema_ops_fast'] - state['ema_ops_slow'])
        if state['ema_ops_slow'] > 0:
            urgency = min(1.0, ema_diff / state['ema_ops_slow'])
        else:
            urgency = 0.0
        iu_state['urgency_score'] = urgency
        
        # Determine quadrant
        if importance >= 0.5 and urgency >= 0.5:
            quadrant = 'Q1'  # Important and urgent
        elif importance >= 0.5 and urgency < 0.5:
            quadrant = 'Q2'  # Important but not urgent
        elif importance < 0.5 and urgency >= 0.5:
            quadrant = 'Q3'  # Not important but urgent
        else:
            quadrant = 'Q4'  # Not important but not urgent
        
        iu_state['quadrant'] = quadrant
        
        # Calculate donor score (higher values indicate better donors)
        # Consider surplus level (hit rate close to 1 indicates possible surplus)
        surplus = max(0, current_hr_norm - 0.8)  # Portion exceeding 80% hit rate
        donor_score = (1 - importance) * (1 - urgency) * surplus
        iu_state['donor_score'] = donor_score
        
        if db_id in ['db_high_priority', 'db_medium_priority', 'db_low_priority']:
            self.logger.debug(f"📊 IU_MATRIX[{db_id}]: importance={importance:.2f}, urgency={urgency:.2f}, "
                            f"quadrant={quadrant}, donor_score={donor_score:.3f}")
        
        return {
            'importance': importance,
            'urgency': urgency,
            'quadrant': quadrant,
            'donor_score': donor_score
        }
    
    def _calculate_dynamic_weight_variance_based(self, h_history: Dict[str, List[float]], v_history: Dict[str, List[float]]) -> float:
        """
        SNR-based automatic weight calculation
        Factors with smaller variance should get higher weight (more stable signal)
        ρ = SNR_v / (SNR_h + SNR_v), where SNR = 1/Var
        """
        epsilon = 1e-6
        
        # Collect all historical H and V values
        all_h_values = []
        all_v_values = []
        
        for db_id in h_history:
            if db_id in h_history and h_history[db_id]:
                all_h_values.extend(h_history[db_id])
            if db_id in v_history and v_history[db_id]:
                all_v_values.extend(v_history[db_id])
        
        # Calculate variance
        var_h = statistics.variance(all_h_values) if len(all_h_values) > 1 else 1.0  # Default variance is 1
        var_v = statistics.variance(all_v_values) if len(all_v_values) > 1 else 1.0
        
        # Calculate signal-to-noise ratio (SNR = 1/Var)
        snr_h = 1.0 / (var_h + epsilon)
        snr_v = 1.0 / (var_v + epsilon)
        
        # Calculate weight based on SNR
        # V's SNR ratio of total SNR = V's weight (1-ρ)
        # Therefore H's weight ρ = SNR_h / (SNR_h + SNR_v)
        rho = snr_h / (snr_h + snr_v)
        
        # To avoid extreme cases, limit ρ to [0.2, 0.8] range
        rho = max(0.2, min(0.8, rho))
        
        self.logger.debug(f"📊 Signal-to-noise ratio weight calculation: Var(H)={var_h:.6f}, Var(V)={var_v:.6f}")
        self.logger.debug(f"    SNR(H)={snr_h:.3f}, SNR(V)={snr_v:.3f}, ρ(H weight)={rho:.3f}")
        self.logger.debug(f"    Result: H weight = {rho:.1%}, V weight = {(1-rho):.1%}")
        
        return rho
    
    def _calculate_dynamic_weight(self, cv_h: float, db_ids: List[str], orchestrator_states: Dict[str, Any]) -> float:
        """
        Calculate dynamic weight α_t, combining confidence-weighted learning rate
        New formula: η_t = decay(t_e) × C_V, α_t = 1 - η_t
        """
        # 1. Calculate theoretical decay term (decay)
        if not db_ids:
            raise ValueError("db_ids cannot be empty for dynamic weight calculation.")

        t_e_values = [orchestrator_states[db_id]['t_e'] for db_id in db_ids]
        if not t_e_values:
            raise ValueError("t_e_values cannot be empty. Missing 't_e' in orchestrator_states for provided db_ids.")
        avg_t_e = statistics.mean(t_e_values)
        
        eta0 = self.hyperparams['eta0']
        eta_floor = self.hyperparams['eta_floor']
        
        if avg_t_e <= 0:
            raise ValueError("avg_t_e must be positive for decay calculation.")
        else:
            decay_theoretical = eta0 / math.sqrt(avg_t_e)
        
        # Ensure decay value is within [eta_floor, 1.0] range
        decay_theoretical = max(eta_floor, min(1.0, decay_theoretical))
        
        # 2. Calculate average confidence (if audit system is enabled)
        if self.hyperparams.get('audit_enabled', True):
            # Collect confidence levels from all databases
            confidences = []
            for db_id in db_ids:
                state = orchestrator_states[db_id]
                v_audit_result = state.get('v_audit_result', {})
                confidence = v_audit_result.get('confidence', 0.0)
                
                # Only include databases with valid audit results
                if v_audit_result.get('last_update_cycles', 0) > 0:
                    confidences.append(confidence)
            
            # Calculate average confidence
            if confidences:
                avg_confidence = statistics.mean(confidences)
            else:
                # If no valid audit results, use default confidence
                avg_confidence = 0.5
        else:
            # If audit not enabled, use original CV modulation method
            cv_sigmoid_center = self.hyperparams['cv_sigmoid_center']
            cv_sigmoid_slope = self.hyperparams['cv_sigmoid_slope']
            sig = 1 / (1 + math.exp(-cv_sigmoid_slope * (cv_h - cv_sigmoid_center)))
            cv_mod = 0.5 + 0.4 * sig
            avg_confidence = cv_mod
        
        # 3. Calculate confidence-weighted learning rate
        confidence_weight = self.hyperparams.get('audit_confidence_weight', 1.0)
        eta_t = decay_theoretical * (avg_confidence ** confidence_weight)
        
        # 4. Convert to α_t
        alpha_t = 1 - eta_t
        
        if self.hyperparams.get('audit_enabled', True):
            self.logger.debug(f"🎯 Dynamic weight: avg_t_e={avg_t_e:.1f}, decay={decay_theoretical:.3f}, avg_C_V={avg_confidence:.3f} → η_t={eta_t:.3f}, α_t={alpha_t:.3f}")
        else:
            self.logger.debug(f"Dynamic weight calculation: cv_h={cv_h:.3f}, avg_t_e={avg_t_e:.1f}, decay={decay_theoretical:.3f} → alpha_t={alpha_t:.3f}")
        
        return alpha_t
    
    def _update_hr_normalization_metrics(self, db_id: str, hr_current: float, state: Dict[str, Any]):
        """
        Update normalized hit rate related metrics: HR_max_adaptive
        HR_base read from config file, kept fixed
        """
        hr_base = state['hr_base']
        hr_max_adaptive = state['hr_max_adaptive']
        hr_norm_update_threshold = self.hyperparams['hr_norm_update_threshold']
        hr_norm_decay_factor = self.hyperparams['hr_norm_decay_factor']
        hr_norm_p95_window_size = self.hyperparams['hr_norm_p95_window_size']
        hr_norm_max_stagnant_cycles = self.hyperparams['hr_norm_max_stagnant_cycles']
        
        # Initialize 95th percentile window and stagnation counter
        if 'hr_p95_window' not in state:
            state['hr_p95_window'] = []
        if 'hr_max_stagnant_counter' not in state:
            state['hr_max_stagnant_counter'] = 0
            
        # Upward update: Update immediately when current hit rate exceeds adaptive upper bound
        if hr_current > hr_max_adaptive + hr_norm_update_threshold:
            state['hr_max_adaptive'] = hr_current
            state['hr_max_stagnant_counter'] = 0  # Reset stagnation counter
            # Only show for important databases HR_max_adaptiveupdate
            if db_id in ['db_high_priority', 'db_medium_priority']:
                self.logger.debug(f"🔄 HR_max_adaptive[{db_id}]: {hr_current:.3f}")
            else:
                self.logger.debug(f"DB[{db_id}] Update HR_max_adaptive: {hr_current:.4f}")
        else:
            # Increment stagnation counter
            state['hr_max_stagnant_counter'] += 1
        
        # Maintain 95th percentile window
        state['hr_p95_window'].append(hr_current)
        if len(state['hr_p95_window']) > hr_norm_p95_window_size:
            state['hr_p95_window'].pop(0)
        
        # Decay mechanism: When stagnant exceeds threshold, use 95th percentile for decay
        if state['hr_max_stagnant_counter'] >= hr_norm_max_stagnant_cycles and len(state['hr_p95_window']) >= 5:
            hr_p95 = statistics.quantiles(state['hr_p95_window'], n=20)[18]  # 95th percentile
            hr_max_new = hr_norm_decay_factor * hr_max_adaptive + (1 - hr_norm_decay_factor) * hr_p95
            
            if hr_max_new < hr_max_adaptive:
                state['hr_max_adaptive'] = hr_max_new
                state['hr_max_stagnant_counter'] = 0  # Reset counter
                # Only show for important databases HR_max_adaptive decay
                if db_id in ['db_high_priority', 'db_medium_priority']:
                    self.logger.debug(f"🔄 HR_max_adaptive decay[{db_id}]: {hr_max_adaptive:.3f} → {hr_max_new:.3f} (P95: {hr_p95:.3f})")
                else:
                    self.logger.debug(f"DB[{db_id}] HR_max_adaptive decay: {hr_max_adaptive:.4f} -> {hr_max_new:.4f} (based onP95: {hr_p95:.4f})")
    
    def _calculate_hr_normalized(self, db_id: str, hr_current: float, state: Dict[str, Any]) -> float:
        """
        Calculate normalized hit rate
        HR_norm = (HR_current - HR_base) / (HR_max_adaptive - HR_base)
        """
        hr_base = state['hr_base']
        hr_max_adaptive = state['hr_max_adaptive']
        
        # Prevent division by zero
        denominator = hr_max_adaptive - hr_base
        if abs(denominator) < 1e-6:
            raise ValueError("Denominator for HR normalization is too close to zero.")
        
        hr_norm = (hr_current - hr_base) / denominator
        hr_norm = max(0.0, min(1.0, hr_norm))  # Ensure within[0,1]range
        
        # Only show for important databases HR_normcalculation result
        if db_id in ['db_high_priority', 'db_medium_priority'] and hr_norm > 0.1:
            self.logger.debug(f"📊 HR_norm[{db_id}]: raw={hr_current:.3f}, base={hr_base:.2f}, max={hr_max_adaptive:.3f} → norm={hr_norm:.3f}")
        else:
            self.logger.debug(f"DB[{db_id}] HR_norm calculation: raw={hr_current:.4f}, base={hr_base:.4f}, max={hr_max_adaptive:.4f} -> norm={hr_norm:.4f}")
        return hr_norm
    
    def _calculate_horizontal_factor(self, db_id: str, state: Dict[str, Any], norm_ops: float) -> float:
        """
        Calculate horizontal factor H (current efficiency)
        H = normalized_ops * hr_norm
        Use slow EMA hit rate defined in documentation
        """
        # H-factor always uses slow EMA hit rate
        hr_ema_slow = state['ema_hr_slow']
        hr_norm = self._calculate_hr_normalized(db_id, hr_ema_slow, state)
        
        # Horizontal factor calculation
        h_factor = norm_ops * hr_norm
        
        # Only show for important databases H-factorcalculation result
        if db_id in ['db_high_priority', 'db_medium_priority', 'db_low_priority']:
            self.logger.debug(f"📊 H-factor[{db_id}]: norm_ops={norm_ops:.3f}, hr_norm={hr_norm:.3f} → H={h_factor:.4f}")
        else:
            self.logger.debug(f"DB[{db_id}] H-factor calculation: norm_ops={norm_ops:.4f}, hr_norm={hr_norm:.3f} -> H={h_factor:.4f}")
        return h_factor

    def _calculate_v_clean(self, db_id: str, state: Dict[str, Any], db_period_stats: Dict[str, Any], h_params: Dict[str, Any]) -> float:
        """
        V-factor audit system - return V-factor based on audit results
        """
        # Add debug logs for important databases
        if db_id in ['db_high_priority', 'db_medium_priority']:
            self.logger.info(f"🔍 V_CLEAN_START[{db_id}]: Start V-factor calculation")
        
        # Check if first allocation
        if not state.get('first_allocation_done', False):
            if db_id in ['db_high_priority', 'db_medium_priority']:
                self.logger.info(f"V_FACTOR[{db_id}]: First allocation, V=0")
            return 0.0
        
        # Check if audit system is enabled
        if not self.hyperparams.get('audit_enabled', True):
            # If audit disabled, use original simplified V-factor calculation
            return self._calculate_v_clean_legacy(db_id, state, db_period_stats, h_params)
        
        # Get audit state
        audit_state = state.get('audit_state', {})
        v_audit_result = state.get('v_audit_result', {})
        
        # If in audit period, maintain last valid V-factor value
        if audit_state.get('is_auditing', False):
            # Use last audit result
            last_v_direction = v_audit_result.get('direction', 0.0)
            if db_id in ['db_high_priority', 'db_medium_priority']:
                remaining = audit_state.get('audit_cycles_remaining', 0)
                self.logger.info(f"🔍 V_AUDIT[{db_id}]: audit in progress, remaining {remaining}cycles, maintaining V={last_v_direction:.6f}")
            return last_v_direction
        
        # Use audit result V direction
        v_direction = v_audit_result.get('direction', 0.0)
        confidence = v_audit_result.get('confidence', 0.0)
        last_update = v_audit_result.get('last_update_cycles', 0)
        
        # Update cache
        self._v_factors_cache[db_id] = v_direction
        
        # Detailed logging
        if db_id in ['db_high_priority', 'db_medium_priority']:
            self.logger.info(f"📊 V_AUDIT_RESULT[{db_id}]: V_direction={v_direction:.6f}, C_V={confidence:.3f}, last_update_cycles={last_update}")
            
        return v_direction
    
    def _calculate_v_clean_legacy(self, db_id: str, state: Dict[str, Any], db_period_stats: Dict[str, Any], h_params: Dict[str, Any]) -> float:
        """
        Original simplified V-factor calculation (as backup)
        """
        # Check if within 2-cycles waiting period
        cycles_since_change = state.get('cycles_since_major_change', 999)
        if cycles_since_change < 2:
            if db_id in ['db_high_priority', 'db_medium_priority']:
                self.logger.info(f"V_FACTOR[{db_id}]: Waiting period (waited {cycles_since_change} cycles), V=0")
            return 0.0
        
        # Check if historical data is available
        last_allocation = state.get('last_allocation_pages')
        last_hr = state.get('last_ema_hit_rate')
        
        if last_allocation is None or last_hr is None:
            if db_id in ['db_high_priority', 'db_medium_priority']:
                self.logger.info(f"V_FACTOR[{db_id}]: Historical data incomplete, V=0")
            return 0.0
        
        # Get current state
        current_allocation = h_params['current_allocations'][db_id]
        current_hr = state['ema_hr_slow']
        
        # Calculate allocation change
        delta_allocation = current_allocation - last_allocation
        
        if abs(delta_allocation) < 1:  # Allocation barely changed
            if db_id in ['db_high_priority', 'db_medium_priority']:
                self.logger.info(f"V_FACTOR[{db_id}]: Allocation unchanged, V=0")
            return 0.0
        
        # Calculate normalized hit rate change
        hr_norm_current = self._calculate_hr_normalized(db_id, current_hr, state)
        
        # Use saved historical hr_max_adaptive to calculate historical normalized hit rate
        hr_base = state['hr_base']
        last_hr_max = state.get('last_hr_max_adaptive', state['hr_max_adaptive'])
        
        if last_hr_max > hr_base:
            hr_norm_last = (last_hr - hr_base) / (last_hr_max - hr_base)
        else:
            hr_norm_last = 0.0
        hr_norm_last = max(0.0, min(1.0, hr_norm_last))
        
        delta_hr_norm = hr_norm_current - hr_norm_last
        
        # Calculate marginal gain
        raw_marginal_gain = delta_hr_norm / delta_allocation
        
        # Detailed logging
        if db_id in ['db_high_priority', 'db_medium_priority']:
            if delta_allocation > 0:
                self.logger.info(f"📊 V_CALC_LEGACY[{db_id}]: Cache increased {last_allocation} -> {current_allocation} (+{delta_allocation})")
            else:
                self.logger.info(f"📊 V_CALC_LEGACY[{db_id}]: Cache decreased {last_allocation} -> {current_allocation} ({delta_allocation})")
            self.logger.info(f"📊 V_CALC_LEGACY[{db_id}]: Hit rate change {hr_norm_last:.3f} -> {hr_norm_current:.3f} (Δ={delta_hr_norm:.4f})")
            self.logger.info(f"📊 V_CALC_LEGACY[{db_id}]: Marginal gain = {raw_marginal_gain:.6f}")
            
        return raw_marginal_gain
    
    def _apply_tanh_v_normalization(self, v_factors: Dict[str, float]) -> Dict[str, float]:
        """
        Median aggregation + Tanh transformation V-factor normalization scheme
        
        Step 1 (median aggregation) completed in audit system, input v_factors here are stable V_direction values
        Step 2: Use scaled tanh for normalization
        
        V_norm = tanh(V_raw_stable * scaling_factor)
        
        Args:
            v_factors: Raw V-factor for each database (already audited V_direction)
            
        Returns:
            Normalized V-factor dictionary
        """
        if not v_factors:
            return {}
        
        # Get scaling factor (needs adjustment based on experimental environment constants)
        scaling_factor = self.hyperparams.get('v_factor_scaling_factor', 1000.0)
        
        normalized_v_factors = {}
        for db_id, v_raw in v_factors.items():
            # Apply scaled tanh transformation
            # V_norm = tanh(V_raw * scaling_factor)
            v_scaled = v_raw * scaling_factor
            v_norm = math.tanh(v_scaled)
            
            normalized_v_factors[db_id] = v_norm
            
            # Detailed loggingRecord normalization process
            if db_id in ['db_high_priority', 'db_medium_priority']:
                self.logger.info(f"📊 V_NORM[{db_id}]: {v_raw:.6f} * {scaling_factor} = {v_scaled:.3f} → tanh → {v_norm:.4f}")
            else:
                self.logger.debug(f"DB[{db_id}] V normalization: {v_raw:.6f} → {v_norm:.4f}")
        
        return normalized_v_factors
    
    def _apply_median_based_v_normalization(self, v_factors: Dict[str, float], core_members_set: set = None) -> Dict[str, float]:
        """
        Median-based logarithmic ratio normalization method
        Map V-factor values to reasonable range while preserving semantics
        
        Improvement: Use all database V-factors for normalization, improve stability
        
        Args:
            v_factors: Raw V-factor dictionary
            core_members_set: Core memberset (kept as parameter but no longer used for filtering)
        """
        if not v_factors:
            return {}
        
        # Use all database V-factors for normalization (including zeros)
        all_v_values = list(v_factors.values())
        
        # Group V-factor values (including all databases)
        positive_vs = [v for v in all_v_values if v > 0]
        negative_vs = [v for v in all_v_values if v < 0]
        zero_count = sum(1 for v in all_v_values if abs(v) < 1e-9)
        
        # Record V-factor distribution information
        self.logger.info(f"📊 V-factor distribution: positive={len(positive_vs)}, negative={len(negative_vs)}, zero={zero_count}")
        
        # Calculate median baseline
        if positive_vs:
            v_median_pos = statistics.median(positive_vs)
        else:
            # If no positive values, use small default value
            v_median_pos = 0.01
        
        if negative_vs:
            v_median_neg = statistics.median(negative_vs)
        else:
            # ifnonegativeusesmall default value
            v_median_neg = -0.01
        
        # Ensure median is not zero
        if abs(v_median_pos) < 1e-6:
            v_median_pos = 0.01
        if abs(v_median_neg) < 1e-6:
            v_median_neg = -0.01
        
        self.logger.debug(f"📈 V-factor median baseline: positive median={v_median_pos:.6f}, negative median={v_median_neg:.6f}")
        
        # Calculate normalized V-factor
        normalized_v_factors = {}
        for db_id, raw_v in v_factors.items():
            if abs(raw_v) < 1e-9:
                # Zero values remain 0
                normalized_v_factors[db_id] = 0.0
                continue
            
            # Normalize all non-zero V-factors
            if raw_v > 0:
                # Positive: score = log(1 + raw_v/v_median_pos)
                score = math.log(1 + raw_v / v_median_pos)
            else:  # raw_v < 0
                # negativescore = -log(1 + |raw_v|/|v_median_neg|)
                score = -math.log(1 + abs(raw_v) / abs(v_median_neg))
            
            # tanh smoothing
            v_final = math.tanh(score)
            normalized_v_factors[db_id] = v_final
            
            # Add core member marker to log
            is_core = core_members_set and db_id in core_members_set
            core_marker = "🏆" if is_core else "⚪"
            
            # Add especially prominent normalization log for important databases
            if db_id in ['db_high_priority', 'db_medium_priority']:
                self.logger.debug(f"🔥 {core_marker} V_NORM[{db_id}] normalization: {raw_v:.6f} -> score={score:.4f} -> V_final={v_final:.4f}")
            else:
                self.logger.debug(f"{core_marker} DB[{db_id}] V-factor normalization: {raw_v:.6f} -> score={score:.4f} -> V_final={v_final:.4f}")
        
        return normalized_v_factors
    
    def _identify_core_members(self, h_factors: Dict[str, float]) -> Tuple[List[str], Optional[int], float]:
        """
        Identify core members based on H-factor and knee point detection
        
        Returns:
            (core_member_ids, knee_index, knee_score)
        """
        # Use knee point detector to identify core members
        core_member_ids, knee_index, knee_score = self.knee_detector.identify_core_members(
            h_factors,
            min_core_members=self.min_core_members,
            max_core_members_ratio=self.max_core_members_ratio
        )
        
        # Update core member history (for smoothing)
        self._core_member_history.append(set(core_member_ids))
        if len(self._core_member_history) > 3:  # Keep recent 3 history entries
            self._core_member_history.pop(0)
            
        # Check core member changes
        current_core = set(core_member_ids)
        if current_core != self._last_core_members:
            added = current_core - self._last_core_members
            removed = self._last_core_members - current_core
            
            self.logger.info(f"🎯 Core memberchange: added{added if added else 'none'}, removed{removed if removed else 'none'}")
            self.logger.info(f"📊 Current core members({len(core_member_ids)}): {core_member_ids}")
            if knee_index is not None:
                self.logger.info(f"📍 Knee point position: index={knee_index}, score={knee_score:.3f}")
                
        self._last_core_members = current_core
        
        return core_member_ids, knee_index, knee_score
    
    def _batch_calculate_scores_simplified(self, current_metrics: Dict[str, Any], h_params: Dict[str, Any], 
                                         db_ids: List[str], orchestrator_states: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simplified batch calculation: Variance-based automatic H/V balance
        """
        self.logger.debug(f"🆕 Simplified algorithm: _batch_calculate_scores_simplified start")
        
        # Calculate H and V factors
        h_factors = {}
        v_factors = {}
        
        # Calculate H-factors for all databases
        for db_id in db_ids:
            if db_id not in orchestrator_states:
                continue
            state = orchestrator_states[db_id]
            
            # Use slow EMA for stable values
            ops = state['ema_ops_slow']
            hr = state['ema_hr_slow']
            
            # H-factor = normalized ops * hr
            if ops > 0:
                log_ops = math.log(max(ops, 1e-6))
                # Adjust coefficient to put H-factor in more reasonable range
                h_factor = log_ops * hr / 5.0  # Change from 10 to 5, increase H-factor value
            else:
                h_factor = 0.0
            
            h_factors[db_id] = h_factor
            
            # Update history
            if db_id not in self._h_history:
                self._h_history[db_id] = []
            self._h_history[db_id].append(h_factor)
            if len(self._h_history[db_id]) > self.history_window:
                self._h_history[db_id].pop(0)
        
        # Calculate V-factors for all databases
        for db_id in db_ids:
            if db_id not in orchestrator_states:
                continue
            state = orchestrator_states[db_id]
            
            # Simplified algorithm does not use audit system, directly calculate marginal gain
            v_factor = self._calculate_v_factor_simplified(db_id, state, h_params)
            v_factors[db_id] = v_factor
            
            # Update history
            if db_id not in self._v_history:
                self._v_history[db_id] = []
            self._v_history[db_id].append(v_factor)
            if len(self._v_history[db_id]) > self.history_window:
                self._v_history[db_id].pop(0)
        
        # No longer need to update _allocation_history, using snapshot data in state
        
        # Calculate signal-to-noise ratio based dynamic weight
        rho = self._calculate_dynamic_weight_variance_based(self._h_history, self._v_history)
        
        # Calculate final scores
        scores = {}
        total_score = 0.0
        
        # Display current H/V factor values
        self.logger.info(f"🔍 Current factor values:")
        for db_id in db_ids:
            h = h_factors.get(db_id, 0.0)
            v = v_factors.get(db_id, 0.0)
            if db_id in ['db_high_priority', 'db_medium_priority', 'db_low_priority']:
                self.logger.info(f"  {db_id}: H={h:.4f}, V={v:.4f}")
        
        # Note: No longer update allocation history here, should be handled by state_manager when major changes detected
        
        for db_id in db_ids:
            h = h_factors.get(db_id, 0.0)
            v = v_factors.get(db_id, 0.0)
            
            # Use signal-to-noise ratio weight
            score = rho * h + (1 - rho) * v
            score = max(0, score)
            
            scores[db_id] = score
            total_score += score
            
            # Update state
            if db_id in orchestrator_states:
                orchestrator_states[db_id]['current_alpha_t'] = rho
            
            if db_id in ['db_high_priority', 'db_medium_priority', 'db_low_priority']:
                self.logger.info(f"  {db_id} score: {rho:.1%}×{h:.4f} + {(1-rho):.1%}×{v:.4f} = {score:.4f}")
            else:
                self.logger.debug(f"DB[{db_id}] simplified score: ρ={rho:.3f}, H={h:.4f}, V={v:.4f} -> score={score:.4f}")
        
        # Calculate CV for reference
        active_h_factors = [h for h in h_factors.values() if h > 0]
        cv_h = self._calculate_coefficient_of_variation(active_h_factors)
        
        return {
            'scores': scores,
            'total_score': total_score,
            'h_factors': h_factors,
            'v_factors': v_factors,
            'alpha_t': rho,  # Use rho as α_t
            'cv_h': cv_h
        }
    
    def _batch_calculate_scores_and_states(self, current_metrics: Dict[str, Any], h_params: Dict[str, Any], 
                                         db_ids: List[str], orchestrator_states: Dict[str, Any],
                                         frozen_alpha_t: float = None, is_efficiency_only: bool = False) -> Dict[str, Any]:
        """
        Batch calculate H-factors, V-factors and final scores for all databases
        """
        # Check if using simplified algorithm
        if self.use_simplified_algorithm:
            return self._batch_calculate_scores_simplified(current_metrics, h_params, db_ids, orchestrator_states)
        
        self.logger.debug(f"Calculation engine: _batch_calculate_scores_and_states start, db_ids={db_ids}")
        
        # Fix: Mark first allocation complete before calculating V-factor
        for db_id in db_ids:
            if db_id in orchestrator_states:
                state = orchestrator_states[db_id]
                if not state['first_allocation_done']:
                    if db_id in ['db_high_priority', 'db_medium_priority']:
                        self.logger.info(f"🎆 FIRST_ALLOC[{db_id}]: Mark first allocation complete (current allocation: {h_params['current_allocations'][db_id]} pages)")
                    state['first_allocation_done'] = True
                    self.logger.debug(f"V_CALC_DEBUG[{db_id}]: Mark first allocation complete")
        
        h_factors = {}
        v_factors_raw = {}
        
        # --- H-factor calculation ---
        # Step 1: Collect ops_for_decision from all databases
        ops_for_decision_map = {}
        strategic_change_threshold = self.hyperparams['strategic_change_threshold']

        for db_id in db_ids:
            if db_id not in orchestrator_states:
                raise KeyError(f"Missing orchestrator state for db_id: {db_id}")
            state = orchestrator_states[db_id]
            
            # Decide which OPS to use (fast/slow)
            ops_ema_slow = state['ema_ops_slow']
            ops_ema_fast = state['ema_ops_fast']
            
            is_in_cooldown = state['smart_cooldown_remaining'] > 0
            if is_in_cooldown:
                ops_for_decision = ops_ema_fast
            elif ops_ema_fast > ops_ema_slow * strategic_change_threshold and ops_ema_slow > 1e-6:
                ops_for_decision = ops_ema_fast
            else:
                ops_for_decision = ops_ema_slow
            
            ops_for_decision_map[db_id] = ops_for_decision

        # Step 2: Calculate logarithmic normalization parameters based on all ops
        norm_ops_map = {}
        ops_values = [ops for ops in ops_for_decision_map.values() if ops > 0]

        if ops_values:
            log_ops_values = [math.log(max(ops, 1e-6)) for ops in ops_values]
            log_min = min(log_ops_values)
            log_max = max(log_ops_values)
            log_range = log_max - log_min if log_max > log_min else 1.0

            if log_range > 1e-9:
                for db_id, ops in ops_for_decision_map.items():
                    if ops > 0:
                        log_ops = math.log(max(ops, 1e-6))
                        norm_ops = (log_ops - log_min) / log_range
                        # Ensure norm_ops in [0, 1] range
                        norm_ops_map[db_id] = max(0.0, min(1.0, norm_ops))
                    else:
                        norm_ops_map[db_id] = 0.0
            else: # All ops values are the same
                for db_id in ops_for_decision_map:
                    norm_ops_map[db_id] = 1.0
        
        # Step 3: Calculate H-factor for each database
        for db_id in db_ids:
            if db_id not in orchestrator_states: continue
            state = orchestrator_states[db_id]
            norm_ops = norm_ops_map[db_id]
            
            h_factor = self._calculate_horizontal_factor(db_id, state, norm_ops)
            h_factors[db_id] = h_factor
            self._h_factors_cache[db_id] = h_factor

        # --- Core member identification ---
        # Identify core members based on H-factor
        core_member_ids, knee_index, knee_score = self._identify_core_members(h_factors)
        core_members_set = set(core_member_ids)
        
        self.logger.info(f"🎯 Core member count M in this round = {len(core_member_ids)}, Knee point score = {knee_score:.3f}")
        
        # --- V-factor calculation ---
        # Improvement: Calculate V-factors for all databases (no longer limited to core members)
        for db_id in db_ids:
            if db_id not in orchestrator_states: continue
            state = orchestrator_states[db_id]
            db_period_stats = current_metrics[db_id]
            
            # If in audit period, update micro V-factor
            if self.hyperparams.get('audit_enabled', True) and state.get('audit_state', {}).get('is_auditing', False):
                current_allocation = h_params['current_allocations'][db_id]
                current_hr = state['ema_hr_slow']
                current_hr_norm = self._calculate_hr_normalized(db_id, current_hr, state)
                
                # Delegate to state manager to update micro V-factor
                if self.state_manager is not None:
                    self.state_manager.update_audit_micro_v_factor(db_id, current_allocation, current_hr_norm)
                else:
                    self.logger.warning(f"⚠️ V_AUDIT[{db_id}]: Cannot update micro V-factor, state_manager not set")
            
            # Check if using dual-track V-factor
            if self.hyperparams.get('use_dual_track_v', True):
                # Use newDual-track V-factor calculation
                current_allocation = h_params['current_allocations'][db_id]
                current_hr = state['ema_hr_slow']
                current_hr_norm = self._calculate_hr_normalized(db_id, current_hr, state)
                current_cycles = state.get('current_cycles', 0)
                
                dual_v_result = self._calculate_dual_track_v_factor(
                    db_id, state, current_allocation, current_hr_norm, current_cycles
                )
                v_factor = dual_v_result['v_combined']
            else:
                # Use original V-factor calculation
                v_factor = self._calculate_v_clean(db_id, state, db_period_stats, h_params)
            
            v_factors_raw[db_id] = v_factor
            
            # Record whether core member (for logging only)
            if db_id in core_members_set:
                self.logger.debug(f"🏆 Core member[{db_id}]: Calculate V-factor = {v_factor:.6f}")
            else:
                self.logger.debug(f"⚪ Non-core member[{db_id}]: Calculate V-factor = {v_factor:.6f}")
        
        # Show raw V-factors for important databases
        for db_id in ['db_high_priority', 'db_medium_priority']:
            if db_id in v_factors_raw:
                self.logger.info(f"🔍 RAW_V_FACTORS[{db_id}]: {v_factors_raw[db_id]:.6f}")
        
        # New normalization scheme: Median aggregation + Tanh transformation (use all database V-factors)
        v_factors_final = self._apply_median_based_v_normalization(v_factors_raw, core_members_set)
        for db_id, v_factor in v_factors_final.items():
            self._v_factors_cache[db_id] = v_factor
        
        # --- Importance-urgency matrix calculation ---
        iu_matrix_results = {}
        if self.hyperparams.get('use_importance_urgency_matrix', True):
            for db_id in db_ids:
                if db_id not in orchestrator_states: continue
                state = orchestrator_states[db_id]
                
                # Get database priority (from configuration)
                db_priority = 5  # default medium priority
                for db_conf in self.state_manager.orchestrator.db_instance_configs:
                    if db_conf['id'] == db_id:
                        db_priority = db_conf.get('base_priority', 5)
                        break
                
                # Calculate current normalized hit rate
                current_hr = state['ema_hr_slow']
                current_hr_norm = self._calculate_hr_normalized(db_id, current_hr, state)
                
                # Calculate importance-urgency matrix position
                iu_result = self._calculate_importance_urgency_matrix(
                    db_id, state, db_priority, current_hr_norm
                )
                iu_matrix_results[db_id] = iu_result
        
        # --- Final score calculation ---
        active_h_factors = [h for h in h_factors.values() if h > 0]
        cv_h = self._calculate_coefficient_of_variation(active_h_factors)
        
        if is_efficiency_only:
            alpha_t = 1.0
            self.logger.debug("B8_EFFICIENCY_ONLY mode, force alpha_t = 1.0")
        elif frozen_alpha_t is not None: # This branch now handles the user's desired fixed initial value
            alpha_t = frozen_alpha_t
            self.logger.debug(f"Use fixed alpha_t = {alpha_t:.3f}")
        else: # Only calculate dynamically if not efficiency_only and not frozen
            alpha_t = self._calculate_dynamic_weight(cv_h, db_ids, orchestrator_states)
            self.logger.debug(f"CV(H_t) = {cv_h:.4f}, Dynamic weight α_t = {alpha_t:.3f}")
        
        scores = {}
        total_score = 0.0
        for db_id in db_ids:
            if db_id not in h_factors:
                raise KeyError(f"Missing H-factor for db_id: {db_id}")
            if db_id not in v_factors_final:
                raise KeyError(f"Missing V-factor for db_id: {db_id}")
            
            h_factor = h_factors[db_id]
            v_factor = v_factors_final[db_id]
            
            score = alpha_t * h_factor + (1 - alpha_t) * v_factor
            score = max(0, score)
            
            scores[db_id] = score
            total_score += score
            
            if db_id in orchestrator_states:
                orchestrator_states[db_id]['current_alpha_t'] = alpha_t
            
            # Debug log
            self.logger.debug(f"DB[{db_id}] score calculation: α_t={alpha_t:.3f}, H={h_factor:.4f}, V={v_factor:.4f} -> score={score:.4f}")
        
        return {
            'scores': scores,
            'total_score': total_score,
            'h_factors': h_factors,
            'v_factors': v_factors_final,
            'alpha_t': alpha_t,
            'cv_h': cv_h,
            'iu_matrix': iu_matrix_results  # Added importance-urgency matrix result
        }
    
    def get_cached_h_factor(self, db_id: str) -> float:
        """Get cached H-factor"""
        return self._h_factors_cache[db_id]
    
    def get_cached_v_factor(self, db_id: str) -> float:
        """Get cached V-factor"""
        return self._v_factors_cache[db_id]
    
    # V-factor manager related methods removed, use simplified history snapshot method
    
    def clear_cache(self):
        """Clear cache"""
        self._h_factors_cache.clear()
        self._v_factors_cache.clear()