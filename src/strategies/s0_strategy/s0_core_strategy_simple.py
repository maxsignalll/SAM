"""
S0 Strategy - Simplified Core Strategy Class

Version: S0CoreStrategySimple (Basic Simplified Version)
Features: Only retains gradient descent allocation and simple H, V, alpha_t calculations
Purpose: Base class for other simplified versions
Status: Stable, inherited by multiple versions

See: S0_VERSIONS_README.md
"""

import logging
import math
import statistics
import time
# import numpy as np  # No longer need NumPy, using pure Python for performance
from typing import Any, Dict, List, Tuple

from ..base_strategy import BaseStrategy
from .s0_gradient_allocator import S0GradientAllocator
from .s0_epoch_manager_optimized import S0EpochManagerOptimized as S0EpochManager
# from .percentile_optimization import calculate_percentile_partition_interpolated  # Replaced with max value method
from .s0_final_score_calculator import S0FinalScoreCalculator


# MedianFilter class removed - replaced with EMA smoothing


class S0CoreStrategySimple(BaseStrategy):
    """Simplified implementation of S0 strategy"""
    
    def __init__(self, orchestrator: Any, strategy_name: str, strategy_specific_config: Dict = None):
        super().__init__(orchestrator, strategy_name, strategy_specific_config)
        
        # Basic configuration
        self.hyperparams = self.strategy_specific_config["strategy_config"]["strategy_hyperparameters"]
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.strategy_name}")
        # Disable logging in scalability experiments for performance
        if hasattr(orchestrator, 'is_scalability_experiment') and orchestrator.is_scalability_experiment:
            self.logger.setLevel(logging.CRITICAL)
        
        # S0 is a dynamic strategy
        self.is_dynamic = True
        
        # Get reporting interval for TPS calculation
        self.reporting_interval = orchestrator.general_setup.get("reporting_interval_seconds", 4)
        
        # For recording H and V factors
        self._last_h_factors = {}
        self._last_v_factors = {}
        self._last_decision_v_factors = {}  # Expected attribute name by experiment manager
        
        # Simplified V-factor parameters - only keep necessary ones
        self.v_threshold = 0.0001  # Dead zone threshold based on measurement precision
        self.saturation_threshold = self.hyperparams.get('saturation_threshold', 0.9)  # Saturation detection threshold
        self.v_ema_alpha = 0.3  # V-factor EMA smoothing coefficient
        
        # self.logger.info(f"V-factor params: dead_zone={self.v_threshold}, saturation={self.saturation_threshold}, "
        #                 f"EMA_alpha={self.v_ema_alpha}")
        
        # Fixed and elastic pool calculation
        # total_pages already set in BaseStrategy
        
        # Calculate fixed allocation (based on priority)
        total_priority = sum(db["base_priority"] for db in self.db_instance_configs)
        fixed_pool_percentage = self.strategy_specific_config["fixed_pool_percentage_of_total_ram"]
        fixed_pool_pages = int(self.total_pages * fixed_pool_percentage)
        
        self.s0_fixed_allocations = {}
        for db_conf in self.db_instance_configs:
            db_id = db_conf["id"]
            priority = db_conf["base_priority"]
            if total_priority > 0:
                priority_share = priority / total_priority
                fixed_pages = int(fixed_pool_pages * priority_share)
            else:
                fixed_pages = 0
            self.s0_fixed_allocations[db_id] = fixed_pages
        
        # Elastic pool size
        self.elastic_pool_pages = self.total_pages - sum(self.s0_fixed_allocations.values())
        
        # Create epoch manager
        self.epoch_manager = S0EpochManager(self.hyperparams, self.logger)
        
        # Initialize gradient allocator
        self.gradient_allocator = S0GradientAllocator(self, orchestrator, self.hyperparams, self.logger)
        
        # Initialize FinalScore calculator
        self.final_score_calculator = S0FinalScoreCalculator(self.hyperparams, self.logger)
        
        # Pass epoch manager to gradient allocator
        self.gradient_allocator.epoch_manager = self.epoch_manager
        
        # EMA states
        self.ema_states = {}
        
        # Alpha_t calculation state based on V-factor variance
        self.alpha_state = {
            'alpha_prev': 0.5,  # Previous alpha_t value
            'alpha_smoothing': self.hyperparams.get('alpha_smoothing', 0.4),  # alpha_t smoothing parameter
            'v_stats': {},  # V-factor statistics per database {db_id: {'mean_v', 'var_v', 'mag_v'}}
            'ewma_lambda': 0.8,  # EWMA coefficient for statistics
            'initialized': False  # Whether initialized
        }
        
        # Saturation confidence parameters
        self.saturation_params = {
            'v_near_zero_threshold': 0.001,  # V-factor near-zero threshold
            'high_hit_rate_threshold': 0.9,  # High hit rate threshold
            'confidence_increment': 0.34,  # Confidence growth rate (reach 1.0 in 3 cycles)
            'min_samples_for_kappa': 2,  # Minimum samples for calculating kappa
            'saturated_alpha_default': 0.7  # Default alpha_t when many DBs saturated
        }
        
        # Track tuning cycles (for debugging)
        self.tuning_cycle_count = 0
        
        for db_conf in self.db_instance_configs:
            db_id = db_conf["id"]
            self.ema_states[db_id] = {
                'ema_ops_slow': 0.0,
                'ema_ops_fast': 0.0,
                'ema_hr_slow': 0.0,
                'ema_hr_fast': 0.0,
                'hr_base': self.hyperparams.get('hr_base', 0.5),
                'hr_max_adaptive': self.hyperparams.get('hr_base', 0.5) + 0.3,  # Initialize to hr_base+0.3 for more room
                'last_allocation_pages': None,
                'last_ema_hit_rate': None,
                'v_ema': 0.0,  # V-factor EMA value
                # New saturation confidence mechanism
                'saturation_confidence': 0.0  # Saturation confidence [0.0, 1.0]
            }
        
        # Record hr_max_adaptive initialization
        hr_base = self.hyperparams.get('hr_base', 0.5)
        # self.logger.info(f"hr_max_adaptive initialized to hr_base={hr_base} (not 1.0), avoiding saturation detection failure")
        
        # self.logger.info(f"S0 simplified strategy initialized: total_pages={self.total_pages}, "
        #                 f"fixed_pool={sum(self.s0_fixed_allocations.values())}, "
        #                 f"elastic_pool={self.elastic_pool_pages}")
        
        # Set log level to DEBUG to see debug info
        # self.logger.setLevel(logging.DEBUG)
        
        # Ensure logs can propagate to parent logger
        # self.logger.propagate = True
        
        # If orchestrator has logger, use same handlers
        # if hasattr(orchestrator, 'logger') and orchestrator.logger:
        #     for handler in orchestrator.logger.handlers:
        #         self.logger.addHandler(handler)
    
    def calculate_initial_allocations(self) -> Dict[str, int]:
        """Calculate initial allocations: fixed allocation + average elastic pool distribution"""
        allocations = self.s0_fixed_allocations.copy()
        
        # Average distribution of elastic pool
        num_dbs = len(self.db_instance_configs)
        if num_dbs > 0:
            elastic_per_db = self.elastic_pool_pages // num_dbs
            for db_id in allocations:
                allocations[db_id] += elastic_per_db
            
            # Handle remainder
            remainder = self.elastic_pool_pages % num_dbs
            if remainder > 0:
                sorted_dbs = sorted(allocations.keys())
                for i in range(remainder):
                    allocations[sorted_dbs[i]] += 1
        
        return allocations
    
    def update_allocations(self, current_metrics: Dict[str, Any], elapsed_in_phase: float) -> Dict[str, int]:
        """Update cache allocations"""
        # Merge EMA update and H-factor calculation to reduce iterations
        h_factors = self._update_ema_and_calculate_h_factors(current_metrics)
        v_factors = self._calculate_v_factors(current_metrics)
        
        # Check epoch transition
        current_time = time.time()
        epoch_changed, change_reason = self.epoch_manager.check_epoch_transition(h_factors, current_time)
        if epoch_changed:
            self.epoch_manager.start_new_epoch(change_reason, current_time)
            # Optional: reset some states of gradient allocator
            # self.gradient_allocator.velocities = {}
        
        # Increment epoch step
        self.epoch_manager.increment_step()
        
        # Calculate scores
        scores = {}
        
        # Check and initialize orchestrator's strategy_states
        if hasattr(self.orchestrator, 'strategy_states'):
            if self.orchestrator.strategy_states is None:
                self.orchestrator.strategy_states = {}
        
        # Complete all operations in single iteration
        for db_id in h_factors:
            # Get current database state
            state = self.ema_states.get(db_id, {})
            hr_current = state.get('ema_hr_slow', 0.0)
            hr_base = state.get('hr_base', 0.5)
            hr_max = state.get('hr_max_adaptive', 1.0)
            
            # Use FinalScore calculator
            h = h_factors[db_id]
            v = v_factors.get(db_id, 0.0)
            final_score, alpha_t = self.final_score_calculator.calculate_final_score(
                db_id, h, v, hr_current, hr_base, hr_max
            )
            
            # Apply non-negative protection
            scores[db_id] = max(0.0, final_score)
            
            # Update alpha_t in ema_states
            if db_id in self.ema_states:
                self.ema_states[db_id]['current_alpha_t'] = alpha_t
            
            # Update orchestrator's strategy_states (if exists)
            if hasattr(self.orchestrator, 'strategy_states'):
                if db_id not in self.orchestrator.strategy_states:
                    self.orchestrator.strategy_states[db_id] = {}
                self.orchestrator.strategy_states[db_id]['current_alpha_t'] = alpha_t
        
        # Record scores
        total_score = sum(scores.values())
        # self.logger.info(f"Score calculation completed: alpha_t={alpha_t:.3f}, total_score={total_score:.3f}")
        # for db_id, score in scores.items():
        #     saturation_conf = self.ema_states[db_id].get('saturation_confidence', 0.0)
        #     v_orig = v_factors.get(db_id, 0.0)
        #     v_eff = v_orig * (1.0 - saturation_conf)
        #     self.logger.info(f"  {db_id}: H={h_factors[db_id]:.3f}, V={v_orig:.3f}, "
        #                    f"SatConf={saturation_conf:.2f}, V_eff={v_eff:.3f}, Score={score:.3f}")
        
        # If all scores are 0, output more debugging information
        # if total_score == 0:
        #     self.logger.warning("Warning: All scores are 0, outputting debug information")
        #     for db_id in h_factors:
        #         state = self.ema_states[db_id]
        #         self.logger.info(f"  {db_id}: ops={state['ema_ops_slow']:.2f}, hr={state['ema_hr_slow']:.3f}, "
        #                        f"hr_base={state['hr_base']:.3f}, hr_max={state['hr_max_adaptive']:.3f}")
        #     # Check raw metrics
        #     self.logger.info("Current raw metrics:")
        #     for db_id, metrics in current_metrics.items():
        #         ops_count = metrics.get('ops_count', 0)
        #         ops = ops_count / self.reporting_interval if self.reporting_interval > 0 else 0
        #         hr = metrics.get('hit_rate', metrics.get('cache_hit_rate', 0))
        #         self.logger.info(f"  {db_id}: raw_ops={ops:.2f}, raw_hr={hr:.3f}")
        
        # Record H and V factors for plotting
        self._last_h_factors = h_factors
        self._last_v_factors = v_factors
        self._last_decision_v_factors = v_factors  # Attribute name expected by experiment manager
        
        # Use gradient allocator to calculate new allocations
        new_allocations = self.gradient_allocator.calculate_allocations(
            list(scores.keys()),
            scores,
            self.orchestrator.db_current_page_allocations,
            self.s0_fixed_allocations,
            self.total_pages
        )
        
        # Update state
        self._update_allocation_state(new_allocations)
        
        return new_allocations
    
    def _update_ema_metrics(self, current_metrics: Dict[str, Any]):
        """Update EMA metrics"""
        alpha_slow = self.hyperparams['alpha_slow']
        alpha_fast = self.hyperparams['alpha_fast']
        
        for db_id, metrics in current_metrics.items():
            if db_id not in self.ema_states:
                continue
                
            state = self.ema_states[db_id]
            
            # Get raw metrics
            # Compatible with experiment_manager data format
            ops_count = metrics.get('ops_count', 0)
            ops = ops_count / self.reporting_interval if self.reporting_interval > 0 else 0
            
            # hit_rate may come from cache_hit_rate field
            hit_rate = metrics.get('hit_rate', metrics.get('cache_hit_rate', 0))
            
            # Apply hit rate cap (0.99) to avoid statistical anomalies
            hit_rate_capped = min(hit_rate, 0.99)
            if hit_rate > 0.99:
                pass  # Debug log removed: hit rate capping
            
            # Update EMA
            if state['ema_ops_slow'] == 0 and ops > 0:
                # Initialize - only initialize when there are actual operations
                state['ema_ops_slow'] = ops
                state['ema_ops_fast'] = ops
                state['ema_hr_slow'] = hit_rate_capped
                state['ema_hr_fast'] = hit_rate_capped
                # Debug log removed: EMA initialization
            elif ops > 0:  # Only update when there are operations
                # Update
                state['ema_ops_slow'] = alpha_slow * ops + (1 - alpha_slow) * state['ema_ops_slow']
                state['ema_ops_fast'] = alpha_fast * ops + (1 - alpha_fast) * state['ema_ops_fast']
                state['ema_hr_slow'] = alpha_slow * hit_rate_capped + (1 - alpha_slow) * state['ema_hr_slow']
                state['ema_hr_fast'] = alpha_fast * hit_rate_capped + (1 - alpha_fast) * state['ema_hr_fast']
            
            # Update hr_max_adaptive (using capped value)
            if hit_rate_capped > state['hr_max_adaptive']:
                state['hr_max_adaptive'] = hit_rate_capped
                # Debug log removed: hr_max_adaptive update
    
    def _calculate_h_factors(self, current_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate horizontal factors (current efficiency)"""
        h_factors = {}
        
        # Collect all ops values for normalization
        all_ops = []
        for db_id in self.ema_states:
            ops = self.ema_states[db_id]['ema_ops_slow']
            all_ops.append(ops)
        
        # Calculate normalization parameters
        if all_ops:
            max_ops = max(all_ops)
            min_ops = min(all_ops)
        else:
            max_ops = min_ops = 0
        
        # Calculate H factor for each database
        for db_id in self.ema_states:
            state = self.ema_states[db_id]
            
            # Normalize ops
            ops = state['ema_ops_slow']
            if max_ops > min_ops:
                norm_ops = (ops - min_ops) / (max_ops - min_ops)
            else:
                norm_ops = 0.5
            
            # Normalize hit rate (using hybrid normalization method)
            hr = state['ema_hr_slow']
            hr_base = state['hr_base']
            hr_max = state['hr_max_adaptive']
            
            # Hybrid normalization: more tolerant of hit rates below baseline
            if hr < hr_base:
                # Below baseline, linearly map to [0, 0.5]
                hr_norm = 0.5 * (hr / hr_base) if hr_base > 0 else 0.0
            else:
                # Above baseline, map to [0.5, 1.0]
                if hr_max > hr_base:
                    hr_norm = 0.5 + 0.5 * (hr - hr_base) / (hr_max - hr_base)
                else:
                    hr_norm = 0.5  # If hr_max not yet updated, at least give 0.5
            
            hr_norm = max(0.0, min(1.0, hr_norm))
            
            # H = normalized ops * normalized hit rate
            h_factors[db_id] = norm_ops * hr_norm
            
            # Debug information
            # Debug log removed: H-factor calculation details
            #                     f"hr_norm={hr_norm:.3f}, H={h_factors[db_id]:.3f}")
        
        return h_factors
    
    def _update_ema_and_calculate_h_factors(self, current_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Merge EMA update and H factor calculation to reduce iterations"""
        alpha_slow = self.hyperparams['alpha_slow']
        alpha_fast = self.hyperparams['alpha_fast']
        h_factors = {}
        all_ops = []
        
        # First iteration: update EMA and collect ops values
        for db_id, metrics in current_metrics.items():
            if db_id not in self.ema_states:
                continue
                
            state = self.ema_states[db_id]
            
            # Get raw metrics
            ops_count = metrics.get('ops_count', 0)
            ops = ops_count / self.reporting_interval if self.reporting_interval > 0 else 0
            hit_rate = metrics.get('hit_rate', metrics.get('cache_hit_rate', 0))
            hit_rate_capped = min(hit_rate, 0.99)
            
            # Update EMA
            if state['ema_ops_slow'] == 0 and ops > 0:
                state['ema_ops_slow'] = ops
                state['ema_ops_fast'] = ops
                state['ema_hr_slow'] = hit_rate_capped
                state['ema_hr_fast'] = hit_rate_capped
            elif ops > 0:
                state['ema_ops_slow'] = alpha_slow * ops + (1 - alpha_slow) * state['ema_ops_slow']
                state['ema_ops_fast'] = alpha_fast * ops + (1 - alpha_fast) * state['ema_ops_fast']
                state['ema_hr_slow'] = alpha_slow * hit_rate_capped + (1 - alpha_slow) * state['ema_hr_slow']
                state['ema_hr_fast'] = alpha_fast * hit_rate_capped + (1 - alpha_fast) * state['ema_hr_fast']
            
            # Update hr_max_adaptive
            if hit_rate_capped > state['hr_max_adaptive']:
                state['hr_max_adaptive'] = hit_rate_capped
            
            # Collect ops values for normalization
            all_ops.append(state['ema_ops_slow'])
        
        # Calculate normalization parameters
        if all_ops:
            max_ops = max(all_ops)
            min_ops = min(all_ops)
        else:
            max_ops = min_ops = 0
        
        # Second iteration: calculate H factors
        for db_id in self.ema_states:
            state = self.ema_states[db_id]
            
            # Normalize ops
            ops = state['ema_ops_slow']
            if max_ops > min_ops:
                norm_ops = (ops - min_ops) / (max_ops - min_ops)
            else:
                norm_ops = 0.5
            
            # Normalize hit rate (using hybrid normalization method)
            hr = state['ema_hr_slow']
            hr_base = state['hr_base']
            hr_max = state['hr_max_adaptive']
            
            # Hybrid normalization: more tolerant of hit rates below baseline
            if hr < hr_base:
                # Below baseline, linearly map to [0, 0.5]
                hr_norm = 0.5 * (hr / hr_base) if hr_base > 0 else 0.0
            else:
                # Above baseline, map to [0.5, 1.0]
                if hr_max > hr_base:
                    hr_norm = 0.5 + 0.5 * (hr - hr_base) / (hr_max - hr_base)
                else:
                    hr_norm = 0.5  # If hr_max not yet updated, at least give 0.5
            
            hr_norm = max(0.0, min(1.0, hr_norm))
            
            # H = normalized ops * normalized hit rate
            h_factors[db_id] = norm_ops * hr_norm
        
        return h_factors
    
    def _calculate_v_factors(self, current_metrics: Dict[str, Any]) -> Dict[str, float]:
        """Calculate vertical factors (V factors) using EMA smoothing and apply robust statistical normalization"""
        v_factors_raw = {}
        non_zero_v_values = []
        
        # Increment tuning cycle count
        self.tuning_cycle_count += 1
        
        # Single iteration: calculate raw V factors for databases in incoming metrics and collect non-zero values
        for db_id in current_metrics:
            if db_id not in self.ema_states:
                continue
            state = self.ema_states[db_id]
            
            # Calculate raw gradient
            gradient = self._compute_gradient(db_id, state)
            
            # Use adaptive alpha with asymmetric smoothing
            adaptive_alpha = self._calculate_adaptive_v_alpha(gradient, state['v_ema'])
            
            # EMA smooth V factor (using adaptive alpha)
            old_v_ema = state['v_ema']
            state['v_ema'] = adaptive_alpha * gradient + (1 - adaptive_alpha) * old_v_ema
            
            # Record alpha changes for key databases
            # if db_id in ['db_high_priority', 'db_medium_priority'] and abs(gradient) > 1e-6:
            #     self.logger.info(f"DB[{db_id}] V factor asymmetric smoothing: "
            #                    f"gradient={gradient:.6f}, old_ema={old_v_ema:.6f}, "
            #                    f"adaptive_alpha={adaptive_alpha:.3f}, new_ema={state['v_ema']:.6f}")
            
            # Store raw values temporarily
            v_factors_raw[db_id] = state['v_ema']
            
            # Also collect non-zero V factors
            if abs(state['v_ema']) > self.v_threshold:
                non_zero_v_values.append(abs(state['v_ema']))
        
        # Calculate robust scaling factor - use O(K) maximum absolute value method instead of percentile
        if len(non_zero_v_values) >= 2:  # Need at least 2 non-zero values to calculate statistics
            # O(K) - find maximum value, faster than percentile for active sets with same effect
            # Avoids overhead of creating NumPy arrays and potential GC delays
            max_abs_v = max(non_zero_v_values)  # non_zero_v_values are already absolute values
            
            # Calculate scaling factor, target is to map maximum value to 0.1
            # This way all V factors will be in [-0.1, 0.1] range, better balanced with H factors [0, 1]
            if max_abs_v > 1e-6:  # Avoid division by zero
                scale_factor = 0.1 / max_abs_v
                # Debug log removed: V-factor robust normalization
            else:
                scale_factor = 1.0  # If maximum value too small, don't scale
                # Debug log removed: V-factor max value too small
        else:
            scale_factor = 1.0  # Too few samples, don't scale
            # Debug log removed: V-factor non-zero samples too few
        
        # Apply scaling and dead zone processing (process directly on original dictionary)
        v_factors = {}
        for db_id, v_raw in v_factors_raw.items():
            # Apply scaling
            v_scaled = v_raw * scale_factor
            
            # Dead zone threshold processing (based on scaled values)
            if abs(v_scaled) < self.v_threshold:
                v_factors[db_id] = 0.0
                # if db_id in ['db_high_priority', 'db_medium_priority'] and abs(v_raw) > 0:
                #     # Debug log removed: V-factor dead zone filtering
            else:
                v_factors[db_id] = v_scaled
            
            # Record debug information
            # if db_id in ['db_high_priority', 'db_medium_priority']:
            #     state = self.ema_states[db_id]
            #     gradient = self._compute_gradient(db_id, state)
            #     # Debug log removed: V-factor calculation details
        
        # Step 5: Update saturation confidence (no longer freeze V factors)
        self._update_saturation_confidence(v_factors)
        
        return v_factors
    
    def _estimate_measurement_noise(self, db_id: str, state: Dict[str, Any], 
                                   current_metrics: Dict[str, Any]) -> float:
        """Estimate noise level of gradient measurement
        
        Evaluate current measurement reliability based on multiple factors:
        - ops count (sample size)
        - system stability (fast/slow EMA difference)
        - timing factors (just adjusted vs stable running)
        - latency conditions (network jitter)
        """
        # Base noise level
        base_noise = self.hyperparams.get('kalman_base_noise', 0.01)
        noise = base_noise
        
        # Get statistics for this database
        db_stats = current_metrics.get(db_id, {})
        
        # 1. ops too low → high noise (insufficient sampling)
        ops_count = db_stats.get('ops_count', 0)
        if ops_count < 5:
            noise += 0.2
        elif ops_count < 10:
            noise += 0.1
        elif ops_count < 20:
            noise += 0.05
            
        # 2. just adjusted → high noise (system not stabilized)
        cycles_since_change = state.get('cycles_since_major_change', 999)
        if cycles_since_change < 1:
            noise += 0.15
        elif cycles_since_change < 2:
            noise += 0.08
        elif cycles_since_change < 3:
            noise += 0.03
            
        # 3. large fast/slow EMA difference → high noise (system changing)
        hr_fast = state.get('ema_hr_fast', 0.5)
        hr_slow = state.get('ema_hr_slow', 0.5)
        ema_diff = abs(hr_fast - hr_slow)
        if ema_diff > 0.1:
            noise += 0.1
        elif ema_diff > 0.05:
            noise += 0.05
            
        # 4. abnormal latency → high noise (possible network issues)
        avg_latency = db_stats.get('avg_latency_ms', 0)
        p95_latency = db_stats.get('p95_latency_ms', 1)
        if p95_latency > 0 and avg_latency > p95_latency * 0.8:
            noise += 0.05
            
        return noise
    
    def _calculate_adaptive_v_alpha(self, v_new: float, v_old: float) -> float:
        """Calculate adaptive V factor EMA coefficient for asymmetric smoothing
        
        Core idea:
        - Fast upward: quickly capture performance improvement signals (higher alpha)
        - Slow downward: avoid overreacting to temporary performance drops (lower alpha)
        
        Args:
            v_new: newly calculated V factor value (gradient)
            v_old: current V factor EMA value
            
        Returns:
            adaptive alpha coefficient
        """
        # Base alpha value (default value)
        base_alpha = self.v_ema_alpha  # 0.3
        
        # Avoid division by zero
        if abs(v_old) < 1e-9:
            return base_alpha
        
        # Calculate change rate
        if v_new > v_old * 1.1:  # Rise over 10%
            # Significant improvement, fast response
            # But not too aggressive, at most 30% increase in alpha
            alpha = min(0.4, base_alpha * 1.3)
            # Debug log removed: V-factor upward trend
            
        elif v_new < v_old * 0.7:  # Drop over 30%
            # Significant decline, strong damping
            decline_ratio = (v_old - v_new) / (abs(v_old) + 1e-9)
            # The steeper the decline, the smaller alpha (but not below 0.05)
            alpha = max(0.05, base_alpha * (1 - decline_ratio * 0.7))
            # Debug log removed: V-factor downward trend
            
        elif v_new < v_old * 0.9:  # Mild decline (10%-30%)
            # Moderate damping
            alpha = base_alpha * 0.7
            # Debug log removed: V-factor mild decline
            
        else:
            # Normal fluctuation or mild rise, use base value
            alpha = base_alpha
        
        return alpha
    
    def _compute_gradient(self, db_id: str, state: Dict[str, Any]) -> float:
        """Calculate gradient (theoretical asymmetric processing)"""
        # Basic initialization check
        if state['last_allocation_pages'] is None or state['last_ema_hit_rate'] is None:
            # First run, no historical data
            return 0.0
        
        # Get current values
        current_allocation = self.orchestrator.db_current_page_allocations.get(db_id, 0)
        current_hr = state['ema_hr_slow']
        
        baseline_allocation = state['last_allocation_pages']
        baseline_hr = state['last_ema_hit_rate']
        
        # Calculate changes
        delta_allocation = current_allocation - baseline_allocation
        delta_hr = current_hr - baseline_hr
        
        # Calculate gradient (prevent division by zero)
        if abs(delta_allocation) < 0.1:
            return 0.0
        
        gradient = delta_hr / delta_allocation
        
        # Theoretical asymmetric processing
        if delta_allocation < 0 and baseline_allocation > 0:  # Cache reduction
            # Adjustment based on relative change rate
            relative_change = abs(delta_allocation) / baseline_allocation
            
            # Use logarithmic function to model impact (instead of hard-coding)
            # Theory: the larger the relative change, the more discount needed
            discount_factor = 1.0 / (1.0 + math.log(1 + relative_change))
            
            # Consider load stability
            if state.get('ema_ops_fast', 0) > 0 and state.get('ema_ops_slow', 0) > 0:
                load_stability = min(state['ema_ops_fast'] / state['ema_ops_slow'], 1.0)
            else:
                load_stability = 1.0
            
            # Comprehensive adjustment
            gradient = gradient * discount_factor * load_stability
            
            # if db_id in ['db_high_priority', 'db_medium_priority']:
            #     # Debug log removed: asymmetric adjustment
            #                     f"discount={discount_factor:.2f}, "
            #                     f"load_stability={load_stability:.2f}")
        
        return gradient
    
    def _update_saturation_confidence(self, v_factors: Dict[str, float]):
        """Update saturation confidence
        
        Core idea:
        - If V factor close to zero and hit rate very high, confidence gradually increases
        - Any non-zero V factor immediately resets confidence to 0
        """
        v_threshold = self.saturation_params['v_near_zero_threshold']
        hr_threshold = self.saturation_params['high_hit_rate_threshold']
        conf_increment = self.saturation_params['confidence_increment']
        
        for db_id, v_factor in v_factors.items():
            state = self.ema_states[db_id]
            hr = state.get('ema_hr_slow', 0.0)
            
            # Check if saturation conditions are met
            if abs(v_factor) < v_threshold and hr > hr_threshold:
                # Received saturation signal, increase confidence
                old_confidence = state.get('saturation_confidence', 0.0)
                new_confidence = min(1.0, old_confidence + conf_increment)
                state['saturation_confidence'] = new_confidence
                
                if db_id in ['db_high_priority', 'db_medium_priority']:
                    # Debug log removed: saturation confidence increase
                    pass
            else:
                # Received non-saturation signal, immediately reset confidence
                if state.get('saturation_confidence', 0.0) > 0:
                    if db_id in ['db_high_priority', 'db_medium_priority']:
                        # Debug log removed: saturation confidence reset
                        pass
                    state['saturation_confidence'] = 0.0
    
    def _calculate_single_h_factor(self, db_id: str) -> float:
        """Calculate H factor for single database (for saturation state detection)"""
        state = self.ema_states[db_id]
        
        # Get ops values from all databases for normalization
        all_ops = [self.ema_states[db]['ema_ops_slow'] for db in self.ema_states]
        if all_ops:
            max_ops = max(all_ops)
            min_ops = min(all_ops)
        else:
            max_ops = min_ops = 0
        
        # Normalize ops
        ops = state['ema_ops_slow']
        if max_ops > min_ops:
            norm_ops = (ops - min_ops) / (max_ops - min_ops)
        else:
            norm_ops = 0.5
        
        # Normalize hit rate (using hybrid normalization method)
        hr = state['ema_hr_slow']
        hr_base = state['hr_base']
        hr_max = state['hr_max_adaptive']
        
        # Hybrid normalization: more tolerant of hit rates below baseline
        if hr < hr_base:
            # Below baseline, linearly map to [0, 0.5]
            hr_norm = 0.5 * (hr / hr_base) if hr_base > 0 else 0.0
        else:
            # Above baseline, map to [0.5, 1.0]
            if hr_max > hr_base:
                hr_norm = 0.5 + 0.5 * (hr - hr_base) / (hr_max - hr_base)
            else:
                hr_norm = 0.5  # If hr_max not yet updated, at least give 0.5
        
        hr_norm = max(0.0, min(1.0, hr_norm))
        
        # H = normalized ops * normalized hit rate
        return norm_ops * hr_norm
    
    # _apply_hysteresis method removed - simplified to dead zone threshold processing
    # _is_saturated method removed - using new saturation confidence mechanism
    
    def _calculate_kappa(self, v_factors: Dict[str, float]) -> float:
        """
        Calculate relative variance κ of V factors, measuring statistical reliability of V factors
        
        κ = var_V / (mag_V² + var_V + ε)
        κ → 0: V factors stable and reliable
        κ → 1: V factors unstable
        """
        if len(v_factors) < 2:
            return 0.5  # Insufficient samples, return neutral value
        
        v_stats = self.alpha_state['v_stats']
        ewma_lambda = self.alpha_state['ewma_lambda']
        
        # Update V factor statistics for each database
        for db_id, v in v_factors.items():
            if db_id not in v_stats:
                # Initialize statistics
                v_stats[db_id] = {
                    'mean_v': v,
                    'var_v': 0.0,
                    'mag_v': abs(v)
                }
            else:
                # EWMA update
                stats = v_stats[db_id]
                old_mean = stats['mean_v']
                
                # Update mean
                stats['mean_v'] = ewma_lambda * old_mean + (1 - ewma_lambda) * v
                
                # Update variance
                stats['var_v'] = ewma_lambda * stats['var_v'] + (1 - ewma_lambda) * (v - stats['mean_v']) ** 2
                
                # Update magnitude mean
                stats['mag_v'] = ewma_lambda * stats['mag_v'] + (1 - ewma_lambda) * abs(v)
        
        # Calculate κ values for each database
        kappa_values = []
        for db_id in v_factors:
            if db_id in v_stats:
                stats = v_stats[db_id]
                var_v = stats['var_v']
                mag_v = stats['mag_v']
                
                # Calculate κ = var_V / (mag_V² + var_V + ε)
                denominator = mag_v ** 2 + var_v + 1e-9
                kappa = var_v / denominator
                kappa_values.append(kappa)
                
                # Debug information
                pass  # Debug log removed: kappa statistics
        
        # Return global κ (average value)
        global_kappa = np.mean(kappa_values) if kappa_values else 0.5
        return global_kappa
    
    def _calculate_alpha_t(self, h_factors: Dict[str, float], v_factors: Dict[str, float]) -> float:
        """
        Alpha_t calculation based on saturation detection and variance analysis
        
        Core idea:
        1. Identify non-saturated databases
        2. Calculate κ based only on non-saturated databases
        3. Phase adaptation: climbing phase depends on κ, saturation phase depends on H factor
        """
        # Check if dynamic weight is enabled
        if self.hyperparams.get('freeze_alpha_t', False):
            return self.hyperparams.get('initial_alpha_t', 0.5)
        
        # 1. Count saturated databases (based on saturation confidence)
        saturated_count = 0
        for db_id in v_factors:
            state = self.ema_states[db_id]
            # Use saturation confidence to judge, if confidence > 0.7 then considered saturated
            if state.get('saturation_confidence', 0.0) > 0.7:
                saturated_count += 1
        
        total_dbs = len(v_factors)
        saturation_ratio = saturated_count / total_dbs if total_dbs > 0 else 0
        
        # self.logger.info(f"Saturation status: {saturated_count}/{total_dbs} databases saturated ({saturation_ratio:.2%})")
        
        # 2. Calculate α_t based on all databases (V=0 for saturated databases is reliable)
        kappa = self._calculate_kappa(v_factors)
        
        # Map to α_t: α_t = 0.5 + 0.3 * κ
        # This way when κ=0 then α_t=0.5, when κ=1 then α_t=0.8
        # More inclined to use H factor (current efficiency)
        alpha_target = 0.5 + 0.3 * kappa
        
        # self.logger.info(f"Based on {total_dbs} databases (including {saturated_count} saturated): κ={kappa:.3f}, target_α_t={alpha_target:.3f}")
        
        # 3. Time smoothing
        alpha_prev = self.alpha_state.get('alpha_prev', 0.5)
        alpha_smoothing = self.alpha_state.get('alpha_smoothing', 0.4)
        
        alpha_t = alpha_smoothing * alpha_target + (1 - alpha_smoothing) * alpha_prev
        
        # 4. Range limitation
        alpha_t = max(0.3, min(0.8, alpha_t))
        
        # 5. Update state
        self.alpha_state['alpha_prev'] = alpha_t
        
        # Debug information
        # self.logger.info(f"Final α_t={alpha_t:.3f} (target={alpha_target:.3f}, after smoothing)")
        
        return alpha_t
    
    # Old simplified activity rules have been replaced by complete scheme based on saturation detection and variance analysis
    
    def _update_allocation_state(self, new_allocations: Dict[str, int]):
        """Update allocation state"""
        old_allocations = self.orchestrator.db_current_page_allocations
        
        for db_id in self.ema_states:
            state = self.ema_states[db_id]
            
            old_alloc = old_allocations.get(db_id, 0)
            new_alloc = new_allocations.get(db_id, 0)
            
            # Initialization check
            if state['last_allocation_pages'] is None:
                # Initialize
                state['last_allocation_pages'] = old_alloc
                state['last_ema_hit_rate'] = state['ema_hr_slow']
            
            # Update baseline every cycle! Use values from previous cycle
            # This way gradient reflects changes from most recent cycle
            state['last_allocation_pages'] = old_alloc
            state['last_ema_hit_rate'] = state['ema_hr_slow']
            
            # Record allocation changes (for logging only)
            if old_alloc > 0:
                change_pct = abs(new_alloc - old_alloc) / old_alloc
                if change_pct >= 0.10:
                    pass  # Debug log removed: significant allocation change
    
    def update_metrics(self, aggregated_stats: Dict[str, Any]):
        """Update metrics (method called by orchestrator)"""
        # S0 strategy doesn't need to do anything here, all updates are handled in update_allocations
        pass
    
    def get_strategy_specific_state(self, db_id: str) -> Dict[str, Any]:
        """Get strategy-specific state for a particular database (for use by orchestrator and experiment_manager)"""
        if db_id in self.ema_states:
            return self.ema_states[db_id].copy()
        return {}