"""
S0 ablation baseline strategies

These strategies test individual components of the S0 algorithm by removing
one component at a time while keeping all others intact.
"""

from typing import Any, Dict
from ..s0_strategy.s0_core_strategy_simple import S0CoreStrategySimple


class B3_NoFixedElasticByPriorityStrategy(S0CoreStrategySimple):
    """
    Baseline B3: S0 without fixed allocation mechanism - pure elastic pool allocation
    
    Tests the value of priority-based fixed allocation by using only minimum
    survival pages as fixed allocation, making the elastic pool much larger.
    """
    
    def calculate_initial_allocations(self) -> Dict[str, int]:
        # B3 modification: all DBs get same minimal fixed allocation
        min_pages_per_db = 10
        
        self.s0_fixed_allocations = {}
        total_fixed = 0
        for db_conf in self.db_instance_configs:
            db_id = db_conf["id"]
            self.s0_fixed_allocations[db_id] = min_pages_per_db
            total_fixed += min_pages_per_db
        
        self.elastic_pool_pages = self.total_pages - total_fixed
        
        # Initially distribute elastic pool equally (S0 will redistribute based on performance)
        allocations = self.s0_fixed_allocations.copy()
        num_dbs = len(self.db_instance_configs)
        if num_dbs > 0:
            elastic_per_db = self.elastic_pool_pages // num_dbs
            for db_id in allocations:
                allocations[db_id] += elastic_per_db
            
            remainder = self.elastic_pool_pages % num_dbs
            if remainder > 0:
                sorted_dbs = sorted(allocations.keys())
                for i in range(remainder):
                    allocations[sorted_dbs[i]] += 1
        
        self.logger.info(f"B3 initialization: total_pages={self.total_pages}, "
                        f"fixed_allocation={total_fixed}, elastic_pool={self.elastic_pool_pages}")
        
        return allocations


class B8_Ablation_Efficiency_Only_Strategy(S0CoreStrategySimple):
    """
    Baseline B8: S0 without V-factor - pure H-factor (efficiency only) strategy
    
    Tests the value of marginal gain consideration by forcing V=0,
    making allocation decisions based only on current efficiency (ops * hit_rate).
    """
    
    def _calculate_v_factors(self, current_metrics: Dict[str, Any]) -> Dict[str, float]:
        # B8 core modification: disable V-factor calculation
        zero_v_factors = {}
        for db_id in self.ema_states:
            zero_v_factors[db_id] = 0.0
        
        return zero_v_factors
    
    def _calculate_alpha_t(self, h_factors: Dict[str, float], v_factors: Dict[str, float]) -> float:
        # Force alpha_t=1.0 so score = 1.0*H + 0.0*V = H
        return 1.0


class B9_Ablation_EMG_AS_Single_EMA_Strategy(S0CoreStrategySimple):
    """
    Baseline B9: S0 with all EMAs using fast parameters (alpha=0.7)
    
    Tests the value of dual-speed EMA by forcing all EMAs to use fast alpha,
    making the strategy more reactive but potentially less stable.
    """
    
    def _update_ema_metrics(self, current_metrics: Dict[str, Any]):
        # B9 modification: use fast alpha for both slow and fast EMAs
        alpha_fast = self.hyperparams['alpha_fast']
        alpha_slow_replaced = alpha_fast  # Override slow with fast
        
        for db_id, metrics in current_metrics.items():
            if db_id not in self.ema_states:
                continue
                
            state = self.ema_states[db_id]
            
            ops_count = metrics.get('ops_count', 0)
            ops = ops_count / self.reporting_interval if self.reporting_interval > 0 else 0
            hit_rate = metrics.get('hit_rate', metrics.get('cache_hit_rate', 0))
            
            # Update all EMAs with fast parameter for quicker response
            if state['ema_ops_slow'] == 0 and ops > 0:
                state['ema_ops_slow'] = ops
                state['ema_ops_fast'] = ops
                state['ema_hr_slow'] = hit_rate
                state['ema_hr_fast'] = hit_rate
            elif ops > 0:
                state['ema_ops_slow'] = alpha_slow_replaced * ops + (1 - alpha_slow_replaced) * state['ema_ops_slow']
                state['ema_ops_fast'] = alpha_fast * ops + (1 - alpha_fast) * state['ema_ops_fast']
                state['ema_hr_slow'] = alpha_slow_replaced * hit_rate + (1 - alpha_slow_replaced) * state['ema_hr_slow']
                state['ema_hr_fast'] = alpha_fast * hit_rate + (1 - alpha_fast) * state['ema_hr_fast']
            
            if hit_rate > state['hr_max_adaptive']:
                state['hr_max_adaptive'] = hit_rate


class B10_Pure_V_Factor_Strategy(S0CoreStrategySimple):
    """
    Baseline B10: Pure V-factor (marginal gain) strategy - S0 without H-factor
    
    Tests allocation based purely on marginal gains, ignoring current efficiency.
    Note: This is NOT a UCP implementation (which uses shadow tags for prediction).
    """
    
    def _calculate_h_factors(self, current_metrics: Dict[str, Any]) -> Dict[str, float]:
        # B10 core modification: disable H-factor calculation
        zero_h_factors = {}
        for db_id in self.ema_states:
            zero_h_factors[db_id] = 0.0
        
        return zero_h_factors
    
    def _calculate_alpha_t(self, h_factors: Dict[str, float], v_factors: Dict[str, float]) -> float:
        # Force alpha_t=0.0 so score = 0.0*H + 1.0*V = V
        return 0.0