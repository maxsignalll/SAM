from .base_strategy import BaseStrategy

# S0 strategy version selection - see s0_strategy/S0_VERSIONS_README.md
# Uncomment one of the lines below to switch S0 strategy implementation version:

# Base version - simple and stable, serves as base class for other versions
# from .s0_strategy.s0_core_strategy_simple import S0CoreStrategySimple as S0_EMG_AS_Strategy

# Performance optimized version - for performance testing
# from .s0_strategy.s0_core_strategy_simple_vectorized import S0CoreStrategySimpleVectorized as S0_EMG_AS_Strategy
# from .s0_strategy.s0_core_strategy_simple_optimized import S0CoreStrategySimpleOptimized as S0_EMG_AS_Strategy

# Experimental version - exploratory optimizations
# from .s0_strategy.s0_wild_optimization import S0WildOptimization as S0_EMG_AS_Strategy
# from .s0_strategy.s0_core_strategy_simple_active_set import S0CoreStrategySimpleActiveSet as S0_EMG_AS_Strategy
# from .s0_strategy.s0_core_strategy_simple_active_set_merged import S0CoreStrategySimpleActiveSetMerged as S0_EMG_AS_Strategy

# ★ Current production version - active set optimization V2, balancing performance and accuracy
# from .s0_strategy.s0_core_strategy_active_set_v2 import S0CoreStrategyActiveSetV2 as S0_EMG_AS_Strategy
# Use performance optimized version
# from .s0_strategy.s0_core_strategy_active_set_v2_optimized import S0CoreStrategyActiveSetV2Optimized as S0_EMG_AS_Strategy
# Use performance optimized version V3 - further optimize performance peak for 80DB scenario
# from .s0_strategy.s0_core_strategy_active_set_v2_optimized_v3 import S0CoreStrategyActiveSetV2OptimizedV3 as S0_EMG_AS_Strategy
# Import simple version and active set version
from .s0_strategy.s0_core_strategy_simple import S0CoreStrategySimple
from .s0_strategy.s0_core_strategy_active_set_v2_optimized_v3 import S0CoreStrategyActiveSetV2OptimizedV3
# Only import actually existing files
# Other versions have been cleaned up
# from .s1_strategy import S1_Predictive_Strategy  # S1 not included in paper submission

# Import all baseline strategies from their new organized modules
from .baseline_strategies import (
    B1_StaticAverageStrategy,
    B2_NoElasticFixedByPriorityStrategy,
    B3_NoFixedElasticByPriorityStrategy,
    B4_IndividualOptimizedCacheStrategy,
    B5_SimulatedGlobalLruStrategy,
    B6_DataSizeProportionalStaticStrategy,
    B7_Dynamic_Need_Strategy,
    B8_Ablation_Efficiency_Only_Strategy,
    B9_Ablation_EMG_AS_Single_EMA_Strategy,
    B10_Pure_V_Factor_Strategy,
    B11_ML_Driven_Strategy,
    B12_MT_LRU_Inspired_Strategy,
    # B13_Tiered_UCP_Strategy,  # Not included in paper submission
    # B14_Smooth_OPT_Strategy  # Not included in paper submission
)

def create_strategy(strategy_name: str, orchestrator, specific_config: dict) -> BaseStrategy:
    """
    Factory function to create a strategy instance based on its name.

    Args:
        strategy_name: The name of the strategy to create.
        orchestrator: The ExperimentOrchestrator instance.
        specific_config: The configuration dictionary specific to this strategy.

    Returns:
        An instance of a class that inherits from BaseStrategy.
    """
    
    # Special handling for S0_EMG_AS, select implementation based on scalability_experiment config
    if strategy_name == "S0_EMG_AS":
        # Get scalability_experiment configuration
        scalability_config = orchestrator.config.get("ycsb_general_config", {}).get("scalability_experiment", {})
        scalability_enabled = scalability_config.get("enabled", False)
        
        # Select S0 strategy implementation based on configuration
        if scalability_enabled:
            # Scalability experiment: use active set version
            return S0CoreStrategyActiveSetV2OptimizedV3(orchestrator, strategy_name, specific_config)
        else:
            # Non-scalability experiment: use simple version (no active set)
            return S0CoreStrategySimple(orchestrator, strategy_name, specific_config)
    
    strategies = {
        # S0: Our main proposal (kept for compatibility, but not actually used)
        "S0_EMG_AS": S0CoreStrategySimple,  # Default to simple version
        
        # S1: Predictive strategy (new)
        # "S1_Predictive": S1_Predictive_Strategy,  # S1 not included in paper submission

        # B1: Baseline - Static Average
        "B1_StaticAverage": B1_StaticAverageStrategy,

        # B2: Baseline - Fixed Priority
        "B2_NoElasticFixedByPriority": B2_NoElasticFixedByPriorityStrategy,
        "B2_FixedPriority": B2_NoElasticFixedByPriorityStrategy,  # Alias for compatibility

        # B3: Ablation - Purely Elastic
        "B3_NoFixedElasticByPriority": B3_NoFixedElasticByPriorityStrategy,

        # B4: Baseline - Selfish / Individual Optimized
        "B4_IndividualOptimizedCache": B4_IndividualOptimizedCacheStrategy,

        # B5: Baseline - Simulated Global LRU
        "B5_SimulatedGlobalLru": B5_SimulatedGlobalLruStrategy,

        # B6: Baseline - Datasize Proportional Static
        "B6_DataSizeProportionalStatic": B6_DataSizeProportionalStaticStrategy,

        # B7: Baseline - Dynamic Need (Miss Rate)
        "B7_DYNAMIC_NEED": B7_Dynamic_Need_Strategy,

        # B8: Ablation - Efficiency Only
        "B8_EFFICIENCY_ONLY": B8_Ablation_Efficiency_Only_Strategy,

        # B9: Ablation - S0 with Single-Speed EMA
        "B9_EMG_AS_SINGLE_EMA": B9_Ablation_EMG_AS_Single_EMA_Strategy,

        # B10: Baseline - Pure V Factor (Only Marginal Gain)
        "B10_Pure_V_Factor": B10_Pure_V_Factor_Strategy,

        # B11: Baseline - ML Driven Strategy
        "B11_ML_Driven": B11_ML_Driven_Strategy,

        # B12: Baseline - MT-LRU Inspired (SLA Firefighter)
        "B12_MT_LRU_Inspired": B12_MT_LRU_Inspired_Strategy,
        "B12_SLA_Firefighter": B12_MT_LRU_Inspired_Strategy,  # Alias for compatibility
        
        # B13: Baseline - Tiered UCP Strategy (SLA-aware baseline with pure UCP)
        # "B13_Active_Sampling": B13_Tiered_UCP_Strategy,  # Not included in paper submission
        
        # B14: Baseline - Smooth-OPT (Oracle with gradual adjustment)
        # "B14_Smooth_OPT": B14_Smooth_OPT_Strategy,  # Not included in paper submission
    }
    
    strategy_class = strategies.get(strategy_name)
    
    if not strategy_class:
        # Fallback for unknown or purely static strategies that might not need a class.
        # However, our design mandates a class for every strategy.
        raise ValueError(f"Unknown or unsupported strategy: {strategy_name}")
        
    return strategy_class(
        orchestrator=orchestrator, 
        strategy_name=strategy_name, 
        strategy_specific_config=specific_config
    ) 