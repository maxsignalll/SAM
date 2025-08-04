# Baseline Strategies Package
"""
This package contains all baseline strategy implementations organized by complexity and type.
"""

# Import all baseline strategies
from .static_strategies import (
    B1_StaticAverageStrategy,
    B2_NoElasticFixedByPriorityStrategy,
    B6_DataSizeProportionalStaticStrategy
)

from .simple_dynamic_strategies import (
    B4_IndividualOptimizedCacheStrategy,
    B5_SimulatedGlobalLruStrategy,
    B7_Dynamic_Need_Strategy
)

from .advanced_strategies import (
    B11_ML_Driven_Strategy,
    B12_MT_LRU_Inspired_Strategy
)

from .s0_ablation_strategies import (
    B3_NoFixedElasticByPriorityStrategy,
    B8_Ablation_Efficiency_Only_Strategy,
    B9_Ablation_EMG_AS_Single_EMA_Strategy,
    B10_Pure_V_Factor_Strategy
)

# from .b13_tiered_ucp import (
#     B13_Tiered_UCP_Strategy
# )
# 
# from .b14_smooth_opt import (
#     B14_Smooth_OPT_Strategy
# )

# Export all strategy classes
__all__ = [
    # Static strategies
    'B1_StaticAverageStrategy',
    'B2_NoElasticFixedByPriorityStrategy', 
    'B6_DataSizeProportionalStaticStrategy',
    
    # Simple dynamic strategies
    'B4_IndividualOptimizedCacheStrategy',
    'B5_SimulatedGlobalLruStrategy',
    'B7_Dynamic_Need_Strategy',
    
    # Advanced strategies
    'B11_ML_Driven_Strategy',
    'B12_MT_LRU_Inspired_Strategy',
    
    # S0 ablation strategies
    'B3_NoFixedElasticByPriorityStrategy',
    'B8_Ablation_Efficiency_Only_Strategy',
    'B9_Ablation_EMG_AS_Single_EMA_Strategy',
    'B10_Pure_V_Factor_Strategy',
    
    # Active exploration strategy
    # 'B13_Tiered_UCP_Strategy',  # Not included in paper submission
    
    # Oracle-based strategy
    # 'B14_Smooth_OPT_Strategy'  # Not included in paper submission
]