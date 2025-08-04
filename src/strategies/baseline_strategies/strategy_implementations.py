# S0 strategy has been modularized to src/strategies/s0_strategy/ directory
# Import kept here for compatibility

from .s0_strategy import S0_EMG_AS_Strategy

# All baseline strategies (B1-B12) have been moved to src/strategies/baseline_strategies/
# For compatibility, you can still import them through:
# from src.strategies.baseline_strategies import B1_StaticAverageStrategy, etc.
# or through the main __init__.py file:
# from src.strategies import B1_StaticAverageStrategy, etc.