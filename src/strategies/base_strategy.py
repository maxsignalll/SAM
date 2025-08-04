from abc import ABC, abstractmethod
from typing import Dict, Any
import logging

class BaseStrategy(ABC):
    """
    Abstract base class for all cache allocation strategies.
    It defines the common interface that the ExperimentOrchestrator will use.
    """

    def __init__(self, orchestrator: Any, strategy_name: str, strategy_specific_config: Dict = None):
        """
        Initializes the strategy with a reference to the orchestrator.
        This provides access to the orchestrator's state and configuration.

        Args:
            orchestrator: A reference to the ExperimentOrchestrator instance.
            strategy_name: The name of the strategy.
            strategy_specific_config: A dictionary with configuration specific to this strategy.
        """
        self.orchestrator = orchestrator
        self.strategy_name = strategy_name
        self.logger = orchestrator.logger
        self.config = orchestrator.config
        self.strategy_specific_config = strategy_specific_config or {}
        
        # Used for decision time measurement in scalability experiments
        self.pure_decision_time = None
        
        # --- Add logging here ---
        self.logger.debug(f"BaseStrategy '{self.strategy_name}' initialized.")
        self.logger.debug(f"Strategy specific config: {self.strategy_specific_config}")

        self.system_config = orchestrator.system_config
        self.db_instances = orchestrator.db_instances
        self.general_setup = orchestrator.general_setup
        
        # Calculate and store the total available pages for all strategies to use
        total_ram_mb = self.system_config.get("total_ram_for_cache_mb", 32)
        page_size = self.system_config.get("page_size_bytes", 4096)
        self.total_pages = (total_ram_mb * 1024 * 1024) // page_size
        self.page_size_bytes = page_size
        self.logger.debug(f"Base strategy initialized. Total RAM: {total_ram_mb}MB, Page Size: {page_size}B -> Total Pages: {self.total_pages}")

        self.db_instance_configs = self.config.get("database_instances", [])

    @abstractmethod
    def calculate_initial_allocations(self) -> Dict[str, int]:
        """
        Calculates the initial cache page allocations for each database.
        This method contains the logic that was previously in the large if/elif
        block inside `_calculate_initial_cache_allocations`.

        Returns:
            A dictionary mapping db_id to its initial page allocation.
        """
        pass

    def update_allocations(self, current_metrics: Dict[str, Any], elapsed_in_phase: float) -> Dict[str, int]:
        """
        Updates cache allocations based on current metrics.
        This is the primary method for dynamic strategies. Static strategies
        will rely on this default implementation, which does nothing.
        """
        # Default behavior for static strategies: log that it was called but do nothing.
        self.logger.debug(f"Base 'update_allocations' called at {elapsed_in_phase:.2f}s, no dynamic adjustment will be performed.")
        return self.orchestrator.db_current_page_allocations

    def cleanup(self):
        """
        Provides a hook for strategies to clean up any resources they might hold,
        like file handles or separate threads. Called by the orchestrator at the
        end of an experiment run for a specific strategy.
        """
        # Default is to do nothing.
        pass 