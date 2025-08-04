import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import os

# 1. Move ConfigError here since it's closely related to config loading
class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

# 2. Define a data class to hold the final processed config for clearer structure
@dataclass
class ProcessedConfig:
    """Holds the fully processed and validated configuration."""
    raw_config: Dict[str, Any]
    general_setup: Dict[str, Any]
    system_config: Dict[str, Any]
    ycsb_general_config: Dict[str, Any]
    db_instance_configs: List[Dict[str, Any]]
    active_strategy_name: str
    strategy_specific_config: Dict[str, Any]
    dynamic_workload_phases: Optional[List[Dict[str, Any]]] = None


class ConfigLoader:
    """
    Handles loading, parsing, and processing the experiment configuration file.
    """
    def __init__(self, config_file_path: str):
        if not os.path.exists(config_file_path):
            raise ConfigError(f"Configuration file not found: {config_file_path}")
        self.config_file_path = config_file_path
        self.config = self._load_config()

    def _load_config(self) -> dict:
        try:
            with open(self.config_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ConfigError(f"Failed to load or parse configuration file: {self.config_file_path} - {e}")

    def get_full_config(self) -> dict:
        """Return the complete configuration dictionary."""
        return self.config

    def _load_raw_config(self) -> dict:
        """Load raw configuration dictionary, prioritizing preloaded config."""
        if self.preloaded_config:
            # If preloaded config is provided, use it directly
            return self.preloaded_config
        
        # Otherwise, load from file
        if not os.path.exists(self.config_file_path):
            raise ConfigError(f"Configuration file not found: {self.config_file_path}")
        
        try:
            with open(self.config_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigError(f"Failed to parse configuration file: {self.config_file_path} - {e}")
        except Exception as e:
            # Catch other possible exceptions, like permission issues
            raise ConfigError(f"Unknown error occurred while loading configuration file: {self.config_file_path} - {e}")

    def load_and_process(self) -> ProcessedConfig:
        """
        The main method to orchestrate the configuration processing.
        """
        general_setup = self.raw_config.get("general_experiment_setup", {})
        
        active_strategy_name = self._determine_active_strategy(general_setup)
        
        strategy_specific_config = self._load_strategy_specific_config()

        db_instance_configs = self._process_db_instances(active_strategy_name, strategy_specific_config)

        return ProcessedConfig(
            raw_config=self.raw_config,
            general_setup=general_setup,
            system_config=self.raw_config.get("system_sqlite_config", {}),
            ycsb_general_config=self.raw_config.get("ycsb_benchmark_config", {}),
            db_instance_configs=db_instance_configs,
            active_strategy_name=active_strategy_name,
            strategy_specific_config=strategy_specific_config,
            dynamic_workload_phases=self.raw_config.get("dynamic_workload_phases")
        )

    def _determine_active_strategy(self, general_setup: Dict[str, Any]) -> str:
        """Determines the active strategy name using the lookup ID."""
        if not self.strategy_id:
             raise ConfigError("A 'strategy_id_for_config_lookup' must be provided to determine the active strategy.")
        
        strategy_config = self._load_strategy_specific_config()
        
        # Uniformly get the final strategy name from inside the strategy block
        strategy_name = strategy_config.get("name")
        if not strategy_name:
            strategy_name = strategy_config.get("strategy_name") # Fallback for older format

        if not strategy_name:
            raise ConfigError(f"Could not determine strategy name from config block for ID '{self.strategy_id}'")

        self.logger.info(f"Determined active strategy: '{strategy_name}' from lookup ID: '{self.strategy_id}'")
        return strategy_name


    def _load_strategy_specific_config(self) -> Dict[str, Any]:
        """Loads the configuration block for the active strategy using its ID."""
        strategy_collection_key = "strategy_configurations"
        lookup_id = self.strategy_id
        
        if not lookup_id:
            raise ConfigError("Cannot load strategy-specific config without 'strategy_id_for_config_lookup'.")

        strategy_lookup_table = self.raw_config.get("strategy_lookup", {})
        config_key_name = strategy_lookup_table.get(lookup_id)

        if not config_key_name:
             # If not in lookup table, assume lookup_id (e.g. "S0") is a prefix of config block key
             config_key_name = lookup_id
             self.logger.debug(f"Strategy ID '{lookup_id}' not in lookup table. Assuming it's a prefix.")
        
        if strategy_collection_key in self.raw_config:
            strategy_configs = self.raw_config[strategy_collection_key]
            
            # Try direct key matching
            if config_key_name in strategy_configs:
                self.logger.info(f"Loaded strategy config using direct key: '{config_key_name}' for lookup ID: '{lookup_id}'.")
                return strategy_configs[config_key_name]
            
            # If direct match fails, try prefix matching
            for config_key, config_block in strategy_configs.items():
                if config_key.startswith(config_key_name):
                    self.logger.info(f"Loaded strategy config using prefix '{config_key_name}' on key '{config_key}' for lookup ID: '{lookup_id}'.")
                    return config_block

        raise ConfigError(f"Strategy with lookup ID '{lookup_id}' (maps to key '{config_key_name}') not found in '{strategy_collection_key}'.")
        
    def _process_db_instances(self, active_strategy_name: str, strategy_specific_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Processes database instances, handling special cases like B5 consolidation.
        """
        original_db_instance_configs = list(self.raw_config.get("database_instances", []))

        if active_strategy_name != "SIMULATED_GLOBAL_LRU_OR_LFU":
            return original_db_instance_configs

        # B5 (SIMULATED_GLOBAL_LRU_OR_LFU) strategy active. Consolidate DB instances.
        self.logger.info("B5 strategy active. Consolidating DB instances.")
        if not original_db_instance_configs:
            raise ConfigError("B5 is active, but no database instances found to consolidate.")

        total_record_count = sum(db.get("ycsb_initial_record_count", 0) for db in original_db_instance_configs)
        total_target_tps = sum(db.get("target_tps", 0) for db in original_db_instance_configs)
        
        unified_db_id = "db_unified_b5"
        unified_db_config = {
            "id": unified_db_id,
            "db_filename": f"/mnt/share_via_ssh/{unified_db_id}.sqlite",
            "base_priority": 5,
            "ycsb_initial_record_count": total_record_count,
            "num_worker_threads": 1, # Default, might be overridden below
            "target_tps": total_target_tps,
            "experimental_role": "unified_global_lru_instance",
            "description": "Unified instance for B5 Global LRU simulation",
            "tps_control": original_db_instance_configs[0].get("tps_control", {})
        }

        b5_specific_settings = strategy_specific_config.get("b5_config")
        if b5_specific_settings:
            unified_db_config["b5_settings"] = b5_specific_settings
            if "num_workers_per_db" in b5_specific_settings:
                unified_db_config["num_worker_threads"] = b5_specific_settings["num_workers_per_db"]
            self.logger.info(f"B5: Loaded b5_settings into unified_db_config: {b5_specific_settings}")
        else:
            self.logger.warning("B5: 'b5_config' not found in strategy config. B5 may not function as intended.")
        
        self.logger.info(f"B5: Reconfigured to use a single unified database instance '{unified_db_id}'")
        return [unified_db_config]
