import json
import threading
import time
import os
from config_loader import ConfigLoader, ConfigError
from ycsb_utils import  load_initial_data, create_ycsb_table
from priority_calculator import create_priority_calculator
from cache_allocator import create_cache_allocator
from database_manager import check_and_reuse_database, print_database_cache_summary
from database_utils import *
from database_instance import DatabaseInstance
import logging
from strategies import create_strategy
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from pathlib import Path
import uuid
from queue import Queue, Empty, Full
from collections import defaultdict


class ExperimentOrchestrator:

    def __init__(self, final_config: dict):
        if not final_config:
            raise ValueError("Must provide a valid configuration dictionary (final_config).")
        
        self.config = final_config
        
        try:
            # Get core configurations in strict mode
            self.general_setup = self.config["general_experiment_setup"]
            self.system_config = self.config["system_sqlite_config"]
            self.db_instance_configs = self.config["database_instances"]
            self.strategy_configurations = self.config["strategy_configurations"]
            
            # Get corresponding config based on workload type
            workload_type = self.general_setup.get("workload_type", "ycsb")
            if workload_type == "ycsb":
                self.ycsb_general_config = self.config.get("ycsb_general_config", {})
            elif workload_type == "tpcc":
                self.ycsb_general_config = self.config.get("tpcc_general_config", {})
            elif workload_type == "trace":
                self.ycsb_general_config = self.config.get("trace_general_config", {})
            else:
                # Default to try getting ycsb config
                self.ycsb_general_config = self.config.get("ycsb_general_config", {})
                
            # If no workload config found, use empty dict to avoid crash
            if not self.ycsb_general_config:
                # Note: logger not initialized yet, skip warning
                self.ycsb_general_config = {}
            
            # Strictly get from general_setup
            self.active_strategy_id = self.general_setup["active_strategy"]
            self.output_directory = Path(self.general_setup["output_directory"])
            log_level_str = self.general_setup.get("log_level", "INFO").upper() # Log level can have default
            
            # Non-required config, allow empty
            self.dynamic_workload_phases = self.config.get("dynamic_workload_phases")
            
            # Cache miss simulation config
            self.cache_miss_simulation = self.config.get("cache_miss_simulation", {
                "enabled": False,
                "miss_penalty_ms": 50,
                "compensation_method": "batch",
                "compensation_interval_seconds": 8
            })
            
            # Track pending compensation delay to apply at next cycle
            self.pending_compensation_delay = 0.0
            
            # Initialize current phase index (for B14 and other strategies)
            self.current_phase_idx = 0
            
        except KeyError as e:
            raise ConfigError(f"Missing one or more required top-level keys in config: {e}. Please check your config file structure.")

        unique_id = str(uuid.uuid4())[:8]
        strategy_name_for_logger = self.strategy_configurations.get(self.active_strategy_id, {}).get("strategy_name", self.active_strategy_id)
        logger_name = f"{self.__class__.__name__}_{strategy_name_for_logger}_{unique_id}"
        self.logger = logging.getLogger(logger_name)
        
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        log_level = getattr(logging, log_level_str, logging.INFO)
        self.logger.setLevel(log_level)
        ch = logging.StreamHandler()
        formatter = logging.Formatter(f'%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.propagate = False

        self.config_file_path = self.general_setup.get("config_source", "In-Memory") # This can have default
        
        # Handle scalability experiment - dynamically generate background database instances
        self._handle_scalability_experiment()

        self.stats_queue = Queue()
        self.command_queues = {db_conf["id"]: Queue() for db_conf in self.db_instance_configs}
        
        self.strategy_specific_config = self.strategy_configurations.get(self.active_strategy_id, {}).copy()
        
        self.strategy_id = self.active_strategy_id

        if "strategy_name" not in self.strategy_specific_config:
            self.logger.warning(f"'strategy_name' field not found in strategy config '{self.active_strategy_id}'. Using strategy ID as fallback name and injecting into config.")
            self.strategy_specific_config["strategy_name"] = self.active_strategy_id
        
        self.active_strategy_name = self.strategy_specific_config["strategy_name"]
        
        self.db_instances = []
        
        self.logger.info(f"Orchestrator for strategy '{self.active_strategy_name}' initialized with unique logger '{logger_name}'.")

        self.results_df = pd.DataFrame()
        
        self.db_current_page_allocations = {}
        self.db_fixed_page_allocations = {}
        self.strategy_states = {}
        
        # Decision CPU time collector (for scalability experiments)
        self.decision_cpu_times = []
        scalability_config = self.ycsb_general_config.get("scalability_experiment", {})
        self.is_scalability_experiment = scalability_config.get("enabled", False)

        self.strategy_handler = create_strategy(
            strategy_name=self.active_strategy_name,
            orchestrator=self,
            specific_config=self.strategy_specific_config
        )

    # Note: This method is deprecated, actual use is _perform_cache_reallocation
    # Keep this method for reference only, avoid breaking possible external references
    def _request_cache_reallocation(self, aggregated_stats: Dict[str, Any]):
        """
        [Deprecated] Call current strategy to get new cache allocation scheme and distribute commands to database instances.
        Please use _perform_cache_reallocation method instead.
        """
        if not self.strategy_handler or not hasattr(self.strategy_handler, 'update_allocations'):
            self.logger.debug("Current strategy does not support dynamic updates, skipping cache reallocation.")
            return

        # 1. Call strategy's update_allocations method
        self.logger.debug(f"Calling strategy '{self.active_strategy_name}' to calculate new cache allocation.")
        
        # Measure decision time, but note this includes strategy internal logging time
        # For scalability experiments, we need more precise measurement inside strategy
        if self.is_scalability_experiment:
            # Temporarily disable strategy logging for more accurate measurement
            strategy_logger = getattr(self.strategy_handler, 'logger', None)
            if strategy_logger:
                original_level = strategy_logger.level
                strategy_logger.setLevel(logging.CRITICAL)  # Only keep CRITICAL level logs
            
            start_cpu_time = time.process_time()
            new_allocations = self.strategy_handler.update_allocations(aggregated_stats, 0)
            end_cpu_time = time.process_time()
            
            # Restore original log level
            if strategy_logger:
                strategy_logger.setLevel(original_level)
            
            decision_cpu_time = end_cpu_time - start_cpu_time
            
            # If strategy reports pure computation time, use it; otherwise use our measurement
            if hasattr(self.strategy_handler, 'pure_decision_time') and self.strategy_handler.pure_decision_time is not None:
                decision_cpu_time = self.strategy_handler.pure_decision_time
                self.strategy_handler.pure_decision_time = None  # Reset
            
            self._record_decision_cpu_time(decision_cpu_time)
        else:
            new_allocations = self.strategy_handler.update_allocations(aggregated_stats, 0)

        if not new_allocations or new_allocations == self.db_current_page_allocations:
            self.logger.debug("Strategy did not return new allocation scheme or scheme unchanged.")
            return

        self.logger.info(f"Strategy returned new allocation scheme: {new_allocations}")

        # 2. Distribute RECONFIGURE_CACHE commands by sending instructions to worker threads, not direct operation
        for db_id, new_page_size in new_allocations.items():
            if self.db_current_page_allocations.get(db_id) != new_page_size:
                old_pages = self.db_current_page_allocations.get(db_id, 0)
                self.logger.info(f"Requesting cache reconfiguration for DB[{db_id}]: from {old_pages} -> {new_page_size} pages")
                command = {'type': 'RECONFIGURE_CACHE', 'page_count': new_page_size}
                self._broadcast_command(command, specific_db_id=db_id)

        # 3. Update Orchestrator's internal current allocation state
        self.db_current_page_allocations = new_allocations.copy()

    def _record_decision_cpu_time(self, cpu_time: float):
        """Record strategy decision CPU time (only in scalability experiments)"""
        if self.is_scalability_experiment:
            self.decision_cpu_times.append(cpu_time)
            # Output statistics every certain number of times
            if len(self.decision_cpu_times) % 10 == 0:
                avg_time = np.mean(self.decision_cpu_times)
                self.logger.info(f"[Scalability Experiment] Decision CPU time stats: avg={avg_time:.6f}s, samples={len(self.decision_cpu_times)}")

    def _handle_scalability_experiment(self):
        """Handle scalability experiment configuration, dynamically generate background database instances"""
        # Check scalability experiment configuration
        scalability_config = self.ycsb_general_config.get("scalability_experiment", {})
        
        if not scalability_config.get("enabled", False):
            self.logger.debug("Scalability experiment not enabled")
            return
        
        # Get scalability experiment parameters
        total_db_count = scalability_config.get("total_database_count", 50)
        bg_db_prefix = scalability_config.get("background_db_prefix", "db_bg_")
        bg_db_priority = scalability_config.get("background_db_priority", 1)
        bg_db_record_count = scalability_config.get("background_db_record_count", 10000)
        bg_db_path = scalability_config.get("background_db_path", "/mnt/share_via_ssh/")
        
        # Get existing database instances
        existing_db_ids = {db["id"] for db in self.db_instance_configs}
        
        # Count existing background databases and find template database
        existing_bg_dbs = [db for db in self.db_instance_configs if db["id"].startswith(bg_db_prefix)]
        existing_bg_count = len(existing_bg_dbs)
        
        # List all existing background database IDs
        existing_bg_ids = sorted([db["id"] for db in existing_bg_dbs])
        
        # Scan existing background database files in filesystem
        filesystem_bg_dbs = []
        filesystem_bg_ids = []
        if os.path.exists(bg_db_path):
            import glob
            # Search all matching database files
            bg_file_pattern = os.path.join(bg_db_path, f"{bg_db_prefix}*.sqlite")
            bg_files = glob.glob(bg_file_pattern)
            
            for bg_file in bg_files:
                # Extract database ID from filename
                filename = os.path.basename(bg_file)
                if filename.endswith('.sqlite'):
                    db_id = filename[:-7]  # Remove .sqlite suffix
                    # Ensure valid background database ID (starts with prefix and followed by numbers)
                    if db_id.startswith(bg_db_prefix):
                        suffix = db_id[len(bg_db_prefix):]
                        if suffix.isdigit() and db_id not in existing_db_ids:
                            # This is a background database that exists in filesystem but not in configuration
                            filesystem_bg_dbs.append({
                                "id": db_id,
                                "db_filename": bg_file,
                                "base_priority": bg_db_priority,
                                "ycsb_initial_record_count": bg_db_record_count,
                                "experimental_role": "background_noise",
                                "_from_filesystem": True  # Mark as from filesystem
                            })
                            filesystem_bg_ids.append(db_id)
            
            filesystem_bg_ids.sort(key=lambda x: int(x[len(bg_db_prefix):]))
        
        # Merge background databases from configuration and filesystem
        all_existing_bg_dbs = existing_bg_dbs + filesystem_bg_dbs
        all_existing_bg_count = len(all_existing_bg_dbs)
        all_existing_bg_ids = sorted(existing_bg_ids + filesystem_bg_ids, 
                                     key=lambda x: int(x[len(bg_db_prefix):]))
        
        # Calculate number of background databases needed
        main_db_count = len([db for db in self.db_instance_configs if not db["id"].startswith(bg_db_prefix)])
        main_db_ids = sorted([db["id"] for db in self.db_instance_configs if not db["id"].startswith(bg_db_prefix)])
        target_bg_count = total_db_count - main_db_count
        
        self.logger.info(f"Scalability experiment database check:")
        self.logger.info(f"  - Main databases ({main_db_count}): {', '.join(main_db_ids)}")
        self.logger.info(f"  - Background databases from config ({existing_bg_count}): {', '.join(existing_bg_ids) if existing_bg_ids else 'None'}")
        if filesystem_bg_ids:
            self.logger.info(f"  - Background databases found in filesystem ({len(filesystem_bg_ids)}): {', '.join(filesystem_bg_ids)}")
        self.logger.info(f"  - All existing background databases ({all_existing_bg_count}): {', '.join(all_existing_bg_ids) if all_existing_bg_ids else 'None'}")
        
        if target_bg_count <= all_existing_bg_count:
            self.logger.info(f"  - Conclusion: Already have {all_existing_bg_count} background databases, meets target {target_bg_count}, no need to create more")
            # If too many background databases, only use the needed amount
            if all_existing_bg_count > target_bg_count:
                excess_count = all_existing_bg_count - target_bg_count
                self.logger.info(f"  - Note: Background databases exceed target by {excess_count}, will only use first {target_bg_count}")
                
                # Remove excess background databases from configuration
                # Prioritize keeping original config ones, then filesystem ones with smaller numbers
                if existing_bg_count > target_bg_count:
                    # Config background databases already exceed target, keep only first target_bg_count
                    bg_dbs_to_keep = existing_bg_dbs[:target_bg_count]
                    self.db_instance_configs = [db for db in self.db_instance_configs if not db["id"].startswith(bg_db_prefix) or db in bg_dbs_to_keep]
                    self.logger.info(f"  - Removed {existing_bg_count - target_bg_count} excess background databases from config")
                elif existing_bg_count + len(filesystem_bg_dbs) > target_bg_count:
                    # Need to select some from filesystem databases
                    needed_from_filesystem = target_bg_count - existing_bg_count
                    selected_filesystem_dbs = filesystem_bg_dbs[:needed_from_filesystem]
                    self.logger.info(f"  - Selected {len(selected_filesystem_dbs)} background databases from filesystem to add to config")
                    self.db_instance_configs.extend(selected_filesystem_dbs)
                    # Update dynamic workload phase configuration
                    self._update_workload_phases_for_new_dbs(selected_filesystem_dbs, bg_db_prefix)
            else:
                # Exactly equals target count, but may need to add filesystem databases to configuration
                if filesystem_bg_dbs and existing_bg_count < target_bg_count:
                    needed_from_filesystem = target_bg_count - existing_bg_count
                    selected_filesystem_dbs = filesystem_bg_dbs[:needed_from_filesystem]
                    self.logger.info(f"  - Selected {len(selected_filesystem_dbs)} background databases from filesystem to add to config")
                    self.db_instance_configs.extend(selected_filesystem_dbs)
                    # Update dynamic workload phase configuration
                    self._update_workload_phases_for_new_dbs(selected_filesystem_dbs, bg_db_prefix)
                else:
                    self.logger.info(f"  - Config already has enough background databases, no need to add from filesystem")
            return
        
        self.logger.info(f"Scalability experiment configuration:")
        self.logger.info(f"  - Target total database count: {total_db_count}")
        self.logger.info(f"  - Main database count: {main_db_count}")
        self.logger.info(f"  - Background databases from config: {existing_bg_count}")
        self.logger.info(f"  - Background databases from filesystem: {len(filesystem_bg_ids)}")
        self.logger.info(f"  - All existing background databases: {all_existing_bg_count}")
        self.logger.info(f"  - Target background databases: {target_bg_count}")
        self.logger.info(f"  - Need to create background databases: {target_bg_count - all_existing_bg_count}")
        
        # Only add needed filesystem databases to configuration
        if filesystem_bg_dbs:
            # Calculate how many more background databases needed from filesystem
            needed_from_filesystem = min(len(filesystem_bg_dbs), target_bg_count - existing_bg_count)
            if needed_from_filesystem > 0:
                selected_filesystem_dbs = filesystem_bg_dbs[:needed_from_filesystem]
                self.logger.info(f"  - Selected {len(selected_filesystem_dbs)} background databases from filesystem to add to config")
                self.db_instance_configs.extend(selected_filesystem_dbs)
                # Update configuration background database count
                existing_bg_count += len(selected_filesystem_dbs)
                # Note: Don't add to all_existing_bg_dbs again, already merged before
                # all_existing_bg_dbs.extend(selected_filesystem_dbs)  # Removed to avoid duplication
                # all_existing_bg_count = len(all_existing_bg_dbs)     # Removed
        
        # Check existing background database file status
        existing_files_count = 0
        for bg_db in all_existing_bg_dbs:
            if os.path.exists(bg_db["db_filename"]):
                existing_files_count += 1
        
        if all_existing_bg_dbs:
            self.logger.info(f"  - Existing background database file status: {existing_files_count}/{all_existing_bg_count} files exist")
        
        # Find an existing background database as template
        template_bg_db = None
        template_db_file = None
        if all_existing_bg_dbs:
            # Check all existing background databases, find first existing file as template
            for bg_db in all_existing_bg_dbs:
                db_file = bg_db["db_filename"]
                if os.path.exists(db_file):
                    template_bg_db = bg_db
                    template_db_file = db_file
                    self.logger.info(f"  - Using template database: {template_bg_db['id']} ({template_db_file})")
                    break
            
            if not template_bg_db:
                self.logger.warning(f"  - All existing background database files do not exist, will create new databases")
        
        # Generate new background database instances
        new_bg_instances = []
        # Find the maximum existing background database number
        max_existing_bg_num = 0
        if all_existing_bg_ids:
            for bg_id in all_existing_bg_ids:
                try:
                    num = int(bg_id[len(bg_db_prefix):])
                    max_existing_bg_num = max(max_existing_bg_num, num)
                except ValueError:
                    pass
        
        # Generate new background databases starting from max number + 1
        start_num = max_existing_bg_num + 1
        end_num = start_num + (target_bg_count - all_existing_bg_count)
        
        for i in range(start_num, end_num):
            db_id = f"{bg_db_prefix}{i}"
            new_db_filename = f"{bg_db_path}{db_id}.sqlite"
            
            new_instance = {
                "id": db_id,
                "db_filename": new_db_filename,
                "base_priority": bg_db_priority,
                "ycsb_initial_record_count": bg_db_record_count,
                "experimental_role": "background_noise"
            }
            
            # If template database exists, try to copy
            if template_bg_db and os.path.exists(template_db_file):
                try:
                    import shutil
                    # Ensure target directory exists
                    os.makedirs(os.path.dirname(new_db_filename), exist_ok=True)
                    # Copy database file
                    shutil.copy2(template_db_file, new_db_filename)
                    self.logger.debug(f"  - Copying template database to: {new_db_filename}")
                    # Mark this database as "copied" to skip validation in _prepare_databases
                    new_instance["_cloned_from"] = template_bg_db["id"]
                except Exception as e:
                    self.logger.warning(f"  - Failed to copy database ({db_id}): {e}, will create new database")
            
            new_bg_instances.append(new_instance)
            self.logger.debug(f"  - Generated background database config: {db_id}")
        
        # Update database instance list
        self.db_instance_configs.extend(new_bg_instances)
        
        # Update dynamic workload phase configuration
        self._update_workload_phases_for_new_dbs(new_bg_instances, bg_db_prefix)
        
        self.logger.info(f"Scalability experiment: Successfully generated {len(new_bg_instances)} new background database instances")
    
    def _update_workload_phases_for_new_dbs(self, new_db_instances, bg_db_prefix):
        """Update dynamic workload phase configuration, add TPS and access pattern configuration for new databases"""
        if not self.dynamic_workload_phases or not new_db_instances:
            return
        
        for phase in self.dynamic_workload_phases:
            phase_overrides = phase.get("ycsb_config_overrides", {})
            
            # Update tps_distribution_per_db
            tps_dist = phase_overrides.get("tps_distribution_per_db", {})
            for instance in new_db_instances:
                db_id = instance["id"]
                if db_id not in tps_dist:
                    # Set same TPS for new background databases as existing ones
                    tps_dist[db_id] = 1  # Default TPS is 1, consistent with existing background databases
            
            # Update access_pattern_per_db
            access_pattern = phase_overrides.get("access_pattern_per_db", {})
            for instance in new_db_instances:
                db_id = instance["id"]
                if db_id not in access_pattern:
                    # Set uniform distribution pattern for new background databases
                    access_pattern[db_id] = {"distribution": "uniform"}

    def _prepare_databases(self):
        """
        Check all configured databases, create them if they don't exist or are invalid.
        This is a centralized preparation step executed before starting any worker threads.
        """
        self.logger.info("====== Starting database preparation phase ======")
        for db_conf in self.db_instance_configs:
            db_id = db_conf['id']
            self.logger.info(f"--- Checking database: {db_id} ---")
            
            # Merge global YCSB configuration with DB-specific configuration
            final_db_ycsb_config = self.ycsb_general_config.copy()
            final_db_ycsb_config.update(db_conf)
            
            # Map operation_distribution to ycsb_workload_proportions for backward compatibility
            if 'operation_distribution' in final_db_ycsb_config and 'ycsb_workload_proportions' not in final_db_ycsb_config:
                final_db_ycsb_config['ycsb_workload_proportions'] = final_db_ycsb_config['operation_distribution']
            
            # Get workload type
            workload_type = self.config.get('general_experiment_setup', {}).get('workload_type', 'ycsb')
            can_reuse = check_and_reuse_database(db_conf, final_db_ycsb_config, workload_type)
            
            if not can_reuse:
                self.logger.info(f"Database {db_id} is not reusable, will create as needed.")
                try:
                    # Use the class method we added to DatabaseInstance
                    # Pass orchestrator config to support different workload types
                    DatabaseInstance.create_and_load_data(final_db_ycsb_config, self.logger, self.config)
                except Exception as e:
                    self.logger.critical(f"Fatal error occurred while creating database for {db_id}: {e}", exc_info=True)
                    raise  # Cannot continue experiment
            else:
                self.logger.info(f"Database {db_id} validation passed, will be reused。")
        self.logger.info("====== Database preparation phase completed ======")

    def _start_db_instance_threads(self, initial_cache_allocations):
        """Start database instance threads and pass communication queues"""
        
        self.logger.info("Starting database instance threads...")
        
        self.db_instance_threads = {}
        # Core fix: Extract initial access pattern from dynamic phase configuration
        initial_access_pattern = {}
        if self.dynamic_workload_phases:
            first_phase_overrides = self.dynamic_workload_phases[0].get("ycsb_config_overrides", {})
            # We only need distribution and alpha value, not the entire per_db structure
            # Assume all DBs have same initial distribution pattern in first phase
            if "access_pattern_per_db" in first_phase_overrides and first_phase_overrides["access_pattern_per_db"]:
                # Get first database's access pattern as template
                template_pattern = next(iter(first_phase_overrides["access_pattern_per_db"].values()))
                initial_access_pattern['ycsb_request_distribution'] = template_pattern.get('distribution')
                if 'zipf_alpha' in template_pattern:
                    initial_access_pattern['ycsb_zipfian_constant'] = template_pattern.get('zipf_alpha')
                self.logger.info(f"Extracted initial access pattern configuration from first phase: {initial_access_pattern}")

        for db_config in self.db_instance_configs:
            db_id = db_config["id"]
            
            # Get command queue allocated for this DB instance
            command_queue = self.command_queues[db_id]

            # Core fix: Merge global YCSB configuration and instance-specific configuration
            # This ensures all required keys (like proportions and field_count) exist
            final_db_ycsb_config = self.ycsb_general_config.copy()
            final_db_ycsb_config.update(db_config)
            
            # Map operation_distribution to ycsb_workload_proportions for backward compatibility
            if 'operation_distribution' in final_db_ycsb_config and 'ycsb_workload_proportions' not in final_db_ycsb_config:
                final_db_ycsb_config['ycsb_workload_proportions'] = final_db_ycsb_config['operation_distribution']
            
            # Inject extracted initial access pattern to avoid KeyError
            if initial_access_pattern:
                final_db_ycsb_config.update(initial_access_pattern)

            db_instance = DatabaseInstance(
                db_id=db_id,
                db_config=final_db_ycsb_config, # Pass merged complete configuration
                initial_cache_pages=initial_cache_allocations.get(db_id, 1),  # Default at least 1 page
                command_queue=command_queue,
                stats_queue=self.stats_queue,
                orchestrator_config=self.config,
                strategy_id=self.strategy_id # Core fix: pass strategy_id
            )
            
            thread = threading.Thread(target=db_instance.run, name=f"DBInstance_{db_id}")
            thread.daemon = True # Set as daemon thread
            self.db_instance_threads[db_id] = thread
            self.db_instances.append(db_instance) # Keep reference to instance

        for db_id, thread in self.db_instance_threads.items():
            thread.start()
            self.logger.info(f"Database instance thread '{db_id}' has started.")
        
        self.logger.info(f"All database instance threads ({len(self.db_instance_threads)}) have been created and started.")

    def _wait_for_all_workers_ready(self, timeout_per_worker=900):
        """Wait for all worker threads to be ready by listening for 'READY' messages in statistics queue."""
        self.logger.info("Waiting for all database instance worker threads to initialize...")
        
        ready_workers = set()
        expected_workers = set(self.db_instance_threads.keys())
        
        start_time = time.monotonic()
        
        while ready_workers != expected_workers:
            # Calculate remaining waiting time
            elapsed_time = time.monotonic() - start_time
            remaining_time = (len(expected_workers) * timeout_per_worker) - elapsed_time
            
            if remaining_time <= 0:
                self.logger.error(f"Timeout waiting for worker threads to be ready. Expected: {expected_workers}, Actually ready: {ready_workers}")
                # Send stop signal after timeout
                self._broadcast_command({'type': 'STOP'})
                return False 

            try:
                # Use timeout to avoid infinite blocking
                message = self.stats_queue.get(timeout=remaining_time)
                
                if isinstance(message, dict) and message.get('type') == 'READY':
                    worker_id = message.get('db_id')
                    if worker_id in expected_workers:
                        ready_workers.add(worker_id)
                        self.logger.info(f"Worker thread '{worker_id}' confirmed ready. ({len(ready_workers)}/{len(expected_workers)})")
                else:
                    # If receiving other messages while waiting for READY, record but don't process
                    self.logger.debug(f"Received non-READY message while waiting for ready: {message}")

            except Empty:
                self.logger.error(f"Queue empty with timeout while waiting for worker threads. Expected: {expected_workers}, Actually ready: {ready_workers}")
                self._broadcast_command({'type': 'STOP'})
                return False

        self.logger.info(f"All {len(expected_workers)} database worker threads are ready.")
        return True 
    
    def _run_measurement_loop(self):
        """
        Decide whether to run static phase or dynamic phases based on dynamic_workload_phases configuration.
        """
        if self.dynamic_workload_phases:
            self._run_dynamic_phases()
        else:
            self.logger.warning("Configuration file missing 'dynamic_workload_phases' section, will execute a default static phase.")
            self._run_static_phase()

    def _run_static_phase(self):
        """Run a single, longer duration static workload phase."""
        duration = self.general_setup.get("default_static_duration_s")
        self.logger.info(f"Starting default static measurement phase, duration: {duration} seconds.")
        self._process_experiment_phase(duration, "StaticPhase")
        self.logger.info("Static measurement phase completed.")

    def _run_dynamic_phases(self):
        """Execute all dynamic workload phases defined in configuration in sequence."""
        self.logger.info(f"Starting execution of {len(self.dynamic_workload_phases)} dynamic workload phases...")
        total_elapsed_time = 0
        for i, phase_config in enumerate(self.dynamic_workload_phases):
            duration = phase_config["duration_seconds"]
            phase_name = phase_config.get("name")
            
            self.logger.info(f"\n---==> Entering phase: '{phase_name}' (duration: {duration}s) <==---")
            
            # Set current phase index for strategies like B14 to use
            self.current_phase_idx = i
            self.logger.debug(f"Settings current_phase_idx = {i} for phase '{phase_name}'")
            
            # Broadcast RECONFIGURE command for this phase
            reconfigure_command = {'type': 'RECONFIGURE', 'new_config': phase_config}
            self._broadcast_command(reconfigure_command)

            self._process_experiment_phase(duration, phase_name)
            total_elapsed_time += duration

        self.logger.info("All dynamic workload phases completed.")

    def _process_experiment_phase(self, duration: int, phase_name: str):
        """
        Core loop for processing a single experiment phase, including data collection, reporting and cache adjustment.
        This is the common logic called by _run_static_phase and _run_dynamic_phases.
        """
        phase_start_time = time.monotonic()
        phase_end_time = phase_start_time + duration

        reporting_interval = self.general_setup.get("reporting_interval_seconds")
        
        # Batch collection configuration (backward compatible)
        batch_collection_enabled = self.general_setup.get("batch_collection_enabled", False)
        batch_collection_window = self.general_setup.get("batch_collection_window_seconds", reporting_interval)
        
        # Core fix: Use strategy's is_dynamic attribute to determine, not hasattr
        is_dynamic_strategy = self.strategy_handler.is_dynamic
        adjustment_interval = self.strategy_specific_config.get("adjustment_interval_seconds")
        next_adjustment_time = phase_start_time + adjustment_interval if is_dynamic_strategy else float('inf')
        next_report_time = phase_start_time + reporting_interval
        
        # Batch collection status
        batch_window_start = phase_start_time
        batch_window_end = phase_start_time + batch_collection_window
        batch_stats_buffer = []  # Buffer for storing statistics data within batch collection window

        while time.monotonic() < phase_end_time:
            current_time = time.monotonic()

            # 1. Check if reporting time has arrived, unconditionally request statistics data and update timer
            if current_time >= next_report_time:
                self.logger.debug("Reporting period reached, requesting new statistics data.")
                self._broadcast_command({'type': 'REPORT_STATS'})
                next_report_time = current_time + reporting_interval

            # 2. Collect statistics data
            newly_collected_stats = []
            while not self.stats_queue.empty():
                try:
                    msg = self.stats_queue.get_nowait()
                    if msg.get('type') == 'STATS_UPDATE':
                        if batch_collection_enabled:
                            # Batch collection mode: store data in buffer
                            batch_stats_buffer.append(msg)
                        else:
                            # Original mode: process immediately
                            newly_collected_stats.append(msg)
                except Empty:
                    break
            
            # 3. Process batch collection window
            if batch_collection_enabled and current_time >= batch_window_end:
                # Batch window ended, process all data in buffer
                if batch_stats_buffer:
                    self.logger.debug(f"Batch collection window ended, processing {len(batch_stats_buffer)} statistics data entries")
                    aggregated_stats = self._aggregate_interval_stats(batch_stats_buffer)
                    
                    if aggregated_stats:
                        # Call strategy's metrics update
                        if hasattr(self.strategy_handler, 'update_metrics'):
                            self.logger.debug(f"Calling update_metrics for strategy {self.active_strategy_name}")
                            self.strategy_handler.update_metrics(aggregated_stats)

                        # Unified reporting and recording
                        elapsed_in_phase = current_time - phase_start_time
                        self._report_interval_stats(aggregated_stats, elapsed_in_phase, phase_name)
                        self._log_interval_to_dataframe(aggregated_stats, elapsed_in_phase, phase_name)
                        
                        # EvaluationCacheAdjustment
                        if current_time >= next_adjustment_time:
                            self._perform_cache_reallocation(aggregated_stats, elapsed_in_phase)
                            next_adjustment_time = current_time + adjustment_interval
                
                # Reset batch collection window
                batch_stats_buffer = []
                
                # Apply pending compensation delay at the beginning of new measurement cycle
                if self.pending_compensation_delay > 0:
                    self.logger.info(f"Applying cache miss compensation delay: {self.pending_compensation_delay:.3f}s")
                    time.sleep(self.pending_compensation_delay)
                    self.pending_compensation_delay = 0.0
                
                batch_window_start = current_time
                batch_window_end = current_time + batch_collection_window
            
            # 4. Original mode processing (non-batch collection)
            elif not batch_collection_enabled and newly_collected_stats:
                aggregated_stats = self._aggregate_interval_stats(newly_collected_stats)
                
                if aggregated_stats:
                    # Call strategy's metrics update (high frequency)
                    if hasattr(self.strategy_handler, 'update_metrics'):
                        self.logger.debug(f"Calling update_metrics for strategy {self.active_strategy_name}")
                        self.strategy_handler.update_metrics(aggregated_stats)

                    # Report and record log (medium frequency)
                    elapsed_in_phase = current_time - phase_start_time
                    self._report_interval_stats(aggregated_stats, elapsed_in_phase, phase_name)
                    self._log_interval_to_dataframe(aggregated_stats, elapsed_in_phase, phase_name)
                    
                    # 4. Only evaluate adjustment period when there's new data (low frequency)
                    if current_time >= next_adjustment_time:
                        self._perform_cache_reallocation(aggregated_stats, elapsed_in_phase)
                        next_adjustment_time = current_time + adjustment_interval
            
            # 5. Main loop tempo control
            time.sleep(0.1)

    def _perform_cache_reallocation(self, current_metrics: Dict[str, Any], elapsed_in_phase: float):
        """
        New reallocation method, properly interfaces with S0 strategy.
        """
        if not hasattr(self.strategy_handler, 'update_allocations'):
            return

        self.logger.info(f"--- Evaluating cache reallocation (T+{elapsed_in_phase:.2f}s) ---")
        
        # CPU time measurement for scalability experiment
        if self.is_scalability_experiment:
            # Temporarily disable strategy log output for more accurate measurement
            strategy_logger = getattr(self.strategy_handler, 'logger', None)
            original_level = None
            if strategy_logger:
                original_level = strategy_logger.level
                strategy_logger.setLevel(logging.CRITICAL)  # Only keep CRITICAL level logs
            
            # Measure pure decision time
            start_cpu_time = time.process_time()
            new_allocations = self.strategy_handler.update_allocations(current_metrics, elapsed_in_phase)
            end_cpu_time = time.process_time()
            
            # Restore original log level
            if strategy_logger and original_level is not None:
                strategy_logger.setLevel(original_level)
            
            decision_cpu_time = end_cpu_time - start_cpu_time
            
            # If strategy reports pure computation time, use it; otherwise use our measurement
            if hasattr(self.strategy_handler, 'pure_decision_time') and self.strategy_handler.pure_decision_time is not None:
                decision_cpu_time = self.strategy_handler.pure_decision_time
                self.strategy_handler.pure_decision_time = None  # Reset
            
            self._record_decision_cpu_time(decision_cpu_time)
        else:
            new_allocations = self.strategy_handler.update_allocations(current_metrics, elapsed_in_phase)

        if new_allocations and new_allocations != self.db_current_page_allocations:
            self.logger.info(f"Detected cache allocation changes: {new_allocations}")
            for db_id, new_pages in new_allocations.items():
                if self.db_current_page_allocations.get(db_id) != new_pages:
                    self.logger.info(f"  > Reallocation: DB[{db_id}] -> {new_pages} pages")
                    # [Core fix] Send DatabaseInstance-compatible command
                    command = {'type': 'RECONFIGURE_CACHE', 'page_count': new_pages}
                    self._broadcast_command(command, specific_db_id=db_id)
            
            self.db_current_page_allocations = new_allocations.copy()
        else:
            self.logger.info("Strategy decided not to change current cache allocation.")
        
        # Apply cache miss delay compensation if enabled
        self._apply_cache_miss_compensation(current_metrics)
    
    def _record_decision_cpu_time(self, cpu_time: float):
        """
        Record CPU decision time for scalability experiments.
        
        Args:
            cpu_time: CPU time in seconds for the decision
        """
        self.decision_cpu_times.append(cpu_time)
        
        # Log summary every 10 decisions
        if len(self.decision_cpu_times) % 10 == 0:
            recent_times = self.decision_cpu_times[-10:]
            avg_time = sum(recent_times) / len(recent_times)
            self.logger.debug(f"CPU Decision Time - Last 10 decisions avg: {avg_time*1000:.3f}ms")

    def _aggregate_interval_stats(self, interval_stats_list: List[Dict]) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate raw statistics data from queue.
        [Core fix] This method now returns format compatible with S0 strategy.
        """
        agg_data = defaultdict(lambda: {
            'ops_count': 0, 'cache_hits': 0, 'cache_misses': 0, 
            'total_latency_ms': 0, 'p99_latencies_ms': []
        })

        for msg in interval_stats_list:
            db_id = msg['db_id']
            # Convert us to ms
            agg_data[db_id]['total_latency_ms'] += msg.get('total_latency_us', 0) / 1000.0
            agg_data[db_id]['ops_count'] += msg.get('ops_count', 0)
            agg_data[db_id]['cache_hits'] += msg.get('cache_hits', 0)
            agg_data[db_id]['cache_misses'] += msg.get('cache_misses', 0)
            
            # Core fix: Correctly collect P99 latency
            if 'p99_latency_us' in msg:
                 agg_data[db_id]['p99_latencies_ms'].append(msg['p99_latency_us'] / 1000.0)

        # Post-processing, calculate average and maximum latency
        processed_data = {}
        for db_id, data in agg_data.items():
            processed_data[db_id] = data.copy()
            p99s = data['p99_latencies_ms']
            # Core fix: Calculate peak P99 value for this interval
            processed_data[db_id]['p99_latency_ms'] = max(p99s) if p99s else 0.0
            
            # Core fix: Calculate and add avg_latency_ms for subsequent DataFrame operations
            ops_count = data.get('ops_count', 0)
            total_latency = data.get('total_latency_ms', 0)
            processed_data[db_id]['avg_latency_ms'] = total_latency / ops_count if ops_count > 0 else 0.0
            
            # Core fix: Calculate and add cache_hit_rate, required by S0 strategy
            cache_hits = data.get('cache_hits', 0)
            cache_misses = data.get('cache_misses', 0)
            total_accesses = cache_hits + cache_misses
            if total_accesses > 0:
                processed_data[db_id]['cache_hit_rate'] = cache_hits / total_accesses
            else:
                processed_data[db_id]['cache_hit_rate'] = 0.0
            
            del processed_data[db_id]['p99_latencies_ms'] # Clean temporary list

        return dict(processed_data)

    def _report_interval_stats(self, interval_stats_data: Dict[str, Dict[str, Any]], elapsed_time: float, phase_name: str):
        """
        Record summary of the most recent reporting period.
        [Core fix] This method now uses the new data format.
        """
        report_lines = [
            f"\n--- Phase '{phase_name}' Stats at T+{elapsed_time:.2f}s ---"
        ]
        
        total_ops = sum(stats['ops_count'] for stats in interval_stats_data.values())
        total_hits = sum(stats['cache_hits'] for stats in interval_stats_data.values())
        total_misses = sum(stats['cache_misses'] for stats in interval_stats_data.values())
        
        reporting_interval = self.general_setup.get("reporting_interval_seconds")
        total_tps = total_ops / reporting_interval if reporting_interval > 0 else 0
        overall_hit_rate = (total_hits / (total_hits + total_misses)) if (total_hits + total_misses) > 0 else 0
        
        report_lines.append(f"  Overall: TPS: {total_tps:.2f}, Hit Rate: {overall_hit_rate:.2%}")

        for db_id, stats in sorted(interval_stats_data.items()):
            ops = stats['ops_count']
            tps = ops / reporting_interval if reporting_interval > 0 else 0
            
            hits = stats['cache_hits']
            misses = stats['cache_misses']
            hit_rate = (hits / (hits + misses)) if (hits + misses) > 0 else 0
            
            avg_latency_ms = stats['avg_latency_ms']
            p99_latency_ms = stats['p99_latency_ms']
            
            current_alloc = self.db_current_page_allocations.get(db_id, 0)

            report_lines.append(
                f"  DB[{db_id:<20}]: Alloc: {current_alloc:<7} pgs | "
                f"TPS: {tps:<7.2f} | HR: {hit_rate:<7.2%} | "
                f"AvgLat: {avg_latency_ms:<7.3f}ms | P99Lat: {p99_latency_ms:<7.3f}ms"
            )
        
        self.logger.info("\n".join(report_lines))

    def _log_interval_to_dataframe(self, interval_stats_data: Dict[str, Dict[str, Any]], elapsed_time: float, phase_name: str):
        """Append current interval statistics to the main result DataFrame."""
        new_rows = []
        
        # Check if we need to extract S0-related factors
        extract_factors = False
        h_factors = {}
        v_factors = {}
        alpha_t = None
        
        # S0 and its ablation versions all have these factors
        s0_variants = ['S0_EMG_AS', 'S0', 'B3_NoFixedElasticByPriority', 'B8_EFFICIENCY_ONLY', 
                      'B9_EMG_AS_SINGLE_EMA', 'B10_Pure_V_Factor']
        
        if self.active_strategy_name in s0_variants and hasattr(self.strategy_handler, '_last_h_factors'):
            extract_factors = True
            h_factors = getattr(self.strategy_handler, '_last_h_factors', {})
            v_factors = getattr(self.strategy_handler, '_last_v_factors', {})
            # Get alpha_t (may be stored in different places)
            if hasattr(self.strategy_handler, 'alpha_state') and 'alpha_prev' in self.strategy_handler.alpha_state:
                alpha_t = self.strategy_handler.alpha_state['alpha_prev']
            elif hasattr(self.strategy_handler, '_last_alpha_t'):
                alpha_t = self.strategy_handler._last_alpha_t
            
            # DebugLog
            self.logger.debug(f"Extracting {self.active_strategy_name} factors: "
                            f"H={len(h_factors)} databases, V={len(v_factors)} databases, alpha_t={alpha_t}")
            
        for db_id, stats in interval_stats_data.items():
            # Core fix: ensure we record the current cache page size
            current_pages = self.db_current_page_allocations.get(db_id, 0)
            row_data = {
                'timestamp': pd.Timestamp.now(),
                'phase_name': phase_name,
                'elapsed_seconds': elapsed_time,
                'db_id': db_id,
                'strategy_name': self.active_strategy_name,
                'current_cache_pages': current_pages,  # Fix: use correct column name
                **stats
            }
            
            # Add S0-related factors
            if extract_factors:
                row_data['h_factor'] = h_factors.get(db_id, 0.0)
                row_data['v_factor'] = v_factors.get(db_id, 0.0)
                row_data['alpha_t'] = alpha_t if alpha_t is not None else 0.5
                # Calculate total score
                if alpha_t is not None:
                    row_data['total_score'] = (alpha_t * row_data['h_factor'] + 
                                             (1 - alpha_t) * row_data['v_factor'])
                else:
                    row_data['total_score'] = 0.5 * row_data['h_factor'] + 0.5 * row_data['v_factor']
            
            new_rows.append(row_data)

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            self.results_df = pd.concat([self.results_df, new_df], ignore_index=True)

    def _report_final_stats(self, output_csv_path: str | None = None):
        """Report final statistics and save results at the end of experiment."""
        self.logger.info("====== ExperimentFinalStatistics ======")
        
        if self.results_df.empty:
            self.logger.warning("Result DataFrame is empty, cannot calculate final statistics.")
            return

        # Calculate overall statistics for each database
        final_stats = self.results_df.groupby('db_id').agg(
            total_ops=('ops_count', 'sum'),
            total_cache_hits=('cache_hits', 'sum'),
            total_cache_misses=('cache_misses', 'sum'),
            avg_latency_ms=('avg_latency_ms', 'mean'),
            p99_latency_ms=('p99_latency_ms', 'mean') # Core fix: add P99 aggregation
        ).reset_index()

        if (final_stats['total_cache_hits'] + final_stats['total_cache_misses']).sum() > 0:
            final_stats['overall_hit_rate'] = final_stats['total_cache_hits'] / (final_stats['total_cache_hits'] + final_stats['total_cache_misses'])
        else:
            final_stats['overall_hit_rate'] = 0

        final_stats.fillna({'overall_hit_rate': 0}, inplace=True)

        self.logger.info("--- Overall Performance per Database ---")
        for _, row in final_stats.iterrows():
            self.logger.info(
                f"  > DB[{row['db_id']}]: "
                f"Total Ops={row['total_ops']}, "
                f"Overall HitRate={row['overall_hit_rate']:.2%}, "
                f"Avg Latency={row['avg_latency_ms']:.3f}ms, "
                f"Avg P99 Latency={row['p99_latency_ms']:.3f}ms"
            )

        # Calculate total system throughput and weighted hit rate
        total_system_ops = final_stats['total_ops'].sum()
        experiment_duration = self.results_df['elapsed_seconds'].max()
        system_throughput = total_system_ops / experiment_duration if experiment_duration > 0 else 0
        
        weighted_hit_rate = (final_stats['total_cache_hits'].sum() / (final_stats['total_cache_hits'].sum() + final_stats['total_cache_misses'].sum())) if (final_stats['total_cache_hits'].sum() + final_stats['total_cache_misses'].sum()) > 0 else 0
        
        self.logger.info("--- Overall System Performance ---")
        self.logger.info(f"  > Total run time: {experiment_duration:.2f} seconds")
        self.logger.info(f"  > Total system operations: {total_system_ops}")
        self.logger.info(f"  > System average throughput (TPS): {system_throughput:.2f}")
        self.logger.info(f"  > System weighted average hit rate: {weighted_hit_rate:.2%}")
        
        # Decision CPU time statistics for scalability experiment
        if self.is_scalability_experiment and self.decision_cpu_times:
            self.logger.info("--- Scalability Experiment Decision Performance ---")
            avg_cpu_time = np.mean(self.decision_cpu_times)
            std_cpu_time = np.std(self.decision_cpu_times)
            min_cpu_time = np.min(self.decision_cpu_times)
            max_cpu_time = np.max(self.decision_cpu_times)
            p50_cpu_time = np.percentile(self.decision_cpu_times, 50)
            p95_cpu_time = np.percentile(self.decision_cpu_times, 95)
            p99_cpu_time = np.percentile(self.decision_cpu_times, 99)
            
            self.logger.info(f"  > Decision count: {len(self.decision_cpu_times)}")
            self.logger.info(f"  > Average decision CPU time: {avg_cpu_time:.6f} seconds ({avg_cpu_time*1000:.3f} ms)")
            self.logger.info(f"  > Standard deviation: {std_cpu_time:.6f} seconds")
            self.logger.info(f"  > Min value: {min_cpu_time:.6f} seconds")
            self.logger.info(f"  > Max value: {max_cpu_time:.6f} seconds")
            self.logger.info(f"  > P50 (median): {p50_cpu_time:.6f} seconds")
            self.logger.info(f"  > P95: {p95_cpu_time:.6f} seconds")
            self.logger.info(f"  > P99: {p99_cpu_time:.6f} seconds")
            self.logger.info(f"  > Database count: {len(self.db_instance_configs)}")
            
            # Save scalability experiment results to separate file
            self._save_scalability_results(output_csv_path, {
                'strategy_name': self.active_strategy_name,
                'database_count': len(self.db_instance_configs),
                'decision_count': len(self.decision_cpu_times),
                'avg_cpu_time_ms': avg_cpu_time * 1000,
                'std_cpu_time_ms': std_cpu_time * 1000,
                'min_cpu_time_ms': min_cpu_time * 1000,
                'max_cpu_time_ms': max_cpu_time * 1000,
                'p50_cpu_time_ms': p50_cpu_time * 1000,
                'p95_cpu_time_ms': p95_cpu_time * 1000,
                'p99_cpu_time_ms': p99_cpu_time * 1000,
                'raw_cpu_times': self.decision_cpu_times  # Save raw data
            })

        if output_csv_path:
            try:
                # Ensure directory exists
                output_dir = os.path.dirname(output_csv_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                
                self.results_df.to_csv(output_csv_path, index=False)
                self.logger.info(f"✅ Experiment results successfully saved to: {output_csv_path}")
            except Exception as e:
                self.logger.error(f"❌ Failed to save experiment results to {output_csv_path}: {e}")
        else:
            self.logger.warning("⚠️ No output path provided, experiment results only kept in memory and won't be saved.")
        
        self.logger.info("Experiment completed, detailed data is in results_df.")

        self.logger.info(f"Calling cleanup method for strategy '{self.active_strategy_name}'.")
        self.strategy_handler.cleanup()

        self.logger.info("Orchestrator cleanup complete.")

    def run(self, output_csv_path: str | None = None):
        """
        Main entry point for experiment execution.
        """
        try:
            self.logger.info(f"Starting experiment for strategy '{self.active_strategy_name}' executing experiment。")
            
            # 1. Prepare all databases (if needed)
            self._prepare_databases()

            self.logger.info("====== Calculating initial cache allocation ======")
            self.logger.info(f"strategy_handlerType: {type(self.strategy_handler)}")
            self.logger.info(f"About to call calculate_initial_allocations method...")
            try:
                initial_allocations = self.strategy_handler.calculate_initial_allocations()
                self.logger.info(f"calculate_initial_allocations call complete, returned: {type(initial_allocations)}")
            except Exception as e:
                self.logger.error(f"Initial allocation calculation failed: {e}", exc_info=True)
                # Emergency: create default allocation
                initial_allocations = {}
                for db_instance in self.db_instances:
                    initial_allocations[db_instance.db_id] = 1  # At least 1 page per database
                self.logger.error(f"Using emergency default allocation: {initial_allocations}")
            
            self.db_current_page_allocations = initial_allocations.copy()
            self.logger.info(f"Initial cache allocation plan: {len(initial_allocations)} databases, total allocation: {sum(initial_allocations.values())} pages")
            
            # Debug: check consistency between database instances and initial allocation
            db_instance_ids = [db.db_id for db in self.db_instances]
            missing_in_allocation = [db_id for db_id in db_instance_ids if db_id not in initial_allocations]
            if missing_in_allocation:
                self.logger.error(f"Warning: The following database instances have no initial allocation: {missing_in_allocation}")
            
            extra_in_allocation = [db_id for db_id in initial_allocations if db_id not in db_instance_ids]
            if extra_in_allocation:
                self.logger.error(f"Warning: Initial allocation contains non-existent databases: {extra_in_allocation}")
            
            self.logger.info("====== Starting database instances ======")
            self._start_db_instance_threads(initial_allocations)
            
            if not self._wait_for_all_workers_ready():
                self.logger.critical("Worker threads failed to be fully ready. Experiment terminated.")
                self.cleanup()
                return

            # [Core fix] Send START_BENCHMARK here
            self.logger.info("====== All instances ready, starting benchmark test ======")
            self._broadcast_command({'type': 'START_BENCHMARK'})

            self.logger.info("====== Entering measurement loop ======")
            self._run_measurement_loop()

        except (ConfigError, RuntimeError, ValueError) as e:
            self.logger.critical(f"Experiment terminated due to configuration or runtime error: {e}", exc_info=True)
        except Exception as e:
            self.logger.critical(f"Unexpected fatal error occurred during experiment: {e}", exc_info=True)
        finally:
            self.cleanup()
            self._report_final_stats(output_csv_path)
            self.logger.info("Experiment execution process completed.")
            # Core fix: return DataFrame containing all results
            return self.results_df

    def _broadcast_command(self, command: Dict[str, Any], specific_db_id: str = None):
        """Broadcast a command to all or specified worker threads."""
        
        target_queues = {}
        if specific_db_id:
            if specific_db_id in self.command_queues:
                target_queues[specific_db_id] = self.command_queues[specific_db_id]
            else:
                self.logger.error(f"Attempting to send command to non-existent DB instance '{specific_db_id}'.")
                return
        else:
            target_queues = self.command_queues

        for db_id, q in target_queues.items():
            try:
                q.put_nowait(command)
            except Full:
                self.logger.warning(f"Command queue for {db_id} is full. Command {command.get('type')} might be dropped.")

    def _save_scalability_results(self, base_output_path: str, results: dict):
        """Save scalability experiment results to separate file"""
        if not base_output_path:
            return
            
        try:
            # Build scalability result file path
            base_dir = os.path.dirname(base_output_path)
            base_name = os.path.basename(base_output_path).replace('.csv', '')
            scalability_file = os.path.join(base_dir, f"{base_name}_scalability_results.json")
            
            # Prepare data to save
            save_data = {
                'experiment_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'strategy_name': results['strategy_name'],
                'database_count': results['database_count'],
                'decision_count': results['decision_count'],
                'statistics': {
                    'avg_cpu_time_ms': results['avg_cpu_time_ms'],
                    'std_cpu_time_ms': results['std_cpu_time_ms'],
                    'min_cpu_time_ms': results['min_cpu_time_ms'],
                    'max_cpu_time_ms': results['max_cpu_time_ms'],
                    'p50_cpu_time_ms': results['p50_cpu_time_ms'],
                    'p95_cpu_time_ms': results['p95_cpu_time_ms'],
                    'p99_cpu_time_ms': results['p99_cpu_time_ms']
                },
                'raw_cpu_times_seconds': results['raw_cpu_times']  # Raw data in seconds
            }
            
            # Save as JSON format
            with open(scalability_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"✅ Scalability experiment results saved to: {scalability_file}")
            
            # Also save summary info as CSV format
            csv_file = scalability_file.replace('.json', '.csv')
            scalability_df = pd.DataFrame([{
                'timestamp': save_data['experiment_timestamp'],
                'strategy_name': save_data['strategy_name'],
                'database_count': save_data['database_count'],
                'decision_count': save_data['decision_count'],
                'avg_cpu_time_ms': save_data['statistics']['avg_cpu_time_ms'],
                'std_cpu_time_ms': save_data['statistics']['std_cpu_time_ms'],
                'min_cpu_time_ms': save_data['statistics']['min_cpu_time_ms'],
                'max_cpu_time_ms': save_data['statistics']['max_cpu_time_ms'],
                'p50_cpu_time_ms': save_data['statistics']['p50_cpu_time_ms'],
                'p95_cpu_time_ms': save_data['statistics']['p95_cpu_time_ms'],
                'p99_cpu_time_ms': save_data['statistics']['p99_cpu_time_ms']
            }])
            
            scalability_df.to_csv(csv_file, index=False)
            self.logger.info(f"✅ Scalability experiment summary saved to: {csv_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving scalability experiment results: {e}")

    def cleanup(self):
        """
        Clean up resources created by this orchestrator instance, including worker threads and log handlers.
        """
        self.logger.info(f"CurrentlyCleanOrchestrator for strategy '{self.active_strategy_id}'...")

        # 1. Gracefully stop all worker threads
        if hasattr(self, 'db_instance_threads') and self.db_instance_threads:
            self.logger.info("Sending stop signals to all database worker threads...")
            self._broadcast_command({'type': 'STOP'})
            
            # Wait for all threads to terminate
            shutdown_timeout = self.general_setup.get("worker_shutdown_timeout_s")
            for db_id, thread in self.db_instance_threads.items():
                try:
                    thread.join(timeout=shutdown_timeout)
                    if thread.is_alive():
                        self.logger.warning(f"Worker thread {db_id} did not terminate after timeout ({shutdown_timeout}s).")
                    else:
                        self.logger.info(f"Worker thread {db_id} successfully terminated.")
                except Exception as e:
                    self.logger.error(f"Error waiting for worker thread {db_id} to terminate: {e}")
            self.logger.info("All database worker threads have been processed.")

        # 2. Clean up log handlers
        if self.logger:
            # Key: Remove all handlers to prevent log handle leaks and cross-contamination
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)
            self.logger.info(f"Logger '{self.logger.name}' all handlers removed.")

    def _apply_cache_miss_compensation(self, current_metrics: Dict[str, Any]):
        """
        Calculate batch cache miss delay compensation to simulate network latency.
        The delay will be applied at the beginning of the next measurement cycle.
        """
        if not self.cache_miss_simulation.get("enabled", False):
            return
        
        miss_penalty_ms = self.cache_miss_simulation.get("miss_penalty_ms", 50)
        compensation_method = self.cache_miss_simulation.get("compensation_method", "batch")
        
        if compensation_method != "batch":
            return
        
        # Calculate total cache misses across all databases in this tuning interval
        total_misses = 0
        for db_id, db_stats in current_metrics.items():
            cache_misses = db_stats.get("cache_misses", 0)
            total_misses += cache_misses
        
        if total_misses > 0:
            # Calculate total compensation delay
            compensation_delay_ms = total_misses * miss_penalty_ms
            compensation_delay_seconds = compensation_delay_ms / 1000.0
            
            self.logger.info(f"Cache miss compensation: {total_misses} misses × {miss_penalty_ms}ms = {compensation_delay_ms}ms delay (will apply at next cycle)")
            
            # Store delay to apply at the beginning of next cycle
            self.pending_compensation_delay = compensation_delay_seconds
        else:
            self.pending_compensation_delay = 0.0


def main():
    """Program main entry point"""
    # Removed try-except, let any configuration or runtime errors directly cause program failure
    orchestrator = ExperimentOrchestrator()
    orchestrator.run()

if __name__ == "__main__":
    main()
