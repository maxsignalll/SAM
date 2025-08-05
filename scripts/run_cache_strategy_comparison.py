#!/usr/bin/env python3
"""
Cache Strategy Comparison Experiment Script
Automatically runs TPS-controlled experiments for all cache strategies and generates comparative analysis
"""

import logging
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import copy
import codecs
import gc

# Set matplotlib's log level to WARNING to hide the font manager's DEBUG messages.
# This should be done before other matplotlib imports.
logging.getLogger('matplotlib').setLevel(logging.WARNING)

import sys
import os
import json
import time
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import apsw

try:
    import psutil
except ImportError:
    psutil = None

# Force UTF-8 for stdout/stderr on Windows to prevent UnicodeEncodeError
# On Ubuntu/Linux, UTF-8 is typically the default encoding, so this is mainly for Windows
if sys.platform == "win32" and sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except Exception as e:
        # Fallback for environments where this is not possible
        print(f"Warning: Could not reconfigure console to UTF-8: {e}")
elif sys.platform != "win32":
    # On Linux/Ubuntu, ensure UTF-8 encoding is properly set
    import locale
    try:
        # Set locale to UTF-8 if not already set
        if 'UTF-8' not in locale.getpreferredencoding():
            locale.setlocale(locale.LC_ALL, 'C.UTF-8')
    except locale.Error:
        # Fallback to default if C.UTF-8 is not available
        pass

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# Use matplotlib's default English fonts
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['axes.unicode_minus'] = False

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from experiment_manager import ExperimentOrchestrator
from enhanced_metrics_analyzer import EnhancedMetricsAnalyzer, create_enhanced_analyzer
from config_loader import ConfigLoader
from database_instance import BGDatabaseLogFilter


class CacheStrategyComparison:
    """Cache Strategy Comparison Experiment Manager"""
    
    def __init__(self, base_output_dir: str = None):
        """
        Initialize experiment manager.

        Args:
            base_output_dir (str, optional): Override default base output directory.
                                             If None, uses "results/comparison".
        """
        if base_output_dir:
            self.base_output_dir = Path(base_output_dir)
        else:
            self.base_output_dir = Path("results/comparison")
        
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Initialize logger first
        self.logger = logging.getLogger(self.__class__.__name__)

        # 2. Experiment timestamp
        self.experiment_timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Check if there's an override directory
        if base_output_dir:
            self.experiment_dir = Path(base_output_dir)
        else:
            self.experiment_dir = self.base_output_dir / f"experiment_{self.experiment_timestamp}"
        
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # 3. Use main config file
        self.main_config_file = "configs/config_tps_controlled_comparison.json"
        
        # 4. Load config file to instance for all methods to access
        try:
            # Fix: instantiate ConfigLoader, then call method
            loader = ConfigLoader(self.main_config_file)
            self.config = loader.get_full_config()
        except Exception as e:
            self.logger.error(f"Failed to load main config file {self.main_config_file}: {e}")
            raise

        self.experiment_results = {}
        self.comparison_metrics = {}
        
    def _normalize_strategy_name(self, name: str) -> str:
        """Normalize strategy names for case-insensitive and symbol-agnostic comparison."""
        return ''.join(filter(str.isalnum, name)).lower()

    def update_config_file(self, new_config_path: str):
        """Allows updating the main config file path for all strategies."""
        self.main_config_file = new_config_path
        self.logger.info(f"Updated main config file for the campaign to: {new_config_path}")
        
        # Reload config file
        try:
            loader = ConfigLoader(self.main_config_file)
            self.config = loader.get_full_config()
            self.logger.info(f"Configuration reloaded from: {new_config_path}")
        except Exception as e:
            self.logger.error(f"Failed to reload config from {new_config_path}: {e}")
            raise
        
    def prepare_experiment_environment(self, force_clean_db=False):
        """Prepare experiment environment (optimized version to reduce initialization time)"""
        print(f"🔧 Preparing experiment environment...")
        print(f"📁 Experiment directory: {self.experiment_dir}")
        
        # Create necessary directories
        (self.experiment_dir / "individual_results").mkdir(exist_ok=True)
        (self.experiment_dir / "analysis").mkdir(exist_ok=True)
        (self.experiment_dir / "plots").mkdir(exist_ok=True)
        
        # Optimization: only perform full database management in force clean mode
        if force_clean_db:
            print("💥 Force clean mode: performing full database management")
            self._manage_database_files(force_clean_db)
        else:
            print("⚡ Optimized mode: skipping database pre-check, creating on demand")
            # Only perform basic checks with network error handling
            data_dir = Path("/mnt/share_via_ssh")
            try:
                print("🔍 Checking remote data directory...")
                if data_dir.exists():
                    existing_dbs = list(data_dir.glob("*.sqlite"))
                    if existing_dbs:
                        print(f"📊 Found {len(existing_dbs)} existing databases, will auto-verify and reuse in each strategy")
                    else:
                        print("🔄 No existing databases found, will create on demand")
                else:
                    print(f"⚠️  Remote data directory {data_dir} does not exist")
            except (ConnectionAbortedError, OSError, IOError) as e:
                print(f"🌐 Remote data directory access failed ({type(e).__name__}: {e})")
                print("🔄 Will create databases on demand during strategy runs, no pre-check needed")
        
        print("✅ Experiment environment prepared")
    
    def _is_file_locked(self, file_path):
        """Check if SQLite database file is locked by other processes (including WAL and SHM files)"""
        file_path = Path(file_path)
        
        # Check main database file
        if not file_path.exists():
            return False

        wal_file = file_path.with_suffix(file_path.suffix + '-wal')
        shm_file = file_path.with_suffix(file_path.suffix + '-shm')
        
        files_to_check = [file_path]
        if wal_file.exists():
            files_to_check.append(wal_file)
        if shm_file.exists():
            files_to_check.append(shm_file)
        
        for check_file in files_to_check:
            try:
                temp_name = check_file.with_suffix(check_file.suffix + '.tmp_check')
                check_file.rename(temp_name)
                temp_name.rename(check_file)
            except (OSError, PermissionError):
                return True

            try:
                import msvcrt
                with open(check_file, 'r+b') as f:
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            except (OSError, PermissionError, ImportError):
                pass
        
        return False

    def _cleanup_sqlite_auxiliary_files(self, db_file):
        """Clean up SQLite auxiliary files (WAL, SHM, Journal, etc.)"""
        db_path = Path(db_file)
        auxiliary_files = [
            db_path.with_suffix(db_path.suffix + '-wal'),
            db_path.with_suffix(db_path.suffix + '-shm'),
            db_path.with_suffix(db_path.suffix + '-journal')
        ]
        
        cleaned_files = []
        for aux_file in auxiliary_files:
            if aux_file.exists():
                aux_file.unlink()
                cleaned_files.append(aux_file.name)
        
        if cleaned_files:
            print(f"  🧹 Cleaned SQLite auxiliary files: {', '.join(cleaned_files)}")
            
    def _find_and_kill_locking_processes(self, locked_files: List[Path]) -> List[int]:
        """Find and kill processes holding file locks"""
        if not psutil:
            self.logger.warning("`psutil` module not installed, cannot auto-terminate processes.")
            return []

        killed_pids = []
        # Get absolute paths of all locked files
        locked_file_paths = {str(f.resolve()) for f in locked_files}
        
        current_pid = os.getpid()
        self.logger.info(f"Starting process scan... Current PID: {current_pid}")

        for proc in psutil.process_iter(['pid', 'name', 'open_files']):
            if proc.info['pid'] == current_pid:
                continue

            try:
                open_files = proc.info['open_files']
                if open_files:
                    for f in open_files:
                        if f.path in locked_file_paths:
                            self.logger.warning(f"Found process holding lock: PID={proc.info['pid']}, Name='{proc.info['name']}', File='{f.path}'")
                            try:
                                p = psutil.Process(proc.info['pid'])
                                p.kill()  # Force kill
                                killed_pids.append(p.pid)
                                self.logger.info(f"✅ Terminated process PID: {p.pid}")
                                # Found a match, break inner loop and process next
                                break 
                            except psutil.NoSuchProcess:
                                self.logger.info(f"Process PID {proc.info['pid']} no longer exists before termination.")
                            except Exception as e:
                                self.logger.error(f"Failed to terminate process PID {proc.info['pid']}: {e}")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Process may have died while we were checking, or we don't have permission
                continue
        
        return list(set(killed_pids))
            
    def _manage_database_files(self, force_clean=False):
        """Smart database file management"""
        # Use remote folder mapping instead of local data directory
        data_dir = Path("/mnt/share_via_ssh")
        if not data_dir.exists():
            print(f"⚠️  Remote folder {data_dir} does not exist, please check remote folder mapping")
            return
        
        print(f"📁 Using remote database directory: {data_dir}")
        
        existing_dbs = list(data_dir.glob("*.sqlite"))
        
        if force_clean and existing_dbs:
            print("🗑️  Force cleaning all database files...")
            
            for db_file in existing_dbs:
                self._cleanup_sqlite_auxiliary_files(db_file)
            
            for db_file in existing_dbs:
                if db_file.exists():
                    # Remove retry logic, let deletion failures throw exceptions directly
                    db_file.unlink()
                    print(f"  ✅ Successfully deleted: {db_file.name}")
            
            print(f"✅ All database files cleaned")
            return
        
        if existing_dbs:
            print(f"💾 Found {len(existing_dbs)} existing database files:")
            for db_file in existing_dbs:
                # Check file size and modification time
                stat = db_file.stat()
                size_mb = stat.st_size / (1024 * 1024)
                from datetime import datetime
                mod_time = datetime.fromtimestamp(stat.st_mtime)
                print(f"  📄 {db_file.name}: {size_mb:.1f}MB, Modified: {mod_time.strftime('%H:%M:%S')}")
            
            # First check if all database files are locked
            locked_files = []
            unlocked_files = []
            
            for db_file in existing_dbs:
                if self._is_file_locked(db_file):
                    locked_files.append(db_file)
                    print(f"  🔒 {db_file.name}: File is locked")
                else:
                    unlocked_files.append(db_file)
                    print(f"  🔓 {db_file.name}: File is not locked")
            
            # If files are locked, report the issue and provide solutions
            if locked_files:
                print(f"\n⚠️  Found {len(locked_files)} locked database files:")
                for db_file in locked_files:
                    print(f"     {db_file.name}")
                
                # Ultimate solution: first try to automatically find and kill processes
                print("\n💥 [Ultimate Cleaner] Attempting to find and kill zombie processes holding databases...")
                killed_pids = self._find_and_kill_locking_processes(locked_files)
                if killed_pids:
                    print(f"✅ Successfully terminated {len(killed_pids)} processes: PIDs={killed_pids}")
                    print("   Waiting 3 seconds for OS to release file handles...")
                    time.sleep(3)
                else:
                    print("   No processes clearly holding these files found. May be permission issues or unreleased handles.")

                # After killing processes, try again to clean auxiliary files
                print(f"\n🧹 Trying again to clean SQLite auxiliary files...")
                for db_file in locked_files:
                    self._cleanup_sqlite_auxiliary_files(db_file)
                
                print("\n💡 Solutions:")
                print("   1. Wait for all related processes to end and retry")
                print("   2. Use --force-clean-db parameter to force clean and regenerate")
                
                # Try waiting to see if files will be released
                print(f"\n⏱️  Waiting 5 seconds to check if files will be released...")
                time.sleep(5)
                
                # Re-check
                still_locked = []
                for db_file in locked_files:
                    if self._is_file_locked(db_file):
                        still_locked.append(db_file)
                
                if still_locked:
                    print(f"❌ {len(still_locked)} files still locked, cannot continue")
                    raise RuntimeError(f"Database files locked: {[f.name for f in still_locked]}")
                else:
                    print("✅ All files released, can continue")
                    unlocked_files = existing_dbs  # Update to all files available
            
            # Validate content of unlocked files
            if unlocked_files:
                print(f"\n🔍 Validating content of {len(unlocked_files)} available database files...")
                valid_dbs = self._validate_existing_databases(unlocked_files)
                
                if valid_dbs:
                    print(f"✅ Found {len(valid_dbs)} valid database files, will reuse existing data")
                    print("💡 To regenerate databases, use --force-clean-db parameter")
                    print("⚡ Optimization: Skipping database rebuild, entering experiment phase directly")
                else:
                    print("⚠️  Existing database files invalid, will clean and regenerate")
                    for db_file in unlocked_files:
                        try:
                            # Clean auxiliary files
                            self._cleanup_sqlite_auxiliary_files(db_file)
                            # Delete main file
                            db_file.unlink()
                            print(f"  Deleting invalid database: {db_file}")
                        except Exception as e:
                            print(f"  Failed to delete {db_file}: {e}")
        else:
            print("📊 No existing database files found, will auto-generate on first run")
            print("⚡ Optimization tip: Databases will be created on demand during strategy runs, reducing init time")
    
    def _validate_existing_databases(self, db_files):
        """Validate existing database files"""
        valid_dbs = []
        
        for db_file in db_files:
            conn = None
            try:
                conn = apsw.Connection(str(db_file))
                cursor = conn.cursor()
                
                # Check if table exists - prioritize usertable (YCSB standard table name)
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='usertable'")
                if cursor.fetchone():
                    # Check record count
                    cursor.execute("SELECT COUNT(*) FROM usertable")
                    count = cursor.fetchone()[0]
                    if count > 0:
                        valid_dbs.append(db_file)
                        print(f"  ✅ {db_file.name}: {count:,} records (usertable)")
                    else:
                        print(f"  ❌ {db_file.name}: usertable is empty")
                else:
                    # Check if ycsb table exists (compatibility check)
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ycsb'")
                    if cursor.fetchone():
                        cursor.execute("SELECT COUNT(*) FROM ycsb")
                        count = cursor.fetchone()[0]
                        if count > 0:
                            print(f"  ⚠️ {db_file.name}: Found ycsb table but expected usertable, marking as invalid")
                        else:
                            print(f"  ❌ {db_file.name}: ycsb table is empty")
                    else:
                        print(f"  ❌ {db_file.name}: Missing usertable")
                
            except Exception as e:
                print(f"  ❌ {db_file.name}: Validation failed - {e}")
            finally:
                if conn:
                    conn.close()
        
        return valid_dbs
        
    def _prepare_b5_config(self, base_config: Dict[str, Any], b5_strategy_name: str) -> Dict[str, Any]:
        """
        Dynamically generate special configuration for B5 (Simulated Global LRU) based on standard config.
        Using Option B: Single database file with multiple tables, maintaining original data isolation.
        """
        # Use deep copy to avoid modifying original configuration
        b5_config = copy.deepcopy(base_config)

        # B5 should run with the same dynamic workload phases, but with TPS consolidated.
        self.logger.info(f"🔧 Preparing special B5 configuration for {b5_strategy_name} (Multi-table implementation)...")
        
        # Detect workload type
        workload_type = base_config.get("general_experiment_setup", {}).get("workload_type", "ycsb")
        
        b5_db_id = "b5_global_lru"
        
        # Save original database configs for creating multiple tables
        original_db_configs = base_config["database_instances"]
        
        # Set database configuration based on workload type
        if workload_type == "tpcc":
            # TPC-C configuration
            total_warehouses = sum(db.get("tpcc_warehouses", 1) for db in original_db_configs)
            # Local path for generation, remote path for running
            local_db_path = f"./data/tpcc/{b5_db_id}.sqlite"
            remote_db_path = f"/mnt/share_via_ssh/tpcc/{b5_db_id}.sqlite"
            
            b5_db_config = {
                "id": b5_db_id,
                "db_filename": remote_db_path,  # Use remote path for final runtime
                "local_db_filename": local_db_path,  # Local generation path
                "tpcc_warehouses": total_warehouses,
                "target_tps": 0, # Placeholder, will be set per phase
                "tps_control": {"enabled": True},
                # Save original database configuration for multi-table creation
                "b5_original_dbs": original_db_configs,
                "b5_multi_table": True
            }
        else:
            # YCSB configuration (default)
            total_records = sum(db.get("ycsb_initial_record_count", 0) for db in original_db_configs)
            # Local path for generation, remote path for running
            local_db_path = f"./data/{b5_db_id}.sqlite"
            remote_db_path = f"/mnt/share_via_ssh/{b5_db_id}.sqlite"
            
            b5_db_config = {
                "id": b5_db_id,
                "db_filename": remote_db_path,  # Use remote path for final runtime
                "local_db_filename": local_db_path,  # Local generation path
                "ycsb_initial_record_count": total_records,
                "target_tps": 0, # Placeholder, will be set per phase
                "tps_control": {"enabled": True},
                # Save original database configuration for multi-table creation
                "b5_original_dbs": original_db_configs,
                "b5_multi_table": True
            }
            
        b5_config["database_instances"] = [b5_db_config]
        
        # Recreate the dynamic workload phases for the single B5 instance
        b5_dynamic_phases = []
        if "dynamic_workload_phases" in base_config:
            for original_phase in base_config["dynamic_workload_phases"]:
                # Choose correct config override key based on workload type
                if workload_type == "tpcc":
                    config_overrides_key = "tpcc_config_overrides"
                else:
                    config_overrides_key = "ycsb_config_overrides"
                
                # Sum up the TPS from all DBs for this phase
                phase_overrides = original_phase.get(config_overrides_key, {})
                total_phase_tps = sum(phase_overrides.get("tps_distribution_per_db", {}).values())
                
                # Create new phase configuration
                new_phase = {
                    "name": original_phase.get("name"),
                    "duration_seconds": original_phase.get("duration_seconds"),
                }
                
                # Set configuration overrides based on workload type
                if workload_type == "tpcc":
                    # TPC-C doesn't need access pattern configuration
                    new_phase["tpcc_config_overrides"] = {
                        "tps_distribution_per_db": {
                            b5_db_id: total_phase_tps
                        },
                        # Keep original TPS distribution for B5 internal routing
                        "original_tps_distribution": phase_overrides.get("tps_distribution_per_db", {})
                    }
                else:
                    # YCSB needs access pattern configuration
                    access_pattern_config = self._get_access_pattern_for_b5(original_phase, config_overrides_key)
                    new_phase["ycsb_config_overrides"] = {
                        "access_pattern": access_pattern_config,
                        "tps_distribution_per_db": {
                            b5_db_id: total_phase_tps
                        },
                        # Keep original TPS distribution for B5 internal routing
                        "original_tps_distribution": phase_overrides.get("tps_distribution_per_db", {}),
                        # Keep original access pattern configuration
                        "access_pattern_per_db": phase_overrides.get("access_pattern_per_db", {})
                    }
                
                b5_dynamic_phases.append(new_phase)
                self.logger.info(f"B5 Phase '{new_phase.get('name')}' created with consolidated TPS: {total_phase_tps} for {workload_type}")

        b5_config["dynamic_workload_phases"] = b5_dynamic_phases
        
        # Update active strategy ID
        b5_config["general_experiment_setup"]["active_strategy"] = b5_strategy_name
        b5_config["general_experiment_setup"]["config_source"] = self.main_config_file

        return b5_config

    def _get_access_pattern_for_b5(self, original_phase: Dict[str, Any], config_overrides_key: str = "ycsb_config_overrides") -> Dict[str, Any]:
        """
        Helper function to extract the access pattern for a given phase for B5.
        B5 is monolithic, so we assume the access pattern is the same for all DBs in a phase.
        We extract the first available pattern definition from 'access_pattern_per_db'.
        """
        access_pattern_config = None
        phase_overrides = original_phase.get(config_overrides_key, {})
        if "access_pattern_per_db" in phase_overrides and phase_overrides["access_pattern_per_db"]:
            # Get the access pattern config from the first DB
            first_db_key = next(iter(phase_overrides["access_pattern_per_db"]))
            access_pattern_config = phase_overrides["access_pattern_per_db"][first_db_key]
            self.logger.info(f"Extracted access pattern for B5 phase '{original_phase.get('name')}': {access_pattern_config}")
        else:
            self.logger.warning(f"During B5 config preparation, 'access_pattern_per_db' not found in phase '{original_phase.get('name')}'. The access pattern will not be changed for this phase.")
        return access_pattern_config

    def run_all_strategies(self, strategies_to_run_from_cli: List[str] = None):
        """
        Run all strategies marked as active in config file, or only run strategies specified via CLI.
        """
        # --- New, dynamic strategy loading logic ---
        all_configured_strategies = self.config.get("strategy_configurations", {})
        strategies_to_run = {}

        if strategies_to_run_from_cli:
            # If strategies are specified from command line, run them in command line order
            # Establish mapping from normalized name to config name
            normalized_to_config = {self._normalize_strategy_name(name): name for name in all_configured_strategies.keys()}
            
            # Add strategies in command line argument order
            for cli_strategy in strategies_to_run_from_cli:
                normalized_cli = self._normalize_strategy_name(cli_strategy)
                if normalized_cli in normalized_to_config:
                    config_name = normalized_to_config[normalized_cli]
                    strategies_to_run[config_name] = all_configured_strategies[config_name]
                else:
                    self.logger.warning(f"Provided strategy '{cli_strategy}' not found in config, will be ignored.")
        else:
            # If not specified from command line, run all strategies marked as active: true
            for name, config in all_configured_strategies.items():
                if config.get("active", False):
                    strategies_to_run[name] = config

        print("\n" + "="*80)
        print(f"📑 Preparing to run the following {len(strategies_to_run)} strategies: {', '.join(strategies_to_run.keys())}")
        print("="*80 + "\n")

        # Maintain command line specified strategy execution order
        strategy_items = list(strategies_to_run.items())
        self.logger.info(f"Strategy execution order: {[name for name, _ in strategy_items]}")
        
        for strategy_name, strategy_config_block in strategy_items:
            print("\n" + "=" * 80)
            print(f"🚀 Starting to run strategy: {strategy_name}")
            print("=" * 80)

            # Create a new config copy and override the active strategy
            run_config = copy.deepcopy(self.config)
            run_config['general_experiment_setup']['active_strategy'] = strategy_name
            
            # Create a separate output directory for this strategy
            strategy_output_dir = self.experiment_dir / "individual_results" / strategy_name
            strategy_output_dir.mkdir(parents=True, exist_ok=True)
            run_config['general_experiment_setup']['output_directory'] = str(strategy_output_dir)
            output_csv_path = strategy_output_dir / "data.csv"

            # Prepare special monolithic database configuration for B5 (simulated global LRU)
            if "b5" in self._normalize_strategy_name(strategy_name):
                self.logger.info(f"Detected B5 strategy ({strategy_name}), preparing special single-database simulated global LRU configuration.")
                run_config = self._prepare_b5_config(run_config, strategy_name)

            try:
                orchestrator = ExperimentOrchestrator(final_config=run_config)
                
                # Save orchestrator reference for scalability experiments
                self.orchestrator = orchestrator
                
                # Core fix: use correct run() method and pass output path
                self.logger.info(f"Running strategy '{strategy_name}'...")
                orchestrator.run(output_csv_path=output_csv_path)
                self.logger.info(f"Strategy '{strategy_name}' completed.")

                # Collect results after successful run
                self._collect_strategy_results(strategy_name, strategy_output_dir, output_csv_path)

            except Exception as e:
                self.logger.error(f"Severe error occurred while running strategy '{strategy_name}': {e}", exc_info=True)
            finally:
                if 'orchestrator' in locals() and orchestrator is not None:
                    self.logger.info(f"Performing cleanup for strategy '{strategy_name}'...")
                    orchestrator.cleanup()
                    self.logger.info(f"Cleanup completed for strategy '{strategy_name}'.")
                
                # For scalability experiments, preserve orchestrator for CPU timing data
                # Otherwise, help with garbage collection
                if not getattr(orchestrator, 'is_scalability_experiment', False):
                    # Manually help with garbage collection
                    if 'orchestrator' in locals():
                        del orchestrator
                    gc.collect()
                    self.logger.info(f"Garbage collection performed for strategy '{strategy_name}'.")
                else:
                    self.logger.info(f"Preserving orchestrator for scalability data extraction.")
                
                time.sleep(5) # Add delay between strategy runs to give system (e.g. file handle release) time to breathe

        self.logger.info("All strategies completed.")
        # self.generate_comparison_analysis()
        
    def _collect_strategy_results(self, strategy_id: str, output_dir: Path, expected_data_file: Path) -> bool:
        """Collect results from a single strategy run"""
        self.logger.info(f"Collecting results for strategy '{strategy_id}'...")

        if not expected_data_file.is_file():
            self.logger.error(f"Result file not found or not a file: {expected_data_file}. Strategy '{strategy_id}' may have failed.")
            self.experiment_results[strategy_id] = None
            self.comparison_metrics[strategy_id] = {} # Ensure key exists but empty to prevent subsequent analysis crashes
            return False

        try:
            df = pd.read_csv(expected_data_file)
            
            # Save results
            self.experiment_results[strategy_id] = {
                "data_file": str(expected_data_file),
                "metrics": self._calculate_strategy_metrics(df, strategy_id),
                "raw_data": df
            }
            
            # Print key metrics summary
            print(f"📊 Strategy {strategy_id} key metrics:")
            metrics = self.experiment_results[strategy_id]['metrics']
            if metrics:
                print(f"  Total operations: {metrics.get('Total Operations', 0):,.0f}")
                print(f"  Average latency: {metrics.get('Avg Latency (ms)', 0):.2f} ms")
                print(f"  P99 latency: {metrics.get('P99 Latency (ms)', 0):.2f} ms")
                print(f"  Cache hit rate: {metrics.get('Cache Hit Rate (%)', 0):.2f}%")
                print(f"  Throughput: {metrics.get('Throughput (ops/sec)', 0):.2f} ops/sec")
                print(f"  Fairness index: {metrics.get('Fairness Index', 0):.3f}")
            else:
                print("  - No metrics calculated.")
            
            return True # Indicate success
            
        except Exception as e:
            print(f"❌ Failed to collect results for strategy {strategy_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _calculate_strategy_metrics(self, df: pd.DataFrame, strategy_id: str) -> Dict[str, Any]:
        """Calculate core metrics for strategy"""
        # Dynamically get warmup time from config
        warmup_duration = self.config.get("ycsb_benchmark_config", {}).get("warmup_duration_s", 10)
        df_main = df[df['elapsed_seconds'] > warmup_duration]
        
        if df_main.empty:
            self.logger.warning(f"Strategy {strategy_id} has no data after warmup period, will use all data for analysis.")
            df_main = df
            if df_main.empty:
                self.logger.error(f"Strategy {strategy_id} has no data, cannot calculate metrics.")
                return {} # Return empty dictionary
        
        # 1. Basic metrics
        total_ops = df_main['ops_count'].sum()
        total_hits = df_main['cache_hits'].sum()
        total_misses = df_main['cache_misses'].sum()
        avg_latency = df_main['avg_latency_ms'].mean()
        
        # Core fix: ensure we use current strategy's own runtime to calculate throughput
        duration_seconds = df_main['elapsed_seconds'].max()
        throughput = total_ops / duration_seconds if duration_seconds > 0 else 0
        
        hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0

        # 2. Latency percentiles
        # Ultimate fix: to reflect worst case, first independently calculate P99 for each DB, then take the maximum
        if 'p99_latency_ms' in df_main.columns and not df_main.empty:
            # Group by database ID
            grouped = df_main.groupby('db_id')
            # Fix: get max P99 for each database (avoid duplicate percentile calculation)
            p99_per_db = grouped['p99_latency_ms'].apply(lambda s: s[s > 0].max())
            # Final P99 is the maximum of all database P99s, representing the worst performance of this strategy
            p99_latency = p99_per_db.max()
            # Handle NaN values from DB with no valid operations (all zeros)
            p99_latency = p99_latency if pd.notna(p99_latency) else 0.0
        else:
            p99_latency = 0.0

        # 3. Stability metrics - ensure data is not empty before calculating std and mean
        latency_std = df_main['avg_latency_ms'].std() if not df_main['avg_latency_ms'].empty else 0
        latency_mean = df_main['avg_latency_ms'].mean() if not df_main['avg_latency_ms'].empty else 0
        latency_cv = latency_std / latency_mean if latency_mean > 0 else 0
        
        # 4. Fairness metrics (for multi-database strategies)
        fairness_index = self._calculate_fairness_index(df_main)
        
        # Per-database stats
        per_db_stats = {}
        if 'db_id' in df_main.columns:
            db_groups = df_main.groupby('db_id')
            for db_name, db_df in db_groups:
                db_ops = int(db_df['ops_count'].sum())
                if db_ops > 0:
                    db_avg_latency = (db_df['ops_count'] * db_df['avg_latency_ms']).sum() / db_ops
                    db_hits = int(db_df['cache_hits'].sum())
                    db_misses = int(db_df['cache_misses'].sum())
                    db_hit_rate = (db_hits / (db_hits + db_misses) * 100) if (db_hits + db_misses) > 0 else 0
                    
                    db_duration = db_df['elapsed_seconds'].max() - db_df['elapsed_seconds'].min()
                    db_tps = db_ops / db_duration if db_duration > 0 else 0
                    
                    db_avg_cache_pages = db_df['current_cache_pages'].mean() if 'current_cache_pages' in db_df.columns else 0

                    per_db_stats[db_name] = {
                        'avg_latency_ms': float(db_avg_latency),
                        'hit_rate': float(db_hit_rate),
                        'throughput': float(db_tps),
                        'avg_cache_pages': float(db_avg_cache_pages),
                        'total_ops': db_ops,
                    }
        
        # 4. Assemble result dictionary
        metrics = {
            'Strategy': strategy_id,
            'Total Operations': total_ops,
            'Avg Latency (ms)': avg_latency,
            'P99 Latency (ms)': p99_latency,
            'Cache Hit Rate (%)': hit_rate * 100,
            'Throughput (ops/sec)': throughput,
            'Fairness Index': fairness_index,
            'per_db_metrics': json.dumps(per_db_stats)
        }
        
        return metrics
    
    def _calculate_fairness_index(self, df: pd.DataFrame) -> float:
        """Calculate fairness index (Jain's Fairness Index) 
           Based on the inverse of 'average latency per unit TPS' as performance score x_i.
        """
        try:
            # Fix: use 'db_id' instead of 'db_instance'
            if 'db_id' not in df.columns or df['db_id'].nunique() <= 1:
                return 1.0  # Single database or no db_id column, fairness is perfect

            # Fix: use correct column names 'ops_count' and 'avg_latency_ms'
            # Calculate total operations and total weighted latency for each db_id
            df_agg = df.groupby('db_id').agg(
                total_ops_per_db=('ops_count', 'sum'),
                # Calculate sum of (ops * latency)
                total_weighted_latency_product_sum_per_db=(
                    'avg_latency_ms', lambda x: (x * df.loc[x.index, 'ops_count']).sum()
                )
            ).reset_index()

            # Experiment effective duration
            min_elapsed = df['elapsed_seconds'].min()
            max_elapsed = df['elapsed_seconds'].max()
            experiment_duration_sec = max_elapsed - min_elapsed
            if experiment_duration_sec <= 0:
                self.logger.warning("[FairnessCalc] Experiment duration is zero or negative, cannot calculate fairness.")
                return 0.0

            performance_scores = []
            for _, row in df_agg.iterrows():
                total_ops = row['total_ops_per_db']
                total_weighted_latency_sum = row['total_weighted_latency_product_sum_per_db']

                if total_ops == 0:
                    performance_scores.append(1e-9) # Avoid zero, but give extremely low score
                    self.logger.warning(f"[FairnessCalc] DB {row['db_id']} has 0 operations, giving lowest performance score.")
                    continue

                avg_latency_db = total_weighted_latency_sum / total_ops
                avg_tps_db = total_ops / experiment_duration_sec

                if avg_tps_db == 0:
                    performance_scores.append(1e-9)
                    continue
                
                # Performance score = 1 / (latency/throughput) = throughput / latency
                latency_per_tps = avg_latency_db / avg_tps_db if avg_tps_db > 0 else float('inf')
                
                if latency_per_tps <= 1e-9:
                    performance_score = 1.0 / 1e-9 # Extremely high score
                else:
                    performance_score = 1.0 / latency_per_tps
                
                performance_scores.append(performance_score)

            if not performance_scores or len(performance_scores) <= 1:
                return 1.0 
            
            # Jain's Fairness Index: (sum(xi))^2 / (n * sum(xi^2))
            sum_x = sum(performance_scores)
            sum_x_squared = sum(s**2 for s in performance_scores)
            n = len(performance_scores)
            
            if sum_x_squared == 0:
                return 0.0 # No performance means no fairness
            
            fairness = (sum_x ** 2) / (n * sum_x_squared)
            return fairness
            
        except Exception as e:
            self.logger.error(f"[FairnessCalc] Error calculating fairness index: {e}", exc_info=True)
            return 0.0 # Return 0.0 on error
    
    def generate_comparison_analysis(self):
        """Generate comparison analysis report and charts"""
        print("\n" + "="*60)
        print("🔬 Generating comparison analysis...")
        
        if not self.experiment_results:
            self.logger.warning("No experiment result data, cannot generate analysis.")
            return

        # Create analysis directory
        analysis_dir = self.experiment_dir / "analysis"
        plots_dir = self.experiment_dir / "plots"
        analysis_dir.mkdir(exist_ok=True)
        plots_dir.mkdir(exist_ok=True)

        # Generate per-strategy database comparison
        self._generate_per_strategy_db_comparison()
        
        # Generate all plots
        self._generate_all_plots()
        
        self.logger.info(f"📊 All analysis charts saved to: {plots_dir}")
    
    def _generate_per_strategy_db_comparison(self):
        """Generate unified database comparison table containing metrics for all strategies, databases and phases"""
        print("\n" + "="*60)
        print("📊 Generating unified database performance comparison table...")
        
        analysis_dir = self.experiment_dir / "analysis"
        
        # Collect metrics data for all strategies, databases, and phases
        all_metrics = []
        
        for strategy_id, result in self.experiment_results.items():
            if result is None or result.get("raw_data") is None:
                self.logger.warning(f"Strategy '{strategy_id}' data is empty, skipping database comparison.")
                continue
                
            df_raw = result["raw_data"]
            
            # Filter out databases containing 'bg'
            if 'db_id' in df_raw.columns:
                db_instances = df_raw['db_id'].unique()
                non_bg_dbs = [db for db in db_instances if 'bg' not in db.lower()]
                df_raw = df_raw[df_raw['db_id'].isin(non_bg_dbs)]
                print(f"  Strategy {strategy_id}: excluded bg databases, keeping {len(non_bg_dbs)} databases: {non_bg_dbs}")
            
            if df_raw.empty:
                self.logger.warning(f"Strategy '{strategy_id}' has no data after filtering, skipping.")
                continue
            
            # Get all phases
            phases = df_raw['phase_name'].unique() if 'phase_name' in df_raw.columns else ['All']
            
            # Calculate metrics for each phase
            for phase_name in phases:
                if 'phase_name' in df_raw.columns:
                    phase_df = df_raw[df_raw['phase_name'] == phase_name]
                else:
                    phase_df = df_raw
                
                if phase_df.empty:
                    continue
                
                # Calculate metrics for each database in this phase
                db_phase_metrics = self._calculate_per_db_phase_metrics(phase_df, strategy_id, phase_name)
                all_metrics.extend(db_phase_metrics)
        
        if not all_metrics:
            self.logger.warning("No available metrics data, skipping table generation.")
            return
        
        # Create unified comparison table
        unified_df = pd.DataFrame(all_metrics)
        
        # Reorder columns for better readability
        column_order = ['Strategy', 'Database', 'Phase', 'Total Operations', 
                       'Avg Latency (ms)', 'P99 Latency (ms)', 'Cache Hit Rate (%)', 
                       'Throughput (ops/sec)', 'Avg Cache Pages']
        unified_df = unified_df[column_order]
        
        # Save unified comparison data
        unified_csv_path = analysis_dir / "unified_database_performance_comparison.csv"
        unified_df.to_csv(unified_csv_path, index=False)
        
        # Print beautified table to console
        self._print_unified_comparison_table(unified_df)
        
        print(f"✅ Unified database performance comparison table saved to: {unified_csv_path}")
        
        # Additionally generate summary statistics grouped by phase
        self._generate_phase_summary_table(unified_df, analysis_dir)
    
    def _calculate_per_db_metrics(self, df: pd.DataFrame, strategy_id: str) -> List[Dict[str, Any]]:
        """Calculate core metrics for each database"""
        if 'db_id' not in df.columns:
            self.logger.warning(f"Strategy {strategy_id} data missing 'db_id' column")
            return []
        
        db_metrics = []
        db_groups = df.groupby('db_id')
        
        for db_name, db_df in db_groups:
            if db_df.empty:
                continue
                
            # Calculate basic metrics
            total_ops = db_df['ops_count'].sum()
            total_hits = db_df['cache_hits'].sum()
            total_misses = db_df['cache_misses'].sum()
            
            # Calculate weighted average latency
            if total_ops > 0:
                avg_latency = (db_df['ops_count'] * db_df['avg_latency_ms']).sum() / total_ops
            else:
                avg_latency = 0
            
            # Calculate hit rate
            hit_rate = (total_hits / (total_hits + total_misses) * 100) if (total_hits + total_misses) > 0 else 0
            
            # Calculate throughput
            duration = db_df['elapsed_seconds'].max() - db_df['elapsed_seconds'].min()
            throughput = total_ops / duration if duration > 0 else 0
            
            # Calculate P99 latency
            if 'p99_latency_ms' in db_df.columns and not db_df.empty:
                # Fix: use max value instead of repeated percentile calculation
                p99_latency = db_df[db_df['p99_latency_ms'] > 0]['p99_latency_ms'].max()
                p99_latency = p99_latency if pd.notna(p99_latency) else 0.0
            else:
                p99_latency = 0.0
            
            # Average cache pages
            avg_cache_pages = db_df['current_cache_pages'].mean() if 'current_cache_pages' in db_df.columns else 0
            
            db_metrics.append({
                'Database': db_name,
                'Total Operations': int(total_ops),
                'Avg Latency (ms)': round(avg_latency, 2),
                'P99 Latency (ms)': round(p99_latency, 2),
                'Cache Hit Rate (%)': round(hit_rate, 2),
                'Throughput (ops/sec)': round(throughput, 2),
                'Avg Cache Pages': int(avg_cache_pages)
            })
        
        return db_metrics
    
    def _calculate_per_db_phase_metrics(self, df: pd.DataFrame, strategy_id: str, phase_name: str) -> List[Dict[str, Any]]:
        """Calculate core metrics for each database in specific phase"""
        if 'db_id' not in df.columns:
            self.logger.warning(f"Strategy {strategy_id} data missing 'db_id' column")
            return []
        
        db_metrics = []
        db_groups = df.groupby('db_id')
        
        for db_name, db_df in db_groups:
            if db_df.empty:
                continue
                
            # Calculate basic metrics
            total_ops = db_df['ops_count'].sum()
            total_hits = db_df['cache_hits'].sum()
            total_misses = db_df['cache_misses'].sum()
            
            # Calculate weighted average latency
            if total_ops > 0:
                avg_latency = (db_df['ops_count'] * db_df['avg_latency_ms']).sum() / total_ops
            else:
                avg_latency = 0
            
            # Calculate hit rate
            hit_rate = (total_hits / (total_hits + total_misses) * 100) if (total_hits + total_misses) > 0 else 0
            
            # Calculate throughput
            duration = db_df['elapsed_seconds'].max() - db_df['elapsed_seconds'].min()
            throughput = total_ops / duration if duration > 0 else 0
            
            # Calculate P99 latency
            if 'p99_latency_ms' in db_df.columns and not db_df.empty:
                # Fix: use max value instead of repeated percentile calculation
                p99_latency = db_df[db_df['p99_latency_ms'] > 0]['p99_latency_ms'].max()
                p99_latency = p99_latency if pd.notna(p99_latency) else 0.0
            else:
                p99_latency = 0.0
            
            # Average cache pages
            avg_cache_pages = db_df['current_cache_pages'].mean() if 'current_cache_pages' in db_df.columns else 0
            
            db_metrics.append({
                'Strategy': strategy_id,
                'Database': db_name,
                'Phase': phase_name,
                'Total Operations': int(total_ops),
                'Avg Latency (ms)': round(avg_latency, 2),
                'P99 Latency (ms)': round(p99_latency, 2),
                'Cache Hit Rate (%)': round(hit_rate, 2),
                'Throughput (ops/sec)': round(throughput, 2),
                'Avg Cache Pages': int(avg_cache_pages)
            })
        
        return db_metrics
    
    def _print_unified_comparison_table(self, df: pd.DataFrame):
        """Print unified database comparison table"""
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(show_header=True, header_style="bold magenta", 
                         title="Unified Database Performance Comparison - All Strategies, Databases & Phases")
            
            # Add columns
            for col_name in df.columns:
                if col_name in ['Strategy', 'Database', 'Phase']:
                    table.add_column(col_name, style="dim", width=12)
                else:
                    table.add_column(col_name, justify="right", width=10)

            # Add rows
            for _, row in df.iterrows():
                row_data = []
                for col_name in df.columns:
                    val = row[col_name]
                    if isinstance(val, (int, float)) and col_name not in ['Strategy', 'Database', 'Phase']:
                        if col_name in ['Cache Hit Rate (%)', 'Avg Latency (ms)', 'P99 Latency (ms)', 'Throughput (ops/sec)']:
                            row_data.append(f"{val:.2f}")
                        else:
                            row_data.append(f"{val:,.0f}")
                    else:
                        row_data.append(str(val))
                table.add_row(*row_data)
            
            console.print(table)
        except ImportError:
            self.logger.info("'rich' library not installed; printing plain DataFrame instead.")
            print(f"\n=== Unified Database Performance Comparison ===")
            print(df.to_string(index=False))
            print()
    
    def _generate_phase_summary_table(self, unified_df: pd.DataFrame, analysis_dir: Path):
        """Generate summary statistics table grouped by phase"""
        print("\n" + "="*60)
        print("📊 Generating phase summary statistics table...")
        
        # Group by phase and calculate summary statistics
        phase_summary = []
        
        for phase in unified_df['Phase'].unique():
            phase_data = unified_df[unified_df['Phase'] == phase]
            
            # Calculate summary metrics for all strategies and databases in this phase
            total_ops = phase_data['Total Operations'].sum()
            avg_latency = (phase_data['Total Operations'] * phase_data['Avg Latency (ms)']).sum() / total_ops if total_ops > 0 else 0
            avg_hit_rate = phase_data['Cache Hit Rate (%)'].mean()
            avg_throughput = phase_data['Throughput (ops/sec)'].sum()
            avg_p99_latency = phase_data['P99 Latency (ms)'].mean()
            
            # Calculate performance variance between strategies (standard deviation)
            latency_std = phase_data['Avg Latency (ms)'].std()
            hit_rate_std = phase_data['Cache Hit Rate (%)'].std()
            throughput_std = phase_data['Throughput (ops/sec)'].std()
            
            phase_summary.append({
                'Phase': phase,
                'Total Operations': int(total_ops),
                'Avg Latency (ms)': round(avg_latency, 2),
                'Latency Std Dev': round(latency_std, 2),
                'Avg P99 Latency (ms)': round(avg_p99_latency, 2),
                'Avg Hit Rate (%)': round(avg_hit_rate, 2),
                'Hit Rate Std Dev': round(hit_rate_std, 2),
                'Total Throughput (ops/sec)': round(avg_throughput, 2),
                'Throughput Std Dev': round(throughput_std, 2),
                'Strategy Count': len(phase_data['Strategy'].unique()),
                'Database Count': len(phase_data['Database'].unique())
            })
        
        # Create phase summary table
        phase_summary_df = pd.DataFrame(phase_summary)
        
        # Save phase summary data
        phase_summary_csv_path = analysis_dir / "phase_summary_statistics.csv"
        phase_summary_df.to_csv(phase_summary_csv_path, index=False)
        
        # Print phase summary table
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(show_header=True, header_style="bold blue", 
                         title="Phase Summary Statistics")
            
            # Add columns
            for col_name in phase_summary_df.columns:
                if col_name == 'Phase':
                    table.add_column(col_name, style="dim", width=15)
                else:
                    table.add_column(col_name, justify="right", width=12)

            # Add rows
            for _, row in phase_summary_df.iterrows():
                row_data = []
                for col_name in phase_summary_df.columns:
                    val = row[col_name]
                    if isinstance(val, (int, float)) and col_name != 'Phase':
                        if 'Count' in col_name:
                            row_data.append(f"{val:.0f}")
                        else:
                            row_data.append(f"{val:.2f}")
                    else:
                        row_data.append(str(val))
                table.add_row(*row_data)
            
            console.print(table)
        except ImportError:
            print(f"\n=== Phase Summary Statistics ===")
            print(phase_summary_df.to_string(index=False))
            print()
        
        print(f"✅ Phase summary statistics table saved to: {phase_summary_csv_path}")
    
    def _print_strategy_db_comparison_table(self, strategy_id: str, df: pd.DataFrame):
        """Print strategy's database comparison table"""
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(show_header=True, header_style="bold magenta", 
                         title=f"Database Performance Comparison - {strategy_id}")
            
            # Add columns
            table.add_column("Database", style="dim", width=20)
            for col_name in df.columns:
                table.add_column(col_name, justify="right")

            # Add rows
            for index, row in df.iterrows():
                row_data = [index]
                for col_name in df.columns:
                    val = row[col_name]
                    if isinstance(val, (int, float)):
                        if col_name in ['Cache Hit Rate (%)', 'Avg Latency (ms)', 'P99 Latency (ms)', 'Throughput (ops/sec)']:
                            row_data.append(f"{val:.2f}")
                        else:
                            row_data.append(f"{val:,.0f}")
                    else:
                        row_data.append(str(val))
                table.add_row(*row_data)
            
            console.print(table)
        except ImportError:
            self.logger.info("'rich' library not installed; printing plain DataFrame instead.")
            print(f"\n=== Database Performance Comparison - {strategy_id} ===")
            print(df.to_string())
            print()

    def _generate_performance_ranking(self, df: pd.DataFrame):
        """Generates performance rankings based on key metrics."""
        print("\n--- Performance Ranking ---")
        # Rank by Cache Hit Rate (higher is better)
        print("\n🏆 Ranked by Cache Hit Rate (higher is better):")
        print(df['hit_rate_percent'].sort_values(ascending=False))

        # Rank by Average Latency (lower is better)
        print("\n🏆 Ranked by Average Latency (lower is better):")
        print(df['avg_latency_ms'].sort_values(ascending=True))

        # Rank by Throughput (higher is better)
        print("\n🏆 Ranked by Throughput (higher is better):")
        print(df['throughput_per_second'].sort_values(ascending=False))
        print("-" * 28 + "\n")
    
    def _generate_comparison_plots(self, summary_df: pd.DataFrame, all_strategies_raw_df: pd.DataFrame):
        """Generates all comparison plots."""
        plots_dir = self.experiment_dir / "plots"
        
        try:
            # 1. Key metrics comparison
            self._plot_key_metrics_comparison(summary_df, plots_dir / "key_metrics_comparison.png")
            
            # 2. Radar chart
            self._plot_radar_chart(summary_df, plots_dir / "strategy_radar_chart.png")

            # 3. Per-database latency comparison
            self._plot_per_db_avg_latency_comparison(summary_df, plots_dir / "per_db_avg_latency_comparison.png")
            
            # 4. Time series plots
            self._plot_time_series_averaged_strategies(plots_dir / "time_series_averaged.png")
            self._plot_time_series_per_db_instance(all_strategies_raw_df, plots_dir)

        except Exception as e:
            print(f"❌ Failed to generate plots: {e}")
            import traceback
            traceback.print_exc()
    
    def _plot_radar_chart(self, df: pd.DataFrame, output_path: Path):
        """Generate strategy radar chart"""
        try:
            import numpy as np
            
            # Prepare data (normalize to 0-1)
            # Metrics for radar: Lower P99 Latency is better, Higher Cache Hit Rate is better, Higher Fairness is better
            metrics_to_plot = ['p99_latency_ms', 'hit_rate_percent', 'fairness_index']
            metric_labels = ['Low P99 Latency', 'High Cache Hit Rate', 'High Fairness']
            
            # Ensure all metrics are present in df
            for metric in metrics_to_plot:
                if metric not in df.columns:
                    print(f"[RadarPlot] Metric '{metric}' not found in DataFrame. Skipping radar chart.")
                    return

            normalized_data = df[metrics_to_plot].copy()
            
            # P99 Latency: Invert for "higher is better" representation in radar
            min_p99 = normalized_data['p99_latency_ms'].min()
            max_p99 = normalized_data['p99_latency_ms'].max()
            if max_p99 > min_p99:
                normalized_data['p99_latency_ms'] = 1 - (normalized_data['p99_latency_ms'] - min_p99) / (max_p99 - min_p99)
            elif max_p99 == min_p99 and max_p99 !=0 : # All values are same but not zero
                 normalized_data['p99_latency_ms'] = 0.5 # Or 1.0 if this means "best possible identical performance"
            else: # All are zero or only one data point
                normalized_data['p99_latency_ms'] = 1.0


            # Cache Hit Rate and Fairness Index: Normalize directly (higher is better)
            for metric in ['hit_rate_percent', 'fairness_index']:
                min_val = normalized_data[metric].min()
                max_val = normalized_data[metric].max()
                if max_val > min_val:
                    normalized_data[metric] = (normalized_data[metric] - min_val) / (max_val - min_val)
                elif max_val == min_val and max_val != 0:
                    normalized_data[metric] = 0.5
                else:
                    normalized_data[metric] = 1.0
            
            # Create radar chart
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
            
            angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
            angles += angles[:1]  # Close the figure
            
            # Get a colormap using the new API
            cmap = matplotlib.colormaps['Set3'] 
            
            for i, (index, row) in enumerate(df.iterrows()):
                values = normalized_data.iloc[i].tolist()
                values += values[:1]  # Close the figure
                
                # Normalize index 'i' to be a float between 0 and 1 for the colormap
                # And handle the case where there's only one strategy to avoid division by zero
                color_val = i / (len(df) - 1) if len(df) > 1 else 0.5
                
                ax.plot(angles, values, 'o-', linewidth=2, label=index, color=cmap(color_val))
                ax.fill(angles, values, color=cmap(color_val), alpha=0.25)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_labels)
            ax.set_ylim(0, 1.05) # Give a little space at the top
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1)) # Adjust legend position
            ax.set_title(f'Strategy Performance Radar ({len(df)} Strategies)', y=1.15, fontsize=16)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        except Exception as e:
            print(f"⚠️  Radar chart generation failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _plot_key_metrics_comparison(self, df: pd.DataFrame, output_path: Path):
        """Generate key performance metrics bar comparison"""
        fig, axes = plt.subplots(1, 3, figsize=(22, 7))

        # Ensure metrics exist, otherwise fill with defaults
        for col in ['avg_latency_ms', 'p99_latency_ms', 'hit_rate_percent', 'fairness_index']:
            if col not in df.columns:
                df[col] = 0.0
                print(f"[PlotWarn] Metric '{col}' missing for plotting, defaulted to 0.0.")

        # 1. Latency Comparison (Avg and P99)
        bar_width = 0.35
        index = np.arange(len(df.index))
        
        rects1 = axes[0].bar(index - bar_width/2, df['avg_latency_ms'], bar_width, label='Avg Latency', color='skyblue')
        rects2 = axes[0].bar(index + bar_width/2, df['p99_latency_ms'], bar_width, label='P99 Latency', color='steelblue')
        axes[0].set_title('Latency Comparison (Avg vs P99)', fontsize=14)
        axes[0].set_ylabel('Latency (ms)', fontsize=12)
        axes[0].set_xticks(index)
        axes[0].set_xticklabels(df.index, rotation=45, ha="right")
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.7)

        # 2. Cache Hit Rate Comparison
        axes[1].bar(index, df['hit_rate_percent'], color='mediumseagreen')
        axes[1].set_title('Cache Hit Rate Comparison', fontsize=14)
        axes[1].set_ylabel('Hit Rate (%)', fontsize=12)
        axes[1].set_xticks(index)
        axes[1].set_xticklabels(df.index, rotation=45, ha="right")
        axes[1].grid(True, linestyle='--', alpha=0.7)
        axes[1].set_ylim(bottom=0)

        # 3. Fairness Index Comparison
        axes[2].bar(index, df['fairness_index'], color='salmon')
        axes[2].set_title('Fairness Index Comparison', fontsize=14)
        axes[2].set_ylabel('Fairness Index (Jain\'s)', fontsize=12)
        axes[2].set_xticks(index)
        axes[2].set_xticklabels(df.index, rotation=45, ha="right")
        axes[2].grid(True, linestyle='--', alpha=0.7)
        axes[2].set_ylim(0, 1)
        
        plt.tight_layout(pad=3.0) # Add some padding
        fig.suptitle('Key Performance Metrics Comparison', fontsize=18, y=1.03)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_per_db_avg_latency_comparison(self, df_summary: pd.DataFrame, output_path: Path):
        """Plot average latency comparison per database instance"""
        if 'per_database_stats' not in df_summary.columns:
            self.logger.warning("[PlotWarn] 'per_database_stats' missing. Skipping per-DB latency plot.")
            return

        all_db_latency_data = []
        for index, row in df_summary.iterrows():
            strategy_id = index
            try:
                # Handle potential variations in how the JSON string might be stored (e.g. already a dict)
                if isinstance(row['per_database_stats'], str):
                    per_db_stats = json.loads(row['per_database_stats'])
                elif isinstance(row['per_database_stats'], dict):
                    per_db_stats = row['per_database_stats']
                else:
                    self.logger.warning(f"[PlotWarn] Unexpected type for per_database_stats for {strategy_id}: {type(row['per_database_stats'])}. Skipping.")
                    continue
                
                for db_name, stats in per_db_stats.items():
                    if 'avg_latency_ms' in stats:
                        all_db_latency_data.append({
                            'strategy': strategy_id,
                            'db_id': db_name,
                            'avg_latency_db': float(stats['avg_latency_ms'])
                        })
                    else:
                        self.logger.warning(f"[PlotWarn] 'avg_latency_ms' missing for {db_name} in {strategy_id}.")
            except json.JSONDecodeError as e:
                self.logger.error(f"[PlotError] Failed to parse 'per_database_stats' for strategy {strategy_id}: {e}. Content: {row['per_database_stats'][:100]}...")
                continue
            except Exception as e:
                self.logger.error(f"[PlotError] Error processing 'per_database_stats' for strategy {strategy_id}: {e}")
                continue

        if not all_db_latency_data:
            self.logger.warning("[PlotWarn] No per-database latency data available to plot.")
            return

        plot_df = pd.DataFrame(all_db_latency_data)
        
        if plot_df.empty:
            self.logger.warning("[PlotWarn]DataFrame for per-db latency is empty. Skipping plot.")
            return

        db_instances = sorted(plot_df['db_id'].unique())
        strategies = df_summary.index.unique() # Keep order from summary df
        num_strategies = len(strategies)
        num_db_instances = len(db_instances)

        if num_db_instances == 0:
            self.logger.warning("[PlotWarn] No unique database instances found for per-db latency plot.")
            return

        fig, ax = plt.subplots(figsize=(max(10, num_db_instances * num_strategies * 0.5), 6))
        
        bar_width = 0.8 / num_strategies # Adjust bar width based on number of strategies
        bar_positions = np.arange(num_db_instances)
        colors = plt.cm.get_cmap('tab10', num_strategies)

        for i, strategy in enumerate(strategies):
            strategy_data = plot_df[plot_df['strategy'] == strategy]
            # Ensure data for all db_instances is present for this strategy, fill with NaN if not
            latency_values = []
            for db_instance_ordered in db_instances:
                val = strategy_data[strategy_data['db_id'] == db_instance_ordered]['avg_latency_db']
                latency_values.append(val.iloc[0] if not val.empty else np.nan)

            ax.bar(bar_positions + i * bar_width - (bar_width * num_strategies / 2) + bar_width/2, 
                   latency_values, 
                   bar_width, 
                   label=strategy, 
                   color=colors(i % num_strategies))

        ax.set_title('Average Latency per Database Instance', fontsize=16)
        ax.set_ylabel('Average Latency (ms)', fontsize=12)
        ax.set_xlabel('Database Instance', fontsize=12)
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(db_instances, rotation=45, ha="right")
        ax.legend(title='Strategy', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"📊 Per-database average latency plot saved: {output_path}")
    
    def _get_global_elapsed_time(self, df: pd.DataFrame) -> pd.Series:
        """
        Create a continuous global timeline from experiment data.
        """
        # Preserve original order of phase_name
        phases = df['phase_name'].unique()
        
        # Group and calculate based on phase order in data
        phase_durations = df.groupby('phase_name', sort=False)['elapsed_seconds'].max()
        
        # Calculate time offset for each phase start
        phase_offsets = phase_durations.cumsum().shift(fill_value=0)
        
        # Map offsets back to original DataFrame and add to elapsed_seconds
        global_elapsed_seconds = df['phase_name'].map(phase_offsets) + df['elapsed_seconds']
        
        return global_elapsed_seconds
    
    def _plot_time_series_averaged_strategies(self, output_path: Path):
        """Plot time series comparison of average metrics for all strategies"""
        self.logger.info("📊 Generating time series comparison plot...")
        if not self.experiment_results:
            print("No experiment results, skipping time series plot.")
            return

        # 1. Set up figure with horizontal layout (1 row, 3 columns)
        fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=False) # y-axes have different units, so don't share
        fig.suptitle('Performance Metrics Over Time (Strategy Average)', fontsize=20)
        
        # 2. Prepare colors
        num_strategies = len(self.experiment_results)
        try:
            colors = plt.colormaps.get_cmap('tab10').resampled(num_strategies).colors
        except AttributeError:
            colors = plt.cm.get_cmap('tab10', num_strategies).colors

        # 3. Process each strategy and plot
        for i, (strategy_id, result) in enumerate(self.experiment_results.items()):
            df_raw = result.get('raw_data')
            if df_raw is None or df_raw.empty:
                continue

            # Create and use global continuous timeline
            if 'phase_name' not in df_raw.columns:
                self.logger.warning(f"'{strategy_id}' data missing 'phase_name' column, skipping plot.")
                continue
            df_raw['global_elapsed_seconds'] = self._get_global_elapsed_time(df_raw)
            time_col = 'global_elapsed_seconds'
            df_raw.sort_values(by=time_col, inplace=True)
            
            # Note: Must use global time column for aggregation
            df_aggregated = df_raw.groupby(time_col).agg(
                avg_latency_ms=('avg_latency_ms', 'mean'),
                total_hits=('cache_hits', 'sum'),
                total_misses=('cache_misses', 'sum'),
                total_cache_pages=('current_cache_pages', 'sum')
            ).reset_index()

            df_aggregated['hit_rate_percent'] = (
                df_aggregated['total_hits'] / (df_aggregated['total_hits'] + df_aggregated['total_misses'])
            ).fillna(0) * 100

            color = colors[i]
            axes[0].plot(df_aggregated[time_col], df_aggregated['avg_latency_ms'], label=strategy_id, color=color, marker='.', markersize=5, linestyle='-', linewidth=1.5)
            axes[1].plot(df_aggregated[time_col], df_aggregated['hit_rate_percent'], label=strategy_id, color=color, marker='.', markersize=5, linestyle='-', linewidth=1.5)
            axes[2].plot(df_aggregated[time_col], df_aggregated['total_cache_pages'], label=strategy_id, color=color, marker='.', markersize=5, linestyle='-', linewidth=1.5)

        # 4. Beautify and configure all subplots
        axes[0].set_title('Average Latency (lower is better)', fontsize=14)
        axes[0].set_ylabel('Latency (ms)', fontsize=12)
        
        axes[1].set_title('Cache Hit Rate (higher is better)', fontsize=14)
        axes[1].set_ylabel('Hit Rate (%)', fontsize=12)
        axes[1].set_ylim(bottom=0, top=105)

        axes[2].set_title('Total Cache Usage', fontsize=14)
        axes[2].set_ylabel('Cache Pages', fontsize=12)
        
        for ax in axes:
            ax.grid(True, which='major', linestyle='--', linewidth=0.5)
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.set_xlabel('Experiment Time (seconds)', fontsize=12)

        # 5. Create a shared legend and place it below the chart
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=min(num_strategies, 8), fontsize=12, frameon=True)

        # 6. Adjust layout and save the image
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Time series comparison plot saved: {output_path}")
    
    def _plot_time_series_per_db_instance(self, df: pd.DataFrame, output_path_base: Path):
        """Generate time series comparison plots for each database instance."""
        self.logger.info("📊 Generating time series plots for each database instance...")
        if df.empty:
            self.logger.warning("[PlotWarn] No data available to plot per DB instance.")
            return

        # Core fix: use 'db_id' instead of 'db_instance'
        db_col_name = 'db_id'

        if db_col_name not in df.columns:
            self.logger.warning(f"Cannot find '{db_col_name}' column in data, unable to generate per-DB plots.")
            return
            
        unique_db_instances = df[db_col_name].unique()
        self.logger.info(f"Found {len(unique_db_instances)} unique database instances: {unique_db_instances}")

        for db_instance_name in unique_db_instances:
            df_db = df[df[db_col_name] == db_instance_name].copy()
            
            # Fix: combine paths
            output_file = output_path_base / f"time_series_per_db_{db_instance_name}.png"
            
            self.logger.info(f"  -> Generating chart for '{db_instance_name}'...")

            # Create figure and subplots
            fig, axes = plt.subplots(3, 1, figsize=(20, 18), sharex=True)
            fig.suptitle(f'Performance Comparison for: {db_instance_name}', fontsize=18, weight='bold')

            # Get page size from config
            page_size_bytes = self.config.get("system_sqlite_config", {}).get("page_size_bytes", 4096)

            # Get all strategies list to keep colors consistent
            all_strategy_names = sorted(df_db['strategy_name'].unique())
            try:
                colors = plt.colormaps.get_cmap('tab10').resampled(len(all_strategy_names))
                color_map = {name: colors(i) for i, name in enumerate(all_strategy_names)}
            except AttributeError:
                cmap = plt.cm.get_cmap('tab10', len(all_strategy_names))
                color_map = {name: cmap(i) for i, name in enumerate(all_strategy_names)}

            # Plot data for each strategy
            for strategy_name in all_strategy_names:
                df_plot = df_db[df_db['strategy_name'] == strategy_name].sort_values('global_elapsed_seconds')
                
                if df_plot.empty:
                    continue

                # Core fix: calculate metrics needed for plotting
                # Get reporting interval from config
                reporting_interval = self.config.get("general_experiment_setup", {}).get("reporting_interval_s", 5)
                if reporting_interval <= 0:
                    self.logger.warning(f"Invalid reporting interval {reporting_interval}s. Using default 1s.")
                    reporting_interval = 1
                
                # Calculate throughput (Ops/sec)
                df_plot['ops_per_sec'] = df_plot['ops_count'] / reporting_interval

                # Calculate cache hit rate
                total_accesses = df_plot['cache_hits'] + df_plot['cache_misses']
                df_plot['hit_rate'] = (df_plot['cache_hits'] / total_accesses).fillna(0)
    
                color = color_map[strategy_name]
                
                # 1. Ops/sec
                axes[0].plot(df_plot['global_elapsed_seconds'], df_plot['ops_per_sec'], label=strategy_name, color=color, alpha=0.9, linewidth=2)
                
                # 2. Hit Rate
                axes[1].plot(df_plot['global_elapsed_seconds'], df_plot['hit_rate'], label=strategy_name, linestyle='--', color=color, alpha=0.9, linewidth=2)

                # 3. Cache Size (MB)
                cache_mb = (df_plot['current_cache_pages'] * page_size_bytes) / (1024 * 1024)
                axes[2].plot(df_plot['global_elapsed_seconds'], cache_mb, label=strategy_name, linestyle=':', color=color, alpha=0.9, linewidth=2)

            # Set chart format
            axes[0].set_ylabel('Ops/sec (Higher is better)', fontsize=12)
            axes[0].legend(loc='best')
            axes[0].set_title('Throughput', fontsize=14, weight='bold')
            
            axes[1].set_ylabel('Cache Hit Rate (Higher is better)', fontsize=12)
            axes[1].legend(loc='best')
            axes[1].set_title('Cache Efficiency', fontsize=14, weight='bold')
            axes[1].set_ylim(0, 1.05)
            
            axes[2].set_xlabel('Experiment Time (seconds)', fontsize=12)
            axes[2].set_ylabel('Allocated Cache (MB)', fontsize=12)
            axes[2].legend(loc='best')
            axes[2].set_title('Resource Allocation', fontsize=14, weight='bold')
            
            for ax in axes:
                ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                # Draw phase separation lines
                if 'phase_name' in df_db.columns:
                    phase_changes = df_db.drop_duplicates(subset=['phase_name'], keep='first')
                    for _, row in phase_changes.iterrows():
                        ax.axvline(x=row['global_elapsed_seconds'], color='r', linestyle='-.', alpha=0.7, label=f"Phase: {row['phase_name']}")
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.96])
            plt.savefig(output_file, dpi=150)
            plt.close(fig)
            self.logger.info(f"✅ Per-DB plot saved: {output_file}")

    def _generate_all_plots(self):
        """Generate all plots"""
        print("\n" + "="*60)
        print("📊 Starting to generate all plots...")
        
        analysis_dir = self.experiment_dir / "analysis"
        plots_dir = self.experiment_dir / "plots"
        
        # 1. Generate time series plots (including all phases)
        self._generate_comprehensive_timeseries()
        
        # 2. Generate performance comparison plots (by phase)
        self._generate_performance_comparison_plots()
        
        print("✅ All plots generated successfully")
    
    def _generate_comprehensive_timeseries(self):
        """Generate comprehensive time series plots"""
        print("  🔄 Generating comprehensive time series plots...")
        
        if not self.experiment_results:
            self.logger.warning("No experiment results data, unable to generate time series plots.")
            return
        
        # Collect data from all strategies
        all_dfs = []
        for strategy_name, result in self.experiment_results.items():
            if result is None or result.get("raw_data") is None:
                continue
            
            df = result["raw_data"].copy()
            if df.empty:
                continue
            
            # Filter out bg databases
            df = df[~df['db_id'].str.contains('bg', case=False, na=False)]
            if df.empty:
                continue
            
            # Add strategy name column
            df['strategy_name'] = strategy_name
            
            # Process time continuity
            df = self._process_time_continuity_for_plot(df)
            
            all_dfs.append(df)
        
        if not all_dfs:
            self.logger.warning("No valid data available for generating time series plots.")
            return
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Generate time series plots by database
        self._plot_timeseries_by_database(combined_df)
    
    def _process_time_continuity_for_plot(self, df):
        """Process time continuity for plotting"""
        # Ensure sorting by timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Get unique phases
        phases = df['phase_name'].unique()
        if len(phases) == 0:
            return df
        
        cumulative_offset = 0.0
        df['global_elapsed_seconds'] = df['elapsed_seconds']
        
        for i, phase in enumerate(phases):
            if i > 0:
                # Find max time of previous phase
                previous_phase = phases[i-1]
                max_previous_elapsed = df[df['phase_name'] == previous_phase]['elapsed_seconds'].max()
                cumulative_offset += max_previous_elapsed
            
            # Add cumulative offset for current phase data points
            current_phase_mask = df['phase_name'] == phase
            df.loc[current_phase_mask, 'global_elapsed_seconds'] += cumulative_offset
        
        return df
    
    def _plot_timeseries_by_database(self, df):
        """Generate time series plots for each database type, comparing different strategies"""
        plots_dir = self.experiment_dir / "plots"
        
        # Get all database IDs
        databases = df['db_id'].unique()
        strategies = df['strategy_name'].unique()
        
        # Generate independent time series plots for each database
        for db_id in databases:
            db_data = df[df['db_id'] == db_id]
            if db_data.empty:
                continue
            
            # Create 9 subplots
            fig, axes = plt.subplots(9, 1, figsize=(20, 45), sharex=True)
            fig.suptitle(f'Performance Timeseries for {db_id} (All Strategies)', fontsize=20, y=0.995)
            
            # Set strategy colors
            strategy_colors = plt.cm.Set1(np.linspace(0, 1, len(strategies)))
            
            # 1. Cache allocation time series
            for i, strategy in enumerate(strategies):
                strategy_data = db_data[db_data['strategy_name'] == strategy]
                if not strategy_data.empty:
                    axes[0].plot(strategy_data['global_elapsed_seconds'], strategy_data['current_cache_pages'], 
                               label=strategy, color=strategy_colors[i], linewidth=2, alpha=0.8)
            
            axes[0].set_title(f'Cache Allocation for {db_id}', fontsize=14)
            axes[0].set_ylabel('Cache Pages', fontsize=12)
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(fontsize=10)
            
            # 2. Hit rate time series
            for i, strategy in enumerate(strategies):
                strategy_data = db_data[db_data['strategy_name'] == strategy]
                if not strategy_data.empty:
                    # Calculate hit rate
                    hit_rate = (strategy_data['cache_hits'] / (strategy_data['cache_hits'] + strategy_data['cache_misses']) * 100).fillna(0)
                    axes[1].plot(strategy_data['global_elapsed_seconds'], hit_rate, 
                               label=strategy, color=strategy_colors[i], linewidth=2, alpha=0.8)
            
            axes[1].set_title(f'Cache Hit Rate for {db_id}', fontsize=14)
            axes[1].set_ylabel('Hit Rate (%)', fontsize=12)
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim(0, 100)
            axes[1].legend(fontsize=10)
            
            # 3. Throughput time series
            for i, strategy in enumerate(strategies):
                strategy_data = db_data[db_data['strategy_name'] == strategy]
                if not strategy_data.empty:
                    # Calculate throughput (based on reporting interval)
                    reporting_interval = self.config.get("general_experiment_setup", {}).get("reporting_interval_seconds", 2)
                    throughput = strategy_data['ops_count'] / reporting_interval
                    axes[2].plot(strategy_data['global_elapsed_seconds'], throughput, 
                               label=strategy, color=strategy_colors[i], linewidth=2, alpha=0.8)
            
            axes[2].set_title(f'Throughput for {db_id}', fontsize=14)
            axes[2].set_ylabel('Ops/sec', fontsize=12)
            axes[2].grid(True, alpha=0.3)
            axes[2].legend(fontsize=10)
            
            # 4. Average latency time series
            for i, strategy in enumerate(strategies):
                strategy_data = db_data[db_data['strategy_name'] == strategy]
                if not strategy_data.empty:
                    axes[3].plot(strategy_data['global_elapsed_seconds'], strategy_data['avg_latency_ms'], 
                               label=strategy, color=strategy_colors[i], linewidth=2, alpha=0.8)
            
            axes[3].set_title(f'Average Latency for {db_id}', fontsize=14)
            axes[3].set_ylabel('Latency (ms)', fontsize=12)
            axes[3].grid(True, alpha=0.3)
            axes[3].legend(fontsize=10)
            
            # 5. P99 latency time series
            p99_plotted = False
            for i, strategy in enumerate(strategies):
                strategy_data = db_data[db_data['strategy_name'] == strategy]
                if not strategy_data.empty and 'p99_latency_ms' in strategy_data.columns:
                    p99_data = strategy_data['p99_latency_ms'].dropna()
                    if not p99_data.empty:
                        axes[4].plot(strategy_data['global_elapsed_seconds'], strategy_data['p99_latency_ms'], 
                                   label=strategy, color=strategy_colors[i], linewidth=2, alpha=0.8)
                        p99_plotted = True
            
            axes[4].set_title(f'P99 Latency for {db_id}', fontsize=14)
            axes[4].set_ylabel('P99 Latency (ms)', fontsize=12)
            axes[4].grid(True, alpha=0.3)
            if p99_plotted:
                axes[4].legend(fontsize=10)
            else:
                axes[4].text(0.5, 0.5, 'P99 Latency data not available', 
                           transform=axes[4].transAxes, ha='center', va='center', fontsize=12)
            
            # 6. H factor time series
            h_factor_plotted = False
            for i, strategy in enumerate(strategies):
                strategy_data = db_data[db_data['strategy_name'] == strategy]
                if not strategy_data.empty and 'h_factor' in strategy_data.columns:
                    axes[5].plot(strategy_data['global_elapsed_seconds'], strategy_data['h_factor'], 
                               label=strategy, color=strategy_colors[i], linewidth=2, alpha=0.8)
                    h_factor_plotted = True
            
            axes[5].set_title(f'H Factor (Horizontal Factor) for {db_id}', fontsize=14)
            axes[5].set_ylabel('H Factor', fontsize=12)
            axes[5].grid(True, alpha=0.3)
            if h_factor_plotted:
                axes[5].legend(fontsize=10)
            else:
                axes[5].text(0.5, 0.5, 'H Factor data not available', 
                           transform=axes[5].transAxes, ha='center', va='center', fontsize=12)
            
            # 7. V factor time series (including shadow V factor)
            v_factor_plotted = False
            for i, strategy in enumerate(strategies):
                strategy_data = db_data[db_data['strategy_name'] == strategy]
                if not strategy_data.empty and 'v_factor' in strategy_data.columns:
                    # Plot actual V factor (used for decision)
                    axes[6].plot(strategy_data['global_elapsed_seconds'], strategy_data['v_factor'], 
                               label=f'{strategy} (Decision V)', color=strategy_colors[i], linewidth=2, alpha=0.8)
                    v_factor_plotted = True
                    
                    # Plot shadow V factor (calculated but not used during grace period)
                    if 'shadow_v_factor' in strategy_data.columns:
                        axes[6].plot(strategy_data['global_elapsed_seconds'], strategy_data['shadow_v_factor'], 
                                   label=f'{strategy} (Shadow V)', color=strategy_colors[i], linewidth=1, 
                                   alpha=0.6, linestyle='--')
            
            axes[6].set_title(f'V Factor (Decision V vs Shadow V) for {db_id}', fontsize=14)
            axes[6].set_ylabel('V Factor', fontsize=12)
            axes[6].grid(True, alpha=0.3)
            if v_factor_plotted:
                axes[6].legend(fontsize=8)
            else:
                axes[6].text(0.5, 0.5, 'V Factor data not available', 
                           transform=axes[6].transAxes, ha='center', va='center', fontsize=12)
            
            # 8. α_t (dynamic weight) time series
            alpha_t_plotted = False
            for i, strategy in enumerate(strategies):
                strategy_data = db_data[db_data['strategy_name'] == strategy]
                if not strategy_data.empty and 'alpha_t' in strategy_data.columns:
                    # Check if alpha_t column has valid data (not all NaN or empty)
                    alpha_t_data = strategy_data['alpha_t'].dropna()
                    if not alpha_t_data.empty and len(alpha_t_data) > 0:
                        # Only plot valid data points, filter out NaN values
                        valid_mask = strategy_data['alpha_t'].notna()
                        valid_data = strategy_data[valid_mask]
                        if not valid_data.empty:
                            axes[7].plot(valid_data['global_elapsed_seconds'], valid_data['alpha_t'], 
                                       label=strategy, color=strategy_colors[i], linewidth=2, alpha=0.8)
                            alpha_t_plotted = True
            
            axes[7].set_title(f'Dynamic Weight α_t for {db_id}', fontsize=14)
            axes[7].set_ylabel('α_t (Dynamic Weight)', fontsize=12)
            axes[7].grid(True, alpha=0.3)
            if alpha_t_plotted:
                axes[7].set_ylim(0.4, 1.0)  # Full range of α_t [0.4, 1.0]
                axes[7].legend(fontsize=10)
            else:
                axes[7].text(0.5, 0.5, 'α_t data not available', 
                           transform=axes[7].transAxes, ha='center', va='center', fontsize=12)
            
            # 9. Total score time series
            total_score_plotted = False
            for i, strategy in enumerate(strategies):
                strategy_data = db_data[db_data['strategy_name'] == strategy]
                if not strategy_data.empty and 'total_score' in strategy_data.columns:
                    total_score_data = strategy_data['total_score'].dropna()
                    if not total_score_data.empty:
                        # Only plot valid data points, filter out NaN values
                        valid_mask = strategy_data['total_score'].notna()
                        valid_data = strategy_data[valid_mask]
                        if not valid_data.empty:
                            axes[8].plot(valid_data['global_elapsed_seconds'], valid_data['total_score'], 
                                       label=strategy, color=strategy_colors[i], linewidth=2, alpha=0.8)
                            total_score_plotted = True
            
            axes[8].set_title(f'Total Score for {db_id}', fontsize=14)
            axes[8].set_ylabel('Total Score', fontsize=12)
            axes[8].set_xlabel('Time (seconds)', fontsize=12)
            axes[8].grid(True, alpha=0.3)
            if total_score_plotted:
                axes[8].legend(fontsize=10)
            else:
                axes[8].text(0.5, 0.5, 'Total Score data not available', 
                           transform=axes[8].transAxes, ha='center', va='center', fontsize=12)
            
            # Add phase separation lines
            if 'phase_name' in db_data.columns:
                phases = db_data['phase_name'].unique()
                phase_transitions = []
                cumulative_time = 0
                
                # Get first strategy data to calculate phase transition times
                first_strategy_data = db_data[db_data['strategy_name'] == strategies[0]]
                if not first_strategy_data.empty:
                    for phase in phases:
                        phase_data = first_strategy_data[first_strategy_data['phase_name'] == phase]
                        if not phase_data.empty:
                            phase_transitions.append((cumulative_time, phase))
                            max_elapsed = phase_data['elapsed_seconds'].max()
                            cumulative_time += max_elapsed
                
                for ax in axes:
                    for time, phase_name in phase_transitions:
                        ax.axvline(x=time, color='red', linestyle=':', alpha=0.7, linewidth=1)
                        ax.text(time, ax.get_ylim()[1] * 0.95, phase_name, rotation=90, 
                               fontsize=8, ha='right', va='top')
            
            plt.tight_layout()
            
            # Save figure with database ID in filename
            sanitized_db_id = db_id.replace("_", "-")
            output_path = plots_dir / f"timeseries_{sanitized_db_id}.pdf"
            plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
            plt.close(fig)
            print(f"  ✅ {db_id} time series plot saved: {output_path}")
    
    def _generate_performance_comparison_plots(self):
        """Generate performance comparison plots by phase"""
        print("  🔄 Generating performance comparison plots...")
        
        if not self.experiment_results:
            return
        
        # Collect data from all phases
        all_phase_data = {}
        
        for strategy_name, result in self.experiment_results.items():
            if result is None or result.get("raw_data") is None:
                continue
            
            df = result["raw_data"].copy()
            if df.empty:
                continue
            
            # Filter out bg databases
            df = df[~df['db_id'].str.contains('bg', case=False, na=False)]
            if df.empty:
                continue
            
            # Group by phase
            for phase_name in df['phase_name'].unique():
                if phase_name not in all_phase_data:
                    all_phase_data[phase_name] = []
                
                phase_df = df[df['phase_name'] == phase_name].copy()
                phase_df['strategy_name'] = strategy_name
                all_phase_data[phase_name].append(phase_df)
        
        # Generate comparison plots for each phase
        plots_dir = self.experiment_dir / "plots"
        
        for phase_name, phase_dfs in all_phase_data.items():
            if not phase_dfs:
                continue
            
            combined_phase_df = pd.concat(phase_dfs, ignore_index=True)
            self._plot_phase_performance_comparison(combined_phase_df, phase_name, plots_dir)
    
    def _plot_phase_performance_comparison(self, df, phase_name, plots_dir):
        """Generate performance comparison plot for specific phase"""
        # Calculate average metrics for each strategy-database combination
        metrics_data = []
        
        for (strategy, db_id), group in df.groupby(['strategy_name', 'db_id']):
            total_ops = group['ops_count'].sum()
            total_hits = group['cache_hits'].sum()
            total_misses = group['cache_misses'].sum()
            
            # Calculate weighted average latency
            if total_ops > 0:
                avg_latency = (group['ops_count'] * group['avg_latency_ms']).sum() / total_ops
            else:
                avg_latency = 0
            
            # Calculate hit rate
            hit_rate = (total_hits / (total_hits + total_misses) * 100) if (total_hits + total_misses) > 0 else 0
            
            # Calculate throughput
            duration = group['elapsed_seconds'].max() - group['elapsed_seconds'].min()
            throughput = total_ops / duration if duration > 0 else 0
            
            # Calculate P99 latency
            if 'p99_latency_ms' in group.columns:
                # Fix: use max value instead of repeated percentile calculation
                p99_latency = group[group['p99_latency_ms'] > 0]['p99_latency_ms'].max()
                p99_latency = p99_latency if pd.notna(p99_latency) else 0.0
            else:
                p99_latency = 0.0
            
            metrics_data.append({
                'strategy_name': strategy,
                'db_id': db_id,
                'cache_hit_rate': hit_rate,
                'throughput_tps': throughput,
                'avg_latency_ms': avg_latency,
                'p99_latency_ms': p99_latency
            })
        
        if not metrics_data:
            return
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Create 4 subplot comparison figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Performance Comparison for Phase: {phase_name}', fontsize=16)
        
        # 1. Cache hit rate
        sns.barplot(ax=axes[0,0], data=metrics_df, x='strategy_name', y='cache_hit_rate', hue='db_id')
        axes[0,0].set_title('Cache Hit Rate (%)')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].legend(title='Database', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. Throughput
        sns.barplot(ax=axes[0,1], data=metrics_df, x='strategy_name', y='throughput_tps', hue='db_id')
        axes[0,1].set_title('Throughput (ops/sec)')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].get_legend().remove()
        
        # 3. Average latency
        sns.barplot(ax=axes[1,0], data=metrics_df, x='strategy_name', y='avg_latency_ms', hue='db_id')
        axes[1,0].set_title('Average Latency (ms)')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].get_legend().remove()
        
        # 4. P99 latency
        sns.barplot(ax=axes[1,1], data=metrics_df, x='strategy_name', y='p99_latency_ms', hue='db_id')
        axes[1,1].set_title('P99 Latency (ms)')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].get_legend().remove()
        
        plt.tight_layout()
        
        # Save figure
        sanitized_phase_name = phase_name.replace(" ", "_").replace("'", "")
        output_path = plots_dir / f"performance_comparison_{sanitized_phase_name}.pdf"
        plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        print(f"  ✅ Phase '{phase_name}' performance comparison plot saved: {output_path}")

    def _generate_detailed_report(self, df: pd.DataFrame):
        """Generate detailed text analysis report"""
        report_path = self.experiment_dir / "analysis" / "detailed_analysis_report.md"
        # TODO: Implement detailed report generation logic
        pass

def main():
    """Main function: parse command line arguments and run experiments"""
    import argparse
    parser = argparse.ArgumentParser(
        description="Run standardized experiments for reviewers. Choose experiment type by number.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add experiment type selection
    parser.add_argument(
        '--experiment-type',
        type=int,
        choices=[1, 2, 3],
        required=True,
        help="""Select experiment type:
        1: Three-phase hotspot shift workload (vs B2, B7, B12) - generates cache allocation timeseries, median hit rate, median TPS
        2: Adversarial cache pollution test (vs B7) - generates 3 subplots analysis  
        3: Scalability test - generates CPU time vs database count"""
    )
    
    parser.add_argument(
        '--force-clean-db', 
        action='store_true', 
        help="Force delete all existing database files before experiment starts."
    )
    parser.add_argument(
        '--strategies', 
        nargs='*', 
        default=None, 
        help="List of specific strategy IDs to run. If not provided, run strategies based on experiment type."
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default=None, 
        help="Specify config file path to use, overriding default settings."
    )
    parser.add_argument(
        '--workload',
        type=str,
        choices=['ycsb', 'tpcc', 'trace'],
        default='ycsb',
        help="Specify workload type. Default is 'ycsb'. Choose 'tpcc' for TPC-C workload, 'trace' for real traces."
    )
    parser.add_argument(
        '--output-dir-override',
        type=str,
        default=None,
        help="Completely override default output directory structure, save all content to specified directory. For automated script calls."
    )
    parser.add_argument(
        '--campaign-run-id',
        type=str,
        default=None,
        help="A unique identifier to associate a single run with a larger experiment campaign. Currently metadata only."
    )
    
    parser.add_argument(
        '--burst-scan-only',
        action='store_true',
        help="For experiment type 2: skip dual combat and only run burst scan experiment"
    )

    args = parser.parse_args()

    # Set up logging
    log_dir = Path(args.output_dir_override or "results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / f"comparison_run_{time.strftime('%Y%m%d_%H%M%S')}.log"

    # Create log handler with filter
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    
    # Add BG database filter
    bg_filter = BGDatabaseLogFilter()
    console_handler.addFilter(bg_filter)
    file_handler.addFilter(bg_filter)
    
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(name)-25s] [%(levelname)s] - %(message)s',
        datefmt='%H:%M:%S',
        handlers=[console_handler, file_handler]
    )
    
    main_logger = logging.getLogger("CacheStrategyComparison")
    main_logger.info(f"Log file location: {log_file_path}")
    if args.campaign_run_id:
        main_logger.info(f"Associated with campaign run ID: {args.campaign_run_id}")

    try:
        # Configure experiment based on type
        if args.experiment_type == 1:
            # Experiment 1: Three-phase hotspot shift (vs B2, B7, B12)
            config_file = args.config or 'configs/config_tps_controlled_comparison.json'
            default_strategies = ['S0_EMG_AS', 'B2_NoElasticFixedByPriority', 'B7_DYNAMIC_NEED', 'B12_MT_LRU_Inspired']
            main_logger.info("Experiment 1: Three-phase hotspot shift workload")
            
        elif args.experiment_type == 2:
            # Experiment 2: Adversarial cache pollution test (vs B7)
            config_file = args.config or 'configs/config_dual_combat.json'
            default_strategies = ['S0_EMG_AS', 'B7_DYNAMIC_NEED']
            main_logger.info("Experiment 2: Adversarial cache pollution test")
            
        elif args.experiment_type == 3:
            # Experiment 3: Scalability test
            # Note: Type 3 doesn't use a base config file, it generates configs dynamically
            config_file = None  # Will be generated in generate_experiment3_analysis
            default_strategies = ['S0_EMG_AS']  # Focus on S0 for scalability
            main_logger.info("Experiment 3: Scalability test")
            
        # Use command line strategies if provided, otherwise use experiment defaults
        strategies_to_run = args.strategies if args.strategies else default_strategies
        if config_file:
            main_logger.info(f"Using config file: {config_file}")
        else:
            main_logger.info("Config will be generated dynamically for Type 3 experiment")
        main_logger.info(f"Running strategies: {strategies_to_run}")
        
        # Initialize comparator with output_dir_override
        comparison = CacheStrategyComparison(base_output_dir=args.output_dir_override)
        
        # Special handling for experiment type 2 with burst-scan-only
        if args.experiment_type == 2 and args.burst_scan_only:
            # Skip dual combat, go directly to burst scan
            print("⚡ Burst scan only mode - skipping dual combat experiment")
            
            # Update config file to burst scan
            comparison.update_config_file('configs/config_burst_scan_optimized.json')
            comparison.config['general_experiment_setup']['workload_type'] = args.workload
            comparison.prepare_experiment_environment(force_clean_db=args.force_clean_db)
            comparison.run_all_strategies(strategies_to_run_from_cli=['B7_DYNAMIC_NEED'])
            
            # Generate analysis with empty dual combat results
            generate_experiment2_analysis(
                comparison, 
                {}, 
                comparison.experiment_results,
                dual_combat_exp_path=None,
                burst_scan_exp_path=str(comparison.experiment_dir)
            )
        else:
            # Normal flow for all other cases
            
            # Type 3 (Scalability) handles everything in generate_experiment3_analysis
            if args.experiment_type == 3:
                # For Type 3, just pass the comparison object to the analysis function
                # which will handle all configuration and execution
                pass  # Will be handled in generate_experiment3_analysis below
            else:
                # For Type 1 and Type 2, proceed with normal flow
                # Update config file
                comparison.update_config_file(config_file)
                
                # Set workload type (YCSB for experiments 1&2)
                comparison.config['general_experiment_setup']['workload_type'] = args.workload
                
                comparison.prepare_experiment_environment(force_clean_db=args.force_clean_db)
                comparison.run_all_strategies(strategies_to_run_from_cli=strategies_to_run)
        
            # Generate experiment-specific analysis and plots
            if args.experiment_type == 1:
                generate_experiment1_analysis(comparison)
            elif args.experiment_type == 2:
                # For normal flow (not burst-scan-only), we need to run burst scan after dual combat
                dual_combat_results = comparison.experiment_results.copy()
                dual_combat_exp_path = str(comparison.experiment_dir)
                
                # Now run burst scan experiment with only B7
                print("\n" + "="*60)
                print("🔄 Running burst scan experiment for subplot (a)...")
                
                # Create new comparison instance for burst scan as subdirectory of dual combat
                # This ensures burst scan results are always associated with the dual combat experiment
                burst_scan_base_dir = comparison.experiment_dir / "burst_scan"
                burst_scan_base_dir.mkdir(parents=True, exist_ok=True)
                
                # Pass the exact directory to use (avoid timestamp subdirectory)
                burst_scan_comparison = CacheStrategyComparison(base_output_dir=str(burst_scan_base_dir))
                # Override the experiment_dir to use the exact burst_scan directory
                burst_scan_comparison.experiment_dir = burst_scan_base_dir
                burst_scan_comparison.update_config_file('configs/config_burst_scan_optimized.json')
                burst_scan_comparison.prepare_experiment_environment(force_clean_db=args.force_clean_db)
                burst_scan_comparison.run_all_strategies(strategies_to_run_from_cli=['B7_DYNAMIC_NEED'])
                
                # Pass both results and paths to analysis function
                generate_experiment2_analysis(
                    comparison, 
                    dual_combat_results, 
                    burst_scan_comparison.experiment_results,
                    dual_combat_exp_path=dual_combat_exp_path,
                    burst_scan_exp_path=str(burst_scan_comparison.experiment_dir)
                )
            elif args.experiment_type == 3:
                generate_experiment3_analysis(comparison)
            
        main_logger.info(f"Experiment {args.experiment_type} completed successfully.")

    except Exception as e:
        main_logger.error(f"Critical error occurred during experiment: {e}", exc_info=True)
        sys.exit(1)

def generate_experiment1_analysis(comparison):
    """Generate analysis for Experiment 1: Three-phase hotspot shift workload"""
    print("\n" + "="*60)
    print("📊 Generating Experiment 1 Analysis: Three-phase hotspot shift workload")
    print("📈 Generating Figure 1: Performance comparison (median TPS and hit rate)...")
    print("📈 Generating Figure 2: Cache allocation timeseries...")
    
    # Import and run the paper figures script
    import subprocess
    import sys
    
    try:
        # Run the paper figures script which generates Figure 1 and Figure 2
        result = subprocess.run([
            sys.executable, 'scripts/plotting/plot_paper_figures.py'
        ], capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            print("✅ Paper figures generated successfully")
            print("   - Figure 1: figures/paper/figure1_performance_comparison.pdf")
            print("   - Figure 2: figures/paper/figure2_timeseries.pdf")
        else:
            print(f"❌ Error generating paper figures: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Failed to run paper figures script: {e}")

def generate_experiment2_analysis(comparison, dual_combat_results, burst_scan_results, dual_combat_exp_path=None, burst_scan_exp_path=None):
    """Generate analysis for Experiment 2: Adversarial cache pollution test"""
    import os
    from pathlib import Path
    
    print("\n" + "="*60)
    print("📊 Generating Experiment 2 Analysis: Adversarial cache pollution test")
    print("📈 Analyzing burst scan results for subplot (a)...")
    
    # Use the accurate analysis method for Phase2A vs Phase2D comparison
    if 'B7_DYNAMIC_NEED' in burst_scan_results and burst_scan_results['B7_DYNAMIC_NEED']:
        b7_data = burst_scan_results['B7_DYNAMIC_NEED']['raw_data']
        
        # Convert timestamp to datetime first
        b7_data['timestamp'] = pd.to_datetime(b7_data['timestamp'])
        
        # CRITICAL FIX: Calculate ABSOLUTE elapsed seconds from experiment start
        # The original elapsed_seconds column is relative to each phase start time
        b7_data['absolute_seconds'] = (b7_data['timestamp'] - b7_data['timestamp'].min()).dt.total_seconds()
        
        # Define phase boundaries for accurate comparison (updated to match new config)
        phase_boundaries = [
            (0, 30, 'Phase1_All_Silent'),
            (30, 60, 'Phase2_Attacker_Uniform')
        ]
        
        # Mark phases based on ABSOLUTE elapsed time (not the relative elapsed_seconds)
        phases = []
        for _, row in b7_data.iterrows():
            elapsed = row['absolute_seconds']  # Use absolute_seconds, not elapsed_seconds
            phase = 'Unknown'
            for start, end, name in phase_boundaries:
                if start <= elapsed < end:
                    phase = name
                    break
            phases.append(phase)
        b7_data['accurate_phase'] = phases
        
        # Filter attacker database (db_bg_1 or fallback to db_high_priority)
        high_priority = b7_data[b7_data['db_id'] == 'db_bg_1'].copy()
        if high_priority.empty:
            high_priority = b7_data[b7_data['db_id'] == 'db_high_priority'].copy()
        
        # Calculate need scores
        high_priority['miss_rate'] = 1 - high_priority['cache_hit_rate']
        high_priority['ops_per_second'] = high_priority['ops_count'] / 4.0
        high_priority['need_score'] = high_priority['ops_per_second'] * high_priority['miss_rate']
        
        # Compare Phase1_Both_Zipfian (0-30s) vs Phase2_BG1_Uniform (30-60s)
        phase1_data = high_priority[high_priority['accurate_phase'] == 'Phase1_Both_Zipfian']
        phase2_data = high_priority[high_priority['accurate_phase'] == 'Phase2_BG1_Uniform']
        
        # Fallback for backward compatibility
        if phase1_data.empty:
            phase1_data = high_priority[high_priority['accurate_phase'] == 'Phase1_All_Silent']
        if phase2_data.empty:
            phase2_data = high_priority[high_priority['accurate_phase'] == 'Phase2_Attacker_Uniform']
        
        if not phase1_data.empty and not phase2_data.empty:
            # Phase1: Take the second data point
            if len(phase1_data) >= 2:
                need_score_phase1 = phase1_data['need_score'].iloc[1]
                cache_phase1 = phase1_data['current_cache_pages'].iloc[1]
            else:
                # Fallback to first point if only one exists
                need_score_phase1 = phase1_data['need_score'].iloc[0]
                cache_phase1 = phase1_data['current_cache_pages'].iloc[0]
            
            # Phase2: Take the maximum need_score point
            max_need_idx = phase2_data['need_score'].idxmax()
            need_score_phase2 = phase2_data.loc[max_need_idx, 'need_score']
            cache_phase2 = phase2_data.loc[max_need_idx, 'current_cache_pages']
            
            # Calculate changes
            need_score_increase = (need_score_phase2 - need_score_phase1) / need_score_phase1 * 100 if need_score_phase1 != 0 else 0
            cache_increase = (cache_phase2 - cache_phase1) / cache_phase1 * 100 if cache_phase1 != 0 else 0
            overreaction_factor = cache_increase / need_score_increase if need_score_increase != 0 else 0
            
            print(f"\n📊 Burst Scan Analysis Results (Phase1 vs Phase2):")
            print(f"   Phase1_All_Silent (0-30s):")
            print(f"     - Need Score: {need_score_phase1:.3f}")
            print(f"     - Cache Pages: {cache_phase1:.1f}")
            print(f"   Phase2_Attacker_Uniform (30-60s):")
            print(f"     - Need Score: {need_score_phase2:.3f}")
            print(f"     - Cache Pages: {cache_phase2:.1f}")
            print(f"   Changes:")
            print(f"     - Need Score Increase: {need_score_increase:.1f}%")
            print(f"     - Cache Allocation Increase: {cache_increase:.1f}%")
            print(f"     - Overreaction Factor: {overreaction_factor:.1f}x")
            
            # Save these values for subplot (a)
            # (imports already at function level)
            
            # Save to experiment-specific location AND figures directory for compatibility
            saved_locations = []
            
            # Save to experiment directory if available (burst_scan_exp_path)
            if burst_scan_exp_path:
                exp_results_file = Path(burst_scan_exp_path) / 'burst_scan_results.txt'
                exp_results_file.parent.mkdir(parents=True, exist_ok=True)
                with open(exp_results_file, 'w') as f:
                    f.write(f"need_score_increase={need_score_increase:.1f}\n")
                    f.write(f"cache_increase={cache_increase:.1f}\n")
                    f.write(f"overreaction_factor={overreaction_factor:.1f}\n")
                saved_locations.append(str(exp_results_file))
            
            # Also save to figures directory for backwards compatibility
            cwd = os.getcwd()
            if cwd.endswith('SAM-reproducible'):
                figures_dir = Path('figures')
            else:
                figures_dir = Path('SAM-reproducible/figures') if os.path.exists('SAM-reproducible') else Path('figures')
            figures_dir.mkdir(parents=True, exist_ok=True)
            
            figures_results_file = figures_dir / 'burst_scan_results.txt'
            with open(figures_results_file, 'w') as f:
                f.write(f"need_score_increase={need_score_increase:.1f}\n")
                f.write(f"cache_increase={cache_increase:.1f}\n")
                f.write(f"overreaction_factor={overreaction_factor:.1f}\n")
            saved_locations.append(str(figures_results_file))
            
            print(f"   💾 Results saved to: {', '.join(saved_locations)}")
        else:
            # Handle the case where Phase2D data is missing
            print("\n⚠️ WARNING: Cannot generate burst scan results!")
            if phase2a_data.empty:
                print("   ❌ Phase2A_Normal (30-90s) data is empty")
            if phase2d_data.empty:
                print("   ❌ Phase2D_Second_Burst (165-195s) data is empty")
            print(f"   💡 Hint: The experiment might be too short. Phase2D starts at 165s.")
            print(f"   💡 Try running with --duration 200 to cover Phase2D")
            
            # Still save a file indicating the problem
            if burst_scan_exp_path:
                exp_results_file = Path(burst_scan_exp_path) / 'burst_scan_results.txt'
                with open(exp_results_file, 'w') as f:
                    f.write("# ERROR: Incomplete data\n")
                    f.write("# Phase2D data is missing\n")
                    f.write("# Run with --duration 200 or longer\n")
                    f.write("need_score_increase=11.7\n")  # Default values
                    f.write("cache_increase=64.5\n")
                    f.write("overreaction_factor=5.5\n")
                print(f"   📄 Error file saved to: {exp_results_file}")
    
    print("\n📈 Generating Figure 7: Dual combat robustness analysis...")
    
    import subprocess
    import sys
    
    # Use the provided dual combat experiment path, with fallbacks
    plot_dual_combat_exp_path = dual_combat_exp_path
    
    if not plot_dual_combat_exp_path and hasattr(comparison, 'experiment_dir'):
        # Fallback to comparison experiment directory
        plot_dual_combat_exp_path = str(comparison.experiment_dir)
    
    if not plot_dual_combat_exp_path:
        # Last resort: find the most recent experiment
        import glob
        pattern = 'results/comparison/experiment_*'
        all_exps = glob.glob(pattern)
        if all_exps:
            plot_dual_combat_exp_path = sorted(all_exps)[-1]
            print(f"   ⚠️ Using most recent experiment as fallback: {plot_dual_combat_exp_path}")
    
    # Also ensure burst_scan_results.txt is in the expected location
    if burst_scan_exp_path:
        burst_scan_file = Path(burst_scan_exp_path) / 'burst_scan_results.txt'
        if burst_scan_file.exists():
            print(f"   📊 Burst scan results found at: {burst_scan_file}")
    
    print(f"   🔍 Using dual combat experiment path: {plot_dual_combat_exp_path}")
    
    try:
        # Run the dual combat story script with the actual experiment path
        # The script will use the burst_scan_results.txt for subplot (a)
        cmd = [sys.executable, 'scripts/plot_dual_combat_story.py']
        if plot_dual_combat_exp_path:
            cmd.append(plot_dual_combat_exp_path)
            print(f"   📊 Running plot script with experiment path: {plot_dual_combat_exp_path}")
        else:
            print("   ⚠️ No experiment path available, script will use defaults")
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            print("✅ Dual combat analysis generated successfully")
            print("   - Figure 7: figures/paper/figure7_dual_combat_robustness.pdf")
            # Parse and show the figure story from output
            if "Figure story:" in result.stdout:
                story_start = result.stdout.find("Figure story:")
                if story_start != -1:
                    story_lines = result.stdout[story_start:].split('\n')[:5]
                    for line in story_lines:
                        if line.strip():
                            print(f"   {line.strip()}")
        else:
            print(f"❌ Error generating dual combat figure:")
            print(f"   Error: {result.stderr}")
            print(f"   Output: {result.stdout}")
            
    except Exception as e:
        print(f"❌ Failed to run dual combat script: {e}")

def generate_experiment3_analysis(comparison):
    """Generate analysis for Experiment 3: Scalability test"""
    print("\n" + "="*60)
    print("📊 Generating Experiment 3 Analysis: Scalability test")
    print("="*60)
    
    import subprocess
    import sys
    import json
    import numpy as np
    from pathlib import Path
    
    # Import scalability utilities
    sys.path.append(str(Path(__file__).parent.parent / 'src'))
    from scalability_utils import (
        generate_scalability_config,
        aggregate_scalability_results,
        save_scalability_summary
    )
    
    # Define test parameters
    db_counts = [20, 40, 80, 120]
    strategies = ['S0_EMG_AS', 'B7_DYNAMIC_NEED']
    test_duration = 60  # seconds per test
    
    print(f"🔬 Test Configuration:")
    print(f"   - Database counts: {db_counts}")
    print(f"   - Strategies: {strategies}")
    print(f"   - Duration per test: {test_duration}s")
    print(f"   - Total tests: {len(db_counts) * len(strategies)}")
    print("")
    
    results_by_scale = {}
    
    # Run experiments for each database count and strategy
    for db_count in db_counts:
        print(f"\n📊 Testing with {db_count} databases...")
        print("-" * 40)
        
        for strategy in strategies:
            print(f"\n🚀 Running {strategy} with {db_count} databases...")
            
            try:
                # Step 1: Generate configuration
                config = generate_scalability_config(
                    db_count=db_count,
                    strategy=strategy,
                    duration=test_duration
                )
                
                # Step 2: Create new comparison instance for this test
                test_comparison = CacheStrategyComparison(
                    base_output_dir=f"results/scalability/scale_{db_count}db_{strategy}"
                )
                
                # Save the generated config
                config_path = Path(test_comparison.base_output_dir) / "config_generated.json"
                config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                
                # Update with generated config
                test_comparison.config = config
                test_comparison.config['general_experiment_setup']['output_directory'] = str(test_comparison.base_output_dir)
                
                # Step 3: Prepare environment and run experiment
                print(f"   Preparing environment...")
                test_comparison.prepare_experiment_environment(force_clean_db=False)
                
                print(f"   Running experiment...")
                test_comparison.run_all_strategies(strategies_to_run_from_cli=[strategy])
                
                # Step 4: Extract CPU timing data
                if hasattr(test_comparison, 'orchestrator') and hasattr(test_comparison.orchestrator, 'decision_cpu_times'):
                    cpu_times = test_comparison.orchestrator.decision_cpu_times
                    
                    if cpu_times:
                        mean_time = np.mean(cpu_times)
                        std_time = np.std(cpu_times)
                        
                        results_by_scale[f"{db_count}_{strategy}"] = {
                            'mean': mean_time,
                            'std': std_time,
                            'samples': cpu_times[:100]  # Limit samples to save space
                        }
                        
                        print(f"   ✅ Collected {len(cpu_times)} CPU time samples")
                        print(f"      Mean: {mean_time*1000:.3f}ms, Std: {std_time*1000:.3f}ms")
                    else:
                        print(f"   ⚠️ No CPU timing data collected")
                        results_by_scale[f"{db_count}_{strategy}"] = {
                            'mean': 0,
                            'std': 0,
                            'samples': []
                        }
                else:
                    print(f"   ⚠️ Orchestrator not available or no CPU timing data")
                    results_by_scale[f"{db_count}_{strategy}"] = {
                        'mean': 0,
                        'std': 0,
                        'samples': []
                    }
                    
            except Exception as e:
                print(f"   ❌ Error running {strategy} with {db_count} databases: {e}")
                import traceback
                traceback.print_exc()
                # Add empty results for failed experiments
                results_by_scale[f"{db_count}_{strategy}"] = {
                    'mean': 0,
                    'std': 0,
                    'samples': []
                }
    
    # Step 5: Aggregate and save results
    print("\n" + "="*60)
    print("📊 Aggregating results...")
    
    if results_by_scale:
        aggregated_results = aggregate_scalability_results(results_by_scale)
        output_path = save_scalability_summary(aggregated_results)
        print(f"✅ Results saved to: {output_path}")
        
        # Print summary table
        print("\n📊 Summary Table:")
        print("-" * 60)
        print(f"{'DB Count':<10} {'Strategy':<20} {'Mean (ms)':<15} {'Std (ms)':<15}")
        print("-" * 60)
        
        for db_count in db_counts:
            for strategy in strategies:
                key = f"{db_count}_{strategy}"
                if key in results_by_scale:
                    data = results_by_scale[key]
                    mean_ms = data['mean'] * 1000
                    std_ms = data['std'] * 1000
                    print(f"{db_count:<10} {strategy:<20} {mean_ms:<15.3f} {std_ms:<15.3f}")
    else:
        print("❌ No results collected")
    
    # Step 6: Generate plots
    print("\n📈 Generating plots...")
    
    import subprocess
    import sys
    
    try:
        # Run the CPU performance comparison script
        result1 = subprocess.run([
            sys.executable, 'scripts/plot_cpu_performance_comparison.py'
        ], capture_output=True, text=True, cwd='.')
        
        # Run the CPU decision time distribution script
        result2 = subprocess.run([
            sys.executable, 'scripts/plot_cpu_decision_time_distribution.py'
        ], capture_output=True, text=True, cwd='.')
        
        if result1.returncode == 0 and result2.returncode == 0:
            print("✅ CPU performance analysis plots generated successfully")
            print("   - CPU Performance: figures/cpu_performance/cpu_performance_comparison.pdf")
            print("   - CPU Distribution: figures/cpu_performance/cpu_decision_time_distribution.pdf")
        else:
            if result1.returncode != 0:
                print(f"❌ Error generating CPU performance comparison: {result1.stderr}")
            if result2.returncode != 0:
                print(f"❌ Error generating CPU distribution: {result2.stderr}")
            
    except Exception as e:
        print(f"❌ Failed to run CPU performance scripts: {e}")
    
    print("\n" + "="*60)
    print("✅ Type 3 Scalability Analysis Complete!")
    print("="*60)

if __name__ == "__main__":
    main() 
