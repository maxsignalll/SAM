import apsw
import threading
import time
import os
import queue
from ycsb_utils import YCSBWorkload, load_initial_data, create_ycsb_table
from tps_controller import PreciseTpsController
from database_manager import check_and_reuse_database
from database_utils import *
import logging
import numpy as np
from perf_counter import PerfCounter

# Import workload factory (backward compatible)
try:
    from workload.workload_factory import create_workload, create_initial_data as create_workload_data
except ImportError:
    # If import fails, set flag to use original implementation
    create_workload = None
    create_workload_data = None


class BGDatabaseLogFilter(logging.Filter):
    """
    Custom log filter to filter out database logs containing 'bg'
    """
    def filter(self, record):
        # If log record's name contains 'bg', don't display it
        if 'bg' in record.name.lower():
            return False
        # If log message contains 'DB[db_bg', don't display it
        if hasattr(record, 'msg') and 'DB[db_bg' in str(record.msg):
            return False
        return True


class DatabaseInstance(threading.Thread):
    """
    A queue-driven, independent database worker thread.
    Receives commands through command_queue, reports status through stats_queue.
    """
    
    def __init__(self, db_id: str, db_config: dict, 
                 initial_cache_pages: int, 
                 command_queue: queue.Queue,
                 stats_queue: queue.Queue,
                 orchestrator_config: dict,
                 strategy_id: str
                 ):
        super().__init__()
        self.db_id = db_id
        self.db_config = db_config
        self.initial_cache_pages = initial_cache_pages
        self.command_queue = command_queue
        self.stats_queue = stats_queue
        self.orchestrator_config = orchestrator_config
        self.strategy_id = strategy_id

        self.db_filename = self.db_config["db_filename"]
        
        # Internal state
        self._is_stopped = False
        self._is_paused = True # Default to paused at startup, waiting for 'START' command
        self.is_warming_up = False
        self.warmup_end_time = 0
        self.benchmark_running = False # Core fix: initialize benchmark_running state
        
        # Logger
        self.logger = logging.getLogger(f"DBInstance.{self.db_id}")
        if not self.logger.handlers:
            log_level_str = self.orchestrator_config.get("log_level", "INFO").upper()
            self.logger.setLevel(getattr(logging, log_level_str, logging.DEBUG))
            ch = logging.StreamHandler()
            formatter = logging.Formatter(f'%(asctime)s - %(name)s - %(levelname)s - DB[{self.db_id}] - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            self.logger.propagate = False

        self.conn = None
        self.db_lock = threading.Lock()
        self.workload_generator = None
        self.tps_controller = None

        # Statistics related
        self.operation_count = 0
        self.phase_start_time = 0
        self.phase_name = "init"
        self._last_stats_update_time = 0
        
        # Index management related
        self.index_enabled = True  # Index is enabled by default
        self.phase_should_have_index = True  # Flag indicating current phase should have index

    @classmethod
    def create_and_load_data(cls, db_config: dict, logger: logging.Logger, orchestrator_config: dict = None):
        """
        An independent class method to create and load databases based on configuration.
        This allows Orchestrator to prepare databases before starting worker threads.
        """
        db_id = db_config.get('id', 'UnknownDB')
        logger.info(f"[{db_id}] Starting on-demand database creation and data loading...")
        
        db_filename = db_config.get('db_filename')
        if not db_filename:
            logger.error(f"[{db_id}] Missing 'db_filename' in config, cannot create.")
            return

        db_dir = os.path.dirname(db_filename)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            
        temp_conn = None
        try:
            # Use exclusive connection for fast loading
            temp_conn = safe_connect_database_with_retry(db_filename)
            
            # Get workload type
            workload_type = 'ycsb'  # Default value
            if orchestrator_config:
                workload_type = orchestrator_config.get('general_experiment_setup', {}).get('workload_type', 'ycsb')
            
            # Check if B5 multi-table mode
            if db_config.get('b5_multi_table', False):
                # B5 multi-table mode - use local path for generation
                local_db_filename = db_config.get('local_db_filename')
                if local_db_filename:
                    logger.info(f"[{db_id}] 🎯 B5 multi-table mode: using local cache mechanism")
                    logger.info(f"[{db_id}] 📁 Local path: {local_db_filename}")
                    logger.info(f"[{db_id}] 🌐 Remote path: {db_filename}")
                    
                    # Ensure local directory exists
                    local_db_dir = os.path.dirname(local_db_filename)
                    if local_db_dir:
                        os.makedirs(local_db_dir, exist_ok=True)
                    
                    # If local database exists and is valid, copy directly to remote
                    if os.path.exists(local_db_filename):
                        logger.info(f"[{db_id}] Found existing local database, checking validity...")
                        try:
                            # Simple check if database is valid
                            local_conn = safe_connect_database_with_retry(local_db_filename)
                            cursor = local_conn.cursor()
                            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
                            if cursor.fetchone():
                                logger.info(f"[{db_id}] Local database valid, copying to remote...")
                                local_conn.close()
                                temp_conn.close()
                                
                                # Copy to remote (including all related files)
                                from database_utils_extended import copy_sqlite_with_related_files
                                copy_sqlite_with_related_files(local_db_filename, db_filename)
                                logger.info(f"[{db_id}] Successfully copied local database to remote: {db_filename}")
                                return
                            local_conn.close()
                        except Exception as e:
                            logger.warning(f"[{db_id}] Local database invalid: {e}, will regenerate")
                    
                    # Close remote connection, generate locally
                    temp_conn.close()
                    temp_conn = safe_connect_database_with_retry(local_db_filename)
                    logger.info(f"[{db_id}] Creating B5 multi-table database at local path...")
                
                # B5 multi-table data generation
                from b5_multi_table_loader import B5MultiTableLoader
                
                original_dbs = db_config.get('b5_original_dbs', [])
                if workload_type == 'ycsb':
                    # Create multiple YCSB tables
                    ycsb_config = orchestrator_config.get('ycsb_general_config', {})
                    B5MultiTableLoader.create_ycsb_multi_tables(temp_conn, original_dbs, ycsb_config)
                elif workload_type == 'tpcc':
                    # Create multiple TPC-C table groups
                    tpcc_config = orchestrator_config.get('tpcc_general_config', {})
                    B5MultiTableLoader.create_tpcc_multi_tables(temp_conn, original_dbs, tpcc_config)
                else:
                    logger.error(f"[{db_id}] B5 multi-table mode does not support workload type: {workload_type}")
                
                # If generated locally, copy to remote
                if local_db_filename and local_db_filename != db_filename:
                    # Ensure database connection is properly closed
                    from database_utils_extended import ensure_sqlite_closed_properly, copy_sqlite_with_related_files
                    ensure_sqlite_closed_properly(temp_conn)
                    
                    logger.info(f"[{db_id}] Copying local database to remote: {local_db_filename} -> {db_filename}")
                    copy_sqlite_with_related_files(local_db_filename, db_filename)
                    logger.info(f"[{db_id}] B5 database and related files copy completed")
                else:
                    # If generated directly at target location, ensure connection closed
                    if temp_conn:
                        temp_conn.close()
                
                # B5 multi-table database created successfully
                logger.info(f"[{db_id}] B5 database creation completed")
                    
            # Create data based on workload type
            elif workload_type == 'ycsb':
                # YCSB workload - keep original logic
                if 'ycsb_row_size_bytes' in db_config:
                    # New mode: based on row size
                    create_ycsb_table(temp_conn, row_size_bytes=db_config['ycsb_row_size_bytes'])
                    load_initial_data(
                        temp_conn,
                        db_config['ycsb_initial_record_count'],
                        db_id=db_id,
                        row_size_bytes=db_config['ycsb_row_size_bytes']
                    )
                else:
                    # Old mode: based on field count
                    create_ycsb_table(temp_conn, field_count=db_config.get('ycsb_field_count', 4))
                    load_initial_data(
                        temp_conn,
                        db_config['ycsb_initial_record_count'],
                        field_count=db_config.get('ycsb_field_count', 4),
                        field_length=db_config.get('ycsb_field_length', 1000),
                        db_id=db_id
                    )
                    
            elif create_workload_data is not None:
                # Use factory pattern to create other types of workload data
                if workload_type == 'tpcc':
                    record_count = db_config.get('tpcc_warehouses', 1)
                elif workload_type == 'trace':
                    record_count = 0  # Trace doesn't need to generate data, load from file
                else:
                    record_count = db_config.get('ycsb_initial_record_count', 10000)
                    
                create_workload_data(temp_conn, db_id, orchestrator_config, record_count, workload_type)
                
            else:
                # Fallback to YCSB
                logger.warning(f"[{db_id}] Unable to create {workload_type} type data, falling back to YCSB")
                create_ycsb_table(temp_conn, row_size_bytes=db_config.get('ycsb_row_size_bytes', 2048))
                load_initial_data(
                    temp_conn,
                    db_config.get('ycsb_initial_record_count', 10000),
                    db_id=db_id,
                    row_size_bytes=db_config.get('ycsb_row_size_bytes', 2048)
                )
            
            # Data loading successful
            logger.info(f"[{db_id}] Data loading successful.")
            
        except Exception as e:
            logger.error(f"[{db_id}] Severe error occurred while creating and loading data: {e}", exc_info=True)
            raise
        finally:
            if temp_conn:
                temp_conn.close()

    def run(self):
        """Thread main loop, process commands and execute workload."""
        self.logger.info(f"Worker thread started, PID: {os.getpid()}")
        try:
            self._setup_and_initialize()
            self._main_loop()
        except Exception as e:
            self.logger.error(f"Uncaught exception during runtime: {e}", exc_info=True)
            self.stats_queue.put({'type': 'FATAL', 'db_id': self.db_id, 'error': str(e)})
        finally:
            # Restore index to default state before closing connection
            if self.conn and not self.index_enabled:
                self.logger.info("Experiment ended, restoring priority_value index")
                self._create_priority_index()
                self.index_enabled = True
            
            if self.conn:
                self.conn.close()
            self.logger.info("Worker thread terminated.")
            self.stats_queue.put({'type': 'STOPPED', 'db_id': self.db_id})

    def _setup_and_initialize(self):
        """Establish connection, apply settings, and initialize workload."""
        self.conn = safe_connect_database_with_retry(self.db_config["db_filename"])
        self.logger.info("Database connection successful.")
        self._apply_initial_settings()
        self._initialize_workload()
        self.stats_queue.put({'type': 'READY', 'db_id': self.db_id})
        self.logger.info("Setup and initialization complete, READY signal sent.")

    def _get_actual_record_count(self) -> int:
        """Get actual record count of database, preferring cached metadata."""
        # Get workload type
        workload_type = self.orchestrator_config.get('general_experiment_setup', {}).get('workload_type', 'ycsb')
        
        # First try to get from metadata cache
        try:
            from database_manager import db_manager
            cached_metadata = db_manager.load_database_metadata(self.db_config["db_filename"])
            if cached_metadata and 'record_count' in cached_metadata:
                cached_count = cached_metadata['record_count']
                
                # Get expected record count based on workload type
                if workload_type == 'ycsb':
                    expected_count = self.db_config.get('ycsb_initial_record_count', 0)
                elif workload_type == 'tpcc':
                    expected_count = self.db_config.get('tpcc_warehouses', 1)
                elif workload_type == 'trace':
                    expected_count = 0  # Trace loads from file, no fixed record count
                else:
                    expected_count = self.db_config.get('ycsb_initial_record_count', 0)
                    
                # Verify if cached record count is within reasonable range
                tolerance = max(1, int(expected_count * 0.20))
                if abs(cached_count - expected_count) <= tolerance:
                    self.logger.info(f"Using cached record count: {cached_count}")
                    return cached_count
        except Exception as e:
            self.logger.debug(f"Unable to get record count from cache: {e}")
        
        # If cache is unavailable or doesn't match, query the database
        try:
            with self.db_lock:
                # Query different tables based on workload type
                if workload_type == 'ycsb':
                    # Check if it's B5 multi-table mode
                    if self.db_config.get('b5_multi_table', False):
                        # B5 multi-table mode: count total records from all ycsb_data_* tables
                        cursor = self.conn.execute("""
                            SELECT SUM(cnt) FROM (
                                SELECT COUNT(*) as cnt 
                                FROM sqlite_master 
                                WHERE type='table' AND name LIKE 'ycsb_data_%'
                            )
                        """)
                        table_count = cursor.fetchone()[0]
                        if table_count and table_count > 0:
                            # Get total record count from all ycsb_data_* tables
                            count = 0
                            cursor = self.conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'ycsb_data_%'")
                            tables = cursor.fetchall()
                            for (table_name,) in tables:
                                cursor = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                                count += cursor.fetchone()[0]
                        else:
                            count = 0
                    else:
                        # Standard single table mode
                        cursor = self.conn.execute("SELECT COUNT(*) FROM usertable")
                        count = cursor.fetchone()[0]
                elif workload_type == 'tpcc':
                    # TPC-C's "record count" can be warehouse count
                    cursor = self.conn.execute("SELECT COUNT(*) FROM warehouse")
                    count = cursor.fetchone()[0]
                elif workload_type == 'trace':
                    # Trace's record count is block count
                    cursor = self.conn.execute("SELECT COUNT(*) FROM block_trace")
                    count = cursor.fetchone()[0]
                else:
                    # For unknown workload types, try multiple possible tables
                    count = 0
                    for table_name in ['usertable', 'warehouse', 'block_trace']:
                        try:
                            cursor = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                            count = cursor.fetchone()[0]
                            if count > 0:
                                break
                        except:
                            continue
                    
                self.logger.info(f"Found actual record count in database: {count}")
                return count
        except Exception as e:
            self.logger.error(f"Error querying actual record count: {e}. Will fallback to using config value.")
            if workload_type == 'ycsb':
                return self.db_config.get('ycsb_initial_record_count', 0)
            elif workload_type == 'tpcc':
                return self.db_config.get('tpcc_warehouses', 1)
            else:
                return 0

    def _initialize_workload(self):
        """Initialize workload and TPS controller."""
        
        # Get workload type
        workload_type = self.orchestrator_config.get('general_experiment_setup', {}).get('workload_type', 'ycsb')
        self.logger.info(f"workload_type from orchestrator_config: {workload_type}")
        
        # Core fix: use real record count from database, not outdated value from config file
        self.actual_record_count = self._get_actual_record_count()
        
        # Perform specific initialization based on workload type
        if workload_type == 'ycsb':
            if self.actual_record_count != self.db_config.get('ycsb_initial_record_count'):
                self.logger.warning(
                    f"Actual record count in database ({self.actual_record_count}) doesn't match "
                    f"config value ({self.db_config.get('ycsb_initial_record_count')})."
                    "Will prioritize using actual count from database."
                )
            
            # Add debug info for hot spot size and theoretical hit rate
            hot_rate = self.orchestrator_config.get('ycsb_general_config', {}).get('hot_rate', 0.167)
            hot_spot_size = int(self.actual_record_count * hot_rate)
            row_size_bytes = self.orchestrator_config.get('ycsb_general_config', {}).get('ycsb_row_size_bytes', 2048)
            effective_row_size = row_size_bytes + 100  # SQLite overhead estimate
            
            # Calculate how many records current cache can hold
            # Ensure initial_cache_pages is not None, set to default 1 if it is
            cache_pages = self.initial_cache_pages if self.initial_cache_pages is not None else 1
            current_cache_bytes = cache_pages * self.page_size_bytes
            cache_record_capacity = int(current_cache_bytes / effective_row_size)
            
            # Calculate theoretical hit rate
            theoretical_hit_rate = min(1.0, cache_record_capacity / hot_spot_size) if hot_spot_size > 0 else 0.0
            
            self.logger.info(
                f"Hot spot analysis - Total records: {self.actual_record_count}, "
                f"Hot rate: {hot_rate:.1%}, Hot spot size: {hot_spot_size} records, "
                f"Cache capacity: {cache_pages} pages ({current_cache_bytes/1024/1024:.1f}MB), "
                f"Cacheable records: {cache_record_capacity}, "
                f"Theoretical hit rate upper limit: {theoretical_hit_rate:.1%}"
            )
        elif workload_type == 'tpcc':
            self.logger.info(f"TPC-C workload - Warehouse count: {self.actual_record_count}")
        elif workload_type == 'trace':
            self.logger.info(f"Trace workload - Block count: {self.actual_record_count}")

        # Use factory pattern to create workload (if available), otherwise use original method
        if create_workload is not None:
            # Use new factory pattern
            self.workload_generator = create_workload(
                conn=self.conn,
                conn_lock=threading.Lock(),
                db_id=self.db_id,
                db_config=self.db_config,
                orchestrator_config=self.orchestrator_config,
                initial_record_count=self.actual_record_count,
                worker_seed_offset=hash(self.db_id),
                strategy_id=self.strategy_id,
                workload_type=workload_type
            )
        else:
            # Backward compatibility: use original method
            self.workload_generator = YCSBWorkload(
                conn=self.conn,
                conn_lock=threading.Lock(),
                db_id=self.db_id,
                ycsb_config=self.db_config,
                general_db_config=self.orchestrator_config,
                initial_record_count=self.actual_record_count, # Use real value
                worker_seed_offset=hash(self.db_id),
                strategy_id=self.strategy_id
            )
        
        self.tps_controller = PreciseTpsController(
            self.db_config.get("target_tps", 1000),
            self.db_id
        )
        self.logger.info("Workload and TPS controller initialized.")

    def _process_command_queue(self):
        """Process commands from Orchestrator."""
        try:
            # Use non-blocking get_nowait to quickly check the queue
            command = self.command_queue.get_nowait()
            command_type = command.get('type')
            self.logger.debug(f"Received command: {command}")

            if command_type == 'START_BENCHMARK':
                self.benchmark_running = True
                self.workload_generator.reset_stats()
            elif command_type == 'STOP_BENCHMARK':
                self.benchmark_running = False
            elif command_type == 'STOP':
                self.benchmark_running = False
                self._is_stopped = True
                self.logger.info(f"Received stop signal")
            elif command_type == 'RECONFIGURE':
                self._handle_reconfigure(command.get('new_config', {}))
            elif command_type == 'RECONFIGURE_CACHE':
                new_page_count = command.get('page_count')
                if new_page_count is not None:
                    self.reconfigure_cache_size(new_page_count)
            elif command_type == 'REPORT_STATS':
                self._update_stats_queue()
            else:
                self.logger.warning(f"Received unknown command: {command}")

        except queue.Empty:
            # Empty queue is normal, no action needed
            pass
        except Exception as e:
            self.logger.error(f"Error processing command: {e}", exc_info=True)

    def _handle_start_warmup(self, command: dict):
        """Handle start warmup command"""
        duration = command.get('duration', 10)
        self.logger.info(f"Received START_WARMUP command, starting warmup for {duration} seconds.")
        self._reset_stats()
        self.is_warming_up = True
        self.warmup_end_time = time.monotonic() + duration
        self._is_paused = False
        self.phase_name = command.get('phase_name', 'benchmark')

    def _handle_start_benchmark(self, command: dict):
        """Handle start benchmark command"""
        self.logger.info("Received START_BENCHMARK command, starting formal measurement.")
        self._reset_stats()
        self.is_warming_up = False # Ensure not in warmup mode
        self._is_paused = False
        self.phase_name = command.get('phase_name', 'benchmark')

    def _handle_reconfigure(self, config: dict):
        """Handle reconfiguration command"""
        phase_name_from_config = config.get('phase_name', 'Unnamed Phase')
        self.logger.info(f"Starting reconfiguration: {phase_name_from_config}")
        
        # Get workload type
        workload_type = self.orchestrator_config.get('general_experiment_setup', {}).get('workload_type', 'ycsb')
        
        # Get config overrides based on workload type
        if workload_type == 'tpcc':
            overrides = config.get("tpcc_config_overrides") or {}
        elif workload_type == 'trace':
            overrides = config.get("trace_config_overrides") or {}
        else:
            overrides = config.get("ycsb_config_overrides") or {}
        
        # Handle index management (for ycsb workload only)
        if workload_type == 'ycsb' and overrides:
            access_pattern_per_db = overrides.get("access_pattern_per_db", {})
            if self.db_id in access_pattern_per_db:
                db_pattern = access_pattern_per_db[self.db_id]
                should_have_index = db_pattern.get("index", True)  # Default to True
                
                # Save index state to restore at phase end
                self.phase_should_have_index = should_have_index
                
                # Check actual index existence status
                table_name = "usertable"
                index_name = f"idx_{table_name}_priority_value"
                actual_index_exists = self._check_index_exists(index_name)
                
                # Decide action based on requirement and actual state
                if should_have_index and not actual_index_exists:
                    self.logger.info(f"Phase requires index but index doesn't exist, creating index")
                    self._create_priority_index()
                    self.index_enabled = True
                elif not should_have_index and actual_index_exists:
                    self.logger.info(f"Phase doesn't need index but index exists, dropping index")
                    self._drop_priority_index()
                    self.index_enabled = False
                else:
                    # Index state already meets requirements
                    self.index_enabled = should_have_index
                    if should_have_index:
                        self.logger.debug(f"Index exists and needs to be kept")
                    else:
                        self.logger.debug(f"Index doesn't exist and no need to create")

        # Centrally call reconfiguration logic
        self.reconfigure_workload(overrides)
        
        # Reset phase state
        self._reset_stats()
        self.phase_name = phase_name_from_config
        self.logger.info("Reconfiguration completed.")

    def _reset_stats(self):
        """Reset performance statistics."""
        if self.conn:
            try:
                # Core fix: directly call apsw interface to reset stats, not UDF
                self.conn.status(apsw.SQLITE_DBSTATUS_CACHE_HIT, True)
                self.conn.status(apsw.SQLITE_DBSTATUS_CACHE_MISS, True)
                self.conn.status(apsw.SQLITE_DBSTATUS_CACHE_WRITE, True)
                self.logger.info("SQLite internal cache statistics reset.")
            except Exception as e:
                self.logger.warning(f"Error directly resetting SQLite cache stats: {e}.")
        
        if self.tps_controller:
            self.tps_controller.reset()
        
        if self.workload_generator:
            self.workload_generator.reset_stats()
        
        self.operation_count = 0
        self.phase_start_time = PerfCounter.instance().perf_counter()
        self.logger.debug("Instance internal statistics and timers reset.")

    def _main_loop(self):
        """Main event loop, process commands and execute workload"""
        self.logger.info(f"DB[{self.db_id}] - Worker thread main loop started")
        while not self._is_stopped:
            try:
                # Process commands from Orchestrator first
                self._process_command_queue()

                if self.benchmark_running:
                    # If benchmark is running, execute YCSB operations
                    if self.tps_controller and self.tps_controller.should_wait():
                        time.sleep(0.001) # Brief sleep, wait for next time slice
                        continue

                    # Use db_lock to protect all database access
                    with self.db_lock:
                        try:
                            # Core fix: use correct method name run_operation()
                            latency_us = self.workload_generator.run_operation()
                            if self.tps_controller:
                                self.tps_controller.record_completion(latency_us)

                        except apsw.BusyError:
                            self.logger.warning("Database busy, skipping operation this cycle.")
                            time.sleep(0.05) # Should not sleep too long while holding lock, but handle brief busy states
                        except Exception as e:
                            self.logger.error(f"Unexpected error executing operation while holding lock: {e}", exc_info=True)
                            self._is_stopped = True
                            self.stats_queue.put({'type': 'FATAL', 'db_id': self.db_id, 'error': str(e)})
                else:
                    # If benchmark not running, sleep briefly to avoid spinning
                    time.sleep(0.01)

            except Exception as e:
                self.logger.error(f"Uncaught exception during runtime: {e}", exc_info=True)
                self.stats_queue.put({'type': 'FATAL', 'db_id': self.db_id, 'error': str(e)})

    def _apply_initial_settings(self):
        """Apply initial database connection settings, including cache size."""
        if not self.conn:
            self.logger.error("Database connection not initialized, cannot apply settings.")
            return
            
        system_config = self.orchestrator_config.get('system_sqlite_config', {})
        self.page_size_bytes = system_config.get('page_size_bytes', 4096)
        
        pragmas = [
            f"PRAGMA page_size = {self.page_size_bytes};",
            f"PRAGMA journal_mode = {system_config.get('journal_mode', 'WAL')};",
            f"PRAGMA synchronous = {system_config.get('synchronous', 'NORMAL')};",
            f"PRAGMA temp_store = {system_config.get('temp_store', 'MEMORY')};",
        ]
        
        with self.db_lock:
            for pragma in pragmas:
                try:
                    self.conn.execute(pragma)
                except apsw.SQLError as e:
                    self.logger.warning(f"Failed to execute PRAGMA '{pragma}': {e}")
        
        self.logger.info("Applied initial PRAGMA settings.")
        # Ensure initial_cache_pages is not None
        cache_pages = self.initial_cache_pages if self.initial_cache_pages is not None else 1
        self.reconfigure_cache_size(cache_pages)

    def reconfigure_workload(self, new_config: dict):
        """Update workload and TPS controller based on new configuration."""
        # Get workload type
        workload_type = self.orchestrator_config.get('general_experiment_setup', {}).get('workload_type', 'ycsb')
        
        # Update access pattern (only for supported workload types)
        if workload_type == 'trace':
            # For trace workload, call apply_config_overrides method
            if hasattr(self.workload_generator, 'apply_config_overrides'):
                self.workload_generator.apply_config_overrides(new_config)
                self.logger.info(f"Trace workload configuration updated")
        elif workload_type == 'ycsb':
            # Core fix: correctly parse DB-instance-specific access pattern config
            if "access_pattern_per_db" in new_config and self.db_id in new_config["access_pattern_per_db"]:
                db_specific_pattern = new_config["access_pattern_per_db"][self.db_id]
                # Core fix: pass current, authoritative record count
                if hasattr(self.workload_generator, 'update_access_pattern'):
                    self.workload_generator.update_access_pattern(db_specific_pattern, self.actual_record_count)
            # Compatible with B5's old config mode
            elif "access_pattern" in new_config:
                if hasattr(self.workload_generator, 'update_access_pattern'):
                    self.workload_generator.update_access_pattern(new_config["access_pattern"], self.actual_record_count)

            # Update YCSB operation proportions
            proportions_map = {
                "read_proportion": "set_read_proportion",
                "update_proportion": "set_update_proportion",
                "insert_proportion": "set_insert_proportion",
                "scan_proportion": "set_scan_proportion",
                "read_modify_write_proportion": "set_read_modify_write_proportion"
            }
            for key, method_name in proportions_map.items():
                if key in new_config:
                    value = new_config[key]
                    if hasattr(self.workload_generator, method_name):
                        getattr(self.workload_generator, method_name)(value)
                        self.logger.info(f"YCSB {key} updated to: {value}")
        
        # Core fix: intelligently update target TPS
        new_target_tps = None
        # Prioritize per-db distribution lookup
        if "tps_distribution_per_db" in new_config:
            dist_map = new_config["tps_distribution_per_db"]
            if self.db_id in dist_map:
                new_target_tps = dist_map[self.db_id]
                self.logger.info(f"Found specific TPS from tps_distribution_per_db: {new_target_tps}")
            else:
                self.logger.warning(f"Did not find own db_id ({self.db_id}) in tps_distribution_per_db.")

        # If not in per-db, fall back to global setting
        if new_target_tps is None and "target_tps" in new_config:
            new_target_tps = new_config["target_tps"]
            self.logger.info(f"Falling back to global target_tps: {new_target_tps}")

        # If found new TPS value, update it
        if new_target_tps is not None:
            self.tps_controller.update_target_tps(new_target_tps)
            self.logger.info(f"Target TPS updated to: {new_target_tps}")
        else:
            self.logger.info("No new target TPS value found in this reconfiguration.")

    def _update_stats_queue(self):
        """Get current statistics and put into queue."""
        try:
            stats = self.get_metrics()
            # Add defensive check to prevent crash when get_metrics fails
            if not stats:
                self.logger.warning("get_metrics() returned empty stats, skipping this cycle's update.")
                return

            # Core fix: ensure all core fields exist, including new p95 and p99 latency
            message = {
                'type': 'STATS_UPDATE',
                'db_id': self.db_id,
                'phase_name': self.phase_name,
                'ops_count': stats.get('ops_count_interval', 0),
                'total_latency_us': stats.get('total_latency_us_interval', 0),
                'p95_latency_us': stats.get('p95_latency_us_interval', 0.0),
                'p99_latency_us': stats.get('p99_latency_us_interval', 0.0),
                'errors': stats.get('errors_interval', 0),
                'cache_hits': stats.get('cache_hits_interval', 0),
                'cache_misses': stats.get('cache_misses_interval', 0),
            }
            self.stats_queue.put(message)
            self.logger.debug(f"Sent stats update: Ops={message['ops_count']}, P99={message['p99_latency_us']:.2f}us, Cache(H={message['cache_hits']}, M={message['cache_misses']})")
        except Exception as e:
            self.logger.error(f"Error updating stats queue: {e}", exc_info=True)

    def reconfigure_cache_size(self, num_pages: int):
        """
        Thread-safely adjust database connection cache size.
        """
        if num_pages is None or num_pages <= 0:
            self.logger.warning(f"Invalid cache page count requested ({num_pages}), skipping adjustment.")
            return
            
        with self.db_lock:
            try:
                # Use PRAGMA cache_size to set page count
                self.conn.execute(f"PRAGMA cache_size = {num_pages}")
                
                # Verify setting
                actual_pages = self.conn.execute("PRAGMA cache_size").fetchone()[0]
                kib_size = actual_pages * (self.page_size_bytes / 1024)
                
                self.logger.info(f"Cache size successfully reconfigured to {actual_pages} pages (approx {kib_size:.0f} KiB).")
            except apsw.SQLError as e:
                self.logger.error(f"SQL error setting cache size via PRAGMA: {e}")
            except Exception as e:
                self.logger.error(f"Unknown error reconfiguring cache size: {e}", exc_info=True)

    def _check_index_exists(self, index_name: str) -> bool:
        """Check if index exists"""
        if not self.conn:
            return False
        
        with self.db_lock:
            try:
                result = self.conn.execute(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name=?", 
                    (index_name,)
                ).fetchone()
                return result[0] > 0
            except Exception as e:
                self.logger.error(f"Error checking if index {index_name} exists: {e}")
                return False
    
    def _drop_priority_index(self):
        """Drop index on priority_value column"""
        if not self.conn:
            return
        
        table_name = "usertable"
        index_name = f"idx_{table_name}_priority_value"
        
        with self.db_lock:
            try:
                self.conn.execute(f"DROP INDEX IF EXISTS {index_name}")
                self.logger.info(f"Successfully dropped index: {index_name}")
            except Exception as e:
                self.logger.error(f"Error dropping index {index_name}: {e}")
    
    def _create_priority_index(self):
        """Create index on priority_value column"""
        if not self.conn:
            return
        
        table_name = "usertable"
        index_name = f"idx_{table_name}_priority_value"
        
        with self.db_lock:
            try:
                self.conn.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} (priority_value)")
                self.logger.info(f"Successfully created index: {index_name}")
            except Exception as e:
                self.logger.error(f"Error creating index {index_name}: {e}")
    
    def get_metrics(self) -> dict | None:
        """
        Thread-safely get incremental performance metrics from SQLite and YCSB workload.
        This method "assembles" a complete report by calling existing stats functions from other components.
        """
        if not self.conn or self._is_stopped:
            return None

        try:
            # 1. Reuse YCSBWorkload's get_stats_and_reset() method to get operation and latency stats
            # This method returns 8 values: (ops, total_latency_us, p50_us, p90_us, p95_us, p99_us, p999_us, errors)
            ycsb_stats = self.workload_generator.get_stats_and_reset()
            if not ycsb_stats:
                self.logger.warning("YCSBWorkload.get_stats_and_reset() returned no stats.")
                ops_interval, latency_us_interval, p50_us, p90_us, p95_us, p99_us, p999_us, errors_interval = 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0
            else:
                ops_interval, latency_us_interval, p50_us, p90_us, p95_us, p99_us, p999_us, errors_interval = ycsb_stats

            # 2. Reuse apsw.conn.status() method to get database-level cache stats (incremental)
            # Setting second parameter to True resets the value to 0 after getting it
            hits_interval = self.conn.status(apsw.SQLITE_DBSTATUS_CACHE_HIT, True)[0]
            misses_interval = self.conn.status(apsw.SQLITE_DBSTATUS_CACHE_MISS, True)[0]

            # 3. Assemble into final metrics dictionary
            metrics = {
                "ops_count_interval": ops_interval,
                "total_latency_us_interval": latency_us_interval,
                "p95_latency_us_interval": p95_us,
                "p99_latency_us_interval": p99_us,
                "errors_interval": errors_interval,
                "cache_hits_interval": hits_interval,
                "cache_misses_interval": misses_interval
            }
            return metrics

        except Exception as e:
            self.logger.error(f"Error getting metrics: {e}", exc_info=True)
            return None
