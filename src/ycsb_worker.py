import apsw
import threading
import time
from tps_controller import PreciseTpsController
from database_utils import *
import logging
from ycsb_utils import YCSBWorkload

# Try to import workload factory
try:
    from workload.workload_factory import create_workload
    WORKLOAD_FACTORY_AVAILABLE = True
except ImportError:
    WORKLOAD_FACTORY_AVAILABLE = False

class YCSBWorker(threading.Thread):
    def __init__(self, db_path: str, worker_id: str, 
                 ycsb_config: dict, db_config: dict, # db_config is worker_specific_db_config from DatabaseInstance for B5
                 is_b5_worker: bool, b5_shared_cache_uri: str | None, 
                 stop_event: threading.Event, warmup_event: threading.Event, 
                 global_experiment_start_event: threading.Event, 
                 results_queue, # Typically a queue.Queue instance for results
                 worker_seed_offset: int, # Seed for YCSBWorkload
                 logging_level: str = "INFO",
                 orchestrator_config: dict = None, # New: for passing orchestrator config
                 strategy_id: str = ""):
        super().__init__(name=worker_id) # Set thread name
        self.db_path = db_path
        self.worker_id = worker_id # e.g., db_unified_b5_w1
        self.ycsb_cfg = ycsb_config
        self.db_cfg = db_config # This contains worker-specific target_tps, tps_control settings

        self.is_b5_worker = is_b5_worker
        self.b5_shared_cache_uri_to_use = b5_shared_cache_uri

        self.stop_event = stop_event
        self.warmup_event = warmup_event
        self.global_experiment_start_event = global_experiment_start_event
        self.results_queue = results_queue # For future use if needed for detailed per-op results
        self.worker_seed_offset = worker_seed_offset
        self.orchestrator_config = orchestrator_config or {}
        self.strategy_id = strategy_id

        # Setup logger for this worker
        self.logger = logging.getLogger(f"YCSBWorker.{self.worker_id}")
        if not self.logger.handlers:
            self.logger.setLevel(getattr(logging, logging_level, logging.INFO))
            ch = logging.StreamHandler()
            formatter = logging.Formatter(f'%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            self.logger.propagate = False

        self.conn = None
        self.ycsb_workload = None
        self.conn_lock = threading.Lock() # For protecting self.conn access within this worker

        # TPS control specific to this worker
        self.target_tps_for_worker = self.db_cfg.get("target_tps", 0)
        tps_control_settings = self.db_cfg.get("tps_control", {})
        self.tps_control_enabled_for_worker = tps_control_settings.get("enabled", False)
        self.tps_debug_enabled_for_worker = tps_control_settings.get("debug_output", False)
        
        self.tps_controller = None
        if self.tps_control_enabled_for_worker and self.target_tps_for_worker > 0:
            self.tps_controller = PreciseTpsController(self.target_tps_for_worker, self.worker_id)
            if self.tps_debug_enabled_for_worker:
                self.tps_controller.enable_debug(True)
            self.logger.info(f"TPS Controller initialized: Target={self.target_tps_for_worker}")
        else:
            self.logger.info(f"TPS Controller NOT initialized (Enabled: {self.tps_control_enabled_for_worker}, TargetTPS: {self.target_tps_for_worker})")

        self.ops_executed_in_warmup = 0
        self.ops_executed_total = 0
        self.latency_sum_total = 0.0
        # self.own_ready_event = threading.Event() # Not strictly needed if Orchestrator waits for DBInstance.ready

    def _connect_and_init_workload(self):
        self.logger.info(f"Connecting to DB and initializing YCSBWorkload...")
        self.conn = safe_connect_database_with_retry(
            self.db_path, 
            is_b5_worker_shared_cache=self.is_b5_worker,
            shared_cache_uri=self.b5_shared_cache_uri_to_use # Pass the URI if B5 worker
        )
        if not self.conn:
            self.logger.error(f"CRITICAL: Failed to establish database connection.")
            raise RuntimeError(f"Worker {self.worker_id} failed to connect to DB.")

        # Try to use workload factory, fallback to original method if not available
        workload_type = self.orchestrator_config.get('general_experiment_setup', {}).get('workload_type', 'ycsb')
        
        if WORKLOAD_FACTORY_AVAILABLE and workload_type in ['ycsb', 'tpcc', 'trace']:
            # Use factory pattern to create workload
            initial_record_count = self.db_cfg.get("ycsb_initial_record_count", 10000)
            
            # Merge configurations
            merged_config = self.orchestrator_config.copy()
            if 'ycsb_general_config' not in merged_config:
                merged_config['ycsb_general_config'] = {}
            merged_config['ycsb_general_config'].update(self.ycsb_cfg)
            
            self.ycsb_workload = create_workload(
                conn=self.conn,
                conn_lock=self.conn_lock,
                db_id=self.worker_id,
                db_config=self.db_cfg,
                orchestrator_config=merged_config,
                initial_record_count=initial_record_count,
                worker_seed_offset=self.worker_seed_offset,
                strategy_id=self.strategy_id,
                workload_type=workload_type
            )
            self.logger.info(f"{workload_type.upper()} Workload initialized via factory with seed_offset {self.worker_seed_offset}.")
        else:
            # Backward compatibility: use original method
            self.ycsb_workload = YCSBWorkload(
                conn=self.conn,
                conn_lock=self.conn_lock, 
                ycsb_config=self.ycsb_cfg,
                db_config=self.db_cfg, # Contains record_count for this DB (unified for B5)
                # record_count=self.db_cfg.get("ycsb_initial_record_count"), # Pulled from db_cfg by YCSBWorkload
                worker_seed_offset=self.worker_seed_offset,
                logger_parent_name=f"YCSBWorker.{self.worker_id}" # For YCSBWorkload's own logger
            )
            self.logger.info(f"YCSBWorkload initialized with seed_offset {self.worker_seed_offset}.")

    def run(self):
        self._connect_and_init_workload()
        self.logger.info(f"Ready. Waiting for global start signal.")

        self.global_experiment_start_event.wait()
        self.logger.info(f"Received global start signal. Beginning operations.")

        try:
            if self.tps_controller:
                self._run_with_tps_control()
            else:
                self._run_without_tps_control()
        finally:
            self._cleanup_connection()
            self.logger.info(f"Execution finished.")
            
    def _cleanup_connection(self):
        if self.conn:
            with self.conn_lock:
                if self.conn: # Re-check inside lock
                    try:
                        self.conn.close()
                        self.logger.info(f"Database connection closed.")
                    except Exception as e:
                        self.logger.error(f"Error closing database connection: {e}")
                    finally:
                        self.conn = None
            
    def _run_with_tps_control(self):
        operation_count = 0
        self.logger.info(f"Starting TPS controlled execution (Target: {self.target_tps_for_worker}).")
        
        while not self.stop_event.is_set():
            # Check if we need to wait to maintain TPS
            if self.tps_controller.should_wait():
                # Dynamically calculate sleep time to avoid fixed 1ms rough waiting
                remaining_wait_time = self.tps_controller.get_remaining_wait_time()
                if remaining_wait_time > 0:
                    # Use smaller sleep time for better TPS precision
                    sleep_time = min(remaining_wait_time, 0.0001)  # Sleep at most 0.1ms
                    time.sleep(sleep_time)
                continue
            
            # YCSBWorkload.run_operation returns latency_ms (float)
            latency_ms = self.ycsb_workload.run_operation()
            
            # Check if operation was successful (non-zero latency indicates success)
            success = latency_ms > 0
            
            # Record operation completion, convert latency from milliseconds to microseconds
            if success:
                latency_us = latency_ms * 1000.0
                self.tps_controller.record_completion(latency_us)
            else:
                self.tps_controller.record_error()
            
            if not self.warmup_event.is_set():
                if success: self.ops_executed_in_warmup += 1
            else:
                if success: 
                    self.ops_executed_total += 1
                    self.latency_sum_total += latency_ms
            if success: operation_count += 1
        self.logger.info(f"TPS controlled execution finished. Total ops (post-warmup): {self.ops_executed_total}")
                
    def _run_without_tps_control(self):
        self.logger.info(f"Starting full speed execution.")
        while not self.stop_event.is_set():
            latency_ms = self.ycsb_workload.run_operation()
            success = latency_ms > 0
            
            if not self.warmup_event.is_set():
                if success: self.ops_executed_in_warmup += 1
            else:
                if success:
                    self.ops_executed_total += 1
                    self.latency_sum_total += latency_ms
        self.logger.info(f"Full speed execution finished. Total ops (post-warmup): {self.ops_executed_total}")

    def get_interval_cache_stats(self):
        # These are interval stats, reset on read by SQLite PRAGMA with reset=True
        hits, misses, writes = 0, 0, 0
        if self.conn:
            with self.conn_lock:
                if self.conn and hasattr(self.conn, 'status'):
                    h_tuple = self.conn.status(apsw.SQLITE_DBSTATUS_CACHE_HIT, True)
                    m_tuple = self.conn.status(apsw.SQLITE_DBSTATUS_CACHE_MISS, True)
                    w_tuple = self.conn.status(apsw.SQLITE_DBSTATUS_CACHE_WRITE, True)
                    hits = h_tuple[0] if isinstance(h_tuple, tuple) else h_tuple
                    misses = m_tuple[0] if isinstance(m_tuple, tuple) else m_tuple
                    writes = w_tuple[0] if isinstance(w_tuple, tuple) else w_tuple
        return {"hits": hits, "misses": misses, "writes": writes}

    def get_stats_and_reset(self):
        # These are cumulative stats for the worker over a reporting interval (post-warmup)
        ops = self.ops_executed_total
        sum_latency = self.latency_sum_total
        
        self.ops_executed_total = 0
        self.latency_sum_total = 0.0
        # Warmup ops are a one-time count, not reset here.
        return ops, sum_latency

    def get_tps_stats(self):
        if self.tps_controller:
            stats = self.tps_controller.get_current_stats()
            return {
                "tps_control_enabled": True,
                "target_tps": stats.target_tps,
                "actual_tps": stats.actual_tps,
                "accuracy_percentage": stats.accuracy_percentage,
                "executed_operations": stats.executed_operations,
                "errors": stats.errors
            }
        else:
            return {"tps_control_enabled": False, "target_tps":0, "actual_tps":0, "accuracy_percentage":0, "executed_operations":0, "errors":0}
