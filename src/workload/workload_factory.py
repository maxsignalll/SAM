"""
Workload factory module
Provides backward-compatible workload creation mechanism
"""

import logging
from typing import Dict, Any, Optional
import threading
import apsw
import random
import time

# Import existing YCSBWorkload to maintain compatibility
try:
    from ..ycsb_utils import YCSBWorkload as LegacyYCSBWorkload
except ImportError:
    # If relative import fails, try absolute import
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ycsb_utils import YCSBWorkload as LegacyYCSBWorkload

# Import new workload implementations
from .ycsb_workload import YCSBWorkload as NewYCSBWorkload
from .tpcc_workload import TPCCWorkload
from .trace_workload import TraceWorkload


class WorkloadAdapter:
    """
    Unified workload adapter, provides interface compatible with existing YCSBWorkload
    """
    
    def __init__(self, workload_impl):
        """
        Args:
            workload_impl: Actual workload implementation (subclass of BaseWorkload)
        """
        self.workload = workload_impl
        self.logger = logging.getLogger(f"WorkloadAdapter.{workload_impl.db_id}")
        
    def run_operation(self) -> float:
        """Run a single operation and return latency (milliseconds)"""
        return self.workload.run_operation()
        
    def get_stats_and_reset(self):
        """Get and reset statistics, returns 8-tuple"""
        return self.workload.get_stats_and_reset()
        
    def reset_stats(self):
        """Reset statistics"""
        self.workload.reset_stats()
        
    def update_access_pattern(self, *args, **kwargs):
        """Update access pattern (dynamic workload)"""
        self.workload.update_access_pattern(*args, **kwargs)
        
    def __getattr__(self, name):
        """Proxy other method calls to underlying workload"""
        return getattr(self.workload, name)


def create_workload(
    conn: apsw.Connection,
    conn_lock: threading.Lock,
    db_id: str,
    db_config: Dict[str, Any],
    orchestrator_config: Dict[str, Any],
    initial_record_count: int,
    worker_seed_offset: int = 0,
    strategy_id: str = "",
    workload_type: Optional[str] = None
):
    """
    Factory function to create workload instance
    
    Args:
        conn: Database connection
        conn_lock: Database connection lock
        db_id: Database ID
        db_config: Database configuration
        orchestrator_config: Orchestrator configuration
        initial_record_count: Initial record count
        worker_seed_offset: Random seed offset
        strategy_id: Strategy ID
        workload_type: Workload type (optional, if not specified, read from configuration)
        
    Returns:
        Workload instance (compatible with existing YCSBWorkload interface)
    """
    logger = logging.getLogger(f"WorkloadFactory.{db_id}")
    
    # Check if it's B5 multi-table mode
    if db_config.get('b5_multi_table', False):
        logger.info(f"Detected B5 multi-table mode, creating B5 multi-table workload executor")
        # Import B5 multi-table workload
        try:
            from ..b5_multi_table_workload import B5MultiTableWorkload
        except ImportError:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from b5_multi_table_workload import B5MultiTableWorkload
        
        # Create B5 multi-table workload and wrap as adapter
        b5_workload = B5MultiTableWorkload(conn, db_config, workload_type or 'ycsb')
        
        # Create a compatible wrapper
        class B5WorkloadWrapper:
            def __init__(self, b5_workload, db_config, orchestrator_config):
                self.b5_workload = b5_workload
                self.db_config = db_config
                self.orchestrator_config = orchestrator_config
                self.db_id = db_config['id']
                self.logger = logging.getLogger(f"B5Wrapper.{self.db_id}")
                self.stats = {'ops_count': 0, 'cache_hits': 0, 'cache_misses': 0}
                self.current_phase_config = {}
                self.access_pattern_config = {}
                self.target_tps = 0
                self.record_count = db_config.get('ycsb_initial_record_count', 0)
                
                # Initialize operation counters
                self.read_ops = 0
                self.update_ops = 0
                self.total_latency = 0.0
                self.latencies = []
                
                # Initialize hot spot sets (for each virtual database)
                self.hot_spot_sets = {}
                self._initialize_hot_spot_sets()
                
            def _initialize_hot_spot_sets(self):
                """Initialize hot spot sets for each virtual database"""
                ycsb_config = self.orchestrator_config.get('ycsb_general_config', {})
                hot_rate = ycsb_config.get('hot_rate', 0.167)  # Default 16.7%
                
                for orig_db in self.db_config.get('b5_original_dbs', []):
                    db_id = orig_db['id']
                    record_count = orig_db.get('ycsb_initial_record_count', 10000)
                    hot_spot_size = max(1, int(record_count * hot_rate))
                    
                    # Use deterministic random seed based on database ID
                    seed = hash(db_id) & 0x7FFFFFFF
                    rng = random.Random(seed)
                    
                    # Generate hot spot set (randomly selected priority_values)
                    all_values = list(range(1, record_count + 1))
                    rng.shuffle(all_values)
                    self.hot_spot_sets[db_id] = all_values[:hot_spot_size]
                    
                    self.logger.info(f"Initialized hot spot set for virtual DB {db_id}: {hot_spot_size} hot spots (total records: {record_count})")
                
            def run_operation(self):
                # Execute B5 multi-table operation
                start_time = time.time()
                
                try:
                    # Determine which virtual database to access
                    virtual_db_id = self.b5_workload.get_virtual_db_for_phase(
                        self.current_phase_config, time.time()
                    )
                    
                    # Check if None is returned
                    if virtual_db_id is None:
                        self.logger.error("B5 failed to get virtual database ID, skipping operation")
                        return 0.0  # Return 0 latency
                    
                    # Check if it's TPC-C or YCSB workload
                    if self.b5_workload.workload_type == 'tpcc':
                        # TPC-C transaction processing
                        # Simple implementation: randomly select transaction type
                        txn_types = ['new_order', 'payment', 'order_status', 'delivery', 'stock_level']
                        weights = [45, 43, 4, 4, 4]  # TPC-C standard weights
                        txn_type = random.choices(txn_types, weights=weights)[0]
                        
                        # Generate transaction parameters (simplified version)
                        # For TPC-C, get_record_count returns warehouse count
                        warehouse_count = self.b5_workload.get_record_count_for_virtual_db(virtual_db_id)
                        if warehouse_count == 0:
                            warehouse_count = 1  # Default at least 1 warehouse
                        params = {
                            'warehouse_id': random.randint(1, warehouse_count),
                            'district_id': random.randint(1, 10),
                            'customer_id': random.randint(1, 3000)
                        }
                        
                        # Execute TPC-C transaction
                        result = self.b5_workload.execute_tpcc_transaction(
                            txn_type, virtual_db_id, params
                        )
                        # Record SQL time
                        if 'sql_time_ms' in result:
                            self.logger.debug(f"B5 TPC-C SQL execution time: {result['sql_time_ms']:.4f} ms")
                    else:
                        # YCSB operation processing (original logic)
                        # Get hot spot set for this virtual database
                        hot_spot_set = self.hot_spot_sets.get(virtual_db_id, [])
                        if not hot_spot_set:
                            # If no hot spot set, fall back to full range
                            record_count = self.b5_workload.get_record_count_for_virtual_db(virtual_db_id)
                            if record_count == 0:
                                record_count = 10000  # Default value
                            priority_value = random.randint(1, record_count)
                        else:
                            # Select from hot spot set
                            # Use Zipfian distribution (if configured) or uniform distribution
                            access_pattern = self.access_pattern_config.get(virtual_db_id, {})
                            if access_pattern.get('type') == 'zipfian':
                                # Simple Zipfian approximation: more likely to select elements at the front of hot spot set
                                # Use power law distribution approximation
                                index = int(random.paretovariate(1.2) - 1)  # alpha=1.2 approximates Zipfian
                                index = min(index, len(hot_spot_set) - 1)
                            else:
                                # Uniform distribution
                                index = random.randint(0, len(hot_spot_set) - 1)
                            priority_value = hot_spot_set[index]
                        
                        # Decide operation type (simple read/write ratio)
                        if random.random() < 0.5:  # 50% read
                            operation = "READ"
                            self.read_ops += 1
                        else:
                            operation = "UPDATE"
                            self.update_ops += 1
                        
                        # Execute operation
                        result = self.b5_workload.execute_ycsb_operation(
                            operation, virtual_db_id, priority_value,
                            f"updated_data_{time.time()}" if operation == "UPDATE" else None
                        )
                    
                    # Update statistics
                    self.stats['ops_count'] += 1
                    if result.get('success'):
                        self.stats['cache_hits'] += 1
                    else:
                        self.stats['cache_misses'] += 1
                    
                except Exception as e:
                    self.logger.error(f"B5 operation failed: {e}")
                    self.stats['cache_misses'] += 1
                    # Calculate latency correctly even in exception cases
                    exception_latency = (time.time() - start_time) * 1000
                    self.total_latency += exception_latency
                    self.latencies.append(exception_latency)
                    return exception_latency
                
                # Calculate latency
                latency_ms = (time.time() - start_time) * 1000
                self.total_latency += latency_ms
                self.latencies.append(latency_ms)
                
                # If TPC-C and latency is abnormally low, log warning
                if self.b5_workload.workload_type == 'tpcc' and latency_ms < 1.0:
                    self.logger.warning(f"B5 TPC-C latency abnormally low: {latency_ms:.4f} ms, transaction type: {result.get('transaction_type', 'unknown')}")
                
                # Keep last 1000 latency values for P99 calculation
                if len(self.latencies) > 1000:
                    self.latencies.pop(0)
                
                return latency_ms
                
            def get_stats_and_reset(self):
                # Calculate P99 latency
                p99_latency = 0
                if self.latencies:
                    sorted_latencies = sorted(self.latencies)
                    p99_index = int(len(sorted_latencies) * 0.99)
                    p99_latency = sorted_latencies[min(p99_index, len(sorted_latencies) - 1)]
                    
                
                # Calculate other percentiles
                p50_latency = 0
                p90_latency = 0
                p95_latency = 0
                p999_latency = 0
                
                if self.latencies:
                    sorted_latencies = sorted(self.latencies)
                    n = len(sorted_latencies)
                    p50_latency = sorted_latencies[int(n * 0.50)]
                    p90_latency = sorted_latencies[int(n * 0.90)]
                    p95_latency = sorted_latencies[int(n * 0.95)]
                    p999_latency = sorted_latencies[min(int(n * 0.999), n-1)]
                
                # Return compatible 8-tuple statistics (consistent with DatabaseInstance expected format)
                # Format: (ops, total_latency_us, p50_us, p90_us, p95_us, p99_us, p999_us, errors)
                stats = (
                    self.stats['ops_count'],           # ops
                    self.total_latency * 1000,         # total_latency_us (convert to microseconds)
                    p50_latency * 1000,                # p50_us (convert to microseconds)
                    p90_latency * 1000,                # p90_us (convert to microseconds)
                    p95_latency * 1000,                # p95_us (convert to microseconds)
                    p99_latency * 1000,                # p99_us (convert to microseconds)
                    p999_latency * 1000,               # p999_us (convert to microseconds)
                    0                                  # errors
                )
                
                # Reset statistics
                self.stats = {'ops_count': 0, 'cache_hits': 0, 'cache_misses': 0}
                self.read_ops = 0
                self.update_ops = 0
                self.total_latency = 0.0
                self.latencies = []
                
                return stats
                
            def reset_stats(self):
                self.stats = {'ops_count': 0, 'cache_hits': 0, 'cache_misses': 0}
                self.read_ops = 0
                self.update_ops = 0
                self.total_latency = 0.0
                self.latencies = []
                
            def update_phase_config(self, phase_config):
                # Update phase configuration
                self.current_phase_config = phase_config
                
            def update_access_pattern(self, access_pattern_config, authoritative_record_count=None):
                # Update access pattern - add record_count parameter to match standard interface
                # For B5, need to extract access pattern for each virtual DB from phase configuration
                if hasattr(self, 'current_phase_config') and self.current_phase_config:
                    phase_access_patterns = self.current_phase_config.get('ycsb_config_overrides', {}).get('access_pattern_per_db', {})
                    self.access_pattern_config = phase_access_patterns
                else:
                    self.access_pattern_config = access_pattern_config
                if authoritative_record_count:
                    self.record_count = authoritative_record_count
                
            def update_tps_target(self, target_tps):
                # Update TPS target
                self.target_tps = target_tps
        
        return B5WorkloadWrapper(b5_workload, db_config, orchestrator_config)
    
    # Get workload type
    if workload_type is None:
        workload_type = orchestrator_config.get('general_experiment_setup', {}).get('workload_type', 'ycsb')
    
    logger.info(f"Creating workload, type: {workload_type}")
    
    # For complete backward compatibility, if YCSB and no explicit requirement to use new implementation, use original implementation
    if workload_type == 'ycsb':
        use_new_impl = orchestrator_config.get('use_new_workload_impl', False)
        
        if not use_new_impl:
            # Use original YCSBWorkload implementation to ensure complete compatibility
            logger.info("Using traditional YCSB implementation (fully backward compatible)")
            return LegacyYCSBWorkload(
                conn=conn,
                conn_lock=conn_lock,
                db_id=db_id,
                ycsb_config=db_config,
                general_db_config=orchestrator_config,
                initial_record_count=initial_record_count,
                worker_seed_offset=worker_seed_offset,
                strategy_id=strategy_id
            )
        else:
            # Use new YCSB implementation
            logger.info("Using new YCSB implementation")
            config = {
                'ycsb_config': orchestrator_config.get('ycsb_general_config', {}),
                'workload_proportions': orchestrator_config.get('ycsb_general_config', {}).get('ycsb_workload_proportions', {})
            }
            workload = NewYCSBWorkload(
                conn=conn,
                db_id=db_id,
                config=config,
                seed_offset=worker_seed_offset,
                logger=logger
            )
            return WorkloadAdapter(workload)
            
    elif workload_type == 'tpcc':
        # Create TPC-C workload - use dedicated adapter to ensure complete compatibility
        logger.info("Creating TPC-C workload (using compatibility adapter)")
        try:
            from ..tpcc_workload_adapter import TPCCWorkloadAdapter
        except ImportError:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from tpcc_workload_adapter import TPCCWorkloadAdapter
        
        return TPCCWorkloadAdapter(
            conn=conn,
            conn_lock=conn_lock,
            db_id=db_id,
            ycsb_config=db_config,
            general_db_config=orchestrator_config,
            initial_record_count=initial_record_count,
            worker_seed_offset=worker_seed_offset,
            strategy_id=strategy_id
        )
        
    elif workload_type == 'trace':
        # Create Trace workload - use dedicated adapter to ensure complete compatibility
        logger.info("Creating Trace workload (using compatibility adapter)")
        try:
            from ..trace_workload_adapter import TraceWorkloadAdapter
        except ImportError:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from trace_workload_adapter import TraceWorkloadAdapter
        
        return TraceWorkloadAdapter(
            conn=conn,
            conn_lock=conn_lock,
            db_id=db_id,
            ycsb_config=db_config,
            general_db_config=orchestrator_config,
            initial_record_count=initial_record_count,
            worker_seed_offset=worker_seed_offset,
            strategy_id=strategy_id
        )
        
    else:
        # Unknown type, fall back to YCSB
        logger.warning(f"Unknown workload type: {workload_type}, using default YCSB")
        return LegacyYCSBWorkload(
            conn=conn,
            conn_lock=conn_lock,
            db_id=db_id,
            ycsb_config=db_config,
            general_db_config=orchestrator_config,
            initial_record_count=initial_record_count,
            worker_seed_offset=worker_seed_offset,
            strategy_id=strategy_id
        )


def create_initial_data(
    conn: apsw.Connection,
    db_id: str,
    orchestrator_config: Dict[str, Any],
    record_count: int,
    workload_type: Optional[str] = None
):
    """
    Factory function to create initial data
    
    Args:
        conn: Database connection
        db_id: Database ID
        orchestrator_config: Orchestrator configuration
        record_count: Record count
        workload_type: Workload type
    """
    logger = logging.getLogger(f"WorkloadFactory.{db_id}")
    
    # Get workload type
    if workload_type is None:
        workload_type = orchestrator_config.get('general_experiment_setup', {}).get('workload_type', 'ycsb')
    
    logger.info(f"Creating initial data for {workload_type} type")
    
    if workload_type == 'ycsb':
        # Use existing YCSB data generation
        try:
            from ..ycsb_utils import create_ycsb_table, load_initial_data
        except ImportError:
            from ycsb_utils import create_ycsb_table, load_initial_data
        create_ycsb_table(conn, row_size_bytes=orchestrator_config.get('ycsb_general_config', {}).get('ycsb_row_size_bytes', 2048))
        load_initial_data(conn, record_count, db_id=db_id, row_size_bytes=orchestrator_config.get('ycsb_general_config', {}).get('ycsb_row_size_bytes', 2048))
        
    elif workload_type == 'tpcc':
        # Create TPC-C schema and data
        config = {'tpcc_general_config': orchestrator_config.get('tpcc_general_config', {})}
        workload = TPCCWorkload(conn, db_id, config, logger=logger)
        workload.create_schema()
        workload.generate_initial_data(record_count)
        
    elif workload_type == 'trace':
        # Create Trace schema and data
        # Get db-specific configuration (copying TPC-C pattern)
        db_config = next((db for db in orchestrator_config.get('database_instances', []) if db['id'] == db_id), {})
        config = {
            'trace_config': orchestrator_config.get('trace_general_config', {}),
            'trace_file': db_config.get('trace_file')
        }
        workload = TraceWorkload(conn, db_id, config, logger=logger)
        workload.create_schema()
        workload.generate_initial_data(record_count)
        
    else:
        # Default to YCSB
        logger.warning(f"Unknown workload type: {workload_type}, using default YCSB data generation")
        try:
            from ..ycsb_utils import create_ycsb_table, load_initial_data
        except ImportError:
            from ycsb_utils import create_ycsb_table, load_initial_data
        create_ycsb_table(conn, row_size_bytes=orchestrator_config.get('ycsb_general_config', {}).get('ycsb_row_size_bytes', 2048))
        load_initial_data(conn, record_count, db_id=db_id, row_size_bytes=orchestrator_config.get('ycsb_general_config', {}).get('ycsb_row_size_bytes', 2048))