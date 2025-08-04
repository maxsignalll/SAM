# Workload module initialization
from .base_workload import BaseWorkload
from .ycsb_workload import YCSBWorkload
from .tpcc_workload import TPCCWorkload
from .trace_workload import TraceWorkload

def create_workload(workload_type: str, conn, db_config: dict, config: dict) -> BaseWorkload:
    """
    Factory function to create appropriate workload instance based on workload type
    
    Args:
        workload_type: Workload type ('ycsb', 'tpcc', 'trace')
        conn: Database connection
        db_config: Database configuration
        config: Complete configuration
        
    Returns:
        Subclass instance of BaseWorkload
    """
    # Create a simple logger
    import logging
    logger = logging.getLogger(f"Workload.{db_config.get('id', 'Unknown')}")
    
    if workload_type == "ycsb":
        # YCSB workload needs special configuration format
        db_id = db_config.get('id', 'Unknown')
        ycsb_config = config.copy()
        ycsb_config['ycsb_config'] = config.get('ycsb_general_config', {})
        seed_offset = hash(db_id)
        return YCSBWorkload(conn, db_id, ycsb_config, seed_offset, logger)
    elif workload_type == "tpcc":
        return TPCCWorkload(conn, db_config.get('id', 'Unknown'), config, 0, logger)
    elif workload_type == "trace":
        return TraceWorkload(conn, db_config.get('id', 'Unknown'), config, 0, logger)
    else:
        raise ValueError(f"Unsupported workload type: {workload_type}")

__all__ = ['BaseWorkload', 'YCSBWorkload', 'TPCCWorkload', 'TraceWorkload', 'create_workload']