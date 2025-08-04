"""
Base workload abstract class
Base class for all workload types (YCSB, TPC-C, Trace)
"""

from abc import ABC, abstractmethod
import apsw
import threading
import time
import logging
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
import numpy as np


class BaseWorkload(ABC):
    """Base abstract class for all workloads"""
    
    def __init__(self, 
                 conn: apsw.Connection,
                 db_id: str,
                 config: Dict[str, Any],
                 seed_offset: int = 0,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize base workload
        
        Args:
            conn: Database connection
            db_id: Database identifier
            config: Workload configuration
            seed_offset: Random seed offset
            logger: Logger
        """
        self.conn = conn
        self.db_id = db_id
        self.config = config
        self.seed_offset = seed_offset
        self.conn_lock = threading.Lock()
        
        # Set up logger
        if logger is None:
            self.logger = logging.getLogger(f"BaseWorkload.{db_id}")
        else:
            self.logger = logger
            
        # Statistics (common to all workload types)
        self.stats_lock = threading.Lock()
        self.reset_stats()
        
        # Latency recording queue (for calculating percentiles)
        self.latency_window_size = 1000
        self.latency_deque = deque(maxlen=self.latency_window_size)
        
    def reset_stats(self):
        """Reset statistics"""
        with self.stats_lock:
            self.stats = {
                'total_ops': 0,
                'read_ops': 0,
                'write_ops': 0,
                'scan_ops': 0,
                'errors': 0,
                'total_latency_us': 0.0,
                'latencies': []
            }
            
    @abstractmethod
    def create_schema(self) -> None:
        """Create database schema (table structure)"""
        pass
        
    @abstractmethod
    def generate_initial_data(self, record_count: int) -> None:
        """Generate initial data"""
        pass
        
    @abstractmethod
    def generate_operation(self) -> Tuple[str, Dict[str, Any]]:
        """Generate next operation (operation type, parameters)"""
        pass
        
    @abstractmethod
    def execute_operation(self, op_type: str, params: Dict[str, Any]) -> bool:
        """Execute specific operation, return whether successful"""
        pass
    
    def update_access_pattern(self, pattern_config: Dict[str, Any], record_count: int = None):
        """
        Update access pattern (for dynamic workloads)
        
        Args:
            pattern_config: New access pattern configuration
            record_count: Current record count (optional)
        """
        # Base class implementation: do nothing
        # Subclasses can override this method to support dynamic access pattern updates
        self.logger.info(f"Access pattern update requested but not implemented for {self.__class__.__name__}")
        pass
    
    def run_single_operation(self) -> Tuple[bool, float]:
        """
        Common flow for running an operation
        
        Returns:
            (success, latency_us): Whether operation succeeded and latency (microseconds)
        """
        try:
            # Generate operation
            op_type, params = self.generate_operation()
            
            # Execute operation and time it
            start_time = time.perf_counter()
            with self.conn_lock:
                success = self.execute_operation(op_type, params)
            end_time = time.perf_counter()
            
            # Calculate latency (microseconds)
            latency_us = (end_time - start_time) * 1_000_000
            
            # Update statistics
            with self.stats_lock:
                self.stats['total_ops'] += 1
                self.stats['total_latency_us'] += latency_us
                
                if op_type.startswith('read'):
                    self.stats['read_ops'] += 1
                elif op_type.startswith('write') or op_type.startswith('update'):
                    self.stats['write_ops'] += 1
                elif op_type.startswith('scan'):
                    self.stats['scan_ops'] += 1
                    
                if not success:
                    self.stats['errors'] += 1
                    
                # Record latency for percentile calculation
                self.latency_deque.append(latency_us)
                
            return success, latency_us
            
        except Exception as e:
            self.logger.error(f"Error in run_operation: {e}", exc_info=True)
            with self.stats_lock:
                self.stats['errors'] += 1
            return False, 0.0
    
    def run_operation(self) -> float:
        """
        Run single operation and return latency (milliseconds)
        Compatible with existing system interface
        
        Returns:
            Latency (milliseconds), return 0 on failure
        """
        success, latency_us = self.run_single_operation()
        if success:
            return latency_us / 1000.0  # Convert to milliseconds
        else:
            return 0.0
    
    def get_stats_and_reset(self) -> Tuple[int, float, float, float, float, float, float, int]:
        """
        Get and reset statistics (compatible with existing YCSB interface)
        
        Returns:
            8-tuple: (ops, total_latency_us, p50_us, p90_us, p95_us, p99_us, p999_us, errors)
        """
        try:
            print(f"[DEBUG] {self.db_id} BaseWorkload.get_stats_and_reset() starting")
            with self.stats_lock:
                print(f"[DEBUG] {self.db_id} Got statistics lock successfully, total_ops={self.stats['total_ops']}")
                print(f"[DEBUG] {self.db_id} latency_deque length: {len(self.latency_deque)}")
                
                # Calculate latency percentiles
                if self.latency_deque:
                    print(f"[DEBUG] {self.db_id} Calculating percentiles...")
                    latencies = list(self.latency_deque)
                    print(f"[DEBUG] {self.db_id} latencies list length: {len(latencies)}")
                    
                    try:
                        p50 = np.percentile(latencies, 50)
                        print(f"[DEBUG] {self.db_id} p50 calculation complete: {p50}")
                        p90 = np.percentile(latencies, 90)
                        print(f"[DEBUG] {self.db_id} p90 calculation complete: {p90}")
                        p95 = np.percentile(latencies, 95)
                        p99 = np.percentile(latencies, 99)
                        p999 = np.percentile(latencies, 99.9)
                        print(f"[DEBUG] {self.db_id} All percentile calculations complete")
                    except Exception as e:
                        print(f"[DEBUG] {self.db_id} Percentile calculation error: {e}")
                        p50 = p90 = p95 = p99 = p999 = 0.0
                else:
                    print(f"[DEBUG] {self.db_id} latency_deque is empty, using default values")
                    p50 = p90 = p95 = p99 = p999 = 0.0
                    
                # Collect current statistics
                result = (
                    self.stats['total_ops'],
                    self.stats['total_latency_us'],
                    p50,
                    p90,
                    p95,
                    p99,
                    p999,
                    self.stats['errors']
                )
                
                print(f"[DEBUG] {self.db_id} Statistics result: {result}")
                
                # Reset statistics
                self.reset_stats()
                self.latency_deque.clear()
                
                print(f"[DEBUG] {self.db_id} BaseWorkload.get_stats_and_reset() complete")
                return result
                
        except Exception as e:
            print(f"[DEBUG] {self.db_id} BaseWorkload.get_stats_and_reset() exception: {e}")
            import traceback
            traceback.print_exc()
            # Return default values to avoid system crash
            return (0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
            
    def get_detailed_stats(self) -> Dict[str, Any]:
        """
        Get detailed statistics (for advanced analysis)
        
        Returns:
            Dictionary containing detailed statistics
        """
        with self.stats_lock:
            total_ops = self.stats['total_ops']
            if total_ops > 0:
                avg_latency_us = self.stats['total_latency_us'] / total_ops
            else:
                avg_latency_us = 0.0
                
            return {
                'total_ops': total_ops,
                'read_ops': self.stats['read_ops'],
                'write_ops': self.stats['write_ops'],
                'scan_ops': self.stats['scan_ops'],
                'errors': self.stats['errors'],
                'avg_latency_us': avg_latency_us,
                'read_ratio': self.stats['read_ops'] / total_ops if total_ops > 0 else 0,
                'write_ratio': self.stats['write_ops'] / total_ops if total_ops > 0 else 0,
                'error_rate': self.stats['errors'] / total_ops if total_ops > 0 else 0
            }
            
    def warmup(self, warmup_ops: int = 1000):
        """
        Warm up workload
        
        Args:
            warmup_ops: Number of warmup operations
        """
        self.logger.info(f"Starting warmup with {warmup_ops} operations")
        for i in range(warmup_ops):
            self.run_operation()
            if i % 100 == 0:
                self.logger.debug(f"Warmup progress: {i}/{warmup_ops}")
        
        # Clear statistics from warmup period
        self.reset_stats()
        self.logger.info("Warmup completed")