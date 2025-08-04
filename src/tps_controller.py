"""
TPS (Transactions Per Second) Controller
Implements precise transaction processing rate control

Main features:
1. Precise control of operation execution frequency
2. Calculate actual TPS performance metrics
3. Provide detailed performance statistics
"""

import time
import threading
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging
from perf_counter import PerfCounter


@dataclass
class TpsStats:
    """TPS statistics data structure"""
    executed_operations: int = 0
    total_time: float = 0.0
    errors: int = 0
    skipped_operations: int = 0
    actual_tps: float = 0.0
    target_tps: int = 0
    accuracy_percentage: float = 0.0


class PreciseTpsController:
    """
    A high-precision pacing smoother to ensure operations are executed at a smooth, stable rate.

    It works by maintaining a "next operation earliest allowed timestamp".
    This design fundamentally avoids the "catch-up burst" cycle, because it doesn't care how many operations were missed in the past,
    only caring about when the next operation should be executed, thus achieving truly smooth rate control.
    """
    
    def __init__(self, target_tps: int, db_id: str):
        """
        Initialize the pacing smoother.
        
        Args:
            target_tps: Target operations per second.
            db_id: Database ID, used for logging.
        """
        self.db_id = db_id
        self.logger = logging.getLogger(f"TpsController.{self.db_id}")
        self.perf_counter = PerfCounter.instance()
        self.next_allowed_op_time = self.perf_counter.perf_counter()
        self.target_tps = 0
        self.operation_interval = float('inf')
        
        # Add statistics tracking
        self._start_time = None
        self._operations_count = 0
        self._total_latency_us = 0.0
        self._errors_count = 0
        self._last_reset_time = self.perf_counter.perf_counter()
        
        self.update_target_tps(target_tps) # Use setter to initialize correctly
        self.logger.info(f"Initialized with Target TPS: {self.target_tps}, Interval: {self.operation_interval*1000:.3f} ms")

    def update_target_tps(self, new_target_tps: int):
        """
        Dynamically and safely update target TPS.

        Args:
            new_target_tps: New target operations per second.
        """
        if new_target_tps == self.target_tps:
            return

        if new_target_tps <= 0:
            self.operation_interval = float('inf')
        else:
            self.operation_interval = 1.0 / new_target_tps
        
        self.target_tps = new_target_tps
        # Reset next operation time to current to immediately apply new rate
        self.next_allowed_op_time = self.perf_counter.perf_counter()
        self.logger.info(f"Target TPS updated to: {self.target_tps}, New Interval: {self.operation_interval*1000:.3f} ms")

    def should_wait(self) -> bool:
        """
        Check if we need to wait to maintain smooth rate.
        
        Returns:
            True if current time is earlier than next allowed operation time.
        """
        if self.target_tps <= 0:
            return True # If TPS is 0, always wait.
        
        return self.perf_counter.perf_counter() < self.next_allowed_op_time
    
    def get_remaining_wait_time(self) -> float:
        """
        Get the remaining wait time (seconds).
        
        Returns:
            Remaining wait time, returns 0.0 if no need to wait
        """
        if self.target_tps <= 0:
            return 0.001  # Default wait 1ms
        
        remaining = self.next_allowed_op_time - self.perf_counter.perf_counter()
        return max(0.0, remaining)

    def record_completion(self, latency_us: float):
        """
        Called after each successful operation to update next allowed operation time.
        """
        now = self.perf_counter.perf_counter()
        
        # Statistics tracking
        if self._start_time is None:
            self._start_time = now
        
        self._operations_count += 1
        self._total_latency_us += latency_us
        
        # Core logic:
        # Use max(now, self.next_allowed_op_time) as the baseline.
        # - If system is running normally (now < next_allowed_op_time), next operation time will advance based on preset pace (next_allowed_op_time).
        # - If system stalls (now > next_allowed_op_time), next operation time will advance based on current time (now).
        #   This prevents the system from trying to immediately execute all "missed" operations after stall ends, causing uncontrolled burst.
        base_time = max(now, self.next_allowed_op_time)
        self.next_allowed_op_time = base_time + self.operation_interval

    def reset(self):
        """Reset controller state for use at the start of a new experiment phase."""
        now = self.perf_counter.perf_counter()
        self.next_allowed_op_time = now
        
        # Reset statistics
        self._start_time = None
        self._operations_count = 0
        self._total_latency_us = 0.0
        self._errors_count = 0
        self._last_reset_time = now
        
        self.logger.debug("Controller state has been reset.")

    def __str__(self) -> str:
        """Return string representation of controller state."""
        return f"PreciseTpsController(target={self.target_tps}, interval={self.operation_interval*1000:.3f}ms)"

    def get_current_stats(self) -> TpsStats:
        """Get current statistics"""
        now = self.perf_counter.perf_counter()
        
        if self._start_time is None or self._operations_count == 0:
            return TpsStats(target_tps=self.target_tps)
        
        elapsed_time = now - self._start_time
        actual_tps = self._operations_count / elapsed_time if elapsed_time > 0 else 0.0
        accuracy_percentage = (actual_tps / self.target_tps * 100) if self.target_tps > 0 else 0.0
        
        return TpsStats(
            executed_operations=self._operations_count,
            total_time=elapsed_time,
            errors=self._errors_count,
            actual_tps=actual_tps,
            target_tps=self.target_tps,
            accuracy_percentage=accuracy_percentage
        )
            
    def get_actual_tps(self) -> float:
        """Get actual TPS"""
        stats = self.get_current_stats()
        return stats.actual_tps
        
    def get_accuracy_percentage(self) -> float:
        """Get TPS accuracy percentage"""
        stats = self.get_current_stats()
        return stats.accuracy_percentage
        
    def record_error(self):
        """Record operation error"""
        self._errors_count += 1

    def enable_debug(self, enabled: bool = True) -> None:
        """Enable or disable debug output"""
        if enabled:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        
    def analyze_timing_precision(self) -> Dict[str, Any]:
        """
        Analyze timing control precision
        
        Returns:
            Dictionary containing precision analysis results
        """
        stats = self.get_current_stats()
        
        return {
            "target_tps": stats.target_tps,
            "actual_tps": stats.actual_tps,
            "accuracy_percentage": stats.accuracy_percentage,
            "total_operations": stats.executed_operations,
            "total_errors": stats.errors,
            "avg_latency_us": self._total_latency_us / max(self._operations_count, 1)
        }
        
    def _generate_periodic_report(self, current_time: float) -> None:
        """Generate periodic TPS status report"""
        if self.logger.isEnabledFor(logging.DEBUG):
            stats = self.get_current_stats()
            self.logger.debug(
                f"TPS Report - Target: {stats.target_tps}, "
                f"Actual: {stats.actual_tps:.2f}, "
                f"Accuracy: {stats.accuracy_percentage:.1f}%, "
                f"Operations: {stats.executed_operations}, "
                f"Errors: {stats.errors}"
            )


class MultipleTpsController:
    """
    Multiple database TPS controller
    
    Manages TPS controllers for multiple database instances, providing unified interface and aggregate statistics.
    """
    
    def __init__(self):
        self.controllers: Dict[str, PreciseTpsController] = {}
        self._lock = threading.Lock()
        
    def add_controller(self, db_id: str, target_tps: int) -> PreciseTpsController:
        """
        Add database TPS controller
        
        Args:
            db_id: Database identifier
            target_tps: Target TPS
            
        Returns:
            Created TPS controller instance
        """
        with self._lock:
            controller = PreciseTpsController(target_tps, db_id)
            self.controllers[db_id] = controller
            return controller
            
    def get_controller(self, db_id: str) -> Optional[PreciseTpsController]:
        """Get TPS controller for specified database"""
        return self.controllers.get(db_id)
        
    def get_total_target_tps(self) -> int:
        """Get total target TPS for all databases"""
        return sum(controller.target_tps for controller in self.controllers.values())
        
    def get_total_actual_tps(self) -> float:
        """Get total actual TPS for all databases"""
        return sum(controller.get_actual_tps() for controller in self.controllers.values())
        
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics"""
        total_target_tps = self.get_total_target_tps()
        total_actual_tps = self.get_total_actual_tps()
        
        db_stats = {}
        total_operations = 0
        total_errors = 0
        
        for db_id, controller in self.controllers.items():
            stats = controller.get_current_stats()
            db_stats[db_id] = {
                "target_tps": stats.target_tps,
                "actual_tps": stats.actual_tps,
                "accuracy_percentage": stats.accuracy_percentage,
                "operations": stats.executed_operations,
                "errors": stats.errors
            }
            total_operations += stats.executed_operations
            total_errors += stats.errors
            
        overall_accuracy = (total_actual_tps / total_target_tps * 100) if total_target_tps > 0 else 0
        
        return {
            "total_target_tps": total_target_tps,
            "total_actual_tps": total_actual_tps,
            "overall_accuracy_percentage": overall_accuracy,
            "total_operations": total_operations,
            "total_errors": total_errors,
            "database_stats": db_stats
        }
        
    def enable_debug_all(self, enabled: bool = True) -> None:
        """Enable or disable debug for all controllers"""
        for controller in self.controllers.values():
            controller.enable_debug(enabled)
            
    def reset_all(self) -> None:
        """Reset all controllers"""
        for controller in self.controllers.values():
            controller.reset() 