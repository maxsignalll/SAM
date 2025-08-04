"""
S0 Strategy - Core Strategy with Active Set Mechanism (Performance Optimized V3)

Version: S0CoreStrategyActiveSetV2OptimizedV3
Features:
1. Removed all logging operations to improve performance
2. Optimized non-active set handling to solve non-linear performance growth in 80DB scenarios
3. Optimized knee detection algorithm to reduce NumPy operation overhead
"""

import logging
import os
import time
import json
from typing import Any, Dict, List, Tuple

from ..base_strategy import BaseStrategy
from .s0_gradient_allocator import S0GradientAllocator
from .s0_epoch_manager_optimized import S0EpochManagerOptimized as S0EpochManager
from .s0_active_set_optimized import S0ActiveSetManagerOptimized, DatabaseState
from .s0_core_strategy_simple import S0CoreStrategySimple


class S0CoreStrategyActiveSetV2OptimizedV3(S0CoreStrategySimple):
    """S0 strategy implementation with active set mechanism (Performance Optimized V3)"""
    
    def __init__(self, orchestrator: Any, strategy_name: str, strategy_specific_config: Dict = None):
        # Call parent class initialization
        super().__init__(orchestrator, strategy_name, strategy_specific_config)
        
        # Create optimized active set manager
        self.active_set_manager = S0ActiveSetManagerOptimized(self.hyperparams, self.logger)
        
        # Active set related state
        self.current_state = "global_scan"  # Initial state should be global scan
        self.cycles_since_last_scan = 0
        self.active_set_converged = False
        
        # Database state cache
        self.db_states_cache: Dict[str, DatabaseState] = {}
        
        # Cache latest metrics
        self._last_metrics: Dict[str, Any] = {}
        
        # Performance mode flag
        self.performance_mode = self.hyperparams.get('performance_mode', True)
        
        # Performance optimization: cache non-active set information
        self._non_active_cache = {
            'pages': 0,
            'db_ids': set(),
            'needs_update': True
        }
        
        # Performance optimization: pre-allocate dictionary to avoid frequent creation
        self._allocation_buffer = {}
        
        # CPU timing collection for scalability testing
        self.cpu_timing_config = strategy_specific_config.get('strategy_config', {}).get('scalability_timing', {})
        self.timing_enabled = self.cpu_timing_config.get('enabled', False)
        self.timing_high_precision = self.cpu_timing_config.get('high_precision_timing', True)
        self.timing_export_raw = self.cpu_timing_config.get('export_raw_timings', True)
        self.timing_sample_every_nth = self.cpu_timing_config.get('sample_every_nth_decision', 1)
        
        # CPU timing data collection
        self.cpu_timing_data = {
            'raw_cpu_times_seconds': [],
            'decision_count': 0,
            'database_count': len(orchestrator.db_current_page_allocations),
            'strategy_name': strategy_name,
            'timing_config': self.cpu_timing_config.copy() if self.timing_enabled else None
        }
        
        if self.timing_enabled:
            self.logger.info(f"CPU timing collection enabled: sampling every {self.timing_sample_every_nth} decisions")
            
        # Export file path for timing data
        if self.timing_enabled and self.timing_export_raw:
            timing_output_file = self.cpu_timing_config.get('timing_output_file', 'scalability_cpu_timings.json')
            self.timing_export_path = os.path.join(orchestrator.output_directory, timing_output_file)
        
        
        # Create detailed debug log file (but not used in performance mode)
        if not self.performance_mode:
            log_dir = os.path.join(orchestrator.output_directory, "active_set_debug")
            os.makedirs(log_dir, exist_ok=True)
            self.debug_log_path = os.path.join(log_dir, "active_set_debug.log")
            
            # Initialize debug log
            with open(self.debug_log_path, 'w') as f:
                f.write(f"=== S0 Active Set Strategy Debug Log ===\n")
                f.write(f"Strategy Name: {strategy_name}\n")
                f.write(f"Initialization Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Cache Pages: {self.total_pages}\n")
                f.write(f"Database Count: {len(orchestrator.db_current_page_allocations)}\n")
                f.write(f"Active Set Parameters: k_max={self.hyperparams.get('k_max', 20)}\n")
                f.write("="*50 + "\n\n")
        
        # Comment out log output to improve performance
        # self.logger.info("S0 active set strategy initialization completed")
    
    def _write_debug_log(self, message: str):
        """Write debug log (skip in performance mode)"""
        if not self.performance_mode:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            with open(self.debug_log_path, 'a') as f:
                f.write(f"[{timestamp}] {message}\n")
    
    def update_allocations(self, current_metrics: Dict[str, Any], elapsed_in_phase: float) -> Dict[str, int]:
        """Update cache allocation (integrated active set mechanism)"""
        # Start CPU timing if enabled
        cpu_start_time = None
        if self.timing_enabled:
            cpu_start_time = time.perf_counter() if self.timing_high_precision else time.time()
        
        # Cache current metrics for global scan use
        self._last_metrics = current_metrics.copy()
        
        # Performance mode: comment out all debug logs
        # self._write_debug_log(f"\n{'='*80}")
        # self._write_debug_log(f"update_allocations started - elapsed time: {elapsed_in_phase:.1f}s")
        # self._write_debug_log(f"Current state: {self.current_state}")
        # self._write_debug_log(f"Active set size: {len(self.active_set_manager.active_set) if self.active_set_manager.active_set else 0}")
        # if self.active_set_manager.active_set:
        #     self._write_debug_log(f"Active set members: {list(self.active_set_manager.active_set)}")
        
        # 1. Determine calculation scope based on state
        if self.current_state == "local_optimization" and self.active_set_manager.active_set:
            # Local optimization: only update EMA and factors for databases in active set
            # Note: we still need to update EMA for all databases to maintain continuity
            # but only calculate H and V factors for active set
            
            # Update EMA for all databases (maintain data continuity)
            self._update_ema_metrics(current_metrics)
            
            # Only calculate H and V factors for active set
            active_metrics = {db_id: current_metrics[db_id] 
                            for db_id in self.active_set_manager.active_set 
                            if db_id in current_metrics}
            
            h_factors = self._calculate_h_factors_only(active_metrics)
            
            v_factors = self._calculate_v_factors(active_metrics)
            
            # Performance mode: comment out logs
            # self.logger.info(f"Active set H factors: {h_factors}")
            # self.logger.info(f"Active set V factors: {v_factors}")
            # self._write_debug_log(f"Local optimization - calculated H/V factors for {len(h_factors)} active set members")
            # for db_id in sorted(h_factors.keys()):
            #     self._write_debug_log(f"  {db_id}: H={h_factors[db_id]:.4f}, V={v_factors.get(db_id, 0.0):.4f}")
        else:
            # Global scan: calculate all databases
                h_factors = self._update_ema_and_calculate_h_factors(current_metrics)
            
                v_factors = self._calculate_v_factors(current_metrics)
            
            # Performance mode: comment out logs
            # self.logger.info(f"H factors: {h_factors}")
            # self.logger.info(f"V factors: {v_factors}")
            # self._write_debug_log(f"Global calculation - calculated H/V factors for {len(h_factors)} databases")
            # # Only record top 10 and bottom 10
            # sorted_by_h = sorted(h_factors.items(), key=lambda x: x[1], reverse=True)
            # self._write_debug_log("  Top 10 by H factor:")
            # for db_id, h in sorted_by_h[:10]:
            #     self._write_debug_log(f"    {db_id}: H={h:.4f}, V={v_factors.get(db_id, 0.0):.4f}")
            # self._write_debug_log("  Bottom 10 by H factor:")
            # for db_id, h in sorted_by_h[-10:]:
            #     self._write_debug_log(f"    {db_id}: H={h:.4f}, V={v_factors.get(db_id, 0.0):.4f}")
        
        # 2. Build database state
        # Only update active set state during local optimization
        if self.current_state == "local_optimization" and self.active_set_manager.active_set:
            # Only update database state within active set
            for db_id in self.active_set_manager.active_set:
                if db_id in h_factors:
                    self._update_single_db_state(db_id, h_factors[db_id], v_factors.get(db_id, 0.0))
        else:
            # Update all database states during global scan
            self._update_db_states(h_factors, v_factors)
        
        # 3. State machine logic
        # Performance mode: comment out logs
        # self.logger.info(f"Current state: {self.current_state}, active set converged: {self.active_set_converged}, "
        #                 f"active set size: {len(self.active_set_manager.active_set)}")
        
        if self.current_state == "local_optimization":
            # Local optimization state
            if self.active_set_converged or not self.active_set_manager.active_set:
                # Active set converged or empty, switch to global scan
                # self.logger.info("Active set converged or empty, switching to global scan state")
                # self._write_debug_log(f"\nState transition: local_optimization -> global_scan")
                # self._write_debug_log(f"Reason: {'active set converged' if self.active_set_converged else 'active set empty'}")
                self.current_state = "global_scan"
                self.active_set_converged = False
                # Mark non-active set cache needs update
                self._non_active_cache['needs_update'] = True
                # Important: immediately trigger a global state update to ensure all database states are up-to-date on next call
                # Achieved by returning current allocation and waiting for next cycle
                return self.orchestrator.db_current_page_allocations.copy()
            else:
                # Continue local optimization
                new_allocations, converged = self._do_local_optimization()
                self.active_set_converged = converged
                # self.logger.info(f"Local optimization completed, convergence status: {converged}")
                
                
                # Update allocation state
                self._update_allocation_state(new_allocations)
                
                return new_allocations
        
        if self.current_state == "global_scan":
            # Global scan state
            new_active_set = self._do_global_scan()
            
            if new_active_set:
                # Found new active set, switch to local optimization
                # self.logger.info(f"Selected new active set with {len(new_active_set)} databases")
                # self._write_debug_log(f"\nState transition: global_scan -> local_optimization")
                # self._write_debug_log(f"New active set size: {len(new_active_set)}")
                self.current_state = "local_optimization"
                self.cycles_since_last_scan = 0
                # Mark non-active set cache needs update
                self._non_active_cache['needs_update'] = True
                
                # Immediately perform local optimization using new active set
                new_allocations, _ = self._do_local_optimization(new_active_set)
            else:
                # System balanced, maintain current allocation
                # self.logger.info("System in balanced state, maintaining current allocation")
                # self._write_debug_log(f"\nSystem balanced, no active set selected")
                new_allocations = self.orchestrator.db_current_page_allocations.copy()
                
                # Set longer wait period before next scan
                self.cycles_since_last_scan = -5  # Wait 5 cycles before next scan
                self.current_state = "local_optimization"
        
        
        # Update allocation state
        self._update_allocation_state(new_allocations)
        
        # Update cycle count
        self.cycles_since_last_scan += 1
        
        # Record H and V factors for plotting
        self._last_h_factors = h_factors
        self._last_v_factors = v_factors
        self._last_decision_v_factors = v_factors
        
        # End CPU timing and record if enabled
        if self.timing_enabled and cpu_start_time is not None:
            cpu_end_time = time.perf_counter() if self.timing_high_precision else time.time()
            cpu_time_seconds = cpu_end_time - cpu_start_time
            
            self.cpu_timing_data['decision_count'] += 1
            
            # Sample every nth decision
            if self.cpu_timing_data['decision_count'] % self.timing_sample_every_nth == 0:
                self.cpu_timing_data['raw_cpu_times_seconds'].append(cpu_time_seconds)
                
                # Export timing data if configured
                if (self.timing_export_raw and 
                    hasattr(self, 'timing_export_path') and 
                    len(self.cpu_timing_data['raw_cpu_times_seconds']) % 50 == 0):  # Export every 50 samples
                    self._export_timing_data()
        
        return new_allocations
    
    def _update_db_states(self, h_factors: Dict[str, float], v_factors: Dict[str, float]):
        """Update database state cache"""
        for db_id in h_factors:
            self._update_single_db_state(db_id, h_factors[db_id], v_factors.get(db_id, 0.0))
    
    def _update_single_db_state(self, db_id: str, h_factor: float, v_factor: float):
        """Update single database state"""
        state = self.ema_states.get(db_id, {})
        
        # Validate H factor range
        if h_factor < 0 or h_factor > 1.0:
            # Performance mode: comment out warning logs
            # self.logger.warning(f"DB[{db_id}] H factor out of range: {h_factor:.4f}")
            # Force limit to reasonable range
            h_factor = max(0.0, min(1.0, h_factor))
        
        # Use active set manager to calculate FinalScore
        hr_current = state.get('ema_hr_slow', 0.0)
        hr_base = state.get('hr_base', 0.5)
        hr_max = state.get('hr_max_adaptive', 1.0)
        
        final_score = self.active_set_manager.calculate_final_score(
            db_id, 
            h_factor, 
            v_factor,
            hr_current,
            hr_base,
            hr_max
        )
        
        # Get current operation count (for Activity-Weighted Convergence)
        current_ops = state.get('ema_ops_slow', 0.0)
        
        # Performance mode: comment out detailed logs
        # if db_id in ['db_high_priority', 'db_medium_priority'] or db_id.startswith('db_bg_77'):
        #     self._write_debug_log(f"  {db_id} score calculation details:")
        #     self._write_debug_log(f"    H factor={h_factor:.4f}, V factor={v_factor:.4f}")
        #     self._write_debug_log(f"    hr_current={hr_current:.3f}, hr_base={hr_base:.3f}, hr_max={hr_max:.3f}")
        #     self._write_debug_log(f"    current_ops={current_ops:.1f}")
        #     self._write_debug_log(f"    final_score={final_score:.4f}")
        
        # Create or update database state
        if db_id not in self.db_states_cache:
            self.db_states_cache[db_id] = DatabaseState(
                db_id=db_id,
                h_factor=h_factor,
                v_factor=v_factor,
                final_score=final_score,
                saturation_confidence=self.active_set_manager.saturation_confidences.get(db_id, 0.0),
                current_allocation=self.orchestrator.db_current_page_allocations.get(db_id, 0),
                min_allocation=self.s0_fixed_allocations.get(db_id, 0),
                current_ops=current_ops
            )
        else:
            # Update existing state
            db_state = self.db_states_cache[db_id]
            db_state.h_factor = h_factor
            db_state.v_factor = v_factor
            db_state.final_score = final_score
            db_state.saturation_confidence = self.active_set_manager.saturation_confidences.get(db_id, 0.0)
            db_state.current_allocation = self.orchestrator.db_current_page_allocations.get(db_id, 0)
            db_state.current_ops = current_ops
    
    def _do_global_scan(self) -> set:
        """Execute global scan"""
        # Performance mode: comment out logs
        # self.logger.info("Execute global scan, select new active set")
        
        # Before selecting active set, ensure all database states are up-to-date
        # Get current metrics
        current_metrics = {}
        for db_id in self.orchestrator.db_current_page_allocations:
            if hasattr(self.orchestrator, 'get_latest_metrics'):
                metrics = self.orchestrator.get_latest_metrics(db_id)
            else:
                # If orchestrator doesn't have get_latest_metrics method, use cached metrics
                metrics = self._last_metrics.get(db_id, {})
            if metrics:
                current_metrics[db_id] = metrics
        
        # Update H and V factors for all databases
        # self.logger.info("Global scan: update H/V factors for all databases")
        h_factors = self._update_ema_and_calculate_h_factors(current_metrics)
        v_factors = self._calculate_v_factors(current_metrics)
        
        # Update all database states
        self._update_db_states(h_factors, v_factors)
        
        # Performance mode: comment out logs
        # self.logger.info(f"Global scan H factors: {h_factors}")
        # self.logger.info(f"Global scan V factors: {v_factors}")
        
        # Performance mode: comment out detailed score logging
        # self._write_debug_log("\nGlobal scan - final scores for all databases:")
        # all_scores = [(db_id, state.final_score) for db_id, state in self.db_states_cache.items()]
        # all_scores.sort(key=lambda x: x[1], reverse=True)
        # for i, (db_id, score) in enumerate(all_scores[:20]):
        #     self._write_debug_log(f"  #{i+1}: {db_id} - score={score:.4f}")
        # self._write_debug_log(f"  ... ({len(all_scores)-20} databases omitted)")
        
        # Use active set manager to select new active set
        # self._write_debug_log("\nStarting active set selection...")
        new_active_set = self.active_set_manager.select_active_set(self.db_states_cache)
        
        # self._write_debug_log(f"Selected {len(new_active_set)} databases for active set:")
        # self._write_debug_log(f"Active set members: {list(new_active_set)}")
        
        return new_active_set
    
    def _do_local_optimization(self, active_set: set = None) -> Tuple[Dict[str, int], bool]:
        """Execute local optimization (performance optimized version)"""
        # If no active set specified, use current active set
        if active_set is None:
            active_set = self.active_set_manager.active_set
        
        # Performance optimization: use cached non-active set information
        if self._non_active_cache['needs_update'] or active_set != self.active_set_manager.active_set:
            # Update non-active set cache
            self._update_non_active_cache(active_set)
            
        available_pages_for_active_set = self.total_pages - self._non_active_cache['pages']
        
        # Performance mode: comment out logs
        # self.logger.info(f"Pages available for active set: {available_pages_for_active_set} (total pages {self.total_pages} - non-active set occupancy {non_active_pages})")
        
        # Use active set manager for optimization, pass in pages available for active set
        # self._write_debug_log(f"\nLocal optimization started:")
        # self._write_debug_log(f"Active set members: {list(active_set)}")
        # self._write_debug_log(f"Pages available for active set: {available_pages_for_active_set}")
        
        # Performance mode: comment out member state logging
        # for db_id in active_set:
        #     if db_id in self.db_states_cache:
        #         state = self.db_states_cache[db_id]
        #         self._write_debug_log(f"  {db_id}: current={state.current_allocation} pages, score={state.final_score:.4f}")
        
        new_allocations, converged = self.active_set_manager.optimize_in_active_set(
            active_set,
            self.db_states_cache,
            self.gradient_allocator,
            available_pages_for_active_set  # Pass in pages actually available for active set
        )
        
        # Performance mode: comment out allocation change logs
        # self._write_debug_log(f"\nAfter gradient allocation:")
        # for db_id, new_alloc in new_allocations.items():
        #     old_alloc = self.db_states_cache[db_id].current_allocation if db_id in self.db_states_cache else 0
        #     change = new_alloc - old_alloc
        #     self._write_debug_log(f"  {db_id}: {old_alloc} -> {new_alloc} pages (change: {change:+d})")
        
        # Record convergence status and active set OPS
        # if converged:
        #     total_active_ops = sum(self.db_states_cache[db_id].current_ops 
        #                          for db_id in active_set 
        #                          if db_id in self.db_states_cache)
        #     self._write_debug_log(f"\nActive set convergence detection: converged, total active set OPS={total_active_ops:.0f}")
        
        # Apply intelligent total adjustment using pages available for active set
        adjusted_allocations = self.active_set_manager.intelligent_adjust_total(
            new_allocations,
            self.db_states_cache,
            available_pages_for_active_set  # Use pages available for active set, not total system pages
        )
        
        # Performance optimization: use pre-allocated buffer for final allocation calculation
        self._allocation_buffer.clear()
        
        # Add non-active set allocations (from cache)
        current_allocations = self.orchestrator.db_current_page_allocations
        for db_id in self._non_active_cache['db_ids']:
            self._allocation_buffer[db_id] = current_allocations[db_id]
        
        # Add new allocations for active set
        for db_id, alloc in adjusted_allocations.items():
            self._allocation_buffer[db_id] = alloc
        
        # If active set needs more cache, reclaim from non-active set
        active_set_need = sum(adjusted_allocations.values())
        if active_set_need > available_pages_for_active_set:
            shortage = active_set_need - available_pages_for_active_set
            # self.logger.info(f"Active set needs additional {shortage} pages, trying to reclaim from non-active set")
            
            # Performance optimization: directly use cached non-active set information for reclamation
            self._reclaim_from_non_active(shortage, active_set)
        
        # Check if total exceeds limit
        total_allocated = sum(self._allocation_buffer.values())
        if total_allocated > self.total_pages:
            # self.logger.warning(f"Total allocation {total_allocated} pages exceeds system limit {self.total_pages} pages, adjustment needed")
            # Force adjustment to meet total page limit
            self._force_adjust_to_limit_optimized()
        
        # Return result (create new dictionary to maintain interface compatibility)
        return dict(self._allocation_buffer), converged
    
    def _update_non_active_cache(self, active_set: set):
        """Update non-active set cache"""
        self._non_active_cache['db_ids'].clear()
        self._non_active_cache['pages'] = 0
        
        current_allocations = self.orchestrator.db_current_page_allocations
        for db_id, alloc in current_allocations.items():
            if db_id not in active_set:
                self._non_active_cache['db_ids'].add(db_id)
                self._non_active_cache['pages'] += alloc
        
        self._non_active_cache['needs_update'] = False
    
    def _reclaim_from_non_active(self, shortage: int, active_set: set):
        """Reclaim cache from non-active set (optimized version)"""
        # Build non-active database list (only include those with reclaimable cache)
        non_active_dbs = []
        for db_id in self._non_active_cache['db_ids']:
            state = self.db_states_cache.get(db_id)
            if state:
                current_alloc = self._allocation_buffer[db_id]
                min_alloc = state.min_allocation
                available = current_alloc - min_alloc
                if available > 0:
                    non_active_dbs.append((state.final_score, db_id, available, min_alloc))
        
        if not non_active_dbs:
            return
        
        # Sort by score ascending (reclaim from low scores first)
        non_active_dbs.sort()
        
        # Reclaim cache
        reclaimed = 0
        for score, db_id, available, min_alloc in non_active_dbs:
            if reclaimed >= shortage:
                break
                
            reclaim_amount = min(available, shortage - reclaimed)
            self._allocation_buffer[db_id] -= reclaim_amount
            reclaimed += reclaim_amount
            # self.logger.info(f"Reclaimed {reclaim_amount} pages from {db_id} (score={score:.3f})")
        
        # if reclaimed < shortage:
        #     self.logger.warning(f"Only reclaimed {reclaimed} pages, still short {shortage - reclaimed} pages")
    
    def _force_adjust_to_limit_optimized(self):
        """Force adjust allocation to meet system total page limit (optimized version)"""
        total = sum(self._allocation_buffer.values())
        if total <= self.total_pages:
            return
        
        # Calculate pages to be reduced
        excess = total - self.total_pages
        
        # Build database list (including scores and reducible amounts)
        db_list = []
        for db_id, alloc in self._allocation_buffer.items():
            state = self.db_states_cache.get(db_id)
            if state:
                min_alloc = state.min_allocation
                available = alloc - min_alloc
                if available > 0:
                    db_list.append((state.final_score, db_id, available, min_alloc))
        
        # Sort by score ascending (reduce from low scores first)
        db_list.sort()
        
        # Reduce cache
        for score, db_id, available, min_alloc in db_list:
            if excess <= 0:
                break
                
            reduction = min(available, excess)
            self._allocation_buffer[db_id] -= reduction
            excess -= reduction
        
        # if excess > 0:
        #     self.logger.error(f"Cannot meet total page limit, still exceeds by {excess} pages")
    
    
    def _calculate_h_factors_only(self, metrics_subset: Dict[str, Any]) -> Dict[str, float]:
        """Calculate H factors only (for active set calculations)"""
        h_factors = {}
        
        # Collect data within active set for normalization
        ops_values = []
        
        for db_id, metrics in metrics_subset.items():
            state = self.ema_states.get(db_id, {})
            ops = state.get('ema_ops_slow', 0.0)
            ops_values.append(ops)
        
        # Calculate normalization parameters
        ops_min = min(ops_values) if ops_values else 0.0
        ops_max = max(ops_values) if ops_values else 1.0
        
        # Prevent normalization explosion from too small ops_max
        # Use global perspective minimum range to avoid normalization issues from local active set
        min_ops_range = 10.0  # Minimum operation count range
        if ops_max - ops_min < min_ops_range:
            # Extend range while keeping center point unchanged
            center = (ops_max + ops_min) / 2
            ops_max = center + min_ops_range / 2
            ops_min = center - min_ops_range / 2
            ops_min = max(0.0, ops_min)  # Ensure non-negative
            
            # Performance mode: comment out debug logs
            # self.logger.debug(f"Active set ops range too small, extended to [{ops_min:.2f}, {ops_max:.2f}]")
        
        # Calculate H factors
        for db_id, metrics in metrics_subset.items():
            state = self.ema_states.get(db_id, {})
            ops = state.get('ema_ops_slow', 0.0)
            hr = state.get('ema_hr_slow', 0.0)
            hr_base = state.get('hr_base', self.hyperparams.get('hr_base', 0.5))
            hr_max = state.get('hr_max_adaptive', 1.0)
            
            # Normalize operation count
            if ops_max > ops_min + 1e-6:
                ops_normalized = (ops - ops_min) / (ops_max - ops_min)
            else:
                ops_normalized = 0.5  # When all databases have same load
            
            # Calculate normalized hit rate
            if hr < hr_base:
                # Below baseline, linear mapping to [0, 0.5]
                hr_norm = 0.5 * (hr / hr_base) if hr_base > 0 else 0.0
            else:
                # Above baseline, mapping to [0.5, 1.0]
                if hr_max > hr_base:
                    hr_norm = 0.5 + 0.5 * (hr - hr_base) / (hr_max - hr_base)
                else:
                    hr_norm = 0.5  # If hr_max not yet updated, at least give 0.5
            
            hr_norm = max(0.0, min(1.0, hr_norm))
            
            # H factor = normalized operation count × normalized hit rate
            h_factor = ops_normalized * hr_norm
            
            # Finally ensure H factor is within [0, 1] range
            h_factor = max(0.0, min(1.0, h_factor))
            h_factors[db_id] = h_factor
            
            # Debug information
            # if h_factor > 1.0 or h_factor < 0.0:
            #     self.logger.warning(f"DB[{db_id}] H factor out of range: {h_factor:.4f}")
        
        return h_factors
    
    def get_strategy_specific_state(self, db_id: str) -> Dict[str, Any]:
        """Get strategy-specific state for a specific database"""
        base_state = super().get_strategy_specific_state(db_id)
        
        # Add active set related information
        if db_id in self.db_states_cache:
            db_state = self.db_states_cache[db_id]
            base_state['final_score'] = db_state.final_score
            base_state['saturation_confidence'] = db_state.saturation_confidence
            base_state['is_in_active_set'] = db_id in self.active_set_manager.active_set
        
        return base_state
    
    def _export_timing_data(self):
        """Export CPU timing data to JSON file"""
        if not self.timing_enabled or not hasattr(self, 'timing_export_path'):
            return
        
        try:
            # Update metadata
            self.cpu_timing_data['total_samples'] = len(self.cpu_timing_data['raw_cpu_times_seconds'])
            self.cpu_timing_data['export_timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
            
            # Calculate basic statistics
            if self.cpu_timing_data['raw_cpu_times_seconds']:
                times = self.cpu_timing_data['raw_cpu_times_seconds']
                self.cpu_timing_data['statistics'] = {
                    'mean_seconds': sum(times) / len(times),
                    'min_seconds': min(times),
                    'max_seconds': max(times),
                    'total_decisions': self.cpu_timing_data['decision_count']
                }
            
            # Export to JSON
            os.makedirs(os.path.dirname(self.timing_export_path), exist_ok=True)
            with open(self.timing_export_path, 'w') as f:
                json.dump(self.cpu_timing_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to export timing data: {e}")
    
    def finalize_strategy(self):
        """Finalize strategy and export any remaining data"""
        # Call parent finalization if it exists
        if hasattr(super(), 'finalize_strategy'):
            super().finalize_strategy()
        
        # Final export of timing data
        if self.timing_enabled:
            self._export_timing_data()
            if hasattr(self, 'timing_export_path'):
                self.logger.info(f"CPU timing data exported to: {self.timing_export_path}")
                total_samples = len(self.cpu_timing_data['raw_cpu_times_seconds'])
                total_decisions = self.cpu_timing_data['decision_count']
                self.logger.info(f"Total timing samples: {total_samples}/{total_decisions} decisions")