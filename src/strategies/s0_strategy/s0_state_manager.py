"""
S0 Strategy - State Management and EMA System

Responsible for state management and metrics updates in the S0 strategy, including:
- Database state initialization and maintenance
- EMA metrics updates
- Historical data management
- Metrics update and synchronization
"""

from typing import Any, Dict


class S0StateManager:
    """State manager for S0 strategy, responsible for database state and metrics updates"""
    
    def __init__(self, hyperparams: Dict[str, Any], logger, orchestrator):
        self.hyperparams = hyperparams
        self.logger = logger
        self.orchestrator = orchestrator
        self._log_counter = 0
        
    def initialize_strategy_states(self, db_instance_configs: list):
        """Initialize strategy states for all databases"""
        if not hasattr(self.orchestrator, 'strategy_states'):
            self.orchestrator.strategy_states = {}
        
        for db_conf in db_instance_configs:
            db_id = db_conf['id']
            self.orchestrator.strategy_states[db_id] = {
                'is_initialized': False,
                'ema_hr_slow': 0.0,
                'ema_hr_fast': 0.0,
                'ema_ops_slow': 0.0,
                'ema_ops_fast': 0.0,
                'last_ema_hit_rate': 0.0,
                'last_allocation_pages': None,  # Initialize as None, set after first allocation
                'last_hr_max_adaptive': 0.8, # New: support for V factor repair
                'last_raw_ops': 0,
                'last_raw_hr': 0.0,
                'smart_cooldown_remaining': 0,
                'first_allocation_done': False,
                
                # Old V factor state variables retained for compatibility, but no longer used
                'last_valid_v_factor': 0.0,
                'v_wait_cycles': 1,
                'cycles_since_major_change': 999,  # Initialize as large value, allow immediate V factor calculation
                'hr_base': self.hyperparams['hr_base'],
                'hr_max_adaptive': 0.8,
                'hr_p95_window': [],
                'hr_max_stagnant_counter': 0,
                'v_factor_no_update_counter': 0,
                'cache_change_cycles_since': 0,
                'current_alpha_t': 0.5,
                't_e': 1,
                'epoch_decision_count': 0,
                
                # V factor audit system state
                'audit_state': {
                    'is_auditing': False,           # Whether in audit period
                    'audit_trigger_cycle': None,    # Cycle number when audit triggered
                    'audit_cycles_remaining': 0,    # Remaining audit cycles
                    'micro_v_factors': [],          # Micro V factor sequence [v_1, v_2, ..., v_k]
                    'pre_audit_allocation': None,   # Allocation before audit
                    'pre_audit_hr': None,           # Hit rate before audit
                    'pre_audit_hr_norm': None,      # Normalized hit rate before audit
                    'frozen_allocation': None,      # Allocation frozen during audit
                },
                'v_audit_result': {
                    'direction': 0.0,               # V_direction = Median(E_v)
                    'confidence': 0.0,              # C_V confidence score
                    'last_update_cycle': 0          # Last update cycle number
                },
                
                # Dual-track V factor state
                'dual_v_track': {
                    'v_fast': 0.0,                  # Fast V factor (2-3 points)
                    'v_slow': 0.0,                  # Slow V factor (8-10 points)
                    'allocation_window': [],        # [(cycle, allocation)] Recent allocation history
                    'hr_norm_window': [],           # [(cycle, hr_norm)] Recent normalized hit rate history
                    'last_v_fast_update': 0,        # Last V_fast update cycle
                    'last_v_slow_update': 0,        # Last V_slow update cycle
                },
                
                # Historical data collection - for median calculation
                'allocation_history': [],           # [(cycle, allocation)]
                'hr_history': [],                   # [(cycle, hr_ema_slow)]
                'hr_norm_history': [],              # [(cycle, hr_norm)]
                'hr_max_history': [],               # [(cycle, hr_max_adaptive)]
                'last_snapshot_cycle': 0,           # Last snapshot cycle number
                'current_cycle': 0,                 # Current cycle number
                
                # Hotspot validation mechanism
                'pending_hotspot': False,           # Whether pending hotspot validation
                'hotspot_detected_cycle': 0,        # Cycle when hotspot detected
                'pre_hotspot_score_ratio': 0.0,     # Score ratio when hotspot detected
                'hotspot_audit_triggered': False,   # Whether hotspot-related audit triggered
                
                # Importance-Urgency matrix state
                'importance_urgency': {
                    'importance_score': 0.5,        # Importance score [0, 1]
                    'urgency_score': 0.0,           # Urgency score [0, 1]
                    'quadrant': 'Q4',               # Quadrant Q1-Q4
                    'donor_score': 0.0,             # Donor score
                    'historical_avg_h': 0.0,        # Historical average H factor
                },
                
                # Momentum mechanism state (Phase 1)
                'momentum': {
                    'velocity': 0.0,                # Cache adjustment velocity
                    'last_v_factor': 0.0,           # Last V factor value
                    'last_adjustment': 0,           # Last actual adjustment (pages)
                    'adjustment_history': [],       # Recent adjustment history [(cycle, adjustment)]
                }
            }
            
            if db_id in ['db_high_priority', 'db_medium_priority']:
                self.logger.info(f"🌱 STATE_INIT[{db_id}]: Initial state created - last_allocation_pages=None, last_ema_hit_rate=0.0")
    
    def update_metrics(self, interval_stats: Dict[str, Any]):
        """
        Update metrics for all databases, including EMA and normalized hit rates
        """
        # Get EMA parameters
        alpha_slow = self.hyperparams['alpha_slow']
        alpha_fast = self.hyperparams['alpha_fast']
        
        # Get reporting_interval for ops_per_second calculation
        reporting_interval = self.orchestrator.general_setup["reporting_interval_seconds"]
        
        for db_id, stats in interval_stats.items():
            if db_id not in self.orchestrator.strategy_states:
                raise KeyError(f"Missing strategy state for db_id: {db_id}")
            
            state = self.orchestrator.strategy_states[db_id]
            
            # Data integrity check
            required_fields = ['cache_hits', 'cache_misses', 'ops_count']
            missing_fields = [field for field in required_fields if field not in stats]
            
            if missing_fields:
                self.logger.debug(f"DB[{db_id}] Missing required fields {missing_fields}, skipping metrics update")
                continue
            
            hits = stats['cache_hits']
            misses = stats['cache_misses']
            ops_curr = stats['ops_count']
            total_accesses = hits + misses
            
            if total_accesses == 0:
                # In early experiment stages, some databases may have no access records, which is normal
                self.logger.debug(f"DB[{db_id}] No cache access records (hits={hits}, misses={misses}), skipping EMA update")
                continue
            
            # Calculate current metrics
            hr_curr = hits / total_accesses
            if reporting_interval <= 0:
                raise ValueError("Reporting interval must be positive for ops_per_second calculation.")
            ops_per_second = ops_curr / reporting_interval
            
            # EMA update
            if not state['is_initialized']:
                if db_id in ['db_high_priority', 'db_medium_priority', 'db_low_priority']:
                    self.logger.debug(f"🔄 {db_id}: EMA initialization")
                else:
                    self.logger.debug(f"DB[{db_id}] First metrics update, initializing EMA...")
                state['ema_ops_slow'] = float(ops_per_second)
                state['ema_ops_fast'] = float(ops_per_second)
                state['ema_hr_slow'] = float(hr_curr)
                state['ema_hr_fast'] = float(hr_curr)
                state['is_initialized'] = True
            else:
                # Normal smoothing update
                state['ema_ops_slow'] = alpha_slow * ops_per_second + (1 - alpha_slow) * state['ema_ops_slow']
                state['ema_ops_fast'] = alpha_fast * ops_per_second + (1 - alpha_fast) * state['ema_ops_fast']
                state['ema_hr_slow'] = alpha_slow * hr_curr + (1 - alpha_slow) * state['ema_hr_slow']
                state['ema_hr_fast'] = alpha_fast * hr_curr + (1 - alpha_fast) * state['ema_hr_fast']
            
            # Update current cycle number
            state['current_cycle'] = state.get('current_cycle', 0) + 1
            
            # Reduce log frequency and only show metrics updates for key databases
            self._log_counter += 1
            if self._log_counter % 20 == 0 and db_id in ['db_high_priority', 'db_medium_priority', 'db_low_priority']:
                self.logger.debug(f"📊 {db_id}: OPS(S={state['ema_ops_slow']:.1f}, F={state['ema_ops_fast']:.1f}), HR(S={state['ema_hr_slow']:.3f}, F={state['ema_hr_fast']:.3f})")
            elif self._log_counter % 50 == 0:
                self.logger.debug(f"DB[{db_id}] Metrics Updated: OPS(S={state['ema_ops_slow']:.2f}, F={state['ema_ops_fast']:.2f}), HR(S={state['ema_hr_slow']:.3f}, F={state['ema_hr_fast']:.3f})")
            
            # Save raw metrics
            state['last_raw_ops'] = ops_curr
            state['last_raw_hr'] = hr_curr
    
    def _batch_update_raw_metrics(self, current_metrics: Dict[str, Any], db_ids: list) -> bool:
        """
        Batch update raw metrics and check data completeness
        Returns True if all data is complete, False if missing
        """
        missing_dbs = []
        
        # Get reporting_interval for ops_per_second calculation
        reporting_interval = self.orchestrator.general_setup["reporting_interval_seconds"]
        
        for db_id in db_ids:
            if db_id not in current_metrics:
                raise KeyError(f"Missing metrics for db_id: {db_id}")
                
            if db_id not in self.orchestrator.strategy_states:
                missing_dbs.append(db_id)
                continue
            
            db_metrics = current_metrics[db_id]
            state = self.orchestrator.strategy_states[db_id]
            
            # Data integrity check
            required_fields = ['cache_hits', 'cache_misses', 'ops_count']
            missing_fields = [field for field in required_fields if field not in db_metrics]
            
            if missing_fields:
                raise KeyError(f"DB[{db_id}] Missing required fields: {missing_fields}")
            
            # Calculate current metrics
            hits = db_metrics['cache_hits']
            misses = db_metrics['cache_misses']
            ops_count = db_metrics['ops_count']
            total_accesses = hits + misses
            
            if total_accesses == 0:
                self.logger.debug(f"DB[{db_id}] No cache access records for current cycle")
                missing_dbs.append(db_id)
                continue
            
            # Update raw metrics
            hit_rate = hits / total_accesses
            if reporting_interval <= 0:
                raise ValueError("Reporting interval must be positive for ops_per_second calculation.")
            ops_per_second = ops_count / reporting_interval
            
            # Critical fix: update last_raw_hr and last_raw_ops in state for trigger checks
            state['last_raw_hr'] = hit_rate
            state['last_raw_ops'] = ops_count
            
            self.logger.debug(f"DB[{db_id}] State update: last_raw_hr={hit_rate:.3f}, last_raw_ops={ops_count}")
            
            # Store in period_stats format for use by other modules
            period_stats = {
                'hit_rate': hit_rate,
                'ops_per_second': ops_per_second,
                'ops_count': ops_count,
                'cache_hits': hits,
                'cache_misses': misses
            }
            
            # Add period_stats to current_metrics
            current_metrics[db_id].update(period_stats)
        
        # Check data completeness
        if missing_dbs:
            self.logger.warning(f"⚠️ Data collection incomplete, missing DB instances: {missing_dbs}")
            return False
        
        return True
    
    def _update_previous_state_values(self, db_ids: list, final_allocations: dict = None):
        """
        [Deprecated] This method has been replaced by save_pre_change_snapshot
        Retained for compatibility but performs no operations
        """
        # This method is deprecated, historical state saving is handled by save_pre_change_snapshot at the correct timing
        pass
    
    def _batch_update_allocation_states(self, new_allocations: Dict[str, int], db_ids: list):
        """
        Batch update allocation states, mark first allocation complete
        """
        for db_id in db_ids:
            if db_id not in self.orchestrator.strategy_states:
                raise KeyError(f"Missing strategy state for db_id: {db_id}")
            
            state = self.orchestrator.strategy_states[db_id]
            
            # Mark first allocation complete
            if not state['first_allocation_done']:
                state['first_allocation_done'] = True
                if db_id in ['db_high_priority', 'db_medium_priority']:
                    self.logger.info(f"🎆 FIRST_DONE[{db_id}]: First allocation marked complete (current allocation: {new_allocations.get(db_id, 'Unknown')} pages)")
                else:
                    self.logger.debug(f"DB[{db_id}] First allocation marked complete")
    
    def _detect_strategic_change(self, db_id: str, ops_slow: float, ops_fast: float) -> bool:
        """
        Detect strategic change points
        Determine if significant changes occurred based on difference between slow and fast EMA
        """
        strategic_change_threshold = self.hyperparams['strategic_change_threshold']
        
        if ops_slow <= 0:
            raise ValueError("ops_slow must be positive for strategic change detection.")
        
        # Calculate relative change ratio
        change_ratio = abs(ops_fast - ops_slow) / ops_slow
        
        if change_ratio > strategic_change_threshold:
            if db_id in ['db_high_priority', 'db_medium_priority']:
                self.logger.debug(f"🔄 {db_id}: Strategic change ratio={change_ratio:.3f}")
            else:
                self.logger.debug(f"DB[{db_id}] Strategic change detected: change_ratio={change_ratio:.3f} > threshold={strategic_change_threshold}")
            return True
        
        return False
    
    def _update_applied_score_proportions(self, scores: Dict[str, float], total_score: float, db_ids: list):
        """
        Update applied score proportions for trigger checking
        """
        if total_score <= 0:
            # When all scores are 0, use uniform allocation as fallback
            self.logger.warning("⚠️ Total score of all databases is 0, using uniform allocation as fallback")
            uniform_proportion = 1.0 / len(db_ids) if db_ids else 0.0
            
            for db_id in db_ids:
                if db_id not in self.orchestrator.strategy_states:
                    raise KeyError(f"Missing strategy state for db_id: {db_id}")
                
                state = self.orchestrator.strategy_states[db_id]
                state['applied_score_proportion'] = uniform_proportion
                self.logger.debug(f"DB[{db_id}] Applied uniform score proportion: {uniform_proportion:.4f}")
            return
        
        for db_id in db_ids:
            if db_id not in self.orchestrator.strategy_states:
                raise KeyError(f"Missing strategy state for db_id: {db_id}")
            
            state = self.orchestrator.strategy_states[db_id]
            
            # Calculate and save score proportion
            if db_id in scores:
                proportion = scores[db_id] / total_score
                state['applied_score_proportion'] = proportion
                self.logger.debug(f"DB[{db_id}] Applied score proportion: {proportion:.4f}")
            else:
                raise KeyError(f"Missing score for db_id: {db_id}")
    
    def get_state(self, db_id: str) -> Dict[str, Any]:
        """Get state of specified database"""
        return self.orchestrator.strategy_states[db_id]
    
    def update_state(self, db_id: str, key: str, value: Any):
        """Update state value of specified database"""
        if db_id not in self.orchestrator.strategy_states:
            raise KeyError(f"Strategy state for db_id: {db_id} not found.")
        self.orchestrator.strategy_states[db_id][key] = value
    
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """Get states of all databases"""
        return self.orchestrator.strategy_states
    
    def reset_state(self, db_id: str, key: str, default_value: Any = None):
        """Reset state value of specified database"""
        if db_id not in self.orchestrator.strategy_states:
            raise KeyError(f"Strategy state for db_id: {db_id} not found.")
        self.orchestrator.strategy_states[db_id][key] = default_value
    
    def save_pre_change_snapshot(self, db_id: str, current_allocation: int):
        """
        Save current state as historical snapshot before allocation change
        Use median of historical data from last snapshot to now as baseline
        These values will be used for V factor calculation: V = (HR_now - HR_snapshot) / (Alloc_now - Alloc_snapshot)
        
        Args:
            db_id: Database ID
            current_allocation: Current (pre-change) allocation
        """
        if db_id not in self.orchestrator.strategy_states:
            return
            
        state = self.orchestrator.strategy_states[db_id]
        
        # Get last snapshot cycle and current cycle
        last_snapshot = state.get('last_snapshot_cycle', 0)
        current_cycle = state.get('current_cycle', 0)
        
        # Extract data from relevant period (from after last snapshot to current)
        relevant_allocations = [a for c, a in state['allocation_history'] 
                              if last_snapshot < c <= current_cycle]
        relevant_hrs = [hr for c, hr in state['hr_history'] 
                       if last_snapshot < c <= current_cycle]
        relevant_hr_maxs = [hrm for c, hrm in state['hr_max_history'] 
                           if last_snapshot < c <= current_cycle]
        
        # If sufficient historical data exists, use median; otherwise use current values
        if relevant_allocations and relevant_hrs:
            import statistics
            state['last_allocation_pages'] = statistics.median(relevant_allocations)
            state['last_ema_hit_rate'] = statistics.median(relevant_hrs)
            state['last_hr_max_adaptive'] = statistics.median(relevant_hr_maxs)
            
            if db_id in ['db_high_priority', 'db_medium_priority']:
                self.logger.info(f"📸 SNAPSHOT[{db_id}]: Using median of {len(relevant_allocations)} historical points - "
                               f"alloc={state['last_allocation_pages']:.1f} (current={current_allocation}), "
                               f"hr={state['last_ema_hit_rate']:.3f} (current={state['ema_hr_slow']:.3f})")
        else:
            # Use current values when no historical data available
            state['last_allocation_pages'] = current_allocation
            state['last_ema_hit_rate'] = state['ema_hr_slow']
            state['last_hr_max_adaptive'] = state['hr_max_adaptive']
            
            if db_id in ['db_high_priority', 'db_medium_priority']:
                self.logger.info(f"📸 SNAPSHOT[{db_id}]: No historical data, using current values - "
                               f"alloc={current_allocation}, hr={state['ema_hr_slow']:.3f}")
        
        # Update snapshot cycle
        state['last_snapshot_cycle'] = current_cycle
        
        # Reset wait counter
        state['cycles_since_major_change'] = 0
    
    def collect_historical_data(self, db_id: str, cycle_num: int, allocation: int, 
                               hr_ema_slow: float, hr_norm: float, hr_max_adaptive: float):
        """
        Collect historical data for each cycle, used for median calculation
        
        Args:
            db_id: Database ID
            cycle_num: Current cycle number
            allocation: Current allocation pages
            hr_ema_slow: Slow EMA hit rate
            hr_norm: Normalized hit rate
            hr_max_adaptive: Adaptive maximum hit rate
        """
        if db_id not in self.orchestrator.strategy_states:
            return
            
        state = self.orchestrator.strategy_states[db_id]
        
        # Update current cycle number
        state['current_cycle'] = cycle_num
        
        # Add data to historical records
        state['allocation_history'].append((cycle_num, allocation))
        state['hr_history'].append((cycle_num, hr_ema_slow))
        state['hr_norm_history'].append((cycle_num, hr_norm))
        state['hr_max_history'].append((cycle_num, hr_max_adaptive))
        
        # Limit history length, keep most recent 50 data points
        max_history = 50
        for hist_key in ['allocation_history', 'hr_history', 'hr_norm_history', 'hr_max_history']:
            if len(state[hist_key]) > max_history:
                state[hist_key].pop(0)
                
        if db_id in ['db_high_priority', 'db_medium_priority'] and cycle_num % 10 == 0:
            self.logger.debug(f"📊 HISTORY[{db_id}]: Cycle {cycle_num}, history length={len(state['allocation_history'])}")
    
    def initialize_allocation_history(self, initial_allocations: Dict[str, int]):
        """
        Set historical snapshot data after initial allocation
        """
        for db_id, allocation in initial_allocations.items():
            if db_id in self.orchestrator.strategy_states:
                state = self.orchestrator.strategy_states[db_id]
                state['last_allocation_pages'] = allocation
                self.logger.debug(f"DB[{db_id}] Initial allocation history set: {allocation} pages")
    
    def increment_cycle_counters(self):
        """
        Increment all cycle counters at the end of each tuning cycle
        """
        for db_id, state in self.orchestrator.strategy_states.items():
            # Increment cycles since last major change
            if 'cycles_since_major_change' in state:
                state['cycles_since_major_change'] += 1
                
            # Increment other cycle counters
            if 'cache_change_cycles_since' in state:
                state['cache_change_cycles_since'] += 1
                
            # Decrement remaining audit cycles
            if 'audit_state' in state and state['audit_state']['is_auditing']:
                state['audit_state']['audit_cycles_remaining'] -= 1
    
    def trigger_v_factor_audit(self, db_id: str, cycle_num: int, current_allocation: int):
        """
        Trigger V factor audit
        
        Args:
            db_id: Database ID
            cycle_num: Current cycle number
            current_allocation: Allocation at trigger time
        """
        if db_id not in self.orchestrator.strategy_states:
            return
            
        state = self.orchestrator.strategy_states[db_id]
        audit_state = state['audit_state']
        
        # Don't trigger again if already auditing
        if audit_state['is_auditing']:
            return
        
        # Initialize audit state
        audit_state['is_auditing'] = True
        audit_state['audit_trigger_cycle'] = cycle_num
        audit_state['audit_cycles_remaining'] = self.hyperparams.get('audit_observation_cycles', 3)
        audit_state['micro_v_factors'] = []
        
        # Use last_allocation_pages as pre-audit allocation (this is the true pre-change value)
        audit_state['pre_audit_allocation'] = state.get('last_allocation_pages', current_allocation)
        audit_state['pre_audit_hr'] = state.get('last_ema_hit_rate', state['ema_hr_slow'])
        audit_state['frozen_allocation'] = current_allocation  # Freeze current (new) allocation
        
        # Calculate and save pre-audit normalized hit rate (using historical values)
        hr_base = state['hr_base']
        pre_audit_hr = audit_state['pre_audit_hr']
        last_hr_max = state.get('last_hr_max_adaptive', state['hr_max_adaptive'])
        
        if last_hr_max > hr_base:
            hr_norm = (pre_audit_hr - hr_base) / (last_hr_max - hr_base)
            hr_norm = max(0.0, min(1.0, hr_norm))
        else:
            hr_norm = 0.0
        audit_state['pre_audit_hr_norm'] = hr_norm
        
        if db_id in ['db_high_priority', 'db_medium_priority']:
            pre_alloc = audit_state['pre_audit_allocation']
            self.logger.info(f"🔍 AUDIT_START[{db_id}]: V factor audit started - cycle={cycle_num}, "
                           f"alloc={pre_alloc}→{current_allocation}(frozen), hr_norm={hr_norm:.3f}")
    
    def update_audit_micro_v_factor(self, db_id: str, current_allocation: int, current_hr_norm: float):
        """
        Update micro V factor during audit period
        
        Args:
            db_id: Database ID
            current_allocation: Current allocation
            current_hr_norm: Current normalized hit rate
        """
        if db_id not in self.orchestrator.strategy_states:
            return
            
        state = self.orchestrator.strategy_states[db_id]
        audit_state = state['audit_state']
        
        # Only calculate during audit period
        if not audit_state['is_auditing']:
            return
        
        # Debug logging
        if db_id in ['db_high_priority', 'db_medium_priority']:
            self.logger.info(f"🔍 MICRO_V_UPDATE[{db_id}]: Starting micro V factor update, current_hr_norm={current_hr_norm:.4f}")
        
        # Calculate micro V factor
        pre_alloc = audit_state['pre_audit_allocation']
        pre_hr_norm = audit_state['pre_audit_hr_norm']
        frozen_alloc = audit_state.get('frozen_allocation', current_allocation)
        
        # Since allocation is frozen, use allocation change at trigger time as baseline
        if pre_alloc is not None and frozen_alloc != pre_alloc:
            # Use allocation change when frozen (this is what triggered the audit)
            delta_alloc = frozen_alloc - pre_alloc
            delta_hr_norm = current_hr_norm - pre_hr_norm
            
            if abs(delta_alloc) > 0.1:  # Avoid division by zero
                micro_v = delta_hr_norm / delta_alloc
                audit_state['micro_v_factors'].append(micro_v)
                
                if db_id in ['db_high_priority', 'db_medium_priority']:
                    self.logger.info(f"📊 AUDIT_MICRO_V[{db_id}]: Cycle {len(audit_state['micro_v_factors'])}, "
                                   f"ΔHR_norm={delta_hr_norm:.4f}, Δalloc(frozen)={delta_alloc} → v={micro_v:.6f}")
            else:
                # Allocation change too small, record as 0
                audit_state['micro_v_factors'].append(0.0)
                if db_id in ['db_high_priority', 'db_medium_priority']:
                    self.logger.info(f"📊 AUDIT_MICRO_V[{db_id}]: Allocation change too small (Δalloc={delta_alloc}), recording v=0")
    
    def complete_audit_if_needed(self, db_id: str, cycle_num: int) -> bool:
        """
        Check and complete audit if needed
        
        Args:
            db_id: Database ID
            cycle_num: Current cycle number
            
        Returns:
            bool: Whether audit was completed
        """
        if db_id not in self.orchestrator.strategy_states:
            return False
            
        state = self.orchestrator.strategy_states[db_id]
        audit_state = state['audit_state']
        
        # Check if audit needs to be completed
        if not audit_state['is_auditing'] or audit_state['audit_cycles_remaining'] > 0:
            return False
        
        # Calculate audit results
        micro_v_factors = audit_state['micro_v_factors']
        if micro_v_factors:
            import statistics
            
            # Calculate direction (median)
            v_direction = statistics.median(micro_v_factors)
            
            # Calculate confidence
            confidence = self._calculate_audit_confidence(micro_v_factors, v_direction)
            
            # Update audit results
            state['v_audit_result']['direction'] = v_direction
            state['v_audit_result']['confidence'] = confidence
            state['v_audit_result']['last_update_cycle'] = cycle_num
            
            if db_id in ['db_high_priority', 'db_medium_priority']:
                self.logger.info(f"✅ AUDIT_COMPLETE[{db_id}]: V_direction={v_direction:.6f}, C_V={confidence:.3f}, samples={len(micro_v_factors)}")
        else:
            # No valid samples, use default values
            state['v_audit_result']['direction'] = 0.0
            state['v_audit_result']['confidence'] = 0.0
            
            if db_id in ['db_high_priority', 'db_medium_priority']:
                self.logger.info(f"⚠️ AUDIT_COMPLETE[{db_id}]: No valid samples, using default values V=0, C_V=0")
        
        # Reset audit state
        audit_state['is_auditing'] = False
        audit_state['micro_v_factors'] = []
        audit_state['audit_cycles_remaining'] = 0
        audit_state['frozen_allocation'] = None  # Clear frozen allocation
        
        return True
    
    def _calculate_audit_confidence(self, micro_v_factors: list, v_direction: float) -> float:
        """
        Calculate audit confidence
        
        Args:
            micro_v_factors: List of micro V factors  
            v_direction: V factor direction (median)
            
        Returns:
            float: Confidence score [0, 1]
        """
        import statistics
        
        if not micro_v_factors:
            return 0.0
        
        # Calculate consistency ratio
        if abs(v_direction) < 1e-9:  # v_direction close to 0
            same_sign_count = len([v for v in micro_v_factors if abs(v) < 1e-9])
        else:
            same_sign_count = len([v for v in micro_v_factors 
                                 if (v > 0 and v_direction > 0) or 
                                    (v < 0 and v_direction < 0)])
        consistency_ratio = same_sign_count / len(micro_v_factors)
        
        # Calculate stability factor
        if len(micro_v_factors) > 1:
            std_dev = statistics.stdev(micro_v_factors)
            # Use configured maximum standard deviation parameter
            max_std_dev = self.hyperparams.get('audit_max_std_dev', 0.01)
            stability_factor = max(0.0, 1.0 - std_dev / max_std_dev)
        else:
            stability_factor = 1.0
        
        # Comprehensive confidence
        confidence = consistency_ratio * stability_factor
        
        # Apply minimum consistency requirement
        min_consistency = self.hyperparams.get('audit_min_consistency', 0.6)
        if consistency_ratio < min_consistency:
            confidence *= (consistency_ratio / min_consistency)
        
        return max(0.0, min(1.0, confidence))