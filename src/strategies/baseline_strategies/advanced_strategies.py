from typing import Any, Dict
from ..base_strategy import BaseStrategy


class B11_ML_Driven_Strategy(BaseStrategy):
    """
    B11: ML-driven strategy
    Uses linear regression to predict performance impact of cache allocation changes.
    """
    is_dynamic = True
    
    def __init__(self, orchestrator: Any, strategy_name: str, strategy_specific_config: Dict = None):
        super().__init__(orchestrator, strategy_name, strategy_specific_config)
        
        # Import ML library - LinearRegression is used to predict cache hit improvements
        try:
            from sklearn.linear_model import LinearRegression
            self.LinearRegression = LinearRegression
        except ImportError:
            raise ImportError("B11 strategy requires scikit-learn. Please run: pip install scikit-learn")
        
        from collections import deque
        self.deque = deque
        
        self.hyperparams = self.strategy_specific_config.get("strategy_config", {}).get("strategy_hyperparameters", {})
        self.training_buffer_size = self.hyperparams.get("training_buffer_size", 100)
        self.prediction_actions = self.hyperparams.get("prediction_actions", [-200, -100, 100, 200])
        self.min_training_samples = self.hyperparams.get("min_training_samples", 10)
        self.action_scale_ratio = self.hyperparams.get("action_scale_ratio", 0.1)
        self.smoothing_factor = self.hyperparams.get("smoothing_factor", 0.3)
        self.min_action_size = self.hyperparams.get("min_action_size", 50)
        
        if not hasattr(self.orchestrator, 'strategy_states'):
            self.orchestrator.strategy_states = {}
            
        # Initialize per-database ML state - each DB has its own model and training data
        for db_conf in self.db_instance_configs:
            db_id = db_conf['id']
            self.orchestrator.strategy_states[db_id] = {
                'ml_model': self.LinearRegression(),
                'training_buffer': self.deque(maxlen=self.training_buffer_size),
                'last_hits': 0,
                'last_allocation': 0,
                'last_ops': 0,
                'last_hit_rate': 0.0,
                'is_initialized': False,
                'smoothed_allocation': 0,
                'prediction_history': self.deque(maxlen=3)
            }
        
        self.logger.info(f"{strategy_name} (B11_ML_Driven) strategy initialized with buffer_size={self.training_buffer_size}")
    
    def calculate_initial_allocations(self):
        num_databases = len(self.db_instance_configs)
        pages_per_db = self.total_pages // num_databases
        
        initial_allocations = {}
        for db_conf in self.db_instance_configs:
            db_id = db_conf["id"]
            initial_allocations[db_id] = pages_per_db
        
        remainder = self.total_pages % num_databases
        for i, db_conf in enumerate(self.db_instance_configs[:remainder]):
            db_id = db_conf["id"]
            initial_allocations[db_id] += 1
        
        for db_id, allocation in initial_allocations.items():
            self.orchestrator.strategy_states[db_id]['smoothed_allocation'] = allocation
        
        self.logger.info(f"B11: Initial allocation: {initial_allocations}")
        return initial_allocations
    
    def update_allocations(self, current_metrics: Dict[str, Any], elapsed_in_phase: float) -> Dict[str, int]:
        """Update allocations using ML predictions"""
        import numpy as np
        
        o = self.orchestrator
        new_allocations = {}
        db_scores = {}
        
        missing_dbs = []
        incomplete_dbs = []
        for db_conf in self.db_instance_configs:
            db_id = db_conf["id"]
            db_metrics = current_metrics.get(db_id, {})
            if not db_metrics:
                missing_dbs.append(db_id)
            elif not all(key in db_metrics for key in ['cache_hits', 'cache_misses', 'ops_count']):
                incomplete_dbs.append(db_id)
        
        if missing_dbs or incomplete_dbs:
            self.logger.warning(f"B11: Incomplete data - missing: {missing_dbs}, incomplete: {incomplete_dbs}")
            self.logger.warning("B11: Skipping allocation")
            return o.db_current_page_allocations.copy()
        
        # Update training data and train models
        for db_conf in self.db_instance_configs:
            db_id = db_conf["id"]
            state = o.strategy_states[db_id]
            
            db_metrics = current_metrics[db_id]
            current_hits = db_metrics["cache_hits"]
            current_misses = db_metrics["cache_misses"]
            current_ops = db_metrics["ops_count"]
            total_accesses = current_hits + current_misses
            
            if total_accesses == 0:
                current_hit_rate = 0.5
            else:
                current_hit_rate = current_hits / total_accesses
                
            current_allocation = o.db_current_page_allocations.get(db_id)
            if current_allocation is None:
                self.logger.warning(f"B11: DB[{db_id}] missing allocation info")
                continue
            
            # Collect training data from actual performance changes
            if state['is_initialized']:
                delta_hits_actual = current_hits - state['last_hits']
                delta_alloc = current_allocation - state['last_allocation']
                delta_hit_rate = current_hit_rate - state['last_hit_rate']
                delta_ops = current_ops - state['last_ops']
                
                # Feature vector: [prev_alloc, prev_ops, prev_hit_rate, delta_alloc, delta_ops, delta_hr]
                features = np.array([[
                    state['last_allocation'],
                    state['last_ops'],
                    state['last_hit_rate'],
                    delta_alloc,
                    delta_ops,
                    delta_hit_rate
                ]])
                
                state['training_buffer'].append((features[0], delta_hits_actual))
                
                if len(state['training_buffer']) >= self.min_training_samples:
                    X = np.array([sample[0] for sample in state['training_buffer']])
                    y = np.array([sample[1] for sample in state['training_buffer']])
                    
                    try:
                        state['ml_model'].fit(X, y)
                    except Exception as e:
                        self.logger.warning(f"B11 DB[{db_id}] model training failed: {e}")
            
            state['last_hits'] = current_hits
            state['last_allocation'] = current_allocation
            state['last_ops'] = current_ops
            state['last_hit_rate'] = current_hit_rate
            state['is_initialized'] = True
        
        # Use models to predict optimal actions
        for db_conf in self.db_instance_configs:
            db_id = db_conf["id"]
            state = o.strategy_states[db_id]
            
            max_predicted_gain = 0.0
            best_action = 0
            
            current_alloc = state['last_allocation']
            action_scale = max(self.min_action_size, int(self.action_scale_ratio * current_alloc))
            
            adaptive_actions = [-2*action_scale, -action_scale, 0, action_scale, 2*action_scale]
            
            # Predict cache hit improvements for different allocation actions
            if len(state['training_buffer']) >= self.min_training_samples:
                for action in adaptive_actions:
                    if current_alloc + action < 0:
                        continue
                        
                    # Simulate feature vector for this potential action
                    pred_features = np.array([[
                        state['last_allocation'],
                        state['last_ops'],
                        state['last_hit_rate'],
                        action,  # The allocation change we're testing
                        0,       # Assume no ops change
                        0        # Assume no hit rate change yet
                    ]])
                    
                    try:
                        predicted_delta_hits = state['ml_model'].predict(pred_features)[0]
                        if predicted_delta_hits > max_predicted_gain:
                            max_predicted_gain = predicted_delta_hits
                            best_action = action
                    except Exception as e:
                        self.logger.warning(f"B11 DB[{db_id}] prediction failed: {e}")
            else:
                # Not enough training data - use ops count as proxy for importance
                max_predicted_gain = state['last_ops']
            
            state['prediction_history'].append(best_action)
            
            db_scores[db_id] = max(1e-6, max_predicted_gain)
        
        # Calculate target allocations by score
        target_allocations = {}
        total_score = sum(db_scores.values())
        if total_score > 0:
            for db_id, score in db_scores.items():
                proportion = score / total_score
                target_allocations[db_id] = self.total_pages * proportion
        else:
            pages_per_db = self.total_pages / len(self.db_instance_configs)
            for db_conf in self.db_instance_configs:
                target_allocations[db_conf["id"]] = pages_per_db
        
        # Apply exponential smoothing to avoid drastic allocation changes
        for db_id in target_allocations:
            state = o.strategy_states[db_id]
            current_smoothed = state['smoothed_allocation']
            target = target_allocations[db_id]
            
            # EMA smoothing: new = (1-α)*old + α*target
            new_smoothed = (1 - self.smoothing_factor) * current_smoothed + self.smoothing_factor * target
            state['smoothed_allocation'] = new_smoothed
            new_allocations[db_id] = int(round(new_smoothed))
        
        # Handle total pages difference
        total_allocated = sum(new_allocations.values())
        diff = self.total_pages - total_allocated
        
        if diff != 0:
            sorted_dbs = sorted(db_scores.items(), key=lambda x: x[1], reverse=True)
            for db_id, _ in sorted_dbs:
                if diff == 0:
                    break
                if diff > 0:
                    new_allocations[db_id] += 1
                    diff -= 1
                elif new_allocations[db_id] > 0:
                    new_allocations[db_id] -= 1
                    diff += 1
        
        self.logger.info(f"B11: ML allocation completed: {new_allocations}")
        return new_allocations


class B12_MT_LRU_Inspired_Strategy(BaseStrategy):
    """
    B12: MT-LRU inspired strategy
    Prioritizes databases that are below their SLA targets.
    """
    is_dynamic = True
    
    def __init__(self, orchestrator: Any, strategy_name: str, strategy_specific_config: Dict = None):
        super().__init__(orchestrator, strategy_name, strategy_specific_config)
        
        self.hyperparams = self.strategy_specific_config.get("strategy_config", {}).get("strategy_hyperparameters", {})
        self.huge_weight = self.hyperparams.get("sla_huge_weight", 1000000.0)
        self.target_hit_rates = self.hyperparams.get("target_hit_rates", {})
        self.default_target_hit_rate = 0.4
        
        self.priority_dbs = set()
        for db_conf in self.db_instance_configs:
            self.priority_dbs.add(db_conf["id"])
        
        self.logger.info(f"{strategy_name} (B12_MT_LRU_Inspired) strategy initialized")
        self.logger.info(f"B12: SLA targets: {self.target_hit_rates}, weight: {self.huge_weight}")
    
    def calculate_initial_allocations(self):
        total_priority = sum(db_conf.get("base_priority", 1) for db_conf in self.db_instance_configs)
        
        initial_allocations = {}
        for db_conf in self.db_instance_configs:
            db_id = db_conf["id"]
            priority = db_conf.get("base_priority", 1)
            proportion = priority / total_priority
            initial_allocations[db_id] = int(self.total_pages * proportion)
        
        total_allocated = sum(initial_allocations.values())
        diff = self.total_pages - total_allocated
        
        if diff != 0:
            sorted_dbs = sorted(self.db_instance_configs, 
                              key=lambda x: x.get("base_priority", 1), reverse=True)
            for db_conf in sorted_dbs:
                if diff == 0:
                    break
                db_id = db_conf["id"]
                if diff > 0:
                    initial_allocations[db_id] += 1
                    diff -= 1
                elif initial_allocations[db_id] > 0:
                    initial_allocations[db_id] -= 1
                    diff += 1
        
        self.logger.info(f"B12: Initial allocation: {initial_allocations}")
        return initial_allocations
    
    def update_allocations(self, current_metrics: Dict[str, Any], elapsed_in_phase: float) -> Dict[str, int]:
        """SLA-based allocation strategy"""
        db_scores = {}
        sla_deficits = {}
        
        missing_dbs = []
        incomplete_dbs = []
        available_dbs = []
        
        for db_conf in self.db_instance_configs:
            db_id = db_conf["id"]
            db_metrics = current_metrics.get(db_id, {})
            if not db_metrics:
                missing_dbs.append(db_id)
            elif not all(key in db_metrics for key in ['cache_hits', 'cache_misses', 'ops_count']):
                incomplete_dbs.append(db_id)
            else:
                available_dbs.append(db_id)
        
        if not available_dbs:
            self.logger.warning(f"B12: All databases missing data, skipping allocation")
            return self.orchestrator.db_current_page_allocations.copy()
        
        if missing_dbs or incomplete_dbs:
            self.logger.info(f"B12: Partial data - missing: {missing_dbs}, incomplete: {incomplete_dbs}")
        
        # Calculate SLA deficit and base scores for each database
        # SLA deficit = how far below target hit rate we are
        for db_conf in self.db_instance_configs:
            db_id = db_conf["id"]
            
            if db_id not in available_dbs:
                continue
                
            db_metrics = current_metrics[db_id]
            
            cache_hits = db_metrics["cache_hits"]
            cache_misses = db_metrics["cache_misses"]
            current_ops = db_metrics["ops_count"]
            total_accesses = cache_hits + cache_misses
            
            if total_accesses == 0:
                current_hit_rate = 0.5
            else:
                current_hit_rate = cache_hits / total_accesses
            
            base_score = current_ops
            
            # Determine target hit rate based on DB type
            sla_deficit = 0.0
            if db_id in self.target_hit_rates:
                target_hit_rate = self.target_hit_rates[db_id]
            elif "bg" in db_id:
                target_hit_rate = 0.3  # Background DBs have lower targets
            else:
                target_hit_rate = self.default_target_hit_rate
            
            sla_deficit = max(0.0, target_hit_rate - current_hit_rate)
            sla_deficits[db_id] = sla_deficit
            
            # Huge weight ensures SLA violations get priority over throughput
            final_score = (sla_deficit * self.huge_weight) + base_score
            db_scores[db_id] = final_score
            
            if sla_deficit > 0:
                if db_id in self.target_hit_rates:
                    log_target = self.target_hit_rates[db_id]
                elif "bg" in db_id:
                    log_target = 0.3
                else:
                    log_target = self.default_target_hit_rate
                self.logger.info(f"B12 DB[{db_id}] SLA deficit: {sla_deficit:.3f} (target: {log_target:.3f}, current: {current_hit_rate:.3f})")
            
        
        # Handle missing databases with minimal allocation
        min_pages_per_db = 10
        reserved_pages = min_pages_per_db * len(missing_dbs + incomplete_dbs)
        available_pages = self.total_pages - reserved_pages
        
        for db_id in missing_dbs + incomplete_dbs:
            db_scores[db_id] = 0
        
        # Allocate cache by score proportion
        total_score = sum(score for db_id, score in db_scores.items() if db_id in available_dbs)
        new_allocations = {}
        
        for db_conf in self.db_instance_configs:
            db_id = db_conf["id"]
            new_allocations[db_id] = min_pages_per_db
        
        if total_score > 0 and available_pages > 0:
            remaining_pages = available_pages
            for db_id in available_dbs:
                score = db_scores.get(db_id, 0)
                proportion = score / total_score
                extra_pages = int(remaining_pages * proportion)
                new_allocations[db_id] += extra_pages
        else:
            if available_pages > 0:
                extra_per_db = available_pages // len(available_dbs) if available_dbs else 0
                for db_id in available_dbs:
                    new_allocations[db_id] += extra_per_db
        
        # Ensure total allocation exactly matches available pages
        total_allocated = sum(new_allocations.values())
        diff = self.total_pages - total_allocated
        
        if diff != 0:
            # Prioritize DBs with worst SLA violations for extra pages
            sla_deficit_dbs = [(db_id, deficit) for db_id, deficit in sla_deficits.items() if deficit > 0]
            sla_deficit_dbs.sort(key=lambda x: x[1], reverse=True)
            
            if sla_deficit_dbs:
                for db_id, _ in sla_deficit_dbs:
                    if diff == 0:
                        break
                    if diff > 0:
                        new_allocations[db_id] += 1
                        diff -= 1
                    elif new_allocations[db_id] > 0:
                        new_allocations[db_id] -= 1
                        diff += 1
            else:
                sorted_dbs = sorted(db_scores.items(), key=lambda x: x[1], reverse=True)
                for db_id, _ in sorted_dbs:
                    if diff == 0:
                        break
                    if diff > 0:
                        new_allocations[db_id] += 1
                        diff -= 1
                    elif new_allocations[db_id] > 0:
                        new_allocations[db_id] -= 1
                        diff += 1
        
        active_sla_deficits = {db_id: deficit for db_id, deficit in sla_deficits.items() if deficit > 0}
        if missing_dbs or incomplete_dbs:
            self.logger.info(f"B12: Allocation with partial data: {new_allocations}")
        elif active_sla_deficits:
            self.logger.info(f"B12: SLA deficits detected: {active_sla_deficits}")
            self.logger.info(f"B12: SLA-driven allocation: {new_allocations}")
        else:
            self.logger.info(f"B12: All SLA met, ops-based allocation: {new_allocations}")
        
        return new_allocations