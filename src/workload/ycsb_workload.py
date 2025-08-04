"""
YCSB workload implementation
Adapter for existing YCSBWorkload class to new infrastructure
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from .base_workload import BaseWorkload
import sys
import os

# Import existing modules
try:
    # Try relative import (when used as package)
    from ..ycsb_utils import (
        zipf_generator, generate_random_string, get_table_name, 
        get_key_prefix, create_ycsb_table
    )
except ImportError:
    # Fall back to absolute import (when run directly)
    from ycsb_utils import (
        zipf_generator, generate_random_string, get_table_name, 
        get_key_prefix, create_ycsb_table
    )


class YCSBWorkload(BaseWorkload):
    """YCSB workload implementation - adapter for existing YCSB code"""
    
    def __init__(self, conn, db_id, config, seed_offset=0, logger=None):
        super().__init__(conn, db_id, config, seed_offset, logger)
        
        # YCSB-specific configuration
        ycsb_config = config.get('ycsb_config', {})
        self.row_size_bytes = ycsb_config.get('ycsb_row_size_bytes', 2048)
        self.hot_rate = ycsb_config.get('hot_rate', 0.167)
        self.workload_proportions = ycsb_config.get('ycsb_workload_proportions', {
            'read': 0.85,
            'update': 0.15,
            'insert': 0,
            'scan': 0,
            'read_modify_write': 0
        })
        self.access_pattern = ycsb_config.get('access_pattern', 'zipfian')
        self.zipf_alpha = ycsb_config.get('zipf_alpha', 0.9)
        
        # Set up random number generator
        self.rng = random.Random(42 + seed_offset)
        
        # Record count and hot spot set
        self.record_count = 0
        self.hot_spot_set = []
        self.operation_count = 0
        
        # Zipfian generator (if needed)
        self.zipf_gen = None
        
        # Recently inserted keys (for read-then-write mode)
        self.recently_inserted_keys = []
        self.max_recent_keys = 1000
        
        self.logger.info(f"YCSB workload initialized with {self.access_pattern} access pattern")
        
        # Get actual record count from database
        self._load_record_count_from_db()
        
    def create_schema(self):
        """Create YCSB table structure"""
        self.logger.info("Creating YCSB schema...")
        
        # Use existing create_ycsb_table function
        create_ycsb_table(self.conn, row_size_bytes=self.row_size_bytes)
        
        self.logger.info("YCSB schema created successfully")
        
    def _load_record_count_from_db(self):
        """Get actual record count from database and initialize hot spot set"""
        try:
            # Get record count
            table_name = get_table_name()
            cursor = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}")
            result = cursor.fetchone()
            if result:
                self.record_count = result[0]
                self.logger.info(f"Loaded record_count from database: {self.record_count}")
                
                # If there are records, initialize hot spot set
                if self.record_count > 0:
                    self._create_hot_spot_set()
                else:
                    self.logger.warning(f"No records in {table_name}, record_count=0, workload will generate virtual operations")
                    self.hot_spot_set = []
            else:
                self.logger.warning("Unable to query database record count, using default value 0")
                self.record_count = 0
                self.hot_spot_set = []
                
        except Exception as e:
            self.logger.error(f"Failed to query database record count: {e}")
            # Check if table exists
            try:
                self.conn.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
                self.logger.info(f"{table_name} exists but record count query failed, setting record_count=0")
            except:
                self.logger.info(f"{table_name} does not exist, setting record_count=0")
            self.record_count = 0
            self.hot_spot_set = []
            
    def _create_hot_spot_set(self):
        """Create hot spot set (based on existing ycsb_utils logic)"""
        try:
            if self.record_count <= 0:
                self.hot_spot_set = []
                return
                
            # Calculate hot spot count (16.7% ratio)
            hot_count = max(1, int(self.record_count * self.hot_rate))
            
            # Randomly select hot spot priority_values from database
            table_name = get_table_name()
            cursor = self.conn.execute(
                f"SELECT priority_value FROM {table_name} ORDER BY RANDOM() LIMIT ?", 
                (hot_count,)
            )
            
            self.hot_spot_set = [row[0] for row in cursor.fetchall()]
            self.logger.info(f"Created hot spot set with {len(self.hot_spot_set)} hot spots")
            
            # Initialize zipfian generator (if needed)
            if self.access_pattern == 'zipfian' and len(self.hot_spot_set) > 1:
                from ycsb_utils import zipf_generator
                self.zipf_gen = zipf_generator(len(self.hot_spot_set), self.zipf_alpha)
                
        except Exception as e:
            self.logger.error(f"Failed to create hot spot set: {e}")
            self.hot_spot_set = []
        
    def generate_initial_data(self, record_count: int):
        """Generate YCSB initial data"""
        self.logger.info(f"Generating {record_count} YCSB records...")
        
        self.record_count = record_count
        
        # Batch insert optimization
        self.conn.execute("PRAGMA synchronous = OFF")
        self.conn.execute("PRAGMA journal_mode = OFF")
        self.conn.execute("BEGIN TRANSACTION")
        
        try:
            # Calculate data size (excluding other fields)
            # id (about 20 bytes) + priority_value (8 bytes) + overhead (about 22 bytes) = 50 bytes
            data_field_size = max(1, self.row_size_bytes - 50)
            data_value = generate_random_string(data_field_size, self.seed_offset)
            
            table_name = get_table_name()
            key_prefix = get_key_prefix()
            
            # Batch insert
            batch_size = 1000
            for i in range(record_count):
                key = f"{self.db_id}_{key_prefix}{i}"
                priority_value = i
                
                self.conn.execute(
                    f"INSERT INTO {table_name} VALUES (?, ?, ?)",
                    (key, priority_value, data_value)
                )
                
                if (i + 1) % batch_size == 0:
                    self.conn.execute("COMMIT")
                    self.conn.execute("BEGIN TRANSACTION")
                    if (i + 1) % 10000 == 0:
                        self.logger.debug(f"Generated {i + 1} records")
                        
            self.conn.execute("COMMIT")
            
            # Create hot spot set
            self._create_hot_spot_set()
            
            self.logger.info(f"YCSB data generation completed")
            
        except Exception as e:
            self.conn.execute("ROLLBACK")
            self.logger.error(f"Error generating YCSB data: {e}")
            raise
            
    def _create_hot_spot_set(self):
        """Create hot spot data set"""
        hot_spot_count = int(self.record_count * self.hot_rate)
        hot_spot_count = max(1, hot_spot_count)
        
        # Randomly select hot spots
        all_indices = list(range(self.record_count))
        self.rng.shuffle(all_indices)
        self.hot_spot_set = sorted(all_indices[:hot_spot_count])
        
        # Initialize Zipfian generator
        if self.access_pattern == 'zipfian':
            self.zipf_gen = zipf_generator(len(self.hot_spot_set), self.zipf_alpha, self.rng)
            
        self.logger.info(f"Created hot spot set with {len(self.hot_spot_set)} items "
                        f"({self.hot_rate * 100:.1f}% of data)")
                        
    def generate_operation(self) -> Tuple[str, Dict[str, Any]]:
        """Generate next YCSB operation"""
        # Check if there are available records
        if self.record_count <= 0:
            self.logger.warning(f"No available records (record_count={self.record_count}), skipping operation")
            # Return a dummy insert operation
            return ('insert', {
                'id': f"{self.db_id}_dummy_0",
                'priority_value': 0,
                'data': 'dummy_data'
            })
        
        # Choose operation type
        op_type = self._choose_operation_type()
        
        if op_type == 'read':
            priority_value = self._generate_priority_value_for_read()
            return ('read', {'priority_value': priority_value})
            
        elif op_type == 'update':
            # Update uses ID instead of priority_value
            record_id = self._generate_record_id_for_update()
            new_priority = self.rng.randint(0, max(0, self.record_count - 1))
            data_size = max(1, self.row_size_bytes - 50)
            new_data = generate_random_string(data_size, self.seed_offset + self.operation_count)
            
            return ('update', {
                'id': record_id,
                'new_priority': new_priority,
                'new_data': new_data
            })
            
        elif op_type == 'insert':
            # Generate new key
            new_key = f"{self.db_id}_{get_key_prefix()}{self.record_count + len(self.recently_inserted_keys)}"
            new_priority = self.record_count + len(self.recently_inserted_keys)
            data_size = max(1, self.row_size_bytes - 50)
            new_data = generate_random_string(data_size, self.seed_offset + self.operation_count)
            
            # Record recently inserted key
            self.recently_inserted_keys.append(new_key)
            if len(self.recently_inserted_keys) > self.max_recent_keys:
                self.recently_inserted_keys.pop(0)
                
            return ('insert', {
                'id': new_key,
                'priority_value': new_priority,
                'data': new_data
            })
            
        elif op_type == 'scan':
            start_key = self._generate_record_id_for_scan()
            scan_length = self.rng.randint(1, 100)
            
            return ('scan', {
                'start_key': start_key,
                'scan_length': scan_length
            })
            
        elif op_type == 'read_modify_write':
            record_id = self._generate_record_id_for_update()
            return ('read_modify_write', {'id': record_id})
            
        else:
            # Default to return read operation
            priority_value = self._generate_priority_value_for_read()
            return ('read', {'priority_value': priority_value})
            
    def execute_operation(self, op_type: str, params: Dict[str, Any]) -> bool:
        """Execute YCSB operation"""
        try:
            table_name = get_table_name()
            
            if op_type == 'read':
                cursor = self.conn.execute(
                    f"SELECT 1 FROM {table_name} WHERE priority_value = ? LIMIT 1",
                    (params['priority_value'],)
                )
                return cursor.fetchone() is not None
                
            elif op_type == 'update':
                self.conn.execute(
                    f"UPDATE {table_name} SET priority_value = ?, data = ? WHERE id = ?",
                    (params['new_priority'], params['new_data'], params['id'])
                )
                return self.conn.changes() > 0
                
            elif op_type == 'insert':
                self.conn.execute(
                    f"INSERT INTO {table_name} VALUES (?, ?, ?)",
                    (params['id'], params['priority_value'], params['data'])
                )
                return True
                
            elif op_type == 'scan':
                cursor = self.conn.execute(
                    f"SELECT * FROM {table_name} WHERE id >= ? ORDER BY id LIMIT ?",
                    (params['start_key'], params['scan_length'])
                )
                results = cursor.fetchall()
                return len(results) > 0
                
            elif op_type == 'read_modify_write':
                # Read first
                cursor = self.conn.execute(
                    f"SELECT data FROM {table_name} WHERE id = ?",
                    (params['id'],)
                )
                row = cursor.fetchone()
                
                if row:
                    # Modify and write back
                    new_data = row[0] + "_modified"
                    self.conn.execute(
                        f"UPDATE {table_name} SET data = ? WHERE id = ?",
                        (new_data, params['id'])
                    )
                    return True
                return False
                
            else:
                self.logger.error(f"Unknown operation type: {op_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error executing {op_type}: {e}")
            return False
        finally:
            self.operation_count += 1
            
    def _choose_operation_type(self) -> str:
        """Choose operation type based on workload proportions"""
        rand = self.rng.random()
        cumulative = 0.0
        
        for op_type, proportion in self.workload_proportions.items():
            cumulative += proportion
            if rand < cumulative:
                return op_type
                
        return 'read'  # Default
        
    def _generate_priority_value_for_read(self) -> int:
        """Generate priority_value for read operation"""
        if not self.hot_spot_set:
            return self.rng.randint(0, max(0, self.record_count - 1))
            
        if self.access_pattern == 'uniform':
            # Uniformly select from hot spot set
            idx = self.rng.randint(0, len(self.hot_spot_set) - 1)
            return self.hot_spot_set[idx]
            
        elif self.access_pattern == 'zipfian':
            # Use Zipfian distribution
            if self.zipf_gen:
                zipf_idx = next(self.zipf_gen)
                # Ensure index is within range
                zipf_idx = min(zipf_idx, len(self.hot_spot_set) - 1)
                return self.hot_spot_set[zipf_idx]
            else:
                # Downgrade to uniform distribution
                idx = self.rng.randint(0, len(self.hot_spot_set) - 1)
                return self.hot_spot_set[idx]
                
        else:
            # Unknown mode, use uniform distribution
            idx = self.rng.randint(0, len(self.hot_spot_set) - 1)
            return self.hot_spot_set[idx]
            
    def _generate_record_id_for_update(self) -> str:
        """Generate record ID for update operation"""
        # 90% chance to update existing records, 10% to update recently inserted records
        if self.recently_inserted_keys and self.rng.random() < 0.1:
            return self.rng.choice(self.recently_inserted_keys)
        else:
            record_num = self.rng.randint(0, max(0, self.record_count - 1))
            return f"{self.db_id}_{get_key_prefix()}{record_num}"
            
    def _generate_record_id_for_scan(self) -> str:
        """Generate starting record ID for scan operation"""
        record_num = self.rng.randint(0, max(0, self.record_count - 100))
        return f"{self.db_id}_{get_key_prefix()}{record_num}"
        
    def update_access_pattern(self, new_pattern: str, new_params: Dict[str, Any] = None):
        """Update access pattern (for dynamic workloads)"""
        self.access_pattern = new_pattern
        
        if new_params:
            if 'zipf_alpha' in new_params:
                self.zipf_alpha = new_params['zipf_alpha']
                
            if 'hot_rate' in new_params:
                self.hot_rate = new_params['hot_rate']
                self._create_hot_spot_set()
                
        # Reinitialize Zipfian generator
        if self.access_pattern == 'zipfian' and self.hot_spot_set:
            self.zipf_gen = zipf_generator(len(self.hot_spot_set), self.zipf_alpha, self.rng)
            
        self.logger.info(f"Access pattern updated to {self.access_pattern}")