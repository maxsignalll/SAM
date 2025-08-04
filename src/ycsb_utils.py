import random
import string
import threading
import time
import apsw # Import apsw here if it's used for type hinting or specific exceptions
import logging # Added import
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

def zipf_generator(n, theta, random_state=None):
    """
    An efficient Zipf distribution generator.
    It pre-computes the cumulative distribution function (CDF) and uses binary search for sampling.
    
    Args:
        n (int): Number of elements (from 0 to n-1).
        theta (float): Skewness parameter of the distribution.
        random_state: A random number generator instance with a .random() method (e.g., random.Random).
    """
    if n <= 0:
        while True:
            yield 0
        return

    # Pre-compute normalization constant (harmonic number)
    h_n = sum(1.0 / (i**theta) for i in range(1, n + 1))
    
    # Pre-compute probability for each element
    p = [1.0 / (i**theta * h_n) for i in range(1, n + 1)]
    
    # Pre-compute cumulative distribution function (CDF)
    cdf = np.cumsum(p)

    # If no random_state is provided, use default random
    if random_state is None:
        import random as default_random
        random_func = default_random.random
    else:
        random_func = random_state.random
        
    while True:
        r = random_func()
        # Use binary search to find which interval r falls into, return its index
        # np.searchsorted finds insertion point in sorted array, which is exactly what we need
        yield np.searchsorted(cdf, r, side='right')

def generate_random_string(length, seed_offset=0):
    # Use different random state for different databases
    rng = random.Random(42 + seed_offset)
    return ''.join(rng.choice(string.ascii_lowercase) for i in range(length))

def get_table_name():
    return "usertable"

def get_key_prefix():
    return "user"

def _create_priority_index(conn):
    """Create index for priority_value column"""
    table_name = get_table_name()
    index_name = f"idx_{table_name}_priority_value"
    conn.execute(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} (priority_value)")


def create_ycsb_table(conn, row_size_bytes: int = None, field_count: int = None):
      """
      Create YCSB table. Supports two modes:
      1. New mode: specify row_size_bytes, will automatically calculate appropriate field configuration
      2. Legacy mode: specify field_count, maintain backward compatibility
      """
      table_name = get_table_name()

      if row_size_bytes is not None:
          # New mode: create table based on total row size
          conn.execute(f"DROP TABLE IF EXISTS {table_name}")
          conn.execute(f"""
              CREATE TABLE IF NOT EXISTS {table_name} (
                  id TEXT PRIMARY KEY,
                  priority_value INTEGER,
                  data TEXT
              )
          """)
      else:
          # Legacy mode: create table based on field count (backward compatible)
          if field_count is None:
              field_count = 4  # Default value
          field_definitions = [f"field{i} TEXT" for i in range(field_count)]
          fields_sql = ", ".join(field_definitions)
          conn.execute(f"DROP TABLE IF EXISTS {table_name}")
          conn.execute(f"""
              CREATE TABLE IF NOT EXISTS {table_name} (
                  id TEXT PRIMARY KEY,
                  priority_value INTEGER,
                  {fields_sql}
              )
          """)

      # Create index for new column
      _create_priority_index(conn)


def load_initial_data(conn, record_count: int, field_count: int = None, field_length: int = None, db_id: str = "default", row_size_bytes: int = None):
    table_name = get_table_name()
    key_prefix = get_key_prefix()
    
    # Create a unique seed offset based on db_id to ensure different data
    seed_offset = hash(db_id) % 10000

    if row_size_bytes is not None:
        # New mode: based on total row size
        print(f"[{db_id}] Loading {record_count} records (row size: {row_size_bytes} bytes) into {conn.filename}...")
        
        # Calculate data field size (total size - estimated key size - priority_value size)
        key_size_estimate = len(f"{db_id}_{key_prefix}") + 10  # db_id + prefix + numeric part
        priority_size_estimate = 4  # INTEGER typically 4 bytes
        data_size = max(1, row_size_bytes - key_size_estimate - priority_size_estimate)
        
        columns_sql = "id, priority_value, data"
        placeholders_sql = "?, ?, ?"
        insert_sql = f"INSERT INTO {table_name} ({columns_sql}) VALUES ({placeholders_sql})"
        
        print(f"[{db_id}] Data field size calculated as: {data_size} bytes (including priority_value column)")
    else:
        # Legacy mode: based on field count and length (backward compatible)
        if field_count is None:
            field_count = 4
        if field_length is None:
            field_length = 1000
            
        print(f"[{db_id}] Loading {record_count} records (fields: {field_count}, length: {field_length}) into {conn.filename}...")
        
        field_names = [f"field{i}" for i in range(field_count)]
        all_column_names = ["id", "priority_value"] + field_names
        columns_sql = ", ".join(all_column_names)
        placeholders_sql = ", ".join(["?"] * (field_count + 2)) # +2 for id and priority_value
        insert_sql = f"INSERT INTO {table_name} ({columns_sql}) VALUES ({placeholders_sql})"

    current_max_numeric_id = -1
    # Core fix: only use db_id as prefix when querying
    instance_key_prefix = f"{db_id}_{key_prefix}"
    query_sql = f"SELECT id FROM {table_name} WHERE id LIKE '{instance_key_prefix}%' ORDER BY CAST(SUBSTR(id, {len(instance_key_prefix) + 1}) AS INTEGER) DESC LIMIT 1"
    
    for row in conn.execute(query_sql):
        numeric_part = row[0][len(instance_key_prefix):]
        if numeric_part.isdigit():
            current_max_numeric_id = int(numeric_part)
        
    start_index = current_max_numeric_id + 1
    records_to_load = record_count - start_index

    if records_to_load <= 0 and start_index > 0: # start_index > 0 implies some records were found
        print(f"[{db_id}] Sufficient data already exists ({start_index} records). Target: {record_count}. Skipping load.")
        return start_index 
    
    print(f"[{db_id}] Starting load from index {start_index} for {records_to_load} new records with seed_offset={seed_offset}.")

    key_padding_length = len(str(record_count)) 

    # Optimize SQLite settings to speed up bulk insert
    original_settings = {}
    # Save original settings
    original_settings['journal_mode'] = conn.execute("PRAGMA journal_mode").fetchone()[0]
    original_settings['synchronous'] = conn.execute("PRAGMA synchronous").fetchone()[0]
    original_settings['cache_size'] = conn.execute("PRAGMA cache_size").fetchone()[0]
    original_settings['temp_store'] = conn.execute("PRAGMA temp_store").fetchone()[0]
    
    # Set high-speed import mode
    conn.execute("PRAGMA journal_mode = MEMORY")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA cache_size = 10000")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA locking_mode = EXCLUSIVE")
    
    print(f"[{db_id}] SQLite optimization settings applied, starting fast import...")

    try:
        conn.execute("BEGIN IMMEDIATE") # Start exclusive transaction
        
        # Use batch insert to significantly improve performance
        batch_size = 10000
        batch_data = []
        
        # Pre-generate random strings for reuse
        rng = random.Random(42 + seed_offset)
        char_pool = string.ascii_lowercase
        
        # According to original implementation: generate data_value once, shared by all records
        if row_size_bytes is not None:
            # New mode: generate single data field of specified size
            data_value = generate_random_string(data_size, seed_offset)
        else:
            # Old mode: also generate once, shared by all records
            field_values = [
                ''.join(rng.choice(char_pool) for _ in range(field_length)) 
                for _ in range(field_count)
            ]
        
        processed_count = 0
        
        for i in range(start_index, record_count):
            # Core fix: generate primary key with only db_id prefix
            key = f"{instance_key_prefix}{i:0{key_padding_length}d}"
            
            if row_size_bytes is not None:
                # New mode: use the same data_value
                priority_value = i  # Use index as unique priority value
                batch_data.append((key, priority_value, data_value))
            else:
                # Old mode: use the same field values
                priority_value = i  # Use index as unique priority value
                batch_data.append(tuple([key, priority_value] + field_values))
            
            # When batch reaches specified size or is the last batch, execute batch insert
            if len(batch_data) >= batch_size or i == record_count - 1:
                # Remove error handling for batch insert, any constraint error should abort
                conn.executemany(insert_sql, batch_data)
                processed_count += len(batch_data)
                
                # Show progress
                progress_pct = (processed_count / records_to_load) * 100
                if processed_count % (batch_size * 5) == 0 or i == record_count - 1:
                    print(f"[{db_id}] Progress: {processed_count:,}/{records_to_load:,} ({progress_pct:.1f}%)")
                
                batch_data.clear()
        
        conn.execute("COMMIT")
        print(f"[{db_id}] Fast import completed! Processed {processed_count:,} records")
        
    finally:
        # Keep finally block to ensure database settings are restored
        if original_settings:
            conn.execute(f"PRAGMA journal_mode = {original_settings['journal_mode']}")
            conn.execute(f"PRAGMA synchronous = {original_settings['synchronous']}")
            conn.execute(f"PRAGMA cache_size = {original_settings['cache_size']}")
            conn.execute(f"PRAGMA temp_store = {original_settings['temp_store']}")
            conn.execute("PRAGMA locking_mode = NORMAL")
            print(f"[{db_id}] SQLite settings restored to original values")

    print(f"[{db_id}] Finished loading/verifying data into {conn.filename}.")
    return record_count


class YCSBWorkload:
    def __init__(self, conn, conn_lock, db_id, ycsb_config, general_db_config, initial_record_count, worker_seed_offset, strategy_id: str, output_file=None):
        self.conn = conn
        self.conn_lock = conn_lock
        self.proportions = ycsb_config["ycsb_workload_proportions"]
        self.initial_record_count = initial_record_count
        self.record_count = initial_record_count
        
        # Support new row size mode and old field mode
        self.row_size_bytes = ycsb_config.get("ycsb_row_size_bytes")
        if self.row_size_bytes is not None:
            # New mode: based on total row size
            self.field_count = 1  # Use single data field
            self.field_length = None  # Don't use field_length
        else:
            # Old mode: based on field count and length (backward compatibility)
            self.field_count = ycsb_config.get("ycsb_field_count", 4)
            self.field_length = ycsb_config.get("ycsb_field_length", 1000)
        
        # Core fix: use single, authoritative db_id and strategy_id attributes
        self.db_id = db_id
        self.strategy_id = strategy_id
        
        self.id_generation_lock = threading.Lock()
        self.table_name = get_table_name()
        self.key_prefix = get_key_prefix()
        self._ops_count = 0
        self._total_latency_ms_sum = 0.0
        self.errors = 0
        self.latencies = []
        self.latency_lock = threading.Lock()

        # New logic to handle per-db access patterns
        access_pattern_config = None
        if 'access_pattern_per_db' in ycsb_config and self.db_id in ycsb_config['access_pattern_per_db']:
            access_pattern_config = ycsb_config['access_pattern_per_db'][self.db_id]
        elif 'ycsb_request_distribution' in ycsb_config:
            # Fallback for older config format
            access_pattern_config = {
                "distribution": ycsb_config.get("ycsb_request_distribution", "uniform"),
                "zipf_alpha": ycsb_config.get("ycsb_zipfian_constant", 0.9)
            }
        else:
            # This case will likely cause an error downstream if no pattern is defined, which is intended.
            access_pattern_config = {}

        self.request_distribution = access_pattern_config.get("distribution")
        self.zipfian_constant = access_pattern_config.get("zipf_alpha", 0.9) # Default to 0.9 if not specified

        self.key_padding_length = len(str(self.initial_record_count)) if self.initial_record_count > 0 else 10
        
        self.worker_seed_offset = worker_seed_offset
        self.rng = random.Random(42 + worker_seed_offset)
        
        # Hot spot data set support
        self.hot_rate = ycsb_config.get('hot_rate', 0.167)  # Default 16.7% of data as hot spots
        self.hot_spot_set = []
        self.hot_spot_hit_count = 0  # Count hot spot hits
        self.total_read_count = 0    # Count total reads

        self.output_file = output_file if output_file else f"ycsb_output_db{db_id}.txt"
        
        # Use authoritative self.db_id for logging
        self.logger = logging.getLogger(f"YCSBWorkload.DB_{self.db_id}")
        if not self.logger.handlers:
            log_level_str = general_db_config.get("log_level", "INFO").upper()
            self.logger.setLevel(getattr(logging, log_level_str, logging.INFO))
            handler = logging.StreamHandler()
            formatter = logging.Formatter(f'%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.propagate = False

        self.logger.info(f"YCSBWorkload for db_id '{self.db_id}/{self.strategy_id}' initializing...")

        self.op_choices = []
        for op, prop in self.proportions.items():
            if prop > 0: 
                self.op_choices.extend([op] * int(prop * 1000))

        # Core fix: complete ID generator initialization once and deterministically in constructor
        self.current_max_key_numeric_idx = self._query_max_inserted_key()
        self.logger.info(f"Key generator for '{self.db_id}/{self.strategy_id}' initialized. Starting new keys from index: {self.current_max_key_numeric_idx + 1}")
        
        # Create hot spot data set
        self._create_hot_spot_set()
        
        # Print example queries to confirm workload type
        self.logger.info(f"[{self.db_id}] YCSB workload started - Example queries: SELECT * FROM usertable WHERE priority_value = ? (READ operation), UPDATE usertable SET field0 = ? WHERE id = ? (UPDATE operation)")
        
        # Flag for first operation logging
        self._first_operation_logged = False
        
    def create_schema(self):
        """Create YCSB table structure"""
        self.logger.info("Creating YCSB schema...")
        
        # Use existing create_ycsb_table function
        create_ycsb_table(self.conn, row_size_bytes=self.row_size_bytes)
        
        self.logger.info("YCSB schema created successfully")

    def reset_stats(self):
        """
        Reset internal statistics of this workload instance, mainly latency list and operation count.
        This is crucial for starting new measurements at different experiment phases (e.g., after warmup, before benchmark).
        """
        self.logger.debug("Resetting YCSB workload statistics (latencies, op count)...")
        with self.latency_lock:
            self.latencies.clear()
        self._ops_count = 0
        self._total_latency_ms_sum = 0.0
        self.errors = 0 # Ensure error count is also reset

    def update_access_pattern(self, new_pattern_config: dict, authoritative_record_count: int):
        """
        Dynamically update data access pattern (e.g., distribution type, zipf alpha).
        This replaces the old, incomplete update_distribution and update_zipfian_constant methods.
        Added authoritative_record_count parameter to ensure Zipfian generator uses the latest, actual record count.
        """
        try:
            if not isinstance(new_pattern_config, dict):
                raise TypeError(f"Access pattern configuration must be a dictionary, but received {type(new_pattern_config)}.")
            
            # Prioritize using passed authoritative count value
            if authoritative_record_count > 0:
                if self.initial_record_count != authoritative_record_count:
                    self.logger.info(f"YCSBWorkload internal record count updated from {self.initial_record_count} to authoritative value {authoritative_record_count}")
                self.initial_record_count = authoritative_record_count
            else:
                self.logger.warning("Passed authoritative record count is 0, will continue using internal count value.")

            new_distribution_type = new_pattern_config["distribution"]
            self.logger.info(f"YCSBWorkload is updating access pattern to: {new_pattern_config}")
            
            self.request_distribution = new_distribution_type
            
            if self.request_distribution == "zipfian":
                new_alpha = float(new_pattern_config["zipf_alpha"])
                # Only recreate generator when alpha value changes
                if new_alpha != self.zipfian_constant or not hasattr(self, 'zipf_generator_instance'):
                    self.logger.info(f"Reconfiguring Zipfian generator. Alpha changed from {self.zipfian_constant:.2f} to {new_alpha:.2f}.")
                    self.zipfian_constant = new_alpha
                    # Core fix: if using hot spot set, create generator based on hot spot set size
                    if self.hot_spot_set:
                        zipf_n_param = len(self.hot_spot_set)
                    else:
                        zipf_n_param = self.initial_record_count if self.initial_record_count > 0 else 1
                    self.zipf_generator_instance = zipf_generator(zipf_n_param, self.zipfian_constant, self.rng)
                    self.logger.info(f"Zipfian generator reinitialized, n={zipf_n_param}")

            elif self.request_distribution == "uniform":
                 # For uniform, we don't need specific generator instance since rng.randint() is sufficient
                 self.logger.info("Access pattern switched to Uniform.")
            else:
                raise ValueError(f"Unsupported distribution type '{self.request_distribution}'.")
            
            # If hot rate changes, recreate hot spot set
            if 'hot_rate' in new_pattern_config:
                new_hot_rate = float(new_pattern_config['hot_rate'])
                if new_hot_rate != self.hot_rate:
                    self.hot_rate = new_hot_rate
                    self.logger.info(f"Hot rate changed from {self.hot_rate:.1%} to {new_hot_rate:.1%}")
                    self._create_hot_spot_set()

        except KeyError as e:
            self.logger.error(f"Failed to update access pattern: missing required key in configuration: {e}. Configuration content: {new_pattern_config}")
            raise  # Re-throw exception to fail the program
        except Exception as e:
            self.logger.error(f"Unexpected error occurred while updating access pattern: {e}")
            raise

    def get_total_ops(self):
        """Returns the total number of operations performed by this workload instance."""
        return self._ops_count
    
    def _get_instance_key_prefix(self) -> str:
        """Generate unique key prefix for current workload instance. This should only include database ID to ensure all strategies operate on the same dataset."""
        # Core fix: strategy ID should not be part of the key
        return f"{self.db_id}_{self.key_prefix}"

    def _query_max_inserted_key(self):
        """
        Robustly query the database to find the numeric part of the maximum ID already inserted under this prefix.
        This is core to ensuring ID conflict prevention when reusing database files.
        """
        # If no insert operations, no need to query database
        insert_proportion = self.proportions.get("insert", 0)
        if insert_proportion == 0:
            self.logger.debug(f"No insert operation configured, skipping max ID query")
            return self.initial_record_count
        
        instance_key_prefix = self._get_instance_key_prefix()
        
        sql = f"""
            SELECT MAX(CAST(SUBSTR(id, {len(instance_key_prefix) + 1}) AS INTEGER)) 
            FROM {self.table_name} 
            WHERE id LIKE '{instance_key_prefix}%'
        """
        
        max_id = None
        try:
            with self.conn_lock:
                cursor = self.conn.execute(sql)
                # Core fix: robustly handle query results
                result = cursor.fetchone()
                if result and result[0] is not None:
                    max_id = result[0]
            
            if max_id is not None:
                self.logger.info(f"Recovered maximum inserted ID for '{self.strategy_id}/{self.db_id}' from database: {max_id}")
                return int(max_id)
            else:
                # This situation is normal, meaning the database is brand new, or there are no keys matching this prefix
                # Key fix: even if not found, we should start from initial_record_count to avoid conflicts with initial data
                self.logger.info(f"No keys with format '{instance_key_prefix}%' found in database. Will start from initial_record_count ({self.initial_record_count}).")
                return self.initial_record_count

        except Exception as e:
            self.logger.error(f"Serious error occurred while querying maximum inserted ID for '{self.strategy_id}/{self.db_id}': {e}. Will fall back to safe mode.")
            # Fallback under serious error: return an absolutely non-conflicting, very large unique value
            unique_fallback = self.initial_record_count + int(time.time() * 1e6) + abs(hash(self.db_id))
            return unique_fallback

    
    def _create_hot_spot_set(self):
        """Create hot spot data set"""
        hot_spot_count = int(self.initial_record_count * self.hot_rate)
        hot_spot_count = max(1, hot_spot_count)
        
        # Randomly select hot spots
        all_indices = list(range(self.initial_record_count))
        self.rng.shuffle(all_indices)
        self.hot_spot_set = sorted(all_indices[:hot_spot_count])
        
        # Initialize Zipfian generator
        if self.request_distribution == 'zipfian':
            self.zipf_generator_instance = zipf_generator(len(self.hot_spot_set), self.zipfian_constant, self.rng)
            
        self.logger.info(f"Created hot spot set with {len(self.hot_spot_set)} items "
                        f"({self.hot_rate * 100:.1f}% of data)")
                        
    def generate_initial_data(self, record_count: int):
        """Generate YCSB initial data"""
        self.logger.info(f"Generating {record_count} YCSB records...")
        
        self.initial_record_count = record_count
        self.record_count = record_count
        
        # Batch insert optimization
        self.conn.execute("PRAGMA synchronous = OFF")
        self.conn.execute("PRAGMA journal_mode = OFF")
        self.conn.execute("BEGIN TRANSACTION")
        
        try:
            # Calculate data size (excluding other fields)
            # id (about 20 bytes) + priority_value (8 bytes) + overhead (about 22 bytes) = 50 bytes
            data_field_size = max(1, self.row_size_bytes - 50) if self.row_size_bytes else None
            
            if self.row_size_bytes is not None:
                data_value = generate_random_string(data_field_size, self.worker_seed_offset)
            else:
                # Old mode: generate field values
                data_value = None
                field_values = [
                    generate_random_string(self.field_length, self.worker_seed_offset) 
                    for _ in range(self.field_count)
                ]
            
            table_name = get_table_name()
            key_prefix = get_key_prefix()
            
            # Batch insert
            batch_size = 1000
            for i in range(record_count):
                key = f"{self.db_id}_{key_prefix}{i}"
                priority_value = i
                
                if self.row_size_bytes is not None:
                    self.conn.execute(
                        f"INSERT INTO {table_name} VALUES (?, ?, ?)",
                        (key, priority_value, data_value)
                    )
                else:
                    # Old mode insert
                    all_values = [key, priority_value] + field_values
                    field_names = ', '.join(f'field{j}' for j in range(self.field_count))
                    placeholders = ", ".join(["?"] * len(all_values))
                    self.conn.execute(
                        f"INSERT INTO {table_name} (id, priority_value, {field_names}) VALUES ({placeholders})",
                        all_values
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
            
    def generate_operation(self) -> Tuple[str, Dict[str, Any]]:
        """Generate next YCSB operation"""
        # Check if there are available records
        if self.initial_record_count <= 0:
            self.logger.warning(f"No available records (record_count={self.initial_record_count}), skipping operation")
            # Return a dummy insert operation
            return ('insert', {
                'id': f"{self.db_id}_dummy_0",
                'priority_value': 0,
                'data': 'dummy_data'
            })
        
        # Choose operation type
        op_type = self._choose_operation_type()
        
        if op_type == 'read':
            priority_value = self._generate_priority_value_to_operate_on()
            
            # Count hot spot access
            self.total_read_count += 1
            if self.hot_spot_set and priority_value in self.hot_spot_set:
                self.hot_spot_hit_count += 1
            
            return ('read', {'priority_value': priority_value})
            
        elif op_type == 'update':
            # Update uses ID instead of priority_value
            record_id = self._generate_key_to_operate_on()
            new_priority = self.rng.randint(0, max(0, self.initial_record_count - 1))
            data_size = max(1, self.row_size_bytes - 50) if self.row_size_bytes else self.field_length
            new_data = generate_random_string(data_size, self.worker_seed_offset + self._ops_count)
            
            return ('update', {
                'id': record_id,
                'new_priority': new_priority,
                'new_data': new_data
            })
            
        elif op_type == 'insert':
            # Generate new key
            new_key = self._generate_key_to_insert()
            new_priority = self.current_max_key_numeric_idx
            data_size = max(1, self.row_size_bytes - 50) if self.row_size_bytes else self.field_length
            new_data = generate_random_string(data_size, self.worker_seed_offset + self._ops_count)
            
            return ('insert', {
                'id': new_key,
                'priority_value': new_priority,
                'data': new_data
            })
            
        elif op_type == 'scan':
            start_key = self._generate_key_to_operate_on()
            scan_length = self.rng.randint(1, 100)
            
            return ('scan', {
                'start_key': start_key,
                'scan_length': scan_length
            })
            
        elif op_type == 'read_modify_write':
            record_id = self._generate_key_to_operate_on()
            return ('read_modify_write', {'id': record_id})
            
        else:
            # Default to return read operation
            priority_value = self._generate_priority_value_to_operate_on()
            return ('read', {'priority_value': priority_value})
            
    def _choose_operation_type(self) -> str:
        """Choose operation type based on workload proportions"""
        rand = self.rng.random()
        cumulative = 0.0
        
        for op_type, proportion in self.proportions.items():
            cumulative += proportion
            if rand < cumulative:
                return op_type
                
        return 'read'  # Default
        
    def execute_operation(self, op_type: str, params: Dict[str, Any]) -> bool:
        """Execute YCSB operation"""
        try:
            table_name = get_table_name()
            
            if op_type == 'read':
                with self.conn_lock:
                    cursor = self.conn.execute(
                        f"SELECT 1 FROM {table_name} WHERE priority_value = ? LIMIT 1",
                        (params['priority_value'],)
                    )
                    return cursor.fetchone() is not None
                    
            elif op_type == 'update':
                with self.conn_lock:
                    self.conn.execute(
                        f"UPDATE {table_name} SET priority_value = ?, data = ? WHERE id = ?",
                        (params['new_priority'], params['new_data'], params['id'])
                    ) if self.row_size_bytes else self.conn.execute(
                        f"UPDATE {table_name} SET priority_value = ?, field0 = ? WHERE id = ?",
                        (params['new_priority'], params['new_data'], params['id'])
                    )
                    return self.conn.changes() > 0
                    
            elif op_type == 'insert':
                with self.conn_lock:
                    self.conn.execute(
                        f"INSERT INTO {table_name} VALUES (?, ?, ?)",
                        (params['id'], params['priority_value'], params['data'])
                    )
                    return True
                    
            elif op_type == 'scan':
                with self.conn_lock:
                    cursor = self.conn.execute(
                        f"SELECT * FROM {table_name} WHERE id >= ? ORDER BY id LIMIT ?",
                        (params['start_key'], params['scan_length'])
                    )
                    results = cursor.fetchall()
                    return len(results) > 0
                    
            elif op_type == 'read_modify_write':
                with self.conn_lock:
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
            self._ops_count += 1

    def _generate_key_to_insert(self) -> str:
        """
        Generate a key guaranteed to be unique in this strategy run.
        """
        with self.id_generation_lock:
            self.current_max_key_numeric_idx += 1
            new_id = self.current_max_key_numeric_idx
        # Format: <strategy_id>_<db_id>_user<numeric_id>
        return f"{self._get_instance_key_prefix()}{new_id}"

    def _generate_key_to_operate_on(self) -> str:
        """
        Generate a target key for read/update operations.
        This key should point to initially loaded data and be selected from hot spot set.
        """
        if not self.hot_spot_set:
            # If hot spot set is empty, fall back to original logic
            if self.request_distribution == 'uniform':
                key_numeric_idx = self.rng.randint(0, self.initial_record_count - 1)
            elif self.request_distribution == 'zipfian':
                key_numeric_idx = next(self.zipf_generator_instance)
            else:
                key_numeric_idx = self.rng.randint(0, self.initial_record_count - 1)
        else:
            # Select from hot spot set
            if self.request_distribution == 'uniform':
                # Uniform distribution: randomly select from hot spot set
                hot_index = self.rng.randint(0, len(self.hot_spot_set) - 1)
                key_numeric_idx = self.hot_spot_set[hot_index]
            elif self.request_distribution == 'zipfian':
                # Zipfian distribution: use generator to select index in hot spot set
                zipf_index = next(self.zipf_generator_instance)
                # Ensure index is within range
                zipf_index = min(zipf_index, len(self.hot_spot_set) - 1)
                key_numeric_idx = self.hot_spot_set[zipf_index]
            else:
                # Default to uniform
                hot_index = self.rng.randint(0, len(self.hot_spot_set) - 1)
                key_numeric_idx = self.hot_spot_set[hot_index]
                
        # Core fix: ensure generated key has the same db_id prefix and zero-padding format as during data loading
        instance_key_prefix = self._get_instance_key_prefix()
        key = f"{instance_key_prefix}{key_numeric_idx:0{self.key_padding_length}d}"
        
        return key
    
    def _generate_priority_value_to_operate_on(self) -> int:
        """
        Generate a target priority value for read operations.
        Generate priority value from hot spot set based on configured distribution.
        """
        if not self.hot_spot_set:
            # If hot spot set is empty, fall back to original logic
            if self.request_distribution == 'uniform':
                return self.rng.randint(0, self.initial_record_count - 1)
            elif self.request_distribution == 'zipfian':
                zipf_index = next(self.zipf_generator_instance)
                return zipf_index
            else:
                return self.rng.randint(0, self.initial_record_count - 1)
        else:
            # Select from hot spot set
            if self.request_distribution == 'uniform':
                # Uniform distribution: randomly select from hot spot set
                hot_index = self.rng.randint(0, len(self.hot_spot_set) - 1)
                return self.hot_spot_set[hot_index]
            elif self.request_distribution == 'zipfian':
                # Zipfian distribution: use generator to select index in hot spot set
                zipf_index = next(self.zipf_generator_instance)
                # Ensure index is within range
                zipf_index = min(zipf_index, len(self.hot_spot_set) - 1)
                selected_value = self.hot_spot_set[zipf_index]
                # Occasionally log selected value
                if self._ops_count % 500 == 0:
                    self.logger.debug(f"[{self.db_id}] Zipfian selection: index={zipf_index}, priority_value={selected_value}")
                return selected_value
            else:
                # Default to uniform
                hot_index = self.rng.randint(0, len(self.hot_spot_set) - 1)
                return self.hot_spot_set[hot_index]

    # _execute_operation method has been removed, functionality moved to execute_operation and run_operation

    def run_operation(self):
        """
        Select a random operation based on configured proportions and execute it.
        This is the main entry point for DatabaseInstance to run a single workload unit.
        """
        start_time = time.perf_counter()
        
        # Generate operation
        op_type, params = self.generate_operation()
        
        # Log specific operation type and parameters on first operation
        if not self._first_operation_logged:
            self.logger.info(f"[{self.db_id}] First YCSB operation: {op_type} - priority_value={params.get('priority_value', 'N/A')}, key={params.get('key', 'N/A')[:20]}...")
            self._first_operation_logged = True
        
        # Execute operation
        success = self.execute_operation(op_type, params)
        
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        
        # Record latency and statistics
        if success and latency_ms > 0:
            with self.latency_lock:
                self.latencies.append(latency_ms)
            self._total_latency_ms_sum += latency_ms
        else:
            self.errors += 1
            
        if self._ops_count % 1000 == 0:
            self.logger.debug(f"Executed {self._ops_count} operations.")
            
        return latency_ms

    def get_stats_and_reset(self) -> tuple:
        """
        Get statistics since last reset (operation count, total latency, P95/P99 latency and error count),
        then reset these statistics.
        Returns a tuple containing (ops, total_latency_us, p95_us, p99_us, errors).
        """
        with self.latency_lock:
            # If no latency data, return 0 directly
            if not self.latencies:
                # Reset counters
                ops_count = self._ops_count
                total_latency_us = self._total_latency_ms_sum * 1000 # Convert to microseconds
                errors_count = self.errors
                self._ops_count = 0
                self._total_latency_ms_sum = 0.0
                self.errors = 0
                return ops_count, total_latency_us, 0, 0, 0, 0, 0, errors_count

            # Copy list for calculation while releasing lock
            latencies_copy = self.latencies.copy()
            ops_count = self._ops_count
            total_latency_us = self._total_latency_ms_sum * 1000 # Convert to microseconds
            errors_count = self.errors
            
            # Reset internal state
            self.latencies.clear()
            self._ops_count = 0
            self._total_latency_ms_sum = 0.0
            self.errors = 0

        # Perform expensive percentile calculations outside of lock
        # Convert latency from milliseconds to microseconds
        latencies_us_copy = [l * 1000 for l in latencies_copy]
        if latencies_us_copy:
            p50_us = np.percentile(latencies_us_copy, 50)
            p90_us = np.percentile(latencies_us_copy, 90)
            p95_us = np.percentile(latencies_us_copy, 95)
            p99_us = np.percentile(latencies_us_copy, 99)
            p999_us = np.percentile(latencies_us_copy, 99.9)
        else:
            p50_us = p90_us = p95_us = p99_us = p999_us = 0
        
        return ops_count, total_latency_us, p50_us, p90_us, p95_us, p99_us, p999_us, errors_count

    def _read(self, priority_value):
        """Execute single READ operation, query based on priority_value column."""
        with self.conn_lock:
            cursor = self.conn.execute(f"SELECT 1 FROM {self.table_name} WHERE priority_value = ? LIMIT 1", (priority_value,))
            # Must call fetchone() or fetchall() to actually execute the query
            result = cursor.fetchone()
            self.logger.debug(f"Read operation for priority_value {priority_value}, result: {'found' if result else 'not found'}")

    def _update(self, key):
        """Execute single UPDATE operation."""
        # This method is no longer used, functionality moved to execute_operation
        pass

    def _insert(self, key):
        """Execute single INSERT operation."""
        # This method is no longer used, functionality moved to execute_operation
        pass

    def _scan(self, key):
        """Execute single SCAN operation. The passed key is ignored, only used to maintain interface consistency."""
        # This method is no longer used, functionality moved to execute_operation
        pass