#!/usr/bin/env python3
"""
Database Manager
Responsible for checking whether existing databases meet configuration requirements, avoiding duplicate database generation
"""

import os
import json
import hashlib
import apsw
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import time
import shutil
import logging


class DatabaseManager:
    """Database Manager"""
    
    def __init__(self):
        self.db_cache_dir = Path("data/db_cache")
        self.db_cache_dir.mkdir(parents=True, exist_ok=True)
        self.config_hash_file = self.db_cache_dir / "db_config_hashes.json"
        
    def get_db_config_hash(self, db_config: Dict[str, Any], workload_type: str = "ycsb") -> str:
        """Calculate hash value of database configuration"""
        # Extract key configurations that affect database generation
        key_config = {
            "workload_type": workload_type,
            "db_filename": db_config.get("db_filename"),
            "page_size": db_config.get("page_size", 4096)
        }
        
        # Add specific configurations based on workload type
        if workload_type == "ycsb":
            key_config.update({
                "ycsb_initial_record_count": db_config.get("ycsb_initial_record_count"),
                "ycsb_field_count": db_config.get("field_count", 4),
                "ycsb_field_length": db_config.get("field_length", 100),
            })
        elif workload_type == "tpcc":
            key_config.update({
                "tpcc_warehouses": db_config.get("tpcc_warehouses", 1)
            })
        elif workload_type == "trace":
            key_config.update({
                "trace_partition": db_config.get("trace_partition", "0-100%")
            })
        
        # Serialize and calculate hash
        config_str = json.dumps(key_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def get_db_metadata_cache_path(self, db_filename: str) -> Path:
        """Get database metadata cache file path"""
        db_path = Path(db_filename)
        # Use database filename (without path) to store metadata in cache directory
        cache_filename = f"{db_path.name}.metadata_cache"
        return self.db_cache_dir / cache_filename
    
    def has_insert_or_delete_operations(self, ycsb_config: Dict[str, Any]) -> bool:
        """Check if YCSB configuration contains Insert or Delete operations"""
        workload_proportions = ycsb_config.get("ycsb_workload_proportions", {})
        
        insert_ratio = workload_proportions.get("insert", 0)
        # Note: Delete operations are usually not directly configured in YCSB, but may exist in some workload types
        # Here we mainly check insert, can be extended if delete support is needed in the future
        
        return insert_ratio > 0
    
    def save_database_metadata(self, db_filename: str, metadata: Dict[str, Any]) -> None:
        """Save database metadata to cache file"""
        cache_path = self.get_db_metadata_cache_path(db_filename)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            print(f"✅ Database metadata cached: {cache_path.name}")
        except Exception as e:
            print(f"Warning: Failed to save database metadata: {e}")
    
    def load_database_metadata(self, db_filename: str) -> Optional[Dict[str, Any]]:
        """Load database metadata from cache file"""
        cache_path = self.get_db_metadata_cache_path(db_filename)
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load database metadata: {e}")
            return None
        
    def load_db_hashes(self) -> Dict[str, str]:
        """Load database configuration hash records"""
        if self.config_hash_file.exists():
            try:
                with open(self.config_hash_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
        
    def save_db_hashes(self, hashes: Dict[str, str]) -> None:
        """Save database configuration hash records"""
        try:
            with open(self.config_hash_file, 'w', encoding='utf-8') as f:
                json.dump(hashes, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save database hash: {e}")
            
    def verify_database_integrity(self, db_file: str, expected_record_count: int, 
                                workload_config: Dict[str, Any] = None, 
                                workload_type: str = "ycsb") -> bool:
        """Verify database integrity, supports metadata caching to reduce repeated verification time"""
        if not os.path.exists(db_file):
            return False
        
        # Check if cache verification can be used
        if workload_config and workload_type == "ycsb" and not self.has_insert_or_delete_operations(workload_config):
            # If no Insert/Delete operations, try to load cached metadata
            cached_metadata = self.load_database_metadata(db_file)
            if cached_metadata:
                print(f"✅ Using cached metadata to verify {db_file}")
                # Verify if cached metadata matches expected values
                cached_count = cached_metadata.get("record_count", 0)
                tolerance = max(1, int(expected_record_count * 0.20))
                min_acceptable = expected_record_count - tolerance
                max_acceptable = expected_record_count + tolerance
                
                if min_acceptable <= cached_count <= max_acceptable:
                    print(f"✅ Cache verification passed: {cached_count} records (expected {expected_record_count}±{tolerance})")
                    return True
                else:
                    print(f"⚠️ Cached record count mismatch, performing full verification")
                    # Continue with full verification
            else:
                print(f"ℹ️ No metadata cache found, performing full verification")
        else:
            print(f"ℹ️ Workload type: {workload_type}, performing full verification")
            
        # Execute complete database validation
        conn = None
        try:
            conn = apsw.Connection(db_file)
            cursor = conn.cursor()
            
            # Validate based on workload type
            if workload_type == "ycsb":
                # Check if table exists - YCSB uses usertable
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='usertable'")
                if not cursor.fetchone():
                    print(f"Database {db_file} missing usertable")
                    return False
                
                # Check record count - allow 20% tolerance range
                cursor.execute("SELECT COUNT(*) FROM usertable")
                actual_count = cursor.fetchone()[0]
                
                # Calculate tolerance range (±20%)
                tolerance = max(1, int(expected_record_count * 0.20))  # At least tolerate 1 record difference
                min_acceptable = expected_record_count - tolerance
                max_acceptable = expected_record_count + tolerance
                
                if not (min_acceptable <= actual_count <= max_acceptable):
                    print(f"Database {db_file} record count out of tolerance: expected {expected_record_count}±{tolerance}, actual {actual_count}")
                    return False
                    
                # Check table structure
                cursor.execute("PRAGMA table_info(usertable)")
                columns = cursor.fetchall()
                
                # Verify based on actual YCSB table structure
                actual_columns = [col[1] for col in columns]  # col[1] is column name
                
                # Check required columns: primary key column + field columns
                required_columns = ['id']  # Primary key column
                # Dynamically check field columns based on configured field count
                field_count = 4  # Default value
                if workload_config:
                    field_count = workload_config.get("ycsb_field_count", 4)
                
                for i in range(field_count):
                    required_columns.append(f'field{i}')
                
                missing_columns = [col for col in required_columns if col not in actual_columns]
                if missing_columns:
                    print(f"Database {db_file} table structure mismatch: missing columns {missing_columns}")
                    print(f"  Expected columns: {required_columns}")
                    print(f"  Actual columns: {actual_columns}")
                    return False
                    
            elif workload_type == "tpcc":
                # TPC-C verification: check if 9 tables exist
                tpcc_tables = ['warehouse', 'district', 'customer', 'history', 
                             'orders', 'new_order', 'item', 'stock', 'order_line']
                for table in tpcc_tables:
                    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
                    if not cursor.fetchone():
                        print(f"Database {db_file} missing TPC-C table: {table}")
                        return False
                
                # Check warehouse table record count (other tables' record counts depend on warehouse count)
                cursor.execute("SELECT COUNT(*) FROM warehouse")
                actual_count = cursor.fetchone()[0]
                print(f"✅ TPC-C database verification passed: {actual_count} warehouses")
                
            elif workload_type == "trace":
                # Trace verification: check block_trace table
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='block_trace'")
                if not cursor.fetchone():
                    print(f"Database {db_file} missing block_trace table")
                    return False
                
                # Check record count
                cursor.execute("SELECT COUNT(*) FROM block_trace")
                actual_count = cursor.fetchone()[0]
                print(f"✅ Trace database verification passed: {actual_count} trace records")
            
            # Verification successful, save metadata to cache
            if workload_type == "ycsb" and workload_config and not self.has_insert_or_delete_operations(workload_config):
                metadata = {
                    "workload_type": workload_type,
                    "record_count": actual_count,
                    "table_columns": actual_columns if workload_type == "ycsb" else [],
                    "validation_time": time.time(),
                    "expected_count": expected_record_count,
                    "field_count": field_count if workload_type == "ycsb" else 0
                }
                self.save_database_metadata(db_file, metadata)
                
            print(f"✅ Database {db_file} verification passed (workload={workload_type})")
            return True
            
        except Exception as e:
            print(f"Database {db_file} verification failed: {e}")
            return False
        finally:
            # Ensure connection is completely released
            if conn:
                    conn.close()
            
    def check_database_reusability(self, db_config: Dict[str, Any], 
                                  workload_config: Dict[str, Any],
                                  workload_type: str = "ycsb") -> Tuple[bool, str]:
        """
        Check if database can be reused
        
        Returns:
            (can_reuse, reason)
        """
        db_filename = db_config.get("db_filename")
        if not db_filename:
            return False, "Missing database filename configuration"
            
        # Calculate current configuration hash
        current_config = {
            **db_config,
            "page_size": workload_config.get("page_size_bytes", 4096)
        }
        
        # Add specific configurations based on workload type
        if workload_type == "ycsb":
            current_config.update({
                "field_count": workload_config.get("ycsb_field_count", 4),
                "field_length": workload_config.get("ycsb_field_length", 100)
            })
        
        current_hash = self.get_db_config_hash(current_config, workload_type)
        
        # Load saved hashes
        saved_hashes = self.load_db_hashes()
        saved_hash = saved_hashes.get(db_filename)
        
        if not saved_hash:
            return False, "Database not cached before"
            
        if saved_hash != current_hash:
            return False, f"Configuration changed (cached: {saved_hash[:8]}... vs current: {current_hash[:8]}...)"
            
        # Verify database file integrity (now supports metadata caching)
        expected_records = 0
        if workload_type == "ycsb":
            expected_records = db_config.get("ycsb_initial_record_count", 0)
        elif workload_type == "tpcc":
            expected_records = db_config.get("tpcc_warehouses", 1)
        
        if not self.verify_database_integrity(db_filename, expected_records, workload_config, workload_type):
            return False, "Database file corrupted or incomplete"
            
        return True, f"Configuration matches, can be reused directly (hash: {current_hash[:8]}...)"
        
    def mark_database_generated(self, db_config: Dict[str, Any], 
                               workload_config: Dict[str, Any],
                               workload_type: str = "ycsb") -> None:
        """Mark database as generated, save configuration hash"""
        db_filename = db_config.get("db_filename")
        if not db_filename:
            return
            
        current_config = {
            **db_config,
            "page_size": workload_config.get("page_size_bytes", 4096)
        }
        
        if workload_type == "ycsb":
            current_config.update({
                "field_count": workload_config.get("ycsb_field_count", 4),
                "field_length": workload_config.get("ycsb_field_length", 100)
            })
            
        current_hash = self.get_db_config_hash(current_config, workload_type)
        
        # Save hash
        saved_hashes = self.load_db_hashes()
        saved_hashes[db_filename] = current_hash
        self.save_db_hashes(saved_hashes)
        
        print(f"✅ Database {db_filename} generation completed (workload={workload_type}), configuration hash saved: {current_hash[:8]}...")
        
    def clean_invalid_databases(self) -> None:
        """Clean up invalid database files"""
        saved_hashes = self.load_db_hashes()
        valid_hashes = {}
        
        for db_filename, hash_value in saved_hashes.items():
            if os.path.exists(db_filename):
                valid_hashes[db_filename] = hash_value
            else:
                print(f"Cleaning invalid database record: {db_filename}")
                
        if len(valid_hashes) != len(saved_hashes):
            self.save_db_hashes(valid_hashes)
            print(f"Cleaned {len(saved_hashes) - len(valid_hashes)} invalid database records")
            
    def get_database_summary(self) -> Dict[str, Any]:
        """Get database cache summary"""
        summary = {}
        db_hashes = self.load_db_hashes()
        
        for db_filename in db_hashes.keys():
            if not os.path.exists(db_filename):
                continue
                
            try:
                # Use apsw connection
                conn = apsw.Connection(db_filename)
                cursor = conn.cursor()
                
                # Prefer to get record count from cache
                record_count = None
                cached_metadata = self.load_database_metadata(db_filename)
                if cached_metadata:
                    record_count = cached_metadata.get("record_count")
                
                # If cache is not available, execute query
                if record_count is None:
                    cursor.execute("SELECT COUNT(*) FROM usertable")
                    record_count = cursor.fetchone()[0]
                
                # Get file size
                file_size = os.path.getsize(db_filename)
                
                db_summary = {
                    "hash": db_hashes[db_filename][:8] + "...",
                    "exists": os.path.exists(db_filename),
                    "size_mb": file_size / (1024 * 1024),
                    "record_count": record_count
                }
                
                summary[db_filename] = db_summary
                
            except Exception as e:
                summary[db_filename] = {"error": str(e)}
            finally:
                if 'conn' in locals() and conn:
                    conn.close()

        return summary


# Global database manager instance
db_manager = DatabaseManager()


# Cache validated databases to avoid repeated checks
_validated_databases = {}
_validation_cache_ttl = 300  # 5-minute cache

def check_and_reuse_database(db_config: Dict[str, Any], 
                             workload_config: Dict[str, Any],
                             workload_type: str = "ycsb") -> bool:
    """
    Check if database meets configuration requirements, if so it can be reused.
    Optimized version: uses metadata cache and memory cache to avoid repeated validation of the same database.
    """
    db_filename = db_config.get("db_filename")
    if not db_filename:
        print("❌ Validation failed: 'db_filename' missing in configuration.")
        return False

    # Check memory cache to avoid repeated validation
    current_time = time.time()
    cache_key = f"{db_filename}:{workload_type}"
    if cache_key in _validated_databases:
        cached_result, cache_time = _validated_databases[cache_key]
        if current_time - cache_time < _validation_cache_ttl:
            # If no Insert/Delete operations, can trust cache for longer time
            if workload_type == "ycsb" and workload_config and not db_manager.has_insert_or_delete_operations(workload_config):
                print(f"✅ Using memory cache verification result: {db_filename} (no Insert/Delete operations)")
                return cached_result == 'valid'
            elif cached_result == 'valid':
                print(f"✅ Using memory cache verification result: {db_filename}")
                return True

    # 1. Check if physical file exists
    try:
        file_exists = os.path.exists(db_filename)
    except (ConnectionAbortedError, OSError, IOError) as e:
        print(f"🌐 Network error accessing database file {db_filename}: {type(e).__name__}: {e}")
        print(f"🔄 Will skip pre-check and create database on demand")
        _validated_databases[cache_key] = ('network_error', current_time)
        return False
    
    if not file_exists:
        result = f"File does not exist"
        _validated_databases[cache_key] = ('missing', current_time)
        print(f"ℹ️  Validation info: Database file {db_filename} does not exist, needs to be generated.")
        return False

    # 2. Success marker check has been removed, proceed directly to next verification

    # 3. If no Insert/Delete operations, try to use metadata cache for quick verification
    if workload_type == "ycsb" and workload_config and not db_manager.has_insert_or_delete_operations(workload_config):
        cached_metadata = db_manager.load_database_metadata(db_filename)
        if cached_metadata:
            expected_records = db_config.get("ycsb_initial_record_count", 0)
            cached_count = cached_metadata.get("record_count", 0)
            tolerance = max(1, int(expected_records * 0.20))
            min_acceptable = expected_records - tolerance
            max_acceptable = expected_records + tolerance
            
            if min_acceptable <= cached_count <= max_acceptable:
                print(f"✅ Metadata cache verification passed: {db_filename} ({cached_count} records)")
                _validated_databases[cache_key] = ('valid', current_time)
                return True
            else:
                print(f"⚠️ Metadata cache record count mismatch, performing full verification")

    # 4. Lightweight content verification (only verify table existence and basic record count)
    conn = None
    try:
        conn = apsw.Connection(str(db_filename), flags=apsw.SQLITE_OPEN_READONLY)
        cursor = conn.cursor()

        # Check different tables based on workload type
        if workload_type == "ycsb":
            # Check if it's B5 multi-table mode
            if db_config.get('b5_multi_table', False):
                # B5 multi-table mode: check if there are tables starting with ycsb_data_
                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name LIKE 'ycsb_data_%'")
                table_count = cursor.fetchone()[0]
                if table_count == 0:
                    result = "Missing ycsb_data_* tables"
                    _validated_databases[cache_key] = ('invalid', current_time)
                    print(f"❌ Validation failed: {db_filename} exists but missing 'ycsb_data_*' tables.")
                    return False
                
                # Get total record count from all tables
                cursor.execute("SELECT SUM(cnt) FROM (SELECT COUNT(*) as cnt FROM sqlite_master WHERE type='table' AND name LIKE 'ycsb_data_%')")
                actual_records = 1  # As long as there are tables, consider it non-empty, detailed verification is done later
            else:
                # Standard single-table mode
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='usertable'")
                if not cursor.fetchone():
                    result = "Missing usertable table"
                    _validated_databases[cache_key] = ('invalid', current_time)
                    print(f"❌ Validation failed: {db_filename} exists but missing 'usertable' table.")
                    return False

                # Quick record count check (only check if empty)
                cursor.execute("SELECT COUNT(*) FROM usertable LIMIT 1")
                actual_records = cursor.fetchone()[0]
        elif workload_type == "tpcc":
            # Check if it's B5 multi-table mode
            if db_config.get('b5_multi_table', False):
                # B5 multi-table mode: check if there are warehouse tables with prefix
                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name LIKE '%_warehouse'")
                table_count = cursor.fetchone()[0]
                if table_count == 0:
                    result = "Missing *_warehouse tables"
                    _validated_databases[cache_key] = ('invalid', current_time)
                    print(f"❌ Validation failed: {db_filename} exists but missing '*_warehouse' tables.")
                    return False
                actual_records = 1  # As long as there are tables, consider it non-empty
            else:
                # Standard single-table mode
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='warehouse'")
                if not cursor.fetchone():
                    result = "Missing warehouse table"
                    _validated_databases[cache_key] = ('invalid', current_time)
                    print(f"❌ Validation failed: {db_filename} exists but missing 'warehouse' table.")
                    return False
                cursor.execute("SELECT COUNT(*) FROM warehouse LIMIT 1")
                actual_records = cursor.fetchone()[0]
        elif workload_type == "trace":
            # Check block_trace table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='block_trace'")
            if not cursor.fetchone():
                result = "Missing block_trace table"
                _validated_databases[cache_key] = ('invalid', current_time)
                print(f"❌ Validation failed: {db_filename} exists but missing 'block_trace' table.")
                return False
            cursor.execute("SELECT COUNT(*) FROM block_trace LIMIT 1")
            actual_records = cursor.fetchone()[0]
        
        if actual_records == 0:
            result = "Table is empty"
            _validated_databases[cache_key] = ('invalid', current_time)
            print(f"❌ Validation failed: {db_filename} main table is empty.")
            return False
            
        # Cache successful verification result
        _validated_databases[cache_key] = ('valid', current_time)
        print(f"✅ Validation passed: {db_filename} file exists with data. Can be reused. (workload={workload_type})")
        return True

    except Exception as e:
        result = f"Access error: {e}"
        _validated_databases[cache_key] = ('error', current_time)
        print(f"❌ Validation failed: Error opening or querying {db_filename} - {e}. File may be corrupted.")
        return False
    finally:
        if conn:
            conn.close()


# mark_database_created function has been deleted, no longer need success marker mechanism


def print_database_cache_summary():
    """Print database cache summary information"""
    summary = db_manager.get_database_summary()
    
    print("\n📊 Database cache summary:")
    print(f"  Number of cached databases: {len(summary)}")
    
    for db_file, info in summary.items():
        status = "✅ Exists" if info["exists"] else "❌ Missing"
        print(f"  📄 {os.path.basename(db_file)}: {status} | "
              f"{info['size_mb']:.1f}MB | {info['record_count']:,} records | "
              f"hash: {info['hash']}")
    
    time.sleep(1.0)  # Wait 1 second to ensure connection is released


def prepare_database_for_strategy(strategy_id: str, db_config: dict, base_db_path: str, live_db_path: str, logger: logging.Logger, workload_type: str = None):
    """Prepare database for given strategy: copy from base copy or reuse existing copy. Optimized version reduces duplicate checks."""
    
    db_id = db_config.get('id', 'UnknownDB')
    log_prefix = f"[{strategy_id}][{db_id}]"

    if os.path.exists(live_db_path):
        # Quick validation of existing active database
        try:
            # Simple validation: check file size and basic structure
            stat = os.stat(live_db_path)
            if stat.st_size > 0:
                logger.info(f"{log_prefix} Reusing existing active database '{live_db_path}' ({stat.st_size/1024/1024:.1f}MB)")
                return
        except Exception as e:
            logger.warning(f"{log_prefix} Existing database validation failed: {e}, will re-copy")
            os.remove(live_db_path)

    if not os.path.exists(base_db_path):
        logger.error(f"{log_prefix} Base database '{base_db_path}' does not exist. Cannot prepare database for strategy.")
        raise FileNotFoundError(f"Base database not found: {base_db_path}")

    try:
        logger.info(f"{log_prefix} Copying from base database...")
        # Key step to get Insert ID: quickly query max ID
        conn = None
        try:
            conn = apsw.Connection(base_db_path, flags=apsw.SQLITE_OPEN_READONLY)
            cursor = conn.cursor()
            
            # Query different tables based on workload type
            if workload_type == 'tpcc':
                # TPC-C has no ROWID concept, use warehouse count
                cursor.execute("SELECT COUNT(*) FROM warehouse")
                max_id = cursor.fetchone()[0]
                logger.debug(f"{log_prefix} TPC-C database warehouse count: {max_id}")
            elif workload_type == 'trace':
                # Trace database uses block_trace table
                cursor.execute("SELECT COUNT(*) FROM block_trace")
                max_id = cursor.fetchone()[0]
                logger.debug(f"{log_prefix} Trace database record count: {max_id}")
            else:
                # Default to YCSB
                cursor.execute("SELECT MAX(ROWID) FROM usertable")
                max_id = cursor.fetchone()[0]
                logger.debug(f"{log_prefix} Base database max ID: {max_id}")
        except Exception as e:
            logger.warning(f"{log_prefix} Unable to get max ID: {e}")
        finally:
            if conn:
                conn.close()
        
        shutil.copy2(base_db_path, live_db_path)
        logger.info(f"{log_prefix} Database copy completed")
    except Exception as e:
        logger.error(f"{log_prefix} Database copy failed: {e}", exc_info=True)
        raise

def create_base_database_from_config(db_config: Dict[str, Any], config: Dict[str, Any], logger: logging.Logger) -> None:
    """
    Check or create base database file, optimized version reduces redundant checks, supports metadata caching
    """
    from database_instance import DatabaseInstance
    
    db_filename = db_config.get("db_filename")
    if not db_filename:
        raise ValueError("Database config missing 'db_filename'")
    
    # Get workload type
    workload_type = config.get("general_experiment_setup", {}).get("workload_type", "ycsb")
    
    logger.info(f"Checking database: {db_filename} (workload={workload_type})")
    
    # Get workload configuration
    workload_config = None
    if workload_type == "ycsb":
        workload_config = config.get("ycsb_general_config", {})
    elif workload_type == "tpcc":
        workload_config = config.get("tpcc_general_config", {})
    elif workload_type == "trace":
        workload_config = config.get("trace_general_config", {})
    
    # Use quick check to avoid repeated verification (now supports metadata caching)
    can_reuse, reason = db_manager.check_database_reusability(db_config, workload_config, workload_type)
    
    if can_reuse:
        logger.info(f"✅ Reusing existing database: {db_filename} ({reason})")
        return
    
    logger.info(f"🔄 Need to create database: {reason}")
    
    # Delete old file (if exists), including metadata cache
    if os.path.exists(db_filename):
        os.remove(db_filename)
        logger.info(f"Deleted old database file: {db_filename}")
        
    # Delete old metadata cache file
    metadata_cache_path = db_manager.get_db_metadata_cache_path(db_filename)
    if metadata_cache_path.exists():
        metadata_cache_path.unlink()
        logger.info(f"Deleted old metadata cache: {metadata_cache_path.name}")
    
    # Create new database
    DatabaseInstance.create_and_load_data(db_config, config, logger, workload_type)
    
    # Database creation completed, no longer need to mark success
    db_manager.mark_database_generated(db_config, workload_config, workload_type)
    
    # If no Insert/Delete operations, immediately create metadata cache
    if workload_type == "ycsb" and not db_manager.has_insert_or_delete_operations(workload_config):
        try:
            expected_records = db_config.get("ycsb_initial_record_count", 0)
            # Calling verify_database_integrity will automatically create metadata cache
            db_manager.verify_database_integrity(db_filename, expected_records, workload_config, workload_type)
            logger.info(f"Created metadata cache for new database")
        except Exception as e:
            logger.warning(f"Failed to create metadata cache: {e}")
    
    logger.info(f"✅ Database creation completed: {db_filename}")


def cleanup_database(db_path: str, logger: logging.Logger):
    """Clean up (delete) the specified database file."""
    try:
        if os.path.exists(db_path):
            os.remove(db_path)
            logger.info(f"Successfully cleaned up database file: {db_path}")
    except OSError as e:
        logger.error(f"Error cleaning up database file {db_path}: {e}")


if __name__ == "__main__":
    # Clean up invalid database records
    db_manager.clean_invalid_databases()
    
    # Print summary
    print_database_cache_summary() 