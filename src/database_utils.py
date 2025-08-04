import apsw
import time
import os
from pathlib import Path
import gc
import psutil


def is_file_locked(file_path):
    """Check if SQLite database file is occupied by other processes (including WAL and SHM files)"""
    file_path = Path(file_path)
    
    # Check main database file
    if not file_path.exists():
        return False

    wal_file = file_path.with_suffix(file_path.suffix + '-wal')
    shm_file = file_path.with_suffix(file_path.suffix + '-shm')
    
    files_to_check = [file_path]
    if wal_file.exists():
        files_to_check.append(wal_file)
    if shm_file.exists():
        files_to_check.append(shm_file)
    
    # Check each file for occupation
    for check_file in files_to_check:
        try:
            # Method 1: Try to rename file
            temp_name = check_file.with_suffix(check_file.suffix + '.tmp_check')
            check_file.rename(temp_name)
            temp_name.rename(check_file)
        except (OSError, PermissionError):
            return True

        # Method 2: Try to open file in exclusive mode (cross-platform implementation)
        try:
            if os.name == 'nt':  # Windows
                import msvcrt
                with open(check_file, 'r+b') as f:
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:  # Linux/Unix (including Ubuntu)
                import fcntl
                with open(check_file, 'r+b') as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except (OSError, PermissionError, ImportError, BlockingIOError):
            return True  # File is locked or operation failed
    
    return False

def safe_connect_database_with_retry(db_filename: str, max_retries: int = 5, retry_delay: float = 1.0) -> apsw.Connection:
    """
    Safely connect to database and retry when encountering locks.
    """
    for attempt in range(max_retries):
        try:
            conn = apsw.Connection(db_filename)
            return conn
        except apsw.BusyError:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise
    raise RuntimeError(f"Unable to connect to database {db_filename} after {max_retries} attempts.")

def cleanup_sqlite_auxiliary_files(db_filename):
    """Clean SQLite auxiliary files"""
    db_path = Path(db_filename)
    auxiliary_files = [
        db_path.with_suffix(db_path.suffix + '-wal'),
        db_path.with_suffix(db_path.suffix + '-shm'),
        db_path.with_suffix(db_path.suffix + '-journal')
    ]
    
    for aux_file in auxiliary_files:
        if aux_file.exists():
            # Remove retry and error catching
            aux_file.unlink()
            print(f"  Cleaning auxiliary file: {aux_file.name}")

def force_terminate_database_processes(db_filename):
    """Force terminate all processes using the specified database file"""
    current_pid = os.getpid()
    terminated_processes = []
    
    print(f"[Force Terminate] Finding processes using database {db_filename}...")
    
    for proc in psutil.process_iter(['pid', 'name', 'open_files']):
        try:
            if proc.info['pid'] == current_pid:
                continue
            
            open_files = proc.info['open_files']
            if open_files:
                for file_info in open_files:
                    if db_filename in file_info.path or any(
                        aux in file_info.path 
                        for aux in [db_filename + '-wal', db_filename + '-shm', db_filename + '-journal']
                    ):
                        print(f"[Force Terminate] Found process occupying: PID={proc.info['pid']}, Name={proc.info['name']}")
                        process = psutil.Process(proc.info['pid'])
                        process.terminate()
                        try:
                            process.wait(timeout=3)
                        except psutil.TimeoutExpired:
                            process.kill()
                            print(f"[Force Terminate] Force killing process PID={proc.info['pid']}")
                        else:
                            print(f"[Force Terminate] Gracefully terminating process PID={proc.info['pid']}")
                        
                        terminated_processes.append(proc.info['pid'])
                        break
                        
        except (psutil.NoSuchProcess, psutil.AccessDenied, AttributeError):
            continue
    
    if terminated_processes:
        print(f"[Force Terminate] Terminated {len(terminated_processes)} occupying processes")
        time.sleep(2)
    else:
        print(f"[Force Terminate] No other processes found occupying the database")
        
    return len(terminated_processes)

def comprehensive_database_release(db_filename):
    """
    Comprehensively release database file, including force terminating occupying processes and cleaning auxiliary files.
    """
    # Force garbage collection to help release file handles held by Python
    gc.collect()
    time.sleep(0.1) # Brief wait

    print(f"[Full Release] Starting to release database file: {os.path.basename(db_filename)}")
    
    db_path = str(db_filename)
    
    # Step 1: Force garbage collection
    for i in range(2):
        gc.collect()
        time.sleep(0.05)
    
    # Step 2: Check and force terminate occupying processes
    force_terminate_database_processes(db_filename)
    
    # Step 3: Clean SQLite auxiliary files
    cleanup_sqlite_auxiliary_files(db_filename)
    
    # Step 4: Final verification
    if not is_file_locked(db_filename):
        print(f"[Full Release] ✅ Database file has been completely released")
        return True
    else:
        # If still locked, throw error instead of failing silently
        raise RuntimeError(f"[Full Release] Database file {db_filename} is still locked after release process.")
