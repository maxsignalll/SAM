"""
Experiment log file handler

Provides experiment runtime log file saving functionality, including:
- Automatic log file creation
- Simultaneous output to console and file
- Log file naming and organization
- Log rotation and compression (optional)
"""

import logging
import os
from datetime import datetime
from typing import Optional
import gzip
import shutil


class ExperimentFileHandler(logging.FileHandler):
    """Experiment-specific file handler with automatic directory creation and file naming"""
    
    def __init__(self, experiment_name: str, strategy_name: str, 
                 base_dir: str = "results/logs", compress_old: bool = False):
        """
        Initialize experiment file handler
        
        Args:
            experiment_name: Experiment name
            strategy_name: Strategy name
            base_dir: Log base directory
            compress_old: Whether to compress old logs
        """
        self.experiment_name = experiment_name
        self.strategy_name = strategy_name
        self.base_dir = base_dir
        self.compress_old = compress_old
        
        # Create log file path
        self.log_file_path = self._create_log_file_path()
        
        # Call parent class initialization
        super().__init__(self.log_file_path, mode='w', encoding='utf-8')
        
        # Set formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.setFormatter(formatter)
        
    def _create_log_file_path(self) -> str:
        """Create log file path"""
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory structure: logs/YYYYMMDD/strategy_name/
        date_dir = datetime.now().strftime("%Y%m%d")
        log_dir = os.path.join(self.base_dir, date_dir, self.strategy_name)
        
        # Ensure directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Create filename
        filename = f"{self.experiment_name}_{timestamp}.log"
        file_path = os.path.join(log_dir, filename)
        
        # If need to compress old logs
        if self.compress_old:
            self._compress_old_logs(log_dir)
        
        return file_path
    
    def _compress_old_logs(self, log_dir: str):
        """Compress log files older than 1 day"""
        try:
            for filename in os.listdir(log_dir):
                if filename.endswith('.log'):
                    file_path = os.path.join(log_dir, filename)
                    # Check file modification time
                    if os.path.getmtime(file_path) < (datetime.now().timestamp() - 86400):
                        # Compress file
                        gz_path = file_path + '.gz'
                        with open(file_path, 'rb') as f_in:
                            with gzip.open(gz_path, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        # Delete original file
                        os.remove(file_path)
        except Exception as e:
            # Compression failure should not affect main program
            pass
    
    def get_log_file_path(self) -> str:
        """Get log file path"""
        return self.log_file_path


class DualLoggingHandler:
    """Dual logging handler: output to both console and file"""
    
    def __init__(self, logger: logging.Logger, experiment_name: str, 
                 strategy_name: str, console_level: int = logging.INFO,
                 file_level: int = logging.DEBUG):
        """
        Initialize dual logging handler
        
        Args:
            logger: Logger object to configure
            experiment_name: Experiment name
            strategy_name: Strategy name
            console_level: Console log level
            file_level: File log level
        """
        self.logger = logger
        self.experiment_name = experiment_name
        self.strategy_name = strategy_name
        
        # Clear existing handlers
        self._remove_existing_handlers()
        
        # Add console handler
        self._add_console_handler(console_level)
        
        # Add file handler
        self.file_handler = self._add_file_handler(file_level)
        
    def _remove_existing_handlers(self):
        """Remove existing handlers"""
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
            
    def _add_console_handler(self, level: int):
        """Add console handler"""
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # Use concise console format
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(console_handler)
        
    def _add_file_handler(self, level: int) -> ExperimentFileHandler:
        """Add file handler"""
        file_handler = ExperimentFileHandler(
            self.experiment_name, 
            self.strategy_name,
            compress_old=True
        )
        file_handler.setLevel(level)
        
        self.logger.addHandler(file_handler)
        return file_handler
    
    def get_log_file_path(self) -> str:
        """Get log file path"""
        return self.file_handler.get_log_file_path()
    
    def close(self):
        """Close log handler"""
        if hasattr(self, 'file_handler'):
            self.file_handler.close()


def setup_experiment_logging(experiment_name: str, strategy_name: str, 
                           logger: logging.Logger, 
                           console_level: int = logging.INFO,
                           file_level: int = logging.DEBUG) -> DualLoggingHandler:
    """
    Convenience function: set up experiment logging
    
    Args:
        experiment_name: Experiment name
        strategy_name: Strategy name
        logger: Logger to configure
        console_level: Console log level
        file_level: File log level
        
    Returns:
        DualLoggingHandler instance
    """
    return DualLoggingHandler(
        logger=logger,
        experiment_name=experiment_name,
        strategy_name=strategy_name,
        console_level=console_level,
        file_level=file_level
    )


def create_log_summary(log_file_path: str, output_path: Optional[str] = None) -> str:
    """
    Create log summary
    
    Args:
        log_file_path: Log file path
        output_path: Summary output path (optional)
        
    Returns:
        Summary content
    """
    summary_lines = []
    
    # Statistics
    total_lines = 0
    error_count = 0
    warning_count = 0
    cycle_count = 0
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            
            if ' - ERROR - ' in line:
                error_count += 1
            elif ' - WARNING - ' in line:
                warning_count += 1
            elif '🔄 S0 strategy allocation cycle' in line:
                cycle_count += 1
    
    # Create summary
    summary_lines.append(f"Log file summary: {os.path.basename(log_file_path)}")
    summary_lines.append(f"Creation time: {datetime.fromtimestamp(os.path.getctime(log_file_path))}")
    summary_lines.append(f"File size: {os.path.getsize(log_file_path) / 1024:.2f} KB")
    summary_lines.append(f"Total lines: {total_lines}")
    summary_lines.append(f"Errors: {error_count}")
    summary_lines.append(f"Warnings: {warning_count}")
    summary_lines.append(f"Allocation cycles: {cycle_count}")
    
    summary = '\n'.join(summary_lines)
    
    # If output path is specified, save summary
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary)
    
    return summary