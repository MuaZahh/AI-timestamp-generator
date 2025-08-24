import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to the level name for console output
        if hasattr(record, 'use_color') and record.use_color:
            level_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            record.levelname = f"{level_color}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)

class ContextualFilter(logging.Filter):
    """Filter to add contextual information to log records"""
    
    def __init__(self, component_name: str = "main"):
        super().__init__()
        self.component_name = component_name
    
    def filter(self, record):
        # Add component name to the record
        record.component = self.component_name
        
        # Add process information
        record.process_id = os.getpid()
        
        # Add timestamp in ISO format
        record.iso_timestamp = datetime.utcnow().isoformat() + 'Z'
        
        return True

class LoggingConfig:
    """Centralized logging configuration for the AI Timestamp Generator"""
    
    def __init__(self, 
                 log_dir: str = "logs",
                 log_level: str = "INFO",
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 component_name: str = "main"):
        
        self.log_dir = Path(log_dir)
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.component_name = component_name
        
        # Create logs directory
        self.log_dir.mkdir(exist_ok=True)
        
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        # Create root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear any existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        if self.enable_console:
            self._setup_console_handler(root_logger)
        
        # File handlers
        if self.enable_file:
            self._setup_file_handlers(root_logger)
        
        # Set up specific component loggers
        self._setup_component_loggers()
        
        # Log startup message
        logging.info(f"Logging configured - Level: {logging.getLevelName(self.log_level)}, Component: {self.component_name}")
    
    def _setup_console_handler(self, logger):
        """Setup console handler with colors"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        
        # Colored formatter for console
        console_format = "%(asctime)s [%(levelname)s] %(component)s: %(message)s"
        console_formatter = ColoredFormatter(console_format, datefmt='%H:%M:%S')
        console_handler.setFormatter(console_formatter)
        
        # Add contextual filter
        console_handler.addFilter(ContextualFilter(self.component_name))
        
        # Mark records for coloring
        class ColorFilter(logging.Filter):
            def filter(self, record):
                record.use_color = True
                return True
        
        console_handler.addFilter(ColorFilter())
        logger.addHandler(console_handler)
    
    def _setup_file_handlers(self, logger):
        """Setup file handlers for different log levels"""
        
        # Main application log (all levels)
        app_log_file = self.log_dir / f"{self.component_name}.log"
        app_handler = logging.handlers.RotatingFileHandler(
            app_log_file, 
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        app_handler.setLevel(self.log_level)
        
        # Error log (errors and critical only)
        error_log_file = self.log_dir / f"{self.component_name}_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        
        # File format (more detailed)
        file_format = "%(iso_timestamp)s [%(levelname)s] PID:%(process_id)s %(component)s:%(name)s:%(funcName)s:%(lineno)d - %(message)s"
        file_formatter = logging.Formatter(file_format)
        
        app_handler.setFormatter(file_formatter)
        error_handler.setFormatter(file_formatter)
        
        # Add contextual filter
        app_handler.addFilter(ContextualFilter(self.component_name))
        error_handler.addFilter(ContextualFilter(self.component_name))
        
        logger.addHandler(app_handler)
        logger.addHandler(error_handler)
    
    def _setup_component_loggers(self):
        """Setup specific loggers for different components"""
        
        # Component-specific configurations
        components = {
            'app': logging.INFO,
            'audio_processor': logging.INFO,
            'speech_processor': logging.INFO,
            'content_analyzer': logging.INFO,
            'timestamp_generator': logging.INFO,
            'export_processor': logging.INFO,
            'batch_processor': logging.INFO,
            'cache_manager': logging.INFO,
            'deepgram': logging.WARNING,  # Less verbose for external API
            'urllib3': logging.WARNING,   # Less verbose for HTTP requests
            'werkzeug': logging.WARNING,  # Less verbose for Flask
        }
        
        for component, level in components.items():
            logger = logging.getLogger(component)
            logger.setLevel(level)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger for a specific module/component"""
        return logging.getLogger(name)
    
    @classmethod
    def setup_for_component(cls, 
                          component_name: str,
                          log_level: str = "INFO",
                          log_dir: str = "logs",
                          enable_console: bool = True) -> 'LoggingConfig':
        """Quick setup for a specific component"""
        return cls(
            log_dir=log_dir,
            log_level=log_level,
            component_name=component_name,
            enable_console=enable_console
        )

# Error handling and exception logging
class ExceptionHandler:
    """Global exception handler with logging"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def log_exception(self, exc_info=True, extra_context: dict = None):
        """Log an exception with additional context"""
        context = extra_context or {}
        
        self.logger.error(
            f"Exception occurred: {context}",
            exc_info=exc_info,
            extra={'context': context}
        )
    
    def setup_global_exception_handler(self):
        """Setup global exception handler for uncaught exceptions"""
        def handle_exception(exc_type, exc_value, exc_traceback):
            if issubclass(exc_type, KeyboardInterrupt):
                # Don't log keyboard interrupts
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return
            
            self.logger.critical(
                "Uncaught exception",
                exc_info=(exc_type, exc_value, exc_traceback)
            )
        
        sys.excepthook = handle_exception

# Structured logging helpers
def log_processing_start(logger: logging.Logger, video_id: str, filename: str):
    """Log start of video processing"""
    logger.info(f"Starting processing for video {video_id} - {filename}")

def log_processing_step(logger: logging.Logger, video_id: str, step: str, duration: float = None):
    """Log completion of processing step"""
    duration_info = f" (took {duration:.2f}s)" if duration else ""
    logger.info(f"Video {video_id}: Completed {step}{duration_info}")

def log_processing_error(logger: logging.Logger, video_id: str, step: str, error: Exception):
    """Log processing error"""
    logger.error(f"Video {video_id}: Failed at {step} - {str(error)}", exc_info=True)

def log_processing_complete(logger: logging.Logger, video_id: str, total_duration: float):
    """Log successful completion of processing"""
    logger.info(f"Video {video_id}: Processing completed in {total_duration:.2f}s")

# Performance monitoring
class PerformanceLogger:
    """Logger for performance metrics"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_api_request(self, endpoint: str, method: str, status_code: int, duration: float):
        """Log API request metrics"""
        self.logger.info(
            f"API {method} {endpoint} - Status: {status_code}, Duration: {duration:.3f}s"
        )
    
    def log_database_query(self, query_type: str, duration: float, record_count: int = None):
        """Log database query performance"""
        count_info = f", Records: {record_count}" if record_count is not None else ""
        self.logger.info(f"DB {query_type} - Duration: {duration:.3f}s{count_info}")
    
    def log_processing_metrics(self, video_id: str, metrics: dict):
        """Log detailed processing metrics"""
        self.logger.info(f"Processing metrics for {video_id}: {metrics}")

# Initialize global logging
def initialize_logging(component_name: str = "ai_timestamp_generator", 
                      log_level: str = "INFO") -> LoggingConfig:
    """Initialize logging for the application"""
    config = LoggingConfig.setup_for_component(
        component_name=component_name,
        log_level=log_level
    )
    
    # Setup global exception handler
    exception_handler = ExceptionHandler()
    exception_handler.setup_global_exception_handler()
    
    return config

# Example usage and testing
if __name__ == "__main__":
    # Test logging configuration
    config = initialize_logging("test_component", "DEBUG")
    
    logger = config.get_logger(__name__)
    
    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test structured logging
    log_processing_start(logger, "test_video_123", "test_video.mp4")
    log_processing_step(logger, "test_video_123", "audio extraction", 2.5)
    log_processing_complete(logger, "test_video_123", 45.2)
    
    # Test exception logging
    exception_handler = ExceptionHandler(logger)
    try:
        raise ValueError("Test exception")
    except ValueError as e:
        exception_handler.log_exception(extra_context={"video_id": "test_123"})
    
    print("Logging test completed. Check logs/ directory for output files.")