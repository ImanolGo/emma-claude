import logging
import logging.config
from pathlib import Path
from typing import Optional, Dict
from rich.logging import RichHandler
import sys
import yaml

def setup_logger(
    config: Optional[Dict] = None,
    default_level: str = "INFO",
    log_file: Optional[Path] = None
) -> None:
    """
    Setup logging configuration
    
    Args:
        config: Dictionary containing logging configuration
        default_level: Default logging level if no config provided
        log_file: Optional path to log file
    """
    if config:
        # Use provided configuration
        logging.config.dictConfig(config)
    else:
        # Setup basic configuration with rich handler
        handlers = [
            RichHandler(
                rich_tracebacks=True,
                markup=True,
                show_time=True,
                show_path=True
            )
        ]
        
        # Add file handler if log file specified
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
                )
            )
            handlers.append(file_handler)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, default_level),
            format="%(message)s",
            datefmt="[%X]",
            handlers=handlers
        )
        
    # Suppress excessive logging from external libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """
    Get logger instance with given name
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

class LoggerContext:
    """Context manager for temporarily changing log level"""
    
    def __init__(self, logger: logging.Logger, level: str):
        self.logger = logger
        self.level = getattr(logging, level)
        self.previous_level = logger.level
        
    def __enter__(self):
        self.logger.setLevel(self.level)
        return self.logger
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.previous_level)