import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Return a pre-configured logger for consistent formatting across modules.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Prevent propagation to avoid double logging
        logger.propagate = False
        
    return logger
