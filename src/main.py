#!/usr/bin/env python3
"""
Time Series Unification (TS-Unify) Main Entry Point

This module serves as the main entry point for the TS-Unify framework,
providing a clean interface to the training pipeline.
"""

import sys
import os
import logging
from pathlib import Path
from typing import Optional

# Add src directory to Python path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ts_unify.log')
    ]
)

logger = logging.getLogger(__name__)


def setup_environment() -> None:
    """Setup the environment for running TS-Unify."""
    # Check Python version
    if sys.version_info < (3, 8):
        raise RuntimeError("Python 3.8 or higher is required")
    
    # Check if required directories exist
    required_dirs = ['src', 'configs']
    for dir_name in required_dirs:
        if not (src_path / dir_name).exists():
            raise RuntimeError(f"Required directory '{dir_name}' not found")
    
    logger.info("Environment setup completed")


def main() -> int:
    """
    Main entry point for TS-Unify.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        setup_environment()
        
        # Import after environment setup
        from core.pipeline import main as pipeline_main
        
        logger.info("Starting TS-Unify pipeline...")
        pipeline_main()
        
        logger.info("TS-Unify pipeline completed successfully")
        return 0
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Please ensure all dependencies are installed")
        return 1
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)