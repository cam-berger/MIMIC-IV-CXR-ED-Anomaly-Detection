"""
AWS Batch processor - Entry point for containerized execution
"""

import os
import sys
import argparse
from loguru import logger

from .utils import setup_logging
from .config_manager import get_config
from .phase1_stay_identification import StayIdentifier
from .phase2_clinical_extraction import ClinicalExtractor
from .phase3_integration import DataIntegrator


def main():
    """Main entry point for AWS Batch job"""
    
    parser = argparse.ArgumentParser(
        description='MIMIC Multi-Modal Preprocessing Pipeline'
    )
    parser.add_argument(
        '--phase',
        type=int,
        required=True,
        choices=[1, 2, 3],
        help='Processing phase to run'
    )
    parser.add_argument(
        '--chunk-id',
        type=int,
        help='Chunk ID for parallel processing (Phase 1 and 2)'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    
    logger.info("="*60)
    logger.info(f"MIMIC Preprocessing - Phase {args.phase}")
    if args.chunk_id is not None:
        logger.info(f"Chunk ID: {args.chunk_id}")
    logger.info("="*60)
    
    try:
        if args.phase == 1:
            processor = StayIdentifier()
            processor.run(chunk_id=args.chunk_id)
            
        elif args.phase == 2:
            processor = ClinicalExtractor()
            processor.run(chunk_id=args.chunk_id)
            
        elif args.phase == 3:
            processor = DataIntegrator()
            processor.run()
        
        logger.info("Processing completed successfully!")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        logger.exception(e)
        sys.exit(1)


if __name__ == '__main__':
    main()