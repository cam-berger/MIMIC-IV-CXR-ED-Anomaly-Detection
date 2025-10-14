"""
Run preprocessing pipeline locally (for testing) or submit to AWS Batch
"""

import argparse
import boto3
from loguru import logger

from src.utils import setup_logging
from src.config_manager import get_config
from src.phase1_stay_identification import StayIdentifier
from src.phase2_clinical_extraction import ClinicalExtractor
from src.phase3_integration import DataIntegrator


def run_local(phase: int, chunk_id: int = None):
    """
    Run pipeline locally
    
    Args:
        phase: Phase number (1, 2, or 3)
        chunk_id: Optional chunk ID for phases 1 and 2
    """
    logger.info(f"Running Phase {phase} locally")
    
    if phase == 1:
        processor = StayIdentifier()
        processor.run(chunk_id=chunk_id)
    elif phase == 2:
        processor = ClinicalExtractor()
        processor.run(chunk_id=chunk_id)
    elif phase == 3:
        processor = DataIntegrator()
        processor.run()


def submit_to_batch(phase: int, chunk_id: int = None):
    """
    Submit job to AWS Batch
    
    Args:
        phase: Phase number (1, 2, or 3)
        chunk_id: Optional chunk ID for phases 1 and 2
    """
    config = get_config()
    batch = boto3.client('batch', region_name=config.get('aws.region'))
    
    job_name = f"mimic-phase{phase}"
    if chunk_id is not None:
        job_name += f"-chunk{chunk_id}"
    
    container_overrides = {
        'command': ['python', '-m', 'src.aws_processor', '--phase', str(phase)]
    }
    
    if chunk_id is not None:
        container_overrides['command'].extend(['--chunk-id', str(chunk_id)])
    
    response = batch.submit_job(
        jobName=job_name,
        jobQueue=config.get('aws.batch.job_queue'),
        jobDefinition=config.get('aws.batch.job_definition'),
        containerOverrides=container_overrides
    )
    
    job_id = response['jobId']
    logger.info(f"Submitted job: {job_name} (ID: {job_id})")
    
    return job_id


def main():
    parser = argparse.ArgumentParser(
        description='Run MIMIC preprocessing pipeline'
    )
    parser.add_argument(
        '--mode',
        choices=['local', 'batch'],
        default='local',
        help='Execution mode'
    )
    parser.add_argument(
        '--phase',
        type=int,
        required=True,
        choices=[1, 2, 3],
        help='Phase to run'
    )
    parser.add_argument(
        '--chunk-id',
        type=int,
        help='Chunk ID (for parallel processing)'
    )
    parser.add_argument(
        '--all-phases',
        action='store_true',
        help='Run all phases sequentially'
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    if args.all_phases:
        logger.info("Running all phases...")
        for phase in [1, 2, 3]:
            if args.mode == 'local':
                run_local(phase)
            else:
                submit_to_batch(phase)
    else:
        if args.mode == 'local':
            run_local(args.phase, args.chunk_id)
        else:
            submit_to_batch(args.phase, args.chunk_id)


if __name__ == '__main__':
    main()