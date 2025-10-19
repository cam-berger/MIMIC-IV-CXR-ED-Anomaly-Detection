"""
Run preprocessing pipeline locally (for testing) or submit to AWS Batch
Supports comprehensive clinical data extraction
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import boto3
import json
from loguru import logger
from typing import Optional, Dict, List

from src.utils import setup_logging
from src.config_manager import get_config
from src.phase1_stay_identification import StayIdentifier
from src.phase2_clinical_extraction import ClinicalExtractor
from src.phase3_integration import DataIntegrator


def run_local(
    phase: int, 
    chunk_id: Optional[int] = None,
    sample_size: Optional[int] = None,
    dry_run: bool = False
):
    """
    Run pipeline locally with enhanced options
    
    Args:
        phase: Phase number (1, 2, or 3)
        chunk_id: Optional chunk ID for phases 1 and 2
        sample_size: Optional sample size for testing (Phase 2)
        dry_run: If True, skip image copying (Phase 3)
    """
    logger.info(f"Running Phase {phase} locally")
    
    if phase == 1:
        processor = StayIdentifier()
        processor.run(chunk_id=chunk_id)
        
    elif phase == 2:
        processor = ClinicalExtractor()
        # Phase 2 now supports sample_size for testing
        processor.run(chunk_id=chunk_id, sample_size=sample_size)
        
    elif phase == 3:
        processor = DataIntegrator()
        if dry_run:
            logger.info("Dry run mode - images will not be copied")
            processor.copy_images = False
        processor.run()
    
    else:
        raise ValueError(f"Invalid phase: {phase}")


def submit_to_batch(
    phase: int, 
    chunk_id: Optional[int] = None,
    sample_size: Optional[int] = None,
    memory_override: Optional[int] = None,
    vcpus_override: Optional[int] = None
):
    """
    Submit job to AWS Batch with enhanced resource allocation
    
    Args:
        phase: Phase number (1, 2, or 3)
        chunk_id: Optional chunk ID for phases 1 and 2
        sample_size: Optional sample size for testing
        memory_override: Override default memory allocation (MB)
        vcpus_override: Override default vCPU count
    """
    config = get_config()
    batch = boto3.client('batch', region_name=config.get('aws.region'))
    
    # Create job name
    job_name = f"mimic-phase{phase}"
    if chunk_id is not None:
        job_name += f"-chunk{chunk_id}"
    if sample_size is not None:
        job_name += f"-sample{sample_size}"
    
    # Build command
    command = ['python', '-m', 'src.aws_processor', '--phase', str(phase)]
    
    if chunk_id is not None:
        command.extend(['--chunk-id', str(chunk_id)])
    
    if sample_size is not None:
        command.extend(['--sample-size', str(sample_size)])
    
    # Container overrides
    container_overrides = {
        'command': command,
        'environment': [
            {'name': 'PHASE', 'value': str(phase)},
            {'name': 'AWS_REGION', 'value': config.get('aws.region')}
        ]
    }
    
    # Resource allocation based on phase
    # Phase 2 needs more memory for comprehensive extraction
    if memory_override:
        container_overrides['memory'] = memory_override
    elif phase == 2:
        # Phase 2 needs more memory for lab/medication extraction
        container_overrides['memory'] = 32000  # 32GB
    elif phase == 3:
        # Phase 3 needs moderate memory for integration
        container_overrides['memory'] = 16000  # 16GB
    else:
        # Phase 1 is lighter
        container_overrides['memory'] = 8000  # 8GB
    
    if vcpus_override:
        container_overrides['vcpus'] = vcpus_override
    elif phase == 2:
        container_overrides['vcpus'] = 8  # More CPUs for parallel chunk processing
    else:
        container_overrides['vcpus'] = 4
    
    # Submit job
    response = batch.submit_job(
        jobName=job_name,
        jobQueue=config.get('aws.batch.job_queue'),
        jobDefinition=config.get('aws.batch.job_definition'),
        containerOverrides=container_overrides
    )
    
    job_id = response['jobId']
    logger.info(f"Submitted job: {job_name} (ID: {job_id})")
    logger.info(f"Allocated resources: {container_overrides.get('memory')}MB RAM, {container_overrides.get('vcpus')} vCPUs")
    
    return job_id


def submit_parallel_chunks(
    phase: int,
    num_chunks: int,
    sample_size: Optional[int] = None
) -> List[str]:
    """
    Submit multiple parallel jobs for chunked processing
    
    Args:
        phase: Phase number (1 or 2)
        num_chunks: Number of chunks to process in parallel
        sample_size: Optional sample size per chunk
        
    Returns:
        List of submitted job IDs
    """
    if phase not in [1, 2]:
        raise ValueError("Parallel chunks only supported for phases 1 and 2")
    
    job_ids = []
    logger.info(f"Submitting {num_chunks} parallel jobs for Phase {phase}")
    
    for chunk_id in range(num_chunks):
        job_id = submit_to_batch(
            phase=phase,
            chunk_id=chunk_id,
            sample_size=sample_size
        )
        job_ids.append(job_id)
    
    logger.info(f"Submitted {len(job_ids)} parallel jobs")
    return job_ids


def check_job_status(job_ids: List[str]) -> Dict[str, str]:
    """
    Check status of submitted batch jobs
    
    Args:
        job_ids: List of job IDs to check
        
    Returns:
        Dictionary mapping job ID to status
    """
    config = get_config()
    batch = boto3.client('batch', region_name=config.get('aws.region'))
    
    response = batch.describe_jobs(jobs=job_ids)
    
    status_map = {}
    for job in response['jobs']:
        status_map[job['jobId']] = job['status']
    
    return status_map


def validate_environment():
    """Validate that required environment and configuration is present"""
    config = get_config()
    
    required_settings = [
        'aws.region',
        'aws.s3.mimic_bucket',
        'aws.s3.output_bucket',
        'aws.s3.temp_bucket'
    ]
    
    missing = []
    for setting in required_settings:
        if not config.get(setting):
            missing.append(setting)
    
    if missing:
        logger.error(f"Missing required configuration: {missing}")
        return False
    
    logger.info("Environment validated successfully")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Run MIMIC preprocessing pipeline with comprehensive clinical extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Phase 1 locally
  python run_local.py --mode local --phase 1
  
  # Run Phase 2 with sample for testing
  python run_local.py --mode local --phase 2 --sample-size 10
  
  # Submit Phase 2 to AWS Batch with custom memory
  python run_local.py --mode batch --phase 2 --memory 64000
  
  # Run parallel chunks in batch
  python run_local.py --mode batch --phase 2 --parallel-chunks 10
  
  # Run all phases sequentially
  python run_local.py --mode local --all-phases
        """
    )
    
    # Execution mode
    parser.add_argument(
        '--mode',
        choices=['local', 'batch'],
        default='local',
        help='Execution mode (local or AWS Batch)'
    )
    
    # Phase selection
    parser.add_argument(
        '--phase',
        type=int,
        choices=[1, 2, 3],
        help='Phase to run (1: Stay ID, 2: Clinical Extraction, 3: Integration)'
    )
    
    parser.add_argument(
        '--all-phases',
        action='store_true',
        help='Run all phases sequentially'
    )
    
    # Chunking options
    parser.add_argument(
        '--chunk-id',
        type=int,
        help='Specific chunk ID for parallel processing'
    )
    
    parser.add_argument(
        '--parallel-chunks',
        type=int,
        help='Number of parallel chunks to process (batch mode only)'
    )
    
    # Testing options
    parser.add_argument(
        '--sample-size',
        type=int,
        help='Process only N patients for testing (Phase 2)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run mode - skip image copying (Phase 3)'
    )
    
    # Resource overrides for batch
    parser.add_argument(
        '--memory',
        type=int,
        help='Memory allocation in MB (batch mode only)'
    )
    
    parser.add_argument(
        '--vcpus',
        type=int,
        help='Number of vCPUs (batch mode only)'
    )
    
    # Other options
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate environment and exit'
    )
    
    parser.add_argument(
        '--check-jobs',
        nargs='+',
        help='Check status of batch job IDs'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed")
        return 1
    
    if args.validate_only:
        logger.info("Validation complete")
        return 0
    
    # Check job status
    if args.check_jobs:
        status_map = check_job_status(args.check_jobs)
        for job_id, status in status_map.items():
            logger.info(f"Job {job_id}: {status}")
        return 0
    
    # Validate phase selection
    if not args.all_phases and not args.phase:
        parser.error("Either --phase or --all-phases must be specified")
    
    # Handle parallel chunks
    if args.parallel_chunks:
        if args.mode != 'batch':
            parser.error("Parallel chunks only supported in batch mode")
        if not args.phase:
            parser.error("--phase must be specified with --parallel-chunks")
        
        job_ids = submit_parallel_chunks(
            phase=args.phase,
            num_chunks=args.parallel_chunks,
            sample_size=args.sample_size
        )
        
        logger.info(f"Submitted jobs: {job_ids}")
        logger.info("Use --check-jobs to monitor status")
        return 0
    
    # Run phases
    try:
        if args.all_phases:
            logger.info("Running all phases sequentially...")
            for phase in [1, 2, 3]:
                logger.info(f"\n{'='*60}")
                logger.info(f"Starting Phase {phase}")
                logger.info(f"{'='*60}")
                
                if args.mode == 'local':
                    run_local(
                        phase=phase,
                        sample_size=args.sample_size if phase == 2 else None,
                        dry_run=args.dry_run if phase == 3 else False
                    )
                else:
                    job_id = submit_to_batch(
                        phase=phase,
                        sample_size=args.sample_size if phase == 2 else None,
                        memory_override=args.memory,
                        vcpus_override=args.vcpus
                    )
                    logger.info(f"Waiting for Phase {phase} job {job_id} to complete...")
                    # In production, implement proper job monitoring here
        else:
            if args.mode == 'local':
                run_local(
                    phase=args.phase,
                    chunk_id=args.chunk_id,
                    sample_size=args.sample_size,
                    dry_run=args.dry_run
                )
            else:
                job_id = submit_to_batch(
                    phase=args.phase,
                    chunk_id=args.chunk_id,
                    sample_size=args.sample_size,
                    memory_override=args.memory,
                    vcpus_override=args.vcpus
                )
                logger.info(f"Job submitted: {job_id}")
        
        logger.info("\nPipeline execution completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        return 1


if __name__ == '__main__':
    exit(main())