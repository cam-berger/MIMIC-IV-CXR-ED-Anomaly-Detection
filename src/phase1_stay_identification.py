"""
Phase 1: Stay ID Identification
Links chest X-rays to ED stays using temporal matching
"""

import pandas as pd
from datetime import timedelta
from typing import Optional
from loguru import logger
from tqdm import tqdm

from .config_manager import get_config
from .utils import S3Handler, DataValidator


class StayIdentifier:
    """Identify stay_id for chest X-rays"""
    
    def __init__(self):
        """Initialize Stay Identifier"""
        self.config = get_config()
        self.s3 = S3Handler(
            region=self.config.get('aws.region'),
            profile=self.config.get('aws.profile')
        )
        self.extended_window_days = self.config.get(
            'preprocessing.phase1.extended_window_days', 7
        )
        
        logger.info("Stay Identifier initialized")
    
    def identify_stay_for_cxr(
        self,
        subject_id: int,
        study_datetime: pd.Timestamp,
        ed_stays: pd.DataFrame
    ) -> Optional[int]:
        """
        Identify stay_id for a single CXR using temporal matching
        
        Args:
            subject_id: Patient ID
            study_datetime: When the CXR was taken
            ed_stays: DataFrame of ED stays
            
        Returns:
            stay_id if found, None otherwise
        """
        # Get all stays for this patient
        patient_stays = ed_stays[ed_stays['subject_id'] == subject_id].copy()
        
        if len(patient_stays) == 0:
            return None
        
        # Ensure datetime columns
        patient_stays['intime'] = pd.to_datetime(patient_stays['intime'])
        patient_stays['outtime'] = pd.to_datetime(patient_stays['outtime'])
        
        # Method 1: CXR taken during ED stay
        mask = (
            (patient_stays['intime'] <= study_datetime) &
            (patient_stays['outtime'] >= study_datetime)
        )
        matches = patient_stays[mask]
        
        if len(matches) > 0:
            return int(matches.iloc[0]['stay_id'])
        
        # Method 2: Extended window (within N days after discharge)
        extended_window = timedelta(days=self.extended_window_days)
        patient_stays['extended_outtime'] = patient_stays['outtime'] + extended_window
        
        mask = (
            (patient_stays['intime'] <= study_datetime) &
            (patient_stays['extended_outtime'] >= study_datetime)
        )
        matches = patient_stays[mask]
        
        if len(matches) > 0:
            logger.debug(
                f"Found stay in extended window for subject {subject_id}"
            )
            return int(matches.iloc[0]['stay_id'])
        
        return None
    
    def process_chunk(
        self,
        cxr_chunk: pd.DataFrame,
        ed_stays: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Process a chunk of CXR metadata
        
        Args:
            cxr_chunk: DataFrame with CXR metadata
            ed_stays: DataFrame with ED stays
            
        Returns:
            DataFrame with stay_id added
        """
        results = []
        
        logger.info(f"Processing {len(cxr_chunk)} CXR records")
        
        for idx, row in tqdm(cxr_chunk.iterrows(), total=len(cxr_chunk), desc="Identifying stays"):
            try:
                # Validate data
                if not DataValidator.validate_subject_id(row['subject_id']):
                    logger.warning(f"Invalid subject_id at row {idx}")
                    continue
                
                # Parse study datetime
                if pd.isna(row['StudyDate']) or pd.isna(row['StudyTime']):
                    logger.debug(f"Missing StudyDate/StudyTime at row {idx}")
                    continue
                
                study_datetime = pd.to_datetime(
                    f"{row['StudyDate']} {row['StudyTime']}"
                )
                
                # Identify stay
                stay_id = self.identify_stay_for_cxr(
                    row['subject_id'],
                    study_datetime,
                    ed_stays
                )
                
                results.append({
                    'dicom_id': row['dicom_id'],
                    'subject_id': row['subject_id'],
                    'study_id': row['study_id'],
                    'study_datetime': study_datetime,
                    'stay_id': stay_id,
                    'ViewPosition': row.get('ViewPosition', None)
                })
                
            except Exception as e:
                logger.error(f"Error processing row {idx}: {str(e)}")
                continue
        
        return pd.DataFrame(results)
    
    def load_cxr_metadata(self, chunk_id: Optional[int] = None) -> pd.DataFrame:
        """
        Load CXR metadata from S3
        
        Args:
            chunk_id: If provided, load specific chunk
            
        Returns:
            DataFrame with CXR metadata
        """
        if chunk_id is not None:
            # Load specific chunk
            temp_bucket = self.config.get('aws.s3.temp_bucket')
            key = f'processing/chunks/chunk_{chunk_id}.csv'
            logger.info(f"Loading chunk {chunk_id} from s3://{temp_bucket}/{key}")
            return self.s3.read_csv(temp_bucket, key)
        else:
            # Load full metadata
            # Check if custom bucket is specified for CXR data
            custom_bucket = self.config.get('data_paths.mimic_cxr.bucket')
            mimic_bucket = custom_bucket if custom_bucket else self.config.get('aws.s3.mimic_bucket')

            metadata_path = self.config.get_data_path('mimic_cxr', 'metadata')
            logger.info(f"Loading CXR metadata from s3://{mimic_bucket}/{metadata_path}")

            # CXR-PRO uses CSV without gzip compression
            if 'cxr-pro' in metadata_path:
                return self.s3.read_csv(mimic_bucket, metadata_path)
            else:
                return self.s3.read_csv(mimic_bucket, metadata_path, compression='gzip')
    
    def load_ed_stays(self) -> pd.DataFrame:
        """
        Load ED stays data from S3

        Returns:
            DataFrame with ED stays
        """
        # Check if custom bucket is specified for ED data
        custom_bucket = self.config.get('data_paths.mimic_ed.bucket')
        mimic_bucket = custom_bucket if custom_bucket else self.config.get('aws.s3.mimic_bucket')

        edstays_path = self.config.get_data_path('mimic_ed', 'edstays')
        logger.info(f"Loading ED stays from s3://{mimic_bucket}/{edstays_path}")
        return self.s3.read_csv(mimic_bucket, edstays_path, compression='gzip')
    
    def run(self, chunk_id: Optional[int] = None):
        """
        Execute Phase 1: Stay ID identification
        
        Args:
            chunk_id: If provided, process only this chunk
        """
        logger.info("="*60)
        logger.info("Starting Phase 1: Stay ID Identification")
        logger.info("="*60)
        
        # Load data
        cxr_metadata = self.load_cxr_metadata(chunk_id)
        ed_stays = self.load_ed_stays()
        
        # Process
        results = self.process_chunk(cxr_metadata, ed_stays)
        
        # Save results
        output_bucket = self.config.get('aws.s3.output_bucket')
        
        if chunk_id is not None:
            output_key = f'processing/phase1_results/chunk_{chunk_id}.csv'
        else:
            output_key = 'processing/cxr_with_stays.csv'
        
        logger.info(f"Saving results to s3://{output_bucket}/{output_key}")
        self.s3.write_csv(results, output_bucket, output_key)
        
        # Log statistics
        self._log_statistics(results)
        
        logger.info("Phase 1 complete!")
        return results
    
    def _log_statistics(self, results: pd.DataFrame):
        """Log processing statistics"""
        total = len(results)
        with_stays = results['stay_id'].notna().sum()
        without_stays = total - with_stays
        percentage = (with_stays / total * 100) if total > 0 else 0
        
        logger.info("="*60)
        logger.info("Phase 1 Statistics:")
        logger.info(f"  Total CXRs processed:     {total:,}")
        logger.info(f"  CXRs with stay_id:        {with_stays:,} ({percentage:.2f}%)")
        logger.info(f"  CXRs without stay_id:     {without_stays:,}")
        logger.info("="*60)


def main():
    """Main entry point for Phase 1"""
    import argparse
    from .utils import setup_logging
    
    parser = argparse.ArgumentParser(
        description='Phase 1: Identify stay_id for chest X-rays'
    )
    parser.add_argument(
        '--chunk-id',
        type=int,
        help='Process specific chunk (for parallel processing)'
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
    
    # Run Phase 1
    identifier = StayIdentifier()
    identifier.run(chunk_id=args.chunk_id)


if __name__ == '__main__':
    main()