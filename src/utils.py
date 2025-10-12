"""
Utility functions for MIMIC preprocessing
"""

import os
import sys
import boto3
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
from io import BytesIO, StringIO
from loguru import logger
from datetime import datetime


def setup_logging(log_dir: str = "logs", log_level: str = "INFO"):
    """
    Setup logging configuration
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Remove default logger
    logger.remove()
    
    # Add console logger
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=log_level
    )
    
    # Add file logger
    log_file = os.path.join(
        log_dir,
        f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    
    logger.add(
        log_file,
        rotation="500 MB",
        retention="10 days",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}"
    )
    
    logger.info(f"Logging initialized. Log file: {log_file}")


class S3Handler:
    """Handle S3 operations efficiently"""
    
    def __init__(self, region: str = 'us-east-1', profile: Optional[str] = None):
        """
        Initialize S3 handler
        
        Args:
            region: AWS region
            profile: AWS profile name (optional)
        """
        session_kwargs = {'region_name': region}
        if profile:
            session_kwargs['profile_name'] = profile
        
        session = boto3.Session(**session_kwargs)
        self.s3_client = session.client('s3')
        self.s3_resource = session.resource('s3')
        
        logger.info(f"S3 Handler initialized (region: {region})")
    
    def read_csv(
        self,
        bucket: str,
        key: str,
        **pandas_kwargs
    ) -> pd.DataFrame:
        """
        Read CSV file from S3
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            **pandas_kwargs: Additional arguments for pd.read_csv
            
        Returns:
            DataFrame
        """
        try:
            logger.debug(f"Reading s3://{bucket}/{key}")
            
            obj = self.s3_client.get_object(Bucket=bucket, Key=key)
            df = pd.read_csv(obj['Body'], **pandas_kwargs)
            
            logger.info(f"Successfully read {len(df)} rows from s3://{bucket}/{key}")
            return df
            
        except Exception as e:
            logger.error(f"Error reading s3://{bucket}/{key}: {str(e)}")
            raise
    
    def write_csv(
        self,
        df: pd.DataFrame,
        bucket: str,
        key: str,
        **pandas_kwargs
    ):
        """
        Write DataFrame to S3 as CSV
        
        Args:
            df: DataFrame to write
            bucket: S3 bucket name
            key: S3 object key
            **pandas_kwargs: Additional arguments for df.to_csv
        """
        try:
            logger.debug(f"Writing {len(df)} rows to s3://{bucket}/{key}")
            
            # Write to string buffer
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False, **pandas_kwargs)
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=csv_buffer.getvalue().encode('utf-8')
            )
            
            logger.info(f"Successfully wrote to s3://{bucket}/{key}")
            
        except Exception as e:
            logger.error(f"Error writing to s3://{bucket}/{key}: {str(e)}")
            raise
    
    def copy_object(
        self,
        source_bucket: str,
        source_key: str,
        dest_bucket: str,
        dest_key: str
    ):
        """
        Copy object within S3 (no download/upload needed)
        
        Args:
            source_bucket: Source bucket name
            source_key: Source object key
            dest_bucket: Destination bucket name
            dest_key: Destination object key
        """
        try:
            copy_source = {'Bucket': source_bucket, 'Key': source_key}
            
            self.s3_client.copy_object(
                CopySource=copy_source,
                Bucket=dest_bucket,
                Key=dest_key
            )
            
            logger.debug(f"Copied s3://{source_bucket}/{source_key} to s3://{dest_bucket}/{dest_key}")
            
        except Exception as e:
            logger.error(f"Error copying object: {str(e)}")
            raise
    
    def object_exists(self, bucket: str, key: str) -> bool:
        """
        Check if object exists in S3
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            
        Returns:
            True if object exists, False otherwise
        """
        try:
            self.s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except:
            return False
    
    def list_objects(
        self,
        bucket: str,
        prefix: str = '',
        max_keys: Optional[int] = None
    ) -> List[str]:
        """
        List objects in S3 bucket with given prefix
        
        Args:
            bucket: S3 bucket name
            prefix: Prefix to filter objects
            max_keys: Maximum number of keys to return
            
        Returns:
            List of object keys
        """
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            
            page_kwargs = {
                'Bucket': bucket,
                'Prefix': prefix
            }
            
            if max_keys:
                page_kwargs['PaginationConfig'] = {'MaxItems': max_keys}
            
            pages = paginator.paginate(**page_kwargs)
            
            keys = []
            for page in pages:
                if 'Contents' in page:
                    keys.extend([obj['Key'] for obj in page['Contents']])
            
            logger.info(f"Found {len(keys)} objects in s3://{bucket}/{prefix}")
            return keys
            
        except Exception as e:
            logger.error(f"Error listing objects: {str(e)}")
            raise
    
    def download_file(self, bucket: str, key: str, local_path: str):
        """
        Download file from S3 to local path
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            local_path: Local file path
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            self.s3_client.download_file(bucket, key, local_path)
            logger.info(f"Downloaded s3://{bucket}/{key} to {local_path}")
            
        except Exception as e:
            logger.error(f"Error downloading file: {str(e)}")
            raise
    
    def upload_file(self, local_path: str, bucket: str, key: str):
        """
        Upload file from local path to S3
        
        Args:
            local_path: Local file path
            bucket: S3 bucket name
            key: S3 object key
        """
        try:
            self.s3_client.upload_file(local_path, bucket, key)
            logger.info(f"Uploaded {local_path} to s3://{bucket}/{key}")
            
        except Exception as e:
            logger.error(f"Error uploading file: {str(e)}")
            raise


class DataValidator:
    """Validate MIMIC data"""
    
    @staticmethod
    def validate_subject_id(subject_id: Union[int, str]) -> bool:
        """Validate subject_id format"""
        try:
            int(subject_id)
            return True
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_stay_id(stay_id: Union[int, str]) -> bool:
        """Validate stay_id format"""
        try:
            int(stay_id)
            return True
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_datetime(dt: Union[str, pd.Timestamp]) -> bool:
        """Validate datetime"""
        try:
            pd.to_datetime(dt)
            return True
        except:
            return False
    
    @staticmethod
    def validate_clinical_features(features: Dict) -> bool:
        """Validate clinical features dictionary"""
        required_fields = [
            'temperature', 'heartrate', 'resprate',
            'o2sat', 'sbp', 'dbp'
        ]
        return all(field in features for field in required_fields)
    
    @staticmethod
    def validate_dataframe_columns(df: pd.DataFrame, required_columns: List[str]) -> bool:
        """Validate DataFrame has required columns"""
        return all(col in df.columns for col in required_columns)


def create_s3_path(bucket: str, *parts: str) -> str:
    """
    Create S3 path from parts
    
    Args:
        bucket: S3 bucket name
        *parts: Path components
        
    Returns:
        Full S3 path
        
    Example:
        >>> create_s3_path('my-bucket', 'data', 'file.csv')
        's3://my-bucket/data/file.csv'
    """
    path = '/'.join(parts)
    return f"s3://{bucket}/{path}"


def parse_s3_path(s3_path: str) -> tuple:
    """
    Parse S3 path into bucket and key
    
    Args:
        s3_path: S3 path (s3://bucket/key)
        
    Returns:
        Tuple of (bucket, key)
        
    Example:
        >>> parse_s3_path('s3://my-bucket/data/file.csv')
        ('my-bucket', 'data/file.csv')
    """
    if not s3_path.startswith('s3://'):
        raise ValueError(f"Invalid S3 path: {s3_path}")
    
    parts = s3_path[5:].split('/', 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ''
    
    return bucket, key