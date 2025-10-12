"""
Configuration Manager
Handles loading and accessing configuration from YAML files and environment variables
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ConfigManager:
    """Manage configuration from YAML files and environment variables"""
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize configuration manager
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self._aws_config = self._load_yaml("aws_config.yaml")
        self._preprocessing_config = self._load_yaml("preprocessing_config.yaml")
        
        # Override with environment variables
        self._apply_env_overrides()
    
    def _load_yaml(self, filename: str) -> Dict:
        """Load YAML configuration file"""
        config_path = self.config_dir / filename
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _apply_env_overrides(self):
        """Override configuration with environment variables"""
        # AWS overrides
        if os.getenv('AWS_REGION'):
            self._aws_config['aws']['region'] = os.getenv('AWS_REGION')
        
        if os.getenv('AWS_PROFILE'):
            self._aws_config['aws']['profile'] = os.getenv('AWS_PROFILE')
        
        if os.getenv('AWS_ACCOUNT_ID'):
            self._aws_config['aws']['account_id'] = os.getenv('AWS_ACCOUNT_ID')
        
        # S3 bucket overrides
        if os.getenv('OUTPUT_BUCKET'):
            self._aws_config['aws']['s3']['output_bucket'] = os.getenv('OUTPUT_BUCKET')
        
        if os.getenv('TEMP_BUCKET'):
            self._aws_config['aws']['s3']['temp_bucket'] = os.getenv('TEMP_BUCKET')
        
        # Processing overrides
        if os.getenv('CHUNK_SIZE'):
            self._preprocessing_config['preprocessing']['phase1']['chunk_size'] = int(os.getenv('CHUNK_SIZE'))
        
        if os.getenv('MAX_WORKERS'):
            self._preprocessing_config['processing']['parallel_workers'] = int(os.getenv('MAX_WORKERS'))
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key_path: Dot-separated path to config value (e.g., 'aws.region')
            default: Default value if key not found
            
        Returns:
            Configuration value
            
        Examples:
            >>> config.get('aws.region')
            'us-east-1'
            >>> config.get('aws.s3.output_bucket')
            'my-output-bucket'
        """
        keys = key_path.split('.')
        
        # Try AWS config first
        value = self._get_nested(self._aws_config, keys)
        if value is not None:
            return value
        
        # Try preprocessing config
        value = self._get_nested(self._preprocessing_config, keys)
        if value is not None:
            return value
        
        return default
    
    def _get_nested(self, config: Dict, keys: list) -> Any:
        """Get nested dictionary value"""
        try:
            value = config
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return None
    
    def get_aws_config(self) -> Dict:
        """Get full AWS configuration"""
        return self._aws_config.copy()
    
    def get_preprocessing_config(self) -> Dict:
        """Get full preprocessing configuration"""
        return self._preprocessing_config.copy()
    
    def get_s3_paths(self) -> Dict[str, str]:
        """Get all S3 bucket paths"""
        return {
            'mimic_bucket': self.get('aws.s3.mimic_bucket'),
            'output_bucket': self.get('aws.s3.output_bucket'),
            'temp_bucket': self.get('aws.s3.temp_bucket')
        }
    
    def get_data_path(self, dataset: str, file_key: str) -> str:
        """
        Get full S3 path for a data file
        
        Args:
            dataset: Dataset name (mimic_iv, mimic_ed, mimic_cxr)
            file_key: File key in configuration
            
        Returns:
            Full S3 path
            
        Example:
            >>> config.get_data_path('mimic_ed', 'edstays')
            'mimic-iv-ed/2.2/ed/edstays.csv.gz'
        """
        prefix = self.get(f'data_paths.{dataset}.prefix', '')
        file_path = self.get(f'data_paths.{dataset}.{file_key}', '')
        return f"{prefix}{file_path}"


# Global config instance
_config_instance: Optional[ConfigManager] = None


def get_config(config_dir: str = "config") -> ConfigManager:
    """
    Get global configuration instance (singleton pattern)
    
    Args:
        config_dir: Directory containing configuration files
        
    Returns:
        ConfigManager instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = ConfigManager(config_dir)
    
    return _config_instance