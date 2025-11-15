"""
Phase 1: Data Preprocessing Pipeline for Enhanced MDF-Net
Adapts MDF-Net's data processing for use with:
- BiomedCLIP-CXR vision encoder
- Clinical ModernBERT text encoder
- RAG knowledge enhancement
- Cross-attention fusion

Google Cloud Storage Support: Reads from and writes to GCS buckets for cloud deployment
Supports multiple buckets (your data + PhysioNet's public MIMIC-CXR bucket)

PERFORMANCE IMPROVEMENTS:
- Parallel batch loading with prefetching
- Optimized GCS I/O with retry policies
- Larger write batches to reduce I/O operations
- Monitoring and timing instrumentation
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import pickle
from tqdm import tqdm
import logging
import argparse
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import lru_cache
import multiprocessing as mp
import time
import queue
import random

# Google Cloud Storage
try:
    from google.cloud import storage
    from google.api_core import retry
    from google.api_core import timeout as api_timeout
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    storage = None
    retry = None
    api_timeout = None

# Image processing
from PIL import Image
import cv2
import torch
from torchvision import transforms

# Text processing
from transformers import AutoTokenizer
import re

# Medical knowledge processing
import faiss
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Configuration for data preprocessing"""
    # Paths (local or GCS prefixes)
    mimic_cxr_path: str = "~/Documents/Portfolio/MIMIC_Data/physionet.org/files/mimic-cxr-jpg-2.1.0"
    mimic_iv_path: str =  "~/Documents/Portfolio/MIMIC_Data/physionet.org/files/mimiciv/3.1"
    mimic_ed_path: str = "~/Documents/Portfolio/MIMIC_Data/physionet.org/files/mimic-iv-ed/2.2"  # Separate MIMIC-IV-ED dataset
    reflacx_path: str = "~/Documents/Portfolio/MIMIC_Data/physionet.org/files/reflacx"  # Eye-gaze annotations like MDF-Net
    output_path: str = "~/Documents/Portfolio/MIMIC_Data/physionet.org/"

    # Google Cloud Storage Settings
    use_gcs: bool = False
    gcs_bucket: Optional[str] = None  # Your bucket (e.g., "bergermimiciv")
    gcs_cxr_bucket: Optional[str] = None  # MIMIC-CXR bucket (e.g., "mimic-cxr-jpg-2.1.0.physionet.org")
    output_gcs_bucket: Optional[str] = None  # Output bucket (usually same as gcs_bucket)
    gcs_project_id: Optional[str] = None  # GCP project ID for requester pays buckets

    # Image settings (BiomedCLIP-CXR requirements)
    image_size: int = 518  # Higher resolution than MDF-Net's 224
    normalize_mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
    normalize_std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)

    # Text settings (Clinical ModernBERT)
    max_text_length: int = 8192  # ModernBERT's extended context
    clinical_vocab_size: int = 100000

    # RAG settings
    knowledge_base_path: str = "/data/medical_knowledge"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k_retrieval: int = 5

    # Fusion settings
    num_visual_tokens: int = 256  # For cross-attention
    num_text_tokens: int = 512

    # Data splits
    # IMPORTANT: Set use_official_splits=True to use MIMIC-CXR's recommended patient-level splits
    # This ensures reproducibility and proper patient-level splitting (no patient in multiple splits)
    use_official_splits: bool = True  # Recommended for all experiments
    official_split_file: str = "mimic-cxr-2.0.0-split.csv.gz"  # Located in mimic_cxr_path

    # Custom split ratios (only used if use_official_splits=False)
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15

    # Clinical features (expanded from MDF-Net)
    clinical_features: List[str] = field(default_factory=lambda: [
        'temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp',
        'pain', 'acuity', 'age', 'gender', 'chiefcomplaint',
        'admittime', 'dischtime', 'los_hours', 'prev_admissions',
        'medications', 'lab_results', 'allergies', 'medical_history'
    ])


class GCSHelper:
    """Helper class for Google Cloud Storage operations with multi-bucket support and optimizations"""

    def __init__(self, config: DataConfig):
        self.config = config
        self.gcs_client = None
        self.bucket = None
        self.cxr_bucket = None
        self.output_bucket = None
        self.retry_policy = None

        if config.use_gcs:
            if not GCS_AVAILABLE:
                raise ImportError("google-cloud-storage is not installed. Run: pip install google-cloud-storage")

            # Initialize client with optimized settings
            if config.gcs_project_id:
                self.gcs_client = storage.Client(
                    project=config.gcs_project_id,
                    client_options={'api_endpoint': 'https://storage.googleapis.com'}
                )
            else:
                self.gcs_client = storage.Client(
                    client_options={'api_endpoint': 'https://storage.googleapis.com'}
                )

            # Configure retry policy for robustness
            if retry:
                self.retry_policy = retry.Retry(
                    initial=1.0,      # Initial delay between retries
                    maximum=60.0,     # Maximum delay between retries
                    multiplier=2.0,   # Exponential backoff multiplier
                    deadline=300.0,   # Total timeout for the operation
                )
            else:
                self.retry_policy = None

            # Initialize your main bucket (MIMIC-IV, MIMIC-IV-ED, etc.)
            if config.gcs_bucket:
                self.bucket = self.gcs_client.bucket(config.gcs_bucket)
                logger.info(f"Initialized GCS client for bucket: {config.gcs_bucket}")

            # Initialize separate MIMIC-CXR bucket (PhysioNet's requester pays bucket)
            if config.gcs_cxr_bucket:
                self.cxr_bucket = self.gcs_client.bucket(config.gcs_cxr_bucket, user_project=config.gcs_project_id)
                logger.info(f"Initialized GCS MIMIC-CXR bucket: {config.gcs_cxr_bucket} (requester pays)")

            # Initialize output bucket
            if config.output_gcs_bucket:
                self.output_bucket = self.gcs_client.bucket(config.output_gcs_bucket)
                logger.info(f"Initialized GCS output bucket: {config.output_gcs_bucket}")

    def _get_bucket_for_path(self, path: str):
        """Determine which bucket to use based on path"""
        path_str = str(path)
        # If path is for MIMIC-CXR data and we have a separate CXR bucket, use it
        # Check for either directory path or MIMIC-CXR metadata files
        cxr_indicators = ['mimic-cxr-jpg', 'mimic-cxr-2.0.0', 'files/p']
        if self.cxr_bucket and any(indicator in path_str for indicator in cxr_indicators):
            return self.cxr_bucket
        # Otherwise use main bucket
        return self.bucket

    def read_csv(self, path, **kwargs) -> pd.DataFrame:
        """Read CSV from GCS or local filesystem"""
        if self.config.use_gcs:
            bucket = self._get_bucket_for_path(path)
            gcs_key = path if isinstance(path, str) else str(path)
            logger.info(f"  Reading from GCS: gs://{bucket.name}/{gcs_key}")
            blob = bucket.blob(gcs_key)
            if self.retry_policy:
                data = blob.download_as_bytes(retry=self.retry_policy)
            else:
                data = blob.download_as_bytes()
            return pd.read_csv(BytesIO(data), **kwargs)
        else:
            return pd.read_csv(path, **kwargs)

    def read_image(self, path):
        """Read image from GCS or local filesystem"""
        if self.config.use_gcs:
            bucket = self._get_bucket_for_path(path)
            gcs_key = path if isinstance(path, str) else str(path)
            blob = bucket.blob(gcs_key)
            if self.retry_policy:
                data = blob.download_as_bytes(retry=self.retry_policy)
            else:
                data = blob.download_as_bytes()
            return Image.open(BytesIO(data))
        else:
            return Image.open(path)

    def read_image_cv2(self, path):
        """Read image with OpenCV from GCS or local"""
        if self.config.use_gcs:
            bucket = self._get_bucket_for_path(path)
            gcs_key = path if isinstance(path, str) else str(path)
            blob = bucket.blob(gcs_key)
            if self.retry_policy:
                data = blob.download_as_bytes(retry=self.retry_policy)
            else:
                data = blob.download_as_bytes()
            img_array = np.frombuffer(data, np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        else:
            return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    def path_exists(self, path) -> bool:
        """Check if path exists in GCS or locally"""
        if self.config.use_gcs:
            bucket = self._get_bucket_for_path(path)
            gcs_key = path if isinstance(path, str) else str(path)
            blob = bucket.blob(gcs_key)
            if self.retry_policy:
                return blob.exists(retry=self.retry_policy)
            else:
                return blob.exists()
        else:
            return Path(path).exists()

    def write_pickle(self, data, path):
        """Write pickle file to GCS or local"""
        if self.output_bucket:
            gcs_key = path if isinstance(path, str) else str(path)
            buffer = BytesIO()
            pickle.dump(data, buffer)
            buffer.seek(0)
            blob = self.output_bucket.blob(gcs_key)
            blob.chunk_size = 8 * 1024 * 1024  # 8MB chunks
            if self.retry_policy:
                blob.upload_from_file(buffer, rewind=True, retry=self.retry_policy)
            else:
                blob.upload_from_file(buffer, rewind=True)
            logger.info(f"Saved to GCS: gs://{self.output_bucket.name}/{gcs_key}")
        else:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved locally: {path}")

    def write_torch(self, data, path):
        """
        Write data using torch.save() with optimized GCS upload
        
        Args:
            data: Data to save (typically list of dicts with tensors)
            path: Output path (will use .pt extension)
        """
        import torch
        import tempfile

        if self.output_bucket:
            # Save to temporary file first, then upload
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
                tmp_path = tmp_file.name
                torch.save(data, tmp_path)
            
            # Use optimized upload settings
            gcs_key = path if isinstance(path, str) else str(path)
            blob = self.output_bucket.blob(gcs_key)
            blob.chunk_size = 8 * 1024 * 1024  # 8MB chunks for multipart upload
            
            # Upload with retry policy if available
            if self.retry_policy:
                blob.upload_from_filename(
                    tmp_path,
                    retry=self.retry_policy,
                    timeout=300
                )
            else:
                blob.upload_from_filename(tmp_path, timeout=300)
            
            # Clean up temp file
            Path(tmp_path).unlink()
            logger.debug(f"Saved to GCS using torch.save: gs://{self.output_bucket.name}/{gcs_key}")
        else:
            # Save locally using torch.save
            torch.save(data, path)
            logger.debug(f"Saved locally using torch.save: {path}")

    def read_torch(self, path):
        """
        Read data using torch.load() with retry support
        
        Args:
            path: Path to .pt file
            
        Returns:
            Loaded data
        """
        import torch
        import tempfile

        if self.output_bucket:
            # Download from GCS to temp file, then load
            gcs_key = path if isinstance(path, str) else str(path)
            blob = self.output_bucket.blob(gcs_key)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
                tmp_path = tmp_file.name
                if self.retry_policy:
                    blob.download_to_filename(tmp_path, retry=self.retry_policy)
                else:
                    blob.download_to_filename(tmp_path)
            
            # Load from temp file
            # Use weights_only=False to allow loading numpy arrays and custom objects
            # This is safe since we trust our own checkpoint files
            data = torch.load(tmp_path, map_location='cpu', weights_only=False)
            
            # Clean up temp file
            Path(tmp_path).unlink()
            logger.debug(f"Loaded from GCS using torch.load: gs://{self.output_bucket.name}/{gcs_key}")
            return data
        else:
            # Load locally using torch.load
            # Use weights_only=False to allow loading numpy arrays and custom objects
            # This is safe since we trust our own checkpoint files
            data = torch.load(path, map_location='cpu', weights_only=False)
            logger.debug(f"Loaded locally using torch.load: {path}")
            return data

    def write_json(self, data, path):
        """Write JSON file to GCS or local"""
        if self.output_bucket:
            gcs_key = path if isinstance(path, str) else str(path)
            json_str = json.dumps(data, indent=2, default=str)
            blob = self.output_bucket.blob(gcs_key)
            if self.retry_policy:
                blob.upload_from_string(json_str, content_type='application/json', retry=self.retry_policy)
            else:
                blob.upload_from_string(json_str, content_type='application/json')
            logger.info(f"Saved to GCS: gs://{self.output_bucket.name}/{gcs_key}")
        else:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Saved locally: {path}")


class MIMICDataJoiner:
    """
    Joins MIMIC-CXR images with MIMIC-IV clinical data
    Adapted from MDF-Net's joining strategy
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.gcs_helper = GCSHelper(config)
        self.subject_mapping = {}
        self.study_mapping = {}
        
    def load_mimic_cxr_metadata(self) -> pd.DataFrame:
        """Load MIMIC-CXR metadata"""
        if self.config.use_gcs:
            # For GCS, mimic_cxr_path should be empty/root since files are at bucket root
            if self.config.mimic_cxr_path:
                metadata_path = f"{self.config.mimic_cxr_path}/mimic-cxr-2.0.0-metadata.csv"
            else:
                metadata_path = "mimic-cxr-2.0.0-metadata.csv"
        else:
            metadata_path = Path(self.config.mimic_cxr_path).expanduser() / "mimic-cxr-2.0.0-metadata.csv"
        
        logger.info(f"Loading MIMIC-CXR metadata from {metadata_path}")
        return self.gcs_helper.read_csv(metadata_path)
    
    def load_mimic_iv_data(self) -> Dict[str, pd.DataFrame]:
        """Load relevant MIMIC-IV tables"""
        tables = {}
        
        # Core patient data
        core_tables = [
            'core/patients.csv',
            'core/admissions.csv', 
            'core/transfers.csv'
        ]
        
        # Hospital data
        hosp_tables = [
            'hosp/diagnoses_icd.csv',
            'hosp/procedures_icd.csv',
            'hosp/prescriptions.csv',
            'hosp/labevents.csv',
            'hosp/microbiologyevents.csv'
        ]
        
        # ICU data (if needed)
        icu_tables = [
            'icu/chartevents.csv',  # Vitals
            'icu/d_items.csv'  # Item definitions
        ]
        
        all_tables = core_tables + hosp_tables + icu_tables
        
        for table in all_tables:
            if self.config.use_gcs:
                table_path = f"{self.config.mimic_iv_path}/{table}"
            else:
                table_path = Path(self.config.mimic_iv_path).expanduser() / table
            
            table_name = table.replace('/', '_').replace('.csv', '')
            logger.info(f"Loading MIMIC-IV table: {table}")
            
            # For large tables, only load relevant columns
            if 'labevents' in table or 'chartevents' in table:
                # These are huge - sample or filter
                tables[table_name] = self.gcs_helper.read_csv(
                    table_path, 
                    nrows=1000000  # Limit for testing
                )
            else:
                tables[table_name] = self.gcs_helper.read_csv(table_path)
                
        return tables
    
    def load_mimic_ed_data(self) -> Dict[str, pd.DataFrame]:
        """Load MIMIC-IV-ED emergency department data"""
        tables = {}
        
        ed_tables = [
            'ed/edstays.csv',
            'ed/triage.csv',
            'ed/vitalsign.csv',
            'ed/pyxis.csv',  # Medication dispensing
            'ed/medrecon.csv'  # Medication reconciliation
        ]
        
        for table in ed_tables:
            if self.config.use_gcs:
                table_path = f"{self.config.mimic_ed_path}/{table}"
            else:
                table_path = Path(self.config.mimic_ed_path).expanduser() / table
            
            table_name = table.replace('/', '_').replace('.csv', '')
            logger.info(f"Loading MIMIC-IV-ED table: {table}")
            tables[table_name] = self.gcs_helper.read_csv(table_path)
            
        return tables
    
    def join_data(self) -> pd.DataFrame:
        """
        Join MIMIC-CXR with clinical data from MIMIC-IV and MIMIC-IV-ED
        Returns a DataFrame with one row per chest X-ray
        """
        # Load all data sources
        cxr_metadata = self.load_mimic_cxr_metadata()
        mimic_iv = self.load_mimic_iv_data()
        mimic_ed = self.load_mimic_ed_data()
        
        # Start with CXR metadata as base
        joined_df = cxr_metadata.copy()
        
        # Join with MIMIC-IV admissions
        if 'core_admissions' in mimic_iv:
            joined_df = pd.merge(
                joined_df,
                mimic_iv['core_admissions'][['subject_id', 'hadm_id', 'admittime', 'dischtime', 
                                             'admission_type', 'admission_location', 'discharge_location',
                                             'insurance', 'marital_status', 'ethnicity']],
                on=['subject_id', 'hadm_id'],
                how='left'
            )
        
        # Join with patient demographics
        if 'core_patients' in mimic_iv:
            joined_df = pd.merge(
                joined_df,
                mimic_iv['core_patients'][['subject_id', 'gender', 'dod']],
                on='subject_id',
                how='left'
            )
            
            # Calculate age at study time
            joined_df['age'] = pd.to_datetime(joined_df['StudyDate'], format='%Y%m%d').dt.year - \
                               pd.to_datetime(joined_df['dod']).dt.year
        
        # Join with ED data if available
        if 'ed_edstays' in mimic_ed:
            # Find ED stays that match the admission
            ed_stays = mimic_ed['ed_edstays'][['subject_id', 'hadm_id', 'stay_id', 
                                               'intime', 'outtime', 'disposition']]
            joined_df = pd.merge(
                joined_df,
                ed_stays,
                on=['subject_id', 'hadm_id'],
                how='left'
            )
            
            # Get ED triage data
            if 'ed_triage' in mimic_ed:
                triage = mimic_ed['ed_triage'][['subject_id', 'stay_id', 'temperature', 
                                                'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp',
                                                'pain', 'acuity', 'chiefcomplaint']]
                joined_df = pd.merge(
                    joined_df,
                    triage,
                    on=['subject_id', 'stay_id'],
                    how='left'
                )
        
        # Add diagnosis information
        if 'hosp_diagnoses_icd' in mimic_iv:
            # Get primary diagnosis
            primary_diag = mimic_iv['hosp_diagnoses_icd'][
                mimic_iv['hosp_diagnoses_icd']['seq_num'] == 1
            ][['subject_id', 'hadm_id', 'icd_code', 'icd_version']]
            primary_diag.rename(columns={'icd_code': 'primary_diagnosis'}, inplace=True)
            
            joined_df = pd.merge(
                joined_df,
                primary_diag,
                on=['subject_id', 'hadm_id'],
                how='left'
            )
        
        logger.info(f"Joined data shape: {joined_df.shape}")
        logger.info(f"Columns: {joined_df.columns.tolist()}")
        
        return joined_df


class ImageProcessor:
    """
    Processes chest X-ray images for BiomedCLIP-CXR
    Handles higher resolution (518x518) compared to MDF-Net's 224x224
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.gcs_helper = GCSHelper(config)
        
        # BiomedCLIP-CXR specific transforms
        self.transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.normalize_mean, std=config.normalize_std)
        ])
        
    def process_image(self, image_path: str) -> torch.Tensor:
        """Process a single chest X-ray image"""
        try:
            # Load image
            image = self.gcs_helper.read_image(image_path)
            
            # Convert to RGB if grayscale
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply transforms
            image_tensor = self.transform(image)
            
            return image_tensor
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            # Return a zero tensor of the expected shape
            return torch.zeros(3, self.config.image_size, self.config.image_size)
    
    def extract_visual_attention_regions(self, image_path: str) -> Dict[str, torch.Tensor]:
        """
        Extract attention regions following MDF-Net's approach
        Uses basic saliency detection for demo - should be replaced with REFLACX eye-gaze data
        """
        # Load grayscale for saliency
        img_gray = self.gcs_helper.read_image_cv2(image_path)
        
        # Simple saliency using gradient magnitude
        grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
        saliency = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
        
        # Find top regions
        threshold = np.percentile(saliency, 90)
        salient_mask = saliency > threshold
        
        return {
            'saliency_map': torch.from_numpy(saliency).float(),
            'salient_regions': torch.from_numpy(salient_mask).float()
        }


class TextProcessor:
    """
    Processes clinical text for Clinical ModernBERT
    Handles extended context (8192 tokens) and clinical terminology
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        
        # Initialize tokenizer (using placeholder - replace with actual Clinical ModernBERT)
        self.tokenizer = AutoTokenizer.from_pretrained(
            'bert-base-uncased',  # Replace with Clinical ModernBERT when available
            max_length=config.max_text_length
        )
        
        # Clinical abbreviation expansion
        self.abbreviations = {
            'htn': 'hypertension',
            'dm': 'diabetes mellitus',
            'cad': 'coronary artery disease',
            'copd': 'chronic obstructive pulmonary disease',
            'chf': 'congestive heart failure',
            'mi': 'myocardial infarction',
            'sob': 'shortness of breath',
            'cp': 'chest pain'
        }
        
    def preprocess_clinical_text(self, text: str) -> str:
        """Preprocess clinical notes"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        # Expand abbreviations
        for abbr, expansion in self.abbreviations.items():
            text = re.sub(r'\b' + abbr + r'\b', expansion, text)
        
        # Remove patient identifiers (simplified - use proper de-identification in production)
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)  # SSN pattern
        text = re.sub(r'\b\d{10}\b', '[MRN]', text)  # MRN pattern
        
        return text
    
    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize clinical text"""
        processed_text = self.preprocess_clinical_text(text)
        
        encoding = self.tokenizer(
            processed_text,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_text_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }
    
    def extract_clinical_features(self, row: pd.Series) -> torch.Tensor:
        """Extract numerical clinical features"""
        features = []
        
        for feature in self.config.clinical_features:
            if feature in row and not pd.isna(row[feature]):
                if isinstance(row[feature], (int, float)):
                    features.append(float(row[feature]))
                else:
                    # Handle categorical as 0/1
                    features.append(1.0)
            else:
                features.append(0.0)  # Missing value
        
        return torch.tensor(features, dtype=torch.float32)


class RAGKnowledgeBase:
    """
    Retrieval-Augmented Generation knowledge base for medical information
    Enhances model with external medical knowledge
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.gcs_helper = GCSHelper(config)
        
        # Initialize embedding model
        logger.info(f"Load pretrained SentenceTransformer: {config.embedding_model}")
        self.encoder = SentenceTransformer(config.embedding_model)
        
        # Initialize FAISS index
        self.index = None
        self.documents = []
        
        # Build knowledge base
        self.build_index()
    
    def build_index(self):
        """Build FAISS index from medical knowledge documents"""
        logger.info("Building knowledge base index...")
        
        # For demo, use synthetic medical knowledge
        # In production, load from actual medical databases
        sample_knowledge = [
            "Pneumonia appears as consolidation or ground-glass opacities on chest X-ray, typically in a lobar or segmental distribution.",
            "Congestive heart failure manifests as cardiomegaly, pulmonary edema, pleural effusions, and Kerley B lines on chest radiography."
        ]
        
        # Encode documents
        embeddings = self.encoder.encode(sample_knowledge, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        self.documents = sample_knowledge
        logger.info(f"Built index with {len(self.documents)} documents")
    
    def retrieve(self, query: str, top_k: int = None) -> List[str]:
        """Retrieve relevant medical knowledge for a query"""
        if top_k is None:
            top_k = self.config.top_k_retrieval
            
        # Encode query
        query_embedding = self.encoder.encode([query])
        
        # Search index
        distances, indices = self.index.search(
            query_embedding.astype('float32'), 
            min(top_k, len(self.documents))
        )
        
        # Return retrieved documents
        retrieved = [self.documents[idx] for idx in indices[0]]
        return retrieved


class DatasetCreator:
    """
    Creates the final dataset by combining all components
    Follows MDF-Net's data pipeline with enhancements
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.gcs_helper = GCSHelper(config)

        # Initialize processors
        self.image_processor = ImageProcessor(config)
        self.text_processor = TextProcessor(config)
        self.knowledge_base = RAGKnowledgeBase(config)

        # Initialize data joiner
        self.data_joiner = MIMICDataJoiner(config)

        # Timing statistics
        self.timing_stats = {
            'load': 0,
            'process': 0,
            'write': 0,
            'total': 0
        }

        # Load official MIMIC-CXR splits if configured
        self.official_splits_map = None
        if self.config.use_official_splits:
            self.official_splits_map = self.load_official_splits()

    def load_official_splits(self) -> Dict[Tuple[int, int], str]:
        """
        Load the official MIMIC-CXR train/validate/test splits.

        These are patient-level splits ensuring all images from the same patient
        are in the same split (prevents data leakage).

        Returns:
            Dictionary mapping (subject_id, study_id) -> split_name
            where split_name is 'train', 'validate', or 'test'
        """
        logger.info("=" * 60)
        logger.info("Loading official MIMIC-CXR splits")
        logger.info("=" * 60)

        try:
            # Build path to split file
            if self.config.use_gcs:
                split_path = f"{self.config.mimic_cxr_path}/{self.config.official_split_file}"
            else:
                base_path = Path(self.config.mimic_cxr_path).expanduser()
                split_path = base_path / self.config.official_split_file

            logger.info(f"Loading split file from: {split_path}")

            # Read split file (it's gzipped)
            split_df = self.gcs_helper.read_csv(split_path)

            logger.info(f"Loaded {len(split_df)} study assignments")

            # Validate columns
            required_cols = ['subject_id', 'study_id', 'split']
            missing_cols = [col for col in required_cols if col not in split_df.columns]
            if missing_cols:
                raise ValueError(f"Split file missing required columns: {missing_cols}")

            # Create mapping: (subject_id, study_id) -> split_name
            splits_map = {}
            for _, row in split_df.iterrows():
                key = (int(row['subject_id']), int(row['study_id']))
                split_name = row['split'].strip().lower()

                # Normalize split names: 'validate' -> 'val' for consistency
                if split_name == 'validate':
                    split_name = 'val'

                splits_map[key] = split_name

            # Count splits
            split_counts = defaultdict(int)
            for split_name in splits_map.values():
                split_counts[split_name] += 1

            logger.info("Split distribution:")
            for split_name in ['train', 'val', 'test']:
                count = split_counts.get(split_name, 0)
                pct = 100.0 * count / len(splits_map) if splits_map else 0
                logger.info(f"  {split_name:6s}: {count:7,} studies ({pct:5.1f}%)")

            logger.info("=" * 60)
            logger.info("✓ Official splits loaded successfully")
            logger.info("✓ Patient-level splitting ensures no data leakage")
            logger.info("=" * 60)

            return splits_map

        except FileNotFoundError as e:
            logger.error(f"Split file not found: {split_path}")
            logger.error("Please ensure mimic-cxr-2.0.0-split.csv.gz exists in the MIMIC-CXR directory")
            logger.error("Falling back to custom stratified splits (NOT RECOMMENDED)")
            return None
        except Exception as e:
            logger.error(f"Error loading official splits: {e}")
            logger.error("Falling back to custom stratified splits (NOT RECOMMENDED)")
            import traceback
            traceback.print_exc()
            return None

    def process_single_record(self, row: pd.Series) -> Dict[str, Any]:
        """Process a single data record"""
        try:
            # Build image path
            if self.config.use_gcs:
                # For GCS with PhysioNet bucket, files are at root
                subject_id = f"p{str(row['subject_id'])[:2]}"
                patient_id = f"p{row['subject_id']}"
                study_id = f"s{row['study_id']}"
                
                if self.config.mimic_cxr_path:
                    # Custom bucket with path prefix
                    image_path = f"{self.config.mimic_cxr_path}/files/{subject_id}/{patient_id}/{study_id}/{row['dicom_id']}.jpg"
                else:
                    # PhysioNet bucket - files at root
                    image_path = f"files/{subject_id}/{patient_id}/{study_id}/{row['dicom_id']}.jpg"
            else:
                # Local path
                subject_id = f"p{str(row['subject_id'])[:2]}"
                patient_id = f"p{row['subject_id']}"
                study_id = f"s{row['study_id']}"
                base_path = Path(self.config.mimic_cxr_path).expanduser()
                image_path = base_path / f"files/{subject_id}/{patient_id}/{study_id}/{row['dicom_id']}.jpg"
                image_path = str(image_path)
            
            # Process image
            image_tensor = self.image_processor.process_image(image_path)
            
            # Extract attention regions (if REFLACX data available)
            attention_data = self.image_processor.extract_visual_attention_regions(image_path)
            
            # Process clinical text (chief complaint)
            text_data = {}
            if 'chiefcomplaint' in row:
                text_data = self.text_processor.tokenize(row['chiefcomplaint'])
            
            # Extract clinical features
            clinical_features = self.text_processor.extract_clinical_features(row)
            
            # Retrieve relevant medical knowledge
            if 'ViewPosition' in row:
                query = f"chest x-ray {row['ViewPosition']} view findings"
                retrieved_knowledge = self.knowledge_base.retrieve(query)
            else:
                retrieved_knowledge = []
            
            # Build record
            record = {
                # Identifiers
                'subject_id': row['subject_id'],
                'study_id': row['study_id'],
                'dicom_id': row['dicom_id'],
                
                # Image data
                'image': image_tensor,
                'attention_regions': attention_data,
                
                # Text data
                'text_tokens': text_data,
                'clinical_features': clinical_features,
                
                # Knowledge
                'retrieved_knowledge': retrieved_knowledge,
                
                # Labels (for supervised training)
                'labels': {
                    'view_position': row.get('ViewPosition', 'UNKNOWN'),
                    # Add more labels as needed
                }
            }
            
            return record
            
        except Exception as e:
            logger.error(f"Error processing record {row.get('dicom_id', 'unknown')}: {e}")
            return None
    
    def process_batch_parallel(self, df_batch: pd.DataFrame, num_workers: int = 4) -> List[Dict]:
        """Process a batch of records in parallel"""
        records = []
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all records for processing
            futures = {
                executor.submit(self.process_single_record, row): idx
                for idx, row in df_batch.iterrows()
            }
            
            # Collect results
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing records", leave=False):
                record = future.result()
                if record is not None:
                    records.append(record)
        
        return records
    
    def create_dataset(self, batch_size: int = 100, num_workers: int = 4,
                      create_small_samples: bool = False, small_sample_size: int = 100):
        """
        Create the complete dataset with batched processing
        
        Args:
            batch_size: Number of records to process in each batch
            num_workers: Number of parallel workers
            create_small_samples: Whether to create small sample datasets
            small_sample_size: Size of small sample datasets
        """
        logger.info("=" * 60)
        logger.info("Starting dataset creation")
        logger.info("=" * 60)
        
        # Load and join data
        logger.info("Loading and joining MIMIC data...")
        joined_data = self.data_joiner.join_data()
        
        # Filter to frontal views only (following MDF-Net)
        frontal_views = joined_data[
            joined_data['ViewPosition'].isin(['PA', 'AP'])
        ].copy()
        
        logger.info(f"Filtered to {len(frontal_views)} frontal view images")
        
        # Process in batches to manage memory
        total_batches = (len(frontal_views) + batch_size - 1) // batch_size
        
        # Create output directory
        if self.config.use_gcs:
            intermediate_path = f"{self.config.output_path}/intermediate_batches"
        else:
            output_dir = Path(self.config.output_path).expanduser()
            output_dir.mkdir(parents=True, exist_ok=True)
            intermediate_path = output_dir / "intermediate_batches"
            intermediate_path.mkdir(exist_ok=True)
        
        logger.info(f"Processing {total_batches} batches of size {batch_size}")
        
        for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(frontal_views))
            
            # Get batch
            batch_df = frontal_views.iloc[start_idx:end_idx]
            
            # Process batch
            batch_records = self.process_batch_parallel(batch_df, num_workers)
            
            # Save intermediate batch using torch.save (more efficient)
            if self.config.use_gcs:
                batch_file = f"{intermediate_path}/batch_{batch_idx:05d}.pt"
            else:
                batch_file = intermediate_path / f"batch_{batch_idx:05d}.pt"
            
            self.gcs_helper.write_torch(batch_records, batch_file)
            logger.info(f"Saved batch {batch_idx}/{total_batches} with {len(batch_records)} records")
        
        # Create train/val/test splits from batches
        logger.info("Creating train/val/test splits from batches...")
        self.create_splits_from_batches_streaming(
            create_small_samples=create_small_samples,
            small_sample_size=small_sample_size
        )
    
    def get_batch_file_list(self, max_batches: Optional[int] = None):
        """Get list of intermediate batch files"""
        if self.config.use_gcs:
            # List blobs in GCS
            from google.cloud import storage
            client = storage.Client(project=self.config.gcs_project_id)
            bucket = client.bucket(self.config.output_gcs_bucket)
            
            # Look for batch files in the output path directly
            prefix = f"{self.config.output_path}/"
            blobs = list(bucket.list_blobs(prefix=prefix))

            # Filter to intermediate_batch_*.pt files only
            batch_blobs = [b for b in blobs if b.name.endswith('.pt') and 'intermediate_batch_' in b.name]
            batch_blobs.sort(key=lambda b: b.name)

            if max_batches:
                batch_blobs = batch_blobs[:max_batches]
                logger.info(f"Found {len(batch_blobs)} intermediate batch files in GCS (limited to {max_batches})")
            else:
                logger.info(f"Found {len(batch_blobs)} intermediate batch files in GCS")

            return batch_blobs
        else:
            # List local files directly in output path
            intermediate_path = Path(self.config.output_path).expanduser()
            batch_files = sorted(intermediate_path.glob("intermediate_batch_*.pt"))

            if max_batches:
                batch_files = batch_files[:max_batches]
                logger.info(f"Found {len(batch_files)} intermediate batch files locally (limited to {max_batches})")
            else:
                logger.info(f"Found {len(batch_files)} intermediate batch files locally")

            return batch_files
    
    def count_and_extract_streaming(self, batch_files) -> Tuple[int, List[Tuple]]:
        """
        Count records and extract stratification keys in a single streaming pass.
        More efficient than two separate passes.
        
        Returns:
            total_count: Total number of records
            strat_keys: List of (view_position, has_clinical) tuples for stratification
        """
        logger.info("Counting records and extracting labels in one pass...")
        
        total_count = 0
        strat_keys = []
        
        # Function to load and process a single batch
        def process_batch_for_counting(batch_file):
            try:
                if self.config.use_gcs:
                    records = self.gcs_helper.read_torch(batch_file.name)
                else:
                    records = self.gcs_helper.read_torch(str(batch_file))
                
                batch_keys = []
                for record in records:
                    # Extract stratification key
                    view = record['labels'].get('view_position', 'UNKNOWN')
                    has_clinical = len(record.get('clinical_features', [])) > 0
                    batch_keys.append((view, has_clinical))
                
                return len(records), batch_keys
            except Exception as e:
                logger.error(f"Error processing batch {batch_file}: {e}")
                return 0, []
        
        # Process with progress bar
        # CRITICAL: max_workers=1 to prevent OOM (4 workers = 4GB+ in memory simultaneously)
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(process_batch_for_counting, bf) for bf in batch_files]

            for future in tqdm(as_completed(futures), total=len(batch_files), desc="Counting & extracting"):
                count, keys = future.result()
                total_count += count
                strat_keys.extend(keys)
        
        logger.info(f"Total: {total_count} records, {len(strat_keys)} stratification keys")
        return total_count, strat_keys
    
    def create_official_split_map_streaming(self, batch_files, total_count: int) -> Dict[int, str]:
        """
        Create split map using official MIMIC-CXR splits by streaming through batch files.

        Args:
            batch_files: List of batch file paths
            total_count: Total number of records

        Returns:
            Dictionary mapping global_record_index -> split_name ('train', 'val', or 'test')
        """
        logger.info("Mapping records to official splits in streaming mode...")

        split_map = {}
        current_idx = 0
        matched_count = 0
        unmatched_count = 0

        # Process each batch file
        for batch_file in tqdm(batch_files, desc="Mapping to official splits"):
            try:
                # Load batch
                if self.config.use_gcs:
                    records = self.gcs_helper.read_torch(batch_file.name)
                else:
                    records = self.gcs_helper.read_torch(str(batch_file))

                # Map each record to its official split
                for record in records:
                    subject_id = record.get('subject_id')
                    study_id = record.get('study_id')

                    if subject_id is not None and study_id is not None:
                        # Look up official split
                        key = (int(subject_id), int(study_id))
                        split_name = self.official_splits_map.get(key)

                        if split_name:
                            split_map[current_idx] = split_name
                            matched_count += 1
                        else:
                            # Record not in official splits - skip it
                            unmatched_count += 1
                            logger.debug(f"Record {current_idx} (subject={subject_id}, study={study_id}) not in official splits")
                    else:
                        # Record missing IDs - skip it
                        unmatched_count += 1
                        logger.warning(f"Record {current_idx} missing subject_id or study_id")

                    current_idx += 1

            except Exception as e:
                logger.error(f"Error processing batch {batch_file}: {e}")
                # Skip this batch and continue
                continue

        # Report statistics
        logger.info("=" * 60)
        logger.info(f"Official split mapping complete:")
        logger.info(f"  Total records processed: {current_idx:,}")
        logger.info(f"  Matched to official splits: {matched_count:,} ({100.0*matched_count/current_idx:.1f}%)")
        logger.info(f"  Unmatched (will be excluded): {unmatched_count:,} ({100.0*unmatched_count/current_idx:.1f}%)")

        # Count by split
        split_counts = defaultdict(int)
        for split_name in split_map.values():
            split_counts[split_name] += 1

        logger.info("\nRecords per split:")
        for split_name in ['train', 'val', 'test']:
            count = split_counts.get(split_name, 0)
            pct = 100.0 * count / matched_count if matched_count > 0 else 0
            logger.info(f"  {split_name:6s}: {count:7,} records ({pct:5.1f}%)")
        logger.info("=" * 60)

        if unmatched_count > 0:
            logger.warning(f"\nNote: {unmatched_count:,} records were not in the official split file")
            logger.warning("These may be from studies not included in MIMIC-CXR-JPG v2.0.0")
            logger.warning("They will be excluded from training/validation/testing")

        return split_map

    def create_stratified_split_indices(self, strat_keys: List[Tuple],
                                       total_count: int) -> Tuple[set, set, set]:
        """
        Create stratified train/val/test split indices.

        NOTE: This is a fallback method. Official MIMIC-CXR splits are STRONGLY RECOMMENDED.

        Args:
            strat_keys: List of stratification keys
            total_count: Total number of records

        Returns:
            train_indices, val_indices, test_indices as sets
        """
        logger.info("Creating custom stratified split indices...")
        logger.warning("REMINDER: Official MIMIC-CXR splits are recommended for reproducibility")

        # Set random seed for reproducibility
        np.random.seed(42)

        # Group indices by stratification key
        strat_groups = defaultdict(list)
        for idx, key in enumerate(strat_keys):
            strat_groups[key].append(idx)

        # Shuffle each group independently
        for key in strat_groups:
            np.random.shuffle(strat_groups[key])

        # Split each group according to train/val/test ratios
        train_indices = set()
        val_indices = set()
        test_indices = set()

        for key, indices in strat_groups.items():
            n_group = len(indices)
            n_train = int(n_group * self.config.train_split)
            n_val = int(n_group * self.config.val_split)

            train_indices.update(indices[:n_train])
            val_indices.update(indices[n_train:n_train + n_val])
            test_indices.update(indices[n_train + n_val:])

        logger.info(f"Split sizes: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
        return train_indices, val_indices, test_indices
    
    def create_splits_from_batches_streaming(self, create_small_samples: bool = False,
                                             small_sample_size: int = 100,
                                             max_batches: Optional[int] = None,
                                             skip_final_combine: bool = False):
        """
        TRUE STREAMING implementation - never accumulates records in memory
        Writes directly to chunk files as records are processed
        
        Args:
            create_small_samples: Whether to create small sample versions
            small_sample_size: Number of records in small samples
            max_batches: Optional limit on number of batches to process (for testing)
            skip_final_combine: If True, keep data as separate chunks (recommended for 4TB dataset)
        """
        import gc
        from dataclasses import dataclass
        
        logger.info("=" * 60)
        logger.info("Creating stratified train/val/test splits (TRUE STREAMING MODE)")
        logger.info("This implementation never accumulates records in memory")
        if max_batches:
            logger.info(f"Testing mode: Processing only first {max_batches} batches")
        logger.info("=" * 60)
        
        # Step 1: Get list of batch files
        batch_files = self.get_batch_file_list(max_batches=max_batches)
        
        if not batch_files:
            logger.error("No intermediate batch files found!")
            return
        
        # Step 2 & 3: Count records and extract stratification keys (existing efficient implementation)
        logger.info("Phase 1: Counting records and extracting stratification keys...")
        total_count, strat_keys = self.count_and_extract_streaming(batch_files)
        
        if total_count == 0:
            logger.error("No records found in intermediate batches!")
            return
        
        logger.info(f"Found {total_count:,} records across {len(batch_files)} batches")

        # Step 4: Create split assignments using official splits or custom stratification
        logger.info("Phase 2: Creating split assignments...")

        if self.official_splits_map:
            # Use official MIMIC-CXR patient-level splits (RECOMMENDED)
            logger.info("Using official MIMIC-CXR patient-level splits")
            logger.info("This ensures proper patient-level splitting (no data leakage)")

            # We need to map records to splits using their subject_id and study_id
            # First, we'll extract these IDs from batch files in a streaming manner
            split_map = self.create_official_split_map_streaming(batch_files, total_count)

            # Free memory
            del strat_keys
            gc.collect()

        else:
            # Fallback to custom stratified splits (NOT RECOMMENDED)
            logger.warning("=" * 60)
            logger.warning("WARNING: Using custom stratified splits")
            logger.warning("Official MIMIC-CXR splits are STRONGLY RECOMMENDED for:")
            logger.warning("  - Reproducibility with published papers")
            logger.warning("  - Proper patient-level splitting (prevents data leakage)")
            logger.warning("  - Fair benchmarking")
            logger.warning("=" * 60)

            train_indices, val_indices, test_indices = self.create_stratified_split_indices(
                strat_keys, total_count
            )

            # Convert to a single lookup dictionary for O(1) access
            split_map = {}
            for idx in train_indices:
                split_map[idx] = 'train'
            for idx in val_indices:
                split_map[idx] = 'val'
            for idx in test_indices:
                split_map[idx] = 'test'

            # Free memory from indices sets
            del train_indices, val_indices, test_indices, strat_keys
            gc.collect()

        logger.info(f"Split assignments created: {len(split_map):,} records mapped")
        
        # Step 5: Initialize streaming writers
        @dataclass
        class StreamWriter:
            """Streaming writer that flushes small batches to disk/GCS"""
            split_name: str
            output_path: str
            gcs_helper: Any
            chunk_size: int = 50  # Flush every 50 records (tunable based on memory)
            
            def __post_init__(self):
                self.records = []
                self.chunk_num = 0
                self.total_written = 0
                self.chunk_files = []  # Track all chunk files created
            
            def write_record(self, record):
                """Add a record and flush if buffer is full"""
                self.records.append(record)
                if len(self.records) >= self.chunk_size:
                    self.flush()
            
            def flush(self):
                """Write current buffer to a chunk file"""
                if not self.records:
                    return
                    
                chunk_filename = f"{self.split_name}_chunk_{self.chunk_num:06d}.pt"
                chunk_path = f"{self.output_path}/{chunk_filename}"
                
                # Write using the GCS helper (handles both GCS and local)
                self.gcs_helper.write_torch(self.records, chunk_path)
                
                self.chunk_files.append(chunk_path)
                self.total_written += len(self.records)
                
                # Clear buffer
                self.records = []
                self.chunk_num += 1
                
                # Aggressive memory cleanup
                gc.collect()
            
            def finalize(self):
                """Flush any remaining records and return statistics"""
                self.flush()
                return {
                    'split': self.split_name,
                    'chunks': self.chunk_num,
                    'records': self.total_written,
                    'files': self.chunk_files
                }
        
        # Create writers for each split
        logger.info("Phase 3: Initializing streaming writers...")
        
        if self.config.use_gcs:
            output_base = self.config.output_path
        else:
            output_base = str(Path(self.config.output_path).expanduser())
        
        writers = {
            'train': StreamWriter('train', output_base, self.gcs_helper, chunk_size=50),
            'val': StreamWriter('val', output_base, self.gcs_helper, chunk_size=50),
            'test': StreamWriter('test', output_base, self.gcs_helper, chunk_size=50)
        }
        
        # For small samples
        small_sample_collectors = {
            'train': [],
            'val': [],
            'test': []
        } if create_small_samples else None
        
        # Step 6: Stream through batches and write directly
        logger.info("Phase 4: Streaming records to split chunk files...")
        logger.info(f"Processing {len(batch_files)} intermediate batches...")
        
        current_global_idx = 0
        processed_batches = 0
        
        # Reset timing stats for this phase
        self.timing_stats = {'load': 0, 'process': 0, 'write': 0, 'total': 0}
        
        # Process batches with progress tracking
        with tqdm(total=len(batch_files), desc="Processing batches") as pbar:
            for batch_idx, batch_file in enumerate(batch_files):
                try:
                    # Load single batch
                    t_load = time.time()
                    if self.config.use_gcs:
                        records = self.gcs_helper.read_torch(batch_file.name)
                    else:
                        records = self.gcs_helper.read_torch(str(batch_file))
                    self.timing_stats['load'] += time.time() - t_load
                    
                    if records is None:
                        logger.warning(f"Batch {batch_idx} returned None, skipping")
                        pbar.update(1)
                        continue
                    
                    # Process each record in the batch
                    t_process = time.time()
                    records_in_batch = 0
                    
                    for record in records:
                        # Determine split assignment
                        split_name = split_map.get(current_global_idx)
                        
                        if split_name:
                            # Write to appropriate split
                            writers[split_name].write_record(record)
                            
                            # Collect for small sample if needed
                            if create_small_samples:
                                if len(small_sample_collectors[split_name]) < small_sample_size:
                                    small_sample_collectors[split_name].append(record)
                        
                        current_global_idx += 1
                        records_in_batch += 1
                    
                    self.timing_stats['process'] += time.time() - t_process
                    
                    # Free batch memory immediately
                    del records
                    gc.collect()
                    
                    processed_batches += 1
                    
                    # Update progress
                    pbar.update(1)
                    pbar.set_postfix({
                        'batch': f"{batch_idx + 1}/{len(batch_files)}",
                        'records': records_in_batch,
                        'load_ms': f"{(self.timing_stats['load']/(batch_idx+1))*1000:.0f}",
                        'proc_ms': f"{(self.timing_stats['process']/(batch_idx+1))*1000:.0f}"
                    })
                    
                    # Periodic status update
                    if (batch_idx + 1) % 100 == 0:
                        logger.info(f"Processed {batch_idx + 1}/{len(batch_files)} batches, "
                                  f"{current_global_idx:,} total records")
                        for split_name, writer in writers.items():
                            logger.info(f"  {split_name}: {writer.total_written:,} records, "
                                      f"{writer.chunk_num} chunks")
                    
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    logger.error(f"Batch file: {batch_file}")
                    # Continue processing other batches
                    pbar.update(1)
                    continue
        
        # Step 7: Finalize all writers
        logger.info("Phase 5: Finalizing writers and flushing remaining records...")
        
        writer_stats = {}
        for split_name, writer in writers.items():
            stats = writer.finalize()
            writer_stats[split_name] = stats
            logger.info(f"{split_name}: {stats['records']:,} records in {stats['chunks']} chunks")
        
        # Step 8: Optionally combine chunks (NOT recommended for 4TB)
        if not skip_final_combine:
            logger.warning("=" * 60)
            logger.warning("WARNING: Combining chunks for 4TB dataset may cause OOM!")
            logger.warning("Consider using --skip-final-combine flag")
            logger.warning("=" * 60)
            
            user_confirm = input("Are you sure you want to combine chunks? (y/N): ")
            if user_confirm.lower() == 'y':
                logger.info("Combining chunks into final files...")
                self._combine_split_chunks_v2('train', writer_stats['train']['files'])
                self._combine_split_chunks_v2('val', writer_stats['val']['files'])
                self._combine_split_chunks_v2('test', writer_stats['test']['files'])
            else:
                logger.info("Skipping chunk combination")
                skip_final_combine = True
        
        # Step 9: Save small samples if requested
        if create_small_samples and small_sample_collectors:
            logger.info(f"Saving small sample datasets ({small_sample_size} records each)...")
            for split_name, samples in small_sample_collectors.items():
                if samples:
                    self._save_small_sample(split_name, samples[:small_sample_size])
                    logger.info(f"  {split_name}_small.pt: {len(samples[:small_sample_size])} records")
        
        # Step 10: Save metadata
        metadata = {
            'config': {k: str(v) for k, v in self.config.__dict__.items()},
            'total_records': current_global_idx,
            'processed_batches': processed_batches,
            'splits': {
                split_name: {
                    'records': stats['records'],
                    'chunks': stats['chunks'],
                    'chunk_files': len(stats['files'])
                }
                for split_name, stats in writer_stats.items()
            },
            'stratified': True,
            'chunked_format': skip_final_combine,
            'small_samples': {
                'created': create_small_samples,
                'size': small_sample_size if create_small_samples else 0
            },
            'processing': {
                'method': 'true_streaming_direct_write',
                'chunk_size_per_writer': 50,
                'memory_efficient': True
            },
            'timing_stats': self.timing_stats
        }
        
        if self.config.use_gcs:
            metadata_file = f"{self.config.output_path}/metadata.json"
        else:
            metadata_file = Path(self.config.output_path) / 'metadata.json'
        
        self.gcs_helper.write_json(metadata, metadata_file)
        
        # Final summary
        logger.info("=" * 60)
        logger.info("TRUE STREAMING SPLIT COMPLETE!")
        logger.info("=" * 60)
        for split_name, stats in writer_stats.items():
            logger.info(f"{split_name:8s}: {stats['records']:8,} records in {stats['chunks']:4} chunks")
        logger.info("=" * 60)
        logger.info(f"Total records processed: {current_global_idx:,}")
        logger.info(f"Output format: {'CHUNKED (recommended)' if skip_final_combine else 'COMBINED'}")
        if create_small_samples:
            logger.info(f"Small samples created: {small_sample_size} records per split")
        logger.info("=" * 60)
        logger.info("✓ Memory-efficient processing completed successfully!")
        logger.info("✓ Data is ready for training (can load chunks directly)")
        logger.info("=" * 60)
    
    def _combine_split_chunks_v2(self, split_name: str, chunk_files: List[str]):
        """
        Optimized chunk combination that processes in smaller groups
        WARNING: Still requires significant memory for final combination
        
        Args:
            split_name: 'train', 'val', or 'test'
            chunk_files: List of chunk file paths to combine
        """
        import tempfile
        
        logger.info(f"Combining {len(chunk_files)} chunks for {split_name}...")
        
        # Process in groups to manage memory
        group_size = 20  # Process 20 chunks at a time
        combined_groups = []
        
        for i in range(0, len(chunk_files), group_size):
            group = chunk_files[i:i + group_size]
            group_records = []
            
            logger.info(f"  Loading chunk group {i//group_size + 1}/{(len(chunk_files) + group_size - 1)//group_size}")
            
            for chunk_file in group:
                records = self.gcs_helper.read_torch(chunk_file)
                group_records.extend(records)
                del records
                gc.collect()
            
            # Save intermediate combined group
            if self.config.use_gcs:
                temp_path = f"{self.config.output_path}/temp_{split_name}_group_{i//group_size}.pt"
            else:
                temp_path = Path(self.config.output_path) / f"temp_{split_name}_group_{i//group_size}.pt"
            
            self.gcs_helper.write_torch(group_records, temp_path)
            combined_groups.append(temp_path)
            
            del group_records
            gc.collect()
        
        # Final combination of groups
        logger.info(f"Final combination of {len(combined_groups)} groups for {split_name}...")
        all_records = []
        
        for group_file in combined_groups:
            records = self.gcs_helper.read_torch(group_file)
            all_records.extend(records)
            del records
            gc.collect()
        
        # Save final file
        if self.config.use_gcs:
            final_path = f"{self.config.output_path}/{split_name}_data.pt"
        else:
            final_path = Path(self.config.output_path) / f"{split_name}_data.pt"
        
        self.gcs_helper.write_torch(all_records, final_path)
        
        # Clean up temporary files
        for temp_file in combined_groups:
            if self.config.use_gcs:
                blob = self.gcs_helper.output_bucket.blob(temp_file)
                blob.delete()
            else:
                Path(temp_file).unlink()
        
        # Clean up original chunks
        logger.info(f"Cleaning up {len(chunk_files)} original chunks...")
        for chunk_file in chunk_files:
            if self.config.use_gcs:
                blob = self.gcs_helper.output_bucket.blob(chunk_file)
                blob.delete()
            else:
                Path(chunk_file).unlink(missing_ok=True)
        
        logger.info(f"Combined {split_name} split: {len(all_records):,} records")
        del all_records
        gc.collect()
    
    def _append_to_split_file(self, split_name: str, records: List[Dict]):
        """
        Append records to a split chunk file with optimized GCS upload
        
        Args:
            split_name: 'train', 'val', or 'test'
            records: Records to append
        """
        import time
        import random
        
        # Generate chunk filename with timestamp and random suffix
        timestamp = int(time.time())
        rand_suffix = random.randint(1000, 9999)
        chunk_filename = f"{split_name}_chunk_{timestamp}_{rand_suffix}.pt"
        
        if self.config.use_gcs:
            chunk_path = f"{self.config.output_path}/{chunk_filename}"
        else:
            output_dir = Path(self.config.output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            chunk_path = output_dir / chunk_filename
        
        self.gcs_helper.write_torch(records, chunk_path)
    
    def _combine_split_chunks(self, split_name: str):
        """
        Combine all chunk files for a split into a single final file.
        
        IMPORTANT: This loads all data into memory. For very large datasets (>10GB),
        consider keeping chunks separate or using a database format.
        
        Args:
            split_name: 'train', 'val', or 'test'
        """
        import gc
        
        logger.info(f"Combining {split_name} chunks...")
        
        # Find all chunk files
        if self.config.use_gcs:
            from google.cloud import storage
            prefix = f"{self.config.output_path}/{split_name}_chunk_"
            
            client = storage.Client(project=self.config.gcs_project_id)
            bucket = client.bucket(self.config.output_gcs_bucket)
            chunk_blobs = list(bucket.list_blobs(prefix=prefix))
            chunk_blobs.sort(key=lambda b: b.name)
            
            if not chunk_blobs:
                logger.warning(f"No {split_name} chunks found to combine")
                return
            
            logger.info(f"Found {len(chunk_blobs)} chunks for {split_name}")
        else:
            output_dir = Path(self.config.output_path)
            chunk_files = sorted(output_dir.glob(f"{split_name}_chunk_*.pt"))
            
            if not chunk_files:
                logger.warning(f"No {split_name} chunks found to combine")
                return
            
            logger.info(f"Found {len(chunk_files)} chunks for {split_name}")
        
        # Strategy: Load chunks in groups to manage memory
        # Process in groups of 10 chunks at a time
        chunk_group_size = 10
        all_records = []
        total_count = 0
        
        if self.config.use_gcs:
            chunk_list = chunk_blobs
            for i in range(0, len(chunk_list), chunk_group_size):
                group = chunk_list[i:i + chunk_group_size]
                logger.info(f"Loading chunk group {i//chunk_group_size + 1}/{(len(chunk_list) + chunk_group_size - 1)//chunk_group_size}")
                
                for blob in tqdm(group, desc=f"Loading chunks", leave=False):
                    records = self.gcs_helper.read_torch(blob.name)
                    all_records.extend(records)
                    total_count += len(records)
                    del records
                    gc.collect()
            
            # Delete chunk files after combining
            logger.info(f"Deleting {len(chunk_blobs)} chunk files from GCS...")
            for blob in chunk_blobs:
                blob.delete()
        else:
            chunk_list = chunk_files
            for i in range(0, len(chunk_list), chunk_group_size):
                group = chunk_list[i:i + chunk_group_size]
                logger.info(f"Loading chunk group {i//chunk_group_size + 1}/{(len(chunk_list) + chunk_group_size - 1)//chunk_group_size}")
                
                for chunk_file in tqdm(group, desc=f"Loading chunks", leave=False):
                    records = self.gcs_helper.read_torch(str(chunk_file))
                    all_records.extend(records)
                    total_count += len(records)
                    del records
                    gc.collect()
            
            # Delete chunk files after combining
            logger.info(f"Deleting {len(chunk_files)} chunk files...")
            for chunk_file in chunk_files:
                chunk_file.unlink()
        
        # Save final combined file using torch.save (more efficient than pickle)
        logger.info(f"Saving final {split_name} file with {total_count:,} records...")
        if self.config.use_gcs:
            output_file = f"{self.config.output_path}/{split_name}_data.pt"
            self.gcs_helper.write_torch(all_records, output_file)
            logger.info(f"Uploaded {split_name} split to GCS")
        else:
            output_file = Path(self.config.output_path) / f"{split_name}_data.pt"
            self.gcs_helper.write_torch(all_records, str(output_file))
            logger.info(f"Saved {split_name} split locally")
        
        logger.info(f"Completed {split_name} split with {total_count:,} records")
        del all_records
        gc.collect()
    
    def _count_chunk_files(self, split_name: str) -> int:
        """
        Count the number of chunk files for a given split.
        
        Args:
            split_name: 'train', 'val', or 'test'
            
        Returns:
            Number of chunk files found
        """
        if self.config.use_gcs:
            # Use output_bucket directly
            bucket = self.gcs_helper.output_bucket
            blobs = list(bucket.list_blobs(prefix=self.config.output_path))
            chunk_blobs = [b for b in blobs if b.name.endswith('.pt') and f'{split_name}_chunk_' in b.name]
            return len(chunk_blobs)
        else:
            output_dir = Path(self.config.output_path)
            chunk_files = list(output_dir.glob(f"{split_name}_chunk_*.pt"))
            return len(chunk_files)
    
    def _save_small_sample(self, split_name: str, records: List[Dict]):
        """Save a small sample dataset for testing"""
        if not records:
            logger.warning(f"No records to save for {split_name} small sample")
            return
        
        if self.config.use_gcs:
            output_file = f"{self.config.output_path}/{split_name}_small.pt"
        else:
            output_file = Path(self.config.output_path) / f"{split_name}_small.pt"
        
        self.gcs_helper.write_torch(records, output_file)
        logger.info(f"Saved {split_name} small sample with {len(records)} records")


class ChunkedMIMICDataset:
    """
    Efficient dataset that loads from multiple chunk files without combining them
    Perfect for the 4TB dataset split across thousands of chunks
    """
    
    def __init__(self, chunk_files: List[str], gcs_helper=None, cache_size: int = 2):
        """
        Args:
            chunk_files: List of chunk file paths
            gcs_helper: GCSHelper instance for cloud storage
            cache_size: Number of chunks to keep in memory cache
        """
        self.chunk_files = chunk_files
        self.gcs_helper = gcs_helper
        self.cache_size = cache_size
        
        # Build index: which chunk and position for each record
        self.index = []
        self._build_index()
        
        # LRU cache for loaded chunks
        self.chunk_cache = {}
        self.cache_order = []
    
    def _build_index(self):
        """Build index mapping global position to (chunk_id, position_in_chunk)"""
        logger.info(f"Building index for {len(self.chunk_files)} chunk files...")
        
        for chunk_id, chunk_file in enumerate(tqdm(self.chunk_files, desc="Indexing")):
            # Load chunk to get size (lightweight - just length)
            chunk_data = self._load_chunk(chunk_id)
            for pos in range(len(chunk_data)):
                self.index.append((chunk_id, pos))
            
            # Don't cache during indexing
            if chunk_id in self.chunk_cache:
                del self.chunk_cache[chunk_id]
        
        logger.info(f"Index built: {len(self.index)} total records")
    
    def _load_chunk(self, chunk_id: int):
        """Load a chunk with caching"""
        if chunk_id in self.chunk_cache:
            # Move to end (most recently used)
            self.cache_order.remove(chunk_id)
            self.cache_order.append(chunk_id)
            return self.chunk_cache[chunk_id]
        
        # Load chunk
        chunk_file = self.chunk_files[chunk_id]
        if self.gcs_helper:
            chunk_data = self.gcs_helper.read_torch(chunk_file)
        else:
            chunk_data = torch.load(chunk_file, map_location='cpu', weights_only=False)
        
        # Add to cache
        self.chunk_cache[chunk_id] = chunk_data
        self.cache_order.append(chunk_id)
        
        # Evict oldest if cache is full
        if len(self.chunk_cache) > self.cache_size:
            oldest = self.cache_order.pop(0)
            del self.chunk_cache[oldest]
        
        return chunk_data
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        chunk_id, position = self.index[idx]
        chunk_data = self._load_chunk(chunk_id)
        return chunk_data[position]


def load_chunked_dataset(output_path: str, split: str, config: DataConfig = None):
    """
    Helper to load a chunked dataset for training
    
    Args:
        output_path: Base output directory
        split: 'train', 'val', or 'test'
        config: DataConfig instance for GCS settings
        
    Returns:
        ChunkedMIMICDataset instance ready for PyTorch DataLoader
    """
    # Load metadata
    if config and config.use_gcs:
        from google.cloud import storage
        metadata_path = f"{output_path}/metadata.json"
        gcs_helper = GCSHelper(config)
        metadata = gcs_helper.read_json(metadata_path)
        
        # Find chunk files
        client = storage.Client(project=config.gcs_project_id)
        bucket = client.bucket(config.output_gcs_bucket)
        prefix = f"{output_path}/{split}_chunk_"
        chunk_blobs = list(bucket.list_blobs(prefix=prefix))
        chunk_files = [blob.name for blob in chunk_blobs]
        chunk_files.sort()  # Ensure consistent ordering
        
        if not chunk_files:
            raise ValueError(f"No chunk files found for {split} split in {output_path}")
        
        dataset = ChunkedMIMICDataset(chunk_files, gcs_helper, cache_size=2)
    else:
        metadata_path = Path(output_path) / 'metadata.json'
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        # Find chunk files
        output_dir = Path(output_path)
        chunk_files = sorted(output_dir.glob(f"{split}_chunk_*.pt"))
        chunk_files = [str(f) for f in chunk_files]
        
        if not chunk_files:
            raise ValueError(f"No chunk files found for {split} split in {output_path}")
        
        dataset = ChunkedMIMICDataset(chunk_files, cache_size=2)
    
    logger.info(f"Loaded {split} dataset: {len(dataset)} records from {len(chunk_files)} chunks")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="MIMIC Enhanced MDF-Net Data Preprocessing")
    
    # Data paths
    parser.add_argument(
        '--mimic-cxr-path',
        type=str,
        default='',  # Empty for GCS root-level access
        help='Path to MIMIC-CXR dataset (local or GCS prefix, empty for root)'
    )
    
    parser.add_argument(
        '--mimic-iv-path',
        type=str,
        default='physionet.org/files/mimiciv/3.1',
        help='Path to MIMIC-IV dataset (local or GCS prefix)'
    )
    
    parser.add_argument(
        '--mimic-ed-path',
        type=str,
        default='physionet.org/files/mimic-iv-ed/2.2',
        help='Path to MIMIC-IV-ED dataset (local or S3 prefix) - separate from MIMIC-IV'
    )
    
    parser.add_argument(
        '--reflacx-path',
        type=str,
        default='physionet.org/files/reflacx',
        help='Path to REFLACX dataset (local or S3 prefix)'
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        default='processed/phase1_preprocess',
        help='Output path for processed data (local or S3 prefix)'
    )
    
    parser.add_argument(
        '--knowledge-base-path',
        type=str,
        default='medical_knowledge',
        help='Path to medical knowledge base (local or S3 prefix)'
    )
    
    # Google Cloud Storage settings
    parser.add_argument(
        '--gcs-bucket',
        type=str,
        default=None,
        help='GCS bucket name for reading MIMIC-IV, MIMIC-IV-ED, REFLACX (enables GCS mode). Example: bergermimiciv'
    )
    
    parser.add_argument(
        '--gcs-cxr-bucket',
        type=str,
        default=None,
        help='GCS bucket name for MIMIC-CXR images. Example: mimic-cxr-jpg-2.1.0.physionet.org (PhysioNet public bucket)'
    )
    
    parser.add_argument(
        '--output-gcs-bucket',
        type=str,
        default=None,
        help='GCS bucket name for writing output data (if different from input bucket)'
    )
    
    parser.add_argument(
        '--gcs-project-id',
        type=str,
        default=None,
        help='GCP project ID for requester pays buckets (required for PhysioNet public bucket)'
    )
    
    # Processing settings
    parser.add_argument(
        '--image-size',
        type=int,
        default=518,
        help='Image size for BiomedCLIP-CXR (default: 518)'
    )
    
    parser.add_argument(
        '--max-text-length',
        type=int,
        default=8192,
        help='Maximum text length for Clinical ModernBERT (default: 8192)'
    )
    
    parser.add_argument(
        '--top-k-retrieval',
        type=int,
        default=5,
        help='Number of knowledge documents to retrieve (default: 5)'
    )
    
    # Performance optimization
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Number of records to process in each batch for parallel downloading (default: 100)'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of parallel workers for image downloading (default: 4)'
    )
    
    parser.add_argument(
        '--skip-to-combine',
        action='store_true',
        help='Skip batch processing and go directly to combining intermediate batches into final splits'
    )
    
    parser.add_argument(
        '--create-small-samples',
        action='store_true',
        help='Create small sample versions of train/val/test for testing purposes (e.g., train_small.pkl)'
    )
    
    parser.add_argument(
        '--small-sample-size',
        type=int,
        default=100,
        help='Number of records in small sample datasets (default: 100)'
    )
    
    parser.add_argument(
        '--max-batches',
        type=int,
        default=None,
        help='Maximum number of intermediate batches to process (for testing). None = process all batches'
    )
    
    parser.add_argument(
        '--skip-final-combine',
        action='store_true',
        help='Skip combining chunk files into single train/val/test files. Keep data as separate chunks (recommended for large datasets >1TB)'
    )
    
    # Data splits
    parser.add_argument(
        '--train-split',
        type=float,
        default=0.7,
        help='Training split ratio (default: 0.7)'
    )
    
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.15,
        help='Validation split ratio (default: 0.15)'
    )
    
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.15,
        help='Test split ratio (default: 0.15)'
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = DataConfig()
    
    # Update paths
    config.mimic_cxr_path = args.mimic_cxr_path
    config.mimic_iv_path = args.mimic_iv_path
    config.mimic_ed_path = args.mimic_ed_path
    config.reflacx_path = args.reflacx_path
    config.output_path = args.output_path
    config.knowledge_base_path = args.knowledge_base_path
    
    # Update GCS settings
    config.use_gcs = args.gcs_bucket is not None
    config.gcs_bucket = args.gcs_bucket
    config.gcs_cxr_bucket = args.gcs_cxr_bucket
    config.output_gcs_bucket = args.output_gcs_bucket if args.output_gcs_bucket else args.gcs_bucket
    config.gcs_project_id = args.gcs_project_id
    
    # For GCS with PhysioNet bucket, set mimic_cxr_path to empty (files are at bucket root)
    if config.use_gcs and config.gcs_cxr_bucket:
        config.mimic_cxr_path = ""  # Files are at bucket root in PhysioNet bucket
        
        # Warn if project ID not provided for requester pays
        if not config.gcs_project_id:
            logger.warning("PhysioNet bucket requires --gcs-project-id for requester pays access")
    
    # Update processing settings
    config.image_size = args.image_size
    config.max_text_length = args.max_text_length
    config.top_k_retrieval = args.top_k_retrieval
    
    # Update data splits
    config.train_split = args.train_split
    config.val_split = args.val_split
    config.test_split = args.test_split
    
    logger.info("=" * 60)
    logger.info("MIMIC Enhanced MDF-Net Data Preprocessing Pipeline")
    logger.info("=" * 60)
    logger.info(f"Mode: {'GCS' if config.use_gcs else 'Local'}")
    if config.use_gcs:
        logger.info(f"Main bucket: {config.gcs_bucket}")
        logger.info(f"CXR bucket: {config.gcs_cxr_bucket}")
        logger.info(f"Output bucket: {config.output_gcs_bucket}")
    logger.info(f"MIMIC-CXR path: {config.mimic_cxr_path}")
    logger.info(f"MIMIC-IV path: {config.mimic_iv_path}")
    logger.info(f"MIMIC-ED path: {config.mimic_ed_path}")
    logger.info(f"Output path: {config.output_path}")
    logger.info("=" * 60)
    
    # Create dataset
    dataset_creator = DatasetCreator(config)
    
    if args.skip_to_combine:
        # Skip batch processing, go directly to combining and splitting
        logger.info("Skipping batch processing, combining intermediate batches directly...")
        dataset_creator.create_splits_from_batches_streaming(
            create_small_samples=args.create_small_samples,
            small_sample_size=args.small_sample_size,
            max_batches=args.max_batches,
            skip_final_combine=args.skip_final_combine
        )
    else:
        # Full pipeline: batch processing + combining + splitting
        dataset_creator.create_dataset(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            create_small_samples=args.create_small_samples,
            small_sample_size=args.small_sample_size
        )
    
    logger.info("=" * 60)
    logger.info("Processing complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()