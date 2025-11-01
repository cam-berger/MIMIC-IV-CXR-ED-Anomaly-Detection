"""
Phase 1: Data Preprocessing Pipeline for Enhanced MDF-Net
Adapts MDF-Net's data processing for use with:
- BiomedCLIP-CXR vision encoder
- Clinical ModernBERT text encoder
- RAG knowledge enhancement
- Cross-attention fusion

Google Cloud Storage Support: Reads from and writes to GCS buckets for cloud deployment
Supports multiple buckets (your data + PhysioNet's public MIMIC-CXR bucket)
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

# Google Cloud Storage
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    storage = None

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

    # Data splits (following MDF-Net)
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
    """Helper class for Google Cloud Storage operations with multi-bucket support"""

    def __init__(self, config: DataConfig):
        self.config = config
        self.gcs_client = None
        self.bucket = None
        self.cxr_bucket = None
        self.output_bucket = None

        if config.use_gcs:
            if not GCS_AVAILABLE:
                raise ImportError("google-cloud-storage is not installed. Run: pip install google-cloud-storage")

            # Initialize client with project for requester pays
            if config.gcs_project_id:
                self.gcs_client = storage.Client(project=config.gcs_project_id)
            else:
                self.gcs_client = storage.Client()

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
            blob.upload_from_file(buffer, rewind=True)
            logger.info(f"Saved to GCS: gs://{self.output_bucket.name}/{gcs_key}")
        else:
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved locally: {path}")

    def write_torch(self, data, path):
        """
        Write data using torch.save() - much more memory efficient for PyTorch tensors

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

            # Upload to GCS
            gcs_key = path if isinstance(path, str) else str(path)
            blob = self.output_bucket.blob(gcs_key)
            blob.upload_from_filename(tmp_path)

            # Clean up temp file
            Path(tmp_path).unlink()
            logger.info(f"Saved to GCS using torch.save: gs://{self.output_bucket.name}/{gcs_key}")
        else:
            # Save locally using torch.save
            torch.save(data, path)
            logger.info(f"Saved locally using torch.save: {path}")

    def read_torch(self, path):
        """
        Read data using torch.load() - companion to write_torch()

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
                blob.download_to_filename(tmp_path)

            # Load from temp file
            # Use weights_only=False to allow loading numpy arrays and custom objects
            # This is safe since we trust our own checkpoint files
            data = torch.load(tmp_path, map_location='cpu', weights_only=False)

            # Clean up temp file
            Path(tmp_path).unlink()
            logger.info(f"Loaded from GCS using torch.load: gs://{self.output_bucket.name}/{gcs_key}")
            return data
        else:
            # Load locally using torch.load
            # Use weights_only=False to allow loading numpy arrays and custom objects
            # This is safe since we trust our own checkpoint files
            data = torch.load(path, map_location='cpu', weights_only=False)
            logger.info(f"Loaded locally using torch.load: {path}")
            return data

    def write_json(self, data, path):
        """Write JSON file to GCS or local"""
        if self.output_bucket:
            gcs_key = path if isinstance(path, str) else str(path)
            json_str = json.dumps(data, indent=2, default=str)
            blob = self.output_bucket.blob(gcs_key)
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
                metadata_path = f"{self.config.mimic_cxr_path}/mimic-cxr-2.0.0-metadata.csv.gz"
            else:
                metadata_path = "mimic-cxr-2.0.0-metadata.csv.gz"
        else:
            metadata_path = Path(self.config.mimic_cxr_path) / "mimic-cxr-2.0.0-metadata.csv"

        logger.info("Loading MIMIC-CXR metadata...")
        metadata = self.gcs_helper.read_csv(metadata_path, compression='gzip' if str(metadata_path).endswith('.gz') else None)

        # Filter for frontal views (AP/PA) like MDF-Net
        frontal_views = metadata[metadata['ViewPosition'].isin(['AP', 'PA'])]
        logger.info(f"Found {len(frontal_views)} frontal view images")

        return frontal_views
    
    def load_mimic_iv_ed(self) -> Dict[str, pd.DataFrame]:
        """Load MIMIC-IV ED data"""
        logger.info("Loading MIMIC-IV ED data...")

        ed_data = {}

        # MIMIC-IV-ED is a separate dataset from MIMIC-IV
        # Use mimic_ed_path if available, otherwise fallback to mimic_iv_path/ed
        if hasattr(self.config, 'mimic_ed_path') and self.config.mimic_ed_path:
            base_ed_path = self.config.mimic_ed_path
        else:
            base_ed_path = self.config.mimic_iv_path

        # Load all ED tables
        tables = ['edstays', 'triage', 'vitalsign', 'pyxis', 'medrecon', 'diagnosis']
        for table in tables:
            if self.config.use_gcs:
                # Try .csv.gz first, then .csv
                file_path_gz = f"{base_ed_path}/ed/{table}.csv.gz"
                file_path_csv = f"{base_ed_path}/ed/{table}.csv"

                if self.gcs_helper.path_exists(file_path_gz):
                    file_path = file_path_gz
                    compression = 'gzip'
                elif self.gcs_helper.path_exists(file_path_csv):
                    file_path = file_path_csv
                    compression = None
                else:
                    logger.warning(f"Could not find {table} at {file_path_gz} or {file_path_csv}")
                    continue
            else:
                file_path = Path(base_ed_path) / "ed" / f"{table}.csv"
                compression = None

                if not self.gcs_helper.path_exists(file_path):
                    logger.warning(f"Could not find {table} at {file_path}")
                    continue

            ed_data[table] = self.gcs_helper.read_csv(file_path, compression=compression)
            logger.info(f"Loaded {table}: {len(ed_data[table])} records")

        return ed_data
    
    def create_pseudo_notes(self, clinical_data: Dict) -> str:
        """
        Convert structured clinical data to pseudo-notes
        Following MDF-Net's approach but enhanced for LLM processing
        """
        pseudo_note_parts = []
        
        # Patient demographics
        if 'age' in clinical_data:
            pseudo_note_parts.append(f"Patient is a {clinical_data['age']} year old {clinical_data.get('gender', 'patient')}.")
        
        # Chief complaint
        if 'chiefcomplaint' in clinical_data:
            pseudo_note_parts.append(f"Presenting with: {clinical_data['chiefcomplaint']}.")
        
        # Vital signs (MDF-Net style)
        vitals = []
        vital_mappings = {
            'temperature': ('T', '°F'),
            'heartrate': ('HR', 'bpm'),
            'resprate': ('RR', '/min'),
            'o2sat': ('O2', '%'),
            'sbp': ('SBP', 'mmHg'),
            'dbp': ('DBP', 'mmHg')
        }
        
        for vital, (abbrev, unit) in vital_mappings.items():
            if vital in clinical_data and clinical_data[vital] is not None:
                vitals.append(f"{abbrev}: {clinical_data[vital]}{unit}")
        
        if vitals:
            pseudo_note_parts.append(f"Vitals: {', '.join(vitals)}.")
        
        # Medications
        if 'medications' in clinical_data and clinical_data['medications']:
            meds = clinical_data['medications']
            if isinstance(meds, list):
                pseudo_note_parts.append(f"Current medications: {', '.join(meds[:5])}.")
        
        # Lab results (if available)
        if 'lab_results' in clinical_data and clinical_data['lab_results']:
            pseudo_note_parts.append(f"Notable labs: {clinical_data['lab_results']}.")
        
        # Medical history
        if 'medical_history' in clinical_data and clinical_data['medical_history']:
            pseudo_note_parts.append(f"PMH: {clinical_data['medical_history']}.")
        
        return " ".join(pseudo_note_parts)
    
    def join_multimodal_data(self) -> pd.DataFrame:
        """
        Main joining function to create multimodal dataset
        Enhanced from MDF-Net to include more clinical context
        """
        # Load data
        cxr_metadata = self.load_mimic_cxr_metadata()
        ed_data = self.load_mimic_iv_ed()

        # Join on subject_id and time windows
        logger.info("Joining CXR images with ED data...")

        joined_data = []
        no_ed_match_count = 0
        no_image_found_count = 0
        successful_joins = 0

        for idx, cxr_row in tqdm(cxr_metadata.iterrows(), total=len(cxr_metadata)):
            subject_id = cxr_row['subject_id']

            # Combine StudyDate and StudyTime into a single datetime
            # StudyDate format: YYYYMMDD, StudyTime format: HHMMSS or HHMMSS.fff (with fractional seconds)
            study_date_str = str(cxr_row['StudyDate'])
            study_time_raw = str(cxr_row.get('StudyTime', '000000'))
            # Remove fractional seconds if present (e.g., "083045.531" -> "083045")
            study_time_str = study_time_raw.split('.')[0].zfill(6)
            study_datetime_str = f"{study_date_str} {study_time_str}"
            study_time = pd.to_datetime(study_datetime_str, format='%Y%m%d %H%M%S')

            # Find matching ED stay within 24 hours
            match_found = False
            if 'edstays' in ed_data:
                ed_stays = ed_data['edstays'][ed_data['edstays']['subject_id'] == subject_id]

                if ed_stays.empty:
                    no_ed_match_count += 1
                    continue

                for _, ed_row in ed_stays.iterrows():
                    ed_time = pd.to_datetime(ed_row['intime'])
                    time_diff = abs((study_time - ed_time).total_seconds() / 3600)

                    if time_diff <= 24:  # Within 24 hours
                        # Get image path and verify it exists before adding to joined data
                        image_path = self._get_image_path(cxr_row)

                        # Verify image exists before adding to joined data
                        if not self.gcs_helper.path_exists(image_path):
                            no_image_found_count += 1
                            logger.debug(f"Image not found: {image_path} (subject_id={subject_id})")
                            match_found = True  # Had ED match but no image
                            break

                        # Collect all clinical data
                        clinical_data = self._collect_clinical_data(subject_id, ed_row, ed_data)

                        # Create pseudo-note
                        pseudo_note = self.create_pseudo_notes(clinical_data)

                        joined_record = {
                            'subject_id': subject_id,
                            'study_id': cxr_row['study_id'],
                            'dicom_id': cxr_row['dicom_id'],
                            'image_path': image_path,
                            'pseudo_note': pseudo_note,
                            'clinical_data': json.dumps(clinical_data),
                            'ed_stay_id': ed_row['stay_id'],
                            'study_time': study_time,
                            'ed_time': ed_time
                        }

                        joined_data.append(joined_record)
                        successful_joins += 1
                        match_found = True
                        break

            if not match_found and 'edstays' in ed_data:
                no_ed_match_count += 1

        # Log detailed statistics
        logger.info("=" * 60)
        logger.info("Multimodal Data Joining Statistics:")
        logger.info(f"  Total CXR records processed: {len(cxr_metadata):,}")
        logger.info(f"  Successful joins (with existing images): {successful_joins:,}")
        logger.info(f"  No ED match found (no ED visit within 24hrs): {no_ed_match_count:,}")
        logger.info(f"  Image file not found in storage: {no_image_found_count:,}")
        logger.info(f"  Success rate: {successful_joins/len(cxr_metadata)*100:.2f}%")
        logger.info("=" * 60)

        result_df = pd.DataFrame(joined_data)

        return result_df
    
    def _collect_clinical_data(self, subject_id: int, ed_row: pd.Series, 
                               ed_data: Dict[str, pd.DataFrame]) -> Dict:
        """Collect all clinical data for a patient"""
        clinical_data = {
            'subject_id': subject_id,
            'age': ed_row.get('age', None),
            'gender': ed_row.get('gender', None)
        }
        
        # Get triage data
        if 'triage' in ed_data:
            triage = ed_data['triage'][ed_data['triage']['stay_id'] == ed_row['stay_id']]
            if not triage.empty:
                triage_row = triage.iloc[0]
                for vital in self.config.clinical_features:
                    if vital in triage_row:
                        clinical_data[vital] = triage_row[vital]
        
        # Get vital signs
        if 'vitalsign' in ed_data:
            vitals = ed_data['vitalsign'][ed_data['vitalsign']['stay_id'] == ed_row['stay_id']]
            if not vitals.empty:
                # Get most recent vitals
                for vital in ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp']:
                    if vital in vitals.columns:
                        recent_vital = vitals[vital].dropna().iloc[-1] if not vitals[vital].dropna().empty else None
                        clinical_data[vital] = recent_vital
        
        return clinical_data
    
    def _get_image_path(self, cxr_row: pd.Series) -> str:
        """
        Get the full path to the image file

        MIMIC-CXR uses 8-digit padded IDs for directory structure:
        - subject_id 1234 -> 00001234 -> p10/p00001234
        - subject_id 10000032 -> 10000032 -> p10/p10000032
        """
        # MIMIC-CXR uses 8-digit padded subject IDs and study IDs
        subject_id_padded = str(cxr_row['subject_id']).zfill(8)
        study_id_padded = str(cxr_row['study_id']).zfill(8)

        # Directory structure: first 2 digits of padded ID determine patient folder
        patient_folder = f"p{subject_id_padded[:2]}"
        subject_folder = f"p{subject_id_padded}"
        study_folder = f"s{study_id_padded}"

        if self.config.use_gcs:
            # For GCS, construct path from root (files are at bucket root)
            if self.config.mimic_cxr_path:
                image_path = f"{self.config.mimic_cxr_path}/files/{patient_folder}/{subject_folder}/{study_folder}/{cxr_row['dicom_id']}.jpg"
            else:
                image_path = f"files/{patient_folder}/{subject_folder}/{study_folder}/{cxr_row['dicom_id']}.jpg"
        else:
            # For local filesystem
            image_path = Path(self.config.mimic_cxr_path) / "files" / patient_folder / \
                         subject_folder / study_folder / f"{cxr_row['dicom_id']}.jpg"
            image_path = str(image_path)

        return image_path


class ImagePreprocessor:
    """
    Preprocess images for BiomedCLIP-CXR
    Enhanced from MDF-Net's basic preprocessing
    Optimized with caching to avoid duplicate downloads
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.gcs_helper = GCSHelper(config)
        self.transform = self._create_transform()
        self._image_cache = {}  # Cache for downloaded images

    def _create_transform(self):
        """Create transformation pipeline for BiomedCLIP-CXR"""
        return transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.normalize_mean,
                               std=self.config.normalize_std)
        ])

    def _download_and_cache_image(self, image_path: str) -> Optional[Tuple[Image.Image, np.ndarray]]:
        """
        Download image once and return both PIL and cv2 versions
        Caches result to avoid duplicate downloads
        """
        # Check cache first
        if image_path in self._image_cache:
            return self._image_cache[image_path]

        try:
            # First check if file exists (already done in calling code, but double-check)
            if not self.gcs_helper.path_exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return None

            # Download once from GCS
            if self.config.use_gcs:
                bucket = self.gcs_helper._get_bucket_for_path(image_path)
                blob = bucket.blob(image_path)

                # Verify blob exists
                if not blob.exists():
                    logger.error(f"GCS blob does not exist: gs://{bucket.name}/{image_path}")
                    return None

                # Download data
                data = blob.download_as_bytes()

                # Validate downloaded data
                if not data or len(data) == 0:
                    logger.error(f"Empty data downloaded for: {image_path}")
                    return None

                # Create PIL image
                pil_image = Image.open(BytesIO(data)).convert('RGB')

                # Create cv2 image (grayscale)
                img_array = np.frombuffer(data, np.uint8)
                cv2_image = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

                # Validate decoded images
                if cv2_image is None:
                    logger.error(f"Failed to decode image with cv2: {image_path}")
                    return None
            else:
                # Local filesystem
                pil_image = Image.open(image_path).convert('RGB')
                cv2_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                # Validate loaded images
                if cv2_image is None:
                    logger.error(f"Failed to load image with cv2: {image_path}")
                    return None

            # Cache the result
            result = (pil_image, cv2_image)
            self._image_cache[image_path] = result

            # Clear cache if it gets too large (keep last 50 images)
            # Each preprocessed image with embeddings uses ~60MB RAM
            # 50 images = ~3GB, safe for 14GB VM
            if len(self._image_cache) > 50:
                # Remove oldest entries (FIFO)
                keys_to_remove = list(self._image_cache.keys())[:-50]
                for key in keys_to_remove:
                    del self._image_cache[key]

            return result

        except Exception as e:
            logger.error(f"Error downloading/processing image {image_path}: {e}")
            return None

    def preprocess_image_and_attention(self, image_path: str) -> Tuple[Optional[torch.Tensor], Optional[np.ndarray]]:
        """
        Preprocess image and extract attention regions in one pass
        Downloads the image only once
        Returns: (image_tensor, attention_mask)
        """
        # Download image once
        result = self._download_and_cache_image(image_path)
        if result is None:
            return None, None

        pil_image, cv2_image = result

        # Process PIL image for BiomedCLIP-CXR
        try:
            image_tensor = self.transform(pil_image)
        except Exception as e:
            logger.error(f"Error transforming image {image_path}: {e}")
            image_tensor = None

        # Process cv2 image for attention regions
        try:
            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(cv2_image)

            # Edge detection to find potential abnormal regions
            edges = cv2.Canny(enhanced, 50, 150)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create attention mask
            attention_mask = np.zeros_like(cv2_image, dtype=np.float32)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Filter small regions
                    cv2.drawContours(attention_mask, [contour], -1, 1.0, -1)
        except Exception as e:
            logger.warning(f"Error extracting attention regions from {image_path}: {e}. Returning zero mask.")
            try:
                attention_mask = np.zeros((self.config.image_size, self.config.image_size), dtype=np.float32)
            except:
                attention_mask = None

        return image_tensor, attention_mask

    def preprocess_image(self, image_path: str) -> Optional[torch.Tensor]:
        """Preprocess a single image (backward compatibility)"""
        image_tensor, _ = self.preprocess_image_and_attention(image_path)
        return image_tensor

    def extract_attention_regions(self, image_path: str) -> Optional[np.ndarray]:
        """Extract attention regions (backward compatibility)"""
        _, attention_mask = self.preprocess_image_and_attention(image_path)
        return attention_mask


class TextPreprocessor:
    """
    Preprocess clinical text for Clinical ModernBERT
    Enhanced with medical terminology handling
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        # Initialize Clinical ModernBERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            "answerdotai/ModernBERT-base",  # Will be replaced with Clinical version
            max_length=config.max_text_length
        )
        self.medical_abbreviations = self._load_medical_abbreviations()
        
    def _load_medical_abbreviations(self) -> Dict[str, str]:
        """Load common medical abbreviations"""
        return {
            'hx': 'history',
            'pt': 'patient',
            'yo': 'year old',
            'hr': 'heart rate',
            'bp': 'blood pressure',
            'rr': 'respiratory rate',
            'o2': 'oxygen',
            'sob': 'shortness of breath',
            'cp': 'chest pain',
            'cxr': 'chest x-ray',
            'ed': 'emergency department',
            'pmh': 'past medical history',
            'nkda': 'no known drug allergies'
        }
    
    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Preprocess clinical text"""
        # Clean and normalize text
        text = self.clean_clinical_text(text)
        
        # Expand medical abbreviations
        text = self.expand_abbreviations(text)
        
        # Tokenize for Clinical ModernBERT
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.config.max_text_length,
            return_tensors='pt'
        )
        
        return encoded
    
    def clean_clinical_text(self, text: str) -> str:
        """Clean clinical text"""
        # Remove de-identification tokens
        text = re.sub(r'\[\*\*.*?\*\*\]', '', text)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Convert to lowercase for consistency
        text = text.lower()
        
        return text
    
    def expand_abbreviations(self, text: str) -> str:
        """Expand medical abbreviations"""
        words = text.split()
        expanded = []
        
        for word in words:
            if word in self.medical_abbreviations:
                expanded.append(self.medical_abbreviations[word])
            else:
                expanded.append(word)
        
        return ' '.join(expanded)
    
    def extract_medical_entities(self, text: str) -> List[str]:
        """
        Extract medical entities for RAG queries
        Simple rule-based extraction (can be enhanced with NER)
        """
        entities = []
        
        # Extract vital sign values
        vital_patterns = [
            r'temperature[:\s]+(\d+\.?\d*)',
            r'heart rate[:\s]+(\d+)',
            r'blood pressure[:\s]+(\d+/\d+)',
            r'oxygen[:\s]+(\d+)%?'
        ]
        
        for pattern in vital_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.extend(matches)
        
        # Extract symptoms
        symptom_keywords = ['pain', 'fever', 'cough', 'dyspnea', 'fatigue', 
                           'nausea', 'vomiting', 'diarrhea', 'headache']
        for symptom in symptom_keywords:
            if symptom in text.lower():
                entities.append(symptom)
        
        return entities


class RAGKnowledgeEnhancer:
    """
    Enhance pseudo-notes with retrieved medical knowledge
    Novel addition beyond MDF-Net
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.embedding_model = SentenceTransformer(config.embedding_model)
        self.knowledge_base = None
        self.index = None
        self._build_knowledge_index()
        
    def _build_knowledge_index(self):
        """Build FAISS index for medical knowledge retrieval"""
        logger.info("Building knowledge base index...")
        
        # Load medical knowledge (textbooks, guidelines, etc.)
        knowledge_docs = self._load_knowledge_documents()
        
        # Create embeddings
        embeddings = []
        self.knowledge_base = []
        
        for doc in tqdm(knowledge_docs, desc="Encoding knowledge documents"):
            embedding = self.embedding_model.encode(doc['text'])
            embeddings.append(embedding)
            self.knowledge_base.append(doc)
        
        # Build FAISS index
        embeddings_np = np.array(embeddings).astype('float32')
        self.index = faiss.IndexFlatL2(embeddings_np.shape[1])
        self.index.add(embeddings_np)
        
        logger.info(f"Built index with {len(self.knowledge_base)} documents")
    
    def _load_knowledge_documents(self) -> List[Dict]:
        """Load medical knowledge documents"""
        knowledge_docs = []
        
        # Example structure - replace with actual knowledge loading
        knowledge_sources = [
            "medical_textbooks.json",
            "clinical_guidelines.json",
            "drug_information.json",
            "disease_descriptions.json"
        ]
        
        for source_file in knowledge_sources:
            source_path = Path(self.config.knowledge_base_path) / source_file
            if source_path.exists():
                with open(source_path, 'r') as f:
                    docs = json.load(f)
                    knowledge_docs.extend(docs)
        
        # If no knowledge base exists, create sample entries
        if not knowledge_docs:
            knowledge_docs = [
                {
                    "text": "Pneumonia typically presents with fever, productive cough, and consolidation on chest X-ray. Common findings include air bronchograms and lobar consolidation.",
                    "source": "Internal Medicine Textbook",
                    "category": "respiratory"
                },
                {
                    "text": "Congestive heart failure shows cardiomegaly, pulmonary edema, and pleural effusions on chest X-ray. Kerley B lines may be visible.",
                    "source": "Cardiology Guidelines",
                    "category": "cardiac"
                }
            ]
        
        return knowledge_docs
    
    def enhance_with_knowledge(self, pseudo_note: str, 
                               medical_entities: List[str]) -> str:
        """
        Enhance pseudo-note with retrieved medical knowledge
        """
        # Create query from pseudo-note and entities
        query = f"{pseudo_note} {' '.join(medical_entities)}"
        
        # Encode query
        query_embedding = self.embedding_model.encode(query)
        
        # Search for relevant knowledge
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'),
            self.config.top_k_retrieval
        )
        
        # Collect retrieved knowledge
        retrieved_knowledge = []
        for idx in indices[0]:
            if idx < len(self.knowledge_base):
                doc = self.knowledge_base[idx]
                retrieved_knowledge.append(doc['text'])
        
        # Combine pseudo-note with knowledge
        enhanced_note = pseudo_note
        if retrieved_knowledge:
            knowledge_section = "\n[Clinical Context]: " + " ".join(retrieved_knowledge[:3])
            enhanced_note += knowledge_section
        
        return enhanced_note
    
    def prepare_for_cross_attention(self, enhanced_note: str) -> Dict:
        """
        Prepare enhanced note for cross-attention fusion
        Different from MDF-Net's spatialization approach
        """
        # Split into segments for cross-attention
        sentences = enhanced_note.split('.')
        
        attention_segments = {
            'clinical_data': [],
            'knowledge_context': [],
            'diagnostic_hints': []
        }
        
        for sentence in sentences:
            if any(vital in sentence.lower() for vital in ['temperature', 'heart rate', 'blood pressure']):
                attention_segments['clinical_data'].append(sentence)
            elif '[Clinical Context]' in sentence:
                attention_segments['knowledge_context'].append(sentence)
            elif any(disease in sentence.lower() for disease in ['pneumonia', 'failure', 'edema']):
                attention_segments['diagnostic_hints'].append(sentence)
        
        return attention_segments


class DatasetCreator:
    """
    Create final dataset with all preprocessing
    Combines MDF-Net approach with new enhancements
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.gcs_helper = GCSHelper(config)
        self.data_joiner = MIMICDataJoiner(config)
        self.image_preprocessor = ImagePreprocessor(config)
        self.text_preprocessor = TextPreprocessor(config)
        self.rag_enhancer = RAGKnowledgeEnhancer(config)
        
    def create_dataset(self, batch_size: int = 100, num_workers: int = 4,
                       create_small_samples: bool = False, small_sample_size: int = 100):
        """
        Main dataset creation pipeline with parallel processing

        Args:
            batch_size: Number of records to process in each batch
            num_workers: Number of parallel workers for image downloading
            create_small_samples: Whether to create small sample datasets
            small_sample_size: Number of records in small sample datasets
        """
        logger.info("Starting dataset creation pipeline...")
        logger.info(f"Using batch_size={batch_size}, num_workers={num_workers}")

        # Step 1: Join multimodal data (MDF-Net approach)
        joined_data = self.data_joiner.join_multimodal_data()
        total_records = len(joined_data)

        # Step 2: Process records in batches with parallel downloading
        processed_records = []
        failed_count = 0

        # Process in batches
        for batch_start in range(0, total_records, batch_size):
            batch_end = min(batch_start + batch_size, total_records)
            batch_data = joined_data.iloc[batch_start:batch_end]

            logger.info(f"Processing batch {batch_start//batch_size + 1}/{(total_records + batch_size - 1)//batch_size}: records {batch_start}-{batch_end}")

            # Batch download images in parallel using ThreadPoolExecutor
            image_paths = batch_data['image_path'].tolist()
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Pre-download all images in this batch
                futures = {executor.submit(self.image_preprocessor._download_and_cache_image, path): path
                          for path in image_paths}

                for future in tqdm(as_completed(futures), total=len(futures),
                                 desc=f"Downloading batch {batch_start//batch_size + 1}", leave=False):
                    try:
                        future.result()  # Just cache the result
                    except Exception as e:
                        path = futures[future]
                        logger.error(f"Error downloading {path}: {e}")

            # Now process each record (images are cached)
            for idx, row in tqdm(batch_data.iterrows(), total=len(batch_data),
                                desc=f"Processing batch {batch_start//batch_size + 1}", leave=False):
                try:
                    record = self.process_single_record(row)
                    if record is not None:
                        processed_records.append(record)
                    else:
                        failed_count += 1
                except KeyboardInterrupt:
                    logger.warning(f"Interrupted by user at record {idx}. Saving {len(processed_records)} successfully processed records...")
                    # Step 3: Create train/val/test splits with partial data
                    if len(processed_records) > 0:
                        self.create_splits(processed_records)
                        logger.info(f"Saved {len(processed_records)} records before interruption")
                    return
                except Exception as e:
                    failed_count += 1
                    logger.error(f"Error processing record {idx} (subject_id={row.get('subject_id', 'unknown')}): {e}")

            # Log progress
            logger.info(f"Batch complete. Total processed: {len(processed_records)}, failed: {failed_count}")

            # Clear image cache after each batch to free memory
            self.image_preprocessor._image_cache.clear()

            # Save intermediate batch after EVERY batch to prevent OOM
            # Using torch.save() instead of pickle for memory-efficient tensor serialization
            # Save every batch (50 records) instead of every 2 batches (100 records)
            current_batch_num = batch_start // batch_size + 1
            if len(processed_records) > 0:
                logger.info(f"Saving intermediate batch checkpoint at batch {current_batch_num} ({len(processed_records)} total records)...")
                self.save_intermediate_batch(processed_records, current_batch_num)
                processed_records = []  # Clear from memory after saving
                import gc
                gc.collect()  # Force garbage collection to free memory immediately

        logger.info(f"Processing complete! Successfully processed {len(processed_records)}/{total_records} records, failed {failed_count} records")

        # Step 3: Combine all records and create splits (streaming to avoid OOM)
        logger.info("Combining all processed records and creating splits...")

        # Save any remaining records not yet saved
        if len(processed_records) > 0:
            final_batch_num = (total_records // batch_size) + 1000  # Use high number to distinguish final batch
            logger.info(f"Saving final batch with {len(processed_records)} records...")
            self.save_intermediate_batch(processed_records, final_batch_num)
            processed_records = []  # Clear from memory

        # Step 4: Create train/val/test splits using streaming (memory-efficient)
        self.create_splits_from_batches_streaming(
            create_small_samples=create_small_samples,
            small_sample_size=small_sample_size
        )
        
    def process_single_record(self, row: pd.Series) -> Optional[Dict]:
        """Process a single multimodal record"""

        # Validate image path exists first
        image_path = row['image_path']
        if not self.image_preprocessor.gcs_helper.path_exists(image_path):
            logger.warning(f"Image not found: {image_path} (subject_id={row.get('subject_id', 'unknown')})")
            return None

        # Process image and extract attention regions in one pass (optimized)
        image_tensor, attention_mask = self.image_preprocessor.preprocess_image_and_attention(image_path)
        if image_tensor is None:
            logger.error(f"Failed to process existing image: {image_path} (subject_id={row.get('subject_id', 'unknown')})")
            return None
        
        # Process pseudo-note
        pseudo_note = row['pseudo_note']
        
        # Extract medical entities for RAG
        entities = self.text_preprocessor.extract_medical_entities(pseudo_note)
        
        # Enhance with knowledge (beyond MDF-Net)
        enhanced_note = self.rag_enhancer.enhance_with_knowledge(pseudo_note, entities)
        
        # Tokenize for Clinical ModernBERT
        text_encoding = self.text_preprocessor.preprocess_text(enhanced_note)
        
        # Prepare for cross-attention (different from MDF-Net spatialization)
        attention_segments = self.rag_enhancer.prepare_for_cross_attention(enhanced_note)
        
        # Create final record
        record = {
            'subject_id': row['subject_id'],
            'study_id': row['study_id'],
            'image_tensor': image_tensor,
            'attention_mask': attention_mask,
            'text_input_ids': text_encoding['input_ids'],
            'text_attention_mask': text_encoding['attention_mask'],
            'enhanced_note': enhanced_note,
            'attention_segments': attention_segments,
            'clinical_data': row['clinical_data'],
            'labels': self.extract_labels(row)  # To be implemented based on task
        }
        
        return record
    
    def extract_labels(self, row: pd.Series) -> Dict:
        """
        Extract labels for classification and localization tasks
        Placeholder - implement based on your specific task
        """
        labels = {
            'disease_labels': [],  # Multi-label classification
            'bbox_coordinates': [],  # Bounding boxes for localization
            'severity_scores': []  # Additional clinical scores
        }

        # Load from REFLACX or other annotation sources
        # This would typically involve loading radiologist annotations

        return labels

    def save_intermediate_batch(self, records: List[Dict], batch_num: int):
        """
        Save intermediate batch to disk to free memory using torch.save()

        torch.save() is much more memory-efficient than pickle for PyTorch tensors
        because it doesn't create temporary copies during serialization.

        Args:
            records: List of processed records
            batch_num: Batch number for filenaming
        """
        import gc

        if self.config.use_gcs:
            output_file = f"{self.config.output_path}/intermediate_batch_{batch_num:04d}.pt"
        else:
            output_dir = Path(self.config.output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"intermediate_batch_{batch_num:04d}.pt"

        # Use torch.save instead of pickle - much more memory efficient for tensors
        self.gcs_helper.write_torch(records, output_file)
        logger.info(f"Saved intermediate batch {batch_num} with {len(records)} records to {output_file}")

        # Force garbage collection to free memory immediately
        gc.collect()

    def load_all_intermediate_batches(self) -> List[Dict]:
        """
        Load all intermediate batch files and combine them using torch.load()

        Returns:
            Combined list of all records from intermediate batches
        """
        import glob
        all_records = []

        if self.config.use_gcs:
            # List all intermediate batch files in GCS
            from google.cloud import storage
            bucket_name = self.config.gcs_bucket
            prefix = f"{self.config.output_path}/intermediate_batch_"

            client = storage.Client(project=self.config.gcs_project_id)
            bucket = client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=prefix)

            for blob in blobs:
                # Load both .pt (torch) and .pkl (legacy pickle) files
                if blob.name.endswith('.pt'):
                    logger.info(f"Loading intermediate batch (torch): {blob.name}")
                    records = self.gcs_helper.read_torch(blob.name)
                    all_records.extend(records)
                elif blob.name.endswith('.pkl'):
                    logger.info(f"Loading intermediate batch (pickle): {blob.name}")
                    records = self.gcs_helper.read_pickle(blob.name)
                    all_records.extend(records)
        else:
            # List all intermediate batch files locally
            output_dir = Path(self.config.output_path)

            # Load .pt files (torch format - preferred)
            batch_files_pt = sorted(output_dir.glob("intermediate_batch_*.pt"))
            for batch_file in batch_files_pt:
                logger.info(f"Loading intermediate batch (torch): {batch_file}")
                records = self.gcs_helper.read_torch(str(batch_file))
                all_records.extend(records)

            # Also load legacy .pkl files if they exist
            batch_files_pkl = sorted(output_dir.glob("intermediate_batch_*.pkl"))
            for batch_file in batch_files_pkl:
                logger.info(f"Loading intermediate batch (pickle): {batch_file}")
                records = self.gcs_helper.read_pickle(str(batch_file))
                all_records.extend(records)

        logger.info(f"Loaded {len(all_records)} total records from intermediate batches")
        return all_records

    def get_batch_file_list(self) -> List:
        """
        Get list of all intermediate batch files without loading them

        Returns:
            List of batch file references (blobs for GCS, paths for local)
        """
        if self.config.use_gcs:
            # List all intermediate batch files in GCS
            from google.cloud import storage
            prefix = f"{self.config.output_path}/intermediate_batch_"

            client = storage.Client(project=self.config.gcs_project_id)
            bucket = client.bucket(self.config.gcs_bucket)
            blobs = list(bucket.list_blobs(prefix=prefix))

            # Filter for .pt files only (ignore legacy .pkl)
            batch_files = [blob for blob in blobs if blob.name.endswith('.pt')]
            batch_files.sort(key=lambda b: b.name)
            logger.info(f"Found {len(batch_files)} intermediate batch files in GCS")
        else:
            # List all intermediate batch files locally
            output_dir = Path(self.config.output_path)
            batch_files = sorted(output_dir.glob("intermediate_batch_*.pt"))
            logger.info(f"Found {len(batch_files)} intermediate batch files locally")

        return batch_files

    def count_total_records_streaming(self, batch_files: List) -> int:
        """
        Count total records across all batches without loading full data

        Args:
            batch_files: List of batch file references

        Returns:
            Total number of records
        """
        total_count = 0
        logger.info("Counting total records...")

        for batch_file in tqdm(batch_files, desc="Counting records"):
            try:
                # Load just to count, then immediately discard
                if self.config.use_gcs:
                    records = self.gcs_helper.read_torch(batch_file.name)
                else:
                    records = self.gcs_helper.read_torch(str(batch_file))

                total_count += len(records)
                del records  # Free memory immediately
            except Exception as e:
                logger.error(f"Error counting records in {batch_file}: {e}")

        logger.info(f"Total records across all batches: {total_count}")
        return total_count

    def extract_labels_for_stratification(self, batch_files: List) -> List[int]:
        """
        Extract stratification keys from all batches
        Uses first available class label or defaults to subject_id-based distribution

        Args:
            batch_files: List of batch file references

        Returns:
            List of stratification_keys for all records (one per record)
        """
        strat_keys = []

        logger.info("Extracting labels for stratified splitting...")

        for batch_file in tqdm(batch_files, desc="Extracting labels"):
            try:
                # Load batch
                if self.config.use_gcs:
                    records = self.gcs_helper.read_torch(batch_file.name)
                else:
                    records = self.gcs_helper.read_torch(str(batch_file))

                # Extract stratification keys (no need to store record_ids - just count)
                for record in records:
                    # Extract stratification key
                    # Try to use disease labels if available, otherwise use subject_id mod 10 for distribution
                    labels = record.get('labels', {})
                    disease_labels = labels.get('disease_labels', [])

                    if disease_labels and len(disease_labels) > 0:
                        # Use first disease label as stratification key
                        strat_key = hash(str(disease_labels[0])) % 10
                    else:
                        # Use subject_id for deterministic pseudo-random stratification
                        strat_key = record['subject_id'] % 10

                    strat_keys.append(strat_key)

                del records  # Free memory immediately
            except Exception as e:
                logger.error(f"Error extracting labels from {batch_file}: {e}")

        logger.info(f"Extracted {len(strat_keys)} stratification keys for splitting")
        return strat_keys

    def create_stratified_split_indices(self, strat_keys: List[int],
                                       total_count: int) -> Tuple[set, set, set]:
        """
        Create stratified train/val/test split indices

        Args:
            strat_keys: Stratification keys for each record
            total_count: Total number of records

        Returns:
            Tuple of (train_indices, val_indices, test_indices) as sets
        """
        from collections import defaultdict

        logger.info("Creating stratified split indices...")

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
                                             small_sample_size: int = 100):
        """
        Create train/val/test splits by streaming through batches (memory-efficient)
        Implements stratified splitting to evenly distribute classes

        Args:
            create_small_samples: Whether to create small sample versions
            small_sample_size: Number of records in small samples
        """
        import gc

        logger.info("=" * 60)
        logger.info("Creating stratified train/val/test splits (streaming mode)")
        logger.info("=" * 60)

        # Step 1: Get list of batch files
        batch_files = self.get_batch_file_list()

        if not batch_files:
            logger.error("No intermediate batch files found!")
            return

        # Step 2: Count total records
        total_count = self.count_total_records_streaming(batch_files)

        if total_count == 0:
            logger.error("No records found in intermediate batches!")
            return

        # Step 3: Extract labels for stratification
        strat_keys = self.extract_labels_for_stratification(batch_files)
        logger.info(f"Memory-efficient extraction complete. Creating split indices...")

        # Step 4: Create stratified split indices
        train_indices, val_indices, test_indices = self.create_stratified_split_indices(
            strat_keys, total_count
        )

        # Free strat_keys after creating indices
        del strat_keys
        gc.collect()

        # Step 5: Stream through batches and write to split files
        logger.info("Streaming records to split files...")

        # Initialize accumulators for each split
        train_records = []
        val_records = []
        test_records = []

        # Also track small samples
        train_small = []
        val_small = []
        test_small = []

        current_idx = 0
        # Write batch size: optimized for low-memory machines
        # Each record is ~6-8 MB, and we have 3 accumulators (train/val/test)
        # 100 records × 6 MB × 3 = ~1.8 GB - safe for 7.5GB RAM machines
        write_batch_size = 100

        for batch_file in tqdm(batch_files, desc="Writing splits"):
            try:
                # Load batch
                if self.config.use_gcs:
                    records = self.gcs_helper.read_torch(batch_file.name)
                else:
                    records = self.gcs_helper.read_torch(str(batch_file))

                # Assign each record to appropriate split
                for record in records:
                    if current_idx in train_indices:
                        train_records.append(record)
                        if create_small_samples and len(train_small) < small_sample_size:
                            train_small.append(record)
                    elif current_idx in val_indices:
                        val_records.append(record)
                        if create_small_samples and len(val_small) < small_sample_size:
                            val_small.append(record)
                    elif current_idx in test_indices:
                        test_records.append(record)
                        if create_small_samples and len(test_small) < small_sample_size:
                            test_small.append(record)

                    current_idx += 1

                del records  # Free memory immediately
                gc.collect()

                # Periodically write accumulated records to free memory
                if len(train_records) >= write_batch_size:
                    self._append_to_split_file('train', train_records)
                    train_records = []
                    gc.collect()

                if len(val_records) >= write_batch_size:
                    self._append_to_split_file('val', val_records)
                    val_records = []
                    gc.collect()

                if len(test_records) >= write_batch_size:
                    self._append_to_split_file('test', test_records)
                    test_records = []
                    gc.collect()

            except Exception as e:
                logger.error(f"Error processing {batch_file}: {e}")

        # Write remaining records
        if train_records:
            self._append_to_split_file('train', train_records)
        if val_records:
            self._append_to_split_file('val', val_records)
        if test_records:
            self._append_to_split_file('test', test_records)

        # Combine all split chunks into final files
        logger.info("Combining split chunks into final files...")
        self._combine_split_chunks('train')
        self._combine_split_chunks('val')
        self._combine_split_chunks('test')

        # Create small samples if requested
        if create_small_samples:
            logger.info(f"Creating small sample datasets ({small_sample_size} records each)...")
            self._save_small_sample('train', train_small)
            self._save_small_sample('val', val_small)
            self._save_small_sample('test', test_small)

        # Save metadata
        metadata = {
            'config': {k: str(v) for k, v in self.config.__dict__.items()},
            'n_train': len(train_indices),
            'n_val': len(val_indices),
            'n_test': len(test_indices),
            'total_records': total_count,
            'stratified': True,
            'small_samples_created': create_small_samples,
            'small_sample_size': small_sample_size if create_small_samples else 0
        }

        if self.config.use_gcs:
            metadata_file = f"{self.config.output_path}/metadata.json"
        else:
            metadata_file = Path(self.config.output_path) / 'metadata.json'

        self.gcs_helper.write_json(metadata, metadata_file)

        logger.info("=" * 60)
        logger.info("Dataset splitting complete!")
        logger.info(f"  Train: {len(train_indices):,} records")
        logger.info(f"  Val: {len(val_indices):,} records")
        logger.info(f"  Test: {len(test_indices):,} records")
        if create_small_samples:
            logger.info(f"  Small samples: {small_sample_size} records each")
        logger.info("=" * 60)

    def _append_to_split_file(self, split_name: str, records: List[Dict]):
        """
        Append records to a split chunk file (for streaming writes)

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
        Combine all chunk files for a split into a single final file

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
            bucket = client.bucket(self.config.gcs_bucket)
            chunk_blobs = list(bucket.list_blobs(prefix=prefix))
            chunk_blobs.sort(key=lambda b: b.name)
        else:
            output_dir = Path(self.config.output_path)
            chunk_files = sorted(output_dir.glob(f"{split_name}_chunk_*.pt"))

        # Load and combine all chunks
        all_records = []

        if self.config.use_gcs:
            for blob in tqdm(chunk_blobs, desc=f"Loading {split_name} chunks"):
                records = self.gcs_helper.read_torch(blob.name)
                all_records.extend(records)
                del records
                gc.collect()

            # Delete chunk files after combining
            for blob in chunk_blobs:
                blob.delete()
        else:
            for chunk_file in tqdm(chunk_files, desc=f"Loading {split_name} chunks"):
                records = self.gcs_helper.read_torch(str(chunk_file))
                all_records.extend(records)
                del records
                gc.collect()

            # Delete chunk files after combining
            for chunk_file in chunk_files:
                chunk_file.unlink()

        # Save final combined file
        if self.config.use_gcs:
            output_file = f"{self.config.output_path}/{split_name}_data.pkl"
        else:
            output_file = Path(self.config.output_path) / f"{split_name}_data.pkl"

        self.gcs_helper.write_pickle(all_records, output_file)
        logger.info(f"Saved {split_name} split with {len(all_records):,} records")

        del all_records
        gc.collect()

    def _save_small_sample(self, split_name: str, records: List[Dict]):
        """
        Save small sample version of a split

        Args:
            split_name: 'train', 'val', or 'test'
            records: Sample records
        """
        if not records:
            logger.warning(f"No records for {split_name}_small")
            return

        if self.config.use_gcs:
            output_file = f"{self.config.output_path}/{split_name}_small.pkl"
        else:
            output_file = Path(self.config.output_path) / f"{split_name}_small.pkl"

        self.gcs_helper.write_pickle(records, output_file)
        logger.info(f"Saved {split_name}_small with {len(records)} records")

    def create_splits(self, records: List[Dict]):
        """Create train/val/test splits and save"""
        n_total = len(records)
        n_train = int(n_total * self.config.train_split)
        n_val = int(n_total * self.config.val_split)

        # Shuffle records
        np.random.shuffle(records)

        # Split data
        train_records = records[:n_train]
        val_records = records[n_train:n_train + n_val]
        test_records = records[n_train + n_val:]

        # Create output directory for local mode
        if not self.config.use_gcs:
            output_dir = Path(self.config.output_path)
            output_dir.mkdir(parents=True, exist_ok=True)

        splits = {
            'train': train_records,
            'val': val_records,
            'test': test_records
        }

        for split_name, split_data in splits.items():
            if self.config.use_gcs:
                output_file = f"{self.config.output_path}/{split_name}_data.pkl"
            else:
                output_file = Path(self.config.output_path) / f"{split_name}_data.pkl"

            self.gcs_helper.write_pickle(split_data, output_file)
            logger.info(f"Saved {split_name} split with {len(split_data)} records")

        # Save metadata
        metadata = {
            'config': {k: str(v) for k, v in self.config.__dict__.items()},
            'n_train': len(train_records),
            'n_val': len(val_records),
            'n_test': len(test_records),
            'total_records': n_total
        }

        if self.config.use_gcs:
            metadata_file = f"{self.config.output_path}/metadata.json"
        else:
            metadata_file = Path(self.config.output_path) / 'metadata.json'

        self.gcs_helper.write_json(metadata, metadata_file)
        logger.info(f"Dataset creation complete! Total records: {n_total}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='MIMIC Enhanced MDF-Net Data Preprocessing Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data paths
    parser.add_argument(
        '--mimic-cxr-path',
        type=str,
        default='physionet.org/files/mimic-cxr-jpg/2.1.0',
        help='Path to MIMIC-CXR dataset (local path or GCS prefix). Automatically set to "" when using --gcs-cxr-bucket (PhysioNet bucket has files at root)'
    )

    parser.add_argument(
        '--mimic-iv-path',
        type=str,
        default='physionet.org/files/mimiciv/3.1',
        help='Path to MIMIC-IV dataset (local or S3 prefix)'
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
            small_sample_size=args.small_sample_size
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