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

            self.gcs_client = storage.Client()

            # Initialize your main bucket (MIMIC-IV, MIMIC-IV-ED, etc.)
            if config.gcs_bucket:
                self.bucket = self.gcs_client.bucket(config.gcs_bucket)
                logger.info(f"Initialized GCS client for bucket: {config.gcs_bucket}")

            # Initialize separate MIMIC-CXR bucket (PhysioNet's public bucket)
            if config.gcs_cxr_bucket:
                self.cxr_bucket = self.gcs_client.bucket(config.gcs_cxr_bucket)
                logger.info(f"Initialized GCS MIMIC-CXR bucket: {config.gcs_cxr_bucket}")

            # Initialize output bucket
            if config.output_gcs_bucket:
                self.output_bucket = self.gcs_client.bucket(config.output_gcs_bucket)
                logger.info(f"Initialized GCS output bucket: {config.output_gcs_bucket}")

    def _get_bucket_for_path(self, path: str):
        """Determine which bucket to use based on path"""
        path_str = str(path)
        # If path contains mimic-cxr-jpg and we have a separate CXR bucket, use it
        if 'mimic-cxr-jpg' in path_str and self.cxr_bucket:
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
            metadata_path = f"{self.config.mimic_cxr_path}/mimic-cxr-2.0.0-metadata.csv.gz"
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
                file_path = f"{base_ed_path}/ed/{table}.csv.gz"
            else:
                file_path = Path(base_ed_path) / "ed" / f"{table}.csv"

            if self.gcs_helper.path_exists(file_path):
                ed_data[table] = self.gcs_helper.read_csv(file_path, compression='gzip' if str(file_path).endswith('.gz') else None)
                logger.info(f"Loaded {table}: {len(ed_data[table])} records")
            else:
                logger.warning(f"Could not find {table} at {file_path}")

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
        
        for idx, cxr_row in tqdm(cxr_metadata.iterrows(), total=len(cxr_metadata)):
            subject_id = cxr_row['subject_id']
            study_time = pd.to_datetime(cxr_row['StudyDate'])
            
            # Find matching ED stay within 24 hours
            if 'edstays' in ed_data:
                ed_stays = ed_data['edstays'][ed_data['edstays']['subject_id'] == subject_id]
                
                for _, ed_row in ed_stays.iterrows():
                    ed_time = pd.to_datetime(ed_row['intime'])
                    time_diff = abs((study_time - ed_time).total_seconds() / 3600)
                    
                    if time_diff <= 24:  # Within 24 hours
                        # Collect all clinical data
                        clinical_data = self._collect_clinical_data(subject_id, ed_row, ed_data)
                        
                        # Create pseudo-note
                        pseudo_note = self.create_pseudo_notes(clinical_data)
                        
                        joined_record = {
                            'subject_id': subject_id,
                            'study_id': cxr_row['study_id'],
                            'dicom_id': cxr_row['dicom_id'],
                            'image_path': self._get_image_path(cxr_row),
                            'pseudo_note': pseudo_note,
                            'clinical_data': json.dumps(clinical_data),
                            'ed_stay_id': ed_row['stay_id'],
                            'study_time': study_time,
                            'ed_time': ed_time
                        }
                        
                        joined_data.append(joined_record)
                        break
        
        result_df = pd.DataFrame(joined_data)
        logger.info(f"Successfully joined {len(result_df)} multimodal records")
        
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
        """Get the full path to the image file"""
        # MIMIC-CXR directory structure: p10/p10000032/s50414267/...
        patient_folder = f"p{str(cxr_row['subject_id'])[:2]}"
        subject_folder = f"p{cxr_row['subject_id']}"
        study_folder = f"s{cxr_row['study_id']}"
        
        image_path = Path(self.config.mimic_cxr_path) / "files" / patient_folder / \
                     subject_folder / study_folder / f"{cxr_row['dicom_id']}.jpg"
        
        return str(image_path)


class ImagePreprocessor:
    """
    Preprocess images for BiomedCLIP-CXR
    Enhanced from MDF-Net's basic preprocessing
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.gcs_helper = GCSHelper(config)
        self.transform = self._create_transform()
        
    def _create_transform(self):
        """Create transformation pipeline for BiomedCLIP-CXR"""
        return transforms.Compose([
            transforms.Resize((self.config.image_size, self.config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.normalize_mean,
                               std=self.config.normalize_std)
        ])
    
    def preprocess_image(self, image_path: str) -> Optional[torch.Tensor]:
        """Preprocess a single image"""
        try:
            image = self.gcs_helper.read_image(image_path).convert('RGB')

            # Apply BiomedCLIP-CXR transformations
            image_tensor = self.transform(image)

            return image_tensor

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None
    
    def extract_attention_regions(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract attention regions using edge detection
        Similar to MDF-Net's approach for finding abnormal regions
        """
        try:
            image = self.gcs_helper.read_image_cv2(image_path)

            # Apply CLAHE for better contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)

            # Edge detection to find potential abnormal regions
            edges = cv2.Canny(enhanced, 50, 150)

            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create attention mask
            attention_mask = np.zeros_like(image, dtype=np.float32)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Filter small regions
                    cv2.drawContours(attention_mask, [contour], -1, 1.0, -1)

            return attention_mask

        except Exception as e:
            logger.error(f"Error extracting attention regions from {image_path}: {e}")
            return None


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
        
    def create_dataset(self):
        """Main dataset creation pipeline"""
        logger.info("Starting dataset creation pipeline...")
        
        # Step 1: Join multimodal data (MDF-Net approach)
        joined_data = self.data_joiner.join_multimodal_data()
        
        # Step 2: Process each record
        processed_records = []
        
        for idx, row in tqdm(joined_data.iterrows(), total=len(joined_data)):
            try:
                record = self.process_single_record(row)
                if record is not None:
                    processed_records.append(record)
            except Exception as e:
                logger.error(f"Error processing record {idx}: {e}")
                continue
        
        # Step 3: Create train/val/test splits
        self.create_splits(processed_records)
        
        logger.info("Dataset creation completed!")
        
    def process_single_record(self, row: pd.Series) -> Optional[Dict]:
        """Process a single multimodal record"""
        
        # Process image
        image_tensor = self.image_preprocessor.preprocess_image(row['image_path'])
        if image_tensor is None:
            return None
        
        # Extract attention regions (MDF-Net style)
        attention_mask = self.image_preprocessor.extract_attention_regions(row['image_path'])
        
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
        help='Path to MIMIC-CXR dataset (local or S3 prefix)'
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
    dataset_creator.create_dataset()

    logger.info("=" * 60)
    logger.info("Processing complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()