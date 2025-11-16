"""
Phase 2: Enhanced Pseudo-Note Generation and RAG Integration
Integrates with Phase 1 outputs to:
- Generate narrative pseudo-notes from structured clinical data
- Enhance notes with RAG knowledge retrieval
- Tokenize enhanced notes for Clinical ModernBERT
- Prepare for cross-modal fusion

Supports both GCS and local file systems
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import pickle
from tqdm import tqdm
import logging
import argparse
import torch
import re
from datetime import datetime

# Import from phase1 for consistency
from phase1_preprocess_streaming import DataConfig, GCSHelper

# Text processing
from transformers import AutoTokenizer

# RAG processing
import faiss
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PseudoNoteGenerator:
    """
    Generate narrative pseudo-notes from structured clinical data
    Converts structured vitals, labs, medications into clinical narrative text
    """

    def __init__(self):
        """Initialize Pseudo-Note Generator"""
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

    def create_pseudo_note(self, clinical_features: torch.Tensor,
                          feature_names: List[str],
                          chiefcomplaint: Optional[str] = None) -> str:
        """
        Convert structured clinical features to narrative text

        Args:
            clinical_features: Tensor of clinical feature values
            feature_names: Names of features in order
            chiefcomplaint: Optional chief complaint text

        Returns:
            Narrative pseudo-note as string
        """
        parts = []

        # Convert tensor to dict for easier access
        feature_dict = {}
        if clinical_features is not None:
            features_list = clinical_features.tolist() if torch.is_tensor(clinical_features) else clinical_features
            for name, value in zip(feature_names, features_list):
                if value != 0.0:  # Skip missing/zero values
                    feature_dict[name] = value

        # Demographics
        demo_parts = []
        if 'age' in feature_dict:
            age = int(feature_dict['age'])
            if age > 0:
                demo_parts.append(f"{age} year old")

        if 'gender' in feature_dict:
            gender = 'M' if feature_dict['gender'] == 1.0 else 'F'
            demo_parts.append(gender)

        if demo_parts:
            parts.append(f"Patient is a {' '.join(demo_parts)}.")

        # Chief complaint
        if chiefcomplaint and str(chiefcomplaint) != 'nan':
            complaint = self._expand_abbreviations(str(chiefcomplaint))
            parts.append(f"Chief complaint: {complaint}.")

        # Vital signs
        vital_parts = []
        if 'temperature' in feature_dict and feature_dict['temperature'] > 0:
            vital_parts.append(f"temperature {feature_dict['temperature']:.1f}°F")
        if 'heartrate' in feature_dict and feature_dict['heartrate'] > 0:
            vital_parts.append(f"heart rate {int(feature_dict['heartrate'])} bpm")
        if 'resprate' in feature_dict and feature_dict['resprate'] > 0:
            vital_parts.append(f"respiratory rate {int(feature_dict['resprate'])} breaths/min")
        if 'o2sat' in feature_dict and feature_dict['o2sat'] > 0:
            vital_parts.append(f"oxygen saturation {int(feature_dict['o2sat'])}%")
        if 'sbp' in feature_dict and feature_dict['sbp'] > 0:
            sbp = int(feature_dict['sbp'])
            dbp = int(feature_dict.get('dbp', 0))
            if dbp > 0:
                vital_parts.append(f"blood pressure {sbp}/{dbp} mmHg")
            else:
                vital_parts.append(f"systolic blood pressure {sbp} mmHg")

        if vital_parts:
            parts.append(f"Vital signs: {', '.join(vital_parts)}.")

        # Clinical assessment
        if 'acuity' in feature_dict and feature_dict['acuity'] > 0:
            acuity = int(feature_dict['acuity'])
            acuity_desc = {
                1: 'critical (Level 1)',
                2: 'emergent (Level 2)',
                3: 'urgent (Level 3)',
                4: 'semi-urgent (Level 4)',
                5: 'non-urgent (Level 5)'
            }
            if acuity in acuity_desc:
                parts.append(f"Triage acuity: {acuity_desc[acuity]}.")

        if 'pain' in feature_dict and feature_dict['pain'] > 0:
            pain = int(feature_dict['pain'])
            parts.append(f"Pain score: {pain}/10.")

        # If no meaningful data, return minimal note
        if not parts:
            return "Patient presenting to emergency department for evaluation."

        return " ".join(parts)

    def _expand_abbreviations(self, text: str) -> str:
        """Expand medical abbreviations in text"""
        text_lower = text.lower()
        for abbr, expansion in self.abbreviations.items():
            text_lower = re.sub(r'\b' + abbr + r'\b', expansion, text_lower)
        return text_lower


class RAGEnhancer:
    """
    RAG Knowledge Enhancement for pseudo-notes
    Retrieves relevant medical knowledge and augments clinical notes
    """

    def __init__(self, config: DataConfig):
        """
        Initialize RAG Enhancer

        Args:
            config: DataConfig with RAG settings
        """
        self.config = config

        # Initialize embedding model
        logger.info(f"Loading sentence transformer: {config.embedding_model}")
        self.encoder = SentenceTransformer(config.embedding_model)

        # Initialize FAISS index
        self.index = None
        self.documents = []

        # Build knowledge base
        self.build_knowledge_base()

    def build_knowledge_base(self):
        """Build FAISS index from medical knowledge documents"""
        logger.info("Building medical knowledge base index...")

        # Load medical knowledge documents
        # In production, load from actual medical knowledge base
        # For now, use expanded sample knowledge
        sample_knowledge = [
            # Pulmonary conditions
            "Pneumonia appears as consolidation or ground-glass opacities on chest X-ray, typically in a lobar or segmental distribution. Common findings include air bronchograms and pleural effusions.",
            "Congestive heart failure manifests as cardiomegaly, pulmonary edema, pleural effusions, and Kerley B lines on chest radiography. Cephalization of pulmonary vessels may be present.",
            "Pulmonary edema shows bilateral perihilar infiltrates with a butterfly pattern. Interstitial edema appears as thickened interlobular septa.",
            "Pneumothorax presents as absence of lung markings peripherally with a visible visceral pleural line. Tension pneumothorax causes mediastinal shift.",

            # Cardiac conditions
            "Cardiomegaly is defined as a cardiothoracic ratio greater than 0.5 on PA chest radiograph. May indicate heart failure, valvular disease, or cardiomyopathy.",
            "Pleural effusion appears as blunting of the costophrenic angle with a meniscus sign. Large effusions show complete opacification of the hemithorax.",

            # Infectious diseases
            "COVID-19 pneumonia typically shows bilateral, peripheral ground-glass opacities with or without consolidation. Distribution is often multilobar.",
            "Tuberculosis may present as upper lobe infiltrates, cavitation, or miliary pattern. Hilar lymphadenopathy is common in primary TB.",

            # Trauma
            "Rib fractures may be visible on chest X-ray but are often better seen on CT. Complications include pneumothorax and hemothorax.",
            "Pulmonary contusion appears as patchy alveolar infiltrates in a non-anatomic distribution following blunt chest trauma.",

            # Chronic conditions
            "COPD shows hyperinflation with flattened diaphragms, increased retrosternal airspace, and attenuated peripheral vascular markings.",
            "Interstitial lung disease presents with reticular or reticulonodular opacities, often with a basilar or peripheral predominance.",

            # Vascular conditions
            "Pulmonary embolism may show oligemia (Westermark sign), pleural effusion, or elevated hemidiaphragm on chest X-ray, but findings are often subtle.",
            "Aortic dissection may show widened mediastinum on chest radiograph. CT angiography is the gold standard for diagnosis.",

            # Common ED presentations
            "Chest pain with normal chest X-ray requires clinical correlation. Consider cardiac ischemia, pulmonary embolism, or musculoskeletal causes.",
            "Shortness of breath can result from cardiac, pulmonary, or metabolic causes. Chest X-ray helps differentiate between these etiologies.",
            "Fever and cough with infiltrate on chest X-ray suggests pneumonia. Clinical context determines appropriate antibiotic therapy.",
        ]

        # Encode documents
        logger.info(f"Encoding {len(sample_knowledge)} medical knowledge documents...")
        embeddings = self.encoder.encode(sample_knowledge, show_progress_bar=True)

        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))

        self.documents = sample_knowledge
        logger.info(f"Built knowledge base with {len(self.documents)} documents")

    def enhance_note(self, pseudo_note: str, view_position: str = None,
                    retrieved_knowledge: List[str] = None) -> str:
        """
        Enhance pseudo-note with RAG-retrieved medical knowledge

        Args:
            pseudo_note: Original pseudo-note text
            view_position: Chest X-ray view position (PA, AP, etc.)
            retrieved_knowledge: Pre-retrieved knowledge from phase1 (optional)

        Returns:
            Enhanced pseudo-note with medical knowledge context
        """
        # If knowledge was already retrieved in phase1, use it
        if retrieved_knowledge:
            knowledge_text = " ".join(retrieved_knowledge)
        else:
            # Retrieve knowledge based on pseudo-note content
            knowledge_docs = self.retrieve_knowledge(pseudo_note, view_position)
            knowledge_text = " ".join(knowledge_docs)

        # Combine pseudo-note with retrieved knowledge
        # Format: [CLINICAL CONTEXT] pseudo-note [MEDICAL KNOWLEDGE] knowledge
        enhanced_note = f"[CLINICAL PRESENTATION] {pseudo_note} [MEDICAL CONTEXT] {knowledge_text}"

        return enhanced_note

    def retrieve_knowledge(self, query_text: str, view_position: str = None,
                          top_k: int = None) -> List[str]:
        """
        Retrieve relevant medical knowledge for a query

        Args:
            query_text: Query text (pseudo-note or clinical context)
            view_position: Optional chest X-ray view position
            top_k: Number of documents to retrieve

        Returns:
            List of relevant medical knowledge texts
        """
        if top_k is None:
            top_k = self.config.top_k_retrieval

        # Enhance query with view position if available
        if view_position:
            query_text = f"chest x-ray {view_position} view {query_text}"

        # Encode query
        query_embedding = self.encoder.encode([query_text])

        # Search index
        distances, indices = self.index.search(
            query_embedding.astype('float32'),
            min(top_k, len(self.documents))
        )

        # Return retrieved documents
        retrieved = [self.documents[idx] for idx in indices[0]]
        return retrieved


class TextEnhancer:
    """
    Enhance and tokenize text for Clinical ModernBERT
    Combines pseudo-notes with RAG knowledge and tokenizes
    """

    def __init__(self, config: DataConfig):
        """
        Initialize Text Enhancer

        Args:
            config: DataConfig with text processing settings
        """
        self.config = config

        # Initialize tokenizer (using BERT as placeholder for Clinical ModernBERT)
        logger.info("Loading tokenizer for Clinical ModernBERT...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            'bert-base-uncased',  # Replace with Clinical ModernBERT when available
            max_length=config.max_text_length
        )
        logger.info(f"Tokenizer loaded with max length: {config.max_text_length}")

    def tokenize_enhanced_note(self, enhanced_note: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize enhanced pseudo-note for Clinical ModernBERT

        Args:
            enhanced_note: RAG-enhanced pseudo-note text

        Returns:
            Dictionary with input_ids and attention_mask tensors
        """
        encoding = self.tokenizer(
            enhanced_note,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_text_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }


class Phase2Processor:
    """
    Phase 2 Processor: Generate pseudo-notes and enhance with RAG
    Integrates with Phase 1 outputs
    """

    def __init__(self, config: DataConfig):
        """
        Initialize Phase 2 Processor

        Args:
            config: DataConfig from phase1
        """
        self.config = config
        self.gcs_helper = GCSHelper(config)

        # Initialize components
        self.pseudo_note_generator = PseudoNoteGenerator()
        self.rag_enhancer = RAGEnhancer(config)
        self.text_enhancer = TextEnhancer(config)

        logger.info("Phase 2 Processor initialized")
        logger.info(f"  Mode: {'GCS' if config.use_gcs else 'Local'}")
        logger.info(f"  Max text length: {config.max_text_length}")
        logger.info(f"  RAG top-k: {config.top_k_retrieval}")

    def process_split(self, split_name: str, use_small_sample: bool = False) -> List[Dict]:
        """
        Process a data split (train/val/test) from Phase 1 output

        Args:
            split_name: 'train', 'val', or 'test'
            use_small_sample: Use small sample files (e.g., train_small.pt)

        Returns:
            List of enhanced records
        """
        # Construct input file path
        if use_small_sample:
            filename = f"{split_name}_small.pt"
        else:
            filename = f"{split_name}_data.pt"

        if self.config.use_gcs:
            input_path = f"{self.config.output_path}/{filename}"
        else:
            input_path = Path(self.config.output_path).expanduser() / filename
            input_path = str(input_path)

        logger.info(f"Loading {split_name} split from: {input_path}")

        # Load data from Phase 1
        try:
            records = self.gcs_helper.read_torch(input_path)
            logger.info(f"Loaded {len(records)} records from {split_name} split")
        except Exception as e:
            logger.error(f"Error loading {split_name} split: {e}")
            raise

        # Process each record
        enhanced_records = []
        for record in tqdm(records, desc=f"Processing {split_name}"):
            enhanced_record = self.process_record(record)
            if enhanced_record:
                enhanced_records.append(enhanced_record)

        logger.info(f"Enhanced {len(enhanced_records)} records in {split_name} split")
        return enhanced_records

    def process_record(self, record: Dict) -> Optional[Dict]:
        """
        Process a single record: generate pseudo-note and enhance with RAG

        Args:
            record: Record from Phase 1 output

        Returns:
            Enhanced record with pseudo-note and enhanced text tokens
        """
        try:
            # Extract data from Phase 1 record
            clinical_features = record.get('clinical_features')
            retrieved_knowledge = record.get('retrieved_knowledge', [])
            view_position = record.get('labels', {}).get('view_position')

            # Get original text tokens to extract chief complaint if available
            chiefcomplaint = None
            # Note: In phase1, chiefcomplaint is tokenized but not stored as raw text
            # We'll generate note from clinical features only

            # Generate pseudo-note from structured clinical data
            pseudo_note = self.pseudo_note_generator.create_pseudo_note(
                clinical_features=clinical_features,
                feature_names=self.config.clinical_features,
                chiefcomplaint=chiefcomplaint
            )

            # Enhance pseudo-note with RAG knowledge
            enhanced_note = self.rag_enhancer.enhance_note(
                pseudo_note=pseudo_note,
                view_position=view_position,
                retrieved_knowledge=retrieved_knowledge
            )

            # Tokenize enhanced note
            enhanced_tokens = self.text_enhancer.tokenize_enhanced_note(enhanced_note)

            # Create enhanced record (keep all original data)
            enhanced_record = record.copy()
            enhanced_record.update({
                'pseudo_note': pseudo_note,
                'enhanced_note': enhanced_note,
                'enhanced_text_tokens': enhanced_tokens,
                'phase2_processed': True
            })

            return enhanced_record

        except Exception as e:
            logger.error(f"Error processing record {record.get('dicom_id', 'unknown')}: {e}")
            return None

    def process_all_splits(self, use_small_sample: bool = False):
        """
        Process all data splits (train/val/test)

        Args:
            use_small_sample: Use small sample files for testing
        """
        logger.info("=" * 60)
        logger.info("Phase 2: Enhanced Pseudo-Note Generation")
        logger.info("=" * 60)

        splits = ['train', 'val', 'test']

        for split_name in splits:
            logger.info(f"\nProcessing {split_name} split...")

            try:
                # Process split
                enhanced_records = self.process_split(split_name, use_small_sample)

                # Save enhanced records
                if use_small_sample:
                    output_filename = f"{split_name}_small_enhanced.pt"
                else:
                    output_filename = f"{split_name}_data_enhanced.pt"

                if self.config.use_gcs:
                    output_path = f"{self.config.output_path}/{output_filename}"
                else:
                    output_dir = Path(self.config.output_path).expanduser()
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / output_filename
                    output_path = str(output_path)

                logger.info(f"Saving enhanced {split_name} split to: {output_path}")
                self.gcs_helper.write_torch(enhanced_records, output_path)

                # Log sample enhanced note
                if enhanced_records:
                    sample = enhanced_records[0]
                    logger.info(f"\nSample enhanced note from {split_name}:")
                    logger.info(f"  Pseudo-note: {sample['pseudo_note'][:200]}...")
                    logger.info(f"  Enhanced note length: {len(sample['enhanced_note'])} chars")

            except Exception as e:
                logger.error(f"Error processing {split_name} split: {e}")
                continue

        logger.info("=" * 60)
        logger.info("Phase 2 Complete!")
        logger.info("=" * 60)

    def save_metadata(self):
        """Save processing metadata"""
        metadata = {
            'phase': 2,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'max_text_length': self.config.max_text_length,
                'top_k_retrieval': self.config.top_k_retrieval,
                'embedding_model': self.config.embedding_model
            },
            'components': {
                'pseudo_note_generator': 'PseudoNoteGenerator',
                'rag_enhancer': 'RAGEnhancer',
                'text_enhancer': 'TextEnhancer'
            }
        }

        if self.config.use_gcs:
            metadata_path = f"{self.config.output_path}/phase2_metadata.json"
        else:
            output_dir = Path(self.config.output_path).expanduser()
            metadata_path = output_dir / "phase2_metadata.json"
            metadata_path = str(metadata_path)

        # Save metadata
        metadata_json = json.dumps(metadata, indent=2)
        if self.config.use_gcs and self.gcs_helper.output_bucket:
            blob = self.gcs_helper.output_bucket.blob(metadata_path)
            blob.upload_from_string(metadata_json)
        else:
            with open(metadata_path, 'w') as f:
                f.write(metadata_json)

        logger.info(f"Saved metadata to: {metadata_path}")


def main():
    """Main entry point for Phase 2"""
    parser = argparse.ArgumentParser(
        description='Phase 2: Enhanced Pseudo-Note Generation and RAG Integration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/Output paths
    parser.add_argument(
        '--input-path',
        type=str,
        required=True,
        help='Path to Phase 1 output directory (local or GCS path)'
    )

    parser.add_argument(
        '--output-path',
        type=str,
        default=None,
        help='Output path for Phase 2 (default: same as input-path with _enhanced suffix)'
    )

    # GCS settings
    parser.add_argument(
        '--gcs-bucket',
        type=str,
        default=None,
        help='GCS bucket name (if using GCS)'
    )

    parser.add_argument(
        '--gcs-project-id',
        type=str,
        default=None,
        help='GCP project ID for requester pays'
    )

    # Processing settings
    parser.add_argument(
        '--max-text-length',
        type=int,
        default=8192,
        help='Maximum text length for Clinical ModernBERT'
    )

    parser.add_argument(
        '--top-k-retrieval',
        type=int,
        default=5,
        help='Number of knowledge documents to retrieve'
    )

    parser.add_argument(
        '--embedding-model',
        type=str,
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='Sentence transformer model for RAG'
    )

    # Sample data option
    parser.add_argument(
        '--use-small-sample',
        action='store_true',
        help='Process small sample files (e.g., train_small.pt) for testing'
    )

    args = parser.parse_args()

    # Create configuration
    config = DataConfig()

    # Set paths
    config.output_path = args.input_path
    if args.output_path:
        # If different output path specified, we'll save there
        # For now, keep same path and add _enhanced to filenames
        pass

    # Set GCS settings
    config.use_gcs = args.gcs_bucket is not None
    config.gcs_bucket = args.gcs_bucket
    config.output_gcs_bucket = args.gcs_bucket
    config.gcs_project_id = args.gcs_project_id

    # Set processing settings
    config.max_text_length = args.max_text_length
    config.top_k_retrieval = args.top_k_retrieval
    config.embedding_model = args.embedding_model

    logger.info("=" * 60)
    logger.info("Phase 2: Enhanced Pseudo-Note Generation and RAG Integration")
    logger.info("=" * 60)
    logger.info(f"Mode: {'GCS' if config.use_gcs else 'Local'}")
    if config.use_gcs:
        logger.info(f"Bucket: {config.gcs_bucket}")
    logger.info(f"Input path: {config.output_path}")
    logger.info(f"Max text length: {config.max_text_length}")
    logger.info(f"RAG top-k: {config.top_k_retrieval}")
    logger.info(f"Use small sample: {args.use_small_sample}")
    logger.info("=" * 60)

    # Create processor
    processor = Phase2Processor(config)

    # Process all splits
    processor.process_all_splits(use_small_sample=args.use_small_sample)

    # Save metadata
    processor.save_metadata()

    logger.info("Phase 2 processing complete!")


if __name__ == "__main__":
    main()
