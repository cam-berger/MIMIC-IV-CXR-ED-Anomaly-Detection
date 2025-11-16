"""
Enhanced RAG Data Adapter

Converts Enhanced RAG format (with attention segments, bbox, RAG-enhanced notes)
to Standard format expected by training scripts.

Enhanced RAG Format:
- image_tensor: [3, 518, 518]
- clinical_data: JSON string with structured clinical fields
- labels: nested dict with disease_labels, bbox_coordinates, severity_scores
- text_input_ids, text_attention_mask: [1, 8192]
- enhanced_note: RAG-augmented clinical note
- attention_segments: dict with clinical_data, knowledge_context, diagnostic_hints

Standard Training Format:
- image: [3, 518, 518]
- clinical_features: [45] tensor
- labels: flat dict with 14 binary disease labels
- text_input_ids, text_attention_mask: [seq_len]
"""

import torch
import json
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


class EnhancedRAGAdapter:
    """Adapter to convert Enhanced RAG format to Standard training format"""

    # Standard clinical feature names (45 features)
    CLINICAL_FEATURE_NAMES = [
        'temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp',
        'pain', 'acuity', 'age', 'gender', 'subject_id', 'los_hours'
    ]

    # CheXpert disease label names (14 classes)
    CHEXPERT_LABELS = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
        'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
        'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
        'Pleural Other', 'Fracture', 'Support Devices'
    ]

    def __init__(self):
        """Initialize the adapter"""
        self.stats = {
            'converted': 0,
            'missing_clinical': 0,
            'missing_labels': 0
        }

    def convert_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a single Enhanced RAG sample to Standard format

        Args:
            sample: Enhanced RAG format sample

        Returns:
            Standard format sample compatible with training scripts
        """
        try:
            converted = {}

            # 1. Image: image_tensor -> image
            if 'image_tensor' in sample:
                converted['image'] = sample['image_tensor']
            else:
                raise ValueError("Missing image_tensor")

            # 2. Text: already in correct format, just squeeze extra dimension
            if 'text_input_ids' in sample:
                # Shape: [1, 8192] -> [8192]
                text_ids = sample['text_input_ids']
                if isinstance(text_ids, torch.Tensor) and text_ids.dim() > 1:
                    converted['text_input_ids'] = text_ids.squeeze(0)
                else:
                    converted['text_input_ids'] = text_ids

            if 'text_attention_mask' in sample:
                # Shape: [1, 8192] -> [8192]
                text_mask = sample['text_attention_mask']
                if isinstance(text_mask, torch.Tensor) and text_mask.dim() > 1:
                    converted['text_attention_mask'] = text_mask.squeeze(0)
                else:
                    converted['text_attention_mask'] = text_mask

            # 3. Clinical data: JSON string -> 45-element tensor
            converted['clinical_features'] = self._parse_clinical_data(
                sample.get('clinical_data', '{}')
            )

            # 4. Labels: nested dict -> flat dict with 14 binary labels
            converted['labels'] = self._parse_labels(
                sample.get('labels', {})
            )

            # 5. Optional metadata
            if 'subject_id' in sample:
                converted['subject_id'] = sample['subject_id']
            if 'study_id' in sample:
                converted['study_id'] = sample['study_id']

            # 6. Store enhanced features for potential future use
            converted['_enhanced'] = {
                'enhanced_note': sample.get('enhanced_note', ''),
                'attention_segments': sample.get('attention_segments', {}),
                'bbox_coordinates': sample.get('labels', {}).get('bbox_coordinates', []),
                'severity_scores': sample.get('labels', {}).get('severity_scores', [])
            }

            self.stats['converted'] += 1
            return converted

        except Exception as e:
            logger.error(f"Error converting sample: {e}")
            raise

    def _parse_clinical_data(self, clinical_json: str) -> torch.Tensor:
        """
        Parse clinical data JSON string to 45-element feature tensor

        Args:
            clinical_json: JSON string with clinical fields

        Returns:
            Tensor of shape [45] with normalized clinical features
        """
        try:
            # Parse JSON
            if isinstance(clinical_json, str):
                clinical_dict = json.loads(clinical_json)
            elif isinstance(clinical_json, dict):
                clinical_dict = clinical_json
            else:
                clinical_dict = {}

            # Extract numeric features
            features = []

            # Vital signs (6 features)
            features.append(self._normalize_temp(clinical_dict.get('temperature')))
            features.append(self._normalize_hr(clinical_dict.get('heartrate')))
            features.append(self._normalize_rr(clinical_dict.get('resprate')))
            features.append(self._normalize_o2(clinical_dict.get('o2sat')))
            features.append(self._normalize_bp(clinical_dict.get('sbp')))
            features.append(self._normalize_bp(clinical_dict.get('dbp')))

            # Pain and acuity (2 features)
            features.append(self._normalize_pain(clinical_dict.get('pain')))
            features.append(self._normalize_acuity(clinical_dict.get('acuity')))

            # Demographics (2 features)
            features.append(self._normalize_age(clinical_dict.get('age')))
            features.append(self._encode_gender(clinical_dict.get('gender')))

            # Patient ID (1 feature - normalized)
            features.append(self._normalize_id(clinical_dict.get('subject_id')))

            # Placeholder for additional features to reach 45
            # (In full implementation, add: medications, labs, allergies, history, etc.)
            remaining = 45 - len(features)
            features.extend([0.0] * remaining)

            return torch.tensor(features, dtype=torch.float32)

        except Exception as e:
            logger.warning(f"Error parsing clinical data: {e}, using zeros")
            self.stats['missing_clinical'] += 1
            return torch.zeros(45, dtype=torch.float32)

    def _parse_labels(self, labels_dict: Dict) -> Dict[str, int]:
        """
        Parse nested labels to flat binary dict

        Args:
            labels_dict: Nested dict with disease_labels, bbox, severity

        Returns:
            Flat dict with 14 binary CheXpert labels
        """
        try:
            # Extract disease_labels list
            disease_labels = labels_dict.get('disease_labels', [])

            # Convert to flat dict
            flat_labels = {}

            if isinstance(disease_labels, list):
                # Assume disease_labels is a list of 14 binary values
                for i, label_name in enumerate(self.CHEXPERT_LABELS):
                    if i < len(disease_labels):
                        # Convert to binary (0 or 1)
                        value = disease_labels[i]
                        if isinstance(value, (int, float)):
                            flat_labels[label_name] = 1 if value > 0 else 0
                        else:
                            flat_labels[label_name] = 0
                    else:
                        flat_labels[label_name] = 0
            else:
                # Fallback: all zeros
                for label_name in self.CHEXPERT_LABELS:
                    flat_labels[label_name] = 0

            return flat_labels

        except Exception as e:
            logger.warning(f"Error parsing labels: {e}, using all zeros")
            self.stats['missing_labels'] += 1
            return {name: 0 for name in self.CHEXPERT_LABELS}

    # Normalization helper functions
    def _normalize_temp(self, value) -> float:
        """Normalize temperature (F) to [0, 1]"""
        if value is None or (isinstance(value, str) and not value.replace('.', '').isdigit()):
            return 0.0
        try:
            temp = float(value)
            if np.isnan(temp):
                return 0.0
            # Normal range: 97-99°F, map to ~0.5, extremes to 0-1
            return np.clip((temp - 95) / 10, 0, 1)
        except:
            return 0.0

    def _normalize_hr(self, value) -> float:
        """Normalize heart rate (bpm) to [0, 1]"""
        if value is None:
            return 0.0
        try:
            hr = float(value)
            if np.isnan(hr):
                return 0.0
            # Normal: 60-100 bpm, map to ~0.5
            return np.clip(hr / 200, 0, 1)
        except:
            return 0.0

    def _normalize_rr(self, value) -> float:
        """Normalize respiratory rate to [0, 1]"""
        if value is None:
            return 0.0
        try:
            rr = float(value)
            if np.isnan(rr):
                return 0.0
            # Normal: 12-20 breaths/min
            return np.clip(rr / 40, 0, 1)
        except:
            return 0.0

    def _normalize_o2(self, value) -> float:
        """Normalize O2 saturation to [0, 1]"""
        if value is None:
            return 0.0
        try:
            o2 = float(value)
            if np.isnan(o2):
                return 0.0
            # Already in percentage, normalize to [0, 1]
            return np.clip(o2 / 100, 0, 1)
        except:
            return 0.0

    def _normalize_bp(self, value) -> float:
        """Normalize blood pressure to [0, 1]"""
        if value is None:
            return 0.0
        try:
            bp = float(value)
            if np.isnan(bp):
                return 0.0
            # Normal range: 80-120 for sbp, 60-80 for dbp
            return np.clip(bp / 200, 0, 1)
        except:
            return 0.0

    def _normalize_pain(self, value) -> float:
        """Normalize pain score (0-10) to [0, 1]"""
        if value is None or value == '':
            return 0.0
        try:
            pain = float(str(value))
            # Check for NaN (from JSON with nan values)
            if np.isnan(pain):
                return 0.0
            return np.clip(pain / 10, 0, 1)
        except:
            return 0.0

    def _normalize_acuity(self, value) -> float:
        """Normalize acuity (1-5) to [0, 1]"""
        if value is None:
            return 0.0
        try:
            acuity = float(value)
            if np.isnan(acuity):
                return 0.0
            # 1 = most critical, 5 = least critical
            return np.clip((6 - acuity) / 5, 0, 1)
        except:
            return 0.0

    def _normalize_age(self, value) -> float:
        """Normalize age to [0, 1]"""
        if value is None:
            return 0.5  # Unknown age -> middle value
        try:
            age = float(value)
            if np.isnan(age):
                return 0.5
            # Max age ~100
            return np.clip(age / 100, 0, 1)
        except:
            return 0.5

    def _encode_gender(self, value) -> float:
        """Encode gender as binary feature"""
        if value is None:
            return 0.5  # Unknown
        value_upper = str(value).upper()
        if value_upper == 'M':
            return 0.0
        elif value_upper == 'F':
            return 1.0
        else:
            return 0.5

    def _normalize_id(self, value) -> float:
        """Normalize patient ID (just for uniqueness feature)"""
        if value is None:
            return 0.0
        try:
            # Use last 4 digits normalized
            id_val = int(value) % 10000
            return id_val / 10000
        except:
            return 0.0

    def convert_batch(self, batch: List[Dict]) -> List[Dict]:
        """Convert a batch of Enhanced RAG samples"""
        return [self.convert_sample(sample) for sample in batch]

    def print_stats(self):
        """Print conversion statistics"""
        logger.info("=" * 60)
        logger.info("Enhanced RAG Adapter Statistics")
        logger.info("=" * 60)
        logger.info(f"Total samples converted: {self.stats['converted']}")
        logger.info(f"Missing clinical data: {self.stats['missing_clinical']}")
        logger.info(f"Missing labels: {self.stats['missing_labels']}")
        logger.info("=" * 60)
