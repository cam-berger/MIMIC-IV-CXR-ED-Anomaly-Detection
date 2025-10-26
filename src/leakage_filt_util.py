#!/usr/bin/env python3
"""
Advanced filtering utilities to prevent diagnosis leakage in clinical notes
"""

import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import spacy
from transformers import AutoTokenizer, AutoModel
import torch

class DiagnosisLeakageFilter:
    """Advanced filtering to remove diagnosis-related information from clinical data"""
    
    def __init__(self, use_nlp_model: bool = True):
        """
        Initialize the filter with optional NLP model for semantic similarity
        
        Args:
            use_nlp_model: Whether to use BioBERT for semantic filtering
        """
        self.use_nlp_model = use_nlp_model
        
        if use_nlp_model:
            # Load BioBERT for semantic similarity
            self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
            self.model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")
            self.model.eval()
        
        # Load spaCy for medical entity recognition (optional)
        self.nlp = None
        try:
            self.nlp = spacy.load("en_core_sci_md")  # SciSpacy medical model
            print("Loaded SciSpacy medical model")
        except:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                print("Loaded standard spaCy model (SciSpacy not available)")
            except:
                print("WARNING: No spaCy models found. Entity extraction will be disabled.")
                print("Install with: python -m spacy download en_core_web_sm")
        
        # Define comprehensive leakage patterns
        self.diagnosis_patterns = self._build_diagnosis_patterns()
        
    def _build_diagnosis_patterns(self) -> Dict[str, List[str]]:
        """Build comprehensive patterns for each CheXpert label"""
        
        patterns = {
            'Atelectasis': [
                r'atelecta\w*', r'collapse\w*\s*lung', r'volume\s*loss',
                r'subsegmental\s*collapse', r'linear\s*opacity'
            ],
            
            'Cardiomegaly': [
                r'cardiomegal\w*', r'enlarge\w*\s*heart', r'cardiac\s*enlarge\w*',
                r'heart\s*size', r'cardiac\s*silhouette', r'CTR\s*>', 
                r'cardiothoracic\s*ratio'
            ],
            
            'Consolidation': [
                r'consolidat\w*', r'air\s*?space\s*opacit\w*', r'alveolar\s*infiltrat\w*',
                r'lobar\s*pneumonia', r'dense\s*opacit\w*'
            ],
            
            'Edema': [
                r'edema', r'pulmonary\s*edema', r'vascular\s*congestion',
                r'kerley\s*[ab]\s*lines', r'interstitial\s*marking',
                r'fluid\s*overload', r'CHF', r'congestive\s*heart'
            ],
            
            'Pleural Effusion': [
                r'effusion', r'pleural\s*fluid', r'costophrenic\s*angle',
                r'hemothorax', r'hydrothorax', r'blunt\w*\s*costophrenic',
                r'meniscus\s*sign', r'layering\s*fluid'
            ],
            
            'Pneumonia': [
                r'pneumonia', r'infectious\s*process', r'infiltrat\w*',
                r'airspace\s*disease', r'focal\s*opacit\w*',
                r'bronchopneumonia', r'multilobar', r'ARDS'
            ],
            
            'Pneumothorax': [
                r'pneumothora\w*', r'collapsed?\s*lung', r'air\s*in\s*pleural',
                r'tension\s*pneumo', r'visceral\s*pleural\s*line',
                r'deep\s*sulcus'
            ],
            
            # General radiology terms to filter
            'Radiology_Terms': [
                r'chest\s*(?:x-?ray|radiograph|film|cxr)', r'pa\s*and\s*lateral',
                r'portable\s*(?:chest|cxr)', r'imaging\s*(?:show|reveal|demonstrate)',
                r'radiolog\w*\s*(?:finding|report|impression)',
                r'(?:increased|decreased)\s*(?:opacity|density|lucency)',
                r'ground[- ]glass', r'reticular', r'nodular', r'miliary',
                r'hilar', r'mediastin\w*', r'parenchyma\w*', r'interstitial',
                r'radiograph\w*\s*(?:normal|clear|unremarkable)',
                r'comparison\s*(?:with|to)\s*prior', r'interval\s*change'
            ]
        }
        
        return patterns
    
    def filter_clinical_note(self, 
                           note_text: str, 
                           positive_findings: List[str],
                           aggressive: bool = True) -> Tuple[str, Dict]:
        """
        Filter clinical note to remove diagnosis-related information
        
        Args:
            note_text: Original clinical note text
            positive_findings: List of positive CheXpert labels for this case
            aggressive: If True, use more aggressive filtering
            
        Returns:
            Tuple of (filtered_text, filtering_stats)
        """
        if pd.isna(note_text) or not note_text:
            return "", {'removed_sentences': 0, 'total_sentences': 0}
        
        original_length = len(note_text)
        sentences = note_text.split('.')
        total_sentences = len(sentences)
        
        filtered_sentences = []
        removed_count = 0
        removal_reasons = []
        
        for sentence in sentences:
            should_remove = False
            reason = ""
            
            # Check for direct pattern matches
            for finding in positive_findings + ['Radiology_Terms']:
                if finding in self.diagnosis_patterns:
                    for pattern in self.diagnosis_patterns[finding]:
                        if re.search(pattern, sentence.lower()):
                            should_remove = True
                            reason = f"Pattern match: {finding}"
                            break
            
            # Additional filtering for aggressive mode
            if aggressive and not should_remove:
                # Remove sentences mentioning imaging modalities
                imaging_terms = ['imaging', 'scan', 'ct', 'mri', 'ultrasound', 'echo']
                if any(term in sentence.lower() for term in imaging_terms):
                    should_remove = True
                    reason = "Imaging reference"
                
                # Remove sentences with diagnostic conclusions
                diagnostic_phrases = [
                    'consistent with', 'suggestive of', 'concerning for',
                    'evidence of', 'no evidence', 'ruled out', 'cannot exclude',
                    'differential includes', 'likely represents', 'compatible with'
                ]
                if any(phrase in sentence.lower() for phrase in diagnostic_phrases):
                    should_remove = True
                    reason = "Diagnostic language"
            
            # Use NLP model for semantic similarity (if enabled)
            if self.use_nlp_model and not should_remove:
                similarity_score = self._check_semantic_similarity(sentence, positive_findings)
                if similarity_score > 0.7:  # Threshold for semantic similarity
                    should_remove = True
                    reason = f"Semantic similarity: {similarity_score:.2f}"
            
            if should_remove:
                removed_count += 1
                removal_reasons.append(reason)
            else:
                filtered_sentences.append(sentence)
        
        filtered_text = '. '.join(filtered_sentences).strip()
        
        # Final safety check - remove if too much was filtered (might indicate heavy contamination)
        if len(filtered_text) < 0.2 * original_length and aggressive:
            filtered_text = "[Note heavily redacted due to diagnosis leakage]"
        
        stats = {
            'original_length': original_length,
            'filtered_length': len(filtered_text),
            'removed_sentences': removed_count,
            'total_sentences': total_sentences,
            'removal_rate': removed_count / max(total_sentences, 1),
            'removal_reasons': removal_reasons[:5]  # Sample of reasons
        }
        
        return filtered_text, stats
    
    def _check_semantic_similarity(self, text: str, findings: List[str]) -> float:
        """
        Check semantic similarity between text and diagnosis terms using BioBERT
        
        Args:
            text: Text to check
            findings: List of diagnosis terms
            
        Returns:
            Maximum similarity score (0-1)
        """
        if not self.use_nlp_model:
            return 0.0
        
        try:
            # Encode text
            text_inputs = self.tokenizer(text, return_tensors="pt", 
                                        truncation=True, max_length=512)
            with torch.no_grad():
                text_embedding = self.model(**text_inputs).last_hidden_state.mean(dim=1)
            
            max_similarity = 0.0
            
            # Check similarity with each finding
            for finding in findings:
                finding_text = f"{finding} in chest x-ray imaging"
                finding_inputs = self.tokenizer(finding_text, return_tensors="pt",
                                              truncation=True, max_length=512)
                with torch.no_grad():
                    finding_embedding = self.model(**finding_inputs).last_hidden_state.mean(dim=1)
                
                # Cosine similarity
                similarity = torch.cosine_similarity(text_embedding, finding_embedding)
                max_similarity = max(max_similarity, similarity.item())
            
            return max_similarity
            
        except Exception as e:
            print(f"Error in semantic similarity: {e}")
            return 0.0
    
    def filter_lab_values(self, 
                         lab_df: pd.DataFrame,
                         positive_findings: List[str]) -> pd.DataFrame:
        """
        Filter lab values that might directly indicate chest pathology
        
        Args:
            lab_df: DataFrame with lab values
            positive_findings: List of positive CheXpert labels
            
        Returns:
            Filtered DataFrame
        """
        if lab_df.empty:
            return lab_df
        
        # Labs that might directly indicate chest pathology
        chest_related_labs = {
            'Pneumonia': [50889, 50893, 50912],  # CRP, Procalcitonin, WBC
            'Pleural Effusion': [50883, 50976],  # LDH, Protein (pleural)
            'Edema': [50963, 50970, 50971],  # BNP, NT-proBNP, Troponin
            'Pneumothorax': [50817, 50818]  # Blood gas values
        }
        
        # Get lab IDs to exclude
        exclude_ids = []
        for finding in positive_findings:
            if finding in chest_related_labs:
                exclude_ids.extend(chest_related_labs[finding])
        
        # Filter out problematic labs
        filtered_df = lab_df[~lab_df['itemid'].isin(exclude_ids)]
        
        return filtered_df
    
    def filter_medications(self,
                          med_df: pd.DataFrame,
                          positive_findings: List[str]) -> pd.DataFrame:
        """
        Filter medications that directly relate to chest pathology treatment
        
        Args:
            med_df: DataFrame with medications
            positive_findings: List of positive CheXpert labels
            
        Returns:
            Filtered DataFrame
        """
        if med_df.empty:
            return med_df
        
        # Medications that might leak diagnosis
        diagnosis_meds = {
            'Pneumonia': ['antibiotic', 'azithromycin', 'ceftriaxone', 'levofloxacin'],
            'Edema': ['furosemide', 'lasix', 'diuretic', 'spironolactone'],
            'Pleural Effusion': ['thoracentesis', 'pleurodesis'],
            'Pneumothorax': ['chest tube', 'pleurodesis']
        }
        
        # Build exclusion patterns
        exclude_patterns = []
        for finding in positive_findings:
            if finding in diagnosis_meds:
                exclude_patterns.extend(diagnosis_meds[finding])
        
        # Filter medications
        if exclude_patterns:
            pattern = '|'.join(exclude_patterns)
            mask = ~med_df['drug'].str.lower().str.contains(pattern, na=False)
            filtered_df = med_df[mask]
        else:
            filtered_df = med_df
        
        return filtered_df
    
    def validate_filtering(self, 
                          original_text: str,
                          filtered_text: str,
                          findings: List[str]) -> Dict:
        """
        Validate that filtering successfully removed diagnosis information
        
        Args:
            original_text: Original clinical text
            filtered_text: Filtered clinical text
            findings: List of positive findings
            
        Returns:
            Validation metrics
        """
        metrics = {
            'text_reduction_rate': 1 - (len(filtered_text) / max(len(original_text), 1)),
            'findings_mentioned_original': 0,
            'findings_mentioned_filtered': 0,
            'radiology_terms_original': 0,
            'radiology_terms_filtered': 0
        }
        
        # Check for finding mentions
        for finding in findings:
            if finding in self.diagnosis_patterns:
                for pattern in self.diagnosis_patterns[finding]:
                    if re.search(pattern, original_text.lower()):
                        metrics['findings_mentioned_original'] += 1
                    if re.search(pattern, filtered_text.lower()):
                        metrics['findings_mentioned_filtered'] += 1
        
        # Check for radiology terms
        for pattern in self.diagnosis_patterns.get('Radiology_Terms', []):
            if re.search(pattern, original_text.lower()):
                metrics['radiology_terms_original'] += 1
            if re.search(pattern, filtered_text.lower()):
                metrics['radiology_terms_filtered'] += 1
        
        # Calculate effectiveness
        if metrics['findings_mentioned_original'] > 0:
            metrics['finding_removal_rate'] = 1 - (
                metrics['findings_mentioned_filtered'] / 
                metrics['findings_mentioned_original']
            )
        else:
            metrics['finding_removal_rate'] = 1.0
        
        metrics['is_valid'] = (
            metrics['findings_mentioned_filtered'] == 0 and
            metrics['radiology_terms_filtered'] == 0
        )
        
        return metrics


def create_temporal_features(clinical_data: Dict, 
                            cxr_time: datetime,
                            window_hours: int = 24) -> Dict:
    """
    Create temporal features from clinical data relative to CXR time
    
    Args:
        clinical_data: Dictionary of clinical data
        cxr_time: Time of chest X-ray
        window_hours: Time window to consider
        
    Returns:
        Dictionary of temporal features
    """
    features = {
        'hours_since_admission': None,
        'notes_count_24h': 0,
        'notes_count_48h': 0,
        'lab_count_24h': 0,
        'vital_trends': {},
        'medication_changes': []
    }
    
    # Calculate time-based features
    window_start = cxr_time - timedelta(hours=window_hours)
    
    # Process notes temporally
    if 'notes' in clinical_data:
        for _, note in clinical_data['notes'].iterrows():
            note_time = pd.to_datetime(note.get('charttime'))
            if pd.notna(note_time):
                hours_diff = (cxr_time - note_time).total_seconds() / 3600
                if 0 <= hours_diff <= 24:
                    features['notes_count_24h'] += 1
                elif 0 <= hours_diff <= 48:
                    features['notes_count_48h'] += 1
    
    # Calculate vital sign trends
    if 'vitals' in clinical_data:
        vital_types = {
            220045: 'heart_rate',
            220050: 'sbp',
            220210: 'resp_rate',
            220277: 'spo2'
        }
        
        for itemid, vital_name in vital_types.items():
            vital_values = clinical_data['vitals'][
                clinical_data['vitals']['itemid'] == itemid
            ]['value'].values
            
            if len(vital_values) >= 2:
                # Calculate trend (increasing/decreasing/stable)
                recent = vital_values[-5:].mean() if len(vital_values) >= 5 else vital_values[-1]
                older = vital_values[-10:-5].mean() if len(vital_values) >= 10 else vital_values[0]
                
                change = (recent - older) / max(older, 1)
                if abs(change) < 0.05:
                    trend = 'stable'
                elif change > 0:
                    trend = 'increasing'
                else:
                    trend = 'decreasing'
                    
                features['vital_trends'][vital_name] = {
                    'trend': trend,
                    'change_percent': change * 100,
                    'last_value': vital_values[-1]
                }
    
    return features


# Example usage
if __name__ == "__main__":
    # Initialize filter
    filter = DiagnosisLeakageFilter(use_nlp_model=False)  # Set True if BioBERT installed
    
    # Example clinical note with diagnosis information
    example_note = """
    Patient admitted with shortness of breath and fever. 
    Chest x-ray shows bilateral infiltrates consistent with pneumonia.
    Started on antibiotics. Consolidation noted in right lower lobe.
    Will continue respiratory support and monitor closely.
    """
    
    positive_findings = ['Pneumonia', 'Consolidation']
    
    # Filter the note
    filtered_note, stats = filter.filter_clinical_note(
        example_note, 
        positive_findings,
        aggressive=True
    )
    
    print("Original Note:")
    print(example_note)
    print("\nFiltered Note:")
    print(filtered_note)
    print("\nFiltering Statistics:")
    print(stats)
    
    # Validate filtering
    validation = filter.validate_filtering(example_note, filtered_note, positive_findings)
    print("\nValidation Results:")
    print(validation)