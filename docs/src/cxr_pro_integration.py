"""
CXR-PRO Integration: Remove Prior References from Radiology Reports

Integrates CXR-PRO methodology to remove hallucinated references to priors
from MIMIC-CXR radiology reports using the GILBERT model.

Based on: "Improving Radiology Report Generation Systems By Removing
Hallucinated References to Non-existent Priors"
GitHub: rajpurkarlab/CXR-ReDonE
HuggingFace: rajpurkarlab/gilbert

GILBERT: Fine-tuned BioBERT for NER-based prior removal
- Token-level classification: REMOVE vs KEEP
- Removes: "unchanged", "stable", "new from prior", etc.
- Preserves: Current findings and clinical observations
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from tqdm import tqdm
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline
)

# Google Cloud Storage
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    storage = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class CXRProConfig:
    """Configuration for CXR-PRO integration"""
    # Model settings
    gilbert_model: str = "rajpurkarlab/gilbert"  # HuggingFace model
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 16

    # Data paths
    mimic_cxr_reports_path: str = "files/mimic-cxr-jpg/2.1.0/reports"
    mimic_cxr_metadata_path: str = "files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz"

    # CXR-PRO pre-processed data (if available)
    cxr_pro_train_path: Optional[str] = None  # mimic_train_impressions.csv
    cxr_pro_test_path: Optional[str] = None   # mimic_test_impressions.csv

    # Output settings
    output_path: str = "processed/cxr_pro_cleaned"
    save_intermediate: bool = True

    # GCS settings
    use_gcs: bool = False
    gcs_bucket: Optional[str] = None
    gcs_project_id: Optional[str] = None


class GILBERTModel:
    """GILBERT: Token-level prior reference removal using BioBERT"""

    def __init__(self, config: CXRProConfig):
        self.config = config
        self.device = config.device

        logger.info(f"Loading GILBERT model from HuggingFace: {config.gilbert_model}")
        logger.info(f"Using device: {self.device}")

        # Load GILBERT model
        self.tokenizer = AutoTokenizer.from_pretrained(config.gilbert_model)
        self.model = AutoModelForTokenClassification.from_pretrained(config.gilbert_model)
        self.model.to(self.device)
        self.model.eval()

        # Create NER pipeline
        self.ner_pipeline = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            aggregation_strategy="simple"
        )

        logger.info("GILBERT model loaded successfully")

    def remove_prior_references(self, text: str) -> Tuple[str, Dict]:
        """
        Remove prior references from a radiology report using GILBERT

        Args:
            text: Input radiology report text

        Returns:
            Tuple of (cleaned_text, metadata)
            - cleaned_text: Report with prior references removed
            - metadata: Statistics about removal (tokens removed, etc.)
        """
        if not text or len(text.strip()) == 0:
            return "", {"tokens_removed": 0, "original_length": 0}

        # Run GILBERT NER
        predictions = self.ner_pipeline(text)

        # Track what to remove
        spans_to_remove = []
        for pred in predictions:
            # GILBERT classifies tokens as REMOVE or KEEP
            # Label IDs: typically "LABEL_1" = REMOVE, "LABEL_0" = KEEP
            if "REMOVE" in pred.get("entity_group", "") or pred.get("score", 0) > 0.5:
                spans_to_remove.append((pred["start"], pred["end"]))

        # Merge overlapping spans
        spans_to_remove = self._merge_spans(spans_to_remove)

        # Remove spans from text
        cleaned_text = self._remove_spans(text, spans_to_remove)

        # Clean up extra whitespace
        cleaned_text = self._clean_whitespace(cleaned_text)

        metadata = {
            "original_length": len(text),
            "cleaned_length": len(cleaned_text),
            "tokens_removed": len(spans_to_remove),
            "removal_rate": 1.0 - (len(cleaned_text) / max(len(text), 1))
        }

        return cleaned_text, metadata

    def _merge_spans(self, spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Merge overlapping or adjacent spans"""
        if not spans:
            return []

        spans = sorted(spans, key=lambda x: x[0])
        merged = [spans[0]]

        for current in spans[1:]:
            previous = merged[-1]
            if current[0] <= previous[1] + 1:  # Overlapping or adjacent
                merged[-1] = (previous[0], max(previous[1], current[1]))
            else:
                merged.append(current)

        return merged

    def _remove_spans(self, text: str, spans: List[Tuple[int, int]]) -> str:
        """Remove character spans from text"""
        if not spans:
            return text

        result = []
        last_end = 0

        for start, end in spans:
            result.append(text[last_end:start])
            last_end = end

        result.append(text[last_end:])
        return "".join(result)

    def _clean_whitespace(self, text: str) -> str:
        """Clean up extra whitespace and formatting"""
        # Remove multiple spaces
        text = " ".join(text.split())

        # Fix spacing around punctuation
        text = text.replace(" ,", ",").replace(" .", ".")
        text = text.replace("( ", "(").replace(" )", ")")

        # Remove leading/trailing whitespace
        text = text.strip()

        return text


class CXRProIntegrator:
    """
    Integrates CXR-PRO methodology into the existing pipeline

    Two modes:
    1. Use pre-processed CXR-PRO data (if available)
    2. Apply GILBERT to MIMIC-CXR reports (process from scratch)
    """

    def __init__(self, config: CXRProConfig):
        self.config = config

        # Initialize GCS if needed
        if config.use_gcs and GCS_AVAILABLE:
            self.storage_client = storage.Client(project=config.gcs_project_id)
            self.bucket = self.storage_client.bucket(config.gcs_bucket)
        else:
            self.storage_client = None
            self.bucket = None

        # Initialize GILBERT model
        self.gilbert = GILBERTModel(config)

        # Statistics
        self.stats = {
            "total_reports": 0,
            "reports_processed": 0,
            "reports_with_priors": 0,
            "total_tokens_removed": 0,
            "average_removal_rate": 0.0
        }

    def load_mimic_cxr_metadata(self) -> pd.DataFrame:
        """Load MIMIC-CXR metadata"""
        logger.info("Loading MIMIC-CXR metadata...")

        if self.config.use_gcs:
            path = f"{self.config.mimic_cxr_metadata_path}"
            blob = self.bucket.blob(path)
            content = blob.download_as_bytes()

            from io import BytesIO
            import gzip

            if path.endswith('.gz'):
                content = gzip.decompress(content)

            df = pd.read_csv(BytesIO(content))
        else:
            path = os.path.expanduser(self.config.mimic_cxr_metadata_path)
            df = pd.read_csv(path)

        logger.info(f"Loaded {len(df)} records from MIMIC-CXR metadata")
        return df

    def extract_impression_from_report(self, report_path: str) -> Optional[str]:
        """
        Extract IMPRESSION section from MIMIC-CXR report

        MIMIC-CXR reports have sections:
        - FINDINGS
        - IMPRESSION (this is what we want)
        """
        try:
            if self.config.use_gcs:
                blob = self.bucket.blob(report_path)
                content = blob.download_as_text()
            else:
                with open(report_path, 'r') as f:
                    content = f.read()

            # Extract IMPRESSION section
            lines = content.split('\n')
            impression_lines = []
            in_impression = False

            for line in lines:
                if line.strip().startswith('IMPRESSION:'):
                    in_impression = True
                    continue
                elif in_impression:
                    if line.strip() and line[0].isupper() and ':' in line:
                        # New section started
                        break
                    impression_lines.append(line)

            impression = '\n'.join(impression_lines).strip()
            return impression if impression else None

        except Exception as e:
            logger.warning(f"Failed to extract impression from {report_path}: {e}")
            return None

    def process_reports(self,
                       study_ids: List[str],
                       report_paths: Dict[str, str]) -> pd.DataFrame:
        """
        Process MIMIC-CXR reports to remove prior references

        Args:
            study_ids: List of study IDs to process
            report_paths: Mapping of study_id -> report_path

        Returns:
            DataFrame with columns: study_id, subject_id, impression,
                                   cleaned_impression, metadata
        """
        results = []

        logger.info(f"Processing {len(study_ids)} reports with GILBERT...")

        for study_id in tqdm(study_ids, desc="Removing priors"):
            report_path = report_paths.get(study_id)
            if not report_path:
                continue

            # Extract impression
            impression = self.extract_impression_from_report(report_path)
            if not impression:
                continue

            # Remove prior references using GILBERT
            cleaned_impression, metadata = self.gilbert.remove_prior_references(impression)

            # Update statistics
            self.stats["total_reports"] += 1
            self.stats["reports_processed"] += 1
            if metadata["tokens_removed"] > 0:
                self.stats["reports_with_priors"] += 1
                self.stats["total_tokens_removed"] += metadata["tokens_removed"]

            results.append({
                "study_id": study_id,
                "impression_original": impression,
                "impression_cleaned": cleaned_impression,
                "tokens_removed": metadata["tokens_removed"],
                "removal_rate": metadata["removal_rate"]
            })

        # Calculate average removal rate
        if self.stats["reports_processed"] > 0:
            self.stats["average_removal_rate"] = (
                self.stats["total_tokens_removed"] / self.stats["reports_processed"]
            )

        df = pd.DataFrame(results)
        logger.info(f"Processed {len(df)} reports successfully")
        logger.info(f"Reports with priors removed: {self.stats['reports_with_priors']} "
                   f"({100*self.stats['reports_with_priors']/max(len(df),1):.1f}%)")

        return df

    def load_cxr_pro_preprocessed(self, split: str = "train") -> pd.DataFrame:
        """
        Load pre-processed CXR-PRO data (if available)

        Args:
            split: "train" or "test"

        Returns:
            DataFrame with cleaned impressions
        """
        if split == "train":
            path = self.config.cxr_pro_train_path
        else:
            path = self.config.cxr_pro_test_path

        if not path:
            raise ValueError(f"CXR-PRO {split} data path not configured")

        logger.info(f"Loading pre-processed CXR-PRO {split} data from {path}")

        if self.config.use_gcs:
            blob = self.bucket.blob(path)
            content = blob.download_as_bytes()
            from io import BytesIO
            df = pd.read_csv(BytesIO(content))
        else:
            df = pd.read_csv(path)

        logger.info(f"Loaded {len(df)} records from CXR-PRO {split} set")
        return df

    def save_results(self, df: pd.DataFrame, output_name: str):
        """Save cleaned reports"""
        os.makedirs(self.config.output_path, exist_ok=True)
        output_file = os.path.join(self.config.output_path, f"{output_name}.csv")

        df.to_csv(output_file, index=False)
        logger.info(f"Saved results to {output_file}")

        # Save statistics
        stats_file = os.path.join(self.config.output_path, f"{output_name}_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        logger.info(f"Saved statistics to {stats_file}")


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description="CXR-PRO: Remove prior references from radiology reports")

    # Mode selection
    parser.add_argument('--mode', choices=['preprocess', 'load_preprocessed'],
                       default='preprocess',
                       help='Use GILBERT to process reports or load pre-processed CXR-PRO data')

    # Data paths
    parser.add_argument('--mimic-cxr-path', type=str,
                       help='Path to MIMIC-CXR data')
    parser.add_argument('--cxr-pro-train', type=str,
                       help='Path to CXR-PRO mimic_train_impressions.csv')
    parser.add_argument('--cxr-pro-test', type=str,
                       help='Path to CXR-PRO mimic_test_impressions.csv')
    parser.add_argument('--output-path', type=str, default='processed/cxr_pro_cleaned',
                       help='Output directory')

    # GCS settings
    parser.add_argument('--use-gcs', action='store_true',
                       help='Use Google Cloud Storage')
    parser.add_argument('--gcs-bucket', type=str,
                       help='GCS bucket name')
    parser.add_argument('--gcs-project-id', type=str,
                       help='GCP project ID')

    # Model settings
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for GILBERT processing')
    parser.add_argument('--device', type=str,
                       choices=['cuda', 'cpu', 'auto'],
                       default='auto',
                       help='Device for model inference')

    args = parser.parse_args()

    # Configure
    config = CXRProConfig(
        cxr_pro_train_path=args.cxr_pro_train,
        cxr_pro_test_path=args.cxr_pro_test,
        output_path=args.output_path,
        use_gcs=args.use_gcs,
        gcs_bucket=args.gcs_bucket,
        gcs_project_id=args.gcs_project_id,
        batch_size=args.batch_size,
        device=args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    )

    # Initialize integrator
    integrator = CXRProIntegrator(config)

    if args.mode == 'load_preprocessed':
        # Load pre-processed CXR-PRO data
        logger.info("Loading pre-processed CXR-PRO data...")
        train_df = integrator.load_cxr_pro_preprocessed('train')
        test_df = integrator.load_cxr_pro_preprocessed('test')

        # Save to output
        integrator.save_results(train_df, "cxr_pro_train")
        integrator.save_results(test_df, "cxr_pro_test")

    else:
        # Process MIMIC-CXR reports with GILBERT
        logger.info("Processing MIMIC-CXR reports with GILBERT...")

        # Load metadata
        metadata_df = integrator.load_mimic_cxr_metadata()

        # Get report paths (you'll need to implement this based on your data structure)
        # This is a placeholder - adapt to your actual data structure
        study_ids = metadata_df['study_id'].unique()[:100]  # Start with 100 for testing
        report_paths = {}  # Map study_id -> report_path

        # Process reports
        results_df = integrator.process_reports(study_ids, report_paths)

        # Save results
        integrator.save_results(results_df, "mimic_cxr_cleaned")

    logger.info("CXR-PRO integration complete!")
    logger.info(f"Statistics: {integrator.stats}")


if __name__ == "__main__":
    main()
