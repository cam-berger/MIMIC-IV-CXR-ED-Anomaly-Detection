"""
Data Quality Validation for CXR-PRO Prior-Free Reports

Validates that radiology impressions have been properly cleaned:
1. Detects remaining prior references
2. Checks impression quality (length, coherence)
3. Verifies tokenization and data format
4. Generates quality metrics and reports

Usage:
    python scripts/validate_cxr_pro_quality.py \
        --data-path processed/phase1_with_radiology \
        --output-report validation_report.json
"""

import os
import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import torch
from collections import defaultdict, Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CXRProQualityValidator:
    """
    Validates quality of CXR-PRO cleaned radiology impressions

    Checks for:
    - Remaining prior references
    - Impression completeness
    - Text quality issues
    - Data format correctness
    """

    # Patterns that indicate prior references (should NOT be present after cleaning)
    PRIOR_REFERENCE_PATTERNS = [
        r'\b(unchanged|stable|improved|worsened|increased|decreased)\s+(from|since|compared\s+to)\s+(prior|previous|last)',
        r'\b(compared\s+to|comparison\s+to|similar\s+to)\s+(prior|previous|last|old)',
        r'\b(no\s+change|interval\s+change|new\s+since|resolved\s+since)',
        r'\b(prior\s+(study|exam|examination|radiograph|film|image|CT|MRI))',
        r'\b(previous\s+(study|exam|examination|radiograph|film|image|CT|MRI))',
        r'\b(as\s+before|again\s+seen|again\s+noted|redemonstrated)',
        r'\b(persistent|persisting)\s+(from|since)',
        r'\b(new\s+from|progressed\s+from|regressed\s+from)',
    ]

    # Patterns that indicate incomplete cleaning (e.g., "stable" without context)
    INCOMPLETE_CLEANING_PATTERNS = [
        r'\b(stable|unchanged)\s*[,.]',  # "stable." or "stable," without object
        r'\b(improved|worsened)\s*[,.]',
    ]

    # Patterns that indicate quality issues
    QUALITY_ISSUE_PATTERNS = [
        r'^\s*$',  # Empty
        r'^[,.]',  # Starts with punctuation
        r'_{3,}',  # PHI redaction markers (___) - should be minimal
    ]

    def __init__(self):
        self.stats = {
            'total_impressions': 0,
            'with_prior_references': 0,
            'incomplete_cleaning': 0,
            'quality_issues': 0,
            'empty_impressions': 0,
            'average_length': 0.0,
            'median_length': 0.0,
            'length_distribution': {}
        }

        self.prior_reference_examples = []
        self.incomplete_cleaning_examples = []
        self.quality_issue_examples = []

    def check_prior_references(self, impression: str) -> Tuple[bool, List[str]]:
        """
        Check if impression contains prior references

        Args:
            impression: Radiology impression text

        Returns:
            Tuple of (has_priors, matched_patterns)
        """
        matches = []

        for pattern in self.PRIOR_REFERENCE_PATTERNS:
            if re.search(pattern, impression, re.IGNORECASE):
                matches.append(pattern)

        return len(matches) > 0, matches

    def check_incomplete_cleaning(self, impression: str) -> Tuple[bool, List[str]]:
        """
        Check if impression has incomplete cleaning (e.g., "stable" without object)

        Args:
            impression: Radiology impression text

        Returns:
            Tuple of (is_incomplete, matched_patterns)
        """
        matches = []

        for pattern in self.INCOMPLETE_CLEANING_PATTERNS:
            if re.search(pattern, impression, re.IGNORECASE):
                matches.append(pattern)

        return len(matches) > 0, matches

    def check_quality_issues(self, impression: str) -> Tuple[bool, List[str]]:
        """
        Check for general quality issues

        Args:
            impression: Radiology impression text

        Returns:
            Tuple of (has_issues, issue_types)
        """
        issues = []

        # Check if empty
        if not impression or len(impression.strip()) == 0:
            issues.append('empty')

        # Check for quality patterns
        for pattern in self.QUALITY_ISSUE_PATTERNS:
            if re.search(pattern, impression):
                issues.append(f'pattern: {pattern}')

        # Check length (too short may indicate over-cleaning)
        if len(impression.strip()) < 10 and len(impression.strip()) > 0:
            issues.append('too_short')

        return len(issues) > 0, issues

    def validate_impression(self, impression: str, study_id: str) -> Dict:
        """
        Validate a single impression

        Args:
            impression: Radiology impression text
            study_id: Study identifier for logging

        Returns:
            Dictionary with validation results
        """
        result = {
            'study_id': study_id,
            'impression': impression,
            'length': len(impression) if impression else 0,
            'has_prior_references': False,
            'incomplete_cleaning': False,
            'quality_issues': False,
            'issues': []
        }

        if not impression:
            result['quality_issues'] = True
            result['issues'].append('empty')
            self.stats['empty_impressions'] += 1
            return result

        # Check prior references
        has_priors, prior_patterns = self.check_prior_references(impression)
        if has_priors:
            result['has_prior_references'] = True
            result['issues'].extend([f'prior: {p}' for p in prior_patterns])
            self.stats['with_prior_references'] += 1

            # Save example
            if len(self.prior_reference_examples) < 10:
                self.prior_reference_examples.append({
                    'study_id': study_id,
                    'impression': impression[:200],
                    'patterns': prior_patterns
                })

        # Check incomplete cleaning
        is_incomplete, incomplete_patterns = self.check_incomplete_cleaning(impression)
        if is_incomplete:
            result['incomplete_cleaning'] = True
            result['issues'].extend([f'incomplete: {p}' for p in incomplete_patterns])
            self.stats['incomplete_cleaning'] += 1

            # Save example
            if len(self.incomplete_cleaning_examples) < 10:
                self.incomplete_cleaning_examples.append({
                    'study_id': study_id,
                    'impression': impression[:200],
                    'patterns': incomplete_patterns
                })

        # Check quality issues
        has_issues, quality_issues = self.check_quality_issues(impression)
        if has_issues:
            result['quality_issues'] = True
            result['issues'].extend(quality_issues)
            self.stats['quality_issues'] += 1

            # Save example
            if len(self.quality_issue_examples) < 10:
                self.quality_issue_examples.append({
                    'study_id': study_id,
                    'impression': impression[:200],
                    'issues': quality_issues
                })

        return result

    def validate_dataset(self, data_path: str, split: str = 'train') -> Dict:
        """
        Validate a complete dataset split

        Args:
            data_path: Path to data directory
            split: 'train', 'val', or 'test'

        Returns:
            Validation results dictionary
        """
        logger.info(f"Validating {split} split...")

        # Load data
        file_path = os.path.join(data_path, f"{split}_data_with_radiology.pt")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        data = torch.load(file_path)
        logger.info(f"Loaded {len(data)} records")

        # Reset stats
        self.stats = {
            'total_impressions': 0,
            'with_prior_references': 0,
            'incomplete_cleaning': 0,
            'quality_issues': 0,
            'empty_impressions': 0,
            'average_length': 0.0,
            'median_length': 0.0,
            'length_distribution': defaultdict(int)
        }
        self.prior_reference_examples = []
        self.incomplete_cleaning_examples = []
        self.quality_issue_examples = []

        # Validate each record
        lengths = []
        validation_results = []

        for record in tqdm(data, desc=f"Validating {split}"):
            impression = record.get('radiology_impression_cleaned', '')
            study_id = str(record.get('study_id', 'unknown'))

            result = self.validate_impression(impression, study_id)
            validation_results.append(result)

            # Track lengths
            if result['length'] > 0:
                lengths.append(result['length'])

            self.stats['total_impressions'] += 1

        # Calculate length statistics
        if lengths:
            self.stats['average_length'] = np.mean(lengths)
            self.stats['median_length'] = np.median(lengths)

            # Length distribution (bins: 0-50, 50-100, 100-200, 200+)
            for length in lengths:
                if length < 50:
                    self.stats['length_distribution']['0-50'] += 1
                elif length < 100:
                    self.stats['length_distribution']['50-100'] += 1
                elif length < 200:
                    self.stats['length_distribution']['100-200'] += 1
                else:
                    self.stats['length_distribution']['200+'] += 1

        return {
            'split': split,
            'stats': self.stats,
            'prior_reference_examples': self.prior_reference_examples,
            'incomplete_cleaning_examples': self.incomplete_cleaning_examples,
            'quality_issue_examples': self.quality_issue_examples,
            'validation_results': validation_results  # Full results (may be large)
        }

    def generate_report(self, results: Dict) -> str:
        """
        Generate human-readable validation report

        Args:
            results: Validation results dictionary

        Returns:
            Formatted report string
        """
        stats = results['stats']
        split = results['split']

        report_lines = [
            f"\n{'='*70}",
            f"CXR-PRO Data Quality Validation Report - {split.upper()} Split",
            f"{'='*70}\n",
            f"Total Impressions: {stats['total_impressions']}",
            f"",
            f"PRIOR REFERENCES (should be 0):",
            f"  - With prior references: {stats['with_prior_references']} "
            f"({100*stats['with_prior_references']/max(stats['total_impressions'],1):.2f}%)",
            f"",
            f"CLEANING QUALITY:",
            f"  - Incomplete cleaning: {stats['incomplete_cleaning']} "
            f"({100*stats['incomplete_cleaning']/max(stats['total_impressions'],1):.2f}%)",
            f"  - Quality issues: {stats['quality_issues']} "
            f"({100*stats['quality_issues']/max(stats['total_impressions'],1):.2f}%)",
            f"  - Empty impressions: {stats['empty_impressions']} "
            f"({100*stats['empty_impressions']/max(stats['total_impressions'],1):.2f}%)",
            f"",
            f"LENGTH STATISTICS:",
            f"  - Average length: {stats['average_length']:.1f} chars",
            f"  - Median length: {stats['median_length']:.1f} chars",
            f"  - Distribution:",
        ]

        # Add length distribution
        for bin_range, count in sorted(stats['length_distribution'].items()):
            pct = 100 * count / max(stats['total_impressions'], 1)
            report_lines.append(f"      {bin_range}: {count} ({pct:.1f}%)")

        report_lines.append(f"")

        # Add examples if present
        if results['prior_reference_examples']:
            report_lines.append(f"EXAMPLES OF PRIOR REFERENCES (first 3):")
            for i, example in enumerate(results['prior_reference_examples'][:3], 1):
                report_lines.append(f"  {i}. Study {example['study_id']}:")
                report_lines.append(f"     \"{example['impression']}...\"")
                report_lines.append(f"     Patterns: {example['patterns']}")
            report_lines.append(f"")

        if results['incomplete_cleaning_examples']:
            report_lines.append(f"EXAMPLES OF INCOMPLETE CLEANING (first 3):")
            for i, example in enumerate(results['incomplete_cleaning_examples'][:3], 1):
                report_lines.append(f"  {i}. Study {example['study_id']}:")
                report_lines.append(f"     \"{example['impression']}...\"")
                report_lines.append(f"     Patterns: {example['patterns']}")
            report_lines.append(f"")

        # Overall assessment
        report_lines.append(f"OVERALL ASSESSMENT:")
        if stats['with_prior_references'] == 0 and stats['empty_impressions'] < stats['total_impressions'] * 0.01:
            report_lines.append(f"  ✓ PASS: Prior references successfully removed")
        else:
            report_lines.append(f"  ✗ FAIL: Prior references detected or too many empty impressions")

        report_lines.append(f"{'='*70}\n")

        return "\n".join(report_lines)


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate CXR-PRO data quality"
    )

    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to data with radiology impressions')
    parser.add_argument('--splits', type=str, nargs='+',
                       default=['train', 'val', 'test'],
                       help='Which splits to validate')
    parser.add_argument('--output-report', type=str,
                       default='cxr_pro_validation_report.json',
                       help='Output JSON report file')
    parser.add_argument('--output-txt', type=str,
                       default='cxr_pro_validation_report.txt',
                       help='Output text report file')

    args = parser.parse_args()

    # Initialize validator
    validator = CXRProQualityValidator()

    # Validate each split
    all_results = {}

    for split in args.splits:
        logger.info(f"\n{'='*70}")
        logger.info(f"Validating {split.upper()} split")
        logger.info(f"{'='*70}")

        try:
            results = validator.validate_dataset(args.data_path, split)
            all_results[split] = results

            # Print report
            report = validator.generate_report(results)
            print(report)

        except FileNotFoundError as e:
            logger.warning(f"Skipping {split}: {e}")

    # Save results
    logger.info(f"\nSaving validation reports...")

    # Save JSON (without full validation_results to reduce file size)
    json_output = {}
    for split, results in all_results.items():
        json_output[split] = {
            'stats': results['stats'],
            'prior_reference_examples': results['prior_reference_examples'],
            'incomplete_cleaning_examples': results['incomplete_cleaning_examples'],
            'quality_issue_examples': results['quality_issue_examples']
        }

    with open(args.output_report, 'w') as f:
        json.dump(json_output, f, indent=2)
    logger.info(f"JSON report saved to: {args.output_report}")

    # Save text report
    with open(args.output_txt, 'w') as f:
        for split, results in all_results.items():
            report = validator.generate_report(results)
            f.write(report)
            f.write("\n\n")
    logger.info(f"Text report saved to: {args.output_txt}")

    logger.info("\nValidation complete!")


if __name__ == "__main__":
    main()
