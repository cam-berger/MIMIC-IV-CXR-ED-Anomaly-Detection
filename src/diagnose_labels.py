#!/usr/bin/env python3
"""
Diagnostic script to analyze CheXpert label attachment in preprocessed MIMIC-CXR data

This script helps identify and diagnose label attachment issues in your preprocessed data.
It can work with both pickle files and PyTorch tensor files.
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import logging

# Try to import torch (optional)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available, .pt files cannot be analyzed")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Expected CheXpert labels from MIMIC-CXR-JPG
EXPECTED_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
    'Pneumonia', 'Pneumothorax', 'Support Devices'
]

class LabelDiagnostics:
    """Comprehensive diagnostics for CheXpert label attachment"""
    
    def __init__(self):
        self.stats = {
            'total_records': 0,
            'records_with_labels': 0,
            'records_with_disease_labels': 0,
            'records_with_positive_findings': 0,
            'label_distribution': {label: 0 for label in EXPECTED_LABELS},
            'issues_found': []
        }
        
    def load_data(self, file_path: str) -> List[Dict]:
        """Load preprocessed data from pickle or PyTorch file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Loading data from: {file_path}")
        
        if file_path.suffix == '.pkl':
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        elif file_path.suffix == '.pt' and TORCH_AVAILABLE:
            data = torch.load(file_path, map_location='cpu')
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.info(f"Loaded {len(data)} records")
        return data
    
    def analyze_record(self, record: Dict, idx: int) -> Dict:
        """Analyze a single record for label issues"""
        issues = []
        record_stats = {
            'has_labels': False,
            'has_disease_labels': False,
            'positive_findings': [],
            'label_format': None,
            'study_id': record.get('study_id', 'unknown'),
            'subject_id': record.get('subject_id', 'unknown')
        }
        
        # Check if labels field exists
        if 'labels' not in record:
            issues.append(f"Record {idx} (study_id={record_stats['study_id']}): Missing 'labels' field entirely")
            return record_stats
        
        record_stats['has_labels'] = True
        labels = record['labels']
        
        # Check label structure
        if isinstance(labels, dict):
            # Check for disease labels
            if 'disease_labels' in labels:
                record_stats['has_disease_labels'] = True
                disease_labels = labels['disease_labels']
                
                if isinstance(disease_labels, list):
                    record_stats['positive_findings'] = disease_labels
                    record_stats['label_format'] = 'list'
                else:
                    issues.append(f"Record {idx}: 'disease_labels' is not a list")
            
            # Check for label array
            if 'label_array' in labels:
                array = labels['label_array']
                if hasattr(array, 'shape'):
                    if array.shape[0] != len(EXPECTED_LABELS):
                        issues.append(f"Record {idx}: label_array has wrong shape {array.shape}, expected ({len(EXPECTED_LABELS)},)")
                    record_stats['label_format'] = 'array'
            
            # Check for individual label fields (alternative format)
            individual_labels = []
            for label in EXPECTED_LABELS:
                if label in labels:
                    if labels[label] == 1.0:
                        individual_labels.append(label)
                    record_stats['label_format'] = 'individual'
            
            if individual_labels and not record_stats['positive_findings']:
                record_stats['positive_findings'] = individual_labels
        
        elif isinstance(labels, list):
            # Old format: just a list of positive findings
            record_stats['positive_findings'] = labels
            record_stats['label_format'] = 'simple_list'
            issues.append(f"Record {idx}: Using old label format (list instead of dict)")
        
        else:
            issues.append(f"Record {idx}: Unexpected label type: {type(labels)}")
        
        # Check if completely empty
        if record_stats['has_disease_labels'] and not record_stats['positive_findings']:
            # This is OK - might be a "No Finding" case
            pass
        
        self.stats['issues_found'].extend(issues)
        return record_stats
    
    def analyze_dataset(self, data: List[Dict]) -> None:
        """Analyze entire dataset for label statistics"""
        logger.info("Analyzing dataset for label issues...")
        
        self.stats['total_records'] = len(data)
        
        all_formats = set()
        sample_records = {'good': None, 'bad': None}
        
        for idx, record in enumerate(data):
            record_stats = self.analyze_record(record, idx)
            
            if record_stats['has_labels']:
                self.stats['records_with_labels'] += 1
            
            if record_stats['has_disease_labels']:
                self.stats['records_with_disease_labels'] += 1
            
            if record_stats['positive_findings']:
                self.stats['records_with_positive_findings'] += 1
                for finding in record_stats['positive_findings']:
                    if finding in self.stats['label_distribution']:
                        self.stats['label_distribution'][finding] += 1
            
            if record_stats['label_format']:
                all_formats.add(record_stats['label_format'])
            
            # Save sample records
            if record_stats['has_disease_labels'] and record_stats['positive_findings']:
                sample_records['good'] = record
            elif not record_stats['has_labels']:
                sample_records['bad'] = record
        
        self.stats['label_formats'] = list(all_formats)
        self.stats['sample_records'] = sample_records
    
    def print_report(self) -> None:
        """Print comprehensive diagnostic report"""
        print("\n" + "=" * 80)
        print("MIMIC-CXR LABEL DIAGNOSTICS REPORT")
        print("=" * 80)
        
        # Overall statistics
        print("\n📊 OVERALL STATISTICS:")
        print(f"  Total records: {self.stats['total_records']}")
        print(f"  Records with 'labels' field: {self.stats['records_with_labels']} "
              f"({100*self.stats['records_with_labels']/max(self.stats['total_records'],1):.1f}%)")
        print(f"  Records with 'disease_labels': {self.stats['records_with_disease_labels']} "
              f"({100*self.stats['records_with_disease_labels']/max(self.stats['total_records'],1):.1f}%)")
        print(f"  Records with positive findings: {self.stats['records_with_positive_findings']} "
              f"({100*self.stats['records_with_positive_findings']/max(self.stats['total_records'],1):.1f}%)")
        
        # Label formats detected
        print(f"\n📋 Label formats detected: {', '.join(self.stats['label_formats']) if self.stats['label_formats'] else 'NONE'}")
        
        # Diagnosis
        print("\n🔍 DIAGNOSIS:")
        if self.stats['records_with_disease_labels'] == 0:
            print("  ❌ CRITICAL: No records have disease labels attached!")
            print("     → CheXpert labels are NOT being loaded during preprocessing")
            print("     → Apply the fix in fix_chexpert_labels.py")
        elif self.stats['records_with_disease_labels'] < self.stats['total_records'] * 0.9:
            print("  ⚠️  WARNING: Some records are missing disease labels")
            print(f"     → {self.stats['total_records'] - self.stats['records_with_disease_labels']} records have no labels")
        else:
            print("  ✅ Labels appear to be properly attached")
        
        # Label distribution
        print("\n📈 LABEL DISTRIBUTION (positive findings only):")
        if self.stats['records_with_positive_findings'] > 0:
            sorted_labels = sorted(self.stats['label_distribution'].items(), 
                                 key=lambda x: x[1], reverse=True)
            for label, count in sorted_labels:
                if count > 0:
                    prevalence = 100 * count / self.stats['total_records']
                    print(f"  {label:30s}: {count:6d} ({prevalence:5.1f}%)")
        else:
            print("  No positive findings found in any records")
        
        # Expected vs actual prevalence
        print("\n📊 EXPECTED VS ACTUAL PREVALENCE:")
        expected_prevalences = {
            'Support Devices': (45, 55),
            'Lung Opacity': (35, 45),
            'Pleural Effusion': (25, 35),
            'Atelectasis': (25, 35),
            'Cardiomegaly': (20, 30),
            'No Finding': (15, 25)
        }
        
        for label, (min_exp, max_exp) in expected_prevalences.items():
            actual = 100 * self.stats['label_distribution'].get(label, 0) / max(self.stats['total_records'], 1)
            if actual < min_exp * 0.5:  # Less than half expected minimum
                status = "❌ MUCH LOWER than expected"
            elif actual < min_exp:
                status = "⚠️  Lower than expected"
            elif actual > max_exp:
                status = "⚠️  Higher than expected"
            else:
                status = "✅ Within expected range"
            
            print(f"  {label:30s}: {actual:5.1f}% (expected {min_exp}-{max_exp}%) {status}")
        
        # Sample record structure
        if self.stats['sample_records']['good']:
            print("\n📄 SAMPLE GOOD RECORD STRUCTURE:")
            record = self.stats['sample_records']['good']
            print(f"  study_id: {record.get('study_id')}")
            print(f"  subject_id: {record.get('subject_id')}")
            print(f"  Top-level keys: {list(record.keys())}")
            if 'labels' in record:
                print(f"  Label keys: {list(record['labels'].keys())}")
                if 'disease_labels' in record['labels']:
                    findings = record['labels']['disease_labels'][:3]  # First 3
                    print(f"  Sample findings: {findings}...")
        
        if self.stats['sample_records']['bad']:
            print("\n📄 SAMPLE PROBLEMATIC RECORD:")
            record = self.stats['sample_records']['bad']
            print(f"  study_id: {record.get('study_id')}")
            print(f"  Has 'labels' field: {'labels' in record}")
            if 'labels' in record:
                print(f"  Labels content: {record['labels']}")
        
        # Issues summary
        if self.stats['issues_found']:
            print(f"\n⚠️  ISSUES FOUND ({len(self.stats['issues_found'])} total):")
            # Show first 5 issues
            for issue in self.stats['issues_found'][:5]:
                print(f"  - {issue}")
            if len(self.stats['issues_found']) > 5:
                print(f"  ... and {len(self.stats['issues_found']) - 5} more issues")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if self.stats['records_with_disease_labels'] == 0:
            print("  1. CheXpert labels are not being loaded - this is your primary issue")
            print("  2. Apply the patch in preprocessing_patch.py to fix the preprocessing pipeline")
            print("  3. Ensure mimic-cxr-2.0.0-chexpert.csv.gz is accessible")
            print("  4. Rerun preprocessing with the fixed pipeline")
            print("  5. Use this diagnostic script again to verify the fix worked")
        elif self.stats['records_with_positive_findings'] / max(self.stats['total_records'], 1) < 0.4:
            print("  1. Label prevalence seems too low")
            print("  2. Check if CheXpert CSV is being loaded correctly")
            print("  3. Verify study_id matching between records and labels")
            print("  4. Consider if you're only loading positive labels (missing negatives/uncertains)")
        else:
            print("  ✅ Labels appear to be properly configured!")
            print("  - Continue with model training")
            print("  - Consider using class weights for imbalanced labels")
        
        print("\n" + "=" * 80)
    
    def save_report(self, output_path: str) -> None:
        """Save detailed report to JSON file"""
        report = {
            'statistics': self.stats,
            'diagnosis': self.get_diagnosis(),
            'recommendations': self.get_recommendations()
        }
        
        # Remove sample records from saved report (might be large)
        report['statistics'] = {k: v for k, v in self.stats.items() 
                               if k != 'sample_records'}
        
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to: {output_path}")
    
    def get_diagnosis(self) -> str:
        """Get diagnosis summary"""
        if self.stats['records_with_disease_labels'] == 0:
            return "CRITICAL: No CheXpert labels found - preprocessing pipeline needs fix"
        elif self.stats['records_with_disease_labels'] < self.stats['total_records'] * 0.9:
            return "WARNING: Partial label attachment - some records missing labels"
        else:
            return "OK: Labels appear to be properly attached"
    
    def get_recommendations(self) -> List[str]:
        """Get list of recommendations"""
        recs = []
        
        if self.stats['records_with_disease_labels'] == 0:
            recs.extend([
                "Apply preprocessing_patch.py to fix label attachment",
                "Ensure mimic-cxr-2.0.0-chexpert.csv.gz is accessible",
                "Rerun preprocessing pipeline",
                "Validate with this diagnostic script"
            ])
        elif self.stats['records_with_positive_findings'] / max(self.stats['total_records'], 1) < 0.4:
            recs.extend([
                "Verify CheXpert CSV loading",
                "Check study_id matching logic",
                "Consider loading all label values (not just positives)"
            ])
        else:
            recs.extend([
                "Proceed with model training",
                "Use class weights for imbalanced labels",
                "Consider data augmentation for rare findings"
            ])
        
        return recs


def main():
    parser = argparse.ArgumentParser(
        description='Diagnose CheXpert label attachment in preprocessed MIMIC-CXR data'
    )
    
    parser.add_argument('data_path', type=str,
                       help='Path to preprocessed data file (.pkl or .pt)')
    parser.add_argument('--save-report', type=str,
                       help='Path to save JSON report')
    parser.add_argument('--check-all-splits', action='store_true',
                       help='Check train, val, and test splits in directory')
    
    args = parser.parse_args()
    
    if args.check_all_splits:
        # Check all splits in directory
        data_dir = Path(args.data_path)
        for split in ['train', 'val', 'test']:
            for ext in ['.pkl', '.pt']:
                file_path = data_dir / f"{split}_data{ext}"
                if file_path.exists():
                    print(f"\n{'='*80}")
                    print(f"Analyzing {split} split: {file_path}")
                    print('='*80)
                    
                    diagnostics = LabelDiagnostics()
                    try:
                        data = diagnostics.load_data(file_path)
                        diagnostics.analyze_dataset(data)
                        diagnostics.print_report()
                        
                        if args.save_report:
                            report_path = Path(args.save_report).parent / f"{split}_label_report.json"
                            diagnostics.save_report(report_path)
                    except Exception as e:
                        print(f"Error analyzing {file_path}: {e}")
    else:
        # Check single file
        diagnostics = LabelDiagnostics()
        
        try:
            data = diagnostics.load_data(args.data_path)
            diagnostics.analyze_dataset(data)
            diagnostics.print_report()
            
            if args.save_report:
                diagnostics.save_report(args.save_report)
            
        except Exception as e:
            logger.error(f"Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
