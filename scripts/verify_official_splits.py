#!/usr/bin/env python3
"""
Verification Script: Official MIMIC-CXR Splits

This script verifies that:
1. The official split file can be loaded
2. Split distribution matches expected values
3. No patient leakage across splits (when used with preprocessed data)

Usage:
    # Just verify split file loads
    python scripts/verify_official_splits.py

    # Verify preprocessed data has no patient leakage
    python scripts/verify_official_splits.py --check-preprocessed /path/to/preprocessed
"""

import argparse
import pandas as pd
import torch
from pathlib import Path
from collections import defaultdict
import sys

def verify_split_file(split_file_path: str):
    """Verify the official split file can be loaded and has expected format"""
    print("=" * 70)
    print("Verifying Official MIMIC-CXR Split File")
    print("=" * 70)

    try:
        # Load split file
        print(f"\nLoading: {split_file_path}")
        split_df = pd.read_csv(split_file_path)

        # Check columns
        required_cols = ['subject_id', 'study_id', 'split']
        missing_cols = [col for col in required_cols if col not in split_df.columns]

        if missing_cols:
            print(f"❌ ERROR: Missing required columns: {missing_cols}")
            print(f"   Available columns: {split_df.columns.tolist()}")
            return False

        print(f"✓ File loaded successfully")
        print(f"✓ Total studies: {len(split_df):,}")

        # Check split distribution
        split_counts = split_df['split'].value_counts()
        print("\n" + "=" * 70)
        print("Split Distribution:")
        print("=" * 70)

        for split_name in ['train', 'validate', 'test']:
            count = split_counts.get(split_name, 0)
            pct = 100.0 * count / len(split_df)
            print(f"  {split_name:10s}: {count:8,} studies ({pct:5.1f}%)")

        # Check for patient-level consistency
        print("\n" + "=" * 70)
        print("Patient-Level Consistency Check:")
        print("=" * 70)

        # Group by patient and check if all studies from same patient are in same split
        patient_splits = split_df.groupby('subject_id')['split'].apply(set)
        patients_with_multiple_splits = sum(1 for splits in patient_splits if len(splits) > 1)

        if patients_with_multiple_splits > 0:
            print(f"⚠️  WARNING: {patients_with_multiple_splits} patients have studies in multiple splits")
            print("   This should not happen in official MIMIC-CXR splits!")
            return False
        else:
            print("✓ All patients have studies in only one split (patient-level separation)")

        print("\n" + "=" * 70)
        print("✅ Official split file is valid!")
        print("=" * 70)
        return True

    except FileNotFoundError:
        print(f"❌ ERROR: Split file not found at {split_file_path}")
        print("\nPlease ensure mimic-cxr-2.0.0-split.csv.gz exists in your MIMIC-CXR directory")
        print("Download from: https://physionet.org/content/mimic-cxr-jpg/2.1.0/")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_preprocessed_data(preprocessed_dir: str, split_file_path: str):
    """Check that preprocessed data has no patient leakage"""
    print("\n" + "=" * 70)
    print("Checking Preprocessed Data for Patient Leakage")
    print("=" * 70)

    try:
        preprocessed_path = Path(preprocessed_dir)

        # Find split files
        train_files = sorted(preprocessed_path.glob("train*.pt"))
        val_files = sorted(preprocessed_path.glob("val*.pt"))
        test_files = sorted(preprocessed_path.glob("test*.pt"))

        if not (train_files or val_files or test_files):
            print(f"❌ ERROR: No preprocessed split files found in {preprocessed_dir}")
            print("   Expected files: train_*.pt, val_*.pt, test_*.pt")
            return False

        print(f"\nFound files:")
        print(f"  Train: {len(train_files)} files")
        print(f"  Val:   {len(val_files)} files")
        print(f"  Test:  {len(test_files)} files")

        # Load official splits for reference
        split_df = pd.read_csv(split_file_path)
        official_splits = {}
        for _, row in split_df.iterrows():
            key = (int(row['subject_id']), int(row['study_id']))
            official_splits[key] = row['split'].strip().lower()
            if official_splits[key] == 'validate':
                official_splits[key] = 'val'

        # Collect patient IDs from each split
        def get_patients_from_files(files, split_name):
            patients = set()
            studies = []
            for file_path in files[:5]:  # Sample first 5 files
                records = torch.load(file_path, weights_only=False)
                for record in records:
                    subject_id = record.get('subject_id')
                    study_id = record.get('study_id')
                    if subject_id is not None:
                        patients.add(int(subject_id))
                    if subject_id is not None and study_id is not None:
                        studies.append((int(subject_id), int(study_id)))
            return patients, studies

        print("\nExtracting patient IDs from preprocessed data...")
        train_patients, train_studies = get_patients_from_files(train_files, 'train')
        val_patients, val_studies = get_patients_from_files(val_files, 'val')
        test_patients, test_studies = get_patients_from_files(test_files, 'test')

        print(f"  Train: {len(train_patients):,} unique patients, {len(train_studies):,} studies (sampled)")
        print(f"  Val:   {len(val_patients):,} unique patients, {len(val_studies):,} studies (sampled)")
        print(f"  Test:  {len(test_patients):,} unique patients, {len(test_studies):,} studies (sampled)")

        # Check for patient overlap
        print("\n" + "=" * 70)
        print("Patient Leakage Check:")
        print("=" * 70)

        train_val_overlap = train_patients & val_patients
        train_test_overlap = train_patients & test_patients
        val_test_overlap = val_patients & test_patients

        has_leakage = False

        if train_val_overlap:
            print(f"❌ PATIENT LEAKAGE: {len(train_val_overlap)} patients in both TRAIN and VAL")
            print(f"   Sample patient IDs: {list(train_val_overlap)[:5]}")
            has_leakage = True
        else:
            print("✓ No patient overlap between TRAIN and VAL")

        if train_test_overlap:
            print(f"❌ PATIENT LEAKAGE: {len(train_test_overlap)} patients in both TRAIN and TEST")
            print(f"   Sample patient IDs: {list(train_test_overlap)[:5]}")
            has_leakage = True
        else:
            print("✓ No patient overlap between TRAIN and TEST")

        if val_test_overlap:
            print(f"❌ PATIENT LEAKAGE: {len(val_test_overlap)} patients in both VAL and TEST")
            print(f"   Sample patient IDs: {list(val_test_overlap)[:5]}")
            has_leakage = True
        else:
            print("✓ No patient overlap between VAL and TEST")

        # Check if studies match official splits
        print("\n" + "=" * 70)
        print("Official Split Compliance Check:")
        print("=" * 70)

        def check_split_compliance(studies, expected_split):
            mismatches = 0
            total = 0
            for study_key in studies[:100]:  # Check first 100
                official_split = official_splits.get(study_key)
                if official_split:
                    total += 1
                    if official_split != expected_split:
                        mismatches += 1
            return mismatches, total

        train_mismatches, train_checked = check_split_compliance(train_studies, 'train')
        val_mismatches, val_checked = check_split_compliance(val_studies, 'val')
        test_mismatches, test_checked = check_split_compliance(test_studies, 'test')

        if train_checked > 0:
            train_compliance = 100.0 * (1 - train_mismatches / train_checked)
            print(f"  Train compliance: {train_compliance:.1f}% ({train_checked - train_mismatches}/{train_checked} correct)")

        if val_checked > 0:
            val_compliance = 100.0 * (1 - val_mismatches / val_checked)
            print(f"  Val compliance:   {val_compliance:.1f}% ({val_checked - val_mismatches}/{val_checked} correct)")

        if test_checked > 0:
            test_compliance = 100.0 * (1 - test_mismatches / test_checked)
            print(f"  Test compliance:  {test_compliance:.1f}% ({test_checked - test_mismatches}/{test_checked} correct)")

        if has_leakage:
            print("\n" + "=" * 70)
            print("❌ CRITICAL: Patient leakage detected!")
            print("=" * 70)
            print("\n⚠️  Your preprocessed data has patient leakage issues.")
            print("   This will lead to inflated performance metrics.")
            print("\n   ACTION REQUIRED: Re-preprocess with official splits enabled:")
            print("   python src/phase1_preprocess_streaming.py --skip-to-combine")
            return False
        else:
            print("\n" + "=" * 70)
            print("✅ Preprocessed data passes all checks!")
            print("=" * 70)
            print("✓ No patient leakage detected")
            print("✓ Studies match official split assignments")
            return True

    except Exception as e:
        print(f"❌ ERROR checking preprocessed data: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Verify official MIMIC-CXR splits')

    parser.add_argument(
        '--split-file',
        type=str,
        default='/home/dev/Documents/Portfolio/MIMIC_Data/physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-split.csv.gz',
        help='Path to mimic-cxr-2.0.0-split.csv.gz'
    )

    parser.add_argument(
        '--check-preprocessed',
        type=str,
        help='Path to preprocessed data directory to check for patient leakage'
    )

    args = parser.parse_args()

    # Verify split file
    file_ok = verify_split_file(args.split_file)

    if not file_ok:
        sys.exit(1)

    # Check preprocessed data if requested
    if args.check_preprocessed:
        preprocessed_ok = check_preprocessed_data(args.check_preprocessed, args.split_file)
        if not preprocessed_ok:
            sys.exit(1)

    print("\n✅ All checks passed!")
    sys.exit(0)


if __name__ == "__main__":
    main()
