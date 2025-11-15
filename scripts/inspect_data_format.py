#!/usr/bin/env python3
"""
Quick script to inspect the actual format of preprocessed data chunks
"""
import torch
from pathlib import Path
import sys

def inspect_chunk(chunk_path):
    """Inspect a data chunk and print its structure"""
    print("=" * 70)
    print(f"Inspecting: {chunk_path}")
    print("=" * 70)

    # Load chunk
    data = torch.load(chunk_path, map_location='cpu', weights_only=False)

    print(f"\n✓ Loaded successfully")
    print(f"  Type: {type(data)}")
    print(f"  Length: {len(data)} samples")

    if len(data) == 0:
        print("  ⚠️  Chunk is empty!")
        return

    # Inspect first sample
    print(f"\n" + "-" * 70)
    print("First sample structure:")
    print("-" * 70)

    sample = data[0]
    print(f"Sample type: {type(sample)}")

    if isinstance(sample, dict):
        print(f"\nTop-level keys ({len(sample)} total):")
        for key in sorted(sample.keys()):
            value = sample[key]
            if isinstance(value, torch.Tensor):
                print(f"  ✓ {key:30s} → Tensor {value.shape} ({value.dtype})")
            elif isinstance(value, dict):
                subkeys = list(value.keys())
                print(f"  📁 {key:30s} → dict with {len(value)} keys: {subkeys[:5]}")
            elif isinstance(value, list):
                print(f"  📋 {key:30s} → list with {len(value)} items")
            elif isinstance(value, (int, float, str, bool)):
                print(f"  📝 {key:30s} → {type(value).__name__}: {value}")
            else:
                print(f"  ❓ {key:30s} → {type(value).__name__}")

        # Check for nested structures
        print(f"\n" + "-" * 70)
        print("Nested structures:")
        print("-" * 70)

        for key, value in sample.items():
            if isinstance(value, dict):
                print(f"\n{key}:")
                for subkey, subval in sorted(value.items()):
                    if isinstance(subval, torch.Tensor):
                        print(f"  ✓ {subkey:28s} → Tensor {subval.shape} ({subval.dtype})")
                    elif isinstance(subval, (int, float)):
                        print(f"  📝 {subkey:28s} → {type(subval).__name__}: {subval}")
                    else:
                        print(f"  ❓ {subkey:28s} → {type(subval).__name__}")
    else:
        print(f"\n⚠️  Expected dict, got {type(sample)}")
        print(f"Sample: {sample}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        chunk_path = Path(sys.argv[1])
    else:
        # Default to first train chunk
        chunk_path = Path("/home/dev/Documents/Portfolio/MIMIC_Data/physionet.org/files/train_test_small_2/train_chunk_000003.pt")

    if not chunk_path.exists():
        print(f"❌ File not found: {chunk_path}")
        sys.exit(1)

    inspect_chunk(chunk_path)
