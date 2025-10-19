#!/usr/bin/env python3
"""
Download MIMIC-CXR-JPG images directly to S3 using IMAGE_FILENAMES
This script uses the official IMAGE_FILENAMES file to download images efficiently
Fixed version that mimics the working wget command
"""

import os
import sys
import requests
from requests.auth import HTTPBasicAuth
import boto3
from pathlib import Path
from tqdm import tqdm
import getpass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random

# Configuration
S3_BUCKET = 'bergermimiciv'
S3_PREFIX = 'mimic-cxr-jpg/2.1.0'
AWS_REGION = 'us-west-2'
AWS_PROFILE = 'default'

# IMPORTANT: Base URL without /files/ since IMAGE_FILENAMES already contains full paths
PHYSIONET_BASE_URL = 'https://physionet.org/files/mimic-cxr-jpg/2.1.0'

# Local path to IMAGE_FILENAMES (already downloaded)
IMAGE_FILENAMES_PATH = Path.home() / 'Documents/Portfolio/MIMIC_Data/physionet.org/files/mimic-cxr-jpg/2.1.0/IMAGE_FILENAMES'

# Download settings
MAX_WORKERS = 4  # Reduced to avoid rate limiting
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5  # seconds


def load_image_filenames() -> list:
    """Load the IMAGE_FILENAMES file from local disk"""
    print(f"\n📥 Loading IMAGE_FILENAMES from local disk...")
    print(f"   Path: {IMAGE_FILENAMES_PATH}")

    try:
        if not IMAGE_FILENAMES_PATH.exists():
            print(f"✗ ERROR: IMAGE_FILENAMES not found at: {IMAGE_FILENAMES_PATH}")
            print(f"\nPlease ensure the file exists at this location.")
            sys.exit(1)

        # Read file list (one filename per line)
        with open(IMAGE_FILENAMES_PATH, 'r') as f:
            filenames = [line.strip() for line in f if line.strip()]

        print(f"✓ Found {len(filenames):,} images in IMAGE_FILENAMES")
        
        # Debug: Print first few filenames to understand the format
        if filenames:
            print("\n📋 Sample filenames from IMAGE_FILENAMES:")
            for i, fname in enumerate(filenames[:5]):
                print(f"   {i+1}. {fname}")
            
            # Check format
            if filenames[0].startswith('files/'):
                print("   ℹ️  Paths include 'files/' prefix")
            else:
                print("   ℹ️  Paths do NOT include 'files/' prefix")
            print()
        
        return filenames

    except Exception as e:
        print(f"✗ Failed to read IMAGE_FILENAMES: {e}")
        sys.exit(1)


def create_session(username: str, password: str) -> requests.Session:
    """Create a requests session with proper authentication"""
    session = requests.Session()
    session.auth = HTTPBasicAuth(username, password)
    
    # Add headers that wget uses
    session.headers.update({
        'User-Agent': 'Wget/1.21.2',  # Mimic wget user agent
        'Accept': '*/*',
        'Accept-Encoding': 'identity',  # Avoid compression issues
        'Connection': 'Keep-Alive',
    })
    
    return session


def test_single_download(image_path: str, session: requests.Session) -> tuple:
    """Test downloading a single image to verify credentials and URL format"""
    try:
        # The IMAGE_FILENAMES contains paths like "files/p10/p10000032/..."
        # The base URL already includes up to /2.1.0/
        # So we just append the path from IMAGE_FILENAMES
        url = f"{PHYSIONET_BASE_URL}/{image_path}"
        
        print(f"\n🔍 Testing download...")
        print(f"   Image path from file: {image_path}")
        print(f"   Full URL: {url}")
        
        response = session.get(url, timeout=30)
        
        print(f"   Response status: {response.status_code}")
        print(f"   Content-Type: {response.headers.get('Content-Type', 'Unknown')}")
        
        if response.status_code == 200:
            print(f"   ✓ Success! Downloaded {len(response.content):,} bytes")
            return True, response.content
        else:
            print(f"   ✗ Failed with status: {response.status_code}")
            if response.status_code == 401:
                print("   ⚠️  Authentication failed - check credentials")
            elif response.status_code == 403:
                print("   ⚠️  Access forbidden - check MIMIC-CXR-JPG access rights")
            elif response.status_code == 404:
                print("   ⚠️  File not found - URL structure issue")
            return False, None
        
    except Exception as e:
        print(f"   ✗ Test failed: {e}")
        return False, None


def download_image_to_s3(image_path: str, session: requests.Session,
                         s3_bucket: str, s3_prefix: str, s3_client) -> tuple:
    """
    Download a single image and upload directly to S3
    
    Args:
        image_path: Path from IMAGE_FILENAMES (e.g., "files/p10/p10000032/...")
        session: Authenticated requests session
        s3_bucket: S3 bucket name
        s3_prefix: S3 prefix/folder
        s3_client: Boto3 S3 client

    Returns:
        (success, s3_key, error_message)
    """
    for attempt in range(RETRY_ATTEMPTS):
        try:
            # Construct URL exactly as wget does
            url = f"{PHYSIONET_BASE_URL}/{image_path}"

            # Download from PhysioNet
            response = session.get(url, timeout=60)
            
            if response.status_code == 200:
                # Construct S3 key - preserve the full path structure
                s3_key = f"{s3_prefix}/{image_path}"
                
                # Upload to S3
                s3_client.put_object(
                    Bucket=s3_bucket,
                    Key=s3_key,
                    Body=response.content,
                    ContentType='image/jpeg'
                )
                
                return (True, s3_key, None)
            else:
                error_msg = f"HTTP {response.status_code}"
                if response.status_code in [401, 403, 404]:
                    # Don't retry for auth/permission errors
                    return (False, image_path, error_msg)
                elif attempt < RETRY_ATTEMPTS - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                else:
                    return (False, image_path, error_msg)

        except requests.exceptions.Timeout:
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
                continue
            else:
                return (False, image_path, "Timeout")
        
        except Exception as e:
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(RETRY_DELAY)
                continue
            else:
                return (False, image_path, str(e))

    return (False, image_path, "Max retries exceeded")


def download_images_parallel(image_paths: list, username: str, password: str, sample_size: int = None):
    """Download images in parallel and upload to S3"""

    # Sample if requested
    if sample_size and sample_size < len(image_paths):
        print(f"\n📊 Sampling {sample_size:,} images from {len(image_paths):,} total...")
        # Use random sampling for better distribution
        random.seed(42)  # For reproducibility
        image_paths = random.sample(image_paths, sample_size)

    total_images = len(image_paths)
    
    # Create authenticated session
    print(f"\n🔐 Creating authenticated session...")
    session = create_session(username, password)
    
    # Test with a single image first
    print(f"\n🧪 Testing download with first image...")
    test_image = image_paths[0]
    success, content = test_single_download(test_image, session)
    
    if not success:
        print("\n❌ Test download failed!")
        print("\nTroubleshooting steps:")
        print("1. Verify your credentials are correct")
        print("2. Confirm you have access to MIMIC-CXR-JPG (not just MIMIC-CXR)")
        print("3. Try the wget command that works:")
        print(f'   grep "p10000032" IMAGE_FILENAMES | wget -r -N -c -np --user {username} --ask-password -i - --base=https://physionet.org/files/mimic-cxr-jpg/2.1.0/')
        sys.exit(1)
    
    print(f"\n✓ Test successful!")
    
    # Test S3 access
    print(f"\n🔍 Testing S3 access...")
    try:
        boto_session = boto3.Session(profile_name=AWS_PROFILE)
        s3_client = boto_session.client('s3', region_name=AWS_REGION)
        s3_client.head_bucket(Bucket=S3_BUCKET)
        print(f"✓ S3 bucket accessible: {S3_BUCKET}")
    except Exception as e:
        print(f"❌ Cannot access S3 bucket {S3_BUCKET}: {e}")
        print("Please check your AWS credentials and bucket permissions")
        sys.exit(1)
    
    print(f"\n🚀 Starting download of {total_images:,} images...")
    print(f"   Workers: {MAX_WORKERS}")
    print(f"   Destination: s3://{S3_BUCKET}/{S3_PREFIX}/")
    print()

    # Prepare download tasks
    tasks = []
    for image_path in image_paths:
        tasks.append((image_path, session, S3_BUCKET, S3_PREFIX, s3_client))

    # Download with progress bar
    success_count = 0
    failed_count = 0
    failed_images = []
    
    # Create a new session for each worker to avoid connection issues
    def worker_download(args):
        image_path, _, s3_bucket, s3_prefix, s3_client = args
        # Create a new session for this worker
        worker_session = create_session(username, password)
        return download_image_to_s3(image_path, worker_session, s3_bucket, s3_prefix, s3_client)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(worker_download, task): task[0] for task in tasks}

        with tqdm(total=total_images, desc="Downloading", unit="img") as pbar:
            for future in as_completed(futures):
                success, key, error = future.result()

                if success:
                    success_count += 1
                else:
                    failed_count += 1
                    failed_images.append((futures[future], error))
                    
                    # Print first few errors for debugging
                    if failed_count <= 3:
                        tqdm.write(f"  ❌ Failed: {futures[future]} - {error}")

                pbar.update(1)
                pbar.set_postfix(success=success_count, failed=failed_count)
                
                # Add small delay to avoid overwhelming the server
                if success_count % 100 == 0:
                    time.sleep(1)

    return success_count, failed_count, failed_images


def main():
    print("="*70)
    print("MIMIC-CXR-JPG Image Download to S3")
    print("Using Official IMAGE_FILENAMES List")
    print("="*70)
    print()

    # Get credentials
    print("PhysioNet Credentials")
    print("-" * 70)
    username = os.environ.get('PHYSIONET_USERNAME')
    password = os.environ.get('PHYSIONET_PASSWORD')

    if not username:
        username = input("PhysioNet Username: ").strip()
    else:
        print(f"Using username from environment: {username}")

    if not password:
        password = getpass.getpass("PhysioNet Password: ")
    else:
        print("Using password from environment variable")

    print()

    # Load IMAGE_FILENAMES list from local disk
    image_paths = load_image_filenames()
    
    # Verify format matches wget expectations
    if not image_paths[0].startswith('files/'):
        print("\n⚠️  WARNING: IMAGE_FILENAMES paths don't start with 'files/'")
        print("This might cause issues. The working wget command expects paths like:")
        print("  files/p10/p10000032/s50414267/xxx.jpg")
        response = input("\nDo you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)

    # Ask about download options
    print("\n" + "="*70)
    print("Download Options")
    print("="*70)
    print(f"\nTotal images available: {len(image_paths):,}")
    print(f"Estimated size: ~500 GB")
    print(f"Estimated S3 cost: ~$11.50/month")
    print()
    print("Options:")
    print("  1. Download ALL images (377,110 images, ~500 GB)")
    print("  2. Download a SAMPLE for testing (100 images)")
    print("  3. Download CUSTOM amount")
    print("  4. Download by patient ID (e.g., p10000032)")
    print("  5. Cancel")
    print()

    choice = input("Enter choice (1-5): ").strip()

    sample_size = None

    if choice == '1':
        sample_size = None
        print("\n⚠️  This will download ALL 377,110 images (~500 GB)")
        print("⚠️  This will take 10-20 hours")
        confirm = input("Are you sure? Type 'yes' to confirm: ").strip().lower()
        if confirm != 'yes':
            print("Cancelled.")
            sys.exit(0)

    elif choice == '2':
        sample_size = 100  # Reduced for initial testing
        print(f"\n✓ Will download {sample_size:,} sample images")

    elif choice == '3':
        try:
            sample_size = int(input("How many images? "))
            if sample_size <= 0 or sample_size > len(image_paths):
                raise ValueError()

            size_gb = sample_size * 1.3 / 1000  # Rough estimate
            print(f"\n✓ Will download {sample_size:,} images (~{size_gb:.1f} GB)")
        except ValueError:
            print("Invalid number. Cancelled.")
            sys.exit(0)

    elif choice == '4':
        patient_id = input("Enter patient ID (e.g., p10000032): ").strip()

        # Filter images for this patient - check different possible formats
        patient_images = []
        for p in image_paths:
            if f'/{patient_id}/' in p or f'p{patient_id}/' in p or patient_id in p:
                patient_images.append(p)

        if not patient_images:
            print(f"✗ No images found for patient {patient_id}")
            
            # Show example of what we're looking for
            print("\nSearching for paths containing:", patient_id)
            print("Example expected format: files/p10/p10000032/s50414267/xxx.jpg")
            
            # Debug: show a few paths to help understand the format
            print("\nSample paths in IMAGE_FILENAMES:")
            for i, p in enumerate(image_paths[:3]):
                print(f"  {p}")
            
            sys.exit(1)

        print(f"\n✓ Found {len(patient_images):,} images for patient {patient_id}")
        print("Sample images:")
        for img in patient_images[:3]:
            print(f"  {img}")
        
        image_paths = patient_images
        sample_size = None  # Use all images for this patient

    else:
        print("Cancelled.")
        sys.exit(0)

    # Download images
    success, failed, failed_list = download_images_parallel(
        image_paths, username, password, sample_size
    )

    # Summary
    print("\n" + "="*70)
    print("Download Summary")
    print("="*70)
    print(f"✓ Successful: {success:,} images")
    print(f"✗ Failed:     {failed:,} images")

    if failed > 0 and failed < 20:
        print("\nFailed images:")
        for img_path, error in failed_list[:10]:
            print(f"  • {img_path}: {error}")

    if failed > 0:
        # Save failed list
        failed_file = "failed_images.txt"
        with open(failed_file, 'w') as f:
            for img_path, error in failed_list:
                f.write(f"{img_path}\t{error}\n")
        print(f"\n📄 Failed images saved to: {failed_file}")

    print()
    print("Verify upload:")
    print(f"  aws s3 ls s3://{S3_BUCKET}/{S3_PREFIX}/files/ --recursive | head -20")
    print()
    print("Count uploaded images:")
    print(f"  aws s3 ls s3://{S3_BUCKET}/{S3_PREFIX}/files/ --recursive | wc -l")
    print()


if __name__ == '__main__':
    main()