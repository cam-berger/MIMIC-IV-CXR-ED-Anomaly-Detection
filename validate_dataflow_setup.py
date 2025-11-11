#!/usr/bin/env python3
"""
Validation script for Dataflow setup

This script checks that:
1. All required dependencies are installed
2. GCP authentication is configured
3. Required APIs are enabled
4. GCS bucket and batch files exist
5. Permissions are correct

Run this before executing the Dataflow pipeline.
"""

import sys
import subprocess
import json
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_status(check_name, passed, message=""):
    """Print check status"""
    status = "✓ PASS" if passed else "✗ FAIL"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    print(f"{color}{status}{reset} - {check_name}")
    if message:
        print(f"       {message}")


def check_python_version():
    """Check Python version is 3.8+"""
    version = sys.version_info
    passed = version.major == 3 and version.minor >= 8
    print_status(
        "Python version",
        passed,
        f"Current: {version.major}.{version.minor}.{version.micro}, Required: 3.8+"
    )
    return passed


def check_dependencies():
    """Check required Python packages"""
    required = {
        'apache_beam': 'apache-beam[gcp]',
        'google.cloud.storage': 'google-cloud-storage',
        'torch': 'torch',
        'numpy': 'numpy',
        'pandas': 'pandas'
    }

    all_passed = True
    for module_name, package_name in required.items():
        try:
            __import__(module_name)
            print_status(f"Package: {package_name}", True)
        except ImportError:
            print_status(f"Package: {package_name}", False, f"Install: pip install {package_name}")
            all_passed = False

    return all_passed


def check_gcloud_cli():
    """Check if gcloud CLI is installed and authenticated"""
    try:
        result = subprocess.run(['gcloud', '--version'], capture_output=True, text=True)
        passed = result.returncode == 0
        print_status("gcloud CLI installed", passed)

        if passed:
            # Check authentication
            result = subprocess.run(['gcloud', 'auth', 'list', '--filter=status:ACTIVE', '--format=value(account)'],
                                  capture_output=True, text=True)
            account = result.stdout.strip()
            if account:
                print_status("gcloud authenticated", True, f"Account: {account}")
                return True
            else:
                print_status("gcloud authenticated", False, "Run: gcloud auth login")
                return False
        return False
    except FileNotFoundError:
        print_status("gcloud CLI installed", False, "Install: https://cloud.google.com/sdk/docs/install")
        return False


def check_config_file():
    """Check if config file exists and is valid"""
    config_path = Path("dataflow_config.json")

    if not config_path.exists():
        print_status("Config file exists", False, "Copy dataflow_config.json.example to dataflow_config.json")
        return False, None

    try:
        with open(config_path) as f:
            config = json.load(f)

        # Check required fields
        required_fields = ['project_id', 'gcs_bucket', 'batch_files_prefix', 'output_prefix']
        missing = [f for f in required_fields if not config.get(f) or config[f] in ['YOUR_PROJECT_ID', 'YOUR_BUCKET_NAME']]

        if missing:
            print_status("Config file valid", False, f"Set these fields: {', '.join(missing)}")
            return False, None

        print_status("Config file valid", True)
        return True, config

    except json.JSONDecodeError as e:
        print_status("Config file valid", False, f"Invalid JSON: {e}")
        return False, None


def check_gcp_project(project_id):
    """Check if GCP project exists and is accessible"""
    try:
        result = subprocess.run(['gcloud', 'projects', 'describe', project_id],
                              capture_output=True, text=True)
        passed = result.returncode == 0
        print_status(f"GCP project accessible", passed, f"Project: {project_id}")
        return passed
    except Exception as e:
        print_status("GCP project accessible", False, str(e))
        return False


def check_apis_enabled(project_id):
    """Check if required APIs are enabled"""
    required_apis = [
        ('dataflow.googleapis.com', 'Dataflow API'),
        ('compute.googleapis.com', 'Compute Engine API'),
        ('storage-api.googleapis.com', 'Cloud Storage API')
    ]

    all_passed = True
    for api_name, display_name in required_apis:
        try:
            result = subprocess.run(
                ['gcloud', 'services', 'list', '--enabled', '--filter=name:' + api_name,
                 '--format=value(name)', '--project', project_id],
                capture_output=True, text=True
            )
            enabled = api_name in result.stdout
            print_status(f"API: {display_name}", enabled,
                        f"Enable: gcloud services enable {api_name}" if not enabled else "")
            if not enabled:
                all_passed = False
        except Exception as e:
            print_status(f"API: {display_name}", False, str(e))
            all_passed = False

    return all_passed


def check_gcs_bucket(bucket_name, batch_prefix, project_id):
    """Check if GCS bucket exists and contains batch files"""
    try:
        # Check bucket existence
        result = subprocess.run(
            ['gsutil', 'ls', '-b', f'gs://{bucket_name}'],
            capture_output=True, text=True
        )
        bucket_exists = result.returncode == 0

        if not bucket_exists:
            print_status("GCS bucket accessible", False, f"Bucket gs://{bucket_name} not found")
            return False

        print_status("GCS bucket accessible", True, f"gs://{bucket_name}")

        # Check batch files
        result = subprocess.run(
            ['gsutil', 'ls', f'gs://{bucket_name}/{batch_prefix}*.pt'],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            batch_files = result.stdout.strip().split('\n')
            count = len([f for f in batch_files if f.strip()])
            print_status("Batch files found", True, f"Found {count} batch files")
            return True
        else:
            print_status("Batch files found", False,
                        f"No files matching gs://{bucket_name}/{batch_prefix}*.pt")
            return False

    except Exception as e:
        print_status("GCS bucket accessible", False, str(e))
        return False


def check_permissions(project_id):
    """Check if user has required permissions"""
    try:
        # Check if user can create Dataflow jobs
        result = subprocess.run(
            ['gcloud', 'projects', 'get-iam-policy', project_id, '--flatten=bindings[].members',
             '--format=json'],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            print_status("Permissions check", True, "User has project access")
            print("       Note: Full permission validation requires attempting a job")
            return True
        else:
            print_status("Permissions check", False, "Cannot read IAM policy")
            return False

    except Exception as e:
        print_status("Permissions check", False, str(e))
        return False


def main():
    """Run all validation checks"""
    print_header("Dataflow Setup Validation")

    all_checks = []

    # Check Python environment
    print_header("Python Environment")
    all_checks.append(check_python_version())
    all_checks.append(check_dependencies())

    # Check GCP CLI
    print_header("Google Cloud CLI")
    all_checks.append(check_gcloud_cli())

    # Check configuration
    print_header("Configuration")
    config_valid, config = check_config_file()
    all_checks.append(config_valid)

    if config:
        # Check GCP setup
        print_header("Google Cloud Project")
        all_checks.append(check_gcp_project(config['project_id']))
        all_checks.append(check_apis_enabled(config['project_id']))

        # Check GCS
        print_header("Google Cloud Storage")
        all_checks.append(check_gcs_bucket(
            config['gcs_bucket'],
            config['batch_files_prefix'],
            config['project_id']
        ))

        # Check permissions
        print_header("Permissions")
        all_checks.append(check_permissions(config['project_id']))

    # Summary
    print_header("Summary")
    passed_count = sum(all_checks)
    total_count = len(all_checks)

    if passed_count == total_count:
        print("\n✓ All checks passed! You're ready to run the Dataflow pipeline.")
        print("\nNext steps:")
        print("  ./run_dataflow_split.sh")
        print("\nOr manually:")
        print(f"  python src/phase1_dataflow_split.py \\")
        print(f"      --project_id {config['project_id']} \\")
        print(f"      --gcs_bucket {config['gcs_bucket']} \\")
        print(f"      --batch_files_prefix {config['batch_files_prefix']} \\")
        print(f"      --output_prefix {config['output_prefix']} \\")
        print(f"      --runner DataflowRunner \\")
        print(f"      --region us-central1")
        sys.exit(0)
    else:
        print(f"\n✗ {total_count - passed_count} check(s) failed. Please fix the issues above.")
        print("\nFor help, see:")
        print("  docs/DATAFLOW_SETUP.md")
        sys.exit(1)


if __name__ == '__main__':
    main()
