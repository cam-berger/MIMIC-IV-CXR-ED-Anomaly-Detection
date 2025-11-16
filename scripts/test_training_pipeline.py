#!/usr/bin/env python3
"""
Test Training Pipeline with Enhanced RAG Data

This script performs a quick end-to-end test of the training pipeline:
1. Loads Enhanced RAG data with adapter
2. Creates model
3. Runs a few training steps
4. Validates everything integrates correctly

Use this before running full training to catch issues early.
"""

import sys
import torch
from pathlib import Path
import yaml
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.dataloader import MIMICDataModule
from src.model.enhanced_mdfnet import EnhancedMDFNet
from src.model.losses import CombinedLoss

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML config"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def test_data_loading(config: dict):
    """Test 1: Data loading with Enhanced RAG adapter"""
    print("\n" + "=" * 70)
    print("Test 1: Data Loading with Enhanced RAG Adapter")
    print("=" * 70)

    try:
        # Create data module
        data_module = MIMICDataModule(config)

        # Setup for training
        logger.info("Setting up data module...")
        data_module.setup(stage='fit')

        # Get dataloaders
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()

        logger.info(f"✓ Training batches: {len(train_loader)}")
        logger.info(f"✓ Validation batches: {len(val_loader)}")

        # Test loading one batch
        logger.info("\nLoading first training batch...")
        batch = next(iter(train_loader))

        # Verify batch structure
        logger.info(f"✓ Batch keys: {list(batch.keys())}")
        logger.info(f"✓ Image shape: {batch['image'].shape}")
        logger.info(f"✓ Text input IDs shape: {batch['text_input_ids'].shape}")
        logger.info(f"✓ Clinical features shape: {batch['clinical_features'].shape}")
        logger.info(f"✓ Labels: {len(batch['labels'])} classes")

        # Verify data format
        assert batch['image'].shape[0] == config['training']['batch_size'], "Wrong batch size"
        assert batch['image'].shape[1:] == (3, 518, 518), "Wrong image shape"
        assert batch['clinical_features'].shape[1] == 45, "Wrong clinical features dim"
        assert len(batch['labels']) == 14, "Wrong number of label classes"

        print("\n" + "=" * 70)
        print("✅ Test 1 PASSED: Data loading works correctly")
        print("=" * 70)
        return True

    except Exception as e:
        logger.error(f"❌ Test 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation(config: dict):
    """Test 2: Model creation"""
    print("\n" + "=" * 70)
    print("Test 2: Model Creation")
    print("=" * 70)

    try:
        # Create model
        logger.info("Creating EnhancedMDFNet...")
        model = EnhancedMDFNet(
            num_classes=config['model']['num_classes'],
            clinical_feature_dim=config['model']['clinical_feature_dim'],
            modalities=config['model']['modalities'],
            freeze_encoders=config['model']['freeze_encoders'],
            dropout_fusion=config['model']['dropout_fusion'],
            dropout_head1=config['model']['dropout_head1'],
            dropout_head2=config['model']['dropout_head2']
        )

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"✓ Model created successfully")
        logger.info(f"✓ Total parameters: {total_params:,}")
        logger.info(f"✓ Trainable parameters: {trainable_params:,}")

        print("\n" + "=" * 70)
        print("✅ Test 2 PASSED: Model creation works correctly")
        print("=" * 70)
        return True

    except Exception as e:
        logger.error(f"❌ Test 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass(config: dict):
    """Test 3: Forward pass with real data"""
    print("\n" + "=" * 70)
    print("Test 3: Forward Pass")
    print("=" * 70)

    try:
        # Create data module and model
        data_module = MIMICDataModule(config)
        data_module.setup(stage='fit')
        train_loader = data_module.train_dataloader()

        model = EnhancedMDFNet(
            num_classes=config['model']['num_classes'],
            clinical_feature_dim=config['model']['clinical_feature_dim'],
            modalities=config['model']['modalities'],
            freeze_encoders=config['model']['freeze_encoders'],
            dropout_fusion=config['model']['dropout_fusion'],
            dropout_head1=config['model']['dropout_head1'],
            dropout_head2=config['model']['dropout_head2']
        )

        # Get a batch
        batch = next(iter(train_loader))

        # Move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        model = model.to(device)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Check for NaN/inf in input data
        logger.info("Checking input data for NaN/inf...")
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                if torch.isnan(val).any():
                    logger.warning(f"⚠️  {key} contains NaN values!")
                if torch.isinf(val).any():
                    logger.warning(f"⚠️  {key} contains inf values!")
                logger.info(f"  {key}: shape={val.shape}, min={val.min().item():.4f}, max={val.max().item():.4f}")

        # Forward pass
        logger.info("Running forward pass...")
        # Use train() mode for untrained model to avoid BatchNorm issues
        # BatchNorm in eval() mode with uninitialized running stats can cause NaN
        model.train()
        with torch.no_grad():
            outputs = model(batch)

        # Model returns dict, extract probabilities
        if isinstance(outputs, dict):
            probabilities = outputs['probabilities']
            logger.info(f"✓ Model returned dict with keys: {list(outputs.keys())}")
        else:
            probabilities = outputs

        logger.info(f"✓ Output shape: {probabilities.shape}")

        # Check for NaN before asserting
        has_nan = torch.isnan(probabilities).any().item()
        has_inf = torch.isinf(probabilities).any().item()

        if has_nan:
            logger.error(f"❌ Output contains NaN values! {torch.isnan(probabilities).sum().item()} NaN values found")
            logger.error("This suggests a numerical issue in the forward pass")
        if has_inf:
            logger.error(f"❌ Output contains inf values!")

        if not has_nan and not has_inf:
            logger.info(f"✓ Output range: [{probabilities.min().item():.4f}, {probabilities.max().item():.4f}]")

        # Verify output shape
        assert probabilities.shape == (config['training']['batch_size'], 14), "Wrong output shape"
        assert not has_nan, "Outputs contain NaN values - check model initialization and input data"
        assert not has_inf, "Outputs contain inf values"
        assert torch.all((probabilities >= 0) & (probabilities <= 1)), "Outputs should be in [0, 1]"

        print("\n" + "=" * 70)
        print("✅ Test 3 PASSED: Forward pass works correctly")
        print("=" * 70)
        return True

    except Exception as e:
        logger.error(f"❌ Test 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step(config: dict):
    """Test 4: Training step with loss computation"""
    print("\n" + "=" * 70)
    print("Test 4: Training Step")
    print("=" * 70)

    try:
        # Create data module, model, and loss
        data_module = MIMICDataModule(config)
        data_module.setup(stage='fit')
        train_loader = data_module.train_dataloader()

        model = EnhancedMDFNet(
            num_classes=config['model']['num_classes'],
            clinical_feature_dim=config['model']['clinical_feature_dim'],
            modalities=config['model']['modalities'],
            freeze_encoders=config['model']['freeze_encoders'],
            dropout_fusion=config['model']['dropout_fusion'],
            dropout_head1=config['model']['dropout_head1'],
            dropout_head2=config['model']['dropout_head2']
        )

        loss_fn = CombinedLoss(
            lambda_bce=config['loss']['bce_weight'],
            lambda_focal=config['loss']['focal_weight'],
            focal_alpha=config['loss']['focal_alpha'],
            focal_gamma=config['loss']['focal_gamma']
        )

        # Get a batch
        batch = next(iter(train_loader))

        # Move to device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Training step
        logger.info("Running training step...")
        model.train()
        outputs = model(batch)

        # Model returns dict, extract probabilities
        if isinstance(outputs, dict):
            probabilities = outputs['probabilities']
        else:
            probabilities = outputs

        # Check probabilities for NaN
        has_nan = torch.isnan(probabilities).any().item()
        has_inf = torch.isinf(probabilities).any().item()
        if has_nan:
            logger.error(f"❌ Probabilities contain NaN! {torch.isnan(probabilities).sum().item()} NaN values")
        if has_inf:
            logger.error(f"❌ Probabilities contain inf!")
        if not has_nan and not has_inf:
            logger.info(f"✓ Probabilities range: [{probabilities.min().item():.4f}, {probabilities.max().item():.4f}]")

        # Extract labels
        class_names = config['class_names']
        labels_dict = batch['labels']
        batch_size = len(next(iter(labels_dict.values())))

        labels_tensor = torch.zeros(batch_size, len(class_names), device=device)
        for i, class_name in enumerate(class_names):
            if class_name in labels_dict:
                labels_tensor[:, i] = labels_dict[class_name].float()

        # Check labels for NaN
        if torch.isnan(labels_tensor).any():
            logger.error(f"❌ Labels contain NaN!")
        else:
            logger.info(f"✓ Labels range: [{labels_tensor.min().item():.4f}, {labels_tensor.max().item():.4f}]")
            logger.info(f"✓ Positive labels: {(labels_tensor == 1).sum().item()}/{labels_tensor.numel()}")

        # Compute loss
        logger.info("Computing loss...")
        loss_dict = loss_fn(probabilities, labels_tensor)

        # Extract loss values (handle both tensor and float)
        def get_loss_value(loss):
            return loss.item() if hasattr(loss, 'item') else float(loss)

        logger.info(f"✓ Total loss: {get_loss_value(loss_dict['loss']):.4f}")
        if 'bce_loss' in loss_dict:
            logger.info(f"✓ BCE loss: {get_loss_value(loss_dict['bce_loss']):.4f}")
        if 'focal_loss' in loss_dict:
            logger.info(f"✓ Focal loss: {get_loss_value(loss_dict['focal_loss']):.4f}")

        # Verify loss is valid
        loss_value = get_loss_value(loss_dict['loss'])
        import math
        assert not math.isnan(loss_value), "Loss is NaN"
        assert not math.isinf(loss_value), "Loss is Inf"
        assert loss_value > 0, "Loss should be positive"

        print("\n" + "=" * 70)
        print("✅ Test 4 PASSED: Training step works correctly")
        print("=" * 70)
        return True

    except Exception as e:
        logger.error(f"❌ Test 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    import argparse

    parser = argparse.ArgumentParser(description='Test training pipeline')
    parser.add_argument('--config', type=str,
                       default='configs/phase3_enhanced_rag.yaml',
                       help='Path to config file')
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        logger.info("Using default: configs/phase3_enhanced_rag.yaml")
        config_path = Path("configs/phase3_enhanced_rag.yaml")

    # Load config
    logger.info(f"Loading config from: {config_path}")
    config = load_config(str(config_path))

    print("\n" + "=" * 70)
    print("Enhanced RAG Training Pipeline Test Suite")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Data root: {config['data']['data_root']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Modalities: {config['model']['modalities']}")
    print("=" * 70)

    # Run tests
    results = {}
    results['data_loading'] = test_data_loading(config)
    results['model_creation'] = test_model_creation(config)
    results['forward_pass'] = test_forward_pass(config)
    results['training_step'] = test_training_step(config)

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20s}: {status}")
    print("=" * 70)

    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests PASSED! Your training pipeline is ready.")
        print("\nTo start training, run:")
        print(f"  python src/training/train_lightning.py --config {config_path}")
        return 0
    else:
        print("\n❌ Some tests FAILED. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
