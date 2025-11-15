"""
Test Full Training Pipeline with Debug Dataset

Quick test to verify:
- Data loading works
- Model forward pass works
- Loss computation works
- Metrics computation works
- Training loop runs without errors

Should complete in ~2 minutes on CPU.
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.dataloader import MIMICDataModule
from src.training.train_lightning import EnhancedMDFNetLightning, load_config
from pytorch_lightning import Trainer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_dataloader(config: dict):
    """Test that DataLoader works correctly"""
    logger.info("Testing DataLoader...")

    # Create data module
    data_module = MIMICDataModule(config)
    data_module.setup('fit')

    # Get a batch
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))

    logger.info(f"✓ DataLoader working!")
    logger.info(f"  Batch size: {batch['image'].shape[0]}")
    logger.info(f"  Image shape: {batch['image'].shape}")
    logger.info(f"  Text input_ids shape: {batch['text_input_ids'].shape}")
    logger.info(f"  Clinical features shape: {batch['clinical_features'].shape}")
    logger.info(f"  Number of classes: {len(batch['labels'])}")

    return batch


def test_model_forward(config: dict, batch: dict):
    """Test that model forward pass works"""
    logger.info("\nTesting model forward pass...")

    # Create model
    model = EnhancedMDFNetLightning(config)
    model.eval()

    # Forward pass
    with torch.no_grad():
        outputs = model(batch)

    logger.info(f"✓ Model forward pass working!")
    logger.info(f"  Output shape: {outputs.shape}")
    logger.info(f"  Output range: [{outputs.min():.4f}, {outputs.max():.4f}]")

    return outputs


def test_loss_computation(config: dict, outputs: torch.Tensor, batch: dict):
    """Test that loss computation works"""
    logger.info("\nTesting loss computation...")

    # Create model to access loss function
    model = EnhancedMDFNetLightning(config)

    # Extract labels
    labels = model._extract_labels(batch['labels'])

    # Compute loss
    loss_dict = model.loss_fn(outputs, labels)

    logger.info(f"✓ Loss computation working!")
    logger.info(f"  Loss: {loss_dict['loss']:.4f}")
    if 'bce_loss' in loss_dict:
        logger.info(f"  BCE loss: {loss_dict['bce_loss']:.4f}")
    if 'focal_loss' in loss_dict:
        logger.info(f"  Focal loss: {loss_dict['focal_loss']:.4f}")


def test_training_loop(config: dict):
    """Test that training loop runs without errors"""
    logger.info("\nTesting training loop (3 epochs on CPU)...")

    # Create model
    model = EnhancedMDFNetLightning(config)

    # Create data module
    data_module = MIMICDataModule(config)

    # Create trainer (CPU, no logging)
    trainer = Trainer(
        max_epochs=3,
        accelerator='cpu',
        devices=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True
    )

    # Train
    trainer.fit(model, datamodule=data_module)

    logger.info(f"✓ Training loop completed!")
    logger.info(f"  Best validation AUROC: {model.best_val_auroc:.4f}")


def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description='Test full pipeline with debug dataset')

    parser.add_argument('--config', type=str, default='configs/base.yaml',
                       help='Path to config file')

    parser.add_argument('--debug-data-dir', type=str, default='data/debug',
                       help='Path to debug dataset directory')

    parser.add_argument('--skip-training', action='store_true',
                       help='Skip the training loop test (just test components)')

    args = parser.parse_args()

    # Load config
    logger.info(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Override data path to use debug dataset
    config['data']['data_root'] = args.debug_data_dir
    config['training']['max_epochs'] = 3

    logger.info("\n" + "=" * 60)
    logger.info("Testing Full Training Pipeline")
    logger.info("=" * 60)

    # Test 1: DataLoader
    batch = test_dataloader(config)

    # Test 2: Model forward pass
    outputs = test_model_forward(config, batch)

    # Test 3: Loss computation
    test_loss_computation(config, outputs, batch)

    # Test 4: Training loop (optional, slower)
    if not args.skip_training:
        test_training_loop(config)
    else:
        logger.info("\nSkipping training loop test (--skip-training flag)")

    logger.info("\n" + "=" * 60)
    logger.info("✓ All tests passed!")
    logger.info("=" * 60)
    logger.info("\nYour training pipeline is ready to use!")
    logger.info("\nNext steps:")
    logger.info("  1. Run on full dataset:")
    logger.info("     python src/training/train_lightning.py --config configs/base.yaml")
    logger.info("  2. Monitor with TensorBoard:")
    logger.info("     tensorboard --logdir tb_logs")


if __name__ == '__main__':
    main()
