"""
Analyze gradient flow through the model

Usage:
    python scripts/debug_gradients.py --config configs/base.yaml --data-root data/debug
"""

import sys
import argparse
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.train_lightning import EnhancedMDFNetLightning, load_config
from src.training.dataloader import MIMICDataModule
from src.training.debug_utils import GradientFlowAnalyzer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Analyze gradient flow')

    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--data-root', type=str, default='data/debug',
                       help='Path to data directory')
    parser.add_argument('--output', type=str, default='gradient_flow.png',
                       help='Output path for gradient flow plot')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    config['data']['data_root'] = args.data_root

    logger.info("Creating model...")
    model = EnhancedMDFNetLightning(config)

    logger.info("Loading data...")
    data_module = MIMICDataModule(config)
    data_module.setup('fit')
    train_loader = data_module.train_dataloader()

    # Get a batch
    batch = next(iter(train_loader))

    # Create analyzer
    logger.info("Analyzing gradients...")
    analyzer = GradientFlowAnalyzer(model.model)  # Use the inner model
    analyzer.hook_gradients()

    # Forward pass and compute loss
    model.train()
    outputs = model(batch)
    labels = model._extract_labels(batch['labels'])
    loss_dict = model.loss_fn(outputs, labels)

    # Analyze gradients
    results = analyzer.analyze_batch(loss_dict['loss'])

    logger.info("\n" + "=" * 60)
    logger.info("GRADIENT ANALYSIS RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total layers: {results['total_layers']}")
    logger.info(f"Vanishing gradients: {len(results['vanishing_layers'])}")
    logger.info(f"Exploding gradients: {len(results['exploding_layers'])}")
    logger.info(f"Dead layers: {len(results['dead_layers'])}")

    if len(results['vanishing_layers']) > 0:
        logger.warning(f"\nLayers with vanishing gradients:")
        for layer in results['vanishing_layers']:
            logger.warning(f"  - {layer}")

    if len(results['exploding_layers']) > 0:
        logger.warning(f"\nLayers with exploding gradients:")
        for layer in results['exploding_layers']:
            logger.warning(f"  - {layer}")

    # Plot
    analyzer.plot_gradient_flow(save_path=args.output)
    logger.info(f"\nGradient flow plot saved to: {args.output}")

    # Cleanup
    analyzer.remove_hooks()


if __name__ == '__main__':
    main()
