"""
Find optimal learning rate using LR range test

Usage:
    python scripts/find_lr.py --config configs/base.yaml --data-root data/debug
"""

import sys
import argparse
from pathlib import Path
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.train_lightning import EnhancedMDFNetLightning, load_config
from src.training.dataloader import MIMICDataModule
from src.training.debug_utils import LearningRateFinder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Find optimal learning rate')

    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--data-root', type=str, default='data/debug',
                       help='Path to data directory')
    parser.add_argument('--start-lr', type=float, default=1e-7,
                       help='Starting learning rate')
    parser.add_argument('--end-lr', type=float, default=10,
                       help='Ending learning rate')
    parser.add_argument('--num-iter', type=int, default=100,
                       help='Number of iterations')
    parser.add_argument('--output', type=str, default='lr_finder.png',
                       help='Output path for LR finder plot')

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

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.start_lr,
        weight_decay=config['optimizer']['weight_decay']
    )

    # Create LR finder
    logger.info("Running LR range test...")
    logger.info(f"Start LR: {args.start_lr:.2e}")
    logger.info(f"End LR: {args.end_lr:.2e}")
    logger.info(f"Iterations: {args.num_iter}")

    lr_finder = LearningRateFinder(model.model, optimizer, model.loss_fn)

    suggested_lr, fig = lr_finder.find(
        train_loader=train_loader,
        start_lr=args.start_lr,
        end_lr=args.end_lr,
        num_iter=args.num_iter
    )

    logger.info("\n" + "=" * 60)
    logger.info("LR FINDER RESULTS")
    logger.info("=" * 60)
    logger.info(f"Suggested learning rate: {suggested_lr:.2e}")
    logger.info("=" * 60)

    # Save plot
    fig.savefig(args.output, dpi=300, bbox_inches='tight')
    logger.info(f"\nLR finder plot saved to: {args.output}")

    logger.info("\nRecommendations:")
    logger.info(f"  1. Use suggested LR: {suggested_lr:.2e}")
    logger.info(f"  2. Or use 1/10 of max LR before loss increases")
    logger.info(f"  3. Update config file with chosen LR")


if __name__ == '__main__':
    main()
