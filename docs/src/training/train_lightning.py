"""
PyTorch Lightning Training Script for Enhanced MDF-Net

Comprehensive training pipeline with:
- Multi-GPU distributed training (DDP)
- Mixed precision training
- Gradient accumulation
- TensorBoard logging
- Model checkpointing
- Early stopping
- Learning rate scheduling with warmup
- Support for all modality combinations
"""

import os
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    TQDMProgressBar
)
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
import numpy as np
from sklearn.metrics import roc_auc_score

# Import model and loss
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))  # Add project root to path
from src.model.enhanced_mdfnet import EnhancedMDFNet
from src.model.losses import CombinedLoss, WeightedBCELoss, FocalLoss


class EnhancedMDFNetLightning(pl.LightningModule):
    """
    PyTorch Lightning wrapper for Enhanced MDF-Net

    Handles training loop, validation, optimization, and metrics.
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary from YAML
        """
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        # Create model
        self.model = EnhancedMDFNet(
            num_classes=config['model']['num_classes'],
            clinical_feature_dim=config['model']['clinical_feature_dim'],
            modalities=config['model']['modalities'],
            freeze_encoders=config['model']['freeze_encoders'],
            dropout_fusion=config['model']['dropout_fusion'],
            dropout_head1=config['model']['dropout_head1'],
            dropout_head2=config['model']['dropout_head2']
        )

        # Create loss function
        self.loss_fn = self._create_loss_function()

        # Class names for logging
        self.class_names = config['class_names']

        # Track best validation AUROC
        self.best_val_auroc = 0.0

        # For collecting validation outputs (PyTorch Lightning 2.0+)
        self.validation_step_outputs = []

    def _create_loss_function(self):
        """Create loss function based on config"""
        loss_config = self.config['loss']

        if loss_config['name'] == 'CombinedLoss':
            return CombinedLoss(
                lambda_bce=loss_config['bce_weight'],
                lambda_focal=loss_config['focal_weight'],
                focal_alpha=loss_config['focal_alpha'],
                focal_gamma=loss_config['focal_gamma']
            )
        elif loss_config['name'] == 'WeightedBCE':
            return WeightedBCELoss()
        elif loss_config['name'] == 'Focal':
            return FocalLoss(
                alpha=loss_config.get('focal_alpha', 0.25),
                gamma=loss_config.get('focal_gamma', 2.0)
            )
        else:
            raise ValueError(f"Unknown loss function: {loss_config['name']}")

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass

        Returns:
            probabilities: [B, num_classes] tensor (for loss computation)
        """
        outputs = self.model(batch)
        # Model returns dict with 'probabilities', extract for loss function
        if isinstance(outputs, dict):
            return outputs['probabilities']
        else:
            # If model returns tensor directly, return as-is
            return outputs

    def training_step(self, batch: Dict, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Training step

        Args:
            batch: Dictionary with 'image', 'text_input_ids', 'text_attention_mask',
                   'clinical_features', 'labels'
            batch_idx: Batch index

        Returns:
            Loss dictionary
        """
        # Forward pass
        outputs = self(batch)

        # Get labels
        labels = self._extract_labels(batch['labels'])

        # Compute loss
        loss_dict = self.loss_fn(outputs, labels)

        # Log losses
        self.log('train_loss', loss_dict['loss'], on_step=True, on_epoch=True,
                prog_bar=True, logger=True, sync_dist=True)

        if 'bce_loss' in loss_dict:
            self.log('train_bce_loss', loss_dict['bce_loss'], on_step=False,
                    on_epoch=True, logger=True, sync_dist=True)
        if 'focal_loss' in loss_dict:
            self.log('train_focal_loss', loss_dict['focal_loss'], on_step=False,
                    on_epoch=True, logger=True, sync_dist=True)

        return loss_dict

    def validation_step(self, batch: Dict, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step

        Args:
            batch: Dictionary with inputs and labels
            batch_idx: Batch index

        Returns:
            Dictionary with predictions and labels
        """
        # Forward pass
        outputs = self(batch)

        # Get labels
        labels = self._extract_labels(batch['labels'])

        # Compute loss
        loss_dict = self.loss_fn(outputs, labels)

        # Log validation loss
        self.log('val_loss', loss_dict['loss'], on_step=False, on_epoch=True,
                prog_bar=True, logger=True, sync_dist=True)

        # Store outputs for epoch-end processing
        output_dict = {
            'predictions': outputs.detach(),
            'labels': labels.detach(),
            'loss': loss_dict['loss'].detach()
        }
        self.validation_step_outputs.append(output_dict)

        return output_dict

    def on_validation_epoch_end(self) -> None:
        """
        Called at the end of validation epoch

        Computes and logs AUROC for all classes
        """
        # Skip if no outputs (can happen at sanity check)
        if len(self.validation_step_outputs) == 0:
            return

        # Gather all predictions and labels
        all_preds = torch.cat([x['predictions'] for x in self.validation_step_outputs], dim=0)
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs], dim=0)

        # Move to CPU for sklearn
        all_preds_np = all_preds.cpu().numpy()
        all_labels_np = all_labels.cpu().numpy()

        # Compute AUROC for each class
        aurocs = []
        for i in range(self.config['model']['num_classes']):
            try:
                auroc = roc_auc_score(all_labels_np[:, i], all_preds_np[:, i])
                aurocs.append(auroc)

                # Log per-class AUROC
                self.log(f'val_auroc_{self.class_names[i]}', auroc,
                        on_epoch=True, logger=True, sync_dist=True)
            except ValueError:
                # Handle case where class is not present in batch
                aurocs.append(0.0)

        # Compute mean AUROC
        mean_auroc = np.mean(aurocs)
        self.log('val_mean_auroc', mean_auroc, on_epoch=True, prog_bar=True,
                logger=True, sync_dist=True)

        # Update best AUROC
        if mean_auroc > self.best_val_auroc:
            self.best_val_auroc = mean_auroc

        # Log best AUROC
        self.log('best_val_auroc', self.best_val_auroc, on_epoch=True,
                prog_bar=True, logger=True, sync_dist=True)

        # Clear outputs for next epoch
        self.validation_step_outputs.clear()

    def test_step(self, batch: Dict, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step - same as validation"""
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        """Aggregate test outputs - same as validation"""
        self.on_validation_epoch_end()

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler

        Returns:
            Optimizer and scheduler configuration
        """
        opt_config = self.config['optimizer']
        sched_config = self.config['scheduler']

        # Create optimizer with discriminative learning rates if enabled
        if opt_config.get('use_discriminative_lr', False):
            # Different learning rates for different parts of the model
            encoder_params = []
            fusion_params = []
            head_params = []

            for name, param in self.model.named_parameters():
                if 'vision_encoder' in name or 'text_encoder' in name or 'clinical_encoder' in name:
                    encoder_params.append(param)
                elif 'fusion' in name:
                    fusion_params.append(param)
                elif 'classification_head' in name:
                    head_params.append(param)

            param_groups = [
                {'params': encoder_params, 'lr': opt_config['lr_encoders']},
                {'params': fusion_params, 'lr': opt_config['lr_fusion']},
                {'params': head_params, 'lr': opt_config['lr_head']}
            ]
        else:
            # Single learning rate for all parameters
            param_groups = self.model.parameters()

        # Create optimizer
        if opt_config['name'] == 'AdamW':
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=opt_config['lr'],
                weight_decay=opt_config['weight_decay'],
                betas=opt_config.get('betas', [0.9, 0.999])
            )
        elif opt_config['name'] == 'Adam':
            optimizer = torch.optim.Adam(
                param_groups,
                lr=opt_config['lr'],
                weight_decay=opt_config['weight_decay']
            )
        elif opt_config['name'] == 'SGD':
            optimizer = torch.optim.SGD(
                param_groups,
                lr=opt_config['lr'],
                weight_decay=opt_config['weight_decay'],
                momentum=opt_config.get('momentum', 0.9)
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['name']}")

        # Create learning rate scheduler
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(sched_config['warmup_epochs'] * (total_steps / self.config['training']['max_epochs']))

        if sched_config['name'] == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=sched_config.get('min_lr', 1e-6)
            )
            scheduler_config = {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }

        elif sched_config['name'] == 'step':
            from torch.optim.lr_scheduler import StepLR
            scheduler = StepLR(
                optimizer,
                step_size=sched_config['step_size'],
                gamma=sched_config['gamma']
            )
            scheduler_config = {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }

        elif sched_config['name'] == 'plateau':
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=sched_config['factor'],
                patience=sched_config['patience'],
                min_lr=sched_config.get('min_lr', 1e-6)
            )
            scheduler_config = {
                'scheduler': scheduler,
                'monitor': 'val_mean_auroc',
                'interval': 'epoch',
                'frequency': 1
            }

        else:
            raise ValueError(f"Unknown scheduler: {sched_config['name']}")

        # Add warmup if needed
        if warmup_steps > 0:
            from torch.optim.lr_scheduler import LambdaLR

            def warmup_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return 1.0

            warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

            from torch.optim.lr_scheduler import SequentialLR
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler_config['scheduler']],
                milestones=[warmup_steps]
            )
            scheduler_config['scheduler'] = scheduler

        return {'optimizer': optimizer, 'lr_scheduler': scheduler_config}

    def _extract_labels(self, labels_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract labels from dictionary into tensor

        Args:
            labels_dict: Dictionary with class_name: binary_label

        Returns:
            Tensor of shape [batch_size, num_classes]
        """
        batch_size = len(next(iter(labels_dict.values())))
        num_classes = len(self.class_names)

        labels_tensor = torch.zeros(batch_size, num_classes, device=self.device)

        for i, class_name in enumerate(self.class_names):
            if class_name in labels_dict:
                labels_tensor[:, i] = labels_dict[class_name].float()

        return labels_tensor


def load_config(config_path: str, overrides: Optional[Dict] = None) -> Dict:
    """
    Load configuration from YAML file with optional overrides

    Args:
        config_path: Path to YAML config file
        overrides: Dictionary of config overrides

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Apply overrides if provided
    if overrides:
        for key, value in overrides.items():
            # Support nested keys like 'training.batch_size'
            keys = key.split('.')
            d = config
            for k in keys[:-1]:
                d = d[k]
            d[keys[-1]] = value

    return config


def create_callbacks(config: Dict) -> List[pl.Callback]:
    """
    Create PyTorch Lightning callbacks

    Args:
        config: Configuration dictionary

    Returns:
        List of callbacks
    """
    callbacks = []

    # Model checkpoint callback
    checkpoint_dir = Path(config['logging']['log_dir']) / 'checkpoints'
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='{epoch}-{val_mean_auroc:.4f}',
        monitor=config['training']['monitor'],
        mode=config['training']['mode'],
        save_top_k=config['training']['save_top_k'],
        save_last=config['training']['save_last'],
        every_n_epochs=config['training']['save_interval_epochs'],
        verbose=True
    )
    callbacks.append(checkpoint_callback)

    # Early stopping callback
    if config['training']['early_stopping']['enabled']:
        early_stop_callback = EarlyStopping(
            monitor=config['training']['early_stopping']['monitor'],
            patience=config['training']['early_stopping']['patience'],
            mode=config['training']['early_stopping']['mode'],
            min_delta=config['training']['early_stopping']['min_delta'],
            verbose=True
        )
        callbacks.append(early_stop_callback)

    # Learning rate monitor
    if config['logging'].get('log_lr', True):
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)

    # Progress bar
    progress_bar = TQDMProgressBar(refresh_rate=10)
    callbacks.append(progress_bar)

    return callbacks


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Enhanced MDF-Net')

    # Config file
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')

    # Data paths (can override config)
    parser.add_argument('--data-root', type=str, default=None,
                       help='Root directory for data files')
    parser.add_argument('--train-file', type=str, default=None,
                       help='Training data file name')
    parser.add_argument('--val-file', type=str, default=None,
                       help='Validation data file name')

    # Training overrides
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size per GPU')
    parser.add_argument('--max-epochs', type=int, default=None,
                       help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')

    # Distributed training
    parser.add_argument('--gpus', type=int, default=None,
                       help='Number of GPUs to use')
    parser.add_argument('--num-nodes', type=int, default=1,
                       help='Number of nodes')

    # Checkpointing
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Path to checkpoint to resume from')

    # Experiment name
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name for logging')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Apply command-line overrides
    if args.data_root:
        config['data']['data_root'] = args.data_root
    if args.train_file:
        config['data']['train_file'] = args.train_file
    if args.val_file:
        config['data']['val_file'] = args.val_file
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.max_epochs:
        config['training']['max_epochs'] = args.max_epochs
    if args.lr:
        config['optimizer']['lr'] = args.lr
    if args.gpus:
        config['distributed']['devices'] = args.gpus
    if args.experiment_name:
        config['logging']['experiment_name'] = args.experiment_name

    # Set random seed for reproducibility
    seed_everything(config['compute']['seed'], workers=True)

    # Create model
    model = EnhancedMDFNetLightning(config)

    # Create data module
    from src.training.dataloader import MIMICDataModule
    data_module = MIMICDataModule(config)

    # Create callbacks
    callbacks = create_callbacks(config)

    # Create logger
    logger = TensorBoardLogger(
        save_dir=config['logging']['tensorboard_dir'],
        name=config['logging']['experiment_name']
    )

    # Create trainer
    trainer = Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator=config['distributed']['accelerator'],
        devices=config['distributed']['devices'],
        num_nodes=config['distributed']['num_nodes'],
        strategy=DDPStrategy(find_unused_parameters=False) if config['distributed']['devices'] > 1 else 'auto',
        precision=config['training']['precision'],
        gradient_clip_val=config['training']['gradient_clip_val'],
        gradient_clip_algorithm=config['training']['gradient_clip_algorithm'],
        accumulate_grad_batches=config['training']['gradient_accumulation_steps'],
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config['logging']['log_every_n_steps'],
        deterministic=config['compute']['deterministic'],
        benchmark=config['compute']['benchmark'],
        enable_progress_bar=True,
        enable_model_summary=True
    )

    # Train
    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=args.resume_from
    )

    # Test on best checkpoint
    trainer.test(
        model,
        datamodule=data_module,
        ckpt_path='best'
    )

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation AUROC: {model.best_val_auroc:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print(f"TensorBoard logs: {config['logging']['tensorboard_dir']}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
