"""
PyTorch Dataset and DataLoader for MIMIC-IV-CXR-ED

Memory-efficient data loading with support for:
- Local and GCS storage
- Data augmentation
- Weighted sampling for class imbalance
- Distributed training
- Variable-length sequence handling
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
import pytorch_lightning as pl
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from torchvision import transforms
import logging

# Import Enhanced RAG adapter
from src.training.enhanced_rag_adapter import EnhancedRAGAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MIMICDataset(Dataset):
    """
    PyTorch Dataset for MIMIC-IV-CXR-ED multi-modal data

    Loads preprocessed .pt files containing:
    - Images: [3, 518, 518] tensors
    - Text: tokenized sequences (input_ids, attention_mask)
    - Clinical features: [45] feature vectors
    - Labels: Dictionary of binary labels for 14 abnormalities
    """

    def __init__(self,
                 data_path: str,
                 augmentation: Optional[transforms.Compose] = None,
                 max_samples: Optional[int] = None):
        """
        Args:
            data_path: Path to .pt file (e.g., '/path/to/train_final.pt')
            augmentation: Optional transforms for image augmentation
            max_samples: Optional limit on number of samples (for debugging)
        """
        super().__init__()

        self.data_path = data_path
        self.augmentation = augmentation

        # Load data
        logger.info(f"Loading data from: {data_path}")

        if data_path.startswith('gs://'):
            # GCS path - download to temp location
            self.data = self._load_from_gcs(data_path)
        else:
            # Local path
            self.data = torch.load(data_path, map_location='cpu', weights_only=False)

        logger.info(f"Loaded {len(self.data)} samples")

        # Limit samples if specified (for debugging)
        if max_samples is not None and max_samples < len(self.data):
            self.data = self.data[:max_samples]
            logger.info(f"Limited to {max_samples} samples for debugging")

        # Auto-detect data format and initialize adapter if needed
        self.data_format = self._detect_data_format()
        self.adapter = None

        if self.data_format == 'enhanced_rag':
            logger.info("Enhanced RAG format detected - initializing adapter")
            self.adapter = EnhancedRAGAdapter()
        else:
            logger.info("Standard format detected - no adapter needed")

        # Extract class names from first sample
        if len(self.data) > 0:
            # Get class names from converted sample if using adapter
            if self.adapter:
                converted = self.adapter.convert_sample(self.data[0])
                self.class_names = list(converted['labels'].keys())
            else:
                self.class_names = list(self.data[0]['labels'].keys())
        else:
            self.class_names = []

    def _detect_data_format(self) -> str:
        """
        Auto-detect whether data is in Enhanced RAG or Standard format

        Returns:
            'enhanced_rag' or 'standard'
        """
        if len(self.data) == 0:
            logger.warning("Empty dataset, defaulting to standard format")
            return 'standard'

        sample = self.data[0]

        # Check for Enhanced RAG format markers
        has_image_tensor = 'image_tensor' in sample
        has_clinical_data_str = 'clinical_data' in sample and isinstance(sample.get('clinical_data'), str)
        has_nested_labels = 'labels' in sample and isinstance(sample.get('labels'), dict) and 'disease_labels' in sample.get('labels', {})
        has_enhanced_note = 'enhanced_note' in sample
        has_attention_segments = 'attention_segments' in sample

        # Check for Standard format markers
        has_image = 'image' in sample
        has_clinical_features = 'clinical_features' in sample

        if has_image_tensor and has_clinical_data_str and has_nested_labels:
            return 'enhanced_rag'
        elif has_image and has_clinical_features:
            return 'standard'
        else:
            logger.warning(f"Ambiguous data format. Sample keys: {sample.keys()}")
            logger.warning("Defaulting to standard format")
            return 'standard'

    def _load_from_gcs(self, gcs_path: str) -> List[Dict]:
        """
        Load data from Google Cloud Storage

        Args:
            gcs_path: GCS path (gs://bucket/path/file.pt)

        Returns:
            List of data dictionaries
        """
        try:
            from google.cloud import storage
            import tempfile

            # Parse GCS path
            path_parts = gcs_path.replace('gs://', '').split('/')
            bucket_name = path_parts[0]
            blob_path = '/'.join(path_parts[1:])

            # Download to temp file
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_path)

            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
                blob.download_to_filename(tmp_file.name)
                data = torch.load(tmp_file.name, map_location='cpu', weights_only=False)

            # Clean up temp file
            os.unlink(tmp_file.name)

            return data

        except ImportError:
            raise ImportError("google-cloud-storage is required for GCS paths")

    def __len__(self) -> int:
        """Return number of samples"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample

        Args:
            idx: Sample index

        Returns:
            Dictionary with:
                - image: [3, 518, 518] tensor
                - text_input_ids: [seq_len] tensor
                - text_attention_mask: [seq_len] tensor
                - clinical_features: [45] tensor
                - labels: Dictionary of binary labels
                - subject_id: int
                - study_id: int
                - dicom_id: str
        """
        raw_sample = self.data[idx]

        # Convert Enhanced RAG format to Standard format if needed
        if self.adapter:
            sample = self.adapter.convert_sample(raw_sample)
        else:
            sample = raw_sample

        # Get image
        image = sample['image']  # [3, 518, 518]

        # Apply augmentation if provided
        if self.augmentation is not None:
            # Convert to PIL for torchvision transforms
            # (transforms expect PIL images or [C, H, W] tensors)
            image = self.augmentation(image)

        # Get text sequences
        text_input_ids = sample['text_input_ids']
        text_attention_mask = sample['text_attention_mask']

        # Get clinical features
        clinical_features = sample['clinical_features']

        # Handle NaN values in clinical features
        # NaN values can occur due to missing vitals/demographics data
        # Replace with 0 (safe for normalized features)
        if torch.isnan(clinical_features).any():
            clinical_features = torch.nan_to_num(clinical_features, nan=0.0)

        # Get labels
        labels = sample['labels']

        # Metadata
        subject_id = sample.get('subject_id', -1)
        study_id = sample.get('study_id', -1)
        dicom_id = sample.get('dicom_id', 'unknown')

        return {
            'image': image,
            'text_input_ids': text_input_ids,
            'text_attention_mask': text_attention_mask,
            'clinical_features': clinical_features,
            'labels': labels,
            'subject_id': subject_id,
            'study_id': study_id,
            'dicom_id': dicom_id
        }

    def compute_class_weights(self) -> torch.Tensor:
        """
        Compute class weights for handling imbalance

        Returns:
            Tensor of shape [num_classes] with positive weights for each class
        """
        # Count positive samples for each class
        class_counts = {name: 0 for name in self.class_names}

        for raw_sample in self.data:
            # Convert if using adapter
            if self.adapter:
                sample = self.adapter.convert_sample(raw_sample)
            else:
                sample = raw_sample

            for class_name, label in sample['labels'].items():
                if label == 1:
                    class_counts[class_name] += 1

        # Compute weights: w = total / (n_classes * count)
        total_samples = len(self.data)
        n_classes = len(self.class_names)

        weights = []
        for class_name in self.class_names:
            count = class_counts[class_name]
            if count > 0:
                weight = total_samples / (n_classes * count)
            else:
                weight = 1.0  # Default weight if class not present
            weights.append(weight)

        return torch.tensor(weights, dtype=torch.float32)

    def compute_sample_weights(self) -> np.ndarray:
        """
        Compute per-sample weights for WeightedRandomSampler

        Samples with more positive labels get higher weights (for minority classes)

        Returns:
            Array of shape [num_samples] with sample weights
        """
        # Get class weights
        class_weights = self.compute_class_weights().numpy()

        # Compute per-sample weights
        sample_weights = []

        for raw_sample in self.data:
            # Convert if using adapter
            if self.adapter:
                sample = self.adapter.convert_sample(raw_sample)
            else:
                sample = raw_sample

            # Sum weights of positive classes for this sample
            weight = 0.0
            for i, class_name in enumerate(self.class_names):
                if sample['labels'][class_name] == 1:
                    weight += class_weights[i]

            # Ensure minimum weight
            weight = max(weight, 1.0)
            sample_weights.append(weight)

        return np.array(sample_weights)


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle variable-length sequences

    Args:
        batch: List of samples from Dataset

    Returns:
        Batched dictionary with padded sequences
    """
    # Collect all fields
    images = []
    text_input_ids = []
    text_attention_masks = []
    clinical_features = []
    labels = []
    subject_ids = []
    study_ids = []
    dicom_ids = []

    for sample in batch:
        images.append(sample['image'])
        text_input_ids.append(sample['text_input_ids'])
        text_attention_masks.append(sample['text_attention_mask'])
        clinical_features.append(sample['clinical_features'])
        labels.append(sample['labels'])
        subject_ids.append(sample['subject_id'])
        study_ids.append(sample['study_id'])
        dicom_ids.append(sample['dicom_id'])

    # Stack images and clinical features
    images_batch = torch.stack(images)  # [batch, 3, 518, 518]
    clinical_features_batch = torch.stack(clinical_features)  # [batch, 45]

    # Pad text sequences to max length in batch
    max_text_len = max(len(seq) for seq in text_input_ids)

    padded_input_ids = []
    padded_attention_masks = []

    for input_ids, attention_mask in zip(text_input_ids, text_attention_masks):
        # Pad to max length
        pad_len = max_text_len - len(input_ids)

        if pad_len > 0:
            # Pad with zeros
            input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=input_ids.dtype)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)])

        padded_input_ids.append(input_ids)
        padded_attention_masks.append(attention_mask)

    text_input_ids_batch = torch.stack(padded_input_ids)  # [batch, max_len]
    text_attention_mask_batch = torch.stack(padded_attention_masks)  # [batch, max_len]

    # Combine labels into batch dictionary
    # Labels are dict of {class_name: label} for each sample
    # Convert to dict of {class_name: [batch] tensor}
    labels_batch = {}
    for class_name in labels[0].keys():
        labels_batch[class_name] = torch.tensor([sample[class_name] for sample in labels])

    return {
        'image': images_batch,
        'text_input_ids': text_input_ids_batch,
        'text_attention_mask': text_attention_mask_batch,
        'clinical_features': clinical_features_batch,
        'labels': labels_batch,
        'subject_id': torch.tensor(subject_ids),
        'study_id': torch.tensor(study_ids),
        'dicom_id': dicom_ids  # List of strings
    }


def create_augmentation_transform(config: Dict) -> transforms.Compose:
    """
    Create data augmentation transforms for training

    Args:
        config: Configuration dictionary

    Returns:
        Composed transforms
    """
    aug_config = config['data']['augmentation']

    if not aug_config['enabled']:
        return None

    transform_list = []

    # Random horizontal flip
    if aug_config.get('horizontal_flip_prob', 0) > 0:
        transform_list.append(
            transforms.RandomHorizontalFlip(p=aug_config['horizontal_flip_prob'])
        )

    # Random rotation
    if aug_config.get('rotation_degrees', 0) > 0:
        transform_list.append(
            transforms.RandomRotation(degrees=aug_config['rotation_degrees'])
        )

    # Color jitter
    if 'color_jitter' in aug_config:
        cj_config = aug_config['color_jitter']
        transform_list.append(
            transforms.ColorJitter(
                brightness=cj_config.get('brightness', 0),
                contrast=cj_config.get('contrast', 0),
                saturation=cj_config.get('saturation', 0),
                hue=cj_config.get('hue', 0)
            )
        )

    if len(transform_list) == 0:
        return None

    return transforms.Compose(transform_list)


class ChunkedDataset(Dataset):
    """
    Dataset wrapper that loads data from multiple chunk files

    Combines multiple .pt files into a single dataset by concatenating them.
    """

    def __init__(self, chunk_paths: List[str], augmentation: Optional[transforms.Compose] = None):
        """
        Args:
            chunk_paths: List of paths to chunk files
            augmentation: Optional transforms for image augmentation
        """
        super().__init__()

        self.chunk_paths = chunk_paths
        self.augmentation = augmentation

        # Load all chunks and concatenate
        logger.info(f"Loading {len(chunk_paths)} chunks...")
        all_data = []
        for chunk_path in chunk_paths:
            chunk_data = torch.load(chunk_path, map_location='cpu', weights_only=False)
            all_data.extend(chunk_data)
            logger.info(f"  Loaded {len(chunk_data)} samples from {Path(chunk_path).name}")

        logger.info(f"Total samples: {len(all_data)}")

        # Create a single MIMICDataset with concatenated data
        # We'll use a temporary file approach or directly set the data
        self.data = all_data

        # Auto-detect data format
        self.data_format = self._detect_data_format()
        self.adapter = None

        if self.data_format == 'enhanced_rag':
            logger.info("Enhanced RAG format detected - initializing adapter")
            self.adapter = EnhancedRAGAdapter()
        else:
            logger.info("Standard format detected - no adapter needed")

        # Extract class names
        if len(self.data) > 0:
            if self.adapter:
                converted = self.adapter.convert_sample(self.data[0])
                self.class_names = list(converted['labels'].keys())
            else:
                self.class_names = list(self.data[0]['labels'].keys())
        else:
            self.class_names = []

    def _detect_data_format(self) -> str:
        """Auto-detect data format"""
        if len(self.data) == 0:
            return 'standard'

        sample = self.data[0]
        has_image_tensor = 'image_tensor' in sample
        has_clinical_data_str = 'clinical_data' in sample and isinstance(sample.get('clinical_data'), str)
        has_nested_labels = 'labels' in sample and isinstance(sample.get('labels'), dict) and 'disease_labels' in sample.get('labels', {})

        if has_image_tensor and has_clinical_data_str and has_nested_labels:
            return 'enhanced_rag'
        return 'standard'

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]

        # Convert format if needed
        if self.adapter:
            sample = self.adapter.convert_sample(sample)

        # Apply augmentation to image if specified
        if self.augmentation and 'image' in sample:
            sample['image'] = self.augmentation(sample['image'])

        return sample


class MIMICDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for MIMIC-IV-CXR-ED

    Handles all data loading logic including:
    - Dataset creation
    - DataLoader creation
    - Distributed sampling
    - Data augmentation
    """

    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config

        # Paths
        self.data_root = Path(config['data']['data_root'])
        self.train_file = config['data']['train_file']
        self.val_file = config['data']['val_file']
        self.test_file = config['data']['test_file']

        # DataLoader settings
        self.batch_size = config['training']['batch_size']
        self.num_workers = config['data']['num_workers']
        self.pin_memory = config['data']['pin_memory']
        self.persistent_workers = config['data'].get('persistent_workers', True)

        # Augmentation
        self.train_transform = create_augmentation_transform(config)
        self.val_transform = None  # No augmentation for validation/test

        # Weighted sampling
        self.use_weighted_sampler = config['data'].get('use_weighted_sampler', False)

        # Datasets (initialized in setup)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _get_data_paths(self, split_name: str, filename: str) -> List[str]:
        """
        Get data path(s), supporting both single files and chunked format

        Args:
            split_name: 'train', 'val', or 'test'
            filename: Configured filename (e.g., 'train_final.pt')

        Returns:
            List of paths to data files (single file or all chunks)
        """
        # Try single file first
        single_file_path = self.data_root / filename
        if single_file_path.exists():
            logger.info(f"Found single file for {split_name}: {filename}")
            return [str(single_file_path)]

        # Try chunked format
        # Look for files like train_chunk_*.pt or val_chunk_*.pt
        chunk_pattern = f"{split_name}_chunk_*.pt"
        chunk_files = sorted(self.data_root.glob(chunk_pattern))

        if chunk_files:
            logger.info(f"Found {len(chunk_files)} chunks for {split_name}")
            return [str(f) for f in chunk_files]

        raise FileNotFoundError(
            f"No data found for {split_name}. Looked for:\n"
            f"  - Single file: {single_file_path}\n"
            f"  - Chunked files: {self.data_root}/{chunk_pattern}"
        )

    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for training/validation/testing

        Args:
            stage: 'fit', 'validate', 'test', or None
        """
        if stage == 'fit' or stage is None:
            # Training dataset
            train_paths = self._get_data_paths('train', self.train_file)
            if len(train_paths) > 1:
                # Multiple chunks - use ChunkedDataset
                self.train_dataset = ChunkedDataset(
                    chunk_paths=train_paths,
                    augmentation=self.train_transform
                )
            else:
                # Single file - use MIMICDataset
                self.train_dataset = MIMICDataset(
                    data_path=train_paths[0],
                    augmentation=self.train_transform
                )

            # Validation dataset
            val_paths = self._get_data_paths('val', self.val_file)
            if len(val_paths) > 1:
                # Multiple chunks - use ChunkedDataset
                self.val_dataset = ChunkedDataset(
                    chunk_paths=val_paths,
                    augmentation=self.val_transform
                )
            else:
                # Single file - use MIMICDataset
                self.val_dataset = MIMICDataset(
                    data_path=val_paths[0],
                    augmentation=self.val_transform
                )

            logger.info(f"Training samples: {len(self.train_dataset)}")
            logger.info(f"Validation samples: {len(self.val_dataset)}")

        if stage == 'test' or stage is None:
            # Test dataset
            test_paths = self._get_data_paths('test', self.test_file)
            if len(test_paths) > 1:
                # Multiple chunks - use ChunkedDataset
                self.test_dataset = ChunkedDataset(
                    chunk_paths=test_paths,
                    augmentation=self.val_transform
                )
            else:
                # Single file - use MIMICDataset
                self.test_dataset = MIMICDataset(
                    data_path=test_paths[0],
                    augmentation=self.val_transform
                )

            logger.info(f"Test samples: {len(self.test_dataset)}")

    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader"""

        # Create sampler
        # Check if we're in a distributed training context
        is_distributed = hasattr(self, 'trainer') and self.trainer is not None and self.trainer.world_size > 1

        if self.use_weighted_sampler and not is_distributed:
            # Weighted sampler for class imbalance (only for single GPU)
            sample_weights = self.train_dataset.compute_sample_weights()
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            shuffle = False
        elif is_distributed:
            # Distributed sampler for multi-GPU
            sampler = DistributedSampler(
                self.train_dataset,
                shuffle=True
            )
            shuffle = False
        else:
            # Regular random sampling
            sampler = None
            shuffle = True

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            collate_fn=collate_fn,
            drop_last=True  # Drop last incomplete batch for stable batch norm
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            collate_fn=collate_fn
        )

    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            collate_fn=collate_fn
        )

    def get_class_weights(self) -> torch.Tensor:
        """Get class weights from training dataset for loss function"""
        if self.train_dataset is None:
            raise RuntimeError("setup() must be called before get_class_weights()")
        return self.train_dataset.compute_class_weights()
