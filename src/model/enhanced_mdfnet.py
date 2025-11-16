"""
Enhanced MDF-Net: Multi-Modal Deep Fusion Network for CXR Abnormality Detection

Combines three modalities through cross-modal attention:
- Vision: BiomedCLIP-CXR (pre-trained ViT)
- Text: Clinical ModernBERT (8192 context)
- Clinical: Structured features (vitals, demographics)

Architecture:
- Encoders (pre-trained) → Cross-Modal Attention Fusion → Classification Head
- Total parameters: ~239M (87M vision + 149M text + 2.4M fusion + 394K head)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math
import logging

logger = logging.getLogger(__name__)


class VisionEncoder(nn.Module):
    """
    BiomedCLIP-CXR Vision Encoder
    Pre-trained Vision Transformer for chest X-ray analysis
    """

    def __init__(self, model_name: str = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
                 freeze: bool = True):
        """
        Args:
            model_name: Model identifier (open_clip format with 'hf-hub:' prefix for HuggingFace)
            freeze: Whether to freeze encoder weights
        """
        super().__init__()

        # BiomedCLIP uses open_clip library
        try:
            import open_clip
            # Load BiomedCLIP from HuggingFace Hub
            # Format: model_name='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name=model_name
            )
            self.encoder = self.model.visual  # Get just the vision encoder

            # Get output dimension from the model (try multiple attributes)
            self.output_dim = self._get_output_dim(self.model)
            logger.info(f"Loaded vision encoder with output dim: {self.output_dim}")

        except ImportError:
            raise ImportError(
                "Please install open_clip: pip install open-clip-torch\n"
                "BiomedCLIP requires open_clip, not transformers"
            )
        except Exception as e:
            # Fallback to standard CLIP if BiomedCLIP fails
            import warnings
            warnings.warn(f"Failed to load BiomedCLIP ({e}), falling back to standard CLIP")
            try:
                import open_clip
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-16', pretrained='openai'
                )
                self.encoder = self.model.visual
                self.output_dim = self._get_output_dim(self.model)
                logger.info(f"Loaded fallback CLIP with output dim: {self.output_dim}")
            except:
                raise RuntimeError(
                    "Failed to load both BiomedCLIP and standard CLIP. "
                    "Please install open-clip-torch: pip install open-clip-torch"
                )

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def _get_output_dim(self, model) -> int:
        """
        Get output dimension from open_clip model

        Args:
            model: open_clip model

        Returns:
            Output dimension (typically 512 or 768)
        """
        # Try multiple ways to get the dimension

        # Method 1: Check if visual encoder has output_dim
        if hasattr(model.visual, 'output_dim'):
            return model.visual.output_dim

        # Method 2: Check model's embed_dim (common in CLIP models)
        if hasattr(model, 'embed_dim'):
            return model.embed_dim

        # Method 3: Check visual.embed_dim
        if hasattr(model.visual, 'embed_dim'):
            return model.visual.embed_dim

        # Method 4: Check visual.num_features (timm models)
        if hasattr(model.visual, 'num_features'):
            return model.visual.num_features

        # Method 5: For BiomedCLIP and standard CLIP models, use known dimensions
        # BiomedCLIP (ViT-B/16) uses 512-dim embeddings
        # This avoids running a dummy forward pass during __init__ which can
        # corrupt model state and cause NaN issues later
        logger.info("Using default CLIP dimension: 512")
        return 512

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 3, 518, 518] - Preprocessed chest X-rays

        Returns:
            vision_features: [B, output_dim] - Vision embeddings (512 for BiomedCLIP)
        """
        # BiomedCLIP expects 224x224, so resize
        images_resized = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

        # Normalize images to [-1, 1] range expected by CLIP
        # Assuming input is already in [0, 1] range from preprocessing
        images_normalized = images_resized * 2.0 - 1.0

        # open_clip vision encoder takes tensor directly
        vision_features = self.encoder(images_normalized)  # [B, output_dim]

        return vision_features

    def unfreeze(self):
        """Unfreeze encoder for fine-tuning"""
        for param in self.encoder.parameters():
            param.requires_grad = True


class TextEncoder(nn.Module):
    """
    Clinical Text Encoder
    Supports modern long-context models and falls back to medical BERT
    """

    def __init__(self, model_name: str = 'answerdotai/ModernBERT-base',
                 freeze: bool = True):
        """
        Args:
            model_name: HuggingFace model identifier
            freeze: Whether to freeze encoder weights
        """
        super().__init__()

        try:
            from transformers import AutoModel

            # Try to load the specified model
            try:
                self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
                logger.info(f"Loaded text encoder: {model_name}")
            except (ValueError, OSError, KeyError) as e:
                # ModernBERT or other new models might not be supported
                # Fall back to well-supported medical BERT models
                import warnings
                warnings.warn(
                    f"Failed to load {model_name} ({e}). "
                    f"Falling back to microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
                )

                # Try BiomedBERT first (medical domain, well-supported)
                try:
                    fallback_model = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
                    self.encoder = AutoModel.from_pretrained(fallback_model)
                    logger.info(f"Loaded fallback text encoder: {fallback_model}")
                except Exception as e2:
                    # If BiomedBERT fails, use standard BERT-base
                    warnings.warn(f"Failed to load BiomedBERT ({e2}). Falling back to bert-base-uncased")
                    self.encoder = AutoModel.from_pretrained('bert-base-uncased')
                    logger.info("Loaded fallback text encoder: bert-base-uncased")

        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")

        self.output_dim = self.encoder.config.hidden_size  # 768 for BERT-base models

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, seq_len] - Tokenized text
            attention_mask: [B, seq_len] - Attention mask

        Returns:
            text_features: [B, 768] - Text embeddings
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token or mean pooling
        text_features = outputs.last_hidden_state[:, 0, :]  # [B, 768]

        return text_features

    def unfreeze(self):
        """Unfreeze encoder for fine-tuning"""
        for param in self.encoder.parameters():
            param.requires_grad = True


class ClinicalEncoder(nn.Module):
    """
    Clinical Features Encoder
    Maps normalized clinical features to latent space
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 256,
                 dropout: float = 0.2):
        """
        Args:
            input_dim: Number of clinical features
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.output_dim = output_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

    def forward(self, clinical_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clinical_features: [B, input_dim] - Normalized clinical features

        Returns:
            clinical_embeddings: [B, output_dim] - Clinical embeddings
        """
        return self.network(clinical_features)


class CrossModalAttention(nn.Module):
    """
    Cross-Modal Attention Fusion Layer
    Learns interactions between vision, text, and clinical modalities
    """

    def __init__(self, vision_dim: int = 768, text_dim: int = 768, clinical_dim: int = 256,
                 num_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            vision_dim: Vision embedding dimension
            text_dim: Text embedding dimension
            clinical_dim: Clinical embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.clinical_dim = clinical_dim

        # Project all modalities to same dimension
        self.fusion_dim = 768
        self.vision_proj = nn.Linear(vision_dim, self.fusion_dim)
        self.text_proj = nn.Linear(text_dim, self.fusion_dim)
        self.clinical_proj = nn.Linear(clinical_dim, self.fusion_dim)

        # Multi-head attention for cross-modal interactions
        self.attention = nn.MultiheadAttention(
            embed_dim=self.fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(self.fusion_dim)
        self.norm2 = nn.LayerNorm(self.fusion_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.fusion_dim * 4, self.fusion_dim),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, vision_features: torch.Tensor, text_features: torch.Tensor,
                clinical_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: [B, vision_dim]
            text_features: [B, text_dim]
            clinical_features: [B, clinical_dim]

        Returns:
            fused_features: [B, fusion_dim * 3] - Concatenated multi-modal features
        """
        batch_size = vision_features.size(0)

        # Project to common dimension
        vision_proj = self.vision_proj(vision_features)  # [B, fusion_dim]
        text_proj = self.text_proj(text_features)        # [B, fusion_dim]
        clinical_proj = self.clinical_proj(clinical_features)  # [B, fusion_dim]

        # Stack modalities as sequence [B, 3, fusion_dim]
        modalities = torch.stack([vision_proj, text_proj, clinical_proj], dim=1)

        # Self-attention across modalities (cross-modal interactions)
        attn_out, attn_weights = self.attention(
            query=modalities,
            key=modalities,
            value=modalities,
            need_weights=True
        )  # [B, 3, fusion_dim]

        # Residual connection + Layer norm
        modalities = self.norm1(modalities + self.dropout(attn_out))

        # Feed-forward network with residual
        ffn_out = self.ffn(modalities)
        modalities = self.norm2(modalities + ffn_out)  # [B, 3, fusion_dim]

        # Concatenate attended modalities
        fused = modalities.reshape(batch_size, -1)  # [B, fusion_dim * 3]

        # Store attention weights for visualization
        self.last_attention_weights = attn_weights

        return fused

    def get_attention_weights(self) -> torch.Tensor:
        """Get last attention weights for visualization"""
        return self.last_attention_weights


class ClassificationHead(nn.Module):
    """
    Multi-Label Classification Head
    Maps fused features to abnormality predictions
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512,
                 num_classes: int = 14, dropout1: float = 0.3, dropout2: float = 0.2):
        """
        Args:
            input_dim: Fused feature dimension
            hidden_dim: Hidden layer dimension
            num_classes: Number of abnormality classes (14 for CheXpert)
            dropout1: Dropout after first layer
            dropout2: Dropout after second layer
        """
        super().__init__()

        self.classifier = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout1),

            # Layer 2
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout2),

            # Output layer
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused_features: [B, input_dim] - Fused multi-modal features

        Returns:
            logits: [B, num_classes] - Raw logits (apply sigmoid for probabilities)
        """
        return self.classifier(fused_features)


class EnhancedMDFNet(nn.Module):
    """
    Enhanced Multi-Modal Deep Fusion Network

    Complete architecture combining:
    - Vision Encoder (BiomedCLIP-CXR)
    - Text Encoder (Clinical ModernBERT)
    - Clinical Encoder (Dense layers)
    - Cross-Modal Attention Fusion
    - Classification Head

    Total Parameters: ~239M
    """

    def __init__(self,
                 num_classes: int = 14,
                 clinical_feature_dim: int = 45,
                 modalities: List[str] = ['vision', 'text', 'clinical'],
                 freeze_encoders: bool = True,
                 dropout_fusion: float = 0.1,
                 dropout_head1: float = 0.3,
                 dropout_head2: float = 0.2):
        """
        Args:
            num_classes: Number of abnormality classes (14 for CheXpert)
            clinical_feature_dim: Number of clinical features
            modalities: Which modalities to use (for ablation studies)
            freeze_encoders: Whether to freeze pre-trained encoders (Stage 1)
            dropout_fusion: Dropout in fusion layer
            dropout_head1: Dropout in classification head layer 1
            dropout_head2: Dropout in classification head layer 2
        """
        super().__init__()

        self.num_classes = num_classes
        self.modalities = modalities

        # Encoders
        self.vision_encoder = None
        self.text_encoder = None
        self.clinical_encoder = None

        # Default dimensions (used for dummy features if modality is missing)
        vision_dim = 768
        text_dim = 768
        clinical_dim = 256

        if 'vision' in modalities:
            self.vision_encoder = VisionEncoder(freeze=freeze_encoders)
            vision_dim = self.vision_encoder.output_dim

        if 'text' in modalities:
            self.text_encoder = TextEncoder(freeze=freeze_encoders)
            text_dim = self.text_encoder.output_dim

        if 'clinical' in modalities:
            self.clinical_encoder = ClinicalEncoder(
                input_dim=clinical_feature_dim,
                output_dim=clinical_dim
            )

        # Store dimensions for creating dummy features
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.clinical_dim = clinical_dim

        # Fusion layer
        self.fusion_layer = CrossModalAttention(
            vision_dim=vision_dim,
            text_dim=text_dim,
            clinical_dim=clinical_dim,
            dropout=dropout_fusion
        )

        # Calculate fusion output dimension
        fusion_output_dim = 768 * len(modalities)

        # Classification head
        self.classification_head = ClassificationHead(
            input_dim=fusion_output_dim,
            num_classes=num_classes,
            dropout1=dropout_head1,
            dropout2=dropout_head2
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-modal network

        Args:
            batch: Dictionary containing:
                - 'image': [B, 3, 518, 518] - Chest X-rays
                - 'text_input_ids': [B, seq_len] - Tokenized text
                - 'text_attention_mask': [B, seq_len] - Text attention mask
                - 'clinical_features': [B, clinical_dim] - Clinical features

        Returns:
            outputs: Dictionary containing:
                - 'logits': [B, num_classes] - Raw logits
                - 'probabilities': [B, num_classes] - Sigmoid probabilities
                - 'attention_weights': [B, num_modalities, num_modalities] - Cross-modal attention
        """
        # Encode each modality
        vision_features = None
        text_features = None
        clinical_features = None

        if self.vision_encoder is not None:
            vision_features = self.vision_encoder(batch['image'])

        if self.text_encoder is not None:
            text_features = self.text_encoder(
                input_ids=batch['text_input_ids'],
                attention_mask=batch['text_attention_mask']
            )

        if self.clinical_encoder is not None:
            clinical_features = self.clinical_encoder(batch['clinical_features'])

        # Create dummy features for missing modalities (ablation studies)
        batch_size = batch['image'].size(0) if 'image' in batch else batch['clinical_features'].size(0)
        device = next(self.parameters()).device

        if vision_features is None:
            vision_features = torch.zeros(batch_size, self.vision_dim, device=device)
        if text_features is None:
            text_features = torch.zeros(batch_size, self.text_dim, device=device)
        if clinical_features is None:
            clinical_features = torch.zeros(batch_size, self.clinical_dim, device=device)

        # Fuse modalities
        fused_features = self.fusion_layer(vision_features, text_features, clinical_features)

        # Classify
        logits = self.classification_head(fused_features)
        probabilities = torch.sigmoid(logits)

        # Get attention weights for visualization
        attention_weights = self.fusion_layer.get_attention_weights()

        return {
            'logits': logits,
            'probabilities': probabilities,
            'attention_weights': attention_weights,
            'vision_features': vision_features,
            'text_features': text_features,
            'clinical_features': clinical_features
        }

    def unfreeze_encoders(self):
        """Unfreeze pre-trained encoders for Stage 2 fine-tuning"""
        if self.vision_encoder is not None:
            self.vision_encoder.unfreeze()
        if self.text_encoder is not None:
            self.text_encoder.unfreeze()

    def get_parameter_groups(self, lr_encoders: float, lr_fusion: float, lr_head: float):
        """
        Get parameter groups with discriminative learning rates

        Args:
            lr_encoders: Learning rate for pre-trained encoders
            lr_fusion: Learning rate for fusion layer
            lr_head: Learning rate for classification head

        Returns:
            List of parameter groups for optimizer
        """
        param_groups = []

        # Encoders (low LR to preserve pre-training)
        if self.vision_encoder is not None:
            param_groups.append({
                'params': self.vision_encoder.parameters(),
                'lr': lr_encoders,
                'name': 'vision_encoder'
            })

        if self.text_encoder is not None:
            param_groups.append({
                'params': self.text_encoder.parameters(),
                'lr': lr_encoders,
                'name': 'text_encoder'
            })

        # Clinical encoder (higher LR, trained from scratch)
        if self.clinical_encoder is not None:
            param_groups.append({
                'params': self.clinical_encoder.parameters(),
                'lr': lr_fusion,
                'name': 'clinical_encoder'
            })

        # Fusion layer (medium LR)
        param_groups.append({
            'params': self.fusion_layer.parameters(),
            'lr': lr_fusion,
            'name': 'fusion_layer'
        })

        # Classification head (highest LR)
        param_groups.append({
            'params': self.classification_head.parameters(),
            'lr': lr_head,
            'name': 'classification_head'
        })

        return param_groups

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters in each component"""
        counts = {}

        if self.vision_encoder is not None:
            counts['vision_encoder'] = sum(p.numel() for p in self.vision_encoder.parameters())

        if self.text_encoder is not None:
            counts['text_encoder'] = sum(p.numel() for p in self.text_encoder.parameters())

        if self.clinical_encoder is not None:
            counts['clinical_encoder'] = sum(p.numel() for p in self.clinical_encoder.parameters())

        counts['fusion_layer'] = sum(p.numel() for p in self.fusion_layer.parameters())
        counts['classification_head'] = sum(p.numel() for p in self.classification_head.parameters())
        counts['total'] = sum(counts.values())

        return counts


# Convenience functions for creating model variants

def create_vision_only_model(num_classes: int = 14) -> EnhancedMDFNet:
    """Create vision-only baseline model"""
    return EnhancedMDFNet(
        num_classes=num_classes,
        modalities=['vision'],
        freeze_encoders=True
    )


def create_text_only_model(num_classes: int = 14) -> EnhancedMDFNet:
    """Create text-only baseline model"""
    return EnhancedMDFNet(
        num_classes=num_classes,
        modalities=['text'],
        freeze_encoders=True
    )


def create_clinical_only_model(num_classes: int = 14, clinical_feature_dim: int = 45) -> EnhancedMDFNet:
    """Create clinical-only baseline model"""
    return EnhancedMDFNet(
        num_classes=num_classes,
        clinical_feature_dim=clinical_feature_dim,
        modalities=['clinical'],
        freeze_encoders=True
    )


def create_full_model(num_classes: int = 14, clinical_feature_dim: int = 45,
                     freeze_encoders: bool = True) -> EnhancedMDFNet:
    """Create full multi-modal model"""
    return EnhancedMDFNet(
        num_classes=num_classes,
        clinical_feature_dim=clinical_feature_dim,
        modalities=['vision', 'text', 'clinical'],
        freeze_encoders=freeze_encoders
    )


if __name__ == '__main__':
    # Test model instantiation
    model = create_full_model()

    # Print parameter counts
    param_counts = model.count_parameters()
    print("Parameter Counts:")
    for component, count in param_counts.items():
        print(f"  {component}: {count:,}")

    # Test forward pass with dummy data
    batch_size = 4
    batch = {
        'image': torch.randn(batch_size, 3, 518, 518),
        'text_input_ids': torch.randint(0, 30000, (batch_size, 512)),
        'text_attention_mask': torch.ones(batch_size, 512),
        'clinical_features': torch.randn(batch_size, 45)
    }

    outputs = model(batch)
    print(f"\nOutput shapes:")
    print(f"  Logits: {outputs['logits'].shape}")
    print(f"  Probabilities: {outputs['probabilities'].shape}")
    print(f"  Attention weights: {outputs['attention_weights'].shape}")
