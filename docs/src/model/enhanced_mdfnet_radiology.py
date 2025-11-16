"""
Enhanced MDF-Net with Radiology Impressions (CXR-PRO)

Extends Enhanced MDF-Net to include a FOURTH modality:
- Vision: BiomedCLIP-CXR (chest X-ray images)
- Text (Clinical): Clinical ModernBERT (ED pseudo-notes + RAG)
- Text (Radiology): BiomedBERT (CXR-PRO cleaned impressions)
- Clinical: Structured features (vitals, demographics)

Architecture:
- 4 Encoders → Cross-Modal Attention Fusion (4-way) → Classification Head

Key Benefit: Combines clinical context (ED visit) with radiological findings (CXR report)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import logging

from enhanced_mdfnet import (
    VisionEncoder,
    TextEncoder,
    ClinicalEncoder,
    ClassificationHead
)

logger = logging.getLogger(__name__)


class RadiologyEncoder(nn.Module):
    """
    Radiology Impression Encoder
    BiomedBERT for encoding CXR-PRO cleaned radiology impressions
    """

    def __init__(self,
                 model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                 max_length: int = 512,
                 freeze: bool = True):
        """
        Args:
            model_name: Pre-trained model (BiomedBERT for radiology text)
            max_length: Max sequence length for impressions
            freeze: Whether to freeze encoder weights initially
        """
        super().__init__()

        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Loading radiology encoder: {model_name}")
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.max_length = max_length
        self.output_dim = self.encoder.config.hidden_size  # 768 for BERT-base

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        logger.info(f"Radiology encoder loaded with output dim: {self.output_dim}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [B, max_length] - Tokenized impressions
            attention_mask: [B, max_length] - Attention mask

        Returns:
            radiology_features: [B, 768] - Radiology impression embeddings
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use [CLS] token embedding
        radiology_features = outputs.last_hidden_state[:, 0, :]  # [B, 768]

        return radiology_features

    def unfreeze(self):
        """Unfreeze encoder for fine-tuning"""
        for param in self.encoder.parameters():
            param.requires_grad = True


class QuadModalAttentionFusion(nn.Module):
    """
    4-way Cross-Modal Attention Fusion

    Fuses four modalities via multi-head attention:
    - Vision (chest X-ray)
    - Clinical text (ED notes + RAG)
    - Radiology text (CXR-PRO impressions)
    - Clinical features (vitals, demographics)

    The attention mechanism learns to:
    - Align visual findings with radiological descriptions
    - Relate clinical presentation to imaging findings
    - Integrate structured data with unstructured text
    """

    def __init__(self,
                 vision_dim: int = 512,
                 clinical_text_dim: int = 768,
                 radiology_text_dim: int = 768,
                 clinical_feature_dim: int = 256,
                 fusion_dim: int = 768,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        """
        Args:
            vision_dim: BiomedCLIP output dimension
            clinical_text_dim: Clinical ModernBERT output dimension
            radiology_text_dim: Radiology BiomedBERT output dimension
            clinical_feature_dim: Clinical encoder output dimension
            fusion_dim: Common fusion dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.fusion_dim = fusion_dim

        # Project each modality to common dimension
        self.vision_proj = nn.Linear(vision_dim, fusion_dim)
        self.clinical_text_proj = nn.Linear(clinical_text_dim, fusion_dim)
        self.radiology_text_proj = nn.Linear(radiology_text_dim, fusion_dim)
        self.clinical_feature_proj = nn.Linear(clinical_feature_dim, fusion_dim)

        # Multi-head attention for 4-way fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(fusion_dim)
        self.norm2 = nn.LayerNorm(fusion_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 4, fusion_dim),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

        # Attention weights for visualization
        self.last_attention_weights = None

    def forward(self,
                vision_features: torch.Tensor,
                clinical_text_features: torch.Tensor,
                radiology_text_features: torch.Tensor,
                clinical_features: torch.Tensor,
                return_attention: bool = False) -> torch.Tensor:
        """
        Args:
            vision_features: [B, vision_dim] - CXR image features
            clinical_text_features: [B, clinical_text_dim] - ED notes + RAG
            radiology_text_features: [B, radiology_text_dim] - CXR-PRO impressions
            clinical_features: [B, clinical_feature_dim] - Vitals, demographics
            return_attention: Whether to return attention weights

        Returns:
            fused_features: [B, fusion_dim * 4] - Concatenated 4-modal features
            attention_weights: [B, num_heads, 4, 4] (optional) - Cross-modal attention
        """
        batch_size = vision_features.size(0)

        # Project to common dimension
        vision_proj = self.vision_proj(vision_features)  # [B, fusion_dim]
        clinical_text_proj = self.clinical_text_proj(clinical_text_features)  # [B, fusion_dim]
        radiology_text_proj = self.radiology_text_proj(radiology_text_features)  # [B, fusion_dim]
        clinical_proj = self.clinical_feature_proj(clinical_features)  # [B, fusion_dim]

        # Stack modalities as sequence [B, 4, fusion_dim]
        # Order: vision, clinical_text, radiology_text, clinical_features
        modalities = torch.stack([
            vision_proj,
            clinical_text_proj,
            radiology_text_proj,
            clinical_proj
        ], dim=1)

        # Self-attention across all 4 modalities (learns cross-modal interactions)
        attn_out, attn_weights = self.attention(
            query=modalities,
            key=modalities,
            value=modalities,
            need_weights=True
        )  # [B, 4, fusion_dim]

        # Residual connection + Layer norm
        modalities = self.norm1(modalities + self.dropout(attn_out))

        # Feed-forward network with residual
        ffn_out = self.ffn(modalities)
        modalities = self.norm2(modalities + ffn_out)  # [B, 4, fusion_dim]

        # Concatenate attended modalities
        fused = modalities.reshape(batch_size, -1)  # [B, fusion_dim * 4]

        # Store attention weights for visualization
        self.last_attention_weights = attn_weights

        if return_attention:
            return fused, attn_weights
        return fused

    def get_attention_weights(self) -> torch.Tensor:
        """
        Get cross-modal attention weights for visualization

        Returns:
            attention_weights: [B, num_heads, 4, 4]
            - Rows/cols: [vision, clinical_text, radiology_text, clinical_features]
            - High attention between vision and radiology_text indicates alignment
        """
        return self.last_attention_weights


class EnhancedMDFNetWithRadiology(nn.Module):
    """
    Enhanced MDF-Net with Radiology Impressions (4 modalities)

    Architecture:
    INPUT:
    - Chest X-ray images (518x518)
    - Clinical pseudo-notes (ED visit + RAG)
    - Radiology impressions (CXR-PRO cleaned)
    - Clinical features (vitals, demographics)

    ENCODERS:
    - BiomedCLIP-CXR (vision)
    - Clinical ModernBERT (clinical text)
    - BiomedBERT (radiology text)
    - Dense layers (clinical features)

    FUSION:
    - 4-way cross-modal attention

    OUTPUT:
    - 14 abnormality probabilities (CheXpert classes)

    Total Parameters: ~327M (87M vision + 149M clinical_text + 109M radiology + 3M fusion + 1M head)
    """

    def __init__(self,
                 num_classes: int = 14,
                 clinical_feature_dim: int = 45,
                 freeze_encoders: bool = True,
                 fusion_dim: int = 768,
                 num_heads: int = 8,
                 dropout_fusion: float = 0.1,
                 dropout_head1: float = 0.3,
                 dropout_head2: float = 0.2,
                 radiology_encoder_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                 radiology_max_length: int = 512):
        """
        Args:
            num_classes: Number of abnormality classes (14 for CheXpert)
            clinical_feature_dim: Number of clinical features
            freeze_encoders: Whether to freeze pre-trained encoders initially
            fusion_dim: Common dimension for cross-modal fusion
            num_heads: Number of attention heads
            dropout_fusion: Dropout in fusion layer
            dropout_head1: Dropout in classification head layer 1
            dropout_head2: Dropout in classification head layer 2
            radiology_encoder_name: Pre-trained model for radiology text
            radiology_max_length: Max sequence length for radiology impressions
        """
        super().__init__()

        self.num_classes = num_classes
        self.fusion_dim = fusion_dim

        # ===== ENCODERS =====

        # 1. Vision Encoder (BiomedCLIP-CXR)
        self.vision_encoder = VisionEncoder(freeze=freeze_encoders)
        vision_dim = self.vision_encoder.output_dim  # 512

        # 2. Clinical Text Encoder (Clinical ModernBERT for ED notes + RAG)
        self.clinical_text_encoder = TextEncoder(freeze=freeze_encoders)
        clinical_text_dim = self.clinical_text_encoder.output_dim  # 768

        # 3. Radiology Text Encoder (BiomedBERT for CXR-PRO impressions) - NEW!
        self.radiology_encoder = RadiologyEncoder(
            model_name=radiology_encoder_name,
            max_length=radiology_max_length,
            freeze=freeze_encoders
        )
        radiology_text_dim = self.radiology_encoder.output_dim  # 768

        # 4. Clinical Feature Encoder (Dense layers for vitals, demographics)
        self.clinical_encoder = ClinicalEncoder(
            input_dim=clinical_feature_dim,
            hidden_dim=128,
            output_dim=256
        )
        clinical_feature_dim_encoded = self.clinical_encoder.output_dim  # 256

        # ===== FUSION =====

        # Quad-modal attention fusion (4 modalities)
        self.fusion = QuadModalAttentionFusion(
            vision_dim=vision_dim,
            clinical_text_dim=clinical_text_dim,
            radiology_text_dim=radiology_text_dim,
            clinical_feature_dim=clinical_feature_dim_encoded,
            fusion_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout_fusion
        )

        # Fused dimension: fusion_dim * 4 (4 modalities)
        fused_dim = fusion_dim * 4  # 768 * 4 = 3072

        # ===== CLASSIFICATION HEAD =====

        self.classification_head = ClassificationHead(
            input_dim=fused_dim,
            hidden_dim=512,
            num_classes=num_classes,
            dropout1=dropout_head1,
            dropout2=dropout_head2
        )

        logger.info(f"EnhancedMDFNetWithRadiology initialized:")
        logger.info(f"  - Vision dim: {vision_dim}")
        logger.info(f"  - Clinical text dim: {clinical_text_dim}")
        logger.info(f"  - Radiology text dim: {radiology_text_dim}")
        logger.info(f"  - Clinical feature dim: {clinical_feature_dim_encoded}")
        logger.info(f"  - Fusion dim: {fusion_dim}")
        logger.info(f"  - Fused feature dim: {fused_dim}")
        logger.info(f"  - Num classes: {num_classes}")
        logger.info(f"  - Encoders frozen: {freeze_encoders}")

    def forward(self,
                images: torch.Tensor,
                clinical_text_input_ids: torch.Tensor,
                clinical_text_attention_mask: torch.Tensor,
                radiology_input_ids: torch.Tensor,
                radiology_attention_mask: torch.Tensor,
                clinical_features: torch.Tensor,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through 4-modal Enhanced MDF-Net

        Args:
            images: [B, 3, 518, 518] - Chest X-ray images
            clinical_text_input_ids: [B, max_len] - ED notes tokens
            clinical_text_attention_mask: [B, max_len] - ED notes mask
            radiology_input_ids: [B, max_len] - CXR-PRO impression tokens (NEW!)
            radiology_attention_mask: [B, max_len] - CXR-PRO impression mask (NEW!)
            clinical_features: [B, clinical_dim] - Vitals, demographics
            return_attention: Whether to return cross-modal attention weights

        Returns:
            outputs: Dictionary with:
                - logits: [B, num_classes] - Raw abnormality scores
                - probabilities: [B, num_classes] - Sigmoid probabilities
                - attention_weights: [B, num_heads, 4, 4] (if return_attention=True)
        """
        # ===== ENCODE MODALITIES =====

        # 1. Vision: Encode chest X-ray
        vision_features = self.vision_encoder(images)  # [B, 512]

        # 2. Clinical Text: Encode ED notes + RAG
        clinical_text_features = self.clinical_text_encoder(
            clinical_text_input_ids,
            clinical_text_attention_mask
        )  # [B, 768]

        # 3. Radiology Text: Encode CXR-PRO impression (NEW!)
        radiology_text_features = self.radiology_encoder(
            radiology_input_ids,
            radiology_attention_mask
        )  # [B, 768]

        # 4. Clinical Features: Encode vitals, demographics
        clinical_feature_encoded = self.clinical_encoder(clinical_features)  # [B, 256]

        # ===== CROSS-MODAL FUSION =====

        # Fuse all 4 modalities via attention
        if return_attention:
            fused_features, attention_weights = self.fusion(
                vision_features,
                clinical_text_features,
                radiology_text_features,
                clinical_feature_encoded,
                return_attention=True
            )  # [B, 3072], [B, num_heads, 4, 4]
        else:
            fused_features = self.fusion(
                vision_features,
                clinical_text_features,
                radiology_text_features,
                clinical_feature_encoded
            )  # [B, 3072]
            attention_weights = None

        # ===== CLASSIFICATION =====

        logits = self.classification_head(fused_features)  # [B, num_classes]
        probabilities = torch.sigmoid(logits)  # [B, num_classes]

        # ===== OUTPUT =====

        outputs = {
            'logits': logits,
            'probabilities': probabilities
        }

        if return_attention:
            outputs['attention_weights'] = attention_weights

        return outputs

    def unfreeze_encoders(self):
        """Unfreeze all encoders for fine-tuning (Stage 2)"""
        logger.info("Unfreezing encoders for fine-tuning...")
        self.vision_encoder.unfreeze()
        self.clinical_text_encoder.unfreeze()
        self.radiology_encoder.unfreeze()
        # Clinical encoder is not frozen (always trainable)

    def get_num_parameters(self) -> Dict[str, int]:
        """Get parameter counts for each component"""
        counts = {
            'vision_encoder': sum(p.numel() for p in self.vision_encoder.parameters()),
            'clinical_text_encoder': sum(p.numel() for p in self.clinical_text_encoder.parameters()),
            'radiology_encoder': sum(p.numel() for p in self.radiology_encoder.parameters()),
            'clinical_encoder': sum(p.numel() for p in self.clinical_encoder.parameters()),
            'fusion': sum(p.numel() for p in self.fusion.parameters()),
            'classification_head': sum(p.numel() for p in self.classification_head.parameters())
        }
        counts['total'] = sum(counts.values())
        return counts
