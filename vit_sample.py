""" Minimal Vision Transformer (ViT) implementation from scratch.
    Useful reference implementations: 
    - https://github.com/quickgrid/paper-implementations/blob/main/pytorch/vision_transformer/vit.py
    - https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
    - https://github.com/google-research/vision_transformer/blob/4317e064a0a54b825b5b9ff634482954788b8d84/vit_jax/models.py (Official)
"""

import torch
from torch import nn


class TransformerEncoder(nn.Module):
    """Transformer Encoder Module for ViT Model."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout_rate: float,
        mlp_hidden_dim: int,
    ):
        """
        Arguments:
            embedding_dim (int): Embedding dimension of the tensors (D)
            num_heads (int): Number of heads for Multihead Attention
            dropout_rate (float): Dropout rate for MSA
            mlp_hidden_dim (int): Number of hidden dimensions for MLP 
        """
        super(TransformerEncoder, self).__init__()

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            patches (torch Tensor [B x N x D]

        Returns:
            Tensor: the output of a single Transformer Encoder (shape [B x N x D])
        """

        raise NotImplementedError


class ViT(nn.Module):
    """Full Vision Transformer Model."""

    def __init__(
        self,
        num_layers: int,
        num_patches: int,
        input_dim: int,
        embedding_dim: int,
        num_heads: int,
        dropout_rate: float,
        attention_mlp_hidden: int,
        classify_mlp_hidden: int,
        num_classes: int,
    ):
        """
        Arguments:
            num_layers (int): Number of Transformer Encoder Blocks (L)
            num_patches (int): Number of patches (N)
            input_dim (int): Input Dimension of the Patch Sequence (size P^2 * C)
            embedding_dim (int): Internal embedding dimension for transformer
            num_heads (int): Number of heads for Transformer MSA
            dropout_rate (float): Dropout rate for Attention
            attention_mlp_hidden (int): Size of MLP hidden layer for the attention module
            classify_mlp_hidden (int): Size of MLP hidden layer for classification
            num_classes (int): Number of classes for output predictions
        """
        super(ViT, self).__init__()

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            patches (torch.Tensor): Input sequence of patches, size [B x N x (P^2 * C)]
        
        Returns:
            class_pred (torch.Tensor): Resulting prediction over the batch for the class, size [B x num_classes]
        """

        raise NotImplementedError
