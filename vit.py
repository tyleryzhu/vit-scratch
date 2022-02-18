""" Minimal Vision Transformer (ViT) implementation from scratch.
    Useful reference implementations: 
    - https://github.com/quickgrid/paper-implementations/blob/main/pytorch/vision_transformer/vit.py
    - https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
    - https://github.com/google-research/vision_transformer/blob/4317e064a0a54b825b5b9ff634482954788b8d84/vit_jax/models.py (Official)
"""

from turtle import position
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

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_hidden_dim = mlp_hidden_dim

        self.layernorm = nn.LayerNorm(embedding_dim)
        self.multihead_attn = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout_rate
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embedding_dim),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            patches (torch Tensor [B x N x (P^2 * C)]

        Returns:
            Tensor: the output of a single Transformer Encoder
        """
        x = self.layernorm(patches)

        # Multi-head Self Attention (MSA)
        attn_output, _ = self.multihead_attn(x, x, x)
        msa_out = attn_output + patches

        out = self.layernorm(msa_out)
        out = self.mlp(out)
        return out + msa_out


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

        self.num_layers = num_layers
        self.num_patches = num_patches
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.attention_mlp_hidden = attention_mlp_hidden
        self.classify_mlp_hidden = classify_mlp_hidden

        self.linear_proj = nn.Linear(input_dim, embedding_dim)
        self.class_token = nn.Parameter(
            torch.randn((1, 1, self.embedding_dim))
        )  # randn or zeros? authors said zeros
        # Learnable Positional Embedding, as opposed to sinusoidal
        self.position_embeddings = nn.Parameter(
            torch.randn((1, num_patches + 1, embedding_dim))
        )
        self.transformers = nn.ModuleList()
        for _ in range(num_layers):
            self.transformers.append(
                TransformerEncoder(
                    embedding_dim, num_heads, dropout_rate, attention_mlp_hidden
                )
            )
        self.layernorm = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, classify_mlp_hidden),
            nn.Tanh(),
            nn.Linear(classify_mlp_hidden, num_classes),
        )

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            patches (torch.Tensor): Input sequence of patches, size [B x N x (P^2 * C)]
        """
        B = patches.shape[0]
        patch_embeddings = self.linear_proj(patches)

        patch_embeddings = torch.cat(
            [torch.tile(self.class_token, [B, 1, 1]), patch_embeddings], dim=1
        )
        position_embeddings = torch.tile(self.position_embeddings, [B, 1, 1])
        out = patch_embeddings + position_embeddings

        for module in self.transformers:
            out = module(out)

        class_head = self.layernorm(out[:, 0, :].squeeze(1))
        class_pred = self.mlp(class_head)
        return class_pred
