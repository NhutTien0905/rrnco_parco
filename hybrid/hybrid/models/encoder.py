"""Hybrid Encoder: RRNCO's AttnFreeNet wrapped in PARCO-compatible interface.

This module will be fully implemented in Phase 4.
It wraps RRNCO's dual-stream AttnFreeNet encoder and fuses the output
into a single-stream representation compatible with PARCO's decoder.
"""

import torch
import torch.nn as nn

from typing import Optional, Tuple


class HybridEncoder(nn.Module):
    """Hybrid encoder combining RRNCO's Attention-Free Transformer with
    PARCO-compatible output interface.

    Takes a TensorDict with node locations, distance matrices, and agent
    features, and produces fused node embeddings [B, m+N, H].

    The encoder internally uses:
    - HybridInitEmbedding: dual-stream init (RRNCO-style for customers,
      PARCO-style for agents)
    - AttnFreeNet: RRNCO's AFT encoder with Neural Adaptive Bias
    - Fusion layer: merges dual-stream (row_emb, col_emb) into single stream

    Output interface matches PARCOEncoder: returns (h, init_h) where both
    are [B, N, H] tensors.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        normalization: str = "instance",
        feedforward_hidden: int = 512,
        nab_type: str = "gating",
        env_name: str = "hcvrp_real",
        init_embedding: Optional[nn.Module] = None,
        init_embedding_kwargs: dict = {},
        use_pos_token: bool = False,
        trainable_pos_token: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.env_name = env_name
        # TODO: Phase 4 — initialize:
        # self.init_embedding = HybridInitEmbedding(...)
        # self.net = AttnFreeNet(...)
        # self.fusion = nn.Linear(2 * embed_dim, embed_dim)
        raise NotImplementedError(
            "HybridEncoder will be implemented in Phase 4. "
            "See plan.md for architecture details."
        )

    def forward(
        self, td, mask=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode the input TensorDict.

        Args:
            td: TensorDict with keys: locs, distance_matrix, demand,
                agents_capacity, agents_speed, etc.
            mask: Optional attention mask.

        Returns:
            h: Fused node embeddings [B, m+N, H]
            init_h: Initial fused embeddings [B, m+N, H] (for SymNCO loss)
        """
        raise NotImplementedError("Phase 4")
