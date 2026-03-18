"""Hybrid Init Embedding: Combines RRNCO's distance-aware encoding with
PARCO's agent embedding.

This module will be fully implemented in Phase 3.
"""

import torch
import torch.nn as nn

from typing import Tuple


class HybridInitEmbedding(nn.Module):
    """Hybrid init embedding combining RRNCO and PARCO approaches.

    For depot/agent nodes (first m nodes):
        - PARCO-style: coordinate embedding + positional encoding + agent features
    For customer nodes (last N nodes):
        - RRNCO-style: CoordinateExpert + DistanceExpert + ContextualGating
        - Plus demand embedding

    Returns dual-stream (row_emb, col_emb) for AttnFreeNet consumption.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        linear_bias: bool = False,
        demand_scaler: float = 40.0,
        speed_scaler: float = 1.0,
        use_polar_feats: bool = True,
        sample_type: str = "prob",
        sample_size: int = 25,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        # TODO: Phase 3 — initialize all sub-modules
        raise NotImplementedError(
            "HybridInitEmbedding will be implemented in Phase 3. "
            "See plan.md for architecture details."
        )

    def forward(
        self, td
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute initial embeddings from TensorDict.

        Args:
            td: TensorDict with locs, distance_matrix, demand,
                agents_capacity, agents_speed.

        Returns:
            row_emb: [B, m+N, H] row stream embeddings
            col_emb: [B, m+N, H] column stream embeddings
            distance: [B, m+N, m+N] normalized distance matrix
        """
        raise NotImplementedError("Phase 3")
