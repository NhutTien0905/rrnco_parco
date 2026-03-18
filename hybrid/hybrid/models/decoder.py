"""Hybrid Decoder: PARCO's multi-pointer decoder with optional distance bias.

This module will be fully implemented in Phase 5.
It extends PARCODecoder with an optional distance-based inductive bias
from RRNCO's decoder.
"""

import torch
import torch.nn as nn

from typing import Optional, Tuple


class HybridDecoder(nn.Module):
    """Hybrid decoder extending PARCO's multi-pointer decoder.

    Adds optional distance-based inductive bias from RRNCO:
        logits = logits - alpha * distance[current_node -> all_nodes]

    This encourages the model to prefer nearby nodes, which is especially
    valuable with real-world (non-Euclidean) distances.

    All multi-agent functionality is inherited from PARCODecoder:
    - Communication layers (agent self-attention in context embedding)
    - Multi-pointer mechanism (per-agent logits)
    - POS token support
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        env_name: str = "hcvrp_real",
        use_distance_bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        # TODO: Phase 5 — extend PARCODecoder
        raise NotImplementedError(
            "HybridDecoder will be implemented in Phase 5. "
            "See plan.md for architecture details."
        )

    def forward(
        self, td, cached, num_starts=0, do_unbatchify=False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute logits for all agents.

        Args:
            td: Current TensorDict state.
            cached: PrecomputedCache from pre_decoder_hook.
            num_starts: Number of multi-start samples.
            do_unbatchify: Whether to unbatchify augmented samples.

        Returns:
            logits: [B, m, N] per-agent logits over nodes.
            mask: [B, m, N] per-agent feasibility mask.
        """
        raise NotImplementedError("Phase 5")
