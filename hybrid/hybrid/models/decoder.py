"""Hybrid Decoder: PARCO's multi-pointer decoder with optional distance bias.

Extends PARCODecoder with RRNCO's distance-based inductive bias:
    logits_adjusted = log(exp(logits - α·dist[cur→all]) + ε)

All multi-agent functionality is inherited from PARCODecoder:
- Communication layers (agent self-attention in HCVRPContextEmbedding)
- Multi-pointer mechanism (per-agent logits via PointerAttention)
- POS token support
- Multi-start decoding
"""

import torch
import torch.nn as nn

from typing import Tuple

from rl4co.utils.ops import gather_by_index
from rl4co.utils.pylogger import get_pylogger
from tensordict import TensorDict
from torch import Tensor

from parco.models.decoder import PARCODecoder

log = get_pylogger(__name__)


class HybridDecoder(PARCODecoder):
    """Hybrid decoder extending PARCO's multi-pointer decoder.

    Adds optional distance-based inductive bias from RRNCO:
        inductive_bias = α · distance[current_node → all_nodes]
        logits = log(exp(logits - inductive_bias) + ε)

    This encourages the model to prefer nearby nodes, especially
    valuable with real-world (non-Euclidean) distances.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        env_name: Environment name for context/dynamic embedding lookup.
            Uses 'hcvrp' to reuse PARCO's HCVRPContextEmbedding.
        use_distance_bias: If True, apply distance-based inductive bias.
        **kwargs: Forwarded to PARCODecoder.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        env_name: str = "hcvrp_real",
        use_distance_bias: bool = True,
        **kwargs,
    ):
        # Use "hcvrp" for PARCO's context/dynamic embedding registry
        # since "hcvrp_real" extends hcvrp with the same TD keys
        parco_env_name = "hcvrp"

        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            env_name=parco_env_name,
            **kwargs,
        )

        # Store the actual env_name for reference
        self.hybrid_env_name = env_name
        self.use_distance_bias = use_distance_bias

        if use_distance_bias:
            self.alpha = nn.Parameter(torch.ones(1))

    def forward(
        self,
        td: TensorDict,
        cached,
        num_starts: int = 0,
        do_unbatchify: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Compute logits for all agents, optionally with distance bias.

        Args:
            td: TensorDict with current environment state.
            cached: Precomputed embeddings cache from pre_decoder_hook.
            num_starts: Number of starts for multi-start decoding.
            do_unbatchify: Whether to unbatchify for multi-start.

        Returns:
            logits: [B, m, m+N] logits per agent.
            mask: [B, m, m+N] action mask.
        """
        # ── NaN guard: check cached embeddings before pointer attention ──
        # Guard with isinstance(..., Tensor) because some fields (e.g. graph_context)
        # may be plain ints (e.g. 0) when the context is not yet computed.
        _ne = getattr(cached, "node_embeddings", None)
        if isinstance(_ne, Tensor) and torch.isnan(_ne).any():
            log.warning("NaN detected in cached.node_embeddings — clamping to zero")
            cached.node_embeddings = torch.nan_to_num(_ne, nan=0.0, posinf=1e4, neginf=-1e4)
        _gc = getattr(cached, "graph_context", None)
        if isinstance(_gc, Tensor) and torch.isnan(_gc).any():
            log.warning("NaN detected in cached.graph_context — clamping to zero")
            cached.graph_context = torch.nan_to_num(_gc, nan=0.0, posinf=1e4, neginf=-1e4)
        _lk = getattr(cached, "logit_key", None)
        if isinstance(_lk, Tensor) and torch.isnan(_lk).any():
            log.warning("NaN detected in cached.logit_key — clamping to zero")
            cached.logit_key = torch.nan_to_num(_lk, nan=0.0, posinf=1e4, neginf=-1e4)

        # Call parent PARCODecoder forward
        logits, mask = super().forward(td, cached, num_starts, do_unbatchify)

        # Sanitize logits coming out of pointer attention before distance bias
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            log.warning("NaN/Inf in logits after pointer attention — clamping")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-1e9)

        # Apply distance-based inductive bias
        if self.use_distance_bias and "distance_matrix" in td:
            # current_node: [B, m] — each agent's current position
            cur_nodes = td["current_node"]  # [B, m]
            dist_matrix = td["distance_matrix"]  # [B, m+N, m+N]

            # Gather distance from each agent's current node to all nodes
            # dist_bias: [B, m, m+N]
            dist_bias = gather_by_index(
                dist_matrix, cur_nodes, dim=1, squeeze=False
            )

            # If POS token is active, logits have an extra column
            # (the POS action). Pad dist_bias with zeros so POS gets
            # no distance penalty.
            if logits.size(-1) > dist_bias.size(-1):
                pad_cols = logits.size(-1) - dist_bias.size(-1)
                dist_bias = torch.nn.functional.pad(
                    dist_bias, (0, pad_cols), value=0.0
                )

            # RRNCO-style: logits = log(exp(logits - α·dist) + ε)
            # This softly penalizes distant nodes.
            # Numerically stable: clamp the exponent to avoid exp overflow.
            logits = logits.float()
            inductive_bias = (self.alpha.abs() * dist_bias).float()
            exponent = (logits - inductive_bias).clamp(min=-80.0, max=80.0)
            logits = torch.log(torch.exp(exponent) + 1e-6)

        return logits, mask
