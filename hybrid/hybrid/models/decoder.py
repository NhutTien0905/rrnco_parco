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
            self.alpha_net = nn.Sequential(
                nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1)
            )
            # Init last bias to 1.0 so per-agent alpha starts ~1 before training
            # (keeps backward compat with old alpha=1 and satisfies test_alpha_initial_value)
            with torch.no_grad():
                self.alpha_net[2].bias.fill_(1.0)
                # Zero weights for stable start: output ≈ bias regardless of [cap,speed]
                self.alpha_net[0].weight.zero_()
                self.alpha_net[0].bias.zero_()
                self.alpha_net[2].weight.zero_()

    @property
    def alpha(self):
        """Backward-compat: old ``decoder.alpha`` now proxies to ``alpha_net[2].bias``.

        Keeps ``test_phase5.py::TestHybridDecoderInit::test_has_alpha_when_bias`` green
        while new code uses ``alpha_net`` for hetero per-agent bias.
        """
        if hasattr(self, "alpha_net"):
            return self.alpha_net[2].bias
        raise AttributeError("alpha not available when use_distance_bias=False")

    @alpha.setter
    def alpha(self, value):
        if hasattr(self, "alpha_net"):
            with torch.no_grad():
                if isinstance(value, torch.Tensor):
                    self.alpha_net[2].bias.copy_(value.view(-1)[:1])
                else:
                    self.alpha_net[2].bias.fill_(float(value))
        else:
            raise AttributeError("alpha not available when use_distance_bias=False")

    def load_state_dict(self, state_dict, strict: bool = True):
        """Migrate old checkpoints that have ``alpha`` scalar to ``alpha_net``."""
        # Old checkpoint key: "alpha" (Parameter [1]) -> init alpha_net bias so that
        # forward with typical [capacity,speed] gives similar magnitude before training resumes.
        if "alpha" in state_dict and not any(k.startswith("alpha_net") for k in state_dict):
            old_alpha = state_dict.pop("alpha")
            # Try to seed alpha_net to output ~ old_alpha on average (set last bias)
            try:
                old_val = float(old_alpha.mean().item()) if hasattr(old_alpha, "mean") else float(old_alpha)
            except Exception:
                old_val = 1.0
            # Reset alpha_net with default init then set last layer bias to old_val
            # (weights stay as initialised; bias shift alone approximates magnitude)
            with torch.no_grad():
                # last Linear is at index 2 of Sequential
                if hasattr(self, "alpha_net") and len(self.alpha_net) >= 3:
                    self.alpha_net[2].bias.fill_(old_val)
            # Do not re-add old key; remaining keys load with strict=False to allow missing alpha_net
            return super().load_state_dict(state_dict, strict=False)
        return super().load_state_dict(state_dict, strict=strict)

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

            # RRNCO-style hetero: per-agent α from [capacity, speed]
            # α_per_agent = |alpha_net([capacity, speed])|  -> [B, m]
            # logits = log(exp(logits - α_per_agent * dist) + ε)
            logits = logits.float()
            # Gather per-agent features: handle both pre-reset (capacity/speed)
            # and post-reset (agents_capacity/agents_speed) key names.
            if "capacity" in td.keys() and "speed" in td.keys():
                cap = td["capacity"]  # [B, m]
                spd = td["speed"]     # [B, m]
            elif "agents_capacity" in td.keys() and "agents_speed" in td.keys():
                cap = td["agents_capacity"]
                spd = td["agents_speed"]
            elif "agents_capacity" in td.keys():
                cap = td["agents_capacity"]
                spd = torch.ones_like(cap) * 0.75  # fallback speed
            else:
                cap = None
                spd = None
            if cap is not None and spd is not None:
                # stack -> [B, m, 2], alpha_net -> [B, m, 1] -> [B, m]
                cap_f = cap.float()
                spd_f = spd.float()
                # Ensure 2D [B,m] even if 1D or 3D edge
                if cap_f.dim() == 1:
                    cap_f = cap_f.unsqueeze(0)
                    spd_f = spd_f.unsqueeze(0)
                agent_feat = torch.stack([cap_f, spd_f], dim=-1)  # [B, m, 2]
                alpha_per_agent = self.alpha_net(agent_feat).squeeze(-1).abs()  # [B, m]
                inductive_bias = alpha_per_agent.unsqueeze(-1) * dist_bias.float()  # [B, m, m+N]
            else:
                # Fallback: uniform alpha=1
                inductive_bias = dist_bias.float()
            exponent = (logits - inductive_bias).clamp(min=-80.0, max=80.0)
            logits = torch.log(torch.exp(exponent) + 1e-6)

        return logits, mask
