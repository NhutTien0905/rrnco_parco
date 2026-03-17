"""Hybrid Policy: Wires HybridEncoder + PARCODecoder + communication layers.

This module will be fully implemented in Phase 6.
It extends PARCOPolicy with the hybrid encoder/decoder components.
"""

import torch.nn as nn

from typing import Optional


class HybridPolicy(nn.Module):
    """Hybrid policy combining RRNCO encoder with PARCO decoder.

    The forward pass is inherited from PARCOPolicy:
    1. encoder(td) → (h, init_h)
    2. decoder.pre_decoder_hook(td, env, h) → cache
    3. Autoregressive loop: decoder(td, cache) → logits → strategy → handler
    4. env.step(td)

    The key difference is:
    - Encoder: HybridEncoder (RRNCO's AttnFreeNet + fusion)
    - Decoder: HybridDecoder (PARCODecoder + optional distance bias)
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        embed_dim: int = 128,
        num_encoder_layers: int = 6,
        num_heads: int = 8,
        env_name: str = "hcvrp_real",
        nab_type: str = "gating",
        feedforward_hidden: int = 512,
        normalization_encoder: str = "instance",
        agent_handler: str = "highprob",
        use_distance_bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        # TODO: Phase 6 — extend PARCOPolicy and wire components
        raise NotImplementedError(
            "HybridPolicy will be implemented in Phase 6. "
            "See plan.md for architecture details."
        )

    def forward(self, td, env, **kwargs):
        """Run the full encode → decode loop.

        Inherited from PARCOPolicy — no changes needed.
        """
        raise NotImplementedError("Phase 6")
