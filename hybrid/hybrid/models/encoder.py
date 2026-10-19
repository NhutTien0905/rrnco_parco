"""Hybrid Encoder: RRNCO's AttnFreeNet wrapped in PARCO-compatible interface.

Wraps RRNCO's dual-stream AttnFreeNet encoder and fuses the output
into a single-stream representation compatible with PARCO's decoder.

Pipeline:
    td → HybridInitEmbedding → (row_emb, col_emb, distance)
       → AttnFreeNet × L      → (row_emb, col_emb)
       → Fusion                → h  [B, m+N, H]
    Returns (h, init_h) matching PARCOEncoder interface.
"""

import torch
import torch.nn as nn

from typing import Optional, Tuple, Union

from tensordict import TensorDict
from torch import Tensor

from rrnco.models.nn.attn_freenet import AttnFreeNet

from hybrid.models.env_embeddings import HybridInitEmbedding


class HybridEncoder(nn.Module):
    """Hybrid encoder combining RRNCO's Attention-Free Transformer with
    PARCO-compatible output interface.

    Internally uses:
    - HybridInitEmbedding: dual-stream init embedding (Phase 3)
    - AttnFreeNet: RRNCO's AFT encoder with Neural Adaptive Bias
    - Fusion layer: merges dual-stream (row_emb, col_emb) → single stream

    Output matches PARCOEncoder: returns ``(h, init_h)`` where both are
    ``[B, m+N, H]`` tensors.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads (unused by AFT but kept for
            interface compatibility with PARCO configs).
        num_layers: Number of AttnFreeNet layers.
        normalization: Normalization type for AFT blocks.
        feedforward_hidden: Hidden dim in AFT feedforward layers.
        nab_type: Neural Adaptive Bias type ('gating', 'naive', 'heuristic').
        env_name: Environment name (for init embedding registry).
        init_embedding: Custom init embedding module. If None, builds
            HybridInitEmbedding from init_embedding_kwargs.
        init_embedding_kwargs: Kwargs forwarded to HybridInitEmbedding.
        use_pos_token: If True, append a trainable POS token (PARCO feature).
        trainable_pos_token: Whether the POS token is trainable.
        fusion_type: How to merge dual streams — 'concat' (default),
            'gate', or 'add'.
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
        init_embedding_kwargs: Optional[dict] = None,
        use_pos_token: bool = False,
        trainable_pos_token: bool = True,
        fusion_type: str = "concat",
        use_checkpoint: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.env_name = env_name
        self.fusion_type = fusion_type

        # ── Init Embedding ───────────────────────────────────────────────
        if init_embedding is not None:
            self.init_embedding = init_embedding
        else:
            _kwargs = dict(embed_dim=embed_dim)
            if init_embedding_kwargs is not None:
                _kwargs.update(init_embedding_kwargs)
            self.init_embedding = HybridInitEmbedding(**_kwargs)

        # ── AttnFreeNet (RRNCO encoder) ──────────────────────────────────
        self.net = AttnFreeNet(
            embed_dim=embed_dim,
            feedforward_hidden=feedforward_hidden,
            num_layers=num_layers,
            normalization=normalization,
            nab_type=nab_type,
            use_duration_matrix=False,  # HCVRP has no duration matrix
            use_checkpoint=use_checkpoint,
        )
        self.use_checkpoint = use_checkpoint

        # ── Fusion: dual-stream → single-stream ─────────────────────────
        if fusion_type == "concat":
            self.fusion = nn.Linear(2 * embed_dim, embed_dim)
        elif fusion_type == "gate":
            self.gate_proj = nn.Linear(2 * embed_dim, embed_dim)
            self.fusion = None  # gate uses sigmoid, see _fuse()
        elif fusion_type == "add":
            self.fusion = None  # simple addition
        else:
            raise ValueError(
                f"Unknown fusion_type '{fusion_type}'. "
                "Choose from 'concat', 'gate', 'add'."
            )

        # ── POS token (PARCO feature) ───────────────────────────────────
        if use_pos_token and trainable_pos_token:
            self.pos_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        elif use_pos_token:
            self.register_buffer(
                "pos_token", torch.zeros(1, 1, embed_dim)
            )
        else:
            self.pos_token = None
        self.use_pos_token = use_pos_token

    def _fuse(
        self, row_emb: Tensor, col_emb: Tensor
    ) -> Tensor:
        """Merge dual-stream embeddings into a single tensor.

        Args:
            row_emb: [B, m+N, H]
            col_emb: [B, m+N, H]

        Returns:
            Fused embedding [B, m+N, H]
        """
        if self.fusion_type == "concat":
            return self.fusion(torch.cat([row_emb, col_emb], dim=-1))
        elif self.fusion_type == "gate":
            gate = torch.sigmoid(
                self.gate_proj(torch.cat([row_emb, col_emb], dim=-1))
            )
            return gate * row_emb + (1 - gate) * col_emb
        else:  # add
            return row_emb + col_emb

    def forward(
        self,
        td: TensorDict,
        mask: Union[Tensor, None] = None,
        phase: str = "train",
    ) -> Tuple[Tensor, Tensor]:
        """Encode the input TensorDict.

        Args:
            td: TensorDict after env.reset() with keys:
                locs, distance_matrix, demand, capacity, speed, action_mask, …
            mask: Optional attention mask (unused by AFT, kept for interface
                compatibility).
            phase: 'train' or 'test' — forwarded to init embedding's
                DistanceExpert sampling strategy.

        Returns:
            h: Fused node embeddings [B, m+N, H]
            init_h: Initial fused embeddings [B, m+N, H] (for SymNCO loss)
        """
        # 1. Init embedding → dual-stream + distance
        row_emb, col_emb, distance = self.init_embedding(td, phase=phase)
        # row_emb, col_emb: [B, m+N, H]  distance: [B, m+N, m+N]

        # Compute init_h before encoder (for SymNCO symmetricity loss)
        init_h = self._fuse(row_emb, col_emb)  # [B, m+N, H]

        # 2. AttnFreeNet encoder
        locs = td["locs"].float()  # [B, m+N, 2]
        duration_mat = td.get("duration_matrix", None)
        if duration_mat is not None:
            duration_mat = duration_mat.float()

        row_emb, col_emb = self.net(
            row_emb, col_emb, distance, locs, duration_mat
        )
        # row_emb, col_emb: [B, m+N, H]

        # 3. Fuse dual-stream → single-stream
        h = self._fuse(row_emb, col_emb)  # [B, m+N, H]

        # 4. Optional POS token
        if self.use_pos_token:
            B = h.size(0)
            pos = self.pos_token.expand(B, -1, -1)
            h = torch.cat([h, pos], dim=1)         # [B, m+N+1, H]
            init_h = torch.cat([init_h, pos], dim=1)

        return h, init_h
