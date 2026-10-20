"""Hybrid Init Embedding: Combines RRNCO's distance-aware encoding with
PARCO's agent embedding.

For depot/agent nodes (first m nodes):
    - PARCO-style: coordinate embedding + positional encoding + agent features
For customer nodes (last N nodes):
    - RRNCO-style: CoordinateExpert + DistanceExpert + ContextualGating
    - Plus demand embedding

Returns dual-stream (row_emb, col_emb, distance_matrix) for AttnFreeNet.
"""

import glob
import os

import torch
import torch.nn as nn

from typing import Tuple

from rrnco.models.env_embeddings.rcvrp import (
    CoordinateExpert,
    ContextualGating,
    DistanceExpert,
)
from parco.models.nn.positional_encoder import PositionalEncoder


class HybridInitEmbedding(nn.Module):
    """Hybrid init embedding combining RRNCO and PARCO approaches.

    For depot/agent nodes (first m nodes):
        - PARCO-style: coordinate embedding + positional encoding + agent features
    For customer nodes (last N nodes):
        - RRNCO-style: CoordinateExpert + DistanceExpert + ContextualGating
        - Plus demand embedding

    Returns dual-stream (row_emb, col_emb) for AttnFreeNet consumption,
    plus the full distance matrix.

    Args:
        embed_dim: Embedding dimension.
        linear_bias: Whether to use bias in linear layers.
        demand_scaler: Scaling factor for demand features (should match max capacity).
        speed_scaler: Scaling factor for speed features.
        use_polar_feats: If True, add polar coordinate features for customers.
        sample_type: Sampling strategy for DistanceExpert ('prob' or 'random').
        sample_size: Number of distance columns to sample in DistanceExpert.
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
        use_svd: bool = True,
        svd_cache_dir: str = "data/dataset",
        svd_k: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.demand_scaler = demand_scaler
        self.speed_scaler = speed_scaler
        self.use_polar_feats = use_polar_feats
        self.use_svd = use_svd
        self.svd_cache_dir = svd_cache_dir
        self.svd_k = svd_k
        self._svd_cache: dict = {}
        self._svd_found_path: str | None = None
        self._svd_search_done: bool = False

        # ── Depot/Agent embedding (PARCO-style) ──────────────────────────
        # depot_locs [B, m, 2] → Linear → [B, m, H]
        self.init_embed_depot = nn.Linear(2, embed_dim, linear_bias)
        # Positional encoding to distinguish agent ordering
        self.pos_encoder = PositionalEncoder(embed_dim)
        self.pos_embedding_proj = nn.Linear(embed_dim, embed_dim, linear_bias)
        self.alpha_pos = nn.Parameter(torch.ones(1))
        # Agent features: [x, y, capacity, speed] → [B, m, H]
        self.init_embed_agents = nn.Linear(4, embed_dim, linear_bias)
        # Combine depot + agent embeddings → [B, m, H]
        self.init_embed_depot_agents = nn.Linear(
            2 * embed_dim, embed_dim, linear_bias
        )

        # ── Customer embedding (RRNCO-style) ─────────────────────────────
        # CoordinateExpert: locs → coord_emb [B, N+1, H] (includes depot)
        # We'll split off the depot part and only keep customer part
        # input_dim=3 because CoordinateExpert uses [x, y, angle_to_depot]
        self.coord_expert = CoordinateExpert(input_dim=3, embed_dim=embed_dim)

        # DistanceExpert: distance_matrix → (row_dist_emb, col_dist_emb) [B, N, H]
        self.distance_expert = DistanceExpert(
            embed_dim=embed_dim,
            sample_type=sample_type,
            sample_size=sample_size,
        )

        # SVD low-rank path: k=8 → 2*k=16 → embed_dim
        self.svd_proj = nn.Linear(2 * self.svd_k, embed_dim, linear_bias)

        # ContextualGating: fuse coordinate + distance embeddings
        self.gating_network_row = ContextualGating(embed_dim)
        self.gating_network_col = ContextualGating(embed_dim)

        # Demand embedding
        self.demand_embed = nn.Linear(1, embed_dim, linear_bias)

        # Combine [fused_emb, demand_emb] → [B, N, H] per stream
        self.combine_row_embed = nn.Linear(
            2 * embed_dim, embed_dim, linear_bias
        )
        self.combine_col_embed = nn.Linear(
            2 * embed_dim, embed_dim, linear_bias
        )

    def forward(
        self, td, phase: str = "train"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute initial embeddings from TensorDict.

        Args:
            td: TensorDict after env.reset() with keys:
                - locs [B, m+N, 2]: depot copies + customers
                - distance_matrix [B, m+N, m+N]: normalized distances
                - demand [B, m, m+N]: demand per agent view
                - capacity [B, m]: agent capacities (original generator key)
                - speed [B, m]: agent speeds (original generator key)
                - action_mask [B, m, m+N]: to determine num_agents
            phase: 'train' or 'test' — passed to DistanceExpert for sampling.

        Returns:
            row_emb [B, m+N, H]: row embeddings for AttnFreeNet
            col_emb [B, m+N, H]: col embeddings for AttnFreeNet
            distance [B, m+N, m+N]: full distance matrix (pass-through)
        """
        num_agents = td["action_mask"].shape[-2]  # m
        device = td["locs"].device

        # ── Depot/Agent nodes (first m) ──────────────────────────────────
        depot_locs = td["locs"][..., :num_agents, :]  # [B, m, 2]

        # Coordinate embedding + positional encoding
        depot_emb = self.init_embed_depot(depot_locs)  # [B, m, H]
        pos_emb = self.pos_encoder(depot_emb, add=False)  # [B, m, H]
        pos_emb = self.alpha_pos * self.pos_embedding_proj(pos_emb)
        depot_emb = depot_emb + pos_emb

        # Agent features: [x, y, capacity/scaler, speed/scaler]
        agents_feats = torch.cat(
            [
                depot_locs,
                td["capacity"][..., None] / self.demand_scaler,
                td["speed"][..., None] / self.speed_scaler,
            ],
            dim=-1,
        )  # [B, m, 4]
        agents_emb = self.init_embed_agents(agents_feats)  # [B, m, H]

        # Fuse depot + agent → depot_agent_emb [B, m, H]
        depot_agent_emb = self.init_embed_depot_agents(
            torch.cat([depot_emb, agents_emb], dim=-1)
        )  # [B, m, H]

        # ── Customer nodes (last N) ──────────────────────────────────────
        # Build full locs for CoordinateExpert: [depot(1), customers(N)]
        # CoordinateExpert expects depot as first row, customers after
        # Use the first depot copy (index 0) as the reference depot
        depot_single = td["locs"][..., 0:1, :]  # [B, 1, 2]
        customer_locs = td["locs"][..., num_agents:, :]  # [B, N, 2]
        coord_expert_locs = torch.cat(
            [depot_single, customer_locs], dim=-2
        )  # [B, N+1, 2]

        # CoordinateExpert → [B, N+1, H] (depot + customers)
        coord_emb_full = self.coord_expert(coord_expert_locs)  # [B, N+1, H]
        # Take only customer part for gating (depot handled above)
        customer_coord_emb = coord_emb_full[:, 1:, :]  # [B, N, H]

        # ── Distance / SVD embedding (B1) ────────────────────────────────
        # Extract customer-to-customer submatrix [B, N, N]
        customer_dist = td["distance_matrix"][
            ..., num_agents:, num_agents:
        ]  # [B, N, N]
        N = customer_dist.size(-1)

        # Try SVD path first: td["svd_emb"] or cache file
        svd_emb = None
        if "svd_emb" in td.keys():
            raw = td["svd_emb"]
            # raw expected [B, N, 2*k] or [B, m+N, 2*k]; slice if needed
            if raw.shape[-2] != N:
                # full m+N case → keep last N rows
                if raw.shape[-2] == num_agents + N:
                    raw = raw[..., num_agents:, :]
                else:
                    raw = raw[..., :N, :]
            # ensure shape [..., 2*k]
            if raw.shape[-1] == 2 * self.svd_k:
                svd_emb = self.svd_proj(raw.to(device=device, dtype=torch.float32))
        elif self.use_svd:
            # cache lookup: try td["city"] or generic svd_k*.npz under cache dir
            # if td carries city name, use city-specific cache
            city_name = None
            if "city" in td.keys():
                try:
                    c = td["city"]
                    if isinstance(c, (list, tuple)):
                        city_name = c[0]
                    elif torch.is_tensor(c):
                        city_name = str(c[0].item()) if c.numel() == 1 else None
                    else:
                        city_name = str(c)
                except Exception:
                    city_name = None
            # memoize candidate discovery after first hit to avoid per-forward glob
            if self._svd_search_done and self._svd_found_path is not None:
                found = self._svd_found_path if os.path.isfile(self._svd_found_path) else None
            else:
                candidate_paths = []
                if city_name is not None:
                    candidate_paths.append(
                        os.path.join(self.svd_cache_dir, city_name, f"svd_k{self.svd_k}.npz")
                    )
                candidate_paths.append(os.path.join(self.svd_cache_dir, f"svd_k{self.svd_k}.npz"))
                if not self._svd_search_done:
                    glob_candidates = glob.glob(
                        os.path.join(self.svd_cache_dir, "*", f"svd_k{self.svd_k}.npz")
                    )
                    candidate_paths.extend(glob_candidates)
                else:
                    # reuse previously discovered glob result (no new glob)
                    if self._svd_found_path is not None:
                        candidate_paths.append(self._svd_found_path)
                found = None
                for p in candidate_paths:
                    if os.path.isfile(p):
                        found = p
                        break
                if found is not None:
                    self._svd_found_path = found
                    self._svd_search_done = True
                elif not self._svd_search_done:
                    # remember negative if glob was attempted and nothing found; still mark searched
                    # so next forward without city_name skips glob re-scan
                    self._svd_search_done = True
            if found is not None:
                # memoize loaded array per path to avoid per-step IO
                if found not in self._svd_cache:
                    try:
                        import numpy as np

                        d = np.load(found, allow_pickle=True)
                        U_S = d["U_S"]  # [Nc, k]
                        V_S = d["V_S"]  # [Nc, k]
                        full = np.concatenate([U_S, V_S], axis=-1)  # [Nc, 2k]
                        self._svd_cache[found] = torch.from_numpy(full).float()
                    except Exception:
                        self._svd_cache[found] = None
                cached = self._svd_cache.get(found, None)
                if cached is not None and cached.shape[0] >= N:
                    raw = cached[:N].unsqueeze(0).expand(customer_dist.shape[0], -1, -1)
                    raw = raw.to(device=device)
                    svd_emb = self.svd_proj(raw)

        if svd_emb is not None:
            # svd_emb [B, N, H] used for both row/col streams (VRAM-friendly)
            row_dist_emb = svd_emb
            col_dist_emb = svd_emb
        else:
            # Fallback: DistanceExpert with padding for N < sample_size
            sample_size = self.distance_expert.sample_size
            dist_input = customer_dist
            if N < sample_size:
                pad_n = sample_size - N
                dist_input = torch.nn.functional.pad(
                    dist_input, (0, pad_n, 0, pad_n), value=0.0
                )
            row_dist_emb, col_dist_emb = self.distance_expert(dist_input, phase)
            if N < sample_size:
                row_dist_emb = row_dist_emb[:, :N, :]
                col_dist_emb = col_dist_emb[:, :N, :]

        # ContextualGating: fuse coordinate + distance
        fused_row = self.gating_network_row(
            customer_coord_emb, row_dist_emb
        )  # [B, N, H]
        fused_col = self.gating_network_col(
            customer_coord_emb, col_dist_emb
        )  # [B, N, H]

        # Demand embedding
        # td["demand"] is [B, m, m+N] — take first agent's view, customer part
        demands = td["demand"][..., 0, num_agents:]  # [B, N]
        demand_emb = self.demand_embed(
            demands[..., None] / self.demand_scaler
        )  # [B, N, H]

        # Combine [fused, demand] → customer embeddings per stream
        customer_row_emb = self.combine_row_embed(
            torch.cat([fused_row, demand_emb], dim=-1)
        )  # [B, N, H]
        customer_col_emb = self.combine_col_embed(
            torch.cat([fused_col, demand_emb], dim=-1)
        )  # [B, N, H]

        # ── Concatenate depot/agent + customer ───────────────────────────
        # depot_agent_emb is shared for both streams (symmetric for depots)
        row_emb = torch.cat(
            [depot_agent_emb, customer_row_emb], dim=-2
        )  # [B, m+N, H]
        col_emb = torch.cat(
            [depot_agent_emb, customer_col_emb], dim=-2
        )  # [B, m+N, H]

        # Pass through the full distance matrix
        distance = td["distance_matrix"]  # [B, m+N, m+N]

        return row_emb, col_emb, distance
