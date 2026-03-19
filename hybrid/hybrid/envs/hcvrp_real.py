"""Hybrid Environment: HCVRP with distance matrix support.

Extends PARCO's HCVRPEnv to include distance matrices in the TensorDict,
following RRNCO's pattern for distance normalization and denormalization.
"""

from typing import Callable, Optional, Union

import torch

from parco.envs.hcvrp.env import HCVRPEnv
from parco.envs.hcvrp.generator import HCVRPGenerator
from rl4co.envs.common.utils import get_sampler
from rl4co.utils.ops import gather_by_index, get_distance
from rl4co.utils.pylogger import get_pylogger
from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

log = get_pylogger(__name__)


class HCVRPRealGenerator(HCVRPGenerator):
    """Generator for HCVRP with distance matrix support.

    Extends PARCO's HCVRPGenerator to optionally produce distance matrices.
    By default, Euclidean distance matrices are computed from sampled locations.
    For real-world data, external distance matrices can be passed via TensorDict.

    Args:
        compute_distance_matrix: If True, compute Euclidean distance matrix from locs.
            Set to False if distance matrices will be provided externally (e.g., OSRM data).
        cdist_compute_mode: Compute mode for torch.cdist. Default 'use_mm_for_euclid_dist'.
        **kwargs: All arguments from HCVRPGenerator (num_loc, num_agents, etc.)
    """

    def __init__(
        self,
        compute_distance_matrix: bool = True,
        cdist_compute_mode: str = "use_mm_for_euclid_dist",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.compute_distance_matrix = compute_distance_matrix
        self.cdist_compute_mode = cdist_compute_mode

    def _generate(self, batch_size) -> TensorDict:
        """Generate HCVRP data with optional Euclidean distance matrix.

        Returns TensorDict with standard HCVRP keys plus:
            - distance_matrix [B, num_loc+1, num_loc+1]: pairwise distances
              (depot + customers, before agent-depot expansion)
        """
        td = super()._generate(batch_size)

        if self.compute_distance_matrix:
            # Build full location set: depot (1) + customers (num_loc)
            depot = td["depot"]  # [B, 2]
            locs = td["locs"]  # [B, num_loc, 2]
            all_locs = torch.cat(
                [depot.unsqueeze(-2), locs], dim=-2
            )  # [B, num_loc+1, 2]

            # Compute pairwise Euclidean distances
            distance_matrix = torch.cdist(
                all_locs, all_locs, p=2, compute_mode=self.cdist_compute_mode
            )  # [B, num_loc+1, num_loc+1]
            td.set("distance_matrix", distance_matrix)

        return td


class HCVRPRealEnv(HCVRPEnv):
    """HCVRP environment extended with real-world distance matrices.

    Extends PARCO's HCVRPEnv to include:
    - distance_matrix: [B, m+N, m+N] — real-world or Euclidean distances (normalized)
    - min_distance, max_distance: [B] — for denormalization

    The distance matrix flows through the following pipeline:
    1. Generator produces raw distance_matrix [B, N+1, N+1] (depot + customers)
    2. _reset() expands depot→agents: [B, m+N, m+N] and normalizes per-instance
    3. _step() uses Euclidean distances for tour length (as in PARCO)
       but distance_matrix is available in TensorDict for encoder/decoder
    4. _get_reward() uses PARCO's min-max time objective

    Args:
        generator: HCVRPRealGenerator instance (or None to auto-create).
        generator_params: Dict of params for HCVRPRealGenerator.
        normalize_distance: If True, apply per-instance min-max normalization to
            the distance matrix.
        check_solution: If True, validate solutions (slow, for debugging).
        **kwargs: Additional arguments passed to HCVRPEnv.
    """

    name = "hcvrp_real"

    def __init__(
        self,
        generator: HCVRPRealGenerator = None,
        generator_params: dict = {},
        normalize_distance: bool = True,
        check_solution: bool = False,
        **kwargs,
    ):
        # Build our generator before calling super().__init__
        if generator is None:
            generator = HCVRPRealGenerator(**generator_params)
        self.normalize_distance = normalize_distance
        # Pass our generator to HCVRPEnv.__init__
        super().__init__(
            generator=generator,
            generator_params={},
            check_solution=check_solution,
            **kwargs,
        )

    def _reset(
        self,
        td: Optional[TensorDict] = None,
        batch_size: Optional[list] = None,
    ) -> TensorDict:
        """Reset the environment, adding distance matrix to the TensorDict.

        Extends PARCO's _reset() to:
        1. Expand the distance matrix from [B, N+1, N+1] to [B, m+N, m+N]
           by repeating depot rows/cols for each agent
        2. Apply per-instance min-max normalization (following RRNCO)
        3. Store min_distance, max_distance for potential denormalization
        """
        device = td.device
        num_agents = td["speed"].size(-1)

        # --- Handle distance matrix before calling super ---
        has_distance = "distance_matrix" in td.keys()

        if has_distance:
            raw_distance = td["distance_matrix"]  # [B, N+1, N+1]
        else:
            # Compute Euclidean fallback from locs + depot
            depot = td["depot"]  # [B, 2]
            locs = td["locs"]  # [B, num_loc, 2]
            all_locs = torch.cat(
                [depot.unsqueeze(-2), locs], dim=-2
            )  # [B, N+1, 2]
            raw_distance = torch.cdist(all_locs, all_locs, p=2)  # [B, N+1, N+1]

        # Expand depot row/col to agent rows/cols: [B, N+1, N+1] → [B, m+N, m+N]
        # The depot is index 0 in the raw matrix. We need to repeat it m times.
        # After expansion: indices 0..m-1 = depot (same distances), m..m+N-1 = customers
        depot_row = raw_distance[:, :1, :]  # [B, 1, N+1]
        customer_rows = raw_distance[:, 1:, :]  # [B, N, N+1]

        # Repeat depot row for each agent
        depot_rows_expanded = depot_row.repeat(1, num_agents, 1)  # [B, m, N+1]

        # Stack: [B, m+N, N+1]
        expanded_rows = torch.cat([depot_rows_expanded, customer_rows], dim=1)

        # Now expand columns similarly
        depot_col = expanded_rows[:, :, :1]  # [B, m+N, 1]
        customer_cols = expanded_rows[:, :, 1:]  # [B, m+N, N]

        # Repeat depot column for each agent
        depot_cols_expanded = depot_col.repeat(1, 1, num_agents)  # [B, m+N, m]

        # Final: [B, m+N, m+N]
        distance_matrix = torch.cat(
            [depot_cols_expanded, customer_cols], dim=2
        )  # [B, m+N, m+N]

        # --- Normalize per-instance (following RRNCO pattern) ---
        if self.normalize_distance:
            min_dist = distance_matrix.amin(dim=(-2, -1), keepdim=True)  # [B, 1, 1]
            max_dist = distance_matrix.amax(dim=(-2, -1), keepdim=True)  # [B, 1, 1]
            distance_matrix = (distance_matrix - min_dist) / (
                max_dist - min_dist + 1e-6
            )
            distance_matrix = distance_matrix.to(dtype=torch.float32)
        else:
            min_dist = None
            max_dist = None

        # --- Call parent _reset to build standard HCVRP TensorDict ---
        td_reset = super()._reset(td, batch_size)

        # --- Inject distance matrix and normalization metadata ---
        td_reset.set("distance_matrix", distance_matrix)

        if self.normalize_distance and min_dist is not None:
            td_reset.set(
                "min_distance", min_dist.squeeze(-1).squeeze(-1)
            )  # [B]
            td_reset.set(
                "max_distance", max_dist.squeeze(-1).squeeze(-1)
            )  # [B]

        return td_reset

    def _step(self, td: TensorDict) -> TensorDict:
        """Step the environment, preserving distance matrix in TensorDict.

        The parent _step() handles all multi-agent logic (current_length via
        Euclidean distance, capacity, visited, action_mask). We just ensure
        the distance_matrix and normalization metadata persist.
        """
        # Parent handles everything — distance_matrix is already in td
        # and TensorDict preserves keys through updates
        return super()._step(td)

    def _get_reward(self, td: TensorDict, actions: TensorDict) -> TensorDict:
        """Compute reward using PARCO's min-max time objective.

        We keep PARCO's original reward computation which uses Euclidean
        distances via `current_length` accumulated in _step(). The distance
        matrix is used by the encoder/decoder, not for reward computation.

        Returns:
            Negative max time across agents: -max_i(tour_length_i / speed_i)
        """
        return super()._get_reward(td, actions)
