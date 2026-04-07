"""Hybrid Environment: HCVRP with distance matrix support.

Extends PARCO's HCVRPEnv to include distance matrices in the TensorDict,
following RRNCO's pattern for distance normalization and denormalization.
"""

import os
from typing import Callable, Optional, Union

import numpy as np
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


class HCVRPRealWorldGenerator(HCVRPGenerator):
    """Generator that samples HCVRP instances from real-world OSRM city data.

    Instead of generating random Euclidean coordinates, this generator loads
    pre-computed distance matrices from RRNCO's city dataset and samples
    subproblems (depot + customers) from them.

    Data flow per batch:
        1. Pick a random city from the loaded list
        2. Sample ``num_loc + 1`` points (1 depot + num_loc customers)
        3. Extract the distance sub-matrix for those points
        4. Normalise coordinates to [0, 1]
        5. Generate random demand, per-agent capacity/speed (from HCVRPGenerator)

    Args:
        city_data_dir: Path to RRNCO's ``data/dataset/`` folder containing
            ``{city}/{city}_data.npz`` files.
        cities_list: List of city names to sample from. If ``None``, loads
            from ``splited_cities_list.json`` using ``split``.
        split: Which split to use from ``splited_cities_list.json``
            (``"train"`` or ``"test"``).  Ignored if ``cities_list`` is given.
        loc_dist: Sampling strategy — ``"uniform"`` or ``"cluster"``.
        **kwargs: All arguments from :class:`HCVRPGenerator`
            (``num_loc``, ``num_agents``, ``min_capacity``, etc.)
    """

    def __init__(
        self,
        city_data_dir: str = "data/dataset",
        cities_list: Optional[list] = None,
        split: str = "train",
        loc_dist: str = "uniform",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.city_data_dir = city_data_dir
        self.loc_dist = loc_dist

        # Load city list
        if cities_list is not None:
            self.cities = cities_list
        else:
            import orjson

            list_path = os.path.join(city_data_dir, "splited_cities_list.json")
            with open(list_path, "r") as f:
                all_cities = orjson.loads(f.read())
            self.cities = all_cities[split]

        # Pre-load all city data into memory for fast sampling
        self._city_data = {}
        for city in self.cities:
            path = os.path.join(city_data_dir, city, f"{city}_data.npz")
            if not os.path.exists(path):
                log.warning(f"City data not found: {path}, skipping")
                continue
            raw = np.load(path, allow_pickle=True)
            points = raw["points"]
            distance = raw["distance"]

            # Remove outlier points (OSRM routing failures)
            if distance.max() > 1e5:
                outlier_mask = (
                    np.any(distance > 1e5, axis=1) | np.any(distance > 1e5, axis=0)
                )
                keep = np.where(~outlier_mask)[0]
                points = points[keep]
                distance = distance[keep][:, keep]

            n_needed = self.num_loc + 1  # 1 depot + num_loc customers
            if len(points) < n_needed:
                log.warning(
                    f"{city}: only {len(points)} points, need {n_needed}. Skipping."
                )
                continue

            self._city_data[city] = {
                "points": points.astype(np.float32),
                "distance": distance.astype(np.float32),
            }

        self._city_names = list(self._city_data.keys())
        if len(self._city_names) == 0:
            raise RuntimeError(
                f"No valid cities found in {city_data_dir} for split={split}"
            )
        log.info(
            f"Loaded {len(self._city_names)} cities for real-world sampling "
            f"(split={split}, loc_dist={loc_dist})"
        )

    def _generate(self, batch_size) -> TensorDict:
        """Generate HCVRP data with real-world OSRM distance matrices.

        Returns TensorDict with:
            - depot [B, 2], locs [B, N, 2] (normalised)
            - demand [B, N], capacity [B, m], speed [B, m]
            - distance_matrix [B, N+1, N+1] (raw, not normalised)
            - num_agents [B]
        """
        if isinstance(batch_size, int):
            batch_size = [batch_size]
        B = batch_size[0]

        # ── Sample from real-world cities ──
        all_points = []
        all_distance = []
        rng = np.random.default_rng()

        for _ in range(B):
            # Pick a random city
            city = self._city_names[rng.integers(len(self._city_names))]
            data = self._city_data[city]
            n_available = len(data["points"])
            n_needed = self.num_loc + 1

            # Sample indices
            if self.loc_dist == "cluster":
                center_idx = rng.integers(n_available)
                dists = np.linalg.norm(
                    data["points"] - data["points"][center_idx], axis=1
                )
                idx = np.argsort(dists)[:n_needed]
            else:  # uniform
                idx = rng.choice(n_available, n_needed, replace=False)

            all_points.append(data["points"][idx])
            all_distance.append(data["distance"][idx][:, idx])

        # Stack and normalise coordinates
        points = np.stack(all_points, axis=0)  # [B, N+1, 2]
        p_min = points.min(axis=1, keepdims=True)
        p_max = points.max(axis=1, keepdims=True)
        denom = np.where(p_max - p_min == 0, 1, p_max - p_min)
        points_norm = (points - p_min) / denom

        depot = torch.tensor(points_norm[:, 0, :], dtype=torch.float32)
        locs = torch.tensor(points_norm[:, 1:, :], dtype=torch.float32)
        distance_matrix = torch.tensor(
            np.stack(all_distance, axis=0), dtype=torch.float32
        )

        # ── Generate standard HCVRP fields via parent ──
        # We override locs/depot but need parent for demand, capacity, speed
        td_parent = super()._generate(batch_size)

        # Replace coordinates with real-world ones
        td_parent.set("depot", depot)
        td_parent.set("locs", locs)
        td_parent.set("distance_matrix", distance_matrix)

        return td_parent


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
        balance_weight: float = 0.0,
        **kwargs,
    ):
        # Build our generator before calling super().__init__
        # Auto-dispatch to HCVRPRealWorldGenerator when city_data_dir is provided
        if generator is None:
            if "city_data_dir" in generator_params:
                generator = HCVRPRealWorldGenerator(**generator_params)
            else:
                generator = HCVRPRealGenerator(**generator_params)
        self.normalize_distance = normalize_distance
        self.balance_weight = balance_weight
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

        # --- Track per-agent visit counts for workload balancing ---
        reset_batch_size = td_reset.batch_size
        reset_num_agents = td_reset["agents_speed"].shape[-1]
        td_reset.set(
            "num_visits_per_agent",
            torch.zeros(
                (*reset_batch_size, reset_num_agents),
                dtype=torch.float32,
                device=device,
            ),
        )

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
        Euclidean distance, capacity, visited, action_mask). We also track
        per-agent visit counts for workload balancing.
        """
        num_agents = td["current_node"].size(-1)

        # Determine which agents are visiting a customer (not depot)
        is_customer = td["action"] >= num_agents  # [B, m]

        # Determine which agents are staying at the same node (no new visit)
        is_staying = td["action"] == td["current_node"]  # [B, m]

        # Parent handles everything — distance_matrix is already in td
        # and TensorDict preserves keys through updates
        td = super()._step(td)

        # Update per-agent visit counts (only for new customer visits)
        if "num_visits_per_agent" in td.keys():
            new_visit = (is_customer & ~is_staying).float()  # [B, m]
            td["num_visits_per_agent"] = td["num_visits_per_agent"] + new_visit

        return td

    def _get_reward(self, td: TensorDict, actions: TensorDict) -> TensorDict:
        """Compute reward: PARCO's min-max time + workload balance penalty.

        Extends the original reward with a workload balance term that
        penalizes uneven agent completion times:

            reward = -(makespan + balance_weight * time_std)

        where:
        - makespan = max_i(tour_length_i / speed_i)  (PARCO original)
        - time_std = std(tour_length_i / speed_i)     (balance penalty)
        - balance_weight: hyperparameter (0.0 = disabled, default)

        The balance penalty encourages the model to distribute work more
        evenly across agents, reducing the gap between the busiest and
        least busy agents.

        Returns:
            Negative (makespan + balance_weight * time_std) per batch.
        """
        from rl4co.utils.ops import gather_by_index, get_distance

        current_length = td["current_length"]

        # Adding the final distance to the depot
        current_loc = gather_by_index(td["locs"], td["current_node"])
        depot_loc = gather_by_index(td["locs"], td["depot_node"])
        current_length = current_length + get_distance(depot_loc, current_loc)

        # Calculate per-agent completion time
        current_time = current_length / td["agents_speed"]  # [B, m]

        # Original makespan objective
        max_time = current_time.max(dim=1)[0]  # [B]

        if self.balance_weight > 0:
            # Workload balance penalty: std of agent completion times
            time_std = current_time.std(dim=1)  # [B]
            reward = -(max_time + self.balance_weight * time_std)
        else:
            reward = -max_time

        return reward
