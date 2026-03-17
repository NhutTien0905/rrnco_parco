"""Hybrid Environment: HCVRP with distance matrix support.

This module will be fully implemented in Phase 2.
It extends PARCO's HCVRPEnv to include distance matrices in the TensorDict.
"""


class HCVRPRealEnv:
    """HCVRP environment extended with real-world distance matrices.

    Extends PARCO's HCVRPEnv to include:
    - distance_matrix: [B, m+N, m+N] — real-world or Euclidean distances
    - duration_matrix: [B, m+N, m+N] — optional travel time matrix
    - min_distance, max_distance: for denormalization

    The generator can either:
    1. Compute Euclidean distances from coordinates (default)
    2. Load real-world OSRM distances from pre-computed data
    """

    def __init__(self, **kwargs):
        # TODO: Phase 2 — extend HCVRPEnv
        raise NotImplementedError(
            "HCVRPRealEnv will be implemented in Phase 2. "
            "See plan.md for architecture details."
        )
