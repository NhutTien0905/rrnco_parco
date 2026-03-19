"""Phase 2 Tests: Hybrid Environment (HCVRPRealEnv + HCVRPRealGenerator).

Tests cover:
- Generator output shapes and properties
- Distance matrix computation (Euclidean)
- Environment reset TensorDict shapes
- Distance matrix expansion (depot → agents)
- Per-instance min-max normalization
- Environment step cycle
- Action mask correctness
- Reward computation
- Backward compatibility (no distance matrix)
"""

import pytest
import torch
from tensordict.tensordict import TensorDict

from hybrid.envs import HCVRPRealEnv, HCVRPRealGenerator


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def small_params():
    """Small problem for fast testing: 10 customers, 3 agents."""
    return dict(num_loc=10, num_agents=3)


@pytest.fixture
def generator(small_params):
    return HCVRPRealGenerator(**small_params)


@pytest.fixture
def env(small_params):
    return HCVRPRealEnv(generator_params=small_params)


@pytest.fixture
def env_no_norm(small_params):
    return HCVRPRealEnv(generator_params=small_params, normalize_distance=False)


@pytest.fixture
def batch_size():
    return [4]


@pytest.fixture
def td_gen(generator, batch_size):
    """TensorDict from generator (before env reset)."""
    return generator._generate(batch_size)


@pytest.fixture
def td_reset(env, td_gen, batch_size):
    """TensorDict after env reset."""
    return env._reset(td_gen, batch_size)


# ─── Generator Tests ────────────────────────────────────────────────────────

class TestHCVRPRealGenerator:
    """Tests for the generator."""

    def test_inherits_hcvrp_generator(self):
        from parco.envs.hcvrp.generator import HCVRPGenerator
        assert issubclass(HCVRPRealGenerator, HCVRPGenerator)

    def test_generates_standard_keys(self, td_gen):
        """Generator should produce all standard HCVRP keys."""
        for key in ["locs", "depot", "demand", "capacity", "speed"]:
            assert key in td_gen.keys(), f"Missing key: {key}"

    def test_generates_distance_matrix(self, td_gen):
        """Generator should produce distance_matrix when compute_distance_matrix=True."""
        assert "distance_matrix" in td_gen.keys()

    def test_distance_matrix_shape(self, td_gen, small_params, batch_size):
        """distance_matrix should be [B, N+1, N+1] (depot + customers)."""
        num_loc = small_params["num_loc"]
        expected = (*batch_size, num_loc + 1, num_loc + 1)
        assert td_gen["distance_matrix"].shape == expected

    def test_distance_matrix_symmetric(self, td_gen):
        """Euclidean distance matrix should be symmetric."""
        dm = td_gen["distance_matrix"]
        assert torch.allclose(dm, dm.transpose(-2, -1), atol=1e-5)

    def test_distance_matrix_nonneg(self, td_gen):
        """All distances should be non-negative."""
        assert (td_gen["distance_matrix"] >= 0).all()

    def test_distance_matrix_zero_diagonal(self, td_gen):
        """Diagonal entries (self-distance) should be approximately zero.
        Note: torch.cdist with use_mm_for_euclid_dist can have tiny numerical noise."""
        dm = td_gen["distance_matrix"]
        diag = torch.diagonal(dm, dim1=-2, dim2=-1)
        assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-3)

    def test_no_distance_matrix_when_disabled(self, small_params, batch_size):
        """Generator with compute_distance_matrix=False should not produce distance_matrix."""
        gen = HCVRPRealGenerator(compute_distance_matrix=False, **small_params)
        td = gen._generate(batch_size)
        assert "distance_matrix" not in td.keys()

    def test_locs_shape(self, td_gen, small_params, batch_size):
        expected = (*batch_size, small_params["num_loc"], 2)
        assert td_gen["locs"].shape == expected

    def test_depot_shape(self, td_gen, batch_size):
        expected = (*batch_size, 2)
        assert td_gen["depot"].shape == expected

    def test_demand_shape(self, td_gen, small_params, batch_size):
        expected = (*batch_size, small_params["num_loc"])
        assert td_gen["demand"].shape == expected

    def test_capacity_shape(self, td_gen, small_params, batch_size):
        expected = (*batch_size, small_params["num_agents"])
        assert td_gen["capacity"].shape == expected

    def test_speed_shape(self, td_gen, small_params, batch_size):
        expected = (*batch_size, small_params["num_agents"])
        assert td_gen["speed"].shape == expected


# ─── Environment Reset Tests ────────────────────────────────────────────────

class TestHCVRPRealEnvReset:
    """Tests for _reset() method."""

    def test_inherits_hcvrp_env(self):
        from parco.envs.hcvrp.env import HCVRPEnv
        assert issubclass(HCVRPRealEnv, HCVRPEnv)

    def test_has_standard_keys(self, td_reset):
        """Reset TensorDict should have all standard HCVRP keys."""
        required = [
            "locs", "demand", "current_length", "current_node", "depot_node",
            "used_capacity", "agents_capacity", "agents_speed", "i",
            "visited", "action_mask", "done",
        ]
        for key in required:
            assert key in td_reset.keys(), f"Missing standard key: {key}"

    def test_has_distance_matrix(self, td_reset):
        """Reset TensorDict should include distance_matrix."""
        assert "distance_matrix" in td_reset.keys()

    def test_distance_matrix_expanded_shape(self, td_reset, small_params, batch_size):
        """distance_matrix should be [B, m+N, m+N] after agent expansion."""
        m = small_params["num_agents"]
        n = small_params["num_loc"]
        expected = (*batch_size, m + n, m + n)
        assert td_reset["distance_matrix"].shape == expected

    def test_locs_shape(self, td_reset, small_params, batch_size):
        """locs should be [B, m+N, 2] (agent depots + customers)."""
        m = small_params["num_agents"]
        n = small_params["num_loc"]
        expected = (*batch_size, m + n, 2)
        assert td_reset["locs"].shape == expected

    def test_has_normalization_metadata(self, td_reset, batch_size):
        """Should have min_distance and max_distance when normalize=True."""
        assert "min_distance" in td_reset.keys()
        assert "max_distance" in td_reset.keys()
        assert td_reset["min_distance"].shape == tuple(batch_size)
        assert td_reset["max_distance"].shape == tuple(batch_size)

    def test_no_normalization_metadata_when_disabled(self, env_no_norm, td_gen, batch_size):
        """Should NOT have min/max_distance when normalize_distance=False."""
        td_reset = env_no_norm._reset(td_gen, batch_size)
        assert "min_distance" not in td_reset.keys()
        assert "max_distance" not in td_reset.keys()

    def test_distance_matrix_normalized_range(self, td_reset):
        """Normalized distance matrix should be in [0, 1] range."""
        dm = td_reset["distance_matrix"]
        assert dm.min() >= -1e-6, f"Min distance {dm.min()} < 0"
        assert dm.max() <= 1.0 + 1e-6, f"Max distance {dm.max()} > 1"

    def test_distance_matrix_unnormalized(self, env_no_norm, td_gen, batch_size):
        """Without normalization, distances should match raw Euclidean."""
        td_reset = env_no_norm._reset(td_gen, batch_size)
        dm = td_reset["distance_matrix"]
        # Unnormalized Euclidean distances in [0,1] unit square:
        # max possible = sqrt(2) ≈ 1.414
        assert dm.max() <= 1.5  # sanity check

    def test_expanded_distance_matrix_depot_rows_equal(self, td_reset, small_params):
        """All agent depot rows should have identical distances (same physical location)."""
        m = small_params["num_agents"]
        dm = td_reset["distance_matrix"]
        # Rows 0..m-1 should all be identical (depot distances)
        for i in range(1, m):
            assert torch.allclose(dm[:, 0, m:], dm[:, i, m:], atol=1e-5), \
                f"Agent depot row {i} differs from row 0"

    def test_expanded_distance_matrix_depot_cols_equal(self, td_reset, small_params):
        """All agent depot columns should have identical distances."""
        m = small_params["num_agents"]
        dm = td_reset["distance_matrix"]
        for i in range(1, m):
            assert torch.allclose(dm[:, m:, 0], dm[:, m:, i], atol=1e-5), \
                f"Agent depot col {i} differs from col 0"

    def test_expanded_distance_matrix_symmetric(self, td_reset):
        """Expanded distance matrix should still be symmetric."""
        dm = td_reset["distance_matrix"]
        assert torch.allclose(dm, dm.transpose(-2, -1), atol=1e-5)

    def test_expanded_distance_matrix_zero_diagonal(self, td_reset):
        """Diagonal should be approximately zero after expansion and normalization.
        Note: torch.cdist with use_mm_for_euclid_dist + normalization can amplify tiny noise."""
        dm = td_reset["distance_matrix"]
        diag = torch.diagonal(dm, dim1=-2, dim2=-1)
        assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-3)

    def test_action_mask_shape(self, td_reset, small_params, batch_size):
        """action_mask should be [B, m, m+N]."""
        m = small_params["num_agents"]
        n = small_params["num_loc"]
        expected = (*batch_size, m, m + n)
        assert td_reset["action_mask"].shape == expected

    def test_demand_shape(self, td_reset, small_params, batch_size):
        """demand should be [B, m, m+N] (replicated per agent)."""
        m = small_params["num_agents"]
        n = small_params["num_loc"]
        expected = (*batch_size, m, m + n)
        assert td_reset["demand"].shape == expected

    def test_current_node_init(self, td_reset, small_params, batch_size):
        """Each agent should start at its own depot index."""
        m = small_params["num_agents"]
        expected = torch.arange(m).unsqueeze(0).expand(*batch_size, m)
        assert (td_reset["current_node"] == expected).all()


# ─── Environment Step Tests ──────────────────────────────────────────────────

class TestHCVRPRealEnvStep:
    """Tests for _step() and full rollout."""

    def test_step_preserves_distance_matrix(self, env, td_gen, batch_size, small_params):
        """distance_matrix should persist through steps."""
        td = env._reset(td_gen, batch_size)
        m = small_params["num_agents"]
        n = small_params["num_loc"]

        # Take a valid action: each agent visits a different customer
        action = torch.arange(m, m + m).unsqueeze(0).expand(*batch_size, m)
        td.set("action", action)
        td = env._step(td)

        assert "distance_matrix" in td.keys()
        expected_shape = (*batch_size, m + n, m + n)
        assert td["distance_matrix"].shape == expected_shape

    def test_step_preserves_normalization_metadata(self, env, td_gen, batch_size, small_params):
        """min_distance and max_distance should persist through steps."""
        td = env._reset(td_gen, batch_size)
        m = small_params["num_agents"]

        action = torch.arange(m, m + m).unsqueeze(0).expand(*batch_size, m)
        td.set("action", action)
        td = env._step(td)

        assert "min_distance" in td.keys()
        assert "max_distance" in td.keys()

    def test_step_updates_visited(self, env, td_gen, batch_size, small_params):
        """Visited nodes should be marked after step."""
        td = env._reset(td_gen, batch_size)
        m = small_params["num_agents"]

        action = torch.arange(m, m + m).unsqueeze(0).expand(*batch_size, m)
        td.set("action", action)
        td = env._step(td)

        # Customer nodes m..2m-1 should now be visited
        for i in range(m):
            assert td["visited"][:, m + i].all()

    def test_step_updates_current_node(self, env, td_gen, batch_size, small_params):
        """current_node should reflect the last action."""
        td = env._reset(td_gen, batch_size)
        m = small_params["num_agents"]

        action = torch.arange(m, m + m).unsqueeze(0).expand(*batch_size, m)
        td.set("action", action)
        td = env._step(td)

        assert (td["current_node"] == action).all()

    def test_full_rollout_greedy(self, env, small_params, batch_size):
        """Run a full greedy rollout: agents visit customers round-robin, then return to depot."""
        m = small_params["num_agents"]
        n = small_params["num_loc"]
        td = env.generator._generate(batch_size)
        td = env._reset(td, batch_size)

        # Assign customers round-robin to agents
        customer_indices = list(range(m, m + n))  # customer node indices
        steps = []
        for start in range(0, n, m):
            action = []
            for agent_idx in range(m):
                cust_idx = start + agent_idx
                if cust_idx < n:
                    action.append(m + cust_idx)
                else:
                    # Agent goes to its depot if no more customers
                    action.append(agent_idx)
            steps.append(action)

        # Execute steps
        for step_action in steps:
            action = torch.tensor(step_action).unsqueeze(0).expand(*batch_size, m)
            td.set("action", action)
            td = env._step(td)

        # Send all agents back to depot
        depot_action = torch.arange(m).unsqueeze(0).expand(*batch_size, m)
        td.set("action", depot_action)
        td = env._step(td)

        # All customers should be visited
        assert td["visited"][:, m:].all(), "Not all customers visited"

    def test_reward_computation(self, env, small_params, batch_size):
        """Reward should be negative (min-max time) and finite."""
        m = small_params["num_agents"]
        n = small_params["num_loc"]
        td = env.generator._generate(batch_size)
        td = env._reset(td, batch_size)

        # Simple rollout: assign customers in order
        for start in range(0, n, m):
            action = []
            for agent_idx in range(m):
                cust_idx = start + agent_idx
                if cust_idx < n:
                    action.append(m + cust_idx)
                else:
                    action.append(agent_idx)
            action_tensor = torch.tensor(action).unsqueeze(0).expand(*batch_size, m)
            td.set("action", action_tensor)
            td = env._step(td)

        # Return to depot
        depot_action = torch.arange(m).unsqueeze(0).expand(*batch_size, m)
        td.set("action", depot_action)
        td = env._step(td)

        reward = env._get_reward(td, None)
        assert reward.shape == tuple(batch_size)
        assert (reward < 0).all(), "Reward should be negative (min-max time)"
        assert torch.isfinite(reward).all(), "Reward should be finite"


# ─── Backward Compatibility Tests ───────────────────────────────────────────

class TestBackwardCompatibility:
    """Test that env works without pre-computed distance matrix."""

    def test_euclidean_fallback(self, small_params, batch_size):
        """If generator doesn't produce distance_matrix, env should compute Euclidean."""
        gen = HCVRPRealGenerator(compute_distance_matrix=False, **small_params)
        env = HCVRPRealEnv(generator=gen)
        td = gen._generate(batch_size)
        assert "distance_matrix" not in td.keys()

        td_reset = env._reset(td, batch_size)
        assert "distance_matrix" in td_reset.keys()

        m = small_params["num_agents"]
        n = small_params["num_loc"]
        expected_shape = (*batch_size, m + n, m + n)
        assert td_reset["distance_matrix"].shape == expected_shape

    def test_euclidean_fallback_matches_explicit(self, small_params, batch_size):
        """Euclidean fallback should give same result as explicit computation."""
        gen = HCVRPRealGenerator(compute_distance_matrix=True, **small_params)
        env_with = HCVRPRealEnv(generator=gen, normalize_distance=False)
        env_without = HCVRPRealEnv(
            generator=HCVRPRealGenerator(compute_distance_matrix=False, **small_params),
            normalize_distance=False,
        )

        # Generate data with distance matrix, then make a copy without it
        torch.manual_seed(42)
        td = gen._generate(batch_size)
        td_no_dm = td.select(*[k for k in td.keys() if k != "distance_matrix"])

        td_reset_with = env_with._reset(td.clone(), batch_size)
        td_reset_without = env_without._reset(td_no_dm.clone(), batch_size)

        assert torch.allclose(
            td_reset_with["distance_matrix"],
            td_reset_without["distance_matrix"],
            atol=1e-3,
        ), "Euclidean fallback should match explicit distance matrix"


# ─── Edge Case Tests ────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_single_agent(self, batch_size):
        """Should work with a single agent."""
        env = HCVRPRealEnv(generator_params=dict(num_loc=5, num_agents=1))
        td = env.generator._generate(batch_size)
        td_reset = env._reset(td, batch_size)

        assert td_reset["distance_matrix"].shape == (*batch_size, 6, 6)
        assert td_reset["locs"].shape == (*batch_size, 6, 2)
        assert td_reset["action_mask"].shape == (*batch_size, 1, 6)

    def test_many_agents(self, batch_size):
        """Should work with many agents."""
        env = HCVRPRealEnv(generator_params=dict(num_loc=20, num_agents=10))
        td = env.generator._generate(batch_size)
        td_reset = env._reset(td, batch_size)

        assert td_reset["distance_matrix"].shape == (*batch_size, 30, 30)
        assert td_reset["locs"].shape == (*batch_size, 30, 2)

    def test_batch_size_1(self):
        """Should work with batch_size=1."""
        env = HCVRPRealEnv(generator_params=dict(num_loc=5, num_agents=2))
        td = env.generator._generate([1])
        td_reset = env._reset(td, [1])
        assert td_reset["distance_matrix"].shape == (1, 7, 7)

    def test_larger_batch(self):
        """Should work with larger batch sizes."""
        env = HCVRPRealEnv(generator_params=dict(num_loc=5, num_agents=2))
        td = env.generator._generate([32])
        td_reset = env._reset(td, [32])
        assert td_reset["distance_matrix"].shape == (32, 7, 7)

    def test_distance_matrix_dtype(self, td_reset):
        """Distance matrix should be float32."""
        assert td_reset["distance_matrix"].dtype == torch.float32

    def test_normalization_per_instance(self, batch_size):
        """Each instance should be normalized independently."""
        env = HCVRPRealEnv(generator_params=dict(num_loc=10, num_agents=3))
        td = env.generator._generate(batch_size)
        td_reset = env._reset(td, batch_size)

        dm = td_reset["distance_matrix"]
        for b in range(batch_size[0]):
            instance_dm = dm[b]
            # Check that max normalized value ≈ 1 (max distance maps to 1)
            assert instance_dm.max() > 0.99, f"Instance {b} max not close to 1"
            # Check that min normalized value ≈ 0 (min distance = 0 from diagonal)
            assert instance_dm.min() < 0.01, f"Instance {b} min not close to 0"
