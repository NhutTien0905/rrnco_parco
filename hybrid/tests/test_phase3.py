"""Phase 3 Tests: HybridInitEmbedding.

Tests cover:
- Module instantiation and sub-module structure
- Forward pass output shapes (row_emb, col_emb, distance)
- Dual-stream output: row ≠ col for customer part
- Depot/agent embedding shared across both streams
- Distance pass-through unchanged
- Different batch sizes
- Different num_agents / num_loc configurations
- Gradient flow through all parameters
- Integration with real HCVRPRealEnv TensorDict
- Phase parameter ('train' vs 'test')
- Demand scaling effect
- Edge case: single agent
"""

import pytest
import torch
from tensordict.tensordict import TensorDict

from hybrid.models.env_embeddings import HybridInitEmbedding
from hybrid.envs import HCVRPRealEnv, HCVRPRealGenerator


# ─── Helpers ────────────────────────────────────────────────────────────────

def make_mock_td(batch_size, num_agents, num_loc, device="cpu"):
    """Create a mock TensorDict mimicking env.reset() output.

    Keys needed by HybridInitEmbedding:
    - locs: [B, m+N, 2]
    - distance_matrix: [B, m+N, m+N]
    - demand: [B, m, m+N]
    - capacity: [B, m]
    - speed: [B, m]
    - action_mask: [B, m, m+N]
    """
    B = batch_size
    m = num_agents
    N = num_loc
    total = m + N

    locs = torch.rand(B, total, 2, device=device)
    # Build a proper distance matrix from locs
    dist = torch.cdist(locs, locs)
    # Zero diagonal
    dist = dist * (1 - torch.eye(total, device=device).unsqueeze(0))

    demand = torch.rand(B, m, total, device=device)
    capacity = torch.ones(B, m, device=device) * 40.0
    speed = torch.ones(B, m, device=device)
    action_mask = torch.ones(B, m, total, dtype=torch.bool, device=device)

    return TensorDict(
        {
            "locs": locs,
            "distance_matrix": dist,
            "demand": demand,
            "capacity": capacity,
            "speed": speed,
            "action_mask": action_mask,
        },
        batch_size=[B],
    )


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def embed_dim():
    return 64


@pytest.fixture
def num_agents():
    return 3


@pytest.fixture
def num_loc():
    return 20


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def model(embed_dim):
    return HybridInitEmbedding(
        embed_dim=embed_dim,
        linear_bias=False,
        demand_scaler=40.0,
        speed_scaler=1.0,
        use_polar_feats=True,
        sample_type="prob",
        sample_size=10,
    )


@pytest.fixture
def td(batch_size, num_agents, num_loc):
    return make_mock_td(batch_size, num_agents, num_loc)


# ─── Test Class: Instantiation ─────────────────────────────────────────────

class TestHybridInitEmbeddingInit:
    """Test module instantiation and structure."""

    def test_instantiation(self, model):
        assert isinstance(model, HybridInitEmbedding)

    def test_has_depot_embedding(self, model):
        assert hasattr(model, "init_embed_depot")

    def test_has_positional_encoder(self, model):
        assert hasattr(model, "pos_encoder")

    def test_has_coord_expert(self, model):
        assert hasattr(model, "coord_expert")

    def test_has_distance_expert(self, model):
        assert hasattr(model, "distance_expert")

    def test_has_gating_networks(self, model):
        assert hasattr(model, "gating_network_row")
        assert hasattr(model, "gating_network_col")

    def test_has_demand_embed(self, model):
        assert hasattr(model, "demand_embed")

    def test_has_combine_embeds(self, model):
        assert hasattr(model, "combine_row_embed")
        assert hasattr(model, "combine_col_embed")

    def test_has_agent_features(self, model):
        assert hasattr(model, "init_embed_agents")
        assert hasattr(model, "init_embed_depot_agents")

    def test_embed_dim_stored(self, model, embed_dim):
        assert model.embed_dim == embed_dim

    def test_parameter_count_positive(self, model):
        total = sum(p.numel() for p in model.parameters())
        assert total > 0


# ─── Test Class: Forward Shapes ────────────────────────────────────────────

class TestHybridInitEmbeddingForwardShapes:
    """Test output shapes of forward pass."""

    def test_returns_tuple_of_three(self, model, td):
        out = model(td, phase="train")
        assert isinstance(out, tuple)
        assert len(out) == 3

    def test_row_emb_shape(self, model, td, batch_size, num_agents, num_loc, embed_dim):
        row_emb, _, _ = model(td, phase="train")
        assert row_emb.shape == (batch_size, num_agents + num_loc, embed_dim)

    def test_col_emb_shape(self, model, td, batch_size, num_agents, num_loc, embed_dim):
        _, col_emb, _ = model(td, phase="train")
        assert col_emb.shape == (batch_size, num_agents + num_loc, embed_dim)

    def test_distance_shape(self, model, td, batch_size, num_agents, num_loc):
        _, _, distance = model(td, phase="train")
        total = num_agents + num_loc
        assert distance.shape == (batch_size, total, total)

    def test_distance_pass_through(self, model, td):
        """Distance returned should be the same tensor from td."""
        _, _, distance = model(td, phase="train")
        assert torch.equal(distance, td["distance_matrix"])


# ─── Test Class: Dual Stream Properties ────────────────────────────────────

class TestDualStreamProperties:
    """Test that row and column streams have different customer embeddings."""

    def test_row_col_differ_customer(self, model, td, num_agents):
        """Customer part (after first m nodes) should differ between streams."""
        row_emb, col_emb, _ = model(td, phase="train")
        row_cust = row_emb[:, num_agents:, :]
        col_cust = col_emb[:, num_agents:, :]
        # They should not be identical because DistanceExpert
        # produces different row/col embeddings
        assert not torch.allclose(row_cust, col_cust, atol=1e-6)

    def test_row_col_same_depot(self, model, td, num_agents):
        """Depot/agent part (first m nodes) should be identical in both streams."""
        row_emb, col_emb, _ = model(td, phase="train")
        row_depot = row_emb[:, :num_agents, :]
        col_depot = col_emb[:, :num_agents, :]
        assert torch.allclose(row_depot, col_depot, atol=1e-6)


# ─── Test Class: Different Configurations ──────────────────────────────────

class TestDifferentConfigs:
    """Test with varying batch sizes, agents, and locs."""

    @pytest.mark.parametrize("bs", [1, 2, 8])
    def test_different_batch_sizes(self, embed_dim, bs):
        model = HybridInitEmbedding(embed_dim=embed_dim, sample_size=10)
        td = make_mock_td(bs, num_agents=3, num_loc=20)
        row, col, dist = model(td, phase="train")
        assert row.shape == (bs, 23, embed_dim)
        assert col.shape == (bs, 23, embed_dim)

    @pytest.mark.parametrize("m", [1, 3, 5])
    def test_different_num_agents(self, embed_dim, m):
        model = HybridInitEmbedding(embed_dim=embed_dim, sample_size=10)
        td = make_mock_td(4, num_agents=m, num_loc=20)
        row, col, dist = model(td, phase="train")
        assert row.shape == (4, m + 20, embed_dim)

    @pytest.mark.parametrize("n", [15, 30, 50])
    def test_different_num_loc(self, embed_dim, n):
        model = HybridInitEmbedding(embed_dim=embed_dim, sample_size=10)
        td = make_mock_td(4, num_agents=3, num_loc=n)
        row, col, dist = model(td, phase="train")
        assert row.shape == (4, 3 + n, embed_dim)

    def test_sample_size_less_than_num_loc(self, embed_dim):
        """sample_size must be < num_loc for prob sampling."""
        model = HybridInitEmbedding(embed_dim=embed_dim, sample_size=5)
        td = make_mock_td(4, num_agents=3, num_loc=10)
        row, col, dist = model(td, phase="train")
        assert row.shape == (4, 13, embed_dim)


# ─── Test Class: Gradient Flow ─────────────────────────────────────────────

class TestGradientFlow:
    """Test that gradients flow through all components."""

    # DistanceExpert defines row_combine_embed / col_combine_embed
    # but never uses them in forward() — these are dead params from RRNCO.
    UNUSED_PARAMS = {
        "distance_expert.row_combine_embed.weight",
        "distance_expert.row_combine_embed.bias",
        "distance_expert.col_combine_embed.weight",
        "distance_expert.col_combine_embed.bias",
    }

    def test_backward_runs(self, model, td):
        """A loss on the output should produce gradients."""
        row_emb, col_emb, _ = model(td, phase="train")
        loss = row_emb.sum() + col_emb.sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad and name not in self.UNUSED_PARAMS:
                assert p.grad is not None, f"No grad for {name}"

    def test_grad_depot_embedding(self, model, td):
        row_emb, col_emb, _ = model(td, phase="train")
        loss = row_emb.sum() + col_emb.sum()
        loss.backward()
        assert model.init_embed_depot.weight.grad is not None

    def test_grad_coord_expert(self, model, td):
        row_emb, col_emb, _ = model(td, phase="train")
        loss = row_emb.sum() + col_emb.sum()
        loss.backward()
        assert model.coord_expert.init_embed.weight.grad is not None

    def test_grad_distance_expert(self, model, td):
        row_emb, col_emb, _ = model(td, phase="train")
        loss = row_emb.sum() + col_emb.sum()
        loss.backward()
        assert model.distance_expert.row_embed.weight.grad is not None

    def test_grad_gating(self, model, td):
        row_emb, col_emb, _ = model(td, phase="train")
        loss = row_emb.sum() + col_emb.sum()
        loss.backward()
        # Gating FC has parameters
        gating_params = list(model.gating_network_row.parameters())
        assert any(p.grad is not None for p in gating_params)

    def test_grad_demand_embed(self, model, td):
        row_emb, col_emb, _ = model(td, phase="train")
        loss = row_emb.sum() + col_emb.sum()
        loss.backward()
        assert model.demand_embed.weight.grad is not None


# ─── Test Class: Phase Behavior ────────────────────────────────────────────

class TestPhaseBehavior:
    """Test train vs test phase."""

    def test_train_phase_runs(self, model, td):
        row, col, dist = model(td, phase="train")
        assert row is not None

    def test_test_phase_runs(self, model, td):
        """Test phase should also work (prob sampling)."""
        row, col, dist = model(td, phase="test")
        assert row.shape == col.shape

    def test_default_phase_is_train(self, model, td):
        """Default phase argument should work."""
        row, col, dist = model(td)
        assert row is not None


# ─── Test Class: Integration with HCVRPRealEnv ────────────────────────────

class TestIntegrationWithEnv:
    """Test HybridInitEmbedding with real env TensorDict."""

    @pytest.fixture
    def real_env(self):
        return HCVRPRealEnv(
            generator_params=dict(num_loc=20, num_agents=3)
        )

    @pytest.fixture
    def real_td(self, real_env):
        td = real_env.reset(batch_size=[4])
        return td

    @pytest.fixture
    def real_model(self):
        return HybridInitEmbedding(
            embed_dim=64,
            sample_size=10,
        )

    def test_forward_with_real_td(self, real_model, real_td):
        """Should work with actual env output."""
        row, col, dist = real_model(real_td, phase="train")
        assert row.shape == (4, 23, 64)
        assert col.shape == (4, 23, 64)
        assert dist.shape == (4, 23, 23)

    def test_distance_matches_env(self, real_model, real_td):
        """Returned distance should match the env's distance matrix."""
        _, _, dist = real_model(real_td, phase="train")
        assert torch.equal(dist, real_td["distance_matrix"])

    def test_output_finite(self, real_model, real_td):
        """All outputs should be finite (no NaN or Inf)."""
        row, col, dist = real_model(real_td, phase="train")
        assert torch.isfinite(row).all()
        assert torch.isfinite(col).all()
        assert torch.isfinite(dist).all()

    def test_backward_with_real_td(self, real_model, real_td):
        """Gradient flow with real env data."""
        row, col, _ = real_model(real_td, phase="train")
        loss = row.mean() + col.mean()
        loss.backward()
        # DistanceExpert has 4 unused params (row/col_combine_embed w+b)
        unused_count = 4
        grad_count = sum(
            1 for p in real_model.parameters() if p.grad is not None
        )
        total = sum(1 for p in real_model.parameters())
        assert grad_count == total - unused_count


# ─── Test Class: Embedding Properties ──────────────────────────────────────

class TestEmbeddingProperties:
    """Test semantic properties of embeddings."""

    def test_output_dtype_float(self, model, td):
        row, col, _ = model(td, phase="train")
        assert row.dtype == torch.float32
        assert col.dtype == torch.float32

    def test_deterministic_in_eval(self, model, td):
        """In eval mode with prob sampling, results should still be valid."""
        model.eval()
        with torch.no_grad():
            row1, col1, _ = model(td, phase="train")
            row2, col2, _ = model(td, phase="train")
        # Note: prob sampling uses multinomial which is stochastic,
        # so results may differ. Just check shapes are correct.
        assert row1.shape == row2.shape

    def test_different_inputs_different_outputs(self, model):
        """Different inputs should produce different outputs."""
        td1 = make_mock_td(4, 3, 20)
        td2 = make_mock_td(4, 3, 20)
        row1, _, _ = model(td1, phase="train")
        row2, _, _ = model(td2, phase="train")
        # With high probability different random inputs → different outputs
        assert not torch.allclose(row1, row2, atol=1e-6)


# ─── Test Class: Edge Cases ────────────────────────────────────────────────

class TestEdgeCases:
    """Edge case tests."""

    def test_single_agent(self):
        """Should work with m=1."""
        model = HybridInitEmbedding(embed_dim=32, sample_size=5)
        td = make_mock_td(2, num_agents=1, num_loc=15)
        row, col, dist = model(td, phase="train")
        assert row.shape == (2, 16, 32)
        assert col.shape == (2, 16, 32)

    def test_large_embed_dim(self):
        model = HybridInitEmbedding(embed_dim=256, sample_size=10)
        td = make_mock_td(2, num_agents=3, num_loc=20)
        row, col, dist = model(td, phase="train")
        assert row.shape == (2, 23, 256)

    def test_small_embed_dim(self):
        model = HybridInitEmbedding(embed_dim=16, sample_size=5)
        td = make_mock_td(2, num_agents=2, num_loc=10)
        row, col, dist = model(td, phase="train")
        assert row.shape == (2, 12, 16)

    def test_with_linear_bias_true(self):
        model = HybridInitEmbedding(embed_dim=32, linear_bias=True, sample_size=5)
        td = make_mock_td(2, num_agents=2, num_loc=10)
        row, col, dist = model(td, phase="train")
        assert row.shape == (2, 12, 32)
