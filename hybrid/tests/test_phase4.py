"""Phase 4 Tests: HybridEncoder.

Tests cover:
- Module instantiation and sub-module structure
- Forward pass output shapes (h, init_h)
- Interface compatibility with PARCOEncoder (returns tuple of two tensors)
- Fusion types: concat, gate, add
- AttnFreeNet layer count
- Different configurations (embed_dim, num_layers, nab_type)
- Gradient flow through all components
- POS token support
- Integration with HCVRPRealEnv TensorDict
- Phase parameter forwarding
- init_h differs from h (encoder actually transforms)
"""

import pytest
import torch
from tensordict.tensordict import TensorDict

from hybrid.models.encoder import HybridEncoder
from hybrid.models.env_embeddings import HybridInitEmbedding
from hybrid.envs import HCVRPRealEnv


# ─── Helpers ────────────────────────────────────────────────────────────────

def make_mock_td(batch_size, num_agents, num_loc, device="cpu"):
    """Create a mock TensorDict mimicking env.reset() output."""
    B, m, N = batch_size, num_agents, num_loc
    total = m + N

    locs = torch.rand(B, total, 2, device=device)
    dist = torch.cdist(locs, locs)
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
def encoder(embed_dim):
    """Default encoder with small config for fast testing."""
    return HybridEncoder(
        embed_dim=embed_dim,
        num_heads=4,
        num_layers=2,
        normalization="instance",
        feedforward_hidden=128,
        nab_type="gating",
        init_embedding_kwargs=dict(sample_size=10),
    )


@pytest.fixture
def td(batch_size, num_agents, num_loc):
    return make_mock_td(batch_size, num_agents, num_loc)


# ─── Test Class: Instantiation ─────────────────────────────────────────────

class TestHybridEncoderInit:
    """Test module instantiation and structure."""

    def test_instantiation(self, encoder):
        assert isinstance(encoder, HybridEncoder)

    def test_has_init_embedding(self, encoder):
        assert hasattr(encoder, "init_embedding")
        assert isinstance(encoder.init_embedding, HybridInitEmbedding)

    def test_has_attn_free_net(self, encoder):
        assert hasattr(encoder, "net")
        from rrnco.models.nn.attn_freenet import AttnFreeNet
        assert isinstance(encoder.net, AttnFreeNet)

    def test_has_fusion(self, encoder):
        assert hasattr(encoder, "fusion")

    def test_num_layers(self, encoder):
        """AttnFreeNet should have 2 layers as configured."""
        assert len(encoder.net.layers) == 2

    def test_embed_dim_stored(self, encoder, embed_dim):
        assert encoder.embed_dim == embed_dim

    def test_default_no_pos_token(self, encoder):
        assert encoder.pos_token is None
        assert not encoder.use_pos_token

    def test_parameter_count_positive(self, encoder):
        total = sum(p.numel() for p in encoder.parameters())
        assert total > 0

    def test_custom_init_embedding(self, embed_dim):
        """Can pass a custom init_embedding module."""
        custom_emb = HybridInitEmbedding(embed_dim=embed_dim, sample_size=5)
        enc = HybridEncoder(
            embed_dim=embed_dim,
            num_layers=1,
            feedforward_hidden=64,
            init_embedding=custom_emb,
        )
        assert enc.init_embedding is custom_emb


# ─── Test Class: Forward Shapes ────────────────────────────────────────────

class TestHybridEncoderForwardShapes:
    """Test output shapes of forward pass."""

    def test_returns_tuple_of_two(self, encoder, td):
        out = encoder(td, phase="train")
        assert isinstance(out, tuple)
        assert len(out) == 2

    def test_h_shape(self, encoder, td, batch_size, num_agents, num_loc, embed_dim):
        h, _ = encoder(td, phase="train")
        assert h.shape == (batch_size, num_agents + num_loc, embed_dim)

    def test_init_h_shape(self, encoder, td, batch_size, num_agents, num_loc, embed_dim):
        _, init_h = encoder(td, phase="train")
        assert init_h.shape == (batch_size, num_agents + num_loc, embed_dim)

    def test_h_and_init_h_differ(self, encoder, td):
        """Encoder should transform embeddings, so h ≠ init_h."""
        h, init_h = encoder(td, phase="train")
        assert not torch.allclose(h, init_h, atol=1e-6)

    def test_output_finite(self, encoder, td):
        h, init_h = encoder(td, phase="train")
        assert torch.isfinite(h).all()
        assert torch.isfinite(init_h).all()

    def test_output_dtype_float(self, encoder, td):
        h, init_h = encoder(td, phase="train")
        assert h.dtype == torch.float32
        assert init_h.dtype == torch.float32


# ─── Test Class: Fusion Types ──────────────────────────────────────────────

class TestFusionTypes:
    """Test different fusion strategies."""

    @pytest.fixture(params=["concat", "gate", "add"])
    def fusion_encoder(self, request, embed_dim):
        return HybridEncoder(
            embed_dim=embed_dim,
            num_layers=1,
            feedforward_hidden=64,
            fusion_type=request.param,
            init_embedding_kwargs=dict(sample_size=10),
        )

    def test_fusion_type_forward(self, fusion_encoder, td, batch_size, num_agents, num_loc, embed_dim):
        h, init_h = fusion_encoder(td, phase="train")
        total = num_agents + num_loc
        assert h.shape == (batch_size, total, embed_dim)
        assert init_h.shape == (batch_size, total, embed_dim)

    def test_invalid_fusion_type(self, embed_dim):
        with pytest.raises(ValueError, match="Unknown fusion_type"):
            HybridEncoder(
                embed_dim=embed_dim,
                num_layers=1,
                feedforward_hidden=64,
                fusion_type="invalid",
                init_embedding_kwargs=dict(sample_size=10),
            )

    def test_concat_has_linear(self, embed_dim):
        enc = HybridEncoder(
            embed_dim=embed_dim,
            num_layers=1,
            feedforward_hidden=64,
            fusion_type="concat",
            init_embedding_kwargs=dict(sample_size=10),
        )
        assert isinstance(enc.fusion, torch.nn.Linear)
        assert enc.fusion.in_features == 2 * embed_dim
        assert enc.fusion.out_features == embed_dim

    def test_gate_has_gate_proj(self, embed_dim):
        enc = HybridEncoder(
            embed_dim=embed_dim,
            num_layers=1,
            feedforward_hidden=64,
            fusion_type="gate",
            init_embedding_kwargs=dict(sample_size=10),
        )
        assert hasattr(enc, "gate_proj")
        assert enc.gate_proj.in_features == 2 * embed_dim

    def test_add_no_extra_params(self, embed_dim):
        enc = HybridEncoder(
            embed_dim=embed_dim,
            num_layers=1,
            feedforward_hidden=64,
            fusion_type="add",
            init_embedding_kwargs=dict(sample_size=10),
        )
        assert enc.fusion is None


# ─── Test Class: Different Configurations ──────────────────────────────────

class TestDifferentConfigs:
    """Test with varying configurations."""

    @pytest.mark.parametrize("n_layers", [1, 3, 6])
    def test_different_num_layers(self, embed_dim, td, n_layers):
        enc = HybridEncoder(
            embed_dim=embed_dim,
            num_layers=n_layers,
            feedforward_hidden=64,
            init_embedding_kwargs=dict(sample_size=10),
        )
        h, init_h = enc(td, phase="train")
        assert h.shape[-1] == embed_dim
        assert len(enc.net.layers) == n_layers

    @pytest.mark.parametrize("nab", ["gating", "heuristic"])
    def test_different_nab_types(self, embed_dim, td, nab):
        enc = HybridEncoder(
            embed_dim=embed_dim,
            num_layers=1,
            feedforward_hidden=64,
            nab_type=nab,
            init_embedding_kwargs=dict(sample_size=10),
        )
        h, init_h = enc(td, phase="train")
        assert h.shape[-1] == embed_dim

    @pytest.mark.parametrize("edim", [32, 64, 128])
    def test_different_embed_dims(self, td, edim):
        # Rebuild td doesn't depend on embed_dim, so reuse
        enc = HybridEncoder(
            embed_dim=edim,
            num_layers=1,
            feedforward_hidden=64,
            init_embedding_kwargs=dict(sample_size=10),
        )
        h, _ = enc(td, phase="train")
        assert h.shape[-1] == edim

    def test_different_problem_size(self, embed_dim):
        td = make_mock_td(2, num_agents=5, num_loc=50)
        enc = HybridEncoder(
            embed_dim=embed_dim,
            num_layers=1,
            feedforward_hidden=64,
            init_embedding_kwargs=dict(sample_size=10),
        )
        h, init_h = enc(td, phase="train")
        assert h.shape == (2, 55, embed_dim)


# ─── Test Class: Gradient Flow ─────────────────────────────────────────────

class TestGradientFlow:
    """Test that gradients flow through all components."""

    def test_backward_runs(self, encoder, td):
        h, init_h = encoder(td, phase="train")
        loss = h.sum() + init_h.sum()
        loss.backward()
        # At least some parameters should have gradients
        grad_count = sum(
            1 for p in encoder.parameters()
            if p.requires_grad and p.grad is not None
        )
        assert grad_count > 0

    def test_grad_init_embedding(self, encoder, td):
        h, init_h = encoder(td, phase="train")
        loss = h.sum() + init_h.sum()
        loss.backward()
        assert encoder.init_embedding.init_embed_depot.weight.grad is not None

    def test_grad_attn_free_net(self, encoder, td):
        h, init_h = encoder(td, phase="train")
        loss = h.sum() + init_h.sum()
        loss.backward()
        # Check first layer of AttnFreeNet
        layer0 = encoder.net.layers[0]
        row_block = layer0.row_encoding_block
        assert row_block.attn_free.to_q.weight.grad is not None

    def test_grad_fusion_concat(self, encoder, td):
        h, init_h = encoder(td, phase="train")
        loss = h.sum() + init_h.sum()
        loss.backward()
        assert encoder.fusion.weight.grad is not None

    def test_grad_fusion_gate(self, embed_dim, td):
        enc = HybridEncoder(
            embed_dim=embed_dim,
            num_layers=1,
            feedforward_hidden=64,
            fusion_type="gate",
            init_embedding_kwargs=dict(sample_size=10),
        )
        h, init_h = enc(td, phase="train")
        loss = h.sum() + init_h.sum()
        loss.backward()
        assert enc.gate_proj.weight.grad is not None


# ─── Test Class: POS Token ─────────────────────────────────────────────────

class TestPosToken:
    """Test POS token support."""

    def test_pos_token_adds_one_node(self, embed_dim, td, batch_size, num_agents, num_loc):
        enc = HybridEncoder(
            embed_dim=embed_dim,
            num_layers=1,
            feedforward_hidden=64,
            use_pos_token=True,
            init_embedding_kwargs=dict(sample_size=10),
        )
        h, init_h = enc(td, phase="train")
        total = num_agents + num_loc + 1  # +1 for POS
        assert h.shape == (batch_size, total, embed_dim)
        assert init_h.shape == (batch_size, total, embed_dim)

    def test_pos_token_trainable(self, embed_dim):
        enc = HybridEncoder(
            embed_dim=embed_dim,
            num_layers=1,
            feedforward_hidden=64,
            use_pos_token=True,
            trainable_pos_token=True,
            init_embedding_kwargs=dict(sample_size=10),
        )
        assert isinstance(enc.pos_token, torch.nn.Parameter)
        assert enc.pos_token.requires_grad

    def test_pos_token_not_trainable(self, embed_dim):
        enc = HybridEncoder(
            embed_dim=embed_dim,
            num_layers=1,
            feedforward_hidden=64,
            use_pos_token=True,
            trainable_pos_token=False,
            init_embedding_kwargs=dict(sample_size=10),
        )
        assert not isinstance(enc.pos_token, torch.nn.Parameter)


# ─── Test Class: Phase Behavior ────────────────────────────────────────────

class TestPhaseBehavior:
    """Test train vs test phase forwarding."""

    def test_train_phase(self, encoder, td):
        h, _ = encoder(td, phase="train")
        assert h is not None

    def test_test_phase(self, encoder, td):
        h, _ = encoder(td, phase="test")
        assert h is not None

    def test_default_mask_none(self, encoder, td):
        """Mask defaults to None and should work."""
        h, _ = encoder(td)
        assert h is not None


# ─── Test Class: Interface Compatibility ───────────────────────────────────

class TestInterfaceCompatibility:
    """Verify output interface matches PARCOEncoder contract."""

    def test_same_return_type_as_parco(self, encoder, td):
        """PARCOEncoder returns (Tensor, Tensor). HybridEncoder should too."""
        result = encoder(td, phase="train")
        assert isinstance(result, tuple)
        h, init_h = result
        assert isinstance(h, torch.Tensor)
        assert isinstance(init_h, torch.Tensor)

    def test_h_can_be_projected(self, encoder, td, embed_dim):
        """Decoder projects h into K, V, logit_K. Check this works."""
        h, _ = encoder(td, phase="train")
        proj = torch.nn.Linear(embed_dim, 3 * embed_dim)
        out = proj(h)
        assert out.shape[-1] == 3 * embed_dim

    def test_mask_kwarg_accepted(self, encoder, td, batch_size, num_agents, num_loc):
        """PARCOPolicy passes mask=None. Verify it's accepted."""
        h, init_h = encoder(td, mask=None, phase="train")
        assert h.shape[0] == batch_size


# ─── Test Class: Integration with HCVRPRealEnv ────────────────────────────

class TestIntegrationWithEnv:
    """Test HybridEncoder with real env TensorDict."""

    @pytest.fixture
    def real_env(self):
        return HCVRPRealEnv(
            generator_params=dict(num_loc=20, num_agents=3)
        )

    @pytest.fixture
    def real_td(self, real_env):
        return real_env.reset(batch_size=[4])

    @pytest.fixture
    def real_encoder(self):
        return HybridEncoder(
            embed_dim=64,
            num_layers=2,
            feedforward_hidden=128,
            init_embedding_kwargs=dict(sample_size=10),
        )

    def test_forward_with_real_td(self, real_encoder, real_td):
        h, init_h = real_encoder(real_td, phase="train")
        assert h.shape == (4, 23, 64)
        assert init_h.shape == (4, 23, 64)

    def test_output_finite_real(self, real_encoder, real_td):
        h, init_h = real_encoder(real_td, phase="train")
        assert torch.isfinite(h).all()
        assert torch.isfinite(init_h).all()

    def test_backward_with_real_td(self, real_encoder, real_td):
        h, init_h = real_encoder(real_td, phase="train")
        loss = h.mean() + init_h.mean()
        loss.backward()
        grad_count = sum(
            1 for p in real_encoder.parameters()
            if p.requires_grad and p.grad is not None
        )
        assert grad_count > 0
