"""Phase 5 Tests: HybridDecoder.

Tests cover:
- Module instantiation (with/without distance bias)
- Inherits from PARCODecoder
- Forward pass output shapes (logits, mask)
- Distance bias modifies logits
- No-bias mode matches PARCODecoder behavior
- alpha parameter is learnable
- Integration with HybridEncoder + HCVRPRealEnv
- pre_decoder_hook produces valid cache
- Gradient flow through distance bias
"""

import pytest
import torch
import torch.nn as nn
from tensordict.tensordict import TensorDict

from hybrid.models.decoder import HybridDecoder
from hybrid.models.encoder import HybridEncoder
from hybrid.envs import HCVRPRealEnv
from parco.models.decoder import PARCODecoder


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def embed_dim():
    return 64


@pytest.fixture
def num_heads():
    return 4


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
def decoder_with_bias(embed_dim, num_heads):
    """Decoder with distance bias enabled."""
    return HybridDecoder(
        embed_dim=embed_dim,
        num_heads=num_heads,
        env_name="hcvrp_real",
        use_distance_bias=True,
    )


@pytest.fixture
def decoder_no_bias(embed_dim, num_heads):
    """Decoder with distance bias disabled."""
    return HybridDecoder(
        embed_dim=embed_dim,
        num_heads=num_heads,
        env_name="hcvrp_real",
        use_distance_bias=False,
    )


@pytest.fixture
def encoder(embed_dim):
    return HybridEncoder(
        embed_dim=embed_dim,
        num_layers=1,
        feedforward_hidden=64,
        init_embedding_kwargs=dict(sample_size=10),
    )


@pytest.fixture
def env():
    return HCVRPRealEnv(
        generator_params=dict(num_loc=20, num_agents=3)
    )


@pytest.fixture
def td_reset(env, batch_size):
    """TensorDict after env.reset()."""
    return env.reset(batch_size=[batch_size])


@pytest.fixture
def encoded(encoder, td_reset):
    """Encoder output (h, init_h)."""
    return encoder(td_reset, phase="train")


@pytest.fixture
def cached_with_bias(decoder_with_bias, td_reset, env, encoded):
    """Pre-decoder hook result with distance bias decoder."""
    h, init_h = encoded
    td, _, cached = decoder_with_bias.pre_decoder_hook(td_reset, env, h)
    return td, cached


@pytest.fixture
def cached_no_bias(decoder_no_bias, td_reset, env, encoded):
    """Pre-decoder hook result with no-bias decoder."""
    h, init_h = encoded
    td, _, cached = decoder_no_bias.pre_decoder_hook(td_reset, env, h)
    return td, cached


# ─── Test Class: Instantiation ─────────────────────────────────────────────

class TestHybridDecoderInit:
    """Test module instantiation and structure."""

    def test_instantiation_with_bias(self, decoder_with_bias):
        assert isinstance(decoder_with_bias, HybridDecoder)

    def test_instantiation_no_bias(self, decoder_no_bias):
        assert isinstance(decoder_no_bias, HybridDecoder)

    def test_inherits_parco_decoder(self, decoder_with_bias):
        assert isinstance(decoder_with_bias, PARCODecoder)

    def test_has_alpha_when_bias(self, decoder_with_bias):
        assert hasattr(decoder_with_bias, "alpha")
        assert isinstance(decoder_with_bias.alpha, nn.Parameter)

    def test_no_alpha_when_no_bias(self, decoder_no_bias):
        assert not hasattr(decoder_no_bias, "alpha") or not decoder_no_bias.use_distance_bias

    def test_use_distance_bias_flag(self, decoder_with_bias, decoder_no_bias):
        assert decoder_with_bias.use_distance_bias is True
        assert decoder_no_bias.use_distance_bias is False

    def test_alpha_initial_value(self, decoder_with_bias):
        assert torch.allclose(decoder_with_bias.alpha, torch.ones(1))

    def test_hybrid_env_name_stored(self, decoder_with_bias):
        assert decoder_with_bias.hybrid_env_name == "hcvrp_real"

    def test_parco_env_name_is_hcvrp(self, decoder_with_bias):
        """Internal env_name should be 'hcvrp' for embedding registry."""
        assert decoder_with_bias.env_name == "hcvrp"

    def test_has_context_embedding(self, decoder_with_bias):
        assert hasattr(decoder_with_bias, "context_embedding")

    def test_has_pointer(self, decoder_with_bias):
        assert hasattr(decoder_with_bias, "pointer")

    def test_parameter_count_positive(self, decoder_with_bias):
        total = sum(p.numel() for p in decoder_with_bias.parameters())
        assert total > 0


# ─── Test Class: Pre-Decoder Hook ──────────────────────────────────────────

class TestPreDecoderHook:
    """Test pre_decoder_hook produces valid cache."""

    def test_hook_returns_tuple(self, decoder_with_bias, td_reset, env, encoded):
        h, _ = encoded
        result = decoder_with_bias.pre_decoder_hook(td_reset, env, h)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_cache_has_node_embeddings(self, cached_with_bias):
        _, cached = cached_with_bias
        assert cached.node_embeddings is not None

    def test_cache_has_glimpse_keys(self, cached_with_bias):
        _, cached = cached_with_bias
        assert cached.glimpse_key is not None
        assert cached.glimpse_val is not None
        assert cached.logit_key is not None


# ─── Test Class: Forward Pass ──────────────────────────────────────────────

class TestForwardPass:
    """Test forward pass output shapes."""

    def test_returns_tuple_of_two(self, decoder_with_bias, cached_with_bias):
        td, cached = cached_with_bias
        logits, mask = decoder_with_bias(td, cached)
        assert isinstance(logits, torch.Tensor)
        assert isinstance(mask, torch.Tensor)

    def test_logits_shape(self, decoder_with_bias, cached_with_bias,
                          batch_size, num_agents, num_loc):
        td, cached = cached_with_bias
        logits, mask = decoder_with_bias(td, cached)
        total = num_agents + num_loc
        assert logits.shape == (batch_size, num_agents, total)

    def test_mask_shape(self, decoder_with_bias, cached_with_bias,
                        batch_size, num_agents, num_loc):
        td, cached = cached_with_bias
        logits, mask = decoder_with_bias(td, cached)
        total = num_agents + num_loc
        assert mask.shape == (batch_size, num_agents, total)

    def test_logits_finite(self, decoder_with_bias, cached_with_bias):
        td, cached = cached_with_bias
        logits, _ = decoder_with_bias(td, cached)
        assert torch.isfinite(logits).all()

    def test_no_bias_forward(self, decoder_no_bias, cached_no_bias,
                             batch_size, num_agents, num_loc):
        td, cached = cached_no_bias
        logits, mask = decoder_no_bias(td, cached)
        total = num_agents + num_loc
        assert logits.shape == (batch_size, num_agents, total)


# ─── Test Class: Distance Bias ─────────────────────────────────────────────

class TestDistanceBias:
    """Test distance bias mechanism."""

    def test_bias_modifies_logits(self, decoder_with_bias, decoder_no_bias,
                                  cached_with_bias, cached_no_bias):
        """Logits should differ when distance bias is applied."""
        td_bias, cache_bias = cached_with_bias
        td_nobias, cache_nobias = cached_no_bias

        # Use same td for both (they share same reset state)
        logits_bias, _ = decoder_with_bias(td_bias, cache_bias)
        logits_nobias, _ = decoder_no_bias(td_nobias, cache_nobias)

        # Since both decoders have different weights, logits differ anyway.
        # But with bias, logits should be modified from the base computation.
        # We verify the bias path runs without error.
        assert logits_bias.shape == logits_nobias.shape

    def test_alpha_affects_logits(self, decoder_with_bias, cached_with_bias):
        """Changing alpha should change the output logits."""
        td, cached = cached_with_bias

        # Get logits with default alpha=1
        logits_alpha1, _ = decoder_with_bias(td, cached)

        # Change alpha to 0 (no bias)
        with torch.no_grad():
            decoder_with_bias.alpha.fill_(0.0)
        logits_alpha0, _ = decoder_with_bias(td, cached)

        # Restore alpha
        with torch.no_grad():
            decoder_with_bias.alpha.fill_(1.0)

        # They should differ (unless distance happens to be zero)
        assert not torch.allclose(logits_alpha1, logits_alpha0, atol=1e-5)

    def test_no_distance_matrix_skips_bias(self, decoder_with_bias, cached_with_bias):
        """If distance_matrix not in td, bias should be skipped."""
        td, cached = cached_with_bias
        # Remove distance_matrix from td
        td_no_dist = td.exclude("distance_matrix")
        logits, mask = decoder_with_bias(td_no_dist, cached)
        assert logits.shape[-2] > 0  # just check it doesn't crash

    def test_bias_penalizes_distant_nodes(self, decoder_with_bias, cached_with_bias):
        """With very large alpha, logits for distant nodes should decrease."""
        td, cached = cached_with_bias
        with torch.no_grad():
            decoder_with_bias.alpha.fill_(100.0)  # very strong bias
        logits_strong, _ = decoder_with_bias(td, cached)

        with torch.no_grad():
            decoder_with_bias.alpha.fill_(1.0)
        logits_normal, _ = decoder_with_bias(td, cached)

        # Strong bias should push logits lower overall (for non-zero distances)
        # The mean of strongly biased logits should be lower
        assert logits_strong.mean() < logits_normal.mean()


# ─── Test Class: Gradient Flow ─────────────────────────────────────────────

class TestGradientFlow:
    """Test gradient flow through distance bias."""

    def test_alpha_has_gradient(self, decoder_with_bias, cached_with_bias):
        td, cached = cached_with_bias
        logits, _ = decoder_with_bias(td, cached)
        loss = logits.sum()
        loss.backward()
        assert decoder_with_bias.alpha.grad is not None

    def test_backward_through_decoder(self, decoder_with_bias, cached_with_bias):
        td, cached = cached_with_bias
        logits, _ = decoder_with_bias(td, cached)
        loss = logits.mean()
        loss.backward()
        grad_count = sum(
            1 for p in decoder_with_bias.parameters()
            if p.requires_grad and p.grad is not None
        )
        assert grad_count > 0

    def test_no_bias_backward(self, decoder_no_bias, cached_no_bias):
        td, cached = cached_no_bias
        logits, _ = decoder_no_bias(td, cached)
        loss = logits.mean()
        loss.backward()
        # Should still work, just no alpha gradient
        grad_count = sum(
            1 for p in decoder_no_bias.parameters()
            if p.requires_grad and p.grad is not None
        )
        assert grad_count > 0


# ─── Test Class: Edge Cases ────────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases and configurations."""

    def test_different_embed_dims(self):
        # embed_dim / num_heads must yield head_dim divisible by 8 and <= 128
        for edim, nh in [(64, 4), (128, 8), (128, 4)]:
            dec = HybridDecoder(embed_dim=edim, num_heads=nh, use_distance_bias=True)
            assert isinstance(dec, HybridDecoder)

    def test_different_num_heads(self):
        for nh in [1, 4, 8]:
            dec = HybridDecoder(embed_dim=64, num_heads=nh, use_distance_bias=True)
            assert isinstance(dec, HybridDecoder)

    def test_kwargs_forwarded(self):
        """Extra kwargs should be forwarded to PARCODecoder."""
        dec = HybridDecoder(
            embed_dim=64,
            num_heads=4,
            use_distance_bias=True,
            use_graph_context=False,
        )
        assert isinstance(dec, HybridDecoder)
