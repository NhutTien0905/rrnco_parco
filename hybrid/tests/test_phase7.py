"""Phase 7 Tests: HybridRLModule.

Tests cover:
1. resample_batch_hybrid (distance_matrix slicing + renormalization)
2. Instantiation (default, custom policy, custom kwargs)
3. Inheritance from PARCORLModule / SymNCO / REINFORCE / RL4COLitModule
4. shared_step (training + validation)
5. Training step gradient flow
6. Hyperparameter defaults
7. Edge cases
"""

import pytest
import torch
import torch.nn as nn

from tensordict import TensorDict


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_env():
    """Create a small HCVRPRealEnv for testing."""
    from hybrid.envs import HCVRPRealEnv, HCVRPRealGenerator

    gen = HCVRPRealGenerator(
        num_loc=20,
        num_agents=5,
        compute_distance_matrix=True,
    )
    return HCVRPRealEnv(generator=gen)


@pytest.fixture
def small_policy():
    """Create a small HybridPolicy for testing."""
    from hybrid.models.policy import HybridPolicy

    return HybridPolicy(
        embed_dim=64,
        num_heads=8,
        num_encoder_layers=1,
        feedforward_hidden=128,
        init_embedding_kwargs={"sample_size": 5},
    )


@pytest.fixture
def small_rl_module(small_env, small_policy):
    """Create a small HybridRLModule.

    Uses num_augment=2 so SymNCO losses produce real tensors
    (with num_augment=1, losses are Python 0 → loss is a float).
    """
    from hybrid.models.rl import HybridRLModule

    return HybridRLModule(
        env=small_env,
        policy=small_policy,
        num_augment=2,
        train_min_agents=3,
        train_max_agents=5,
        train_min_size=10,
        train_max_size=20,
        optimizer_kwargs={"lr": 1e-4},
    )


@pytest.fixture
def batch(small_env):
    """Generate a batch from the environment's generator."""
    return small_env.generator(batch_size=[4])


# ---------------------------------------------------------------------------
# 1. resample_batch_hybrid Tests
# ---------------------------------------------------------------------------

class TestResampleBatchHybrid:
    """Test the resample_batch_hybrid utility function."""

    def test_slices_distance_matrix(self, batch):
        from hybrid.models.rl import resample_batch_hybrid

        num_agents = 3
        num_locs = 10
        resampled = resample_batch_hybrid(batch.clone(), num_agents, num_locs)

        # Raw generator has distance_matrix [B, 1+N_old, 1+N_old].
        # After resample: [B, 1+num_locs, 1+num_locs] (1 depot + customers).
        keep = 1 + num_locs
        assert resampled["distance_matrix"].shape == (4, keep, keep)

    def test_slices_locs(self, batch):
        from hybrid.models.rl import resample_batch_hybrid

        num_agents = 3
        num_locs = 10
        resampled = resample_batch_hybrid(batch.clone(), num_agents, num_locs)

        # PARCO's resample slices locs (customers only) to num_locs
        assert resampled["locs"].shape == (4, num_locs, 2)

    def test_renormalises_distance_matrix(self, batch):
        from hybrid.models.rl import resample_batch_hybrid

        num_agents = 3
        num_locs = 10
        resampled = resample_batch_hybrid(batch.clone(), num_agents, num_locs)

        dist = resampled["distance_matrix"]
        # After renormalization, min should be ≈0, max should be ≈1
        for b in range(dist.shape[0]):
            flat = dist[b].flatten()
            assert flat.min() >= -1e-5, f"Min distance {flat.min()} < 0"
            assert flat.max() <= 1.0 + 1e-5, f"Max distance {flat.max()} > 1"

    def test_updates_min_max_distance(self, batch):
        from hybrid.models.rl import resample_batch_hybrid

        num_agents = 3
        num_locs = 10
        resampled = resample_batch_hybrid(batch.clone(), num_agents, num_locs)

        assert "min_distance" in resampled.keys()
        assert "max_distance" in resampled.keys()
        assert resampled["min_distance"].shape == (4,)
        assert resampled["max_distance"].shape == (4,)

    def test_slices_demand(self, batch):
        from hybrid.models.rl import resample_batch_hybrid

        num_agents = 3
        num_locs = 10
        resampled = resample_batch_hybrid(batch.clone(), num_agents, num_locs)

        if "demand" in resampled.keys():
            assert resampled["demand"].shape[-1] == num_locs

    def test_slices_capacity(self, batch):
        from hybrid.models.rl import resample_batch_hybrid

        num_agents = 3
        num_locs = 10
        resampled = resample_batch_hybrid(batch.clone(), num_agents, num_locs)

        if "capacity" in resampled.keys():
            assert resampled["capacity"].shape[-1] == num_agents

    def test_no_distance_matrix_graceful(self):
        """resample_batch_hybrid should work even without distance_matrix."""
        from hybrid.models.rl import resample_batch_hybrid

        td = TensorDict(
            {
                "locs": torch.rand(2, 20, 2),
                "demand": torch.rand(2, 20),
                "num_agents": torch.full((2,), 5),
            },
            batch_size=[2],
        )
        # Should not raise
        resampled = resample_batch_hybrid(td, num_agents=3, num_locs=10)
        assert resampled["locs"].shape == (2, 10, 2)
        assert "distance_matrix" not in resampled.keys()


# ---------------------------------------------------------------------------
# 2. Instantiation Tests
# ---------------------------------------------------------------------------

class TestInstantiation:
    """Test HybridRLModule creation."""

    def test_instantiation(self, small_env, small_policy):
        from hybrid.models.rl import HybridRLModule

        module = HybridRLModule(
            env=small_env,
            policy=small_policy,
            num_augment=1,
        )
        assert isinstance(module, nn.Module)

    def test_is_parco_rl_module(self, small_env, small_policy):
        from hybrid.models.rl import HybridRLModule
        from parco.models.rl import PARCORLModule

        module = HybridRLModule(
            env=small_env,
            policy=small_policy,
            num_augment=1,
        )
        assert isinstance(module, PARCORLModule)

    def test_is_lightning_module(self, small_env, small_policy):
        import lightning as L
        from hybrid.models.rl import HybridRLModule

        module = HybridRLModule(
            env=small_env,
            policy=small_policy,
            num_augment=1,
        )
        assert isinstance(module, L.LightningModule)

    def test_env_stored(self, small_rl_module, small_env):
        assert small_rl_module.env is small_env

    def test_policy_stored(self, small_rl_module, small_policy):
        assert small_rl_module.policy is small_policy


# ---------------------------------------------------------------------------
# 3. Inherited Attributes
# ---------------------------------------------------------------------------

class TestInheritedAttributes:
    """Test attributes inherited from PARCORLModule / SymNCO."""

    def test_num_augment(self, small_rl_module):
        assert small_rl_module.num_augment == 2

    def test_alpha(self, small_rl_module):
        assert small_rl_module.alpha == 0.2

    def test_beta(self, small_rl_module):
        assert small_rl_module.beta == 1.0

    def test_train_min_max_agents(self, small_rl_module):
        assert small_rl_module.train_min_agents == 3
        assert small_rl_module.train_max_agents == 5

    def test_train_min_max_size(self, small_rl_module):
        assert small_rl_module.train_min_size == 10
        assert small_rl_module.train_max_size == 20

    def test_projection_head_exists(self, small_rl_module):
        assert hasattr(small_rl_module, "projection_head")
        assert small_rl_module.projection_head is not None

    def test_augment_exists(self, small_rl_module):
        assert hasattr(small_rl_module, "augment")

    def test_configure_optimizers(self, small_rl_module):
        """Optimizer should be configurable."""
        opt = small_rl_module.configure_optimizers()
        # Returns either an optimizer or a dict
        assert opt is not None


# ---------------------------------------------------------------------------
# 4. shared_step Tests
# ---------------------------------------------------------------------------

class TestSharedStep:
    """Test the shared_step method (core training/val logic)."""

    def test_train_step_returns_loss(self, small_rl_module, batch):
        result = small_rl_module.shared_step(
            batch.clone(), batch_idx=0, phase="train"
        )
        assert "loss" in result
        assert result["loss"] is not None

    def test_val_step_no_loss(self, small_rl_module, batch):
        """Validation step should not compute loss."""
        result = small_rl_module.shared_step(
            batch.clone(), batch_idx=0, phase="val"
        )
        # Loss is None for val/test
        assert result["loss"] is None

    def test_train_step_loss_is_scalar(self, small_rl_module, batch):
        result = small_rl_module.shared_step(
            batch.clone(), batch_idx=0, phase="train"
        )
        loss = result["loss"]
        assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"

    def test_train_step_loss_is_finite(self, small_rl_module, batch):
        result = small_rl_module.shared_step(
            batch.clone(), batch_idx=0, phase="train"
        )
        loss = result["loss"]
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"


# ---------------------------------------------------------------------------
# 5. Gradient Flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    """Test that gradients flow through the training step."""

    def test_backward_pass(self, small_rl_module, batch):
        result = small_rl_module.shared_step(
            batch.clone(), batch_idx=0, phase="train"
        )
        loss = result["loss"]
        loss.backward()

        # Check some parameter has gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in small_rl_module.policy.parameters()
            if p.requires_grad
        )
        assert has_grad, "No gradients in policy parameters"


# ---------------------------------------------------------------------------
# 6. Custom Hyperparameters
# ---------------------------------------------------------------------------

class TestCustomHyperparameters:
    """Test passing custom hyperparameters."""

    def test_custom_num_augment(self, small_env, small_policy):
        from hybrid.models.rl import HybridRLModule

        module = HybridRLModule(
            env=small_env,
            policy=small_policy,
            num_augment=4,
        )
        assert module.num_augment == 4

    def test_custom_alpha_beta(self, small_env, small_policy):
        from hybrid.models.rl import HybridRLModule

        module = HybridRLModule(
            env=small_env,
            policy=small_policy,
            num_augment=1,
            alpha=0.5,
            beta=2.0,
        )
        assert module.alpha == 0.5
        assert module.beta == 2.0

    def test_custom_lr(self, small_env, small_policy):
        from hybrid.models.rl import HybridRLModule

        module = HybridRLModule(
            env=small_env,
            policy=small_policy,
            num_augment=1,
            optimizer_kwargs={"lr": 2e-4},
        )
        opt = module.configure_optimizers()
        # Check lr is set
        if isinstance(opt, dict):
            lr = opt["optimizer"].param_groups[0]["lr"]
        else:
            lr = opt.param_groups[0]["lr"]
        assert lr == 2e-4


# ---------------------------------------------------------------------------
# 7. Parameter Count
# ---------------------------------------------------------------------------

class TestParameterCount:
    """Sanity check that module has parameters."""

    def test_total_params(self, small_rl_module):
        total = sum(p.numel() for p in small_rl_module.parameters())
        assert total > 0

    def test_policy_params_in_module(self, small_rl_module):
        """Policy parameters should be part of the module."""
        policy_params = set(id(p) for p in small_rl_module.policy.parameters())
        module_params = set(id(p) for p in small_rl_module.parameters())
        assert policy_params.issubset(module_params)
