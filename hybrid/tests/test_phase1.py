"""Phase 1 validation tests.

Verifies that the hybrid project is correctly set up:
1. All packages are importable (hybrid, parco, rrnco)
2. Critical classes from both parent projects are accessible
3. Config files exist and are valid
4. The project structure is complete
"""

import os
import sys

import pytest


# Ensure both parent packages are on the path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARCO_ROOT = os.path.join(os.path.dirname(WORKSPACE_ROOT), "parco")
RRNCO_ROOT = os.path.join(os.path.dirname(WORKSPACE_ROOT), "real-routing-nco")

for path in [PARCO_ROOT, RRNCO_ROOT, WORKSPACE_ROOT]:
    if path not in sys.path:
        sys.path.insert(0, path)


class TestImports:
    """Test that all required packages are importable."""

    def test_import_hybrid(self):
        import hybrid

        assert hybrid is not None

    def test_import_hybrid_models(self):
        from hybrid.models import (
            HybridDecoder,
            HybridEncoder,
            HybridPolicy,
            HybridRLModule,
        )

        assert HybridEncoder is not None
        assert HybridDecoder is not None
        assert HybridPolicy is not None
        assert HybridRLModule is not None

    def test_import_hybrid_envs(self):
        from hybrid.envs import HCVRPRealEnv

        assert HCVRPRealEnv is not None

    def test_import_hybrid_embeddings(self):
        from hybrid.models.env_embeddings import HybridInitEmbedding

        assert HybridInitEmbedding is not None

    def test_import_parco_models(self):
        from parco.models import (
            PARCODecoder,
            PARCOEncoder,
            PARCOPolicy,
            PARCORLModule,
        )

        assert PARCOEncoder is not None
        assert PARCODecoder is not None
        assert PARCOPolicy is not None
        assert PARCORLModule is not None

    def test_import_parco_envs(self):
        from parco.envs import HCVRPEnv

        assert HCVRPEnv is not None

    def test_import_parco_communication(self):
        from parco.models.env_embeddings.communication import (
            BaseMultiAgentContextEmbedding,
        )

        assert BaseMultiAgentContextEmbedding is not None

    def test_import_parco_agent_handlers(self):
        from parco.models.agent_handlers import AgentHandler

        assert AgentHandler is not None

    def test_import_rrnco_models(self):
        from rrnco.models import RRNetDecoder, RRNetEncoder, RRNetPolicy

        assert RRNetEncoder is not None
        assert RRNetDecoder is not None
        assert RRNetPolicy is not None

    def test_import_rrnco_attn_freenet(self):
        from rrnco.models.nn.attn_freenet import AttnFreeNet

        assert AttnFreeNet is not None

    def test_import_rrnco_init_embedding(self):
        from rrnco.models.env_embeddings.rcvrp import RVRPInitEmbedding

        assert RVRPInitEmbedding is not None

    def test_import_rl4co(self):
        import rl4co

        assert rl4co is not None

    def test_import_torch(self):
        import torch

        assert torch is not None


class TestProjectStructure:
    """Test that all required directories and files exist."""

    def test_hybrid_package_exists(self):
        assert os.path.isdir(
            os.path.join(WORKSPACE_ROOT, "hybrid")
        )

    def test_models_dir(self):
        assert os.path.isdir(
            os.path.join(WORKSPACE_ROOT, "hybrid", "models")
        )

    def test_envs_dir(self):
        assert os.path.isdir(
            os.path.join(WORKSPACE_ROOT, "hybrid", "envs")
        )

    def test_utils_dir(self):
        assert os.path.isdir(
            os.path.join(WORKSPACE_ROOT, "hybrid", "utils")
        )

    def test_env_embeddings_dir(self):
        assert os.path.isdir(
            os.path.join(WORKSPACE_ROOT, "hybrid", "models", "env_embeddings")
        )

    def test_configs_dir(self):
        configs_root = os.path.join(WORKSPACE_ROOT, "configs")
        assert os.path.isdir(configs_root)

    def test_config_subdirs(self):
        configs_root = os.path.join(WORKSPACE_ROOT, "configs")
        for subdir in [
            "callbacks",
            "debug",
            "env",
            "experiment",
            "extras",
            "hydra",
            "logger",
            "model",
            "paths",
            "trainer",
        ]:
            assert os.path.isdir(
                os.path.join(configs_root, subdir)
            ), f"Missing config subdir: {subdir}"

    def test_main_yaml_exists(self):
        assert os.path.isfile(
            os.path.join(WORKSPACE_ROOT, "configs", "main.yaml")
        )

    def test_model_config_exists(self):
        assert os.path.isfile(
            os.path.join(WORKSPACE_ROOT, "configs", "model", "hybrid.yaml")
        )

    def test_env_config_exists(self):
        assert os.path.isfile(
            os.path.join(
                WORKSPACE_ROOT, "configs", "env", "hcvrp_real.yaml"
            )
        )

    def test_experiment_config_exists(self):
        assert os.path.isfile(
            os.path.join(
                WORKSPACE_ROOT, "configs", "experiment", "hybrid_hcvrp.yaml"
            )
        )

    def test_train_py_exists(self):
        assert os.path.isfile(os.path.join(WORKSPACE_ROOT, "train.py"))

    def test_test_py_exists(self):
        assert os.path.isfile(os.path.join(WORKSPACE_ROOT, "test.py"))

    def test_pyproject_toml_exists(self):
        assert os.path.isfile(
            os.path.join(WORKSPACE_ROOT, "pyproject.toml")
        )

    def test_gitignore_exists(self):
        assert os.path.isfile(
            os.path.join(WORKSPACE_ROOT, ".gitignore")
        )


class TestModelStubs:
    """Test that stub model classes have correct interfaces."""

    def test_encoder_instantiates(self):
        from hybrid.models.encoder import HybridEncoder

        enc = HybridEncoder(
            embed_dim=32,
            num_layers=1,
            feedforward_hidden=64,
            init_embedding_kwargs=dict(sample_size=5),
        )
        assert isinstance(enc, HybridEncoder)

    def test_decoder_instantiates(self):
        from hybrid.models.decoder import HybridDecoder

        dec = HybridDecoder(
            embed_dim=64, num_heads=4, use_distance_bias=True
        )
        assert isinstance(dec, HybridDecoder)

    def test_policy_instantiates(self):
        from hybrid.models.policy import HybridPolicy

        policy = HybridPolicy(
            embed_dim=64,
            num_heads=8,
            num_encoder_layers=1,
            feedforward_hidden=128,
            init_embedding_kwargs={"sample_size": 5},
        )
        assert isinstance(policy, HybridPolicy)

    def test_rl_module_instantiates(self):
        from hybrid.envs import HCVRPRealEnv, HCVRPRealGenerator
        from hybrid.models.policy import HybridPolicy
        from hybrid.models.rl import HybridRLModule

        gen = HCVRPRealGenerator(
            num_loc=20, num_agents=5, compute_distance_matrix=True
        )
        env = HCVRPRealEnv(generator=gen)
        policy = HybridPolicy(
            embed_dim=64,
            num_heads=8,
            num_encoder_layers=1,
            feedforward_hidden=128,
            init_embedding_kwargs={"sample_size": 5},
        )
        module = HybridRLModule(
            env=env, policy=policy, num_augment=1,
        )
        assert isinstance(module, HybridRLModule)

    def test_init_embedding_instantiates(self):
        from hybrid.models.env_embeddings.hcvrp_real import (
            HybridInitEmbedding,
        )

        model = HybridInitEmbedding(embed_dim=32, sample_size=5)
        assert isinstance(model, HybridInitEmbedding)

    def test_env_instantiates(self):
        from hybrid.envs.hcvrp_real import HCVRPRealEnv

        env = HCVRPRealEnv(generator_params=dict(num_loc=5, num_agents=2))
        assert env.name == "hcvrp_real"


class TestCrossPackageCompatibility:
    """Test that critical components from PARCO and RRNCO can coexist."""

    def test_both_encoders_importable(self):
        from parco.models import PARCOEncoder
        from rrnco.models import RRNetEncoder

        assert PARCOEncoder is not None
        assert RRNetEncoder is not None

    def test_both_decoders_importable(self):
        from parco.models import PARCODecoder
        from rrnco.models import RRNetDecoder

        assert PARCODecoder is not None
        assert RRNetDecoder is not None

    def test_parco_env_importable(self):
        from parco.envs import HCVRPEnv

        assert HCVRPEnv is not None

    def test_rrnco_attnfreenet_importable(self):
        from rrnco.models.nn.attn_freenet import AttnFreeNet

        assert AttnFreeNet is not None

    def test_rl4co_base_classes_importable(self):
        from rl4co.models.nn.attention import PointerAttention
        from rl4co.utils import RL4COTrainer

        assert PointerAttention is not None
        assert RL4COTrainer is not None
