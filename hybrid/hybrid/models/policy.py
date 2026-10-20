"""Hybrid Policy: Wires HybridEncoder + HybridDecoder + communication layers.

Extends ``PARCOPolicy`` to use the hybrid encoder/decoder components while
inheriting the full multi-agent autoregressive decode loop, conflict
handling, and decoding strategies.

The only override to ``PARCOPolicy.forward()`` is the encoder call:
    ``self.encoder(td)``  →  ``self.encoder(td, phase=phase)``
so that the RRNCO DistanceExpert can switch between train/test sampling.
"""

import torch
import torch.nn as nn

from typing import Optional

from rl4co.utils.pylogger import get_pylogger
from tensordict import TensorDict

from parco.models.policy import PARCOPolicy

from hybrid.models.encoder import HybridEncoder
from hybrid.models.decoder import HybridDecoder

log = get_pylogger(__name__)


class HybridPolicy(PARCOPolicy):
    """Hybrid policy combining RRNCO's Attention-Free encoder with
    PARCO's multi-agent decoder.

    Pipeline (inherited from PARCOPolicy with encoder override):
        1. ``encoder(td, phase=phase)`` → ``(h, init_h)``
        2. ``decoder.pre_decoder_hook(td, env, h)`` → cache
        3. Autoregressive loop:
           ``decoder(td, cache)`` → logits → strategy → handler → action
        4. ``env.step(td)``

    The key differences from vanilla PARCOPolicy:
    - Encoder: :class:`HybridEncoder` (RRNCO's AttnFreeNet + fusion)
    - Decoder: :class:`HybridDecoder` (PARCODecoder + optional distance bias)
    - Forward: passes ``phase`` to encoder for DistanceExpert sampling

    Args:
        encoder: Custom encoder module. If ``None``, builds a
            :class:`HybridEncoder` from the keyword arguments.
        decoder: Custom decoder module. If ``None``, builds a
            :class:`HybridDecoder` from the keyword arguments.
        embed_dim: Embedding dimension.
        num_encoder_layers: Number of AttnFreeNet layers (RRNCO default: 6).
        num_heads: Number of attention heads (for decoder pointer attention).
        env_name: Environment name for the hybrid environment.
        nab_type: Neural Adaptive Bias type for the encoder.
        feedforward_hidden: Hidden dim in AttnFreeNet feedforward layers.
        normalization_encoder: Normalization type for the encoder.
        init_embedding_kwargs: Extra kwargs forwarded to
            :class:`HybridInitEmbedding`.
        agent_handler: Agent conflict handler name or instance.
        use_distance_bias: Whether the decoder uses distance-based
            inductive bias.
        use_pos_token: Whether to use POS (pause-of-sequence) token.
        **kwargs: Forwarded to :class:`PARCOPolicy`.
    """

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        embed_dim: int = 128,
        num_encoder_layers: int = 6,
        num_heads: int = 8,
        env_name: str = "hcvrp_real",
        # ── RRNCO encoder kwargs ─────────────────────────────────────
        nab_type: str = "gating",
        feedforward_hidden: int = 512,
        normalization_encoder: str = "instance",
        init_embedding_kwargs: Optional[dict] = None,
        fusion_type: str = "concat",
        use_checkpoint: bool = False,
        # ── PARCO decoder kwargs ─────────────────────────────────────
        agent_handler: str = "highprob",
        use_distance_bias: bool = True,
        use_pos_token: bool = False,
        trainable_pos_token: bool = True,
        # ── Rest ─────────────────────────────────────────────────────
        **kwargs,
    ):
        # Build hybrid encoder if not provided
        if encoder is None:
            log.info("Initializing HybridEncoder")
            encoder = HybridEncoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_encoder_layers,
                normalization=normalization_encoder,
                feedforward_hidden=feedforward_hidden,
                nab_type=nab_type,
                env_name=env_name,
                init_embedding_kwargs=init_embedding_kwargs,
                use_pos_token=use_pos_token,
                trainable_pos_token=trainable_pos_token,
                fusion_type=fusion_type,
                use_checkpoint=use_checkpoint,
            )

        # Build hybrid decoder if not provided
        if decoder is None:
            log.info("Initializing HybridDecoder")
            decoder = HybridDecoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                env_name=env_name,
                use_distance_bias=use_distance_bias,
                use_pos_token=use_pos_token,
            )

        # Ensure context_embedding_kwargs forwarded to PARCOPolicy
        # include num_heads so CommunicationLayer uses matching dims.
        _context_kw = kwargs.pop("context_embedding_kwargs", {})
        _context_kw.setdefault("num_heads", num_heads)

        # Initialise PARCOPolicy with the hybrid components.
        # Note: PARCOPolicy sees encoder/decoder as custom modules
        # (skips its own default construction).
        # env_name for the parent is "hcvrp" so PARCO's internal lookups
        # (embedding registry etc.) work — the decoder already handles this.
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            env_name="hcvrp",  # PARCO registry key
            agent_handler=agent_handler,
            use_pos_token=use_pos_token,
            trainable_pos_token=trainable_pos_token,
            context_embedding_kwargs=_context_kw,
            **kwargs,
        )

        # Store original env_name for reference
        self.hybrid_env_name = env_name

    def forward(
        self,
        td: TensorDict,
        env,
        phase: str = "train",
        calc_reward: bool = True,
        return_actions: bool = True,
        return_sum_log_likelihood: bool = True,
        actions=None,
        max_steps: int = 1_000_000,
        return_init_embeds: bool = True,
        **decoding_kwargs,
    ) -> dict:
        """Run the full encode → autoregressive decode loop.

        Identical to :meth:`PARCOPolicy.forward` except the encoder call
        passes ``phase`` so that the RRNCO DistanceExpert can use the
        correct sampling strategy during train vs. test.

        Args:
            td: TensorDict from ``env.reset()``.
            env: The RL4CO environment instance.
            phase: ``'train'``, ``'val'``, or ``'test'``.
            calc_reward: Whether to compute reward.
            return_actions: Whether to include actions in the output dict.
            return_sum_log_likelihood: Whether to sum log-likelihoods.
            actions: Pre-specified actions for evaluation mode.
            max_steps: Safety limit on decode steps.
            return_init_embeds: Whether to return initial embeddings
                (used by SymNCO loss).
            **decoding_kwargs: Extra kwargs for the decoding strategy.

        Returns:
            Dictionary with keys: ``reward``, ``log_likelihood``,
            ``halting_ratio``, ``actions`` (optional),
            ``init_embeds`` (optional), ``steps``.
        """
        # ── Import helpers lazily (same as PARCOPolicy) ──────────────
        from parco.models.decoding_strategies import (
            PARCODecodingStrategy,
            parco_get_decoding_strategy,
        )
        from parco.models.utils import get_log_likelihood

        # ═══════════════════════════════════════════════════════════════
        # 1. ENCODER — the only line that differs from PARCOPolicy
        # ═══════════════════════════════════════════════════════════════
        hidden, init_embeds = self.encoder(td, phase=phase)

        # ── NaN guard: sanitize encoder outputs before decoding ──────
        if isinstance(hidden, (list, tuple)):
            hidden = tuple(
                torch.nan_to_num(h, nan=0.0, posinf=1e4, neginf=-1e4)
                if isinstance(h, torch.Tensor) else h
                for h in hidden
            )
        elif isinstance(hidden, torch.Tensor):
            if torch.isnan(hidden).any() or torch.isinf(hidden).any():
                log.warning("NaN/Inf detected in encoder hidden — clamping")
                hidden = torch.nan_to_num(hidden, nan=0.0, posinf=1e4, neginf=-1e4)
        if isinstance(init_embeds, torch.Tensor) and (
            torch.isnan(init_embeds).any() or torch.isinf(init_embeds).any()
        ):
            log.warning("NaN/Inf detected in init_embeds — clamping")
            init_embeds = torch.nan_to_num(init_embeds, nan=0.0, posinf=1e4, neginf=-1e4)

        # ═══════════════════════════════════════════════════════════════
        # 2. DECODE SETUP  (same as PARCOPolicy.forward)
        # ═══════════════════════════════════════════════════════════════
        decode_type = decoding_kwargs.pop("decode_type", None)
        if actions is not None:
            decode_type = "evaluate"
        elif decode_type is None:
            decode_type = getattr(self, f"{phase}_decode_type")

        num_samples = decoding_kwargs.pop("num_samples", 1)
        if "sampling" not in decode_type:
            num_samples = 1

        # num_starts is a training-only RL concept (used in shared_step for
        # SymNCO loss_ps). PARCODecodingStrategy does not know about it, so
        # we must strip it here before forwarding kwargs to the strategy.
        decoding_kwargs.pop("num_starts", None)

        num_agents = td["action_mask"].shape[-2]

        decode_strategy: PARCODecodingStrategy = parco_get_decoding_strategy(
            decode_type,
            num_agents=num_agents,
            agent_handler=self.agent_handler,
            use_init_logp=self.use_init_logp,
            mask_handled=self.mask_handled,
            replacement_value_key=self.replacement_value_key,
            num_samples=num_samples,
            **decoding_kwargs,
        )

        # Pre-decoding hooks
        td, env, num_samples = decode_strategy.pre_decoder_hook(td, env)
        td, env, hidden = self.decoder.pre_decoder_hook(
            td, env, hidden, num_samples
        )

        do_unbatchify = num_samples > 1

        # ═══════════════════════════════════════════════════════════════
        # 3. AUTOREGRESSIVE DECODE LOOP  (same as PARCOPolicy.forward)
        # ═══════════════════════════════════════════════════════════════
        step = 0
        while not td["done"].all():
            if step > 1:
                do_unbatchify = False

            logits, mask = self.decoder(
                td, hidden, num_samples, do_unbatchify
            )
            td = decode_strategy.step(
                logits,
                mask,
                td,
                action=actions[..., step] if actions is not None else None,
            )
            td = env.step(td)["next"]
            step += 1
            if step > max_steps:
                log.error(
                    f"Exceeded maximum number of steps ({max_steps}) "
                    "during decoding"
                )
                break

        # ═══════════════════════════════════════════════════════════════
        # 4. POST-DECODE  (same as PARCOPolicy.forward)
        # ═══════════════════════════════════════════════════════════════
        (
            logprobs,
            actions,
            td,
            env,
            halting_ratio,
        ) = decode_strategy.post_decoder_hook(td, env)

        if calc_reward:
            td.set("reward", env.get_reward(td, actions))

        outdict = {
            "reward": td["reward"],
            "log_likelihood": get_log_likelihood(
                logprobs,
                actions,
                td.get("mask", None),
                return_sum_log_likelihood,
            ),
            "halting_ratio": halting_ratio,
            "final_td": td,  # terminal state for diagnostic metrics
        }

        if return_actions:
            outdict["actions"] = actions
        if return_init_embeds:
            outdict["init_embeds"] = init_embeds
        outdict["steps"] = step

        return outdict
