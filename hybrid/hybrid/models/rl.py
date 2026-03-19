"""Hybrid RL Module: Training wrapper for the hybrid RRNCO-PARCO architecture.

Extends :class:`PARCORLModule` (which extends SymNCO → REINFORCE →
RL4COLitModule) to handle the hybrid environment's distance matrix
during random agent/size resampling.

Training loop (inherited from PARCORLModule):
    1. Random agent/size subsampling (curriculum-like)
    2. Symmetric augmentation (dihedral group)
    3. REINFORCE with SymNCO losses:
       - Solution symmetricity loss
       - Invariance loss (via projection head on init_embeds)
"""

import torch
import torch.nn as nn

from typing import Any, Optional

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.pylogger import get_pylogger

from parco.models.rl import PARCORLModule
from parco.models.utils import replace_key_td, resample_batch

from hybrid.models.policy import HybridPolicy
from hybrid.envs import HCVRPRealEnv

log = get_pylogger(__name__)


def resample_batch_hybrid(td, num_agents, num_locs):
    """Resample a batch to a smaller number of agents and locations.

    Extends PARCO's :func:`resample_batch` by additionally slicing the
    ``distance_matrix``, ``min_distance``, and ``max_distance`` tensors
    that the hybrid environment adds.

    **Important**: This operates on the raw generator batch (before
    ``env.reset()``), where:

    - ``locs``: ``[B, N_old, 2]``  (customers only)
    - ``depot``: ``[B, 2]`` (single depot)
    - ``distance_matrix``: ``[B, 1+N_old, 1+N_old]``  (1 depot + customers)

    After slicing the distance matrix to ``[B, 1+num_locs, 1+num_locs]``,
    we re-normalise per instance so values stay in [0, 1].

    Args:
        td: TensorDict batch from the data generator.
        num_agents: Number of agents (depots) to keep.
        num_locs: Number of customer locations to keep.

    Returns:
        Resampled TensorDict.
    """
    # First, delegate to PARCO's standard resample (handles locs, demand,
    # capacity, speed, depots, etc.)
    td = resample_batch(td, num_agents, num_locs)

    # Slice distance_matrix if present.
    # In the raw generator batch, distance_matrix is [B, 1+N_old, 1+N_old]
    # (1 depot + N_old customers).  After slicing locs to num_locs customers,
    # we keep the first (1 + num_locs) rows/cols of the distance matrix
    # (depot row/col + first num_locs customer rows/cols).
    if "distance_matrix" in td.keys():
        dist = td["distance_matrix"]
        keep = 1 + num_locs  # 1 depot + num_locs customers
        dist_sliced = dist[..., :keep, :keep]

        # Re-normalise per instance (min-max to [0, 1])
        B = dist_sliced.shape[0]
        flat = dist_sliced.reshape(B, -1)
        d_min = flat.min(dim=-1).values  # [B]
        d_max = flat.max(dim=-1).values  # [B]
        denom = (d_max - d_min).clamp(min=1e-6)

        dist_norm = (dist_sliced - d_min[:, None, None]) / denom[:, None, None]
        td = replace_key_td(td, "distance_matrix", dist_norm)

        # Update min/max_distance if they exist, otherwise create them
        if "min_distance" in td.keys():
            td = replace_key_td(td, "min_distance", d_min)
        else:
            td["min_distance"] = d_min
        if "max_distance" in td.keys():
            td = replace_key_td(td, "max_distance", d_max)
        else:
            td["max_distance"] = d_max

    return td


class HybridRLModule(PARCORLModule):
    """Training module for the Hybrid RRNCO-PARCO model.

    Extends :class:`PARCORLModule` with:

    - ``resample_batch_hybrid``: slices ``distance_matrix`` alongside
      ``locs``, ``demand``, ``capacity``, ``speed`` when doing random
      agent/size subsampling during training.
    - Default hyperparameters tuned for the deeper RRNCO encoder
      (see ``plan.md`` Phase 7 table).

    Everything else — SymNCO losses, symmetric augmentation, projection
    head, curriculum-style agent/size sampling, multi-dataloader
    val/test — is inherited unchanged from :class:`PARCORLModule`.

    Args:
        env: The :class:`HCVRPRealEnv` (or compatible) environment.
        policy: A :class:`HybridPolicy` (or compatible).  If ``None``,
            a default :class:`HybridPolicy` is built.
        num_augment: Number of symmetric augmentations (default 8).
        alpha: Weight for invariance loss (default 0.2).
        beta: Weight for solution symmetricity loss (default 1.0).
        train_min_agents: Min agents sampled per training step.
        train_max_agents: Max agents sampled per training step.
        train_min_size: Min customer locations sampled per training step.
        train_max_size: Max customer locations sampled per training step.
        **kwargs: Forwarded to :class:`PARCORLModule` → SymNCO →
            REINFORCE → RL4COLitModule.  Useful keys include
            ``optimizer_kwargs``, ``lr_scheduler``, ``batch_size``, etc.
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: Optional[nn.Module] = None,
        # ── SymNCO / augmentation ────────────────────────────────
        num_augment: int = 8,
        alpha: float = 0.2,
        beta: float = 1.0,
        # ── Curriculum sampling ──────────────────────────────────
        train_min_agents: int = 5,
        train_max_agents: int = 15,
        train_min_size: int = 50,
        train_max_size: int = 100,
        # ── Rest ─────────────────────────────────────────────────
        **kwargs,
    ):
        # Build default policy if none provided
        if policy is None:
            log.info("Building default HybridPolicy for HybridRLModule")
            policy = HybridPolicy()

        super().__init__(
            env=env,
            policy=policy,
            num_augment=num_augment,
            alpha=alpha,
            beta=beta,
            train_min_agents=train_min_agents,
            train_max_agents=train_max_agents,
            train_min_size=train_min_size,
            train_max_size=train_max_size,
            **kwargs,
        )

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        """Override to use :func:`resample_batch_hybrid` instead of
        PARCO's :func:`resample_batch`.

        The only change is the resample call — everything else
        (augmentation, policy call, SymNCO losses, logging) is
        delegated to the parent.
        """
        from random import randint

        from rl4co.data.transforms import StateAugmentation
        from rl4co.models.zoo.symnco.losses import (
            invariance_loss,
            solution_symmetricity_loss,
        )
        from rl4co.utils.ops import unbatchify

        # ── Agent / size resampling (curriculum) ─────────────────
        if phase == "train":
            num_agents = randint(self.train_min_agents, self.train_max_agents)
            num_locs = randint(self.train_min_size, self.train_max_size)
            batch = resample_batch_hybrid(batch, num_agents, num_locs)
        else:
            if self.allow_multi_dataloaders:
                if dataloader_idx is not None and self.dataloader_names is not None:
                    num_agents = int(
                        self.dataloader_names[dataloader_idx].split("_")[-1][1:]
                    )
                else:
                    num_agents = self.val_test_num_agents
                batch["num_agents"] = torch.full(
                    (batch.shape[0],), num_agents, device=batch.device
                )

        # ── Env reset ────────────────────────────────────────────
        td = self.env.reset(batch)

        n_aug, n_start = self.num_augment, self.num_starts
        assert n_start <= 1, "PARCO does not support multi-start"

        # ── Symmetric augmentation ───────────────────────────────
        if n_aug > 1:
            td = self.augment(td)

        # ── Policy forward ───────────────────────────────────────
        out = self.policy(
            td,
            self.env,
            phase=phase,
            return_init_embeds=True,
        )

        # Unbatchify reward to [batch_size, n_start, n_aug]
        reward = unbatchify(out["reward"], (n_start, n_aug))

        # ── Training loss (SymNCO) ───────────────────────────────
        if phase == "train":
            ll = unbatchify(out["log_likelihood"], (n_start, n_aug))

            proj_embeds = self.projection_head(out["init_embeds"])
            loss_inv = invariance_loss(proj_embeds, n_aug) if n_aug > 1 else 0
            loss_ps = 0  # no multi-start → no problem symmetricity loss
            loss_ss = (
                solution_symmetricity_loss(reward[..., None], ll, dim=1)
                if n_aug > 1
                else 0
            )
            loss = loss_ps + self.beta * loss_ss + self.alpha * loss_inv

            out.update(
                {
                    "loss": loss,
                    "loss_ss": loss_ss,
                    "loss_ps": loss_ps,
                    "loss_inv": loss_inv,
                }
            )

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}
