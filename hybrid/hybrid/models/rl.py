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

from random import randint
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from typing import Any, Optional

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.zoo.symnco.losses import (
    invariance_loss,
    problem_symmetricity_loss,
    solution_symmetricity_loss,
)
from rl4co.utils.ops import unbatchify
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
    - **Reward shaping**: adds a ``-remaining_nodes_ratio`` term to provide
      dense per-step signal (controlled by ``reward_shaping_coeff``).
    - **Entropy bonus**: adds ``-entropy_coeff * entropy`` to the loss to
      encourage exploration and prevent premature policy collapse.
    - **Multi-start**: supports ``num_starts > 1`` for escaping local optima
      (activates ``loss_ps`` — problem symmetricity loss).

    Everything else — SymNCO losses, symmetric augmentation, projection
    head, curriculum-style agent/size sampling, multi-dataloader
    val/test — is inherited unchanged from :class:`PARCORLModule`.

    Args:
        env: The :class:`HCVRPRealEnv` (or compatible) environment.
        policy: A :class:`HybridPolicy` (or compatible).  If ``None``,
            a default :class:`HybridPolicy` is built.
        num_augment: Number of symmetric augmentations (default 4).
        alpha: Weight for invariance loss (default 0.1).
        beta: Weight for solution symmetricity loss (default 0.5).
        train_min_agents: Min agents sampled per training step.
        train_max_agents: Max agents sampled per training step.
        train_min_size: Min customer locations sampled per training step.
        train_max_size: Max customer locations sampled per training step.
        reward_shaping_coeff: Weight for the ``remaining_nodes_ratio`` penalty
            in the shaped reward.  Set to 0.0 to disable (default 0.05).
        entropy_coeff: Weight for the entropy bonus term in the loss.
            Set to 0.0 to disable (default 0.01).
        warmup_epochs: Number of epochs to linearly warm up the LR from
            0 → base_lr.  Helps prevent NaN logits from large gradients at
            initialisation.  Set to 0 to disable (default 1).
        **kwargs: Forwarded to :class:`PARCORLModule` → SymNCO →
            REINFORCE → RL4COLitModule.  Useful keys include
            ``optimizer_kwargs``, ``lr_scheduler``, ``batch_size``, etc.
    """

    def __init__(
        self,
        env: RL4COEnvBase,
        policy: Optional[nn.Module] = None,
        # ── SymNCO / augmentation ────────────────────────────────
        num_augment: int = 4,
        alpha: float = 0.1,
        beta: float = 0.5,
        # ── Curriculum sampling ──────────────────────────────────
        train_min_agents: int = 5,
        train_max_agents: int = 15,
        train_min_size: int = 50,
        train_max_size: int = 100,
        # ── Reward shaping ───────────────────────────────────────
        reward_shaping_coeff: float = 0.05,
        # ── Entropy bonus ────────────────────────────────────────
        entropy_coeff: float = 0.01,
        # ── LR warmup ────────────────────────────────────────────
        warmup_epochs: int = 1,
        # ── VRAM optimize ────────────────────────────────────────
        micro_batch_aug: int = 0,
        # ── Rest ─────────────────────────────────────────────────
        **kwargs,
    ):
        self.reward_shaping_coeff = reward_shaping_coeff
        self.entropy_coeff = entropy_coeff
        self.warmup_epochs = warmup_epochs
        self.micro_batch_aug = micro_batch_aug
        if micro_batch_aug == 2 and num_augment > 4:
            log.warning(
                "micro_batch_aug=2 with n_aug=%d: forward-chunked only, backward still B*n_aug; "
                "consider micro_batch_aug=1 or per-chunk manual_backward if backward OOM",
                num_augment,
            )
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

    def configure_optimizers(self):
        """Override to prepend a linear LR warmup before the parent scheduler.

        The parent (RL4COLitModule) returns one of:
          - A bare ``optimizer``              (no LR scheduler configured)
          - ``([optimizer], scheduler_cfg)``  (tuple when scheduler is set)

        If ``warmup_epochs <= 0`` the parent behaviour is unchanged.
        """
        result = super().configure_optimizers()

        if self.warmup_epochs <= 0:
            return result

        # ── Unpack parent result ──────────────────────────────────────
        # Case 1: ([optimizer], {scheduler_cfg})  ← tuple returned by parent
        if isinstance(result, tuple) and len(result) == 2:
            optimizer_list, main_scheduler_cfg = result
            optimizer = optimizer_list[0] if isinstance(optimizer_list, list) else optimizer_list
        # Case 2: bare optimizer (no scheduler configured)
        elif isinstance(result, torch.optim.Optimizer):
            optimizer = result
            main_scheduler_cfg = None
        else:
            # Fallback: unknown shape, return as-is
            log.warning("configure_optimizers: unrecognised parent result shape, skipping warmup")
            return result

        # ── Build linear warmup: LR ramps 0.01 % → 100 % of base LR ──
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-4,
            end_factor=1.0,
            total_iters=self.warmup_epochs,
        )

        if main_scheduler_cfg is not None:
            # main_scheduler_cfg is a dict: {"scheduler": ..., "interval": ..., ...}
            main_scheduler = main_scheduler_cfg["scheduler"]
            interval = main_scheduler_cfg.get("interval", self.lr_scheduler_interval)
            monitor = main_scheduler_cfg.get("monitor", self.lr_scheduler_monitor)

            # Chain: warmup for warmup_epochs, then hand off to main scheduler
            combined = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[self.warmup_epochs],
            )
            return [optimizer], {
                "scheduler": combined,
                "interval": interval,
                "monitor": monitor,
            }
        else:
            return [optimizer], {
                "scheduler": warmup_scheduler,
                "interval": "epoch",
                "monitor": self.lr_scheduler_monitor,
            }

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        """Override to use :func:`resample_batch_hybrid` instead of
        PARCO's :func:`resample_batch`.

        Enhancements over the parent:
        - Reward shaping: adds ``-reward_shaping_coeff * remaining_nodes_ratio``
          to give the policy a denser learning signal.
        - Entropy bonus: subtracts ``entropy_coeff * mean_entropy`` from the
          loss to prevent premature policy collapse.
        - Multi-start: activates ``loss_ps`` when ``num_starts > 1``.
        """

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

        # ── NaN guard: check batch inputs after reset ─────────────
        for key in ("locs", "distance_matrix", "demand", "capacity", "speed"):
            val = td.get(key, None)
            if val is not None and torch.isnan(val).any():
                log.warning(f"NaN detected in td['{key}'] after env.reset — clamping")
                td[key] = torch.nan_to_num(val, nan=0.0, posinf=1e4, neginf=-1e4)

        # Augmentation is a training-only trick (SymNCO).
        # At val/test we decode greedily on the original instance to
        # avoid multiplying the batch by n_aug and causing OOM on the
        # [B, N, N, H] tensors inside AttnFreeNet.
        n_aug = self.num_augment if phase == "train" else 1
        # Multi-start is train-only: at val/test use greedy (n_start=1)
        # so that unbatchify(reward, (n_start, n_aug)) always matches the
        # actual output size returned by the policy.
        n_start = self.num_starts if phase == "train" else 1

        # ── Symmetric augmentation (train only) ──────────────────
        if n_aug > 1:
            td = self.augment(td)

        # ── Policy forward ───────────────────────────────────────
        # Pass num_starts only during training so the policy actually
        # performs multi-start decoding (expands the batch internally).
        # VRAM optimize: forward-chunk over aug dimension.
        # shortcut: forward-chunked only, single loss/backward on concatenated
        # tensors — peak backward still B*n_aug. Upgrade to per-chunk
        # manual_backward (requires Lightning automatic_optimization=False) when
        # n_aug > 4 and backward OOM observed. ADR: forward chunking saves
        # encoder/decoder forward activations; backward saving deferred.
        if phase == "train" and n_aug > 1 and getattr(self, "micro_batch_aug", 0):
            # normalize micro_batch_aug: 0=off, int in [1, n_aug) = chunk size in aug units
            mb_raw = self.micro_batch_aug
            mb = int(mb_raw) if isinstance(mb_raw, int) and not isinstance(mb_raw, bool) else 2
            if 0 < mb < n_aug:
                # capture B_orig before augment; fallback to td // n_aug if batch not TensorDict
                try:
                    B_orig = int(batch.batch_size[0])
                except Exception:
                    B_orig = int(td.batch_size[0] // n_aug)
                chunk_B = B_orig * mb
                total = int(td.batch_size[0])
                n_chunks = (total + chunk_B - 1) // chunk_B
                outs = []
                for i in range(n_chunks):
                    chunk = td[i * chunk_B : (i + 1) * chunk_B]
                    out_c = self.policy(
                        chunk,
                        self.env,
                        phase=phase,
                        num_starts=n_start,
                        return_init_embeds=True,
                    )
                    outs.append(out_c)
                # merge outs: concat tensors along batch dim; TensorDict via torch.cat
                out = {}
                for k in outs[0].keys():
                    v0 = outs[0][k]
                    if isinstance(v0, torch.Tensor):
                        out[k] = torch.cat([o[k] for o in outs], dim=0)
                    elif hasattr(v0, "batch_size"):  # TensorDict (final_td, etc.)
                        try:
                            out[k] = torch.cat([o[k] for o in outs], dim=0)
                        except Exception:
                            out[k] = outs[0][k]
                    else:
                        out[k] = outs[0][k]
                # final_td for diagnostics: prefer concatenated full batch; fallback to first chunk
                if "final_td" in outs[0] and outs[0]["final_td"] is not None:
                    if "final_td" not in out or out["final_td"] is outs[0]["final_td"]:
                        # already handled above; if cat failed it is first chunk (biased diagnostic)
                        pass
            else:
                out = self.policy(
                    td,
                    self.env,
                    phase=phase,
                    num_starts=n_start if phase == "train" else 1,
                    return_init_embeds=True,
                )
        else:
            out = self.policy(
                td,
                self.env,
                phase=phase,
                num_starts=n_start if phase == "train" else 1,
                return_init_embeds=True,
            )

        # Unbatchify reward to [batch_size, n_start, n_aug]
        reward = unbatchify(out["reward"], (n_start, n_aug))

        # ── Training loss (SymNCO + shaping + entropy) ───────────
        if phase == "train":
            # out["log_likelihood"]: [B*n_start*n_aug, num_agents, (decode_len)]
            # After unbatchify: [B, n_start, n_aug, num_agents, ...]
            # Sum over all trailing dims (agents + steps) → [B, n_start, n_aug]
            # so that ll matches reward for the SymNCO loss functions.
            ll_raw = unbatchify(out["log_likelihood"], (n_start, n_aug))
            ll = ll_raw.sum(dim=tuple(range(3, ll_raw.dim())))

            # ── 🔥 Reward shaping ────────────────────────────────
            # Penalise having many unvisited nodes at episode end.
            # ``visited_count`` is in [0, num_locs]; normalise to [0, 1].
            shaped_reward = reward.clone()
            if self.reward_shaping_coeff > 0 and "visited_count" in out:
                visited = out["visited_count"].float()  # [B*n_start*n_aug]
                visited = unbatchify(visited, (n_start, n_aug))
                total_nodes = float(num_locs)
                remaining_ratio = (total_nodes - visited) / total_nodes
                shaped_reward = reward - self.reward_shaping_coeff * remaining_ratio

            proj_embeds = self.projection_head(out["init_embeds"])
            loss_inv = invariance_loss(proj_embeds, n_aug) if n_aug > 1 else 0

            # ── 🔥 Multi-start loss_ps ───────────────────────────
            # reward: [B, n_start, n_aug] — average over n_start (dim=1)
            loss_ps = (
                problem_symmetricity_loss(shaped_reward, ll, dim=1)
                if n_start > 1
                else 0
            )

            # ── loss_ss (augmentation symmetricity) ─────────────
            # reward: [B, n_start, n_aug] — average over n_aug (dim=2)
            loss_ss = (
                solution_symmetricity_loss(shaped_reward, ll, dim=2)
                if n_aug > 1
                else 0
            )
            loss = loss_ps + self.beta * loss_ss + self.alpha * loss_inv

            # ── 🔥 Entropy bonus ─────────────────────────────────
            # log_likelihood is the sum of log-probs; approximate entropy as
            # the negative mean of the per-action log-probs.
            if self.entropy_coeff > 0:
                # ll shape: [B, n_start, n_aug]  (sum of log-probs per route)
                # Entropy proxy: -mean(ll) — higher (less negative) ll → lower entropy
                entropy_bonus = -ll.mean()
                loss = loss - self.entropy_coeff * entropy_bonus

            # Ensure loss is a tensor (needed when n_aug=1 → all losses are 0)
            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(loss, device=td.device, requires_grad=True)

            out.update(
                {
                    "loss": loss,
                    "loss_ss": loss_ss,
                    "loss_ps": loss_ps,
                    "loss_inv": loss_inv,
                }
            )

        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)

        # Log SymNCO sub-losses for visibility
        if phase == "train":
            for key in ("loss_inv", "loss_ps", "loss_ss"):
                val = out.get(key, 0)
                if isinstance(val, torch.Tensor):
                    val = val.mean()
                self.log(
                    f"train/{key}",
                    val,
                    on_step=self.log_on_step,
                    on_epoch=False,
                    prog_bar=False,
                    sync_dist=True,
                )

            # Log entropy bonus separately
            if self.entropy_coeff > 0:
                self.log(
                    "train/entropy_proxy",
                    -ll.mean(),
                    on_step=self.log_on_step,
                    on_epoch=False,
                    prog_bar=False,
                    sync_dist=True,
                )

        # ── 📊 Enhanced diagnostic metrics ────────────────────────
        # Use terminal state (after full decode) instead of reset state,
        # so that current_length, visited, etc. reflect the completed episode.
        final_td = out.get("final_td", td)
        self._log_diagnostic_metrics(final_td, out, phase, dataloader_idx)

        return {"loss": out.get("loss", None), **metrics}

    # ══════════════════════════════════════════════════════════════
    # Diagnostic Metrics
    # ══════════════════════════════════════════════════════════════

    def _log_diagnostic_metrics(
        self, td, out, phase: str, dataloader_idx=None
    ):
        """Log comprehensive diagnostic metrics for monitoring training health.

        Metrics logged:
            1. **distance_bias/alpha**: Learnable weight controlling how much
               the decoder relies on the distance matrix.  If it trends → 0
               the model is ignoring distance information.
            2. **agents/max_time**: The worst-agent makespan (= -reward).
            3. **agents/mean_time**: Mean agent completion time.
            4. **agents/time_std**: Std dev of agent times — lower means
               better workload balance.
            5. **agents/time_ratio**: max_time / mean_time — closer to 1.0
               means agents are well balanced.
            6. **decode/steps**: Number of autoregressive decode steps.
            7. **decode/visited_ratio**: Fraction of customers actually
               served (1.0 = feasible solution).
        """
        suffix = ""
        if dataloader_idx is not None and self.dataloader_names is not None:
            suffix = "/" + self.dataloader_names[dataloader_idx]
        on_step = self.log_on_step if phase == "train" else False
        on_epoch = False if phase == "train" else True

        def _log(name, value, prog=False):
            if isinstance(value, torch.Tensor):
                value = value.mean()
            self.log(
                f"{phase}/{name}{suffix}",
                value,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=prog,
                sync_dist=True,
            )

        # ── 1. Distance bias alpha (learnable parameter) ─────────
        # Hetero: alpha_net outputs per-agent [B,m] alpha; log its mean
        decoder = self.policy.decoder
        if hasattr(decoder, "alpha_net") and getattr(decoder, "use_distance_bias", False):
            try:
                if "agents_capacity" in td.keys() and "agents_speed" in td.keys():
                    cap, spd = td["agents_capacity"], td["agents_speed"]
                elif "capacity" in td.keys() and "speed" in td.keys():
                    cap, spd = td["capacity"], td["speed"]
                else:
                    cap = spd = None
                if cap is not None:
                    with torch.no_grad():
                        feat = torch.stack([cap.float(), spd.float()], dim=-1)  # [B,m,2]
                        alphas = decoder.alpha_net(feat).squeeze(-1).abs()  # [B,m]
                        _log("distance_bias/alpha", alphas.mean())
                else:
                    _log("distance_bias/alpha", decoder.alpha_net[2].bias.data.mean())
            except Exception:
                # Fallback: mean of last bias param
                try:
                    _log("distance_bias/alpha", decoder.alpha_net[2].bias.data.mean())
                except Exception:
                    pass
        elif hasattr(decoder, "alpha") and hasattr(decoder, "use_distance_bias"):
            _log("distance_bias/alpha", decoder.alpha.data.mean())

        # ── 2. Per-agent tour stats ──────────────────────────────
        # Compute agent times from the environment's final state
        if "current_length" in td.keys() and "agents_speed" in td.keys():
            from rl4co.utils.ops import gather_by_index, get_distance

            current_length = td["current_length"]  # [B, m]
            # Add final leg back to depot
            if "current_node" in td.keys() and "locs" in td.keys() and "depot_node" in td.keys():
                current_loc = gather_by_index(td["locs"], td["current_node"])
                depot_loc = gather_by_index(td["locs"], td["depot_node"])
                current_length = current_length + get_distance(depot_loc, current_loc)

            agent_times = current_length / td["agents_speed"]  # [B, m]

            max_time = agent_times.max(dim=1)[0]      # [B]
            min_time = agent_times.min(dim=1)[0]       # [B]
            mean_time = agent_times.mean(dim=1)        # [B]
            std_time = agent_times.std(dim=1)           # [B]

            _log("agents/max_time", max_time)
            _log("agents/min_time", min_time)
            _log("agents/mean_time", mean_time)
            _log("agents/time_std", std_time)
            # Ratio > 1.0 means imbalance; = 1.0 means perfect balance
            ratio = max_time / (mean_time + 1e-8)
            _log("agents/time_ratio", ratio)

            # ── Workload balance metrics (Solution 1 monitoring) ──
            # Coefficient of variation: std / mean (scale-invariant balance measure)
            cv = std_time / (mean_time + 1e-8)  # [B]
            _log("balance/time_cv", cv)

            # Max/min ratio: how many times busier is the busiest vs least busy agent
            max_min_ratio = max_time / (min_time + 1e-8)  # [B]
            _log("balance/time_max_min_ratio", max_min_ratio)

            # Balance penalty value (what actually gets added to reward)
            if hasattr(self.env, "balance_weight") and self.env.balance_weight > 0:
                balance_penalty = self.env.balance_weight * std_time  # [B]
                _log("balance/penalty", balance_penalty)
                _log("balance/weight", self.env.balance_weight)

        # ── 2b. Per-agent visit count stats ──────────────────────
        if "num_visits_per_agent" in td.keys():
            visits = td["num_visits_per_agent"]  # [B, m]
            _log("balance/visits_mean", visits.mean(dim=1))
            _log("balance/visits_std", visits.std(dim=1))
            _log("balance/visits_max", visits.max(dim=1)[0])
            _log("balance/visits_min", visits.min(dim=1)[0])
            visits_cv = visits.std(dim=1) / (visits.mean(dim=1) + 1e-8)
            _log("balance/visits_cv", visits_cv)

        # ── 3. Decode steps ──────────────────────────────────────
        if "steps" in out:
            _log("decode/steps", float(out["steps"]))

        # ── 4. Visited ratio (feasibility) ───────────────────────
        if "visited" in td.keys():
            # visited: [B, m+N] or [B, N] boolean mask
            visited = td["visited"]
            if visited.dim() >= 2:
                num_agents = td["agents_speed"].shape[-1] if "agents_speed" in td.keys() else 0
                # Only count customer nodes (skip depot/agent indices)
                customer_visited = visited[..., num_agents:] if visited.shape[-1] > num_agents else visited
                visited_ratio = customer_visited.float().mean()
                _log("decode/visited_ratio", visited_ratio, prog=(phase != "train"))

    def load_state_dict(self, state_dict, strict: bool = True):
        """Allow loading old checkpoints with ``policy.decoder.alpha``.

        Old checkpoints have a single scalar ``alpha``; new code has ``alpha_net``.
        We migrate here so ``HybridRLModule.load_from_checkpoint(..., strict=False)``
        is not required — but if strict=True and old key present, we still load
        with strict=False and seed ``alpha_net`` bias from the old value.
        """
        # Lightning checkpoints nest under "state_dict" already unwrapped by caller;
        # at module level keys look like "policy.decoder.alpha".
        has_old = any(k.endswith("decoder.alpha") for k in state_dict.keys())
        has_new = any("decoder.alpha_net" in k for k in state_dict.keys())
        if has_old and not has_new:
            # Pop old alpha keys and seed new decoder's alpha_net bias
            old_keys = [k for k in list(state_dict.keys()) if k.endswith("decoder.alpha")]
            for k in old_keys:
                old_alpha = state_dict.pop(k)
                try:
                    old_val = float(old_alpha.mean().item()) if hasattr(old_alpha, "mean") else float(old_alpha)
                except Exception:
                    old_val = 1.0
                try:
                    with torch.no_grad():
                        self.policy.decoder.alpha_net[2].bias.fill_(old_val)
                except Exception:
                    pass
            return super().load_state_dict(state_dict, strict=False)
        return super().load_state_dict(state_dict, strict=strict)

    def on_before_optimizer_step(self, optimizer):
        """Log gradient norms before optimizer step.

        This is the most important diagnostic for detecting:
        - Vanishing gradients (norm → 0): model stopped learning
        - Exploding gradients (norm → ∞): training about to diverge
        - Healthy training: norm is stable or slowly decreasing
        """
        # Total gradient norm across all parameters
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        self.log(
            "train/grad_norm",
            total_norm,
            on_step=self.log_on_step,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
        )

        # Gradient norm per component (encoder vs decoder)
        for comp_name, comp in [("encoder", self.policy.encoder),
                                 ("decoder", self.policy.decoder)]:
            comp_norm = 0.0
            for p in comp.parameters():
                if p.grad is not None:
                    comp_norm += p.grad.data.norm(2).item() ** 2
            comp_norm = comp_norm ** 0.5
            self.log(
                f"train/grad_norm/{comp_name}",
                comp_norm,
                on_step=self.log_on_step,
                on_epoch=False,
                prog_bar=False,
                sync_dist=True,
            )
