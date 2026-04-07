"""Diagnose workload imbalance without modifying the model.

Usage examples:
    python scripts/diagnose_workload.py --checkpoint logs/train/runs/2026-03-31_23-36-31/checkpoints/last.ckpt
    python scripts/diagnose_workload.py --checkpoint_dir logs/train/runs/2026-03-31_23-36-31/checkpoints
"""

import argparse
import glob
import os

import torch
from omegaconf import DictConfig, ListConfig
from omegaconf.base import ContainerMetadata
from omegaconf.nodes import ValueNode
from rl4co.data.utils import load_npz_to_tensordict

# Allow loading Hydra/OmegaConf configs from checkpoints (PyTorch 2.6+).
torch.serialization.add_safe_globals([DictConfig, ListConfig, ContainerMetadata, ValueNode])

import pyrootutils
pyrootutils.setup_root(__file__, indicator=".gitignore", pythonpath=True)

from hybrid.envs import HCVRPRealEnv
from hybrid.models import HybridRLModule


def find_best_checkpoint(ckpt_dir: str) -> str:
    """Find the best (latest epoch) checkpoint in a directory."""
    last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
    if os.path.exists(last_ckpt):
        return last_ckpt

    epoch_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.ckpt")))
    if epoch_ckpts:
        return epoch_ckpts[-1]

    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load the HybridRLModule from a checkpoint and return the policy on device."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"

    try:
        model = HybridRLModule.load_from_checkpoint(
            checkpoint_path, map_location="cpu", strict=False
        )
    except Exception:
        print("  Safe-globals load failed; retrying with weights_only=False")
        model = HybridRLModule.load_from_checkpoint(
            checkpoint_path, map_location="cpu", strict=False, weights_only=False
        )
    policy = model.policy.to(device).eval()
    return policy


def main():
    parser = argparse.ArgumentParser(description="Diagnose workload imbalance")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="logs/train/runs/2026-03-31_23-36-31/checkpoints",
        help="Directory containing checkpoints",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/hcvrp_real/val_real_n100_m7.npz",
        help="Path to dataset file",
    )
    parser.add_argument("--num_instances", type=int, default=50, help="Number of instances to analyze")
    parser.add_argument("--num_show", type=int, default=5, help="Number of instances to show in detail")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    opts = parser.parse_args()

    # Device
    if "cuda" in opts.device and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Checkpoint
    if opts.checkpoint is not None:
        checkpoint_path = opts.checkpoint
    else:
        checkpoint_path = find_best_checkpoint(opts.checkpoint_dir)

    # Load model
    policy = load_model(checkpoint_path, device)

    # Load test data
    print(f"Loading dataset: {opts.dataset}")
    td_test = load_npz_to_tensordict(opts.dataset)
    env = HCVRPRealEnv()

    # Run inference and collect statistics
    print(f"\nAnalyzing {opts.num_instances} instances...")
    with torch.inference_mode():
        td = env.reset(td_test[: opts.num_instances]).to(device)
        out = policy(td, env, decode_type="greedy", return_actions=True)

        # Get final state from output
        final_td = out["final_td"]
        
        # Debug: show available keys
        print(f"Final TD keys: {list(final_td.keys())}")

        # Analyze actions
        actions = out["actions"]  # [B, steps, 2] - (agent_idx, node_idx)

        # Count visits per agent
        num_agents = td["capacity"].shape[-1]
        batch_size = actions.shape[0]

        # Aggregate statistics
        all_counts = []
        all_makespans = []
        
        # Use current_length as the time metric (or used_capacity as proxy for workload)
        time_key = "current_length" if "current_length" in final_td.keys() else None

        for b in range(batch_size):
            agent_actions = actions[b, :, 0]  # Agent indices
            counts = [(agent_actions >= a).sum().item() for a in range(num_agents)]  # Fixed: == instead of >=
            all_counts.append(counts)
            
            # Get makespan from current_length or compute from used_capacity
            if time_key:
                all_makespans.append(final_td[time_key][b].max().item())
            else:
                # Fallback: use negative reward as cost
                all_makespans.append(-out["reward"][b].item())

            # Show detailed info for first few instances
            if b < opts.num_show:
                print(f"\n{'='*60}")
                print(f"Instance {b}")
                print(f"{'='*60}")
                print(f"Agent capacities: {[f'{c:.2f}' for c in td['capacity'][b].tolist()]}")
                print(f"Agent speeds:     {[f'{s:.2f}' for s in td['speed'][b].tolist()]}")
                print()

                for a in range(num_agents):
                    count = counts[a]
                    cap = td["capacity"][b, a].item()
                    spd = td["speed"][b, a].item()
                    time = final_td["current_node"][b, a].item()
                    print(f"  Agent {a}: {count:3d} customers | cap={cap:.2f}, speed={spd:.2f}, time={time:.4f}")

                print(f"\nFinal times: {[f'{t:.4f}' for t in final_td['current_node'][b].tolist()]}")
                print(f"Makespan:    {all_makespans[-1]:.4f}")

        # Summary statistics
        print(f"\n{'='*60}")
        print("SUMMARY STATISTICS")
        print(f"{'='*60}")

        all_counts_tensor = torch.tensor(all_counts, dtype=torch.float)
        mean_counts = all_counts_tensor.mean(dim=0)
        std_counts = all_counts_tensor.std(dim=0)
        
        print(f"\nCustomers per agent (mean ± std across {batch_size} instances):")
        for a in range(num_agents):
            mean_cap = td["capacity"][:, a].mean().item()
            mean_spd = td["speed"][:, a].mean().item()
            print(f"  Agent {a}: {mean_counts[a]:.1f} ± {std_counts[a]:.1f} customers | avg_cap={mean_cap:.2f}, avg_speed={mean_spd:.2f}")

        # Workload imbalance metrics
        workload_std = all_counts_tensor.std(dim=1)  # Std across agents per instance
        print(f"\nWorkload imbalance (std of customer counts per instance):")
        print(f"  Mean: {workload_std.mean():.2f}")
        print(f"  Max:  {workload_std.max():.2f}")
        print(f"  Min:  {workload_std.min():.2f}")

        # Correlation with capacity/speed
        capacities = td["capacity"].cpu()
        speeds = td["speed"].cpu()
        
        # Check if high-capacity/fast agents serve more customers
        print(f"\nCorrelation analysis:")
        for b in range(min(10, batch_size)):
            counts_b = torch.tensor(all_counts[b], dtype=torch.float)
            cap_corr = torch.corrcoef(torch.stack([counts_b, capacities[b]]))[0, 1].item()
            spd_corr = torch.corrcoef(torch.stack([counts_b, speeds[b]]))[0, 1].item()
            if b < 3:
                print(f"  Instance {b}: capacity_corr={cap_corr:.3f}, speed_corr={spd_corr:.3f}")

        # Average correlation
        cap_corrs = []
        spd_corrs = []
        for b in range(batch_size):
            counts_b = torch.tensor(all_counts[b], dtype=torch.float)
            cap_corrs.append(torch.corrcoef(torch.stack([counts_b, capacities[b]]))[0, 1].item())
            spd_corrs.append(torch.corrcoef(torch.stack([counts_b, speeds[b]]))[0, 1].item())

        print(f"\n  Average capacity correlation: {sum(cap_corrs)/len(cap_corrs):.3f}")
        print(f"  Average speed correlation:    {sum(spd_corrs)/len(spd_corrs):.3f}")

        print(f"\nAverage makespan: {sum(all_makespans)/len(all_makespans):.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()