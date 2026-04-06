"""Inference script for hybrid RRNCO-PARCO model from a training checkpoint.

Usage examples:
    # Greedy decoding on all test data
    python inference.py --checkpoint logs/train/runs/2026-03-31_23-36-31/checkpoints/last.ckpt

    # Greedy on a specific dataset
    python inference.py \
        --checkpoint logs/train/runs/2026-03-31_23-36-31/checkpoints/epoch_005.ckpt \
        --datasets data/hcvrp_real/test_real_n100_m7.npz

    # Sampling decoding (augmented inference)
    python inference.py \
        --checkpoint logs/train/runs/2026-03-31_23-36-31/checkpoints/last.ckpt \
        --decode_type sampling --sample_size 100

    # Use a specific epoch checkpoint
    python inference.py \
        --checkpoint logs/train/runs/2026-03-31_23-36-31/checkpoints/epoch_003.ckpt
"""

import argparse
import glob
import os
import time
import warnings

import torch
from omegaconf import DictConfig, ListConfig
from omegaconf.base import ContainerMetadata
from omegaconf.nodes import ValueNode
from rl4co.data.utils import load_npz_to_tensordict
from tqdm.auto import tqdm

# Allow loading Hydra/OmegaConf configs from checkpoints (PyTorch 2.6+).
# Checkpoints saved with Hydra configs embed various OmegaConf internal types.
torch.serialization.add_safe_globals([DictConfig, ListConfig, ContainerMetadata, ValueNode])

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".gitignore", pythonpath=True)

from hybrid.envs import HCVRPRealEnv
from hybrid.models import HybridRLModule
from parco.tasks.eval import get_dataloader

warnings.filterwarnings("ignore", category=FutureWarning)
torch.set_float32_matmul_precision("medium")


def find_best_checkpoint(ckpt_dir: str) -> str:
    """Find the best (latest epoch) checkpoint in a directory.

    Prefers 'last.ckpt' if it exists, otherwise picks the highest epoch number.
    """
    last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
    if os.path.exists(last_ckpt):
        return last_ckpt

    epoch_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.ckpt")))
    if epoch_ckpts:
        return epoch_ckpts[-1]  # highest epoch

    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load the HybridRLModule from a checkpoint and return the policy on device."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    assert os.path.exists(checkpoint_path), (
        f"Checkpoint not found at {checkpoint_path}"
    )

    # Try loading with safe globals first; fall back to weights_only=False
    # if the checkpoint contains additional unpickling types.
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


def run_inference(
    policy,
    env,
    data_paths: list,
    device: torch.device,
    decode_type: str = "greedy",
    sample_size: int = None,
    batch_size: int = 1,
    save_results: bool = False,
    output_dir: str = None,
):
    """Run inference on a list of datasets and print results."""

    all_results = {}

    for dataset_path in data_paths:
        dataset_name = os.path.basename(dataset_path)
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        costs = []
        inference_times = []
        eval_steps = []

        td_test = load_npz_to_tensordict(dataset_path)
        dataloader = get_dataloader(td_test, batch_size=batch_size)
        num_instances = len(td_test)

        print(f"  Instances: {num_instances}")
        print(f"  Decode type: {decode_type}")
        if decode_type == "sampling":
            print(f"  Sample size: {sample_size}")

        ctx = (
            torch.autocast("cuda")
            if device.type == "cuda"
            else torch.inference_mode()
        )

        with ctx:
            with torch.inference_mode():
                for td_test_batch in tqdm(dataloader, desc=f"  Evaluating"):
                    td_reset = env.reset(td_test_batch).to(device)

                    start_time = time.time()
                    out = policy(
                        td_reset,
                        env,
                        decode_type=decode_type,
                        num_samples=sample_size,
                        return_actions=False,
                    )
                    end_time = time.time()

                    inference_time = end_time - start_time

                    if decode_type == "greedy":
                        costs.extend((-out["reward"]).tolist())
                    else:
                        costs.extend(
                            (-out["reward"]
                             .reshape(-1, sample_size)
                             .max(dim=-1)[0]).tolist()
                        )

                    inference_times.append(inference_time)
                    if "steps" in out:
                        eval_steps.append(out["steps"])

        avg_cost = sum(costs) / len(costs)
        total_time = sum(inference_times)
        avg_time = total_time / len(inference_times)

        print(f"\n  Results:")
        print(f"    Average cost:           {avg_cost:.4f}")
        print(f"    Per-batch infer time:   {avg_time:.4f}s")
        print(f"    Total inference time:   {total_time:.4f}s")
        if eval_steps:
            avg_steps = sum(eval_steps) / len(eval_steps)
            print(f"    Average eval steps:     {avg_steps:.2f}")

        all_results[dataset_name] = {
            "avg_cost": avg_cost,
            "total_time": total_time,
            "avg_time_per_batch": avg_time,
            "costs": costs,
        }

        if save_results and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            result_file = os.path.join(
                output_dir,
                f"results_{dataset_name.replace('.npz', '')}_{decode_type}.pt",
            )
            torch.save(
                {
                    "costs": costs,
                    "avg_cost": avg_cost,
                    "total_time": total_time,
                    "decode_type": decode_type,
                    "sample_size": sample_size,
                },
                result_file,
            )
            print(f"    Results saved to: {result_file}")

    # Summary table
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"  {'Dataset':<50} {'Avg Cost':>10}")
        print(f"  {'-'*50} {'-'*10}")
        for name, res in all_results.items():
            print(f"  {name:<50} {res['avg_cost']:>10.4f}")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Inference from hybrid RRNCO-PARCO checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (.ckpt). "
             "If not specified, uses last.ckpt from the default run directory.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="logs/train/runs/2026-03-31_23-36-31/checkpoints",
        help="Directory containing checkpoints (used if --checkpoint is not set).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="Path(s) to .npz dataset file(s). "
             "If not specified, uses all .npz files in data/hcvrp_real/.",
    )
    parser.add_argument(
        "--decode_type",
        type=str,
        default="greedy",
        choices=["greedy", "sampling"],
        help="Decoding strategy: greedy or sampling",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples for sampling decoding",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: cuda or cpu",
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        default=False,
        help="Save detailed results to files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results (if --save_results is set)",
    )

    opts = parser.parse_args()

    # --- Device ---
    if "cuda" in opts.device and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Checkpoint ---
    if opts.checkpoint is not None:
        checkpoint_path = opts.checkpoint
    else:
        checkpoint_path = find_best_checkpoint(opts.checkpoint_dir)
    print(f"Checkpoint: {checkpoint_path}")

    # --- Validation ---
    if opts.decode_type == "greedy":
        assert opts.sample_size is None or opts.sample_size == 1, (
            "Greedy decoding only supports sample_size=1"
        )
    if opts.decode_type == "sampling":
        assert opts.sample_size is not None, (
            "Must specify --sample_size for sampling decoding"
        )
        assert opts.batch_size == 1, (
            "Only batch_size=1 is supported for sampling decoding"
        )

    # --- Datasets ---
    if opts.datasets is not None:
        data_paths = opts.datasets
    else:
        data_dir = "./data/hcvrp_real"
        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"Data directory {data_dir} not found. "
                "Please generate or download the data first."
            )
        data_paths = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".npz") and "sol" not in f
        ])

    if not data_paths:
        raise FileNotFoundError("No dataset files found!")

    print(f"Datasets to evaluate: {len(data_paths)}")
    for p in data_paths:
        print(f"  - {p}")

    # --- Load model ---
    policy = load_model(checkpoint_path, device)

    # --- Environment ---
    env = HCVRPRealEnv()

    # --- Run inference ---
    results = run_inference(
        policy=policy,
        env=env,
        data_paths=data_paths,
        device=device,
        decode_type=opts.decode_type,
        sample_size=opts.sample_size,
        batch_size=opts.batch_size,
        save_results=opts.save_results,
        output_dir=opts.output_dir,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
