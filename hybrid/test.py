import argparse
import os
import time
import warnings

import torch

from rl4co.data.utils import load_npz_to_tensordict
from tqdm.auto import tqdm

from hybrid.envs import HCVRPRealEnv
from hybrid.models import HybridRLModule
from parco.tasks.eval import get_dataloader

warnings.filterwarnings("ignore", category=FutureWarning)

torch.set_float32_matmul_precision("medium")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problem",
        type=str,
        default="hcvrp",
        help="Problem name: hcvrp",
    )
    parser.add_argument(
        "--datasets",
        help="Filename of the dataset(s) to evaluate.",
        default=None,
    )
    parser.add_argument(
        "--decode_type",
        type=str,
        default="greedy",
        help="Decoding type. Available: greedy, sampling",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples to use for sampling decoding",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compile the model for faster inference",
    )

    opts = parser.parse_args()

    batch_size = opts.batch_size
    sample_size = opts.sample_size
    decode_type = opts.decode_type
    checkpoint_path = opts.checkpoint
    problem = opts.problem

    if "cuda" in opts.device and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if checkpoint_path is None:
        checkpoint_path = f"./checkpoints/{problem}/hybrid.ckpt"

    if decode_type == "greedy":
        assert (
            sample_size == 1 or sample_size is None
        ), "Greedy decoding only uses 1 sample"

    if opts.datasets is None:
        assert (
            problem is not None
        ), "Problem must be specified if dataset is not provided"
        if not os.path.exists(f"./data/{problem}"):
            raise FileNotFoundError(
                f"Data directory ./data/{problem} not found. "
                "Please generate or download the data first."
            )
        data_paths = [
            f"./data/{problem}/{f}"
            for f in os.listdir(f"./data/{problem}")
        ]
    else:
        data_paths = (
            [opts.datasets]
            if isinstance(opts.datasets, str)
            else opts.datasets
        )

    if decode_type == "sampling":
        assert (
            sample_size is not None
        ), "Sample size must be specified for sampling decoding"
        assert (
            batch_size == 1
        ), "Only batch_size=1 is supported for sampling decoding"

    # Exclude solution files
    data_paths = [f for f in data_paths if "sol" not in f]
    data_paths = sorted(data_paths)

    # Load checkpoint
    print("Loading checkpoint from", checkpoint_path)
    assert os.path.exists(checkpoint_path), (
        f"Checkpoint not found at {checkpoint_path}"
    )

    model = HybridRLModule.load_from_checkpoint(
        checkpoint_path, map_location="cpu", strict=False
    )
    env = HCVRPRealEnv()
    policy = model.policy.to(device).eval()

    if opts.compile:
        print("Compiling the model")
        with torch.inference_mode():
            policy(
                env.reset(env.generator(1)).to(device),
                env,
                decode_type=decode_type,
            )

    for dataset in data_paths:
        costs = []
        inference_times = []
        eval_steps = []

        print(f"Loading {dataset}")
        td_test = load_npz_to_tensordict(dataset)
        dataloader = get_dataloader(td_test, batch_size=batch_size)

        with (
            torch.autocast("cuda")
            if "cuda" in opts.device
            else torch.inference_mode()
        ):
            with torch.inference_mode():
                for td_test_batch in tqdm(dataloader):
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
                        costs.extend(-out["reward"])
                    else:
                        costs.extend(
                            -out["reward"]
                            .reshape(-1, sample_size)
                            .max(dim=-1)[0]
                        )
                    inference_times.append(inference_time)
                    eval_steps.append(out["steps"])

            print(f"Average cost: {sum(costs)/len(costs):.4f}")
            print(
                f"Per step inference time: "
                f"{sum(inference_times)/len(inference_times):.4f}s"
            )
            print(f"Total inference time: {sum(inference_times):.4f}s")
            print(
                f"Average eval steps: "
                f"{sum(eval_steps)/len(eval_steps):.2f}"
            )
