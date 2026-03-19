"""Generate HCVRP datasets from RRNCO real-world OSRM distance matrices.

Bridges the RRNCO city-level data (single-vehicle CVRP with OSRM distances)
to the hybrid multi-agent HCVRP format.

Data flow:
    RRNCO city data → sample subsets → add multi-agent fields → save NPZ
    {points, distance, duration}     → {depot, locs, demand, capacity,
                                         speed, distance_matrix, num_agents}

Usage:
    # Generate in-distribution (train cities) val set
    python scripts/generate_real_data.py \
        --data_dir ../real-routing-nco/data/dataset \
        --output_dir data/hcvrp_real \
        --graph_size 100 --dataset_size 1280 --num_agents 7 \
        --split train --seed 42

    # Generate out-of-distribution (test cities) test set
    python scripts/generate_real_data.py \
        --data_dir ../real-routing-nco/data/dataset \
        --output_dir data/hcvrp_real \
        --graph_size 100 --dataset_size 1280 --num_agents 7 \
        --split test --seed 42

    # Convert existing RRNCO RCVRP npz file (single-vehicle → multi-agent)
    python scripts/generate_real_data.py \
        --from_npz ../real-routing-nco/data/rcvrp/rcvrp_n100_seed3333_in_distribution.npz \
        --output_dir data/hcvrp_real \
        --num_agents 7 --seed 42
"""

import argparse
import os

import numpy as np
import orjson


# ── Constants (matching RRNCO & PARCO) ──────────────────────────────────

CAPACITIES = {
    10: 20.0, 15: 25.0, 20: 30.0, 30: 33.0, 40: 37.0,
    50: 40.0, 60: 43.0, 75: 45.0, 100: 50.0, 125: 55.0,
    150: 60.0, 200: 70.0, 500: 100.0, 1000: 150.0,
}


# ── Sampling helpers (adapted from RRNCO) ──────────────────────────────

def load_cities_list(data_dir, split="train"):
    """Load train/test city split."""
    path = os.path.join(data_dir, "splited_cities_list.json")
    with open(path, "r") as f:
        cities = orjson.loads(f.read())
    return cities[split]


def load_city_data(data_dir, city):
    """Load a city's OSRM data. Returns dict with points, distance, duration."""
    path = os.path.join(data_dir, city, f"{city}_data.npz")
    data = np.load(path, allow_pickle=True)
    result = {k: data[k] for k in data.files}

    # Remove outlier points (distance > 1e5 usually means OSRM routing failure)
    if result["distance"].max() > 1e5:
        n = len(result["points"])
        outlier_mask = np.any(result["distance"] > 1e5, axis=1) | \
                       np.any(result["distance"] > 1e5, axis=0)
        keep = np.where(~outlier_mask)[0]
        result = {
            "points": result["points"][keep],
            "distance": result["distance"][keep][:, keep],
            "duration": result["duration"][keep][:, keep],
        }
    return result


def sample_from_cities(data_dir, cities_list, dataset_size, graph_size,
                        dist_type="uniform", seed=None):
    """Sample subproblems from multiple cities.

    For each city, samples `dataset_size // num_cities` instances.
    Each instance: pick `graph_size + 1` points (1 depot + graph_size customers).

    Returns:
        dict with keys: points [B, graph_size+1, 2], distance [B, graph_size+1, graph_size+1]
    """
    rng = np.random.default_rng(seed)
    num_cities = len(cities_list)
    per_city = dataset_size // num_cities
    # Handle remainder by giving extra samples to first cities
    remainder = dataset_size - per_city * num_cities

    all_points = []
    all_distance = []

    for i, city in enumerate(cities_list):
        city_data = load_city_data(data_dir, city)
        n_available = len(city_data["points"])
        n_needed = graph_size + 1  # 1 depot + graph_size customers

        if n_needed > n_available:
            print(f"Warning: {city} has only {n_available} points, "
                  f"need {n_needed}. Skipping.")
            continue

        n_samples = per_city + (1 if i < remainder else 0)

        for _ in range(n_samples):
            if dist_type == "uniform":
                idx = rng.choice(n_available, n_needed, replace=False)
            elif dist_type == "cluster":
                center_idx = rng.integers(n_available)
                center = city_data["points"][center_idx]
                dists = np.linalg.norm(city_data["points"] - center, axis=1)
                idx = np.argsort(dists)[:n_needed]
            else:
                raise ValueError(f"Unknown dist_type: {dist_type}")

            all_points.append(city_data["points"][idx])
            all_distance.append(city_data["distance"][idx][:, idx])

    return {
        "points": np.array(all_points, dtype=np.float32),      # [B, N+1, 2]
        "distance": np.array(all_distance, dtype=np.float32),   # [B, N+1, N+1]
    }


def normalize_points(points):
    """Per-instance min-max normalization of coordinates to [0, 1]."""
    p_min = points.min(axis=1, keepdims=True)
    p_max = points.max(axis=1, keepdims=True)
    denom = np.where(p_max - p_min == 0, 1, p_max - p_min)
    return (points - p_min) / denom


# ── Main conversion logic ──────────────────────────────────────────────

def prepare_hcvrp_from_sampled(sampled_data, num_agents, graph_size,
                                 min_demand=1, max_demand=10,
                                 min_capacity=20, max_capacity=41,
                                 min_speed=0.5, max_speed=1.0,
                                 seed=None):
    """Convert sampled city data to HCVRP format.

    Input format (from sample_from_cities or RRNCO npz):
        - points/locs: [B, N+1, 2] (first point = depot) or separate depot+locs
        - distance/distance_matrix: [B, N+1, N+1]

    Output NPZ format (compatible with HCVRPRealEnv.load_data → _reset):
        - depot: [B, 2]
        - locs: [B, N, 2]         (customers only)
        - demand: [B, N]           (random integer demands / capacity)
        - capacity: [B, m]         (random per-agent capacity)
        - speed: [B, m]            (random per-agent speed)
        - distance_matrix: [B, N+1, N+1]  (raw, not normalized — env does that)
        - num_agents: [B]
    """
    rng = np.random.default_rng(seed)
    B = sampled_data["points"].shape[0] if "points" in sampled_data else sampled_data["depot"].shape[0]

    # ── Coordinates ──
    if "points" in sampled_data:
        # Raw city format: first point is depot
        points = normalize_points(sampled_data["points"])
        depot = points[:, 0, :]         # [B, 2]
        locs = points[:, 1:, :]         # [B, N, 2]
    else:
        # Already split (from existing RRNCO npz)
        depot = sampled_data["depot"]   # [B, 2]
        locs = sampled_data["locs"]     # [B, N, 2]

    N = locs.shape[1]

    # ── Distance matrix ──
    if "distance" in sampled_data:
        distance_matrix = sampled_data["distance"]  # [B, N+1, N+1]
    elif "distance_matrix" in sampled_data:
        distance_matrix = sampled_data["distance_matrix"]  # [B, N+1, N+1]
    else:
        raise ValueError("No distance data found in sampled_data")

    # ── Demand ──
    if "demand" in sampled_data:
        demand = sampled_data["demand"]
    else:
        demand = rng.integers(min_demand, max_demand + 1,
                              size=(B, N)).astype(np.float32)

    # ── Multi-agent fields (these don't exist in RRNCO) ──
    capacity = rng.uniform(min_capacity, max_capacity,
                           size=(B, num_agents)).astype(np.float32)
    speed = rng.uniform(min_speed, max_speed,
                        size=(B, num_agents)).astype(np.float32)
    num_agents_arr = np.full(B, num_agents, dtype=np.int64)

    return {
        "depot": depot.astype(np.float32),
        "locs": locs.astype(np.float32),
        "demand": demand.astype(np.float32),
        "capacity": capacity.astype(np.float32),
        "speed": speed.astype(np.float32),
        "distance_matrix": distance_matrix.astype(np.float32),
        "num_agents": num_agents_arr,
    }


def convert_rrnco_npz(npz_path, num_agents, seed=None, **kwargs):
    """Convert an existing RRNCO RCVRP npz to HCVRP format.

    RRNCO format: {depot, locs, demand, capacity (scalar), distance_matrix}
    Output: adds per-agent capacity [B, m], speed [B, m], num_agents [B]
    """
    data = dict(np.load(npz_path))
    B = data["depot"].shape[0]
    graph_size = data["locs"].shape[1]

    print(f"Converting {npz_path}")
    print(f"  Instances: {B}, Graph size: {graph_size}")
    print(f"  Adding {num_agents} agents with random capacity/speed")

    return prepare_hcvrp_from_sampled(data, num_agents, graph_size,
                                       seed=seed, **kwargs)


def generate_from_cities(data_dir, split, dataset_size, graph_size,
                          num_agents, dist_type="uniform", seed=None, **kwargs):
    """Generate HCVRP dataset by sampling from real-world cities."""
    cities = load_cities_list(data_dir, split)
    print(f"Loaded {len(cities)} {split} cities")
    print(f"Sampling {dataset_size} instances, graph_size={graph_size}, "
          f"num_agents={num_agents}")

    sampled = sample_from_cities(data_dir, cities, dataset_size,
                                  graph_size, dist_type, seed)
    print(f"Sampled {sampled['points'].shape[0]} instances from cities")

    return prepare_hcvrp_from_sampled(sampled, num_agents, graph_size,
                                       seed=seed, **kwargs)


def save_npz(data, output_path):
    """Save dataset as .npz file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, **data)
    total_mb = sum(v.nbytes for v in data.values()) / 1024 / 1024
    print(f"Saved to {output_path} ({total_mb:.1f} MB)")
    for k, v in data.items():
        print(f"  {k}: {v.shape} {v.dtype}")


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate HCVRP datasets from RRNCO real-world data")

    # Source options (mutually exclusive)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data_dir", type=str,
                        help="Path to RRNCO data/dataset/ dir (with city folders)")
    source.add_argument("--from_npz", type=str,
                        help="Path to existing RRNCO RCVRP .npz file to convert")

    # Output
    parser.add_argument("--output_dir", type=str, default="data/hcvrp_real",
                        help="Output directory for generated .npz files")
    parser.add_argument("--output_name", type=str, default=None,
                        help="Output filename (auto-generated if not set)")

    # Problem parameters
    parser.add_argument("--graph_size", type=int, default=100,
                        help="Number of customer nodes (default: 100)")
    parser.add_argument("--dataset_size", type=int, default=1280,
                        help="Number of instances (default: 1280)")
    parser.add_argument("--num_agents", type=int, default=7,
                        help="Number of agents/vehicles (default: 7)")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "test"],
                        help="City split for sampling (default: train)")
    parser.add_argument("--dist_type", type=str, default="uniform",
                        choices=["uniform", "cluster"],
                        help="Location sampling distribution (default: uniform)")

    # Multi-agent parameters
    parser.add_argument("--min_capacity", type=float, default=20,
                        help="Min agent capacity (default: 20)")
    parser.add_argument("--max_capacity", type=float, default=41,
                        help="Max agent capacity (default: 41)")
    parser.add_argument("--min_speed", type=float, default=0.5,
                        help="Min agent speed (default: 0.5)")
    parser.add_argument("--max_speed", type=float, default=1.0,
                        help="Max agent speed (default: 1.0)")

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.from_npz:
        # ── Convert existing RRNCO npz ──
        data = convert_rrnco_npz(
            args.from_npz, args.num_agents, seed=args.seed,
            min_capacity=args.min_capacity, max_capacity=args.max_capacity,
            min_speed=args.min_speed, max_speed=args.max_speed,
        )
        if args.output_name is None:
            base = os.path.splitext(os.path.basename(args.from_npz))[0]
            args.output_name = f"hcvrp_{base}_m{args.num_agents}.npz"
    else:
        # ── Generate from city data ──
        data = generate_from_cities(
            args.data_dir, args.split, args.dataset_size, args.graph_size,
            args.num_agents, args.dist_type, args.seed,
            min_capacity=args.min_capacity, max_capacity=args.max_capacity,
            min_speed=args.min_speed, max_speed=args.max_speed,
        )
        if args.output_name is None:
            tag = "in_dist" if args.split == "train" else "out_dist"
            args.output_name = (
                f"hcvrp_real_n{args.graph_size}_m{args.num_agents}"
                f"_s{args.dataset_size}_{tag}_seed{args.seed}.npz"
            )

    save_npz(data, os.path.join(args.output_dir, args.output_name))


if __name__ == "__main__":
    main()
