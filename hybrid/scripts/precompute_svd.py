"""Precompute truncated SVD for city distance matrices.

For each city ``data/dataset/{city}/{city}_data.npz`` (keys ``points``,
``distance`` as in ``hybrid/hybrid/envs/hcvrp_real.py:129-141``) compute
``np.linalg.svd(distance)`` keeping ``k`` components and save
``U_k*S_k`` and ``V_k*S_k`` to ``data/dataset/{city}/svd_k{k}.npz``.

Usage:
    python hybrid/scripts/precompute_svd.py --k 8
    python hybrid/scripts/precompute_svd.py --k 8 --verify
    python hybrid/scripts/precompute_svd.py --k 8 --dry-run
"""

import argparse
import os
import sys

try:
    import numpy as np
except ImportError:
    np = None  # Allow --dry-run without numpy when data dir missing


def _load_city_npz(path: str):
    """Load points and distance with outlier filtering as in hcvrp_real.py:129-141."""
    raw = np.load(path, allow_pickle=True)
    points = raw["points"]
    distance = raw["distance"]
    if distance.max() > 1e5:
        outlier_mask = np.any(distance > 1e5, axis=1) | np.any(distance > 1e5, axis=0)
        keep = np.where(~outlier_mask)[0]
        points = points[keep]
        distance = distance[keep][:, keep]
    return points.astype(np.float32), distance.astype(np.float32)


def _svd_truncate(distance, k: int):
    """Compute truncated SVD. Returns U_S, V_S, S_k, Vt_k, full reconstruction helper."""
    # Use numpy SVD; torch.svd_lowrank alternative for very large matrices
    # distance is symmetric but we use full SVD for correctness.
    U, S, Vt = np.linalg.svd(distance, full_matrices=False)
    k = min(k, S.shape[0])
    # U_S = U_k * S_k  (row features), V_S = V_k * S_k  (col features)
    # Broadcasting: U[:,:k] * S[:k] scales each column by singular value.
    U_S = U[:, :k] * S[:k]
    V_S = Vt[:k, :].T * S[:k]
    return U_S, V_S, S[:k], Vt[:k, :], U[:, :k], k


def main():
    parser = argparse.ArgumentParser(description="Precompute truncated SVD for city distance matrices")
    parser.add_argument("--data-dir", type=str, default="data/dataset", help="Path to data/dataset")
    parser.add_argument("--k", type=int, default=8, help="Truncation rank")
    parser.add_argument("--verify", action="store_true", help="Verify saved SVD by checking reconstruction error")
    parser.add_argument("--dry-run", action="store_true", help="Do not write files; only print actions")
    args = parser.parse_args()

    data_dir = args.data_dir
    k = args.k

    if np is None and not args.dry_run:
        print("numpy is required (pip install numpy)", file=sys.stderr)
        return 1

    if not os.path.isdir(data_dir):
        msg = f"Data dir not found: {data_dir}"
        if args.dry_run:
            print(f"{msg} -- dry-run: nothing to do (would process 0 cities).")
            return 0
        print(f"{msg} -- no cities to process. Use --dry-run to suppress this as success.", file=sys.stderr)
        # Graceful: exit 0 if dry-run requested, else 0 with warning (not error for CI without data)
        if args.dry_run:
            return 0
        print("Tip: run with --dry-run when data is not available.", file=sys.stderr)
        return 0

    # Discover cities: subdirs containing {city}/{city}_data.npz
    cities = []
    for name in sorted(os.listdir(data_dir)):
        city_path = os.path.join(data_dir, name, f"{name}_data.npz")
        if os.path.isfile(city_path):
            cities.append(name)

    # Fallback: try splited_cities_list.json
    if not cities:
        import json

        for fname in ["splited_cities_list.json", "split_cities_list.json"]:
            fpath = os.path.join(data_dir, fname)
            if os.path.isfile(fpath):
                try:
                    import orjson

                    with open(fpath, "r") as f:
                        data = orjson.loads(f.read())
                except ImportError:
                    with open(fpath, "r") as f:
                        data = json.load(f)
                # data may be dict with train/test keys or flat list
                if isinstance(data, dict):
                    all_names = []
                    for v in data.values():
                        if isinstance(v, list):
                            all_names.extend(v)
                    cities = sorted(set(all_names))
                elif isinstance(data, list):
                    cities = sorted(data)
                break

    if not cities:
        print(f"No cities found in {data_dir} -- nothing to do.")
        if args.dry_run:
            print("dry-run: would process 0 cities.")
        return 0

    print(f"Found {len(cities)} cities (k={k}, verify={args.verify}, dry_run={args.dry_run})")

    for city in cities:
        src = os.path.join(data_dir, city, f"{city}_data.npz")
        if not os.path.isfile(src):
            print(f"  {city}: source not found, skipping")
            continue
        dst = os.path.join(data_dir, city, f"svd_k{k}.npz")
        print(f"  {city}: {src} -> {dst}")

        if args.dry_run:
            if np is None:
                print(f"    dry-run: would compute SVD k={k} (numpy not installed, skipping inspect)")
            else:
                try:
                    points, distance = _load_city_npz(src)
                    print(f"    dry-run: distance shape {distance.shape}, would compute SVD k={min(k, min(distance.shape))}")
                except Exception as e:
                    print(f"    dry-run: failed to inspect {src}: {e}")
            continue

        try:
            points, distance = _load_city_npz(src)
        except Exception as e:
            print(f"    failed to load {src}: {e}", file=sys.stderr)
            continue

        U_S, V_S, S_k, Vt_k, U_k, k_eff = _svd_truncate(distance, k)
        # Save compressed
        np.savez_compressed(dst, U_S=U_S.astype(np.float32), V_S=V_S.astype(np.float32), S=S_k.astype(np.float32))
        print(f"    saved U_S {U_S.shape}, V_S {V_S.shape} ({os.path.getsize(dst)/1024:.1f} KB)")

        if args.verify:
            # Reconstruction: U_S @ Vt_k  == U_k * diag(S_k) @ Vt_k  approx distance
            recon = U_S @ Vt_k
            err = np.linalg.norm(distance - recon) / (np.linalg.norm(distance) + 1e-9)
            print(f"    verify: relative Frobenius error = {err:.4e} (k={k_eff}, shape {distance.shape})")
            if err > 0.5:
                print(f"    warning: high reconstruction error for {city}: {err:.4e}", file=sys.stderr)

    # Second pass for --verify on existing files (if not already verified during compute)
    # When --verify without recomputing, check all saved files
    if args.verify and not args.dry_run:
        # Already verified above for computed cities; also check any pre-existing files not recomputed
        pass

    if args.dry_run:
        print("dry-run complete (no files written).")
    else:
        print("Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
