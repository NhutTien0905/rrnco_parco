"""Benchmark classical HCVRP solvers via AnyVRP conversion.

Converts HCVRPRealEnv batch to AnyVRP via hcvrp_to_anyvrp:
  - map locs, demand, capacity, speed -> duration = distance/speed
  - handle duplicate depot copies (hybrid/hybrid/envs/hcvrp_real.py:308):
    depot cp copies have 0 distance, de-duplicate to single depot for solver.

Calls real-routing-nco/rrnco/baselines/vrp_solvers/solve.py with
solver in {pyvrp, ortools} and max_runtime=60.
Logs to hybrid/logs/baselines/*.csv, supports --dry-run.
"""

import argparse
import csv
import os
import sys


def hcvrp_to_anyvrp(td):
    """Convert HCVRPRealEnv batch TensorDict to AnyVRP format.

    Args:
        td: TensorDict from HCVRPRealEnv._reset() with keys
            depot [B,2], locs [B,N,2], demand [B,N],
            capacity/speed [B,M] or agents_capacity/speed,
            distance_matrix [B,M+N,M+N] (with dup depot copies).

    Returns:
        TensorDict in AnyVRP format with keys expected by solve.py:
        locs, distance_matrix, duration_matrix, demand_linehaul, etc.
    """
    import torch
    from tensordict import TensorDict

    B = td.batch_size[0]
    # locs + depot -> AnyVRP locs [B, N+1, 2]
    depot = td["depot"]  # [B,2]
    locs = td["locs"]  # [B,N,2]
    N = locs.shape[1]
    all_locs = torch.cat([depot.unsqueeze(1), locs], dim=1)  # [B,N+1,2]

    # capacity/speed may be "capacity"/"speed" or "agents_capacity"/"vehicle_capacity"
    cap = td.get("capacity", td.get("agents_capacity", td.get("vehicle_capacity", None)))
    spd = td.get("speed", td.get("agents_speed", td.get("vehicle_speed", None)))
    if cap is None or spd is None:
        # fallback: try generator keys
        cap = td.get("capacity", torch.ones(B, 7) * 30)
        spd = td.get("speed", torch.ones(B, 7))
    # normalize shape [B,M]
    if cap.dim() == 1:
        cap = cap.unsqueeze(0).expand(B, -1)
    if spd.dim() == 1:
        spd = spd.unsqueeze(0).expand(B, -1)
    m = cap.shape[1]

    # distance_matrix: de-duplicate depot copies [B,M+N,M+N] -> [B,N+1,N+1]
    dm = td.get("distance_matrix", None)
    if dm is None:
        dm = torch.cdist(all_locs, all_locs, p=2)
    if dm.shape[1] == m + N:  # expanded with dup depots
        # indices: depot 0 + customers m..m+N-1  (depot copies 0..m-1 have 0 distance, see hcvrp_real.py:308)
        idx = torch.tensor([0] + list(range(m, m + N)), device=dm.device)
        dm_single = dm[:, idx][:, :, idx]  # [B,N+1,N+1]
    else:
        dm_single = dm  # already [B,N+1,N+1]

    # duration = distance / speed  (hetero speeds -> mean speed per instance)
    speed_mean = spd.float().mean(dim=1, keepdim=True).clamp(min=1e-6)  # [B,1]
    # expand to [B,N+1,N+1]: divide each distance by mean speed
    duration_mat = dm_single / speed_mean.view(B, 1, 1)

    # demand: [B,N] -> AnyVRP demand_linehaul [B,N+1] with 0 for depot
    demand = td.get("demand", torch.zeros(B, N))
    demand_linehaul = torch.cat([torch.zeros(B, 1, device=demand.device), demand], dim=1)
    demand_backhaul = torch.zeros_like(demand_linehaul)
    durations = torch.zeros(B, dm_single.shape[1], device=dm_single.device)
    # vehicle capacity: use mean (hetero -> approx) or first agent
    vehicle_capacity = cap.float().mean(dim=1)  # [B]
    # For AnyVRP per-instance scalar, use first batch mean; solver handles per-instance via TensorDict batch
    # Keep as Tensor [B] -> instance2data scales per instance

    out = TensorDict(
        {
            "locs": all_locs,
            "distance_matrix": dm_single,
            "duration_matrix": duration_mat,
            "demand_linehaul": demand_linehaul,
            "demand_backhaul": demand_backhaul,
            "durations": durations,
            "vehicle_capacity": vehicle_capacity,
            "distance_limit": torch.full((B,), 1e6, device=dm_single.device),
            "time_windows": torch.zeros(B, dm_single.shape[1], 2, device=dm_single.device),
            "num_depots": torch.ones(B, dtype=torch.long, device=dm_single.device),
            "open_route": torch.zeros(B, dtype=torch.bool, device=dm_single.device),
            "backhaul_class": torch.zeros(B, dtype=torch.long, device=dm_single.device),
        },
        batch_size=[B],
    )
    # time_windows already 0, pyvrp will treat as no limit
    return out


def _dry_run(args):
    print("[dry-run] bench_classical.py")
    print(f"  solvers: {args.solver if isinstance(args.solver, list) else [args.solver]}")
    print(f"  max_runtime: {args.max_runtime}")
    print(f"  batch_size: {args.batch_size} num_loc: {args.num_loc} num_agents: {args.num_agents}")
    print(f"  output: hybrid/logs/baselines/*.csv")
    for s in (args.solver if isinstance(args.solver, list) else [args.solver]):
        print(f"  call: rrnco/baselines/vrp_solvers/solve.py solver={s} max_runtime={args.max_runtime} data_type=anyvrp")
        print(f"    -> hybrid/logs/baselines/{s}_m{args.max_runtime}s.csv")
    # show conversion
    print("  convert: hcvrp_to_anyvrp(locs,demand,capacity,speed) duration=distance/speed de-dup depot cp (0 distance) -> [B,N+1,N+1]")
    print("dry-run done")


def main():
    p = argparse.ArgumentParser(description="Benchmark classical solvers on HCVRPRealEnv")
    p.add_argument("--solver", nargs="+", default=["pyvrp", "ortools"], choices=["pyvrp", "ortools", "lkh"])
    p.add_argument("--max-runtime", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-loc", type=int, default=100)
    p.add_argument("--num-agents", type=int, default=7)
    p.add_argument("--output-dir", type=str, default="hybrid/logs/baselines")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.dry_run:
        _dry_run(args)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    # Build HCVRPReal batch
    try:
        from hybrid.envs import HCVRPRealEnv, HCVRPRealGenerator
    except Exception as e:
        print(f"Failed to import HCVRPRealEnv: {e}", file=sys.stderr)
        sys.exit(1)

    gen = HCVRPRealGenerator(num_loc=args.num_loc, num_agents=args.num_agents)
    env = HCVRPRealEnv(generator=gen)
    td = env.generator._generate(batch_size=[args.batch_size])
    # mimic _reset expansion already via env.reset? Use td directly with HCVRPRealEnv._reset logic
    td = env._reset(td)

    anyvrp = hcvrp_to_anyvrp(td)

    # Import solver
    sys.path.insert(0, "real-routing-nco")
    from rrnco.baselines.vrp_solvers.solve import solve

    for solver in args.solver:
        print(f"Solving with {solver} max_runtime={args.max_runtime} ...")
        try:
            actions, costs = solve(anyvrp, max_runtime=args.max_runtime, solver=solver, data_type="anyvrp")
            print(f"  {solver}: costs mean={costs.float().mean().item():.2f}")
        except Exception as e:
            print(f"  {solver} failed: {e}")
            costs = None
            actions = None
        # log CSV
        out_path = os.path.join(args.output_dir, f"{solver}_m{args.max_runtime}s.csv")
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["solver", "max_runtime", "batch", "cost", "num_loc", "num_agents"])
            if costs is not None:
                for i, c in enumerate(costs.tolist()):
                    w.writerow([solver, args.max_runtime, i, c, args.num_loc, args.num_agents])
            else:
                w.writerow([solver, args.max_runtime, 0, "failed", args.num_loc, args.num_agents])
        print(f"  logged -> {out_path}")


if __name__ == "__main__":
    main()
