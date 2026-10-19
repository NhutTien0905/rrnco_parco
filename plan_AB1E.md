# Plan: Hybrid Improve A+B1+E + 4GB VRAM + Serving

> Spec: `docs/superpowers/specs/2026-10-18-hybrid-improve-spec.md` (inline in REVIEW_IMPROVEMENTS.md + RESEARCH_IDEAS_CONTRIBUTION.md)
> Branch: `feat/optimize-fairness` · HW: train 4GB, serve H100/B200 fallback ORT
> Total: 5.5d sequential, 3.5d if parallel A2+E with B1

## Global Constraints

- Stack: PyTorch + Lightning + RL4CO + Hydra. Trainer `hybrid/train.py:5` has `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Precision: `hybrid/configs/trainer/default.yaml:9` is `16-mixed` (or `bf16-mixed`). Keep it.
- Model files: `hybrid/hybrid/models/decoder.py`, `hybrid/hybrid/models/env_embeddings/hcvrp_real.py`, `real-routing-nco/rrnco/models/nn/attn_freenet.py`, `hybrid/hybrid/models/rl.py`, `hybrid/hybrid/envs/hcvrp_real.py`.
- Tests: `hybrid/tests/test_phase5.py` (decoder), `hybrid/tests/test_phase4.py` (encoder), `hybrid/tests/test_phase3.py` (env).
- Paper is gitignored (`paper/` in `.gitignore:1`) — use `git add -f` for paper edits.
- Every task must keep existing tests green and not increase peak VRAM beyond stated budget.

## Task 1: Setup & Profiling + SVD Precompute + 4GB Config

Create `hybrid/scripts/precompute_svd.py` (new) that for each city `data/dataset/{city}/{city}_data.npz` loads `points` and `distance` (as in `hybrid/hybrid/envs/hcvrp_real.py:129-141`), runs `np.linalg.svd(distance)` or `torch.svd_lowrank`, keeps `k=8`, saves `U_k*S_k` and `V_k*S_k` to `data/dataset/{city}/svd_k8.npz` (~6MB total). Provide `--verify` flag that loads and checks reconstruction error. Also create `hybrid/configs/experiment/hybrid_hcvrp_4gb.yaml` with `batch_size: 2`, `num_augment: 2`, `accumulate_grad_batches: 4`, `precision: 16-mixed` (override from `hybrid/configs/experiment/hybrid_hcvrp.yaml:38`). Add a short profiling snippet in `hybrid/scripts/diagnose_workload.py` that prints `torch.cuda.max_memory_allocated()` for batch2/aug2 vs batch8/aug8. Verify by running `python hybrid/scripts/precompute_svd.py --k 8 --verify` (if data exists, otherwise dry-run) and `pytest hybrid/tests/test_phase3.py -q`.

## Task 2: A2 Decoder Hetero Speed-Conditioned Bias

Edit `hybrid/hybrid/models/decoder.py:71,116-143`. Replace `self.alpha = nn.Parameter(torch.ones(1))` with `self.alpha_net = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))` (input [capacity, speed] per agent, output [B,m] alpha). In `forward`, gather `dist_bias` as before, compute `agent_feat = torch.stack([td["capacity"], td["speed"]], dim=-1)` or `td["capacity"]`/`td["speed"]` (handle both key names), squeeze, pass through `alpha_net`, take `abs()`, then `inductive = alpha_per_agent.unsqueeze(-1) * dist_bias`, clamp exponent `[-80,80]`, `logits = log(exp(logits - inductive)+1e-6)`. Add `self.use_distance_bias` guard. Keep NaN guards. Log per-agent mean alpha in `hybrid/hybrid/models/rl.py:459-462` (change `decoder.alpha.data.mean()` to handle `alpha_net`). Ensure old checkpoints with `alpha` still load (`strict=False` or migrate). Verify `pytest hybrid/tests/test_phase5.py -q` and a quick forward test `HybridDecoder(embed_dim=64)(td, cache)` with m=3.

## Task 3: B1 SVD Low-Rank Embedding Lookup

Edit `hybrid/hybrid/models/env_embeddings/hcvrp_real.py:85-234`. Keep `CoordinateExpert`, `DistanceExpert`, `ContextualGating` for fallback, but add SVD path: add `self.svd_proj = nn.Linear(16, embed_dim)` (k=8 → 2*k=16), `self.use_svd = True` flag, and `self.svd_cache_dir = "data/dataset"` . In `forward`, if `td.get("svd_emb")` exists, use it; else if `self.use_svd` and cache file exists, load; else fallback to `DistanceExpert`. For SVD path, compute `svd_emb = self.svd_proj(cat(Uk*Sk, Vk*Sk))` and use it as `row_dist_emb/col_dist_emb`. Keep padding logic for N<sample_size. Add `hybrid/hybrid/models/env_embeddings/__init__.py` export if needed. Verify `pytest hybrid/tests/test_phase3.py::test_hcvrp_real_forward -q` (or similar) and that VRAM with SVD is lower via `torch.cuda.max_memory_allocated()`.

## Task 4: E Fairness Sweep & Logging

Already have `hybrid/hybrid/envs/hcvrp_real.py:399` reward `-(max + balance_weight*std)` and `hybrid/hybrid/models/rl.py:491` balance logs. This task adds a sweep helper: create `hybrid/scripts/sweep_balance.py` (new, <50 lines) that runs short training for `balance_weight ∈ {0,0.05,0.1,0.2}` for 10 epochs each and logs `balance/time_cv, time_max_min_ratio, visits_cv` to CSV. Also ensure `hybrid/configs/experiment/hybrid_hcvrp_4gb.yaml` exposes `env.balance_weight` as sweep param. No model code change besides ensuring `balance_weight` is read from config. Verify `python hybrid/scripts/sweep_balance.py --dry-run` prints configs and `pytest hybrid/tests/test_phase2.py -q` still passes.

## Task 5: VRAM Optimize — Checkpoint + Micro-batch

Edit `real-routing-nco/rrnco/models/nn/attn_freenet.py:517` to wrap each `Attn_Free_Layer` with `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` when `self.use_checkpoint=True` (add `use_checkpoint: bool = False` to `AttnFreeNet.__init__` and pass through `HybridEncoder`). Edit `hybrid/hybrid/models/encoder.py:87-94` to pass `use_checkpoint` from config. Edit `hybrid/hybrid/models/rl.py:296-315` to support `micro_batch_aug`: if `n_aug > 2`, split `td` into chunks and accumulate grad with `loss/accum`. Set `accumulate_grad_batches` via Lightning `trainer`. Ensure `optimizer.zero_grad(set_to_none=True)` as in AMP recipe. Verify `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python hybrid/train.py experiment=hybrid_hcvrp_4gb` does one step without OOM and `train/grad_norm` logs.

## Task 6: Serving INT8 Export & Bench

Create `hybrid/scripts/export_onnx.py` (new, ~80 lines) that loads `HybridRLModule` checkpoint, runs ModelOpt PTQ `mtq.quantize(INT8_SMOOTHQUANT_CFG, calib_loop)` with 256 calibration samples from `HCVRPRealEnv`, then `torch.onnx.export` to `hybrid_int8.onnx` with dynamic axes `B`. Create `hybrid/scripts/bench_serving.py` (new, ~60 lines) that loads ONNX via `onnxruntime` CUDA EP and measures p50/p95 latency for batch 1/32 and `torch.cuda.max_memory_allocated()`, also checks gap vs FP32 <0.5% on 100 samples. Add `deploy/triton/hybrid/config.pbtxt` template with `max_batch_size: 32, dynamic_batching, instance_group count 2`. Verify `python hybrid/scripts/export_onnx.py --help` and `python hybrid/scripts/bench_serving.py --dry-run`.

## Task 7: Paper & Classical Baselines

Create `hybrid/scripts/bench_classical.py` (new, ~100 lines) that converts `HCVRPRealEnv` batch to AnyVRP via `hcvrp_to_anyvrp` (map `locs, demand, capacity, speed → duration = distance/speed`, handle duplicate depot copies as in `hybrid/hybrid/envs/hcvrp_real.py:308`), then calls `real-routing-nco/rrnco/baselines/vrp_solvers/solve.py` with `solver ∈ {pyvrp, ortools}` and `max_runtime=60`. Log results to `hybrid/logs/baselines/*.csv`. Edit paper `paper/2026_IEEE_Trans_on_Industrial_Informatics___AIAgents/tii-articles-template.tex:80-90,633,664,733` to add contributions A+B1+E, replace `Classical HFVRP (planned)` with numbers, fix language `grep -in "superior|first framework|synergistically"`. Verify `grep -n "planned\|TODO"` is clean and `python hybrid/scripts/bench_classical.py --dry-run` prints solver calls.
