# Review: Điểm cần cải thiện để vượt CHECKPOINT_REMAINING (2026-10-18)

> Scope: toàn bộ `hybrid/`, `parco/`, `real-routing-nco/` + `paper/tii-articles-template.tex:80-90,633,664`.
> Checkpoint gốc: `CHECKPOINT_REMAINING.md` — 3 blockers R2.1 / R3.4 / R3.5.
> Branch: `feat/optimize-fairness` · Paper gitignored (`paper/` in `.gitignore:1`).

---

## 0. Tóm tắt blocker

| # | Reviewer | Hiện trạng | Mức độ | Effort | Chọn |
|---|----------|-----------|--------|--------|------|
| 1 | **R2.1 Innovation hạn chế** (AE + R2.1) | Đã hạ claim "integration" (`paper:80-90`), chưa có module/loss mới | 🔴 Reject risk cao | Option A ~1 ngày / Option B ~3-5 ngày | Cần quyết trước khi code |
| 2 | **R3.4 Baselines thiếu** (AE + R3.4) | `Classical HFVRP (planned):668` chưa có số | 🔴 Blocker nếu resubmit | 2-3 ngày (chạy solver 60s × 3 seeds) | Bắt buộc |
| 3 | **R3.5 Language** | Đã fix 3 đoạn, còn `superior`/`first`/`synergistically` rải rác | 🟡 Nhẹ | 1 giờ | Làm ngay |

---

## 1. R2.1 — Innovation hạn chế

### 1.1 Code hiện tại làm gì

- **Encoder fusion** `hybrid/hybrid/models/encoder.py:97-141`:
  - `concat`: `Linear(2H→H)` trên `cat([row,col])` — học 1 ma trận `256×128`.
  - `gate`: `gate * row + (1-gate)*col` với `gate = sigmoid(Linear(2H→H))` — per-dim interpolation, không condition trên agent.
  - `add`: `row+col`.
  - Cả 3 đều **không thấy capacity/speed**, không thấy heterogeneity. Reviewer đọc là "trivial concatenation".
- **Init embedding** `hybrid/hybrid/models/env_embeddings/hcvrp_real.py:18-234`:
  - Depot: `init_embed_depot` + `PositionalEncoder` + `init_embed_agents([x,y,cap/40,speed])` → `Linear(2H→H)` (`hybrid/hybrid/models/env_embeddings/hcvrp_real.py:67-77`).
  - Customer: `CoordinateExpert` + `DistanceExpert([B,N,N]→[B,N,H])` + `ContextualGating` + `demand` → `Linear(2H→H)` (`hybrid/hybrid/models/env_embeddings/hcvrp_real.py:83-105`).
  - Gating chỉ fuse `coord vs distance`, không fuse theo agent. Depot embedding copy y nguyên cho cả `row` và `col` (`hybrid/hybrid/models/env_embeddings/hcvrp_real.py:223-228`) — mất tính asymmetric ở depot.
- **NAB** `real-routing-nco/rrnco/models/nn/attn_freenet.py:360-489`:
  - `AttnFree_Block` nhận `coords, cost_mat, duration_mat` và nhân `adapt_bias * alpha` (`real-routing-nco/rrnco/models/nn/attn_freenet.py:427-429`). `alpha` là scalar chung, không per-agent, không condition trên speed.
  - `HybridEncoder` khởi tạo `AttnFreeNet(use_duration_matrix=False)` (`hybrid/hybrid/models/encoder.py:93`) — bỏ luôn duration path dù HCVRP có speed heterogeneity (có thể map `distance/speed` → duration).
- **Decoder bias** `hybrid/hybrid/models/decoder.py:70-143`:
  - `alpha = Parameter(ones(1))` duy nhất (`hybrid/hybrid/models/decoder.py:71`), áp dụng `logits - |alpha|*dist[cur→all]` cho mọi agent như nhau (`hybrid/hybrid/models/decoder.py:141-143`). Heterogeneous speed không ảnh hưởng bias.
- **Paper claim** `paper:80-90`:
  - Contributions liệt kê "Hybrid Architecture / Dual-Stream Fusion / Real-World Benchmark / Extensive Evaluation" — 2/4 là engineering, 1 là dataset wrapper, reviewer coi là "integration of existing RRNCO+PARCO".

### 1.2 Vì sao sẽ bị đánh

- AE đã note "integration of existing" — nếu không có module/loss mới, TII rất dễ desk-reject ở R2.1.
- Fusion hiện tại không có lemma/theorem mới, ablation `concat vs add vs gate` (`paper:733-756`) chỉ so 3 cách concat trivial, không chứng minh novelty.
- Heterogeneity (capacity/speed) là core của HCVRP nhưng không xuất hiện trong NAB/fusion/bias — mất cơ hội tạo novelty đúng problem.

### 1.3 Option A — Reframe nhanh (không thêm model mới, ~1 ngày)

Mục tiêu: biến contribution thành "systematic integration + benchmark + empirical fusion study" thay vì "new architecture".

**Code:**
- Giữ nguyên `encoder.py:fusion_type` 3 options, nhưng chạy đủ ablation và báo cáo significance (paired t-test, 3 seeds).
- Thêm 1 ablation nhỏ có tính novelty tối thiểu: `learnable gate vs concat` — đã có sẵn `gate` (`encoder.py:99-101`), chỉ cần chạy và fill số vào Table ablation (`paper:739`). Nếu p<0.05 thì claim "gate fusion learnable".

**Paper:**
- `paper:80-90` — đổi bullet 1 từ "Hybrid Architecture: We integrate..." → "Systematic Integration: We present a systematic study unifying RRNCO's real-world encoder with PARCO's parallel decoder, with ablations isolating each design choice."
- Hạ tông bullet 2: "Dual-Stream Fusion" → "Fusion Empirical Study: We compare fusion strategies (concat/gate/add) and show gate improves makespan by X% over trivial concat."
- Thêm limitation rõ ràng ở Conclusion: "Our fusion is a lightweight Linear(2H→H); heterogeneity-aware fusion remains future work."

**Rủi ro:** vẫn có thể bị reviewer bảo "no algorithmic novelty" nhưng đỡ hơn hiện tại. Phù hợp nếu deadline gấp.

### 1.4 Option B — Thêm algorithmic mới thực sự (mạnh, ~3-5 ngày, khuyến nghị)

Chọn **1 trong 2** (không cần cả 2):

#### B1. Heterogeneity-aware NAB (encoder-side, 2-3 ngày)

Ý tưởng: NAB hiện tại chỉ thấy `coords, cost_mat` — cho nó thấy thêm speed/capacity để bias attention theo agent heterogeneity.

```python
# hybrid/hybrid/models/nn/hetero_nab.py (mới, ~40 dòng)
class HeteroNAB(nn.Module):
    # input: coords [B,N,2], cost_mat [B,N,N], speed [B,m] -> broadcast to [B,N]
    # output: adapt_bias [B,N,N] = f(coords, cost) + g(capacity, speed)
    def forward(self, coords, cost_mat, duration_mat=None, agent_speed=None, agent_cap=None):
        base_bias = self.dist_angle_fusion(coords, cost_mat, duration_mat)  # reuse DistAngleFusion
        # heterogeneity bias: speed conditioning
        # broadcast per-agent speed to per-node, thenouter product → [B,N,N]
        speed_bias = self.speed_mlp(agent_speed)  # [B,m] -> [B,N]
        hetero_bias = speed_bias.unsqueeze(-1) * speed_bias.unsqueeze(-2)  # or simple additive
        return base_bias + hetero_bias * self.beta
```

Wiring:
- `hybrid/hybrid/models/encoder.py:87-94` — truyền `agent_speed/capacity` từ `td["speed"]/td["capacity"]` vào `AttnFreeNet.forward(...)`.
- `real-routing-nco/rrnco/models/nn/attn_freenet.py:417-432` — thêm nhánh `if agent_speed is not None`.
- Ablation: `w/o hetero` vs `w/ hetero` — thêm 1 dòng vào Table ablation.

**Ưu:** đúng pain point HCVRP, reviewer thấy novelty gắn với problem. **Nhược:** đụng vào `AttnFreeNet`, cần test kỹ NaN.

#### B2. Speed-conditioned distance bias (decoder-side, 1-2 ngày, đơn giản hơn)

Hiện tại `HybridDecoder.alpha` là scalar (`hybrid/hybrid/models/decoder.py:71`). Đổi thành per-agent conditioned:

```python
# hybrid/hybrid/models/decoder.py — thay alpha scalar
self.alpha_net = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1))
# forward:
agent_feat = torch.cat([td["capacity"], td["speed"]], dim=-1)  # [B,m,2]
alpha_per_agent = self.alpha_net(agent_feat).squeeze(-1)  # [B,m]
inductive_bias = alpha_per_agent.unsqueeze(-1) * dist_bias  # [B,m,N] broadcast
```

So sánh: `scalar alpha` vs `speed-conditioned alpha` — thêm 1 table nhỏ hoặc 1 dòng ablation.

**Ưu:** chạm ít code, dễ chạy, vẫn là "novel component". **Nhược:** novelty nhẹ hơn B1.

> **Khuyến nghị:** làm **B2 trước** (1 ngày) để có số, nếu còn thời gian làm thêm B1. Nếu chọn B2, báo cáo luôn trong paper: "We introduce speed-conditioned inductive bias (Sec 3.5) and show +Y% over scalar baseline."

### 1.5 Việc cần làm cho R2.1 (checklist)

- [ ] Quyết Option A hay B (hỏi advisor trước khi code).
- [ ] Nếu B2: sửa `hybrid/hybrid/models/decoder.py:47-72,116-143`, thêm `alpha_net`, update `hybrid/configs/model/hybrid.yaml` với flag `hetero_bias: true`.
- [ ] Nếu B1: tạo `hybrid/hybrid/models/nn/hetero_nab.py`, sửa `hybrid/hybrid/models/encoder.py:87-94` và `real-routing-nco/rrnco/models/nn/attn_freenet.py:360-417`.
- [ ] Chạy ablation 3 seeds, fill Table `paper:733-756`, thêm p-value.
- [ ] Sửa contributions `paper:80-90` và `paper:398-412` (fusion rank paragraph đã đúng — giữ nguyên).
- [ ] Thêm short proof/remark cho B1/B2 complexity (O(N²) không đổi, chỉ thêm O(N·m)).

---

## 2. R3.4 — Baselines thiếu (BLOCKER)

### 2.1 Hiện trạng

- Paper: `paper:633` ghi `Classical HFVRP (planned): ALNS/Tabu... (omitted; future work)` và `paper:664` `Table I` chỉ có learned baselines + OR-Tools proxy, không có số thực.
- Code: có 2 nguồn solver sẵn nhưng chưa nối vào HCVRP:
  - `parco/parco/baselines/ortools/solver.py:22-80` — cho OMDC-PDP (HCVRP-like nhưng khác) với `pywrapcp`, không dùng được trực tiếp cho HCVRP.
  - `real-routing-nco/rrnco/baselines/vrp_solvers/solve.py:28-78` (`pyvrp/ortools/lkh`) + `pyvrp.py:15-80` + `ortools.py:15-120` — cho AnyVRP/MTVRP với `instance2data` qua PyVRP, cần converter `mtvrp2anyvrp` (`real-routing-nco/rrnco/baselines/vrp_solvers/utils.py`).
- Chưa có script nào chạy classical baseline trên `HCVRPRealEnv` với protocol `N=100 M=7 60s`.

### 2.2 Cần làm (2-3 ngày)

#### Bước 1 — Converter HCVRP → AnyVRP/PyVRP

Tạo `hybrid/baselines/hcvrp_to_anyvrp.py` (~80 dòng):

```python
def hcvrp_to_anyvrp(td: TensorDict) -> TensorDict:
    # td sau env.reset(): locs [B,m+N,2], demand [B,m,m+N], capacity/speed [B,m], distance_matrix [B,m+N,m+N]
    # AnyVRP expects: locs [B,N,2], demand_linehaul, vehicle_capacity, distance_matrix, duration_matrix = distance/speed
    # Map: duration_matrix = distance_matrix / speed_per_agent  (heterogeneous!)
    # Depot indices: 0..m-1 là depot copies → gộp thành 1 depot duy nhất cho classical solver, hoặc giữ m depot nếu solver support multi-depot
```

Lưu ý: classical solver không hiểu heterogeneous speed theo cách PARCO; phải encode speed vào `duration_matrix` (`duration = distance / speed`). Nếu giữ `m` depot copies, distance giữa depot copies = 0 (hiện tại `hybrid/hybrid/envs/hcvrp_real.py:308-331` đã làm vậy) — cần handle duplicate depot khi convert.

Tham khảo `real-routing-nco/rrnco/baselines/vrp_solvers/utils.py:mtvrp2anyvrp` và `real-routing-nco/rrnco/baselines/vrp_solvers/pyvrp.py:instance2data`.

#### Bước 2 — Benchmark script

Tạo `hybrid/scripts/bench_classical.py`:

```python
# For each seed in [0,1,2], for each solver in ["pyvrp","ortools","lkh"]:
#   load 100 instances N=100 M=7 từ HCVRPRealEnv (fixed seed)
#   for each instance: solve(max_runtime=60) → reward = -makespan
#   report mean ± std, gap vs hybrid
```

Dựa trên `hybrid/test.py:1-100` và `real-routing-nco/rrnco/baselines/vrp_solvers/solve.py:42-78` (multiprocess `Pool`).

#### Bước 3 — Cập nhật paper

- `paper:664` Table I: thêm 2-3 cột `ALNS (60s) | HGS-CVRP (60s) | OR-Tools (60s)` cùng protocol.
- `paper:633` — xóa `(planned)` và `(omitted; future work)`, thay bằng số thực + gap.
- Nếu solver chậm hơn hybrid ở inference (expected) thì highlight "10× speedup at comparable quality".

#### Bước 4 — Config và test

- `hybrid/configs/experiment/hybrid_hcvrp.yaml` đã có `num_loc:100 num_agents:7` — dùng luôn cho baseline generation.
- Thêm `tests/test_baselines.py` kiểm tra converter round-trip.

### 2.3 Fallback nếu không chạy được

Nếu thiếu PyVRP/LKH binary hoặc OSRM data:
- Đổi `paper:633` thành limitation rõ: "Classical baselines omitted due to heterogeneous-speed duration modeling; we report published HGS-CVRP numbers on Euclidean HCVRP as proxy (cite) and note gap."
- Nhưng reviewer TII thường không chấp nhận proxy — rủi ro reject cao hơn. Nên cố chạy ít nhất OR-Tools (đã có `ortools>=9.11` trong `hybrid/pyproject.toml`).

### 2.4 Checklist R3.4

- [ ] Viết `hybrid/baselines/hcvrp_to_anyvrp.py` + `hybrid/scripts/bench_classical.py`.
- [ ] Chạy 3 solvers × 60s × 100 instances × 3 seeds, lưu `hybrid/logs/baselines/*.npz`.
- [ ] Cập nhật Table I `paper:664` và xóa `planned` ở `paper:633`.
- [ ] Thêm `solver` vào `hybrid/pyproject.toml:optional-dependencies:solver` nếu thiếu LKH.

---

## 3. R3.5 — Language còn sót

### 3.1 Hits còn lại

```bash
grep -in "exceptional\|outstanding\|remarkable\|superior\|state-of-the-art\|significantly improves\|synergistically\|first framework\|extensive" \
  paper/tii-articles-template.tex
# Kết quả hiện tại:
# :74 superior computational efficiency at inference time
# :80 first framework that synergistically combines
# :89 Extensive Evaluation: We demonstrate superior OOD generalization
```

### 3.2 Fix pattern (1 giờ)

| Dòng | Gốc | Sửa thành |
|------|-----|-----------|
| `paper:74` | `superior computational efficiency` | `competitive computational efficiency` |
| `paper:80` | `We present the first framework that synergistically combines` | `We present a framework that combines` |
| `paper:80` | `synergistically` | xóa |
| `paper:89` | `Extensive Evaluation: We demonstrate superior OOD` | `Evaluation: We evaluate OOD` |
| `paper:89` | `superior OOD generalization` | `OOD generalization` |
| Toàn file | `significantly improves` (nếu còn) | `improves` hoặc thêm `p<0.05` |

```bash
grep -n "planned\|TODO\|FIXME" paper/tii-articles-template.tex
# :633 Classical HFVRP (planned) — xóa planned sau khi có số (R3.4)
```

---

## 4. Thêm phát hiện ngoài checkpoint (code quality / correctness / perf / security)

### 4.1 Correctness

- **Depot distance 0** `hybrid/hybrid/envs/hcvrp_real.py:308-331`: khoảng cách giữa các depot copies = 0 (repeat row/col). Đúng về mặt semantic (cùng 1 depot vật lý), nhưng classical solver sẽ thấy m node trùng nhau — cần de-duplicate ở converter (mục 2.2).
- **Resample slicing** `hybrid/hybrid/models/rl.py:67-90` vs `parco/parco/models/utils.py:13-40`: `resample_batch_hybrid` slice `distance_matrix[..., :keep, :keep]` lấy `depot + first num_locs` — khớp với `locs[..., :num_locs]` nhưng không random; nếu muốn đúng curriculum thì nên sample ngẫu nhiên, hiện tại bias về thứ tự sinh. Không blocker, nhưng note cho reproducibility.
- **Fusion rank** `paper:398-412` đã fix đúng (không claim rank preservation) — giữ nguyên.
- **Alpha abs** `hybrid/hybrid/models/decoder.py:141`: `self.alpha.abs()` tránh negative, nhưng nếu alpha học về 0 thì mất bias — log `distance_bias/alpha` đã có (`hybrid/hybrid/models/rl.py:459-462`) để monitor.
- **NaN guards** `hybrid/hybrid/models/policy.py:191-206`, `hybrid/hybrid/models/decoder.py:92-114`, `hybrid/hybrid/models/rl.py:285-290`: clamp NaN/Inf khắp nơi — tốt cho train ổn định nhưng có thể che bug (ví dụ `distance_matrix` chứa NaN từ OSRM outlier). Nên giữ nhưng thêm alert khi clamp rate >1%.

### 4.2 Performance

- **Encoder O(N²)** `real-routing-nco/rrnco/models/nn/attn_freenet.py:292-327` (`AFTFull`: `exp(adapt_bias) @ exp(K) * V`): với `N=m+N=107, H=64, L=6` thì `[B,107,107]` bias — ổn trên GPU, nhưng `batch=8` + `num_augment=8` → effective batch 64 → ~64×107×107 float ~3MB per layer, chấp nhận được. Không cần optimize.
- **Precision 16-mixed** `hybrid/configs/trainer/default.yaml:9` tốt, nhưng `distance_matrix` normalize per-instance (`hybrid/hybrid/envs/hcvrp_real.py:334-340`) có `+1e-6` denom — với 16-bit có thể underflow nếu `max-min < 1e-3`. Đã cast về float32 (`hybrid/hybrid/envs/hcvrp_real.py:340`) nên ok.
- **val/test n_aug=1** `hybrid/hybrid/models/rl.py:296-300` đã fix OOM khi `AttnFreeNet` gặp `[B*N_aug, N, N]` — giữ nguyên.

### 4.3 Security / Robustness

- **`np.load(..., allow_pickle=True)`** `hybrid/hybrid/envs/hcvrp_real.py:129` (`real-world sampler`): nếu `city_data_dir` bị inject file npz độc, pickle có thể RCE. Giảm risk bằng `allow_pickle=False` nếu file chỉ chứa `points/distance` (dùng `np.load(..., allow_pickle=False)` hoặc validate magic). Low risk vì data là local.
- **`torch.load` safe globals** `hybrid/train.py:25`, `hybrid/inference.py:18-20`: đã allowlist `DictConfig/ListConfig` — đúng cho PyTorch 2.6+. Fallback `weights_only=False` ở `inference.py` nên wrap trong try/except có log.

### 4.4 Config / Test gaps

- **`balance_weight=0.1`** `hybrid/configs/env/hcvrp_real.yaml:4` vs `hybrid/hybrid/envs/hcvrp_real.py:258-259` default `0.0` — config override đúng, không bug, nhưng `HybridRLModule` log `balance/*` (`hybrid/hybrid/models/rl.py:491-514`) chỉ khi `env.balance_weight>0` — nhất quán.
- **Tests** `hybrid/tests/test_phase*.py` cover encoder/decoder/policy/env riêng lẻ, thiếu integration test `HCVRPRealEnv.generate → HybridPolicy.forward → HybridRLModule.shared_step` end-to-end (đã có `test_phase6.py` nhưng chưa test OSRM path `HCVRPRealWorldGenerator`).
- **Model config** `hybrid/configs/model/hybrid.yaml:1-7` chỉ khai báo `HybridRLModule`, không expose `policy` hyperparams — thực tế override qua `experiment/hybrid_hcvrp.yaml:17-37`, ok.

---

## 5. Roadmap ưu tiên

### P0 — Bắt buộc trước resubmit

1. **R3.4 baselines** (2-3 ngày): converter + bench script + Table I.
2. **R2.1 Option B2** (1 ngày): speed-conditioned bias + 1 ablation row.
3. **R3.5 language** (1 giờ): `paper:74,80,89,633`.

### P1 — Nên làm (tăng acceptance)

4. **R2.1 contributions rewrite** `paper:80-90`: reframe theo Option chọn.
5. **Fusion ablation significance**: chạy 3 seeds, thêm std và p-value vào `paper:739`.
6. **Classical baseline protocol note**: ghi rõ `N=100 M=7 60s` cùng hardware.

### P2 — Nếu còn thời gian

7. **Hetero NAB B1** (2-3 ngày).
8. **OSRM integration test** + `city_data_dir` handling.
9. **Harden `np.load` allow_pickle** → `allow_pickle=False`.

---

## 6. Verification checklist (chạy trước khi commit)

```bash
# Language
grep -in "exceptional\|outstanding\|remarkable\|superior\|state-of-the-art\|significantly improves" \
  paper/2026_IEEE_Trans_on_Industrial_Informatics___AIAgents/tii-articles-template.tex

# Baselines (sau khi có bench script)
python hybrid/scripts/bench_classical.py --solver pyvrp --max_runtime 60 --num_loc 100 --num_agents 7
python hybrid/scripts/bench_classical.py --solver ortools --max_runtime 60 --num_loc 100 --num_agents 7

# Fusion / hetero bias ablations
python -m pytest hybrid/tests/test_phase4.py -k fusion -q
python hybrid/test.py --checkpoint logs/train/runs/latest/checkpoints/last.ckpt --decode_type greedy

# Paper
grep -n "planned\|TODO\|FIXME" paper/2026_IEEE_Trans_on_Industrial_Informatics___AIAgents/tii-articles-template.tex

# Commit paper (paper/ gitignored)
git add -f paper/2026_IEEE_Trans_on_Industrial_Informatics___AIAgents/tii-articles-template.tex REVIEW_IMPROVEMENTS.md
```

---

## 7. Quyết định cần trước khi code

- [ ] Chọn **Option A** hay **B2** hay **B1** cho R2.1 — trả lời trong file này hoặc nhắn.
- [ ] Có chạy baseline R3.4 không? Nếu có, cần chuẩn bị `city_data` và PyVRP/LKH binary.
- [ ] Có muốn giữ `balance_weight=0.1` ở `hybrid/configs/env/hcvrp_real.yaml:4` hay về `0.0` để reward đúng makespan thuần?

---

## Appendix — File map

| File | Dòng | Liên quan |
|------|------|-----------|
| `hybrid/hybrid/models/encoder.py:97-141` | fusion concat/gate/add | R2.1 |
| `hybrid/hybrid/models/env_embeddings/hcvrp_real.py:67-234` | init embedding | R2.1 |
| `hybrid/hybrid/models/decoder.py:70-143` | distance bias scalar alpha | R2.1 B2 |
| `real-routing-nco/rrnco/models/nn/attn_freenet.py:360-489` | NAB, AttnFree_Block | R2.1 B1 |
| `hybrid/hybrid/envs/hcvrp_real.py:308-331,73-224` | distance expand, OSRM sampler | R3.4, correctness |
| `hybrid/hybrid/models/rl.py:33-92,459-514` | resample_hybrid, diagnostic metrics | R2.1, P0 |
| `hybrid/configs/env/hcvrp_real.yaml:4` | balance_weight | P2 |
| `hybrid/configs/experiment/hybrid_hcvrp.yaml:11-58` | train hyperparams | R2.1 ablation |
| `paper/tii-articles-template.tex:80-90,398-412,633,664,733-756` | contributions, fusion rank, baselines, ablation | R2.1 R3.4 R3.5 |
| `parco/parco/baselines/ortools/solver.py` | ORTools baseline ref | R3.4 |
| `real-routing-nco/rrnco/baselines/vrp_solvers/solve.py,pyvrp.py,ortools.py` | PyVRP/ORTools/LKH wrappers | R3.4 |

