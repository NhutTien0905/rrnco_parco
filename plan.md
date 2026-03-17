# Implementation Plan: Hybrid RRNCO-PARCO Architecture

> **Goal**: Replace PARCO's Transformer encoder with RRNCO's Attention-Free encoder (AAFM + Neural Adaptive Bias), while keeping PARCO's multi-agent communication layers, multi-pointer decoder, and conflict handler intact.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Interface Analysis & Key Mismatches](#2-interface-analysis--key-mismatches)
3. [Implementation Steps](#3-implementation-steps)
   - [Phase 1: Project Setup](#phase-1-project-setup)
   - [Phase 2: Hybrid Environment](#phase-2-hybrid-environment)
   - [Phase 3: Hybrid Init Embedding](#phase-3-hybrid-init-embedding)
   - [Phase 4: Hybrid Encoder](#phase-4-hybrid-encoder)
   - [Phase 5: Hybrid Decoder Adapter](#phase-5-hybrid-decoder-adapter)
   - [Phase 6: Hybrid Policy](#phase-6-hybrid-policy)
   - [Phase 7: Training Module](#phase-7-training-module)
   - [Phase 8: Configs & Training](#phase-8-configs--training)
4. [File Structure](#4-file-structure)
5. [Detailed Class Designs](#5-detailed-class-designs)
6. [Risk & Fallback](#6-risk--fallback)

---

## 1. Architecture Overview

### Current Architectures

**PARCO** (multi-agent, standard Transformer):
```
td → HCVRPInitEmbedding → [B, m+N, H]
  → TransformerBlock × 3 (self-attention) → [B, m+N, H]
  → HCVRPContextEmbedding (communication layers) → query [B, m, H]
  → PointerAttention → logits [B, m, N]
  → HighProbHandler → actions [B, m]
```

**RRNCO** (single-agent, Attention-Free Transformer):
```
td → RVRPInitEmbedding → (row_emb, col_emb, distance) each [B, N, H]
  → AttnFreeNet × 6 (AFT + Neural Adaptive Bias) → (row_emb, col_emb) each [B, N, H]
  → VRPContext → query [B, 1, H]
  → PointerAttention → logits [B, N]
```

### Target Hybrid Architecture

```
td (with distance_matrix) → HybridInitEmbedding → (row_emb, col_emb, distance)
  → AttnFreeNet × L (RRNCO encoder with AAFM + NAB) → (row_emb, col_emb)
  → Fusion Layer → node_embeddings [B, m+N, H]
  → HCVRPContextEmbedding (PARCO communication layers) → query [B, m, H]
  → PointerAttention (PARCO multi-pointer) → logits [B, m, N]
  → HighProbHandler (PARCO conflict handler) → actions [B, m]
```

---

## 2. Interface Analysis & Key Mismatches

### Encoder Output Format

| Component          | PARCO Encoder          | RRNCO Encoder                 |
|--------------------|------------------------|-------------------------------|
| **Return type**    | `(h, init_h)` — single tensor `[B, N, H]` | `(row_emb, col_emb)` — tuple of 2 tensors `[B, N, H]` |
| **Node count N**   | `m + num_customers` (depots + customers) | `1 + num_customers` (1 depot + customers) |
| **Distance matrix**| ❌ Not used             | ✅ `[B, N, N]` — core input   |
| **Duration matrix**| ❌ Not used             | ✅ Optional `[B, N, N]`       |

### Decoder Input Format

| Component              | PARCO Decoder                       | RRNCO Decoder                        |
|------------------------|-------------------------------------|--------------------------------------|
| **Expects**            | Single tensor `[B, N, H]`          | Tuple `(row_emb, col_emb)`          |
| **node_embeddings**    | Same tensor used for context gather | `row_emb` for context                |
| **glimpse K/V/logit_K**| Same tensor projected              | `col_emb` projected                  |
| **Query shape**        | `[B, m, H]` (multi-agent)          | `[B, 1, H]` (single-agent)          |
| **Mask shape**         | `[B, m, N]` (per-agent masks)      | `[B, N]` (single mask)              |
| **Inductive bias**     | None                                | `logits - α·distance[cur→j]`        |

### Critical Mismatches to Resolve

1. **Dual-stream → Single-stream**: RRNCO outputs `(row_emb, col_emb)`, PARCO decoder expects single tensor → Need **fusion layer**
2. **Node count**: RRNCO has 1 depot, PARCO has `m` depot copies → Need to handle **depot replication** in init embedding
3. **Distance matrix**: RRNCO encoder requires it, PARCO environment doesn't provide it → Need **hybrid environment** with distance matrix generation
4. **`phase` argument**: RRNCO encoder forward takes `phase` parameter → Need to pass through from policy
5. **`init_h` for SymNCO**: PARCO training uses `init_h` for symmetricity loss → Need to return `init_h` from hybrid encoder

---

## 3. Implementation Steps

### Phase 1: Project Setup

**Goal**: Create the hybrid project structure inside the workspace.

#### Tasks:
1. Create a new directory `hybrid/` at the workspace root
2. Set up `pyproject.toml` with dependencies from both PARCO and RRNCO
3. Create the package structure `hybrid/models/`, `hybrid/envs/`, `hybrid/configs/`
4. Ensure both `parco` and `rrnco` are importable (add as local dependencies or copy necessary modules)

#### Key Decision:
- **Approach A** (recommended): Build `hybrid/` as a standalone package that imports from both `parco` and `rrnco` packages
- **Approach B**: Fork PARCO codebase and modify in-place

We go with **Approach A** for clean separation.

---

### Phase 2: Hybrid Environment

**Goal**: Create a multi-agent environment that provides distance matrices (combining HCVRP structure with RRNCO-style distance data).

#### File: `hybrid/envs/hcvrp_real.py`

#### What to do:
1. **Extend PARCO's `HCVRPEnv`** to include `distance_matrix` in the TensorDict
2. The generator should produce/load real-world distance matrices (like RRNCO's `Real_World_Sampler`)
3. Keep all multi-agent logic from HCVRP (heterogeneous capacities, speeds, per-agent masks)
4. Add min-max normalization of distance matrix per instance (as RRNCO does)

#### TensorDict changes (additions from RRNCO):
```python
# Generator output additions:
"distance_matrix": [B, m+N, m+N]   # real-world or Euclidean distance matrix
"min_distance": [B]                  # for denormalization
"max_distance": [B]                  # for denormalization

# Optional:
"duration_matrix": [B, m+N, m+N]    # travel time matrix (if asymmetric)
```

#### Implementation details:
- For initial experiments, compute Euclidean distance matrix from `locs` so we can test without OSRM data:
  ```python
  distance_matrix = torch.cdist(locs, locs, p=2)  # [B, m+N, m+N]
  ```
- Later, integrate RRNCO's `Real_World_Sampler` for real OSRM data
- Normalize: `d_norm = (d - d_min) / (d_max - d_min + 1e-6)` per instance
- **Reward**: Keep PARCO's min-max objective: $-\max_i(\text{tour\_time}_i)$

#### Fallback:
- If distance matrix is not provided (backward compatibility), compute Euclidean from `locs`

---

### Phase 3: Hybrid Init Embedding

**Goal**: Create an init embedding that combines RRNCO's distance-aware node encoding with PARCO's agent encoding.

#### File: `hybrid/models/env_embeddings/hcvrp_real.py`

#### Architecture:
```
Depot/Agent nodes (m):
  depot_locs → CoordinateExpert → depot_coord_emb [B, m, H]
  + PositionalEncoding (from PARCO)
  + agent_features [capacity, speed] → Linear → agent_emb [B, m, H]
  → Linear([depot_coord_emb, agent_emb]) → depot_agent_emb [B, m, H]

Customer nodes (N):
  customer_locs → CoordinateExpert (from RRNCO) → coord_emb [B, N, H]
  distance_matrix → DistanceExpert (from RRNCO) → (row_dist_emb, col_dist_emb) [B, N, H]
  → ContextualGating (from RRNCO) → fused_row_emb, fused_col_emb [B, N, H]
  + demand → Linear → demand_emb [B, N, H]
  → Linear(2H → H) → customer_row_emb, customer_col_emb [B, N, H]

Output:
  row_emb = cat([depot_agent_emb, customer_row_emb]) → [B, m+N, H]
  col_emb = cat([depot_agent_emb, customer_col_emb]) → [B, m+N, H]
  distance = full (m+N) × (m+N) normalized distance matrix
```

#### Key decisions:
1. **Depot/agent nodes** use PARCO-style embedding (with positional encoding + agent features) since RRNCO has no multi-agent concept
2. **Customer nodes** use RRNCO-style embedding (CoordinateExpert + DistanceExpert + ContextualGating) for rich structural features
3. **Distance matrix** for RRNCO encoder is the full `(m+N) × (m+N)` matrix (depots included)
4. Both `row_emb` and `col_emb` are returned (dual-stream for AttnFreeNet)

#### Components to reuse:
- From RRNCO: `CoordinateExpert`, `DistanceExpert`, `ContextualGating`
- From PARCO: `PositionalEncoder`, agent feature projection, `init_embed_depot`, `init_embed_agents`

---

### Phase 4: Hybrid Encoder

**Goal**: Wrap RRNCO's `AttnFreeNet` encoder in PARCO-compatible interface.

#### File: `hybrid/models/encoder.py`

#### Class: `HybridEncoder(nn.Module)`

```python
class HybridEncoder(nn.Module):
    def __init__(self,
        embed_dim=128,
        num_heads=8,
        num_layers=6,           # RRNCO default
        normalization="instance",
        feedforward_hidden=512,
        nab_type="gating",      # Neural Adaptive Bias type
        env_name="hcvrp_real",
        init_embedding=None,
        init_embedding_kwargs={},
        use_pos_token=False,     # PARCO option
        **kwargs,
    ):
        self.init_embedding = HybridInitEmbedding(...)  # Phase 3
        self.net = AttnFreeNet(...)  # from RRNCO
        self.fusion = nn.Linear(2 * embed_dim, embed_dim)  # dual→single fusion
```

#### Forward method:
```python
def forward(self, td, mask=None):
    # 1. Init embedding (hybrid)
    row_emb, col_emb, distance = self.init_embedding(td)
    # row_emb: [B, m+N, H], col_emb: [B, m+N, H], distance: [B, m+N, m+N]

    init_h = self.fusion(cat([row_emb, col_emb], dim=-1))  # [B, m+N, H]

    # 2. Get locs and optional duration_matrix for NAB
    locs = td["locs"]  # [B, m+N, 2]
    duration_matrix = td.get("duration_matrix", None)

    # 3. AttnFreeNet encoder (RRNCO)
    row_emb, col_emb = self.net(row_emb, col_emb, distance, locs, duration_matrix)
    # row_emb: [B, m+N, H], col_emb: [B, m+N, H]

    # 4. Fuse dual-stream → single-stream for PARCO decoder
    h = self.fusion(cat([row_emb, col_emb], dim=-1))  # [B, m+N, H]

    # 5. Optional POS token (PARCO feature)
    if self.use_pos_token:
        h = cat([h, self.pos_token.expand(B, -1, -1)], dim=1)
        init_h = cat([init_h, self.pos_token.expand(B, -1, -1)], dim=1)

    return h, init_h  # Match PARCO encoder output interface
```

#### Why fusion here (not in decoder):
- Keeps PARCO decoder **completely unchanged** — zero modification needed
- The fusion layer learns to combine row/col information into a single representation
- `init_h` is also fused for SymNCO compatibility

#### Alternative (advanced): Keep dual-stream and modify decoder
- Store both `row_emb` and `col_emb` in cache (like RRNCO decoder does)
- Use `row_emb` for node_embeddings (context gather), `col_emb` for K/V projection
- This preserves more information but requires modifying PARCO decoder
- **Save this for Phase 2 of research** — test fusion first

---

### Phase 5: Hybrid Decoder Adapter

**Goal**: Optionally add RRNCO's distance-aware inductive bias to PARCO's multi-pointer decoder.

#### File: `hybrid/models/decoder.py`

#### Class: `HybridDecoder(PARCODecoder)`

```python
class HybridDecoder(PARCODecoder):
    def __init__(self, embed_dim=128, num_heads=8, env_name="hcvrp_real",
                 use_distance_bias=True, **kwargs):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads,
                         env_name=env_name, **kwargs)
        if use_distance_bias:
            self.alpha = nn.Parameter(torch.ones(1))  # learnable distance weight

    def forward(self, td, cached, num_starts=0, do_unbatchify=False):
        logits, mask = super().forward(td, cached, num_starts, do_unbatchify)
        # logits: [B, m, N]

        if hasattr(self, 'alpha') and "distance_matrix" in td:
            # Gather distance from each agent's current node to all nodes
            cur_nodes = td["current_node"]  # [B, m]
            dist_matrix = td["distance_matrix"]  # [B, m+N, m+N]

            # For each agent, get distance vector from current_node
            # dist_bias[b, a, :] = dist_matrix[b, cur_nodes[b,a], :]
            dist_bias = gather_by_index(dist_matrix, cur_nodes, dim=1)  # [B, m, m+N]

            # Apply inductive bias: closer nodes get higher logits
            logits = logits - self.alpha * dist_bias

        return logits, mask
```

#### Why this matters:
- RRNCO's decoder uses `logits - α·distance` as an inductive bias
- This encourages the model to prefer nearby nodes, which is especially valuable with real-world distances
- The `alpha` is learnable, so the model can control the strength

#### This is optional:
- Start without it (use vanilla PARCODecoder) to establish a baseline
- Add it as an ablation study

---

### Phase 6: Hybrid Policy

**Goal**: Wire the hybrid encoder + PARCO decoder + communication layers into a single policy.

#### File: `hybrid/models/policy.py`

#### Class: `HybridPolicy(PARCOPolicy)`

```python
class HybridPolicy(PARCOPolicy):
    def __init__(self,
        encoder=None,
        decoder=None,
        embed_dim=128,
        num_encoder_layers=6,    # RRNCO uses 6 layers
        num_heads=8,
        env_name="hcvrp_real",
        # RRNCO encoder params
        nab_type="gating",
        feedforward_hidden=512,
        normalization_encoder="instance",
        # PARCO decoder params (inherited)
        agent_handler="highprob",
        use_distance_bias=True,
        **kwargs,
    ):
        # Build hybrid encoder if not provided
        if encoder is None:
            encoder = HybridEncoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_encoder_layers,
                normalization=normalization_encoder,
                feedforward_hidden=feedforward_hidden,
                nab_type=nab_type,
                env_name=env_name,
            )
        # Build hybrid decoder if not provided
        if decoder is None:
            decoder = HybridDecoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                env_name=env_name,
                use_distance_bias=use_distance_bias,
            )
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            env_name=env_name,
            agent_handler=agent_handler,
            **kwargs,
        )
```

#### Key: The `forward()` method is inherited from `PARCOPolicy` — no changes needed because:
1. `HybridEncoder.forward(td)` returns `(h, init_h)` — same interface as `PARCOEncoder`
2. `HybridDecoder` extends `PARCODecoder` — same interface
3. Communication layers live inside the decoder's context embedding — unchanged
4. Agent handler is configured via parent class — unchanged

---

### Phase 7: Training Module

**Goal**: Create the training wrapper with appropriate hyperparameters.

#### File: `hybrid/models/rl.py`

#### Class: `HybridRLModule(PARCORLModule)`

```python
class HybridRLModule(PARCORLModule):
    def __init__(self, env, policy, **kwargs):
        super().__init__(env=env, policy=policy, **kwargs)
```

#### Hyperparameter decisions:

| Parameter           | PARCO Default | RRNCO Default | **Hybrid Choice** | Rationale |
|---------------------|---------------|---------------|---------------------|-----------|
| `embed_dim`         | 128           | 128           | **128**             | Same in both |
| `num_encoder_layers` | 3            | 6             | **6**               | RRNCO encoder needs more layers for AFT |
| `num_heads`         | 8             | 8             | **8**               | Same in both |
| `normalization`     | rms           | instance      | **instance**        | RRNCO encoder uses instance norm |
| `feedforward_hidden`| ~341 (SwiGLU) | 512           | **512**             | RRNCO encoder default |
| `nab_type`          | N/A           | gating        | **gating**          | Best NAB variant from RRNCO paper |
| `lr`                | 1e-4          | 4e-4          | **2e-4**            | Compromise; tune later |
| `max_epochs`        | 100           | 200           | **200**             | RRNCO encoder may need more training |
| `num_augment`       | 4→8           | 8             | **8**               | Both use dihedral augmentation |
| `agent_handler`     | highprob      | N/A           | **highprob**        | PARCO's best handler |
| `num_comm_layers`   | 1             | N/A           | **1**               | Start simple |

---

### Phase 8: Configs & Training

**Goal**: Set up Hydra configs and training scripts.

#### Config files to create:

1. **`hybrid/configs/model/hybrid.yaml`**
```yaml
_target_: hybrid.models.policy.HybridPolicy
embed_dim: 128
num_encoder_layers: 6
num_heads: 8
nab_type: "gating"
feedforward_hidden: 512
normalization_encoder: "instance"
agent_handler: "highprob"
use_distance_bias: true
```

2. **`hybrid/configs/env/hcvrp_real.yaml`**
```yaml
_target_: hybrid.envs.hcvrp_real.HCVRPRealEnv
num_loc: 100
num_agents: 10
use_real_distances: false     # true for OSRM data
distance_type: "euclidean"    # or "osrm"
```

3. **`hybrid/configs/experiment/hybrid_hcvrp.yaml`** — main experiment config

#### Training command:
```bash
python train.py experiment=hybrid_hcvrp
```

---

## 4. File Structure

```
hybrid/
├── __init__.py
├── train.py                          # Training entry point
├── test.py                           # Testing entry point
├── pyproject.toml                    # Dependencies
│
├── configs/
│   ├── main.yaml
│   ├── model/
│   │   └── hybrid.yaml               # Hybrid model config
│   ├── env/
│   │   └── hcvrp_real.yaml            # Hybrid environment config
│   ├── experiment/
│   │   └── hybrid_hcvrp.yaml          # Full experiment config
│   ├── callbacks/                     # (reuse from PARCO)
│   ├── trainer/                       # (reuse from PARCO)
│   └── logger/                        # (reuse from PARCO)
│
├── envs/
│   ├── __init__.py
│   └── hcvrp_real.py                  # HCVRP + distance matrix env
│
├── models/
│   ├── __init__.py
│   ├── encoder.py                     # HybridEncoder (RRNCO AttnFreeNet wrapped)
│   ├── decoder.py                     # HybridDecoder (PARCO + distance bias)
│   ├── policy.py                      # HybridPolicy (wires everything)
│   ├── rl.py                          # HybridRLModule (training wrapper)
│   │
│   └── env_embeddings/
│       ├── __init__.py                # Registry for hybrid embeddings
│       └── hcvrp_real.py              # HybridInitEmbedding + context
│
├── scripts/
│   └── generate_data.py               # Data generation with distance matrices
│
└── tests/
    ├── __init__.py
    ├── test_encoder.py
    ├── test_decoder.py
    ├── test_policy.py
    └── test_env.py
```

---

## 5. Detailed Class Designs

### 5.1 `HybridInitEmbedding`

**Inputs** (from TensorDict):
- `td["locs"]`: `[B, m+N, 2]` — all locations (m depots + N customers)
- `td["distance_matrix"]`: `[B, m+N, m+N]` — normalized distance matrix
- `td["demand"]`: `[B, m, m+N]` — demand per agent view
- `td["agents_capacity"]`: `[B, m]`
- `td["agents_speed"]`: `[B, m]`

**Outputs**:
- `row_emb`: `[B, m+N, H]` — row embeddings for AttnFreeNet
- `col_emb`: `[B, m+N, H]` — col embeddings for AttnFreeNet
- `distance`: `[B, m+N, m+N]` — distance matrix (pass-through)

**Internal architecture**:
```
Depot/Agent (first m nodes):
  locs[:, :m] → Linear(2→H) → depot_emb
  depot_emb + PositionalEncoding → depot_pos_emb
  [capacity, speed] → Linear(2→H) → agent_feat_emb
  cat([depot_pos_emb, agent_feat_emb]) → Linear(2H→H) → depot_agent_emb  [B, m, H]
  # Same embedding used for both row and col streams

Customer (last N nodes):
  locs[:, m:] → CoordinateExpert → coord_emb [B, N, H]
  distance_matrix[m:, m:] → DistanceExpert → (row_dist, col_dist) [B, N, H]
  ContextualGating(coord, row_dist) → fused_row [B, N, H]
  ContextualGating(coord, col_dist) → fused_col [B, N, H]
  demand → Linear(1→H) → demand_emb [B, N, H]
  Linear(cat([fused_row, demand_emb])) → customer_row [B, N, H]
  Linear(cat([fused_col, demand_emb])) → customer_col [B, N, H]

Output:
  row_emb = cat([depot_agent_emb, customer_row])  [B, m+N, H]
  col_emb = cat([depot_agent_emb, customer_col])  [B, m+N, H]
```

### 5.2 `HybridEncoder`

**Purpose**: Run RRNCO's AttnFreeNet on the dual-stream embeddings, then fuse for PARCO compatibility.

**Key method**:
```python
def forward(self, td, mask=None) -> Tuple[Tensor, Tensor]:
    row_emb, col_emb, distance = self.init_embedding(td)
    init_h = self.fuse(row_emb, col_emb)  # for SymNCO loss

    locs = td["locs"]
    duration = td.get("duration_matrix", None)
    row_emb, col_emb = self.net(row_emb, col_emb, distance, locs, duration)

    h = self.fuse(row_emb, col_emb)
    return h, init_h  # [B, m+N, H], [B, m+N, H]
```

**Fusion options** (try in order):
1. **Concat + Linear**: `Linear(cat([row, col], -1))` — simplest, recommended start
2. **Gated fusion**: `g = σ(Linear(cat([row, col])))`, `h = g*row + (1-g)*col`
3. **Additive**: `h = row + col` — simplest but may lose information

### 5.3 `HybridDecoder`

**Extends** `PARCODecoder` with one addition:
- Optional distance-based inductive bias from RRNCO decoder

**Everything else inherited**:
- Communication layers (via `HCVRPContextEmbedding`)
- Multi-pointer mechanism
- POS token support

### 5.4 `HybridPolicy`

**Extends** `PARCOPolicy` — primarily a configuration wrapper.

**The forward pass** is 100% inherited from `PARCOPolicy`:
1. `encoder(td)` → `(h, init_h)`
2. `decoder.pre_decoder_hook(td, env, h)` → cache
3. Autoregressive loop: `decoder(td, cache)` → logits → strategy → handler → action
4. `env.step(td)`

### 5.5 `HCVRPRealEnv`

**Extends** `HCVRPEnv` with:
- Distance matrix generation/loading in generator
- Distance matrix in TensorDict
- Min/max distance tracking for denormalization

---

## 6. Risk & Fallback

### Risk 1: RRNCO encoder scales poorly with `m+N` nodes
- **Why**: AttnFreeNet's NAB computes `[B, m+N, m+N]` bias matrix — quadratic in total nodes
- **Mitigation**: Keep `m+N ≤ 110` (100 customers + 10 agents) — same scale as RRNCO experiments
- **Fallback**: Use RRNCO's probabilistic distance sampling (sample 25 columns) to reduce cost

### Risk 2: Dual-stream fusion loses information
- **Why**: Collapsing `(row_emb, col_emb)` into single `h` may lose asymmetric information
- **Mitigation**: Try gated fusion; if insufficient, modify PARCO decoder to accept dual-stream
- **Monitoring**: Compare pure PARCO vs hybrid on symmetric Euclidean instances first

### Risk 3: Training instability
- **Why**: RRNCO encoder (6 AFT layers) + PARCO decoder (communication + pointer) is deeper than either alone
- **Mitigation**: 
  - Use gradient clipping (1.0)
  - Start with smaller encoder (3 layers) and scale up
  - Use warmup learning rate schedule
  - Initialize fusion layer as identity-like (zero init bias)

### Risk 4: DistanceExpert's probabilistic sampling with depot nodes
- **Why**: RRNCO's `DistanceExpert` samples 25 nearest columns — may not work well when first `m` columns are all depot copies
- **Mitigation**: Only apply `DistanceExpert` to customer submatrix `distance_matrix[m:, m:]`, handle depot columns separately
- **Fallback**: Use full distance matrix without sampling for depot rows/columns

### Risk 5: Incompatible rl4co versions
- **Why**: PARCO and RRNCO may depend on different rl4co versions
- **Mitigation**: Pin to a compatible version; test imports early in Phase 1
- **Fallback**: Copy necessary base classes rather than importing

---

## Implementation Priority

| Priority | Phase | Estimated Effort | Dependency |
|----------|-------|-------------------|------------|
| 🔴 P0    | Phase 1: Project Setup | 1 day | None |
| 🔴 P0    | Phase 2: Hybrid Env | 2 days | Phase 1 |
| 🔴 P0    | Phase 3: Init Embedding | 3 days | Phase 1 |
| 🔴 P0    | Phase 4: Encoder | 2 days | Phase 3 |
| 🟡 P1    | Phase 5: Decoder (distance bias) | 1 day | Phase 4 |
| 🔴 P0    | Phase 6: Policy | 1 day | Phase 4, 5 |
| 🔴 P0    | Phase 7: Training Module | 1 day | Phase 6 |
| 🟡 P1    | Phase 8: Configs & Training | 2 days | Phase 7 |
| 🟢 P2    | Ablation: dual-stream decoder | 3 days | Phase 8 |
| 🟢 P2    | Integration: OSRM real data | 3 days | Phase 8 |

**Total estimated effort: ~2-3 weeks**

---

## Validation Checkpoints

1. **After Phase 2**: Environment generates valid TensorDicts with distance matrices; existing PARCO agent handler works
2. **After Phase 4**: `HybridEncoder(td)` produces correct shapes `[B, m+N, H]`; gradient flows
3. **After Phase 6**: `HybridPolicy(td)` produces actions `[B, m]` and valid log-likelihoods
4. **After Phase 7**: Training loop runs for 1 epoch without errors; loss decreases
5. **After Phase 8**: Full training converges; compare vs pure PARCO baseline on HCVRP
