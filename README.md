# Hybrid RRNCO-PARCO

Hybrid architecture combining **RRNCO**'s real-world routing encoder (Attention-Free Transformer with Neural Adaptive Bias) with **PARCO**'s multi-agent parallel decoder (communication layers + multi-pointer + conflict handler) for the **Heterogeneous Capacitated Vehicle Routing Problem (HCVRP)**.

## Architecture

```text
Graph Inputs (coords, distance matrix, demand, capacity, speed)
        │
        ▼
┌─────────────────────────────────────────────┐
│  RRNCO Encoder                              │
│  ├── CoordinateExpert + DistanceExpert      │
│  ├── ContextualGating (dual-stream fusion)  │
│  └── AttnFreeNet (NAB + feedforward × L)    │
└─────────────────────────────────────────────┘
        │
        ▼
    Node Embeddings
        │
        ├── Agent nodes: PARCO-style positional encoding
        │
        ▼
┌─────────────────────────────────────────────┐
│  PARCO Decoder                              │
│  ├── Communication Layer (agent attention)  │
│  ├── Multi-Pointer (per-agent logits)       │
│  ├── Distance-based inductive bias          │
│  └── Conflict Handler (priority-based)      │
└─────────────────────────────────────────────┘
        │
        ▼
    Multi-agent Routes
```

---

## Setup

### Prerequisites

- Python ≥ 3.10
- CUDA-capable GPU with ≥ 8 GB VRAM (recommended)
- The workspace should contain all three sibling projects:

```text
rrnco_parco/
├── parco/              # PARCO codebase
├── real-routing-nco/   # RRNCO codebase
└── hybrid/             # This project
```

### Installation

```bash
# Create and activate a virtual environment (or use PARCO's existing one)
cd parco
python -m venv .venv
source .venv/bin/activate

# Install all three packages in editable mode
pip install -e ../parco -e ../real-routing-nco -e ../hybrid

# (Optional) Install dev dependencies
pip install -e "../hybrid[dev]"
```

### Verify Installation

```bash
cd hybrid
python -m pytest tests/ -x -q
```

---

## Training

All training is managed via [Hydra](https://hydra.cc/) configs. Run from the `hybrid/` directory.

### Option 1: Euclidean Training (Synthetic Data)

Generates random Euclidean HCVRP instances on-the-fly. No data files needed.

```bash
cd hybrid

# Default config (N=100 customers, M=7 agents, 200 epochs)
python train.py experiment=hybrid_hcvrp

# Without W&B logging
python train.py experiment=hybrid_hcvrp logger=none

# With TensorBoard instead
python train.py experiment=hybrid_hcvrp logger=tensorboard

# Override parameters from command line
python train.py experiment=hybrid_hcvrp \
    model.batch_size=64 \
    trainer.max_epochs=100 \
    env.generator_params.num_loc=50 \
    env.generator_params.num_agents=5
```

### Option 2: Real-World Validation (Euclidean Train + OSRM Val/Test)

Train on synthetic Euclidean data but evaluate on real-world OSRM distance matrices.

```bash
cd hybrid

# Step 1: Generate val/test datasets from RRNCO city data
python scripts/generate_real_data.py \
    --data_dir ../real-routing-nco/data/dataset \
    --output_dir data/hcvrp_real \
    --graph_size 100 --dataset_size 1280 --num_agents 7 \
    --split train --seed 42 \
    --output_name val_real_n100_m7.npz

python scripts/generate_real_data.py \
    --data_dir ../real-routing-nco/data/dataset \
    --output_dir data/hcvrp_real \
    --graph_size 100 --dataset_size 1280 --num_agents 7 \
    --split test --seed 42 \
    --output_name test_real_n100_m7.npz

# Step 2: Train
python train.py experiment=hybrid_hcvrp_real
```

### Option 3: Full Real-World Training (OSRM Train + OSRM Val/Test)

Both training and evaluation use real-world OSRM distance matrices sampled from cities.

```bash
cd hybrid

# Step 1: Generate val/test datasets (same as Option 2)
python scripts/generate_real_data.py \
    --data_dir ../real-routing-nco/data/dataset \
    --output_dir data/hcvrp_real \
    --graph_size 100 --dataset_size 1280 --num_agents 7 \
    --split train --seed 42 \
    --output_name val_real_n100_m7.npz

python scripts/generate_real_data.py \
    --data_dir ../real-routing-nco/data/dataset \
    --output_dir data/hcvrp_real \
    --graph_size 100 --dataset_size 1280 --num_agents 7 \
    --split test --seed 42 \
    --output_name test_real_n100_m7.npz

# Step 2: Train (samples from 80 train cities on-the-fly)
python train.py experiment=hybrid_hcvrp_real_osrm
```

### Converting Existing RRNCO Data

You can also convert pre-generated RRNCO RCVRP `.npz` files to HCVRP format:

```bash
python scripts/generate_real_data.py \
    --from_npz ../real-routing-nco/data/rcvrp/rcvrp_n100_seed3333_in_distribution.npz \
    --output_dir data/hcvrp_real \
    --num_agents 7 --seed 42
```

### Resume Training

```bash
python train.py experiment=hybrid_hcvrp \
    ckpt_path=logs/train/runs/<run_dir>/checkpoints/last.ckpt
```

### Multi-GPU Training

```bash
python train.py experiment=hybrid_hcvrp trainer.devices=4
```

---

## Testing / Evaluation

Evaluate a trained checkpoint on test data.

```bash
cd hybrid

# Greedy decoding
python test.py \
    --problem hcvrp \
    --checkpoint checkpoints/hcvrp/hybrid.ckpt \
    --datasets data/hcvrp_real/test_real_n100_m7.npz \
    --decode_type greedy \
    --batch_size 128

# Sampling decoding (1280 samples)
python test.py \
    --problem hcvrp \
    --checkpoint checkpoints/hcvrp/hybrid.ckpt \
    --datasets data/hcvrp_real/test_real_n100_m7.npz \
    --decode_type sampling \
    --sample_size 1280 \
    --batch_size 1

# If --datasets is not specified, evaluates all files in data/{problem}/
python test.py --problem hcvrp --checkpoint checkpoints/hcvrp/hybrid.ckpt

# CPU evaluation
python test.py --problem hcvrp --checkpoint checkpoints/hcvrp/hybrid.ckpt --device cpu

# Compiled model (faster repeated inference)
python test.py --problem hcvrp --checkpoint checkpoints/hcvrp/hybrid.ckpt --compile
```

---

## Data Generation Script

`scripts/generate_real_data.py` bridges RRNCO's single-vehicle city data to the multi-agent HCVRP format.

```text
RRNCO city data                    Hybrid HCVRP format
{points, distance, duration}  →    {depot, locs, demand, capacity,
                                    speed, distance_matrix, num_agents}
```

### Full Options

```text
usage: generate_real_data.py [-h] (--data_dir DATA_DIR | --from_npz FROM_NPZ)
                             [--output_dir OUTPUT_DIR] [--output_name OUTPUT_NAME]
                             [--graph_size GRAPH_SIZE] [--dataset_size DATASET_SIZE]
                             [--num_agents NUM_AGENTS] [--split {train,test}]
                             [--dist_type {uniform,cluster}]
                             [--min_capacity MIN_CAPACITY] [--max_capacity MAX_CAPACITY]
                             [--min_speed MIN_SPEED] [--max_speed MAX_SPEED]
                             [--seed SEED]
```

| Argument | Default | Description |
| --- | --- | --- |
| `--data_dir` | — | Path to RRNCO `data/dataset/` (with city folders) |
| `--from_npz` | — | Path to existing RRNCO RCVRP `.npz` to convert |
| `--graph_size` | 100 | Number of customer nodes |
| `--dataset_size` | 1280 | Number of problem instances |
| `--num_agents` | 7 | Number of vehicles |
| `--split` | train | City split: `train` (80 cities) or `test` (20 cities) |
| `--dist_type` | uniform | Sampling strategy: `uniform` or `cluster` |
| `--min_capacity` | 20 | Min per-agent capacity |
| `--max_capacity` | 41 | Max per-agent capacity |
| `--min_speed` | 0.5 | Min per-agent speed |
| `--max_speed` | 1.0 | Max per-agent speed |

---

## Experiment Configs

| Config | Description |
| --- | --- |
| `experiment/hybrid_hcvrp.yaml` | Euclidean training + Euclidean val/test |
| `experiment/hybrid_hcvrp_real.yaml` | Euclidean training + OSRM val/test |
| `experiment/hybrid_hcvrp_real_osrm.yaml` | OSRM training + OSRM val/test |

### Key Hyperparameters

| Parameter | Default | Description |
| --- | --- | --- |
| `embed_dim` | 128 | Embedding dimension |
| `num_heads` | 8 | Attention heads (decoder) |
| `num_encoder_layers` | 6 | AttnFreeNet depth |
| `feedforward_hidden` | 512 | FFN hidden dimension |
| `nab_type` | gating | Neural Adaptive Bias type |
| `batch_size` | 128 | Training batch size |
| `lr` | 2e-4 | Learning rate |
| `max_epochs` | 200 | Training epochs |
| `num_augment` | 8 | SymNCO augmentations |
| `train_min_agents` | 3 | Curriculum: min agents |
| `train_max_agents` | 7 | Curriculum: max agents |
| `train_min_size` | 60 | Curriculum: min customers |
| `train_max_size` | 100 | Curriculum: max customers |

All parameters can be overridden from the command line:

```bash
python train.py experiment=hybrid_hcvrp \
    model.policy.embed_dim=256 \
    model.policy.num_encoder_layers=8 \
    model.batch_size=64
```

---

## Project Structure

```text
hybrid/
├── train.py                        # Training entry point (Hydra)
├── test.py                         # Evaluation entry point (argparse)
├── pyproject.toml                  # Package metadata & dependencies
├── configs/                        # Hydra configuration
│   ├── main.yaml                   # Root config
│   ├── env/
│   │   ├── hcvrp_real.yaml         # Euclidean generator
│   │   └── hcvrp_real_osrm.yaml    # Real-world OSRM generator
│   ├── experiment/
│   │   ├── hybrid_hcvrp.yaml       # Euclidean experiment
│   │   ├── hybrid_hcvrp_real.yaml  # Euclidean train + OSRM eval
│   │   └── hybrid_hcvrp_real_osrm.yaml  # Full OSRM experiment
│   ├── model/hybrid.yaml           # Model config
│   ├── trainer/default.yaml        # Lightning trainer config
│   ├── callbacks/                   # Callback configs
│   └── logger/                      # Logger configs (wandb, tensorboard, csv)
├── hybrid/                          # Python package
│   ├── envs/
│   │   └── hcvrp_real.py           # HCVRPRealEnv, generators
│   ├── models/
│   │   ├── encoder.py              # HybridEncoder (RRNCO AttnFreeNet)
│   │   ├── decoder.py              # HybridDecoder (PARCO + distance bias)
│   │   ├── policy.py               # HybridPolicy
│   │   ├── rl.py                   # HybridRLModule (SymNCO training)
│   │   └── env_embeddings/
│   │       └── hcvrp_real.py       # Dual-stream init embedding
│   └── utils/
├── scripts/
│   └── generate_real_data.py       # RRNCO → HCVRP data conversion
├── data/                            # Generated datasets
│   └── hcvrp_real/                  # Real-world OSRM .npz files
├── tests/                           # Unit tests (pytest)
└── logs/                            # Training outputs
```
