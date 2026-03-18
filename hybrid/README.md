# Hybrid RRNCO-PARCO

Hybrid architecture combining RRNCO's real-world routing encoder (Attention-Free Transformer with Neural Adaptive Bias) with PARCO's multi-agent parallel decoder (communication layers + multi-pointer + conflict handler).

## Architecture

```
Graph Inputs (coords, distance matrix, attributes)
        │
        ▼
RRNCO Encoder (AAFM + Contextual Gating + Neural Adaptive Bias)
        │
        ▼
Node Embeddings (dual-stream → fused)
        │
        ├── Agent Embeddings (PARCO-style)
        │
        ▼
PARCO Communication Layer (agent self-attention)
        │
        ▼
Multiple Pointer Decoder (parallel per-agent)
        │
        ▼
Conflict Handler (priority-based)
        │
        ▼
Multi-agent Routes
```

## Setup

```bash
# From the workspace root (rrnco_parco/), using PARCO's existing venv:
source parco/.venv/bin/activate

# Install all three packages in editable mode:
pip install -e parco/ -e real-routing-nco/ -e hybrid/
```

## Training

```bash
cd hybrid
python train.py experiment=hybrid_hcvrp
```

## Testing

```bash
cd hybrid
python test.py --problem hcvrp --checkpoint checkpoints/hybrid/best.ckpt
```

## Project Structure

```
hybrid/
├── hybrid/                    # Python package
│   ├── models/                # Model components
│   │   ├── encoder.py         # HybridEncoder (RRNCO AttnFreeNet wrapped)
│   │   ├── decoder.py         # HybridDecoder (PARCO + distance bias)
│   │   ├── policy.py          # HybridPolicy (wires everything)
│   │   ├── rl.py              # HybridRLModule (training wrapper)
│   │   └── env_embeddings/    # Init/context embeddings
│   ├── envs/                  # Environment definitions
│   └── utils/                 # Shared utilities
├── configs/                   # Hydra configuration
├── train.py                   # Training entry point
├── test.py                    # Testing entry point
└── tests/                     # Unit tests
```
