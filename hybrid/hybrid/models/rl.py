"""Hybrid RL Module: Training wrapper for the hybrid architecture.

This module will be fully implemented in Phase 7.
It extends PARCORLModule (which extends SymNCO) with hybrid-specific
training logic.
"""

import lightning as L


class HybridRLModule(L.LightningModule):
    """Training module for the Hybrid RRNCO-PARCO model.

    Extends PARCORLModule with:
    - Adjusted hyperparameters for the deeper RRNCO encoder
    - Support for distance matrix in training data

    The training loop uses REINFORCE with:
    - Symmetric augmentation (dihedral group)
    - SymNCO losses (solution symmetricity + invariance)
    - Random agent/size subsampling (curriculum-like)
    """

    def __init__(self, env, policy=None, **kwargs):
        super().__init__()
        # TODO: Phase 7 — extend PARCORLModule
        raise NotImplementedError(
            "HybridRLModule will be implemented in Phase 7. "
            "See plan.md for architecture details."
        )
