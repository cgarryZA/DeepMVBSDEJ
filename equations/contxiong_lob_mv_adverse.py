# equations/contxiong_lob_mv_adverse.py
#
# Combined: McKean-Vlasov + Adverse Selection.
# This is the full model where mean-field coupling actually matters:
# - Execution probability depends on price signal (adverse selection)
# - Execution probability depends on population distribution (mean-field)
# - The population's aggregate quoting affects the price signal intensity
#
# State: (S, q, signal) — 3D
# Coupling: law encoder sees all agents' (q, signal) states

import time
import numpy as np
import torch

from registry import register_equation
from .contxiong_lob_adverse import ContXiongLOBAdverse
from .law_encoders import create_law_encoder


@register_equation("contxiong_lob_mv_adverse")
class ContXiongLOBMVAdverse(ContXiongLOBAdverse):
    """Full model: adverse selection + distribution-dependent mean-field.

    The competitive factor h(mu_t) now depends on the population's
    joint distribution of (q, signal), not just scalar moments.
    With adverse selection, agents who are all on the same side of
    the market (correlated inventories) face worse execution.
    """

    def __init__(self, eqn_config):
        super().__init__(eqn_config)

        encoder_type = getattr(eqn_config, "law_encoder_type", "deepsets")
        encoder_kwargs = {
            "state_dim": 3,  # (S, q, signal) — full state
            "embed_dim": getattr(eqn_config, "law_embed_dim", 16),
            "n_bins": getattr(eqn_config, "n_bins", 20),
            "q_max": self.q_max,
        }
        self.law_encoder = create_law_encoder(encoder_type, **encoder_kwargs)
        self.law_embed_dim = self.law_encoder.embed_dim

        self._prev_particle_snapshot = None
        self._w2_history = []

    def compute_law_embedding(self, particles):
        """particles: [batch, 3] tensor of (S, q, signal)."""
        return self.law_encoder.encode(particles)

    def compute_w2_distance(self, particles_new):
        """W2 on the inventory dimension."""
        if self._prev_particle_snapshot is None:
            self._prev_particle_snapshot = particles_new.detach().cpu().numpy().copy()
            return 0.0
        q_new = np.sort(particles_new[:, 1].detach().cpu().numpy())
        q_old = np.sort(self._prev_particle_snapshot[:, 1])
        n = min(len(q_new), len(q_old))
        w2 = np.sqrt(np.mean((q_new[:n] - q_old[:n]) ** 2))
        self._prev_particle_snapshot = particles_new.detach().cpu().numpy().copy()
        return float(w2)

    def update_mean_field_mv(self, particles):
        particles_t = torch.tensor(particles, dtype=torch.float64)
        w2 = self.compute_w2_distance(particles_t)
        self._w2_history.append(w2)
