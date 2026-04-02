# equations/contxiong_lob_mv.py
#
# McKean-Vlasov extension of the Cont-Xiong LOB model.
# Distribution-dependent mean-field coupling via law encoders.
#
# Key difference from contxiong_lob.py:
# - Execution probability h(mu_t) depends on a LEARNED embedding
#   of the population distribution, not just 2 scalar moments
# - Subnet inputs include the law embedding: (S_i, q_i, Phi(mu_t))
# - Fictitious play tracks Wasserstein distance between successive mu_t

import time
import numpy as np
import torch
import torch.nn as nn

from registry import register_equation
from .contxiong_lob import ContXiongLOB
from .law_encoders import create_law_encoder


@register_equation("contxiong_lob_mv")
class ContXiongLOBMV(ContXiongLOB):
    """McKean-Vlasov Cont-Xiong LOB with distribution-dependent coupling.

    Extends the base ContXiongLOB with a law encoder that maps the
    empirical population distribution to a fixed-size embedding.
    The embedding enters the subnet inputs so the control depends
    on the full population state, not just two moments.
    """

    def __init__(self, eqn_config):
        super().__init__(eqn_config)

        # Law encoder config
        encoder_type = getattr(eqn_config, "law_encoder_type", "moments")
        encoder_kwargs = {
            "state_dim": 2,  # (S, q)
            "embed_dim": getattr(eqn_config, "law_embed_dim", 16),
            "n_bins": getattr(eqn_config, "n_bins", 20),
            "q_max": self.q_max,
        }
        self.law_encoder = create_law_encoder(encoder_type, **encoder_kwargs)
        self.law_embed_dim = self.law_encoder.embed_dim

        # Wasserstein tracking for fictitious play diagnostics
        self._prev_particle_snapshot = None
        self._w2_history = []

    def compute_law_embedding(self, particles):
        """Compute the law embedding from current particle states.

        Args:
            particles: [batch, 2] tensor of (S, q) states

        Returns:
            embedding: [law_embed_dim] tensor (same for all agents)
        """
        return self.law_encoder.encode(particles)

    def compute_w2_distance(self, particles_new):
        """Compute approximate W2 distance from previous particle snapshot.

        Uses sorted-array approximation: W2 ≈ ||sort(q_new) - sort(q_old)||_2 / sqrt(n)
        """
        if self._prev_particle_snapshot is None:
            self._prev_particle_snapshot = particles_new.detach().cpu().numpy().copy()
            return 0.0

        q_new = np.sort(particles_new[:, 1].detach().cpu().numpy())
        q_old = np.sort(self._prev_particle_snapshot[:, 1])

        # Match lengths (may differ if batch sizes change)
        n = min(len(q_new), len(q_old))
        w2 = np.sqrt(np.mean((q_new[:n] - q_old[:n]) ** 2))

        self._prev_particle_snapshot = particles_new.detach().cpu().numpy().copy()
        return float(w2)

    def update_mean_field_mv(self, particles):
        """Update mean-field state from full particle snapshot.

        Args:
            particles: [batch, 2] numpy array of (S, q) states at current time
        """
        # Track W2 distance for convergence diagnostics
        particles_t = torch.tensor(particles, dtype=torch.float64)
        w2 = self.compute_w2_distance(particles_t)
        self._w2_history.append(w2)

        # Also update the legacy moment proxy for backward compatibility
        q = particles[:, 1]
        self.mean_q_estimate[:] = np.mean(q)
        self.mean_spread_estimate[:] = 2.0 / self.alpha  # will be updated by solver

    def get_w2_history(self):
        """Return the W2 distance history for diagnostics."""
        return self._w2_history.copy()
