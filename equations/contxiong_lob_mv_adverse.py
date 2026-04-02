# equations/contxiong_lob_mv_adverse.py
#
# Combined: McKean-Vlasov + Adverse Selection.
# Full model with law-dependent execution and price signal.

import time
import numpy as np
import torch
import torch.nn as nn

from registry import register_equation
from .contxiong_lob_adverse import ContXiongLOBAdverse
from .law_encoders import create_law_encoder
from .contxiong_lob_mv import CompetitiveFactorNet


@register_equation("contxiong_lob_mv_adverse")
class ContXiongLOBMVAdverse(ContXiongLOBAdverse):
    """Full model: adverse selection + distribution-dependent mean-field.

    f_tf uses h(Phi(mu_t)) from the law embedding, NOT the old scalar proxy.
    """

    def __init__(self, eqn_config):
        super().__init__(eqn_config)

        encoder_type = getattr(eqn_config, "law_encoder_type", "deepsets")
        encoder_kwargs = {
            "state_dim": 3,  # (S, q, signal)
            "embed_dim": getattr(eqn_config, "law_embed_dim", 16),
            "n_bins": getattr(eqn_config, "n_bins", 20),
            "q_max": self.q_max,
        }
        self.law_encoder = create_law_encoder(encoder_type, **encoder_kwargs)
        self.law_embed_dim = self.law_encoder.embed_dim

        # Learned competitive factor from law embedding
        self.competitive_factor_net = CompetitiveFactorNet(
            self.law_embed_dim, dtype=torch.float64
        )
        self._current_law_embed = None
        self._prev_particle_snapshot = None
        self._w2_history = []

    def compute_law_embedding(self, particles):
        return self.law_encoder.encode(particles)

    def compute_competitive_factor(self, law_embed):
        return self.competitive_factor_net(law_embed)

    def set_current_law_embed(self, law_embed):
        self._current_law_embed = law_embed

    def f_tf(self, t, x, y, z):
        """Generator with adverse selection AND law-dependent h(Phi(mu_t))."""
        q = x[:, 1:2]
        signal = x[:, 2:3]
        z_q = z[:, 1:2]

        delta_a, delta_b = self._optimal_quotes_tf(z_q)

        # Adverse selection factors (from price signal)
        adv_a = self._adverse_factor_tf(signal, "ask")
        adv_b = self._adverse_factor_tf(signal, "bid")

        # Competitive factor from law embedding (NOT scalar proxy)
        if self._current_law_embed is not None:
            h_factor = self.compute_competitive_factor(self._current_law_embed)
        else:
            h_factor = 1.0

        # Execution with both adverse selection and MV coupling
        f_a = self._exec_prob_tf(delta_a) * adv_a * h_factor * self.lambda_a
        f_b = self._exec_prob_tf(delta_b) * adv_b * h_factor * self.lambda_b

        psi = self._penalty_tf(q)
        profit_a = f_a * delta_a
        profit_b = f_b * delta_b

        return -self.discount_rate * y - psi + profit_a + profit_b

    def compute_w2_distance(self, particles_new):
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
