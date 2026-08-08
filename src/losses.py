"""Training objectives: OC-Softmax (one-class, plan §6.E) and SAM optimizer
(sharpness-aware minimization, plan §6 preamble — documented domain-
generalization gains for deepfake detection)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OCSoftmax(nn.Module):
    """One-class softmax (Zhang et al., IEEE SPL 2021).

    Embeds bona-fide (real) audio inside a compact cone around a learned
    direction w; pushes fakes outside. Per-sample similarity s = w_hat . x_hat.
      loss_i = softplus(alpha * (m_real - s_i))        if y_i = 0 (real)
      loss_i = softplus(alpha * (s_i - m_fake))        if y_i = 1 (fake)
    Anomaly score at inference = m_real - s  (higher = more fake). Can be
    trained real-only (fake terms simply absent from the batch).
    """

    def __init__(self, emb_dim: int, m_real: float = 0.9, m_fake: float = 0.2,
                 alpha: float = 20.0):
        super().__init__()
        self.center = nn.Parameter(torch.randn(emb_dim))
        self.m_real, self.m_fake, self.alpha = m_real, m_fake, alpha

    def similarity(self, emb: torch.Tensor) -> torch.Tensor:
        return F.normalize(emb, dim=-1) @ F.normalize(self.center, dim=0)

    def forward(self, emb: torch.Tensor, labels: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.similarity(emb)
        margin = torch.where(labels == 1,
                             s - self.m_fake,          # push fakes below m_fake
                             self.m_real - s)          # pull reals above m_real
        loss = F.softplus(self.alpha * margin).mean()
        score = self.m_real - s                        # higher = more anomalous
        return loss, score


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization wrapper (Foret et al. 2021).

    Usage per step:
        loss = closure_forward(); loss.backward(); opt.first_step()
        closure_forward().backward(); opt.second_step()
    """

    def __init__(self, params, base_optimizer_cls, rho: float = 0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = True):
        grad_norm = torch.norm(torch.stack([
            p.grad.norm(p=2) for group in self.param_groups
            for p in group["params"] if p.grad is not None]), p=2)
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = True):
        for group in self.param_groups:
            for p in group["params"]:
                if "e_w" in self.state[p]:
                    p.sub_(self.state[p].pop("e_w"))
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):          # plain fallback (no SAM perturb)
        self.base_optimizer.step()
