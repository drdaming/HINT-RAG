import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelHypothesisSelector(nn.Module):

    def __init__(self, tau_g_init: float = 1.0, tau_g_min: float = 0.1):
        super().__init__()
        self.tau_g = tau_g_init
        self.tau_g_min = tau_g_min
        self.tau_g_init = tau_g_init

    def anneal_temperature(
        self,
        current_step: int,
        total_steps: int,
    ) -> None:
        progress = min(current_step / max(total_steps, 1), 1.0)
        cos_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        self.tau_g = self.tau_g_min + (self.tau_g_init - self.tau_g_min) * cos_decay

    def forward(
        self,
        P_diag: torch.Tensor,
        concept_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_probs = torch.log(P_diag + 1e-9)

        p_tilde = F.gumbel_softmax(log_probs, tau=self.tau_g, hard=False)
        p_tilde = p_tilde.to(concept_embeddings.dtype)

        e_pri = torch.matmul(p_tilde, concept_embeddings)

        with torch.no_grad():
            primary_idx = p_tilde.float().argmax(dim=-1)
            mask = F.one_hot(primary_idx, num_classes=P_diag.shape[-1]).float()

        residual = P_diag.float() - mask * P_diag.float()
        residual_sum = residual.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        r_tilde = (residual / residual_sum).to(concept_embeddings.dtype)

        e_sec = torch.matmul(r_tilde, concept_embeddings)

        return e_pri, e_sec
