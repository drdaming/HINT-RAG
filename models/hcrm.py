from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class HCRM(nn.Module):

    def __init__(
        self,
        d: int = 4096,
        text_dim: int = 768,
        R: int = 128,
        tau_r_init: float = 0.07,
    ):
        super().__init__()
        self.d = d
        self.text_dim = text_dim
        self.R = R

        self.W_pri = nn.Linear(2 * d, d, bias=False)
        self.W_sec = nn.Linear(2 * d, d, bias=False)
        self.ln_pri = nn.LayerNorm(d)
        self.ln_sec = nn.LayerNorm(d)

        self.log_tau_r = nn.Parameter(torch.tensor(float(np.log(tau_r_init))))

        self.value_proj = nn.Linear(text_dim, d, bias=False)

        nn.init.xavier_uniform_(self.W_pri.weight)
        nn.init.xavier_uniform_(self.W_sec.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)

    def build_queries(
        self,
        h_state: torch.Tensor,
        e_pri: torch.Tensor,
        e_sec: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = self.W_pri.weight.dtype
        h = h_state.to(dtype)
        ep = e_pri.to(dtype)
        es = e_sec.to(dtype)

        q_pri = self.ln_pri(self.W_pri(torch.cat([h, ep], dim=-1)))
        q_sec = self.ln_sec(self.W_sec(torch.cat([h, es - ep], dim=-1)))

        return q_pri, q_sec

    def _stage2_attention(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = query.dtype
        tau_r = self.log_tau_r.exp().clamp(min=1e-4)

        q_norm = F.normalize(query.float(), dim=-1).to(dtype)
        k_norm = F.normalize(keys.float(), dim=-1).to(dtype)

        logits = torch.bmm(k_norm, q_norm.unsqueeze(-1)).squeeze(-1) / tau_r
        alpha = F.softmax(logits, dim=-1)

        v_proj = self.value_proj(values.float().to(dtype))

        E = torch.bmm(alpha.unsqueeze(1), v_proj).squeeze(1)

        return E, alpha

    def _retrieve_one(
        self,
        query: torch.Tensor,
        h_state_np: np.ndarray,
        memory_bank,
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        sample_indices, _ = memory_bank.search(h_state_np, top_k=self.R)

        keys, values = memory_bank.get_kv_by_indices(
            sample_indices, device=str(query.device)
        )

        E, alpha = self._stage2_attention(query, keys, values)

        return E, alpha, sample_indices

    def forward(
        self,
        h_state: torch.Tensor,
        e_pri: torch.Tensor,
        e_sec: torch.Tensor,
        memory_bank,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
        q_pri, q_sec = self.build_queries(h_state, e_pri, e_sec)

        h_state_np = h_state.detach().float().cpu().numpy()

        E_pri, alpha_pri, idx_pri = self._retrieve_one(q_pri, h_state_np, memory_bank)
        E_sec, alpha_sec, idx_sec = self._retrieve_one(q_sec, h_state_np, memory_bank)

        return E_pri, E_sec, alpha_pri, alpha_sec, idx_pri, idx_sec
