from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class EvidenceFusion(nn.Module):

    def __init__(self, d: int = 4096, n_heads: int = 8):
        super().__init__()
        self.d = d

        self.w_g = nn.Parameter(torch.tensor(0.0))
        self.b_g = nn.Parameter(torch.tensor(0.0))

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=n_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.layer_norm = nn.LayerNorm(d)

    def forward(
        self,
        H_mid: torch.Tensor,
        E_pri: torch.Tensor,
        E_sec: torch.Tensor,
        S: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dtype = H_mid.dtype
        device = H_mid.device

        g = torch.sigmoid(self.w_g * S.float().to(device) + self.b_g)
        g = g.to(dtype)

        E_kv = torch.stack([E_pri.to(dtype), E_sec.to(dtype)], dim=1)

        H_xattn, _ = self.cross_attn(
            query=H_mid,
            key=E_kv,
            value=E_kv,
            need_weights=False,
        )

        H_xattn = self.layer_norm(H_xattn + H_mid)

        g_expand = g[:, None, None]
        H_fused = g_expand * H_xattn + (1.0 - g_expand) * H_mid

        return H_fused, g
