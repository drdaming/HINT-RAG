import math

import torch
import torch.nn as nn

from ..utils import full_precision


class ContrastiveQuery(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w_pri = nn.Linear(2 * d, d, bias=False)
        self.w_sec = nn.Linear(2 * d, d, bias=False)
        self.norm_pri = nn.LayerNorm(d)
        self.norm_sec = nn.LayerNorm(d)
        nn.init.xavier_uniform_(self.w_pri.weight)
        nn.init.xavier_uniform_(self.w_sec.weight)
        nn.init.constant_(self.norm_pri.weight, d ** -0.5)
        nn.init.constant_(self.norm_sec.weight, d ** -0.5)

    def primary(self, h, e_pri):
        return self.norm_pri(self.w_pri(torch.cat([h.float(), e_pri.float()], dim=-1)))

    def secondary(self, h, e_pri, e_sec):
        return self.norm_sec(self.w_sec(torch.cat([h.float(), (e_sec - e_pri).float()], dim=-1)))


class DifferentiableRetrieval(nn.Module):
    def __init__(self, d, value_dim, top_r=128, tau_r_init=0.07):
        super().__init__()
        self.top_r = int(top_r)
        self.value_proj = nn.Linear(value_dim, d, bias=False)
        self.log_tau_r = nn.Parameter(torch.tensor(math.log(tau_r_init)))
        nn.init.xavier_uniform_(self.value_proj.weight)

    @property
    def tau_r(self):
        return self.log_tau_r.exp()

    def forward(self, query, bank, exclude=None):
        indices = bank.search(query.detach(), self.top_r, exclude)
        keys = bank.keys[indices].to(query.device)
        values = self.value_proj(bank.values[indices].to(query.device, self.value_proj.weight.dtype))
        with full_precision(query):
            logits = torch.einsum("bd,brd->br", query.float(), keys.float()) / self.tau_r
            alpha = logits.softmax(dim=-1)
            evidence = torch.einsum("br,brd->bd", alpha, values.float())
        return evidence, alpha, indices
