import torch
import torch.nn as nn


class UncertaintyGate(nn.Module):
    def __init__(self, w_init=1.0, b_init=-3.0, dynamic=True):
        super().__init__()
        self.dynamic = bool(dynamic)
        self.w_g = nn.Parameter(torch.tensor(float(w_init)))
        self.b_g = nn.Parameter(torch.tensor(float(b_init)))

    def forward(self, entropy):
        if not self.dynamic:
            return torch.ones_like(entropy.float())
        return torch.sigmoid(self.w_g * entropy.float() + self.b_g)


class EvidenceCrossAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, hidden, evidence):
        attended, _ = self.attn(hidden, evidence, evidence, need_weights=False)
        return self.norm(hidden + attended)


class DisambiguateModule(nn.Module):
    def __init__(self, d, n_heads=8, gate_w_init=1.0, gate_b_init=-3.0, dynamic_gate=True):
        super().__init__()
        self.gate = UncertaintyGate(gate_w_init, gate_b_init, dynamic_gate)
        self.cross_attention = EvidenceCrossAttention(d, n_heads)

    def forward(self, h_mid, evidence, g):
        dtype = self.cross_attention.norm.weight.dtype
        hidden = h_mid.to(dtype)
        enhanced = self.cross_attention(hidden, evidence.to(dtype))
        g = g.to(enhanced.dtype).view(-1, 1, 1)
        return (g * enhanced + (1.0 - g) * hidden).to(h_mid.dtype)
