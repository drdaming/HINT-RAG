import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import full_precision


@torch.no_grad()
def encode_concepts(embed_tokens, tokenizer, concepts):
    rows = []
    for text in concepts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids = torch.tensor(ids, dtype=torch.long, device=embed_tokens.weight.device)
        rows.append(embed_tokens(ids).float().mean(dim=0))
    return torch.stack(rows)


class OpenVocabularyProbe(nn.Module):
    def __init__(self, concept_embeddings, tau_p=0.1, normalize=True):
        super().__init__()
        self.tau_p = float(tau_p)
        self.normalize = bool(normalize)
        self.register_buffer("concept_embeddings", concept_embeddings.float(), persistent=False)

    @property
    def num_concepts(self):
        return self.concept_embeddings.size(0)

    def set_concepts(self, concept_embeddings):
        self.concept_embeddings = concept_embeddings.float().to(self.concept_embeddings.device)

    def forward(self, h):
        with full_precision(h):
            h = h.float()
            concepts = self.concept_embeddings
            if self.normalize:
                h = F.normalize(h, dim=-1)
                concepts = F.normalize(concepts, dim=-1)
            logits = h @ concepts.t() / self.tau_p
            prob = logits.softmax(dim=-1)
            entropy = -(prob * prob.clamp_min(1e-12).log()).sum(dim=-1)
        return prob, entropy


class SoftHypothesisRouting(nn.Module):
    def __init__(self, tau_start=1.0, tau_end=0.1, soft=True):
        super().__init__()
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.soft = bool(soft)
        self.register_buffer("tau_g", torch.tensor(self.tau_start))

    def anneal(self, step, total_steps):
        progress = min(max(step / max(total_steps, 1), 0.0), 1.0)
        value = self.tau_end + 0.5 * (self.tau_start - self.tau_end) * (1.0 + math.cos(math.pi * progress))
        self.tau_g.fill_(value)

    def forward(self, prob, concept_embeddings):
        with full_precision(prob):
            prob = prob.float()
            log_prob = prob.clamp_min(1e-12).log()
            if not self.soft:
                p_pri = F.one_hot(prob.argmax(dim=-1), prob.size(-1)).float()
            else:
                if self.training:
                    uniform = torch.rand_like(log_prob).clamp_(1e-10, 1.0 - 1e-10)
                    gumbel = -torch.log(-torch.log(uniform))
                else:
                    gumbel = torch.zeros_like(log_prob)
                p_pri = ((log_prob + gumbel) / self.tau_g).softmax(dim=-1)
            weights = (1.0 - p_pri) * prob
            p_sec = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            e_pri = p_pri @ concept_embeddings.float()
            e_sec = p_sec @ concept_embeddings.float()
        return e_pri, e_sec, p_pri, p_sec
