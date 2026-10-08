import torch
import torch.nn as nn
import torch.nn.functional as F


def generation_loss(logits, target_ids):
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), target_ids.reshape(-1), ignore_index=-100)


def contrastive_retrieval_loss(q_pri, q_sec, positive_keys, negative_keys, tau_c):
    keys = torch.cat([positive_keys, negative_keys], dim=0).float().to(q_pri.device)
    size = q_pri.size(0)
    targets = torch.arange(size, device=q_pri.device)
    loss = F.cross_entropy(q_pri.float() @ keys.t() / tau_c, targets)
    if q_sec is not None:
        loss = loss + F.cross_entropy(q_sec.float() @ keys.t() / tau_c, targets + size)
    return loss


def calibrated_entropy_loss(p_post, s_post, s_pre, labels, gamma, beta, smoothing):
    hinge = F.relu(s_post.float() - s_pre.detach().float() + gamma).mean()
    y = labels.float().to(p_post.device)
    valid = y.sum(dim=-1) > 0
    if not bool(valid.any()):
        return hinge
    num = y.size(-1)
    p_gt = (1.0 - smoothing) * y / y.sum(dim=-1, keepdim=True).clamp_min(1.0) + smoothing / num
    p = p_post.float().clamp_min(1e-12)
    kl = (p * (p.log() - p_gt.log())).sum(dim=-1)
    return hinge + beta * kl[valid].mean()


class HINTRAGLoss(nn.Module):
    def __init__(self, lambda_nce=0.1, lambda_ent=0.05, gamma=0.3, beta=0.1, tau_c=0.07, kl_smoothing=0.1):
        super().__init__()
        self.lambda_nce = float(lambda_nce)
        self.lambda_ent = float(lambda_ent)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.tau_c = float(tau_c)
        self.kl_smoothing = float(kl_smoothing)

    def forward(self, out, target_ids, labels, positive_keys=None, negative_keys=None):
        l_gen = generation_loss(out["logits"], target_ids)
        zero = l_gen.new_zeros(())
        l_nce, l_ent = zero, zero
        if self.lambda_nce > 0 and positive_keys is not None and "q_pri" in out:
            l_nce = contrastive_retrieval_loss(out["q_pri"], out.get("q_sec"), positive_keys, negative_keys, self.tau_c)
        if self.lambda_ent > 0 and out.get("evidence") is not None:
            l_ent = calibrated_entropy_loss(out["P_post"], out["S_post"], out["S"], labels, self.gamma, self.beta, self.kl_smoothing)
        total = l_gen + self.lambda_nce * l_nce + self.lambda_ent * l_ent
        return {"loss": total, "loss_gen": l_gen.detach(), "loss_nce": l_nce.detach(), "loss_ent": l_ent.detach()}
