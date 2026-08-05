from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_loss_gen(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_visual: int,
    pad_token_id: int = -100,
) -> torch.Tensor:
    B, total_seq, vocab_size = logits.shape
    text_seq = labels.shape[1]

    pred_logits = logits[:, n_visual: n_visual + text_seq - 1, :]
    target = labels[:, 1:]

    loss = F.cross_entropy(
        pred_logits.reshape(-1, vocab_size),
        target.reshape(-1),
        ignore_index=pad_token_id,
    )
    return loss

def compute_loss_nce(
    alpha_pri: torch.Tensor,
    alpha_sec: torch.Tensor,
    sample_indices_pri: np.ndarray,
    sample_indices_sec: np.ndarray,
    query_labels: torch.Tensor,
    memory_labels: np.ndarray,
    tau_c: float = 0.07,
) -> torch.Tensor:
    device = alpha_pri.device
    B = alpha_pri.shape[0]

    def _nce_one_query(alpha, sample_indices):
        retrieved_labels = memory_labels[sample_indices]
        q_lbl_np = query_labels.detach().cpu().numpy()
        pos_mask = (retrieved_labels == q_lbl_np[:, None]) & (q_lbl_np[:, None] >= 0)
        pos_mask_t = torch.tensor(pos_mask, dtype=alpha.dtype, device=device)

        pos_sum = (alpha * pos_mask_t).sum(dim=-1)
        pos_sum = pos_sum.clamp(min=1e-9)
        loss = -torch.log(pos_sum).mean()
        return loss

    loss_pri = _nce_one_query(alpha_pri, sample_indices_pri)
    loss_sec = _nce_one_query(alpha_sec, sample_indices_sec)

    return 0.5 * (loss_pri + loss_sec)

def compute_loss_ent(
    P_diag_post: torch.Tensor,
    S_pre: torch.Tensor,
    gt_labels: torch.Tensor,
    K: int = 14,
    gamma: float = 0.3,
    beta: float = 0.1,
) -> torch.Tensor:
    S_post = -(P_diag_post * torch.log(P_diag_post + 1e-9)).sum(dim=-1)

    hinge = F.relu(S_post - S_pre.to(S_post.device) + gamma)
    loss_hinge = hinge.mean()

    valid = (gt_labels >= 0)
    if valid.any():
        valid_post = P_diag_post[valid]
        valid_gt = gt_labels[valid]
        P_gt = F.one_hot(valid_gt, num_classes=K).float().to(valid_post.device)
        kl = -(P_gt * torch.log(valid_post.float() + 1e-9)).sum(dim=-1).mean()
    else:
        kl = torch.tensor(0.0, device=P_diag_post.device)

    return loss_hinge + beta * kl

class HINTRAGLoss(nn.Module):

    def __init__(
        self,
        n_visual: int = 256,
        K: int = 14,
        lambda_nce: float = 0.1,
        lambda_ent: float = 0.05,
        gamma: float = 0.3,
        beta: float = 0.1,
        tau_c: float = 0.07,
        pad_token_id: int = -100,
    ):
        super().__init__()
        self.n_visual = n_visual
        self.K = K
        self.lambda_nce = lambda_nce
        self.lambda_ent = lambda_ent
        self.gamma = gamma
        self.beta = beta
        self.tau_c = tau_c
        self.pad_token_id = pad_token_id

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        alpha_pri: torch.Tensor,
        alpha_sec: torch.Tensor,
        sample_idx_pri: np.ndarray,
        sample_idx_sec: np.ndarray,
        P_diag_post: torch.Tensor,
        S_pre: torch.Tensor,
        gt_labels: torch.Tensor,
        memory_labels: np.ndarray,
    ) -> dict:
        l_gen = compute_loss_gen(logits, labels, self.n_visual, self.pad_token_id)

        l_nce = compute_loss_nce(
            alpha_pri, alpha_sec,
            sample_idx_pri, sample_idx_sec,
            gt_labels, memory_labels, self.tau_c,
        )

        l_ent = compute_loss_ent(
            P_diag_post, S_pre, gt_labels,
            K=self.K, gamma=self.gamma, beta=self.beta,
        )

        total = l_gen + self.lambda_nce * l_nce + self.lambda_ent * l_ent

        return {
            "loss": total,
            "loss_gen": l_gen.detach(),
            "loss_nce": l_nce.detach(),
            "loss_ent": l_ent.detach(),
        }
