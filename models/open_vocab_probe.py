from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class OpenVocabProbe(nn.Module):

    def __init__(
        self,
        embed_tokens: nn.Embedding,
        concept_texts: List[str],
        tokenizer,
        tau_p: float = 0.1,
        d: int = 4096,
    ):
        super().__init__()
        self.tau_p = tau_p
        self.K = len(concept_texts)

        concept_embs = self._encode_concepts(embed_tokens, concept_texts, tokenizer, d)
        self.register_buffer("concept_embeddings", concept_embs)

    @staticmethod
    @torch.no_grad()
    def _encode_concepts(
        embed_tokens: nn.Embedding,
        concept_texts: List[str],
        tokenizer,
        d: int,
    ) -> torch.Tensor:
        embs = []
        for text in concept_texts:
            tok = tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=False,
            )
            input_ids = tok["input_ids"]
            input_ids = input_ids.to(embed_tokens.weight.device)
            token_embs = embed_tokens(input_ids)
            concept_emb = token_embs.mean(dim=1)
            embs.append(concept_emb)

        concept_embs = torch.cat(embs, dim=0)
        return concept_embs

    def forward(self, h_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_norm = F.normalize(h_state.float(), dim=-1)
        c_norm = F.normalize(self.concept_embeddings.float(), dim=-1)

        logits = torch.matmul(h_norm, c_norm.T) / self.tau_p

        P_diag = F.softmax(logits, dim=-1)

        S = -(P_diag * torch.log(P_diag + 1e-9)).sum(dim=-1)

        return P_diag, S
