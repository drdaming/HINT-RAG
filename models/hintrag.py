from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from data.iuxray_dataset import CHEXPERT_CONCEPTS
from models.visual_encoder import VisualEncoder
from models.split_llm import SplitVicuna
from models.open_vocab_probe import OpenVocabProbe
from models.gumbel_hypothesis import GumbelHypothesisSelector
from models.hcrm import HCRM
from models.fusion import EvidenceFusion

class HINTRAGModel(nn.Module):

    def __init__(self, config, tokenizer, memory_bank):
        super().__init__()
        self.config = config
        self.memory_bank = memory_bank

        mc = config.model
        pc = config.pretrained
        mode = getattr(config, "mode", "full")

        self.visual_encoder = VisualEncoder(model_name=pc.biomedclip)
        d_v = self.visual_encoder.output_dim

        self.split_llm = SplitVicuna(
            model_name_or_path=pc.vicuna,
            split_layer=mc.split_layer,
            d_v=d_v,
            lora_rank=mc.lora_rank,
            lora_alpha=mc.lora_alpha,
            lora_dropout=mc.lora_dropout,
            n_visual_tokens=256,
        )
        d = self.split_llm.d

        self.probe = OpenVocabProbe(
            embed_tokens=self.split_llm.embed_tokens,
            concept_texts=CHEXPERT_CONCEPTS,
            tokenizer=tokenizer,
            tau_p=mc.tau_p,
            d=d,
        )

        self.gumbel_selector = GumbelHypothesisSelector(
            tau_g_init=mc.tau_g_init,
            tau_g_min=mc.tau_g_min,
        )

        self.hcrm = HCRM(
            d=d,
            text_dim=768,
            R=mc.R,
            tau_r_init=mc.tau_r_init,
        )

        self.fusion = EvidenceFusion(d=d, n_heads=mc.n_heads)

        self.n_visual_tokens = 256

    def perceive_only(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        _, patch_feats = self.visual_encoder(pixel_values)

        visual_prefix = self.split_llm.embed_visual(patch_feats)

        _, h_state = self.split_llm.forward_first_half(
            visual_prefix, input_ids, attention_mask
        )

        return h_state

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict:

        _, patch_feats = self.visual_encoder(pixel_values)
        visual_prefix = self.split_llm.embed_visual(patch_feats)

        H_mid, h_state = self.split_llm.forward_first_half(
            visual_prefix, input_ids, attention_mask
        )

        P_diag, S = self.probe(h_state)

        e_pri, e_sec = self.gumbel_selector(
            P_diag, self.probe.concept_embeddings
        )

        E_pri, E_sec, alpha_pri, alpha_sec, idx_pri, idx_sec = self.hcrm(
            h_state, e_pri, e_sec, self.memory_bank
        )

        H_fused, g = self.fusion(H_mid, E_pri, E_sec, S)

        logits = self.split_llm.forward_second_half(
            H_fused, attention_mask, n_visual=self.n_visual_tokens
        )

        h_fused_avg = H_fused[:, :self.n_visual_tokens, :].mean(dim=1)
        P_diag_post, _ = self.probe(h_fused_avg)

        return {
            "logits": logits,
            "H_mid": H_mid,
            "H_fused": H_fused,
            "h_state": h_state,
            "P_diag": P_diag,
            "S": S,
            "P_diag_post": P_diag_post,
            "g": g,
            "alpha_pri": alpha_pri,
            "alpha_sec": alpha_sec,
            "sample_idx_pri": idx_pri,
            "sample_idx_sec": idx_sec,
        }

    def _run_pipeline(self, visual_prefix, cur_ids, cur_mask, amp_ctx):
        from contextlib import nullcontext
        with amp_ctx:
            H_mid, h_state = self.split_llm.forward_first_half(
                visual_prefix, cur_ids, cur_mask
            )
            P_diag, S = self.probe(h_state)
            e_pri, e_sec = self.gumbel_selector(P_diag, self.probe.concept_embeddings)
            E_pri, E_sec, _, _, _, _ = self.hcrm(h_state, e_pri, e_sec, self.memory_bank)
            H_fused, _ = self.fusion(H_mid, E_pri, E_sec, S)
            logits = self.split_llm.forward_second_half(
                H_fused, cur_mask, n_visual=self.n_visual_tokens
            )
        return logits, H_mid, H_fused

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tokenizer,
        max_new_tokens: int = 80,
        num_beams: int = 1,
        no_repeat_ngram_size: int = 3,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        self.eval()

        use_amp = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        from contextlib import nullcontext
        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

        B = pixel_values.shape[0]
        device = pixel_values.device
        eos_id = tokenizer.eos_token_id

        with amp_ctx:
            _, patch_feats = self.visual_encoder(pixel_values)
            visual_prefix = self.split_llm.embed_visual(patch_feats)

        if num_beams > 1:
            return self._beam_search(
                visual_prefix, input_ids, attention_mask,
                amp_ctx, B, device, eos_id,
                max_new_tokens, num_beams, no_repeat_ngram_size,
            )

        cur_ids = input_ids.clone()
        cur_mask = attention_mask.clone()
        generated = []

        for step in range(max_new_tokens):
            logits, H_mid, H_fused = self._run_pipeline(visual_prefix, cur_ids, cur_mask, amp_ctx)

            next_token_logits = logits[:, -1, :].float()

            if no_repeat_ngram_size > 1 and len(generated) >= no_repeat_ngram_size - 1:
                gen_tensor = torch.stack(generated, dim=1)
                for b in range(B):
                    seq = gen_tensor[b].tolist()
                    prefix = tuple(seq[-(no_repeat_ngram_size - 1):])
                    banned = set()
                    for i in range(len(seq) - no_repeat_ngram_size + 1):
                        if tuple(seq[i:i + no_repeat_ngram_size - 1]) == prefix:
                            banned.add(seq[i + no_repeat_ngram_size - 1])
                    for tok in banned:
                        next_token_logits[b, tok] = float("-inf")

            next_token_logits = next_token_logits / max(temperature, 1e-8)

            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs - torch.softmax(sorted_logits, dim=-1) > top_p
            for b in range(B):
                next_token_logits[b, sorted_indices[b][sorted_indices_to_remove[b]]] = float("-inf")

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_ids = torch.multinomial(probs, num_samples=1).squeeze(1)

            generated.append(next_token_ids)

            if eos_id is not None and (next_token_ids == eos_id).all():
                break

            cur_ids = torch.cat([cur_ids, next_token_ids.unsqueeze(1)], dim=1)
            cur_mask = torch.cat(
                [cur_mask, torch.ones(B, 1, dtype=cur_mask.dtype, device=device)], dim=1
            )

            del H_mid, H_fused, logits
            torch.cuda.empty_cache()

        if not generated:
            return torch.zeros(B, 0, dtype=torch.long, device=device)

        return torch.stack(generated, dim=1)

    @torch.no_grad()
    def _beam_search(
        self,
        visual_prefix,
        input_ids,
        attention_mask,
        amp_ctx,
        B,
        device,
        eos_id,
        max_new_tokens,
        num_beams,
        no_repeat_ngram_size,
    ) -> torch.Tensor:
        assert B == 1, "Beam search in HINT-RAG requires batch_size=1"

        beam_ids = input_ids.repeat(num_beams, 1)
        beam_mask = attention_mask.repeat(num_beams, 1)
        beam_vp = visual_prefix.repeat(num_beams, 1, 1)

        beam_scores = torch.zeros(num_beams, device=device)
        beam_scores[1:] = float("-inf")

        beam_generated = [[] for _ in range(num_beams)]
        done = [False] * num_beams

        for step in range(max_new_tokens):
            logits, H_mid, H_fused = self._run_pipeline(beam_vp, beam_ids, beam_mask, amp_ctx)
            next_token_logits = logits[:, -1, :].float()

            if no_repeat_ngram_size > 1 and step >= no_repeat_ngram_size - 1:
                for b in range(num_beams):
                    seq = beam_generated[b]
                    prefix = tuple(seq[-(no_repeat_ngram_size - 1):])
                    banned = set()
                    for i in range(len(seq) - no_repeat_ngram_size + 1):
                        if tuple(seq[i:i + no_repeat_ngram_size - 1]) == prefix:
                            banned.add(seq[i + no_repeat_ngram_size - 1])
                    for tok in banned:
                        next_token_logits[b, tok] = float("-inf")

            log_probs = torch.log_softmax(next_token_logits, dim=-1)

            scores = beam_scores.unsqueeze(1) + log_probs

            flat_scores = scores.view(-1)
            topk_scores, topk_ids = flat_scores.topk(num_beams, dim=0)

            beam_idx = topk_ids // next_token_logits.shape[-1]
            token_idx = topk_ids % next_token_logits.shape[-1]

            new_beam_ids = []
            new_beam_mask = []
            new_beam_vp = []
            new_beam_generated = []
            new_beam_scores = []

            for k in range(num_beams):
                b = beam_idx[k].item()
                tok = token_idx[k].item()

                new_ids = torch.cat([beam_ids[b:b+1], torch.tensor([[tok]], device=device)], dim=1)
                new_mask = torch.cat([beam_mask[b:b+1], torch.ones(1, 1, dtype=beam_mask.dtype, device=device)], dim=1)

                new_beam_ids.append(new_ids)
                new_beam_mask.append(new_mask)
                new_beam_vp.append(beam_vp[b:b+1])
                new_beam_generated.append(beam_generated[b] + [tok])
                new_beam_scores.append(topk_scores[k])

            beam_ids = torch.cat(new_beam_ids, dim=0)
            beam_mask = torch.cat(new_beam_mask, dim=0)
            beam_vp = torch.cat(new_beam_vp, dim=0)
            beam_generated = new_beam_generated
            beam_scores = torch.stack(new_beam_scores)

            del H_mid, H_fused, logits
            torch.cuda.empty_cache()

            if eos_id is not None and beam_generated[0][-1] == eos_id:
                break

        best = beam_generated[0]
        if best and eos_id is not None and best[-1] == eos_id:
            best = best[:-1]
        return torch.tensor([best], dtype=torch.long, device=device)
