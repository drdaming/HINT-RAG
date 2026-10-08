import torch
import torch.nn as nn
import torch.nn.functional as F

from .disambiguate import DisambiguateModule
from .perceive import OpenVocabularyProbe, SoftHypothesisRouting, encode_concepts
from .query import ContrastiveQuery, DifferentiableRetrieval

RETRIEVAL_MODES = ("hint", "visual", "none")


def ban_repeated_ngrams(logits, generated, n):
    if n <= 0 or generated.size(1) < n - 1:
        return logits
    history = generated.tolist()
    for row, seq in enumerate(history):
        prefix = tuple(seq[len(seq) - n + 1:]) if n > 1 else tuple()
        banned = {seq[i + n - 1] for i in range(len(seq) - n + 1) if tuple(seq[i : i + n - 1]) == prefix}
        if banned:
            logits[row, list(banned)] = float("-inf")
    return logits


class HINTRAG(nn.Module):
    def __init__(self, cfg, visual_encoder, split_llm, tokenizer, value_dim, prefix_ids, suffix_ids, concepts):
        super().__init__()
        model_cfg = cfg.model
        ablation = model_cfg.ablation
        if ablation.retrieval not in RETRIEVAL_MODES:
            raise ValueError(f"model.ablation.retrieval must be one of {RETRIEVAL_MODES}")
        self.retrieval_mode = ablation.retrieval
        self.use_secondary = bool(ablation.contrastive_query) and self.retrieval_mode == "hint"
        d = split_llm.hidden_size
        self.visual_encoder = visual_encoder
        self.llm = split_llm
        self.projector = nn.Sequential(nn.Linear(visual_encoder.output_dim, d), nn.GELU(), nn.Linear(d, d))
        self.probe = OpenVocabularyProbe(
            encode_concepts(split_llm.embed_tokens, tokenizer, concepts),
            model_cfg.tau_p,
            normalize=model_cfg.get("probe_normalize", True),
        )
        self.routing = SoftHypothesisRouting(model_cfg.tau_g_start, model_cfg.tau_g_end, soft=ablation.soft_routing)
        self.query = ContrastiveQuery(d)
        self.retrieval = DifferentiableRetrieval(d, value_dim, model_cfg.top_r, model_cfg.tau_r_init)
        self.disambiguate = DisambiguateModule(
            d,
            model_cfg.n_heads,
            model_cfg.gate_w_init,
            model_cfg.gate_b_init,
            dynamic_gate=bool(ablation.dynamic_gate) and self.retrieval_mode == "hint",
        )
        self.register_buffer("prefix_ids", torch.tensor(prefix_ids, dtype=torch.long), persistent=False)
        self.register_buffer("suffix_ids", torch.tensor(suffix_ids, dtype=torch.long), persistent=False)
        self.memory_bank = None

    def attach_memory_bank(self, bank):
        self.memory_bank = bank

    def encode_image(self, pixel_values):
        size, views = pixel_values.shape[:2]
        feats = self.visual_encoder(pixel_values.flatten(0, 1))
        feats = feats.reshape(size, views * feats.size(1), feats.size(2))
        return self.projector(feats.to(self.projector[0].weight.dtype))

    def perceive(self, visual_tokens, text_ids, text_mask):
        size = visual_tokens.size(0)
        prefix = self.llm.embed(self.prefix_ids).unsqueeze(0).expand(size, -1, -1)
        text = self.llm.embed(text_ids)
        embeds = torch.cat([prefix, visual_tokens.to(prefix.dtype), text], dim=1)
        start, stop = prefix.size(1), prefix.size(1) + visual_tokens.size(1)
        mask = torch.cat([text_mask.new_ones(size, stop), text_mask], dim=1)
        h_mid = self.llm.perceive(embeds, mask)
        h = h_mid[:, start:stop].float().mean(dim=1)
        return h_mid, h, mask, slice(start, stop)

    def state_vector(self, pixel_values):
        visual = self.encode_image(pixel_values)
        suffix = self.suffix_ids.unsqueeze(0).expand(visual.size(0), -1)
        _, h, _, _ = self.perceive(visual, suffix, torch.ones_like(suffix))
        return h

    def hypothesize(self, h):
        prob, entropy = self.probe(h)
        e_pri, e_sec, p_pri, p_sec = self.routing(prob, self.probe.concept_embeddings)
        return {"P_diag": prob, "S": entropy, "e_pri": e_pri, "e_sec": e_sec, "p_pri": p_pri, "p_sec": p_sec}

    def retrieve(self, h, hyp, exclude=None):
        if self.retrieval_mode == "none":
            return {"evidence": None}
        if self.memory_bank is None:
            raise RuntimeError("memory bank is not attached")
        if self.retrieval_mode == "visual":
            query = F.normalize(h.float(), dim=-1)
            evidence, alpha, indices = self.retrieval(query, self.memory_bank, exclude)
            return {"evidence": evidence.unsqueeze(1), "alpha_pri": alpha, "idx_pri": indices}
        q_pri = self.query.primary(h, hyp["e_pri"])
        e_pri, a_pri, i_pri = self.retrieval(q_pri, self.memory_bank, exclude)
        out = {"q_pri": q_pri, "E_pri": e_pri, "alpha_pri": a_pri, "idx_pri": i_pri}
        evidence = [e_pri]
        if self.use_secondary:
            q_sec = self.query.secondary(h, hyp["e_pri"], hyp["e_sec"])
            e_sec, a_sec, i_sec = self.retrieval(q_sec, self.memory_bank, exclude)
            out.update({"q_sec": q_sec, "E_sec": e_sec, "alpha_sec": a_sec, "idx_sec": i_sec})
            evidence.append(e_sec)
        out["evidence"] = torch.stack(evidence, dim=1)
        return out

    def fuse(self, h_mid, evidence, g):
        if evidence is None:
            return h_mid
        return self.disambiguate(h_mid, evidence, g)

    def forward(self, pixel_values, text_ids, text_mask, exclude=None):
        visual = self.encode_image(pixel_values)
        h_mid, h, mask, visual_slice = self.perceive(visual, text_ids, text_mask)
        hyp = self.hypothesize(h)
        ret = self.retrieve(h, hyp, exclude)
        g = self.disambiguate.gate(hyp["S"])
        h_fused = self.fuse(h_mid, ret["evidence"], g)
        hidden = self.llm.reason(h_fused, mask)
        logits = self.llm.lm_head(hidden[:, visual_slice.stop - 1 : -1])
        p_post, s_post = self.probe(h_fused[:, visual_slice].float().mean(dim=1))
        return {**hyp, **ret, "logits": logits, "h": h, "g": g, "P_post": p_post, "S_post": s_post}

    def _next_token_logits(self, visual, ids, evidence, g):
        h_mid, _, mask, _ = self.perceive(visual, ids, torch.ones_like(ids))
        hidden = self.llm.reason(self.fuse(h_mid, evidence, g), mask)
        return self.llm.lm_head(hidden[:, -1]).float()

    @torch.no_grad()
    def prepare_generation(self, pixel_values):
        visual = self.encode_image(pixel_values)
        suffix = self.suffix_ids.unsqueeze(0).expand(visual.size(0), -1)
        _, h, _, _ = self.perceive(visual, suffix, torch.ones_like(suffix))
        hyp = self.hypothesize(h)
        ret = self.retrieve(h, hyp)
        g = self.disambiguate.gate(hyp["S"])
        return visual, suffix, ret["evidence"], g, {**hyp, **ret}

    @torch.no_grad()
    def generate(self, pixel_values, eos_token_id, pad_token_id, max_new_tokens=100, num_beams=1, no_repeat_ngram_size=3, length_penalty=1.0):
        was_training = self.training
        self.eval()
        visual, ids, evidence, g, _ = self.prepare_generation(pixel_values)
        if num_beams <= 1:
            out = self._greedy(visual, ids, evidence, g, eos_token_id, pad_token_id, max_new_tokens, no_repeat_ngram_size)
        else:
            out = self._beam(visual, ids, evidence, g, eos_token_id, pad_token_id, max_new_tokens, num_beams, no_repeat_ngram_size, length_penalty)
        self.train(was_training)
        return out

    def _greedy(self, visual, ids, evidence, g, eos, pad, max_new_tokens, ngram):
        size = visual.size(0)
        generated = ids.new_zeros(size, 0)
        finished = torch.zeros(size, dtype=torch.bool, device=ids.device)
        for _ in range(max_new_tokens):
            logits = ban_repeated_ngrams(self._next_token_logits(visual, ids, evidence, g), generated, ngram)
            token = logits.argmax(dim=-1)
            token = torch.where(finished, torch.full_like(token, pad), token)
            generated = torch.cat([generated, token[:, None]], dim=1)
            ids = torch.cat([ids, token[:, None]], dim=1)
            finished |= token == eos
            if bool(finished.all()):
                break
        return generated

    def _beam(self, visual, ids, evidence, g, eos, pad, max_new_tokens, beams, ngram, length_penalty):
        size = visual.size(0)
        visual = visual.repeat_interleave(beams, dim=0)
        evidence = None if evidence is None else evidence.repeat_interleave(beams, dim=0)
        g = g.repeat_interleave(beams, dim=0)
        ids = ids.repeat_interleave(beams, dim=0)
        scores = torch.zeros(size, beams, device=ids.device)
        scores[:, 1:] = float("-inf")
        scores = scores.view(-1)
        generated = ids.new_zeros(size * beams, 0)
        finished = torch.zeros(size * beams, dtype=torch.bool, device=ids.device)
        lengths = torch.zeros(size * beams, dtype=torch.float, device=ids.device)
        base = (torch.arange(size, device=ids.device) * beams).unsqueeze(1)
        for _ in range(max_new_tokens):
            logits = ban_repeated_ngrams(self._next_token_logits(visual, ids, evidence, g), generated, ngram)
            log_prob = logits.log_softmax(dim=-1)
            log_prob[finished] = float("-inf")
            log_prob[finished, eos] = 0.0
            vocab = log_prob.size(-1)
            candidates = (scores.unsqueeze(1) + log_prob).view(size, beams * vocab)
            top_scores, top_ids = candidates.topk(beams, dim=-1)
            source = (base + top_ids // vocab).view(-1)
            token = (top_ids % vocab).view(-1)
            lengths = torch.where(finished[source], lengths[source], lengths[source] + 1)
            finished = finished[source] | (token == eos)
            generated = torch.cat([generated[source], token[:, None]], dim=1)
            ids = torch.cat([ids[source], token[:, None]], dim=1)
            scores = top_scores.view(-1)
            if bool(finished.all()):
                break
        normalized = (scores / lengths.clamp_min(1.0) ** length_penalty).view(size, beams)
        best = (torch.arange(size, device=ids.device) * beams) + normalized.argmax(dim=-1)
        out = generated[best]
        after_eos = (out == eos).long().cumsum(dim=1) - (out == eos).long()
        return out.masked_fill(after_eos > 0, pad)
