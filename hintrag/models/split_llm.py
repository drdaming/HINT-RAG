import torch
import torch.nn as nn


def build_causal_mask(attention_mask, dtype):
    size, length = attention_mask.shape
    causal = torch.ones(length, length, dtype=torch.bool, device=attention_mask.device).tril()
    allowed = causal.unsqueeze(0) & attention_mask.bool().unsqueeze(1)
    mask = torch.zeros(size, 1, length, length, dtype=dtype, device=attention_mask.device)
    return mask.masked_fill(~allowed.unsqueeze(1), torch.finfo(dtype).min)


class SplitLLM(nn.Module):
    def __init__(self, llm, split_layer, lora_rank=16, lora_alpha=32, lora_dropout=0.05, lora_targets=("q_proj", "v_proj")):
        super().__init__()
        if lora_rank > 0:
            from peft import LoraConfig, inject_adapter_in_model

            lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=list(lora_targets),
                bias="none",
            )
            llm = inject_adapter_in_model(lora_config, llm)
        for name, param in llm.named_parameters():
            param.requires_grad = "lora_" in name
            if param.requires_grad:
                param.data = param.data.float()
        base = llm.model
        layers = list(base.layers)
        if not 0 < split_layer < len(layers):
            raise ValueError(f"split_layer must be in (0, {len(layers)}), got {split_layer}")
        self.embed_tokens = base.embed_tokens
        self.perceive_layers = nn.ModuleList(layers[:split_layer])
        self.reason_layers = nn.ModuleList(layers[split_layer:])
        self.norm = base.norm
        self.lm_head = llm.lm_head
        self.rotary_emb = getattr(base, "rotary_emb", None)
        self.hidden_size = llm.config.hidden_size
        self.split_layer = split_layer

    @property
    def dtype(self):
        return self.embed_tokens.weight.dtype

    def embed(self, input_ids):
        return self.embed_tokens(input_ids)

    def _run(self, layers, hidden, attention_mask):
        size, length, _ = hidden.shape
        position_ids = torch.arange(length, device=hidden.device).unsqueeze(0).expand(size, -1)
        kwargs = {"attention_mask": build_causal_mask(attention_mask, hidden.dtype), "position_ids": position_ids}
        if self.rotary_emb is not None:
            kwargs["position_embeddings"] = self.rotary_emb(hidden, position_ids)
        for layer in layers:
            out = layer(hidden, **kwargs)
            hidden = out[0] if isinstance(out, (tuple, list)) else out
        return hidden

    def perceive(self, inputs_embeds, attention_mask):
        return self._run(self.perceive_layers, inputs_embeds, attention_mask)

    def reason(self, hidden, attention_mask):
        return self.norm(self._run(self.reason_layers, hidden, attention_mask))
