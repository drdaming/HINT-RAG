import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM, AutoTokenizer, LlamaConfig

class MLPProjector(nn.Module):

    def __init__(self, d_v: int = 1024, d_hidden: int = 8192, d_out: int = 4096):
        super().__init__()
        self.linear1 = nn.Linear(d_v, d_hidden)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(d_hidden, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.act(self.linear1(x)))

class SplitVicuna(nn.Module):

    def __init__(
        self,
        model_name_or_path: str,
        split_layer: int = 16,
        d_v: int = 1024,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        n_visual_tokens: int = 256,
        torch_dtype=torch.bfloat16,
    ):
        super().__init__()
        self.split_layer = split_layer
        self.n_visual_tokens = n_visual_tokens
        self.torch_dtype = torch_dtype

        print(f"[SplitVicuna] Loading {model_name_or_path} ...")
        llm = LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        llm.eval()

        self.embed_tokens = llm.model.embed_tokens
        self.first_layers = nn.ModuleList(llm.model.layers[:split_layer])
        self.second_layers = nn.ModuleList(llm.model.layers[split_layer:])
        self.norm = llm.model.norm
        self.lm_head = llm.lm_head

        for p in self.embed_tokens.parameters():
            p.requires_grad = False
        for p in self.first_layers.parameters():
            p.requires_grad = False
        for p in self.second_layers.parameters():
            p.requires_grad = False
        for p in self.norm.parameters():
            p.requires_grad = False
        for p in self.lm_head.parameters():
            p.requires_grad = False

        self.d = llm.config.hidden_size
        self.n_layers = llm.config.num_hidden_layers

        del llm

        self.mlp_projector = MLPProjector(
            d_v=d_v, d_hidden=self.d * 2, d_out=self.d
        )
        self.mlp_projector = self.mlp_projector.to(torch_dtype)

        if lora_rank > 0:
            self._apply_lora(lora_rank, lora_alpha, lora_dropout)

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[SplitVicuna] Loaded. d={self.d}, split={split_layer}/{self.n_layers}. "
            f"Trainable params: {n_trainable:,}"
        )

    def _apply_lora(self, rank: int, alpha: int, dropout: float) -> None:
        import peft.tuners.lora as lora_module

        target_modules = {"q_proj", "v_proj"}

        def _replace_linear(parent: nn.Module) -> None:
            for name, child in list(parent.named_children()):
                if name in target_modules and isinstance(child, nn.Linear):
                    lora_linear = lora_module.Linear(
                        base_layer=child,
                        adapter_name="default",
                        r=rank,
                        lora_alpha=alpha,
                        lora_dropout=dropout,
                        fan_in_fan_out=False,
                        is_target_conv_1d_layer=False,
                        init_lora_weights=True,
                    )
                    lora_linear.weight.requires_grad = False
                    setattr(parent, name, lora_linear)
                else:
                    _replace_linear(child)

        for layer in list(self.first_layers) + list(self.second_layers):
            _replace_linear(layer)

    @staticmethod
    def _make_causal_mask(
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
        batch_size: int,
    ) -> torch.Tensor:
        min_val = torch.finfo(dtype).min
        mask = torch.full((seq_len, seq_len), fill_value=min_val, dtype=dtype, device=device)
        cond = torch.arange(seq_len, device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len)

    @staticmethod
    def _extend_mask_for_padding(
        causal_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        n_visual: int,
    ) -> torch.Tensor:
        B, text_seq = attention_mask.shape
        dtype = causal_mask.dtype
        device = causal_mask.device
        min_val = torch.finfo(dtype).min

        visual_ones = torch.ones(B, n_visual, dtype=attention_mask.dtype, device=device)
        full_mask = torch.cat([visual_ones, attention_mask], dim=1)

        add_mask = (1.0 - full_mask.float()) * min_val
        add_mask = add_mask[:, None, None, :]

        return causal_mask + add_mask.to(dtype)

    def embed_visual(self, patch_feats: torch.Tensor) -> torch.Tensor:
        patch_feats = patch_feats.to(self.torch_dtype)
        return self.mlp_projector(patch_feats)

    def forward_first_half(
        self,
        visual_prefix: torch.Tensor,
        text_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N_v, d = visual_prefix.shape
        device = visual_prefix.device
        dtype = visual_prefix.dtype

        text_embeds = self.embed_tokens(text_input_ids)
        text_embeds = text_embeds.to(dtype)

        hidden = torch.cat([visual_prefix, text_embeds], dim=1)
        total_seq = hidden.shape[1]

        position_ids = torch.arange(total_seq, device=device).unsqueeze(0)

        causal_mask = self._make_causal_mask(total_seq, dtype, device, B)
        if attention_mask is not None:
            causal_mask = self._extend_mask_for_padding(causal_mask, attention_mask, N_v)

        for layer in self.first_layers:
            layer_out = layer(
                hidden,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            hidden = layer_out[0]

        H_mid = hidden

        h_state = H_mid[:, :N_v, :].mean(dim=1)

        return H_mid, h_state

    def forward_second_half(
        self,
        H_fused: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        n_visual: Optional[int] = None,
    ) -> torch.Tensor:
        B, total_seq, d = H_fused.shape
        device = H_fused.device
        dtype = H_fused.dtype
        n_v = n_visual if n_visual is not None else self.n_visual_tokens

        position_ids = torch.arange(total_seq, device=device).unsqueeze(0)
        causal_mask = self._make_causal_mask(total_seq, dtype, device, B)
        if attention_mask is not None:
            causal_mask = self._extend_mask_for_padding(causal_mask, attention_mask, n_v)

        hidden = H_fused
        for layer in self.second_layers:
            layer_out = layer(
                hidden,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
            )
            hidden = layer_out[0]

        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        return logits
