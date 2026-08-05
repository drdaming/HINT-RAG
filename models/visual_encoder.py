import torch
import torch.nn as nn
from typing import Tuple

class VisualEncoder(nn.Module):

    def __init__(self, model_name: str):
        super().__init__()
        import open_clip

        if "::" in model_name:
            arch, pretrained = model_name.split("::", 1)
            clip_model, _, preprocess = open_clip.create_model_and_transforms(
                arch, pretrained=pretrained
            )
        else:
            hf_name = model_name if model_name.startswith("hf-hub:") else f"hf-hub:{model_name}"
            clip_model, _, preprocess = open_clip.create_model_and_transforms(hf_name)
        self._preprocess = preprocess
        self._vit = clip_model.visual

        for p in self._vit.parameters():
            p.requires_grad = False
        self._vit.eval()

        if hasattr(self._vit, "transformer") and hasattr(self._vit.transformer, "width"):
            self._output_dim = self._vit.transformer.width
        else:
            captured = {}
            def _hook(m, inp, out):
                captured["tokens"] = out
            transformer = self._vit.transformer
            last_block = transformer.resblocks[-1]
            handle = last_block.register_forward_hook(_hook)
            dummy = torch.zeros(1, 3, 224, 224)
            with torch.no_grad():
                self._vit(dummy)
            handle.remove()
            tokens = captured["tokens"]
            if tokens.dim() == 3 and tokens.shape[0] != 1:
                tokens = tokens.permute(1, 0, 2)
            self._output_dim = tokens.shape[-1]

        print(
            f"[VisualEncoder] BiomedCLIP ViT-L/14 loaded. "
            f"output_dim={self._output_dim}, frozen."
        )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def get_preprocess(self):
        return self._preprocess

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            out = self._vit(pixel_values)

        if isinstance(out, (tuple, list)):
            cls_feat, patch_feats = out[0], out[1]
        else:
            cls_feat = out
            patch_feats = self._extract_patch_features(pixel_values)

        return cls_feat, patch_feats

    def _extract_patch_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        captured = {}

        def hook_fn(module, input, output):
            captured["tokens"] = output

        transformer = self._vit.transformer
        last_block = transformer.resblocks[-1]
        handle = last_block.register_forward_hook(hook_fn)

        with torch.no_grad():
            _ = self._vit(pixel_values)

        handle.remove()

        tokens = captured["tokens"]
        if tokens.dim() == 3:
            if tokens.shape[0] != pixel_values.shape[0]:
                tokens = tokens.permute(1, 0, 2)
        patch_feats = tokens[:, 1:, :]
        return patch_feats.float()
