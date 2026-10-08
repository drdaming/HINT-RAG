import torch
import torch.nn as nn

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class VisualEncoder(nn.Module):
    def __init__(self, name, image_size=224):
        super().__init__()
        self.mean, self.std = CLIP_MEAN, CLIP_STD
        if name.startswith("timm:"):
            import timm

            self.backbone = timm.create_model(name[5:], pretrained=False, num_classes=0, img_size=image_size)
            self.kind = "timm"
        else:
            import open_clip

            if "::" in name:
                arch, tag = name.split("::", 1)
                clip = open_clip.create_model(arch, pretrained=tag)
            else:
                clip = open_clip.create_model(name if name.startswith("hf-hub:") else "hf-hub:" + name)
            visual = clip.visual
            self.mean = tuple(getattr(visual, "image_mean", None) or CLIP_MEAN)
            self.std = tuple(getattr(visual, "image_std", None) or CLIP_STD)
            if hasattr(visual, "trunk"):
                self.backbone = visual.trunk
                self.kind = "timm"
            else:
                visual.output_tokens = True
                self.backbone = visual
                self.kind = "open_clip"
            del clip
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
        sample = self.forward(torch.zeros(1, 3, image_size, image_size))
        self.num_patches = sample.size(1)
        self.output_dim = sample.size(2)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, pixel_values):
        if self.kind == "timm":
            tokens = self.backbone.forward_features(pixel_values)
            return tokens[:, getattr(self.backbone, "num_prefix_tokens", 1):]
        _, tokens = self.backbone(pixel_values)
        return tokens
