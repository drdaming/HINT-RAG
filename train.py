import argparse
import os
import sys

import torch
import yaml

def load_config(config_path: str):
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    class Config:
        def __init__(self, d):
            for k, v in d.items():
                if isinstance(v, dict):
                    setattr(self, k, Config(v))
                else:
                    setattr(self, k, v)

        def __repr__(self):
            return str(vars(self))

    return Config(raw)

def main():
    parser = argparse.ArgumentParser(description="Train HINT-RAG on IU X-Ray")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint dir to resume from")
    args = parser.parse_args()

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train.py] Using device: {device}")
    print(f"[train.py] Mode: {config.mode}")

    os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")

    print("[train.py] Loading Vicuna tokenizer ...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.pretrained.vicuna,
        use_fast=False,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[train.py] Loading ClinicalBERT ...")
    from transformers import AutoTokenizer as AutoTok, AutoModel
    text_tokenizer = AutoTok.from_pretrained(config.pretrained.clinicalbert)
    text_encoder = AutoModel.from_pretrained(config.pretrained.clinicalbert)
    text_encoder = text_encoder.to(device).eval()
    for p in text_encoder.parameters():
        p.requires_grad = False

    print("[train.py] Building datasets ...")
    import open_clip
    bm = config.pretrained.biomedclip
    if "::" in bm:
        arch, pretrained = bm.split("::", 1)
        _, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    else:
        hf_name = bm if bm.startswith("hf-hub:") else f"hf-hub:{bm}"
        _, _, preprocess = open_clip.create_model_and_transforms(hf_name)

    from data.iuxray_dataset import IUXRayDataset

    dc = config.dataset
    train_dataset = IUXRayDataset(
        image_dir=config.data.image_dir,
        report_dir=config.data.report_dir,
        split="train",
        processor=preprocess,
        tokenizer=tokenizer,
        max_text_len=dc.max_report_length,
        train_ratio=dc.train_ratio,
        val_ratio=dc.val_ratio,
        seed=dc.seed,
    )
    val_dataset = IUXRayDataset(
        image_dir=config.data.image_dir,
        report_dir=config.data.report_dir,
        split="val",
        processor=preprocess,
        tokenizer=tokenizer,
        max_text_len=dc.max_report_length,
        train_ratio=dc.train_ratio,
        val_ratio=dc.val_ratio,
        seed=dc.seed,
    )

    from data.memory_bank import MemoryBank

    mc = config.model
    visual_dim = (
        mc.visual_feature_dim.full if config.mode == "full" else mc.visual_feature_dim.dev
    )

    memory_bank = MemoryBank(
        bank_size=len(train_dataset),
        visual_dim=visual_dim,
        text_dim=768,
        momentum=config.training.momentum,
        faiss_nlist=mc.faiss_nlist,
        faiss_nprobe=mc.faiss_nprobe,
        device=device,
    )

    print("[train.py] Building HINTRAGModel ...")
    from models.hintrag import HINTRAGModel

    model = HINTRAGModel(config, tokenizer, memory_bank)
    model = model.to(device)

    from training.trainer import Trainer

    trainer = Trainer(
        config=config,
        model=model,
        memory_bank=memory_bank,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        device=device,
    )

    if args.resume:
        print(f"[train.py] Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    trainer.train(text_encoder, text_tokenizer)

    print("[train.py] Training complete.")

if __name__ == "__main__":
    main()
