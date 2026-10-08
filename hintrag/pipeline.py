import json
import os

import torch

from .builder import build_model, build_text_encoder, build_tokenizer, concepts_from_config, encode_reports, text_hidden_size
from .config import save_config
from .data import ReportDataset, build_transform
from .memory_bank import MemoryBank
from .metrics import compute_metrics, format_metrics
from .trainer import Trainer, bank_dtype, generate_reports, load_trainable_state_dict
from .utils import ensure_dir, resolve_device, set_seed


def prepare(cfg):
    set_seed(int(cfg.seed))
    device = resolve_device(cfg.get("device", "auto"))
    tokenizer = build_tokenizer(cfg)
    model = build_model(cfg, tokenizer, text_hidden_size(cfg)).to(device)
    transform = build_transform(cfg.model.image_size, model.visual_encoder.mean, model.visual_encoder.std)
    return device, tokenizer, model, transform


def make_dataset(cfg, split, tokenizer, transform):
    return ReportDataset(cfg.data, split, tokenizer, transform, concepts_from_config(cfg))


def run_training(cfg, resume=None):
    device, tokenizer, model, transform = prepare(cfg)
    ensure_dir(cfg.output_dir)
    save_config(cfg, os.path.join(cfg.output_dir, "config.yaml"))
    train_set = make_dataset(cfg, "train", tokenizer, transform)
    val_set = make_dataset(cfg, "val", tokenizer, transform)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[HINT-RAG] device={device} train={len(train_set)} val={len(val_set)} trainable_params={n_trainable:,}", flush=True)
    trainer = Trainer(cfg, model, tokenizer, train_set, val_set, device)
    if resume:
        trainer.resume(resume)
    else:
        text_tokenizer, text_encoder = build_text_encoder(cfg, tokenizer)
        values = encode_reports(train_set.reports, text_tokenizer, text_encoder, device)
        del text_encoder
        trainer.setup_memory_bank(values)
    history = trainer.fit()
    return trainer, history


def load_for_inference(cfg, checkpoint):
    device, tokenizer, model, transform = prepare(cfg)
    path = checkpoint if checkpoint.endswith(".pt") else os.path.join(checkpoint, "checkpoint.pt")
    state = torch.load(path, map_location="cpu", weights_only=False)
    load_trainable_state_dict(model, state["model"])
    bank = MemoryBank.from_state(
        state["bank"],
        device=device,
        dtype=bank_dtype(cfg, device),
        search=cfg.memory.search,
        faiss_nlist=cfg.memory.faiss_nlist,
        faiss_nprobe=cfg.memory.faiss_nprobe,
    )
    model.attach_memory_bank(bank)
    model.eval()
    return device, tokenizer, model, transform


def run_evaluation(cfg, checkpoint, split="test", output=None):
    device, tokenizer, model, transform = load_for_inference(cfg, checkpoint)
    dataset = make_dataset(cfg, split, tokenizer, transform)
    use_bf16 = bool(cfg.train.bf16) and device.type == "cuda" and torch.cuda.is_bf16_supported()
    records = generate_reports(model, dataset, tokenizer, cfg, device, use_bf16)
    metrics = compute_metrics([r["reference"] for r in records], [r["prediction"] for r in records], with_ce=True)
    print(f"[HINT-RAG] {split}: " + format_metrics(metrics), flush=True)
    output = output or os.path.join(cfg.output_dir, f"{split}_results")
    ensure_dir(output)
    with open(os.path.join(output, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    with open(os.path.join(output, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics, records
