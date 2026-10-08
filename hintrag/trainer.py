import json
import math
import os
import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import ReportCollator, normalize_report
from .losses import HINTRAGLoss
from .memory_bank import MemoryBank
from .metrics import compute_metrics, format_metrics
from .utils import autocast_context, ensure_dir, to_device


def trainable_state_dict(model):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    state = model.state_dict()
    return {k: v.detach().cpu() for k, v in state.items() if k in names or k.endswith("routing.tau_g")}


def load_trainable_state_dict(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    absent = sorted(expected - set(state))
    if absent or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={absent[:5]} unexpected={list(unexpected)[:5]}")


def bank_dtype(cfg, device):
    name = cfg.memory.get("dtype", "float32")
    if device.type != "cuda":
        name = "float32"
    return getattr(torch, name)


def build_memory_bank(cfg, model, dataset, values, collator, device, use_bf16):
    bank = MemoryBank(
        len(dataset),
        model.llm.hidden_size,
        values.size(1),
        model.probe.num_concepts,
        momentum=cfg.memory.momentum,
        device=device,
        dtype=bank_dtype(cfg, device),
        search=cfg.memory.search,
        faiss_nlist=cfg.memory.faiss_nlist,
        faiss_nprobe=cfg.memory.faiss_nprobe,
        seed=cfg.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=cfg.data.num_workers,
    )
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in loader:
            with autocast_context(device, use_bf16):
                h = model.state_vector(batch["pixel_values"].to(device))
            bank.write(batch["index"], keys=h, values=values[batch["index"]], labels=batch["labels"])
    bank.finalize()
    model.train(was_training)
    return bank


@torch.no_grad()
def generate_reports(model, dataset, tokenizer, cfg, device, use_bf16, limit=None):
    collator = ReportCollator(model.suffix_ids.tolist(), tokenizer.pad_token_id, tokenizer.eos_token_id)
    loader = DataLoader(dataset, batch_size=cfg.eval.batch_size, shuffle=False, collate_fn=collator, num_workers=cfg.data.num_workers)
    records = []
    for batch in loader:
        with autocast_context(device, use_bf16):
            sequences = model.generate(
                batch["pixel_values"].to(device),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=cfg.eval.max_new_tokens,
                num_beams=cfg.eval.num_beams,
                no_repeat_ngram_size=cfg.eval.no_repeat_ngram_size,
                length_penalty=cfg.eval.get("length_penalty", 1.0),
            )
        for sample_id, reference, seq in zip(batch["id"], batch["report"], sequences.tolist()):
            if tokenizer.eos_token_id in seq:
                seq = seq[: seq.index(tokenizer.eos_token_id)]
            prediction = normalize_report(tokenizer.decode(seq, skip_special_tokens=True))
            records.append({"id": sample_id, "reference": reference, "prediction": prediction})
        if limit is not None and len(records) >= limit:
            break
    return records


class Trainer:
    def __init__(self, cfg, model, tokenizer, train_set, val_set, device):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.train_set = train_set
        self.val_set = val_set
        self.device = device
        train_cfg = cfg.train
        self.accum = int(train_cfg.grad_accum_steps)
        self.epochs = int(train_cfg.epochs)
        self.output_dir = ensure_dir(cfg.output_dir)
        self.use_bf16 = bool(train_cfg.bf16) and device.type == "cuda" and torch.cuda.is_bf16_supported()
        self.collator = ReportCollator(model.suffix_ids.tolist(), tokenizer.pad_token_id, tokenizer.eos_token_id)
        self.train_loader = DataLoader(
            train_set,
            batch_size=train_cfg.batch_size,
            shuffle=True,
            collate_fn=self.collator,
            num_workers=cfg.data.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=len(train_set) >= train_cfg.batch_size,
        )
        self.val_loader = DataLoader(
            val_set,
            batch_size=cfg.eval.batch_size,
            shuffle=False,
            collate_fn=self.collator,
            num_workers=cfg.data.num_workers,
        )
        self.criterion = HINTRAGLoss(
            lambda_nce=train_cfg.lambda_nce,
            lambda_ent=train_cfg.lambda_ent,
            gamma=train_cfg.gamma,
            beta=train_cfg.beta,
            tau_c=train_cfg.tau_c,
            kl_smoothing=train_cfg.kl_smoothing,
        )
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
        steps_per_epoch = max(1, math.ceil(len(self.train_loader) / self.accum))
        self.total_steps = steps_per_epoch * self.epochs
        from transformers import get_linear_schedule_with_warmup

        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, train_cfg.warmup_steps, self.total_steps)
        self.global_step = 0
        self.start_epoch = 1
        self.best_score = None
        self.history = []

    def setup_memory_bank(self, report_values):
        bank = build_memory_bank(self.cfg, self.model, self.train_set, report_values, self.collator, self.device, self.use_bf16)
        self.model.attach_memory_bank(bank)
        return bank

    @property
    def bank(self):
        return self.model.memory_bank

    def contrastive_keys(self, out, labels, self_indices):
        if "q_pri" not in out or self.criterion.lambda_nce <= 0:
            return None, None
        prob = out["P_diag"].detach().float()
        y = labels.bool().to(prob.device)
        primary = prob.masked_fill(~y, -1.0).argmax(dim=-1)
        primary = torch.where(y.any(dim=-1), primary, prob.argmax(dim=-1))
        confusing = prob.masked_fill(y, -1.0).argmax(dim=-1)
        positive, negative = self.bank.sample_contrastive(primary.cpu(), confusing.cpu(), self_indices)
        return self.bank.keys[positive.to(self.bank.device)], self.bank.keys[negative.to(self.bank.device)]

    def compute_losses(self, batch, training):
        exclude = batch["index"] if training else None
        with autocast_context(self.device, self.use_bf16):
            out = self.model(batch["pixel_values"], batch["text_ids"], batch["text_mask"], exclude=exclude)
        positive, negative = self.contrastive_keys(out, batch["labels"], batch["index"].tolist() if training else None)
        return out, self.criterion(out, batch["target_ids"], batch["labels"], positive, negative)

    def train_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        totals = defaultdict(float)
        count = 0
        log_every = int(self.cfg.train.get("log_every", 50))
        start = time.time()
        for step, batch in enumerate(self.train_loader):
            batch = to_device(batch, self.device)
            out, losses = self.compute_losses(batch, training=True)
            (losses["loss"] / self.accum).backward()
            self.bank.momentum_update(batch["index"], out["h"].detach())
            if (step + 1) % self.accum == 0 or step + 1 == len(self.train_loader):
                nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], self.cfg.train.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
                self.model.routing.anneal(self.global_step, self.total_steps)
            for key, value in losses.items():
                totals[key] += float(value.detach())
            totals["gate"] += float(out["g"].detach().float().mean())
            count += 1
            if log_every > 0 and (step + 1) % log_every == 0:
                avg = {k: v / count for k, v in totals.items()}
                print(
                    f"[epoch {epoch} step {step + 1}/{len(self.train_loader)}] "
                    f"loss={avg['loss']:.4f} gen={avg['loss_gen']:.4f} nce={avg['loss_nce']:.4f} "
                    f"ent={avg['loss_ent']:.4f} g={avg['gate']:.3f} tau_g={float(self.model.routing.tau_g):.3f} "
                    f"{time.time() - start:.1f}s",
                    flush=True,
                )
        return {k: v / max(count, 1) for k, v in totals.items()}

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        totals = defaultdict(float)
        count = 0
        for batch in self.val_loader:
            batch = to_device(batch, self.device)
            _, losses = self.compute_losses(batch, training=False)
            for key, value in losses.items():
                totals[key] += float(value)
            count += 1
        result = {f"val_{k}": v / max(count, 1) for k, v in totals.items()}
        if self.cfg.train.get("select_by", "loss") != "loss":
            limit = self.cfg.train.get("val_generation_limit")
            records = generate_reports(self.model, self.val_set, self.tokenizer, self.cfg, self.device, self.use_bf16, limit)
            metrics = compute_metrics([r["reference"] for r in records], [r["prediction"] for r in records], with_ce=False)
            result.update({f"val_{k}": v for k, v in metrics.items()})
        return result

    def _score(self, result):
        key = self.cfg.train.get("select_by", "loss")
        if key == "loss":
            return -result["val_loss"]
        return result[f"val_{key}"]

    def save(self, name, epoch):
        path = ensure_dir(os.path.join(self.output_dir, name))
        torch.save(
            {
                "model": trainable_state_dict(self.model),
                "bank": self.bank.state_dict(),
                "epoch": epoch,
                "global_step": self.global_step,
                "best_score": self.best_score,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            os.path.join(path, "checkpoint.pt"),
        )
        return path

    def resume(self, path):
        state = torch.load(os.path.join(path, "checkpoint.pt"), map_location="cpu", weights_only=False)
        load_trainable_state_dict(self.model, state["model"])
        bank = MemoryBank.from_state(
            state["bank"],
            device=self.device,
            dtype=bank_dtype(self.cfg, self.device),
            search=self.cfg.memory.search,
            faiss_nlist=self.cfg.memory.faiss_nlist,
            faiss_nprobe=self.cfg.memory.faiss_nprobe,
        )
        self.model.attach_memory_bank(bank)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = int(state["global_step"])
        self.best_score = state.get("best_score")
        self.start_epoch = int(state["epoch"]) + 1

    def fit(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            train_stats = self.train_epoch(epoch)
            val_stats = self.validate()
            record = {"epoch": epoch, **train_stats, **val_stats}
            self.history.append(record)
            print(f"[epoch {epoch}] " + format_metrics({k: v for k, v in record.items() if k != "epoch"}), flush=True)
            score = self._score(val_stats)
            if self.best_score is None or score > self.best_score:
                self.best_score = score
                self.save("best", epoch)
            if epoch % int(self.cfg.train.save_every_epochs) == 0:
                self.save(f"epoch_{epoch:03d}", epoch)
            if self.cfg.memory.search == "faiss" and epoch % int(self.cfg.memory.refresh_every_epochs) == 0:
                self.bank.refresh_index()
            with open(os.path.join(self.output_dir, "history.json"), "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2)
        return self.history
