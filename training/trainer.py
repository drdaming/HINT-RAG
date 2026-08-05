import os
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from data.iuxray_dataset import IUXRayDataset, IUXRayCollator
from data.memory_bank import MemoryBank
from models.hintrag import HINTRAGModel
from training.losses import HINTRAGLoss

class Trainer:

    def __init__(
        self,
        config,
        model: HINTRAGModel,
        memory_bank: MemoryBank,
        train_dataset: IUXRayDataset,
        val_dataset: IUXRayDataset,
        tokenizer,
        device: str = "cuda",
    ):
        self.config = config
        self.model = model
        self.memory_bank = memory_bank
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.device = device

        tc = config.training
        self.batch_size = tc.batch_size
        self.grad_accum_steps = tc.grad_accum_steps
        self.num_epochs = tc.num_epochs
        self.max_grad_norm = tc.max_grad_norm
        self.faiss_refresh_epochs = tc.faiss_refresh_epochs
        self.save_every_n_epochs = tc.save_every_n_epochs
        self.momentum = tc.momentum
        self.checkpoint_dir = config.data.checkpoint_dir

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.criterion = HINTRAGLoss(
            n_visual=model.n_visual_tokens,
            K=14,
            lambda_nce=tc.lambda_nce,
            lambda_ent=tc.lambda_ent,
            gamma=tc.gamma,
            beta=tc.beta,
            tau_c=tc.tau_c,
            pad_token_id=-100,
        )

        collator = IUXRayCollator()
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=config.dataset.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=config.dataset.num_workers,
            pin_memory=True,
        )

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"[Trainer] Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        self.optimizer = AdamW(
            trainable_params,
            lr=tc.learning_rate,
            weight_decay=tc.weight_decay,
        )

        steps_per_epoch = math.ceil(len(train_dataset) / (self.batch_size * self.grad_accum_steps))
        total_steps = steps_per_epoch * self.num_epochs
        self.total_optimizer_steps = total_steps
        self.optimizer_step_count = 0

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=tc.warmup_steps,
            num_training_steps=total_steps,
        )

        self.use_amp = (tc.mixed_precision == "bf16") and torch.cuda.is_bf16_supported()
        self.scaler = None
        if self.use_amp:
            print("[Trainer] Using BF16 mixed precision.")

        self.start_epoch = 1
        self._memory_bank_initialized = False

    def train(self, text_encoder, text_tokenizer) -> None:
        if not self._memory_bank_initialized:
            self.memory_bank.initialize_all(
                self.train_dataset,
                self.model,
                text_encoder,
                text_tokenizer,
                batch_size=self.batch_size * 2,
            )
            self._memory_bank_initialized = True

        best_val_loss = float("inf")

        for epoch in range(self.start_epoch, self.num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.num_epochs}")
            print(f"{'='*60}")

            train_metrics = self._train_epoch(epoch)
            val_metrics = self._val_epoch(epoch)

            print(
                f"[Epoch {epoch}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"(gen={train_metrics['loss_gen']:.4f}, "
                f"nce={train_metrics['loss_nce']:.4f}, "
                f"ent={train_metrics['loss_ent']:.4f}) | "
                f"val_loss={val_metrics['loss']:.4f}"
            )

            if epoch % self.save_every_n_epochs == 0:
                self._save_checkpoint(epoch, val_metrics["loss"])

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                self._save_checkpoint(epoch, val_metrics["loss"], name="best_model")

            if epoch % self.faiss_refresh_epochs == 0:
                print(f"[Trainer] Rebuilding FAISS index after epoch {epoch} ...")
                self.memory_bank.build_faiss_index()

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        total_loss = total_gen = total_nce = total_ent = 0.0
        n_steps = 0

        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            pixel_values = batch["pixel_values"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            gt_labels = batch["pathology_labels"].to(self.device)
            sample_indices = batch["sample_indices"].numpy()

            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if self.use_amp
                else torch.no_grad.__class__()
            )
            if self.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = self.model(pixel_values, input_ids, attention_mask, labels)
            else:
                out = self.model(pixel_values, input_ids, attention_mask, labels)

            loss_dict = self.criterion(
                logits=out["logits"],
                labels=labels,
                alpha_pri=out["alpha_pri"],
                alpha_sec=out["alpha_sec"],
                sample_idx_pri=out["sample_idx_pri"],
                sample_idx_sec=out["sample_idx_sec"],
                P_diag_post=out["P_diag_post"],
                S_pre=out["S"],
                gt_labels=gt_labels,
                memory_labels=self.memory_bank.label_bank,
            )

            loss = loss_dict["loss"] / self.grad_accum_steps
            loss.backward()

            total_loss += loss_dict["loss"].item()
            total_gen += loss_dict["loss_gen"].item()
            total_nce += loss_dict["loss_nce"].item()
            total_ent += loss_dict["loss_ent"].item()
            n_steps += 1

            if (step + 1) % self.grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.optimizer_step_count += 1
                self.model.gumbel_selector.anneal_temperature(
                    self.optimizer_step_count,
                    self.total_optimizer_steps,
                )

            with torch.no_grad():
                h_state_np = out["h_state"].detach().float().cpu().numpy()
                new_text = self.memory_bank.text_bank[sample_indices]
                self.memory_bank.update(sample_indices, h_state_np, new_text)

        return {
            "loss": total_loss / n_steps,
            "loss_gen": total_gen / n_steps,
            "loss_nce": total_nce / n_steps,
            "loss_ent": total_ent / n_steps,
        }

    def _val_epoch(self, epoch: int) -> dict:
        self.model.eval()
        total_loss = 0.0
        n_steps = 0

        with torch.no_grad():
            for batch in self.val_loader:
                pixel_values = batch["pixel_values"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                gt_labels = batch["pathology_labels"].to(self.device)

                if self.use_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        out = self.model(pixel_values, input_ids, attention_mask)
                else:
                    out = self.model(pixel_values, input_ids, attention_mask)

                loss_dict = self.criterion(
                    logits=out["logits"],
                    labels=labels,
                    alpha_pri=out["alpha_pri"],
                    alpha_sec=out["alpha_sec"],
                    sample_idx_pri=out["sample_idx_pri"],
                    sample_idx_sec=out["sample_idx_sec"],
                    P_diag_post=out["P_diag_post"],
                    S_pre=out["S"],
                    gt_labels=gt_labels,
                    memory_labels=self.memory_bank.label_bank,
                )

                total_loss += loss_dict["loss"].item()
                n_steps += 1

        return {"loss": total_loss / max(n_steps, 1)}

    def _save_checkpoint(self, epoch: int, val_loss: float, name: Optional[str] = None) -> None:
        ckpt_name = name or f"epoch_{epoch:03d}"
        ckpt_path = os.path.join(self.checkpoint_dir, ckpt_name)
        os.makedirs(ckpt_path, exist_ok=True)

        try:
            self.model.split_llm.first_layers[0]
            torch.save(
                {k: v for k, v in self.model.state_dict().items()},
                os.path.join(ckpt_path, "model_state.pt"),
            )
        except Exception:
            torch.save(self.model.state_dict(), os.path.join(ckpt_path, "model_state.pt"))

        self.memory_bank.save(os.path.join(ckpt_path, "memory_bank"))

        torch.save(
            {
                "epoch": epoch,
                "val_loss": val_loss,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "optimizer_step_count": self.optimizer_step_count,
                "tau_g": self.model.gumbel_selector.tau_g,
            },
            os.path.join(ckpt_path, "train_state.pt"),
        )

        print(f"[Trainer] Checkpoint saved → {ckpt_path}  (val_loss={val_loss:.4f})")

    def load_checkpoint(self, ckpt_path: str) -> None:
        model_path = os.path.join(ckpt_path, "model_state.pt")
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state, strict=False)

        self.memory_bank.load(os.path.join(ckpt_path, "memory_bank"))

        train_state_path = os.path.join(ckpt_path, "train_state.pt")
        if os.path.isfile(train_state_path):
            ts = torch.load(train_state_path, map_location="cpu")
            self.optimizer.load_state_dict(ts["optimizer"])
            self.scheduler.load_state_dict(ts["scheduler"])
            self.optimizer_step_count = ts.get("optimizer_step_count", 0)
            self.model.gumbel_selector.tau_g = ts.get("tau_g", self.model.gumbel_selector.tau_g)
            saved_epoch = ts.get("epoch", 0)
            self.start_epoch = saved_epoch + 1
            print(f"[Trainer] Resuming from epoch {self.start_epoch}")

        self._memory_bank_initialized = True
        print(f"[Trainer] Loaded checkpoint from {ckpt_path}")
