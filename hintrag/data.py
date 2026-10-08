import json
import os
import re

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .concepts import CHEXPERT_CONCEPTS, label_report


def normalize_report(text):
    text = str(text).replace("\n", " ").lower()
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_annotation(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_label_file(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): np.asarray(v, dtype=np.float32) for k, v in data.items()}


def build_transform(image_size, mean, std):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def build_prompt_ids(tokenizer, prompt_prefix, prompt_suffix):
    prefix = tokenizer(prompt_prefix, add_special_tokens=False)["input_ids"]
    if tokenizer.bos_token_id is not None:
        prefix = [tokenizer.bos_token_id] + prefix
    suffix = tokenizer(prompt_suffix, add_special_tokens=False)["input_ids"]
    return prefix, suffix


class ReportDataset(Dataset):
    def __init__(self, data_cfg, split, tokenizer, transform, concepts=CHEXPERT_CONCEPTS):
        annotation = load_annotation(data_cfg.ann_path)
        self.items = annotation[split]
        self.split = split
        self.image_dir = data_cfg.image_dir
        self.max_views = int(data_cfg.max_views)
        self.max_tokens = int(data_cfg.max_report_tokens)
        self.tokenizer = tokenizer
        self.transform = transform
        self.reports = [normalize_report(item["report"]) for item in self.items]
        label_map = load_label_file(data_cfg.label_path) if data_cfg.get("label_path") else {}
        labels = []
        for item, report in zip(self.items, self.reports):
            key = str(item["id"])
            if key in label_map:
                y = label_map[key]
            elif "labels" in item:
                y = np.asarray(item["labels"], dtype=np.float32)
            else:
                y = label_report(report, concepts)
            labels.append((np.asarray(y, dtype=np.float32) > 0).astype(np.float32))
        self.labels = torch.from_numpy(np.stack(labels)) if labels else torch.zeros(0, len(concepts))

    def __len__(self):
        return len(self.items)

    def _image_paths(self, item):
        paths = item["image_path"]
        if isinstance(paths, str):
            paths = [paths]
        return list(paths)[: self.max_views]

    def __getitem__(self, index):
        item = self.items[index]
        images = []
        for path in self._image_paths(item):
            with Image.open(os.path.join(self.image_dir, path)) as img:
                images.append(self.transform(img.convert("RGB")))
        while len(images) < self.max_views:
            images.append(images[-1])
        report_ids = self.tokenizer(self.reports[index], add_special_tokens=False)["input_ids"][: self.max_tokens]
        return {
            "index": index,
            "id": str(item["id"]),
            "pixel_values": torch.stack(images),
            "report_ids": report_ids,
            "report": self.reports[index],
            "labels": self.labels[index],
        }


class ReportCollator:
    def __init__(self, suffix_ids, pad_token_id, eos_token_id):
        self.suffix_ids = list(suffix_ids)
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    def __call__(self, batch):
        sequences, targets = [], []
        for sample in batch:
            report = list(sample["report_ids"]) + [self.eos_token_id]
            sequences.append(self.suffix_ids + report)
            targets.append([-100] * len(self.suffix_ids) + report)
        length = max(len(s) for s in sequences)
        size = len(batch)
        text_ids = torch.full((size, length), self.pad_token_id, dtype=torch.long)
        text_mask = torch.zeros(size, length, dtype=torch.long)
        target_ids = torch.full((size, length), -100, dtype=torch.long)
        for i, (seq, tgt) in enumerate(zip(sequences, targets)):
            text_ids[i, : len(seq)] = torch.tensor(seq)
            text_mask[i, : len(seq)] = 1
            target_ids[i, : len(tgt)] = torch.tensor(tgt)
        return {
            "index": torch.tensor([s["index"] for s in batch], dtype=torch.long),
            "id": [s["id"] for s in batch],
            "pixel_values": torch.stack([s["pixel_values"] for s in batch]),
            "text_ids": text_ids,
            "text_mask": text_mask,
            "target_ids": target_ids,
            "labels": torch.stack([s["labels"] for s in batch]),
            "report": [s["report"] for s in batch],
        }
