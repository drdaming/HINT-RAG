import contextlib
import io
import math
import re

import numpy as np

from .concepts import CHEXPERT_CONCEPTS, label_report


def tokenize_for_metric(text):
    text = str(text).lower().replace("\n", " ")
    text = re.sub(r"[^a-z0-9.\s]", " ", text)
    text = text.replace(".", " . ")
    return re.sub(r"\s+", " ", text).strip()


def _meteor(gts, res):
    try:
        from pycocoevalcap.meteor.meteor import Meteor

        with contextlib.redirect_stdout(io.StringIO()):
            score, _ = Meteor().compute_score(gts, res)
        return float(score)
    except Exception:
        pass
    try:
        from nltk.translate.meteor_score import meteor_score

        scores = [meteor_score([gts[k][0].split()], res[k][0].split()) for k in gts]
        return float(np.mean(scores))
    except Exception:
        return float("nan")


def compute_nlg_metrics(references, hypotheses):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge

    gts = {i: [tokenize_for_metric(r) or "empty"] for i, r in enumerate(references)}
    res = {i: [tokenize_for_metric(h) or "empty"] for i, h in enumerate(hypotheses)}
    with contextlib.redirect_stdout(io.StringIO()):
        bleu, _ = Bleu(4).compute_score(gts, res)
        rouge, _ = Rouge().compute_score(gts, res)
    return {
        "BLEU_1": float(bleu[0]),
        "BLEU_2": float(bleu[1]),
        "BLEU_3": float(bleu[2]),
        "BLEU_4": float(bleu[3]),
        "METEOR": _meteor(gts, res),
        "ROUGE_L": float(rouge),
    }


def compute_ce_metrics(references, hypotheses, concepts=CHEXPERT_CONCEPTS):
    ref = np.stack([label_report(r, concepts) for r in references])
    hyp = np.stack([label_report(h, concepts) for h in hypotheses])
    tp = float((ref * hyp).sum())
    fp = float(((1 - ref) * hyp).sum())
    fn = float((ref * (1 - hyp)).sum())
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"CE_P": precision, "CE_R": recall, "CE_F1": f1}


def compute_metrics(references, hypotheses, with_ce=True):
    metrics = compute_nlg_metrics(references, hypotheses)
    if with_ce:
        metrics.update(compute_ce_metrics(references, hypotheses))
    return metrics


def format_metrics(metrics):
    return "  ".join(f"{k}={v:.4f}" if isinstance(v, float) and not math.isnan(v) else f"{k}={v}" for k, v in metrics.items())
