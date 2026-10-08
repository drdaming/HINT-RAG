# HINT-RAG (paper-consistent implementation)

Code for **HINT-RAG** (ACM MM 2026).

```
perceive  ->  query  ->  disambiguate  ->  decode
(h, P_diag, S, e_pri, e_sec)  (q_pri, q_sec, E_pri, E_sec)  (g, H_fused)  (report)
```

## Directory layout

```
code_upd/
├── configs/
│   ├── iu_xray.yaml          IU X-Ray
│   ├── mimic_cxr.yaml        MIMIC-CXR
├── hintrag/
│   ├── models/
│   │   ├── visual_encoder.py BiomedCLIP ViT (frozen), patch tokens
│   │   ├── split_llm.py      Vicuna split into perceive / reason stages, LoRA on q_proj, v_proj
│   │   ├── perceive.py       open-vocabulary diagnostic probe, soft hypothesis routing
│   │   ├── query.py          contrastive queries, differentiable retrieval
│   │   ├── disambiguate.py   uncertainty gate, evidence cross-attention + gated residual
│   │   └── hintrag.py        full model, training forward pass, greedy / beam decoding
│   ├── memory_bank.py        dense memory bank M = {(k_m, v_m)}, MIPS (exact or FAISS), contrastive sampling
│   ├── losses.py             L_gen, L_nce, L_ent, L_total
│   ├── data.py               R2Gen-style annotation loader, prompt construction, collator
│   ├── concepts.py           14 CheXpert concepts, negation-aware rule labeller
│   ├── metrics.py            BLEU-1..4, METEOR, ROUGE-L, CE precision / recall / F1
│   ├── trainer.py            training loop, memory-bank construction, checkpoints
│   ├── builder.py            builds tokenizer, LLM, visual encoder, ClinicalBERT
│   ├── pipeline.py           run_training / run_evaluation
│   ├── config.py, utils.py
├── tools/build_labels.py     
├── train.py
├── test.py
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

METEOR uses the Java METEOR in `pycocoevalcap` when Java is available, and otherwise falls back to NLTK.

## Data

Both datasets use the R2Gen annotation format:

```json
{"train": [{"id": "CXR2384_IM-0942", "report": "...", "image_path": ["CXR2384_IM-0942/0.png", "CXR2384_IM-0942/1.png"]}],
 "val": [...], "test": [...]}
```

* **IU X-Ray.** Use the patient-level 7:1:2 split (the standard R2Gen `annotation.json`). Two views per study (`data.max_views: 2`).
* **MIMIC-CXR.** Use the official split (R2Gen `annotation.json`, which includes `study_id`). One view (`data.max_views: 1`).

CheXpert-14 labels feed `L_nce`, `L_ent` and the memory bank. The resolution order is `data.label_path` (json `{id: [14 values]}`), then a `labels` field in the annotation, then the built-in rule labeller.

```bash
python tools/build_labels.py --ann data/mimic_cxr/annotation.json --out data/mimic_cxr/chexpert_labels.json --chexpert_csv mimic-cxr-2.0.0-chexpert.csv
python tools/build_labels.py --ann data/iu_xray/annotation.json --out data/iu_xray/labels.json
```

The concept order follows `hintrag.concepts.CHEXPERT_CONCEPTS`. Labels produced by CheXbert can be passed through `data.label_path` in the same json format.

## Training and evaluation

```bash
python train.py --config configs/iu_xray.yaml
python train.py --config configs/mimic_cxr.yaml
python train.py --config configs/iu_xray.yaml --resume outputs/iu_xray/epoch_010
python test.py  --config configs/iu_xray.yaml --checkpoint outputs/iu_xray/best
```

Any config entry can be overridden with `--opts key=value`, for example `--opts train.batch_size=4 data.image_dir=/path/to/images`.

A checkpoint stores the trainable parameters (projector, LoRA, query and retrieval modules, gate, cross-attention, `τ_g`) and the memory bank. `test.py` writes `predictions.json` and `metrics.json` (BLEU-1..4, METEOR, ROUGE-L, CE-P/R/F1) to `<output_dir>/<split>_results/`.

## Ablations

| Variant | Overrides |
|---|---|
| w/o SHR (top-1 hard hypothesis) | `model.ablation.soft_routing=false` |
| w/o CQ (primary query only) | `model.ablation.contrastive_query=false` |
| w/o DG (fixed `g = 1`) | `model.ablation.dynamic_gate=false` |
| w/o CRL+CERL (only `L_gen`) | `train.lambda_nce=0 train.lambda_ent=0` |
| non-R (no retrieval) | `model.ablation.retrieval=none` |
| SR (visual-similarity retrieval) | `model.ablation.retrieval=visual` |

```bash
python train.py --config configs/iu_xray.yaml --opts model.ablation.contrastive_query=false output_dir=outputs/iu_xray_wo_cq
```

