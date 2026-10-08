import re

import numpy as np

CHEXPERT_CONCEPTS = [
    "no finding",
    "enlarged cardiomediastinum",
    "cardiomegaly",
    "lung opacity",
    "lung lesion",
    "edema",
    "consolidation",
    "pneumonia",
    "atelectasis",
    "pneumothorax",
    "pleural effusion",
    "pleural other",
    "fracture",
    "support devices",
]

CHEXPERT_CSV_COLUMNS = {
    "no finding": "No Finding",
    "enlarged cardiomediastinum": "Enlarged Cardiomediastinum",
    "cardiomegaly": "Cardiomegaly",
    "lung opacity": "Lung Opacity",
    "lung lesion": "Lung Lesion",
    "edema": "Edema",
    "consolidation": "Consolidation",
    "pneumonia": "Pneumonia",
    "atelectasis": "Atelectasis",
    "pneumothorax": "Pneumothorax",
    "pleural effusion": "Pleural Effusion",
    "pleural other": "Pleural Other",
    "fracture": "Fracture",
    "support devices": "Support Devices",
}

_SURFACE_FORMS = {
    "enlarged cardiomediastinum": [
        r"enlarged cardiomediastinum",
        r"widen(?:ed|ing) (?:of the )?mediastin\w*",
        r"mediastinal widening",
        r"enlarged mediastin\w*",
        r"mediastinal enlargement",
    ],
    "cardiomegaly": [
        r"cardiomegaly",
        r"enlarged heart",
        r"heart (?:size )?is enlarged",
        r"cardiac enlargement",
        r"enlarged cardiac silhouette",
        r"cardiac silhouette is enlarged",
        r"enlargement of the cardiac silhouette",
    ],
    "lung opacity": [
        r"opacit(?:y|ies)",
        r"opacification",
        r"infiltrates?",
        r"air ?space disease",
        r"haziness",
    ],
    "lung lesion": [r"nodules?", r"nodular densit(?:y|ies)", r"mass(?:es)?", r"lesions?"],
    "edema": [r"o?edema", r"vascular congestion", r"pulmonary congestion"],
    "consolidation": [r"consolidat(?:ion|ions|ive)"],
    "pneumonia": [r"pneumonias?", r"pneumonic"],
    "atelectasis": [r"atelecta(?:sis|ses|tic)"],
    "pneumothorax": [r"pneumothora(?:x|ces)"],
    "pleural effusion": [r"effusions?", r"pleural fluid"],
    "pleural other": [r"pleural thickening", r"pleural scarring", r"pleural plaques?", r"fibrothorax"],
    "fracture": [r"fractur(?:e|es|ed)"],
    "support devices": [
        r"pacemakers?",
        r"catheters?",
        r"tubes?",
        r"picc",
        r"sternotomy wires?",
        r"wires?",
        r"devices?",
        r"stents?",
        r"surgical clips?",
        r"port-?a-?cath",
    ],
}

_NEG_BEFORE = re.compile(
    r"\b(?:no|not|without|negative for|free of|clear of|absence of|absent|resolution of|"
    r"resolved|rule out|ruled out|exclude|excluded|neither|nor|removal of)\b"
)
_NEG_AFTER = re.compile(
    r"\b(?:resolved|removed|not seen|not identified|not present|not visualized|is absent|are absent|"
    r"has cleared|have cleared|ruled out|excluded)\b"
)
_NOT_NEGATION = re.compile(r"\bno (?:significant |interval |appreciable )?change\b|\bnot significantly changed\b")
_CLAUSE_SPLIT = re.compile(r"\bbut\b|\bhowever\b|\balthough\b|\bexcept\b|\bwhich\b")
_SENTENCE_SPLIT = re.compile(r"[.;\n]")
_COMPILED = {k: [re.compile(r"\b" + p + r"\b") for p in v] for k, v in _SURFACE_FORMS.items()}


def _mention_positive(clause, pattern):
    for match in pattern.finditer(clause):
        before = clause[: match.start()]
        after = clause[match.end():]
        if _NEG_BEFORE.search(before) or _NEG_AFTER.search(after):
            continue
        return True
    return False


def label_report(text, concepts=CHEXPERT_CONCEPTS):
    text = _NOT_NEGATION.sub(" ", text.lower())
    clauses = []
    for sentence in _SENTENCE_SPLIT.split(text):
        clauses.extend(c for c in _CLAUSE_SPLIT.split(sentence) if c.strip())
    found = set()
    for concept, patterns in _COMPILED.items():
        if any(_mention_positive(clause, p) for clause in clauses for p in patterns):
            found.add(concept)
    pathologies = found - {"support devices"}
    if not pathologies:
        found.add("no finding")
    return np.asarray([1.0 if c in found else 0.0 for c in concepts], dtype=np.float32)
