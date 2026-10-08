import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hintrag.concepts import CHEXPERT_CONCEPTS, CHEXPERT_CSV_COLUMNS, label_report
from hintrag.data import load_annotation, normalize_report


def read_chexpert_csv(path, uncertain):
    table = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vector = []
            for concept in CHEXPERT_CONCEPTS:
                raw = row.get(CHEXPERT_CSV_COLUMNS[concept], "")
                value = float(raw) if raw not in ("", None) else 0.0
                if value == -1.0:
                    value = 1.0 if uncertain == "positive" else 0.0
                vector.append(1.0 if value == 1.0 else 0.0)
            table[str(int(float(row["study_id"])))] = vector
    return table


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build CheXpert-14 labels for every annotation entry")
    parser.add_argument("--ann", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--chexpert_csv", default=None)
    parser.add_argument("--uncertain", choices=["positive", "negative"], default="positive")
    args = parser.parse_args(argv)
    annotation = load_annotation(args.ann)
    study_labels = read_chexpert_csv(args.chexpert_csv, args.uncertain) if args.chexpert_csv else {}
    labels = {}
    from_csv = 0
    for split in annotation.values():
        for item in split:
            study = str(item.get("study_id", ""))
            if study in study_labels:
                vector = study_labels[study]
                if not any(vector[1:13]):
                    vector = [1.0] + vector[1:]
                from_csv += 1
            else:
                vector = label_report(normalize_report(item["report"])).tolist()
            labels[str(item["id"])] = vector
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(labels, f)
    print(f"wrote {len(labels)} label vectors to {args.out} ({from_csv} from CheXpert csv, {len(labels) - from_csv} rule-based)")


if __name__ == "__main__":
    main()
