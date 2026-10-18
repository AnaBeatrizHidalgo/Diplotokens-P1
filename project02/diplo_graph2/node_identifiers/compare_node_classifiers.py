"""Compare the LR predictions and NGNI on their respective test references.

Run from project02 with:
    python -m diplo_graph2.node_identifiers.compare_node_classifiers
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, cast

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR.parent))

from project02.models.classificar_nos.step2_weak_labels import label_case
from project02.models.pre_processamento.tokenizer import PREPROC_ID, process_case

from diplo_graph2.node_identifiers import ngni_tuning

PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
LR_PREDICTIONS = PROCESSED_DIR / "test_predictions.csv"
WEAK_DICTIONARY = PROCESSED_DIR / "weak_labels" / "dictionary.json"
NGNI_DIR = PROCESSED_DIR / "ngni_train-data"
NGNI_TRAIN = NGNI_DIR / "train_cases_labeled.csv"
NGNI_TEST = NGNI_DIR / "test_cases_labeled.csv"
NGNI_PARAMS = NGNI_DIR / "best_params.json"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "reports" / "node_classification_metrics"


def _entity_key(case_id: str, start: int, end: int, label: str):
    return case_id, int(start), int(end), label


def _score(gold_by_case: dict[str, set], pred_by_case: dict[str, set],
           model: str, reference: str, excluded_labels=()) -> tuple[dict, list[dict]]:
    excluded = set(excluded_labels)
    gold = {entity for entities in gold_by_case.values() for entity in entities
            if entity[3] not in excluded}
    pred = {entity for entities in pred_by_case.values() for entity in entities
            if entity[3] not in excluded}
    labels = sorted({entity[3] for entity in gold | pred})
    per_label = []

    for label in labels:
        gold_label = {entity for entity in gold if entity[3] == label}
        pred_label = {entity for entity in pred if entity[3] == label}
        tp = len(gold_label & pred_label)
        fp = len(pred_label - gold_label)
        fn = len(gold_label - pred_label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label.append({
            "model": model, "reference": reference, "label": label,
            "support": len(gold_label), "predicted": len(pred_label),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        })

    tp = sum(row["tp"] for row in per_label)
    fp = sum(row["fp"] for row in per_label)
    fn = sum(row["fn"] for row in per_label)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    support = sum(row["support"] for row in per_label)
    macro_precision = sum(row["precision"] for row in per_label) / len(per_label) if labels else 0.0
    macro_recall = sum(row["recall"] for row in per_label) / len(per_label) if labels else 0.0
    macro_f1 = sum(row["f1"] for row in per_label) / len(per_label) if labels else 0.0
    weighted_f1 = (sum(row["f1"] * row["support"] for row in per_label) / support
                   if support else 0.0)
    summary = {
        "model": model, "reference": reference, "cases": len(gold_by_case),
        "gold_nodes": len(gold), "predicted_nodes": len(pred),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "macro_precision": macro_precision, "macro_recall": macro_recall,
        "macro_f1": macro_f1, "weighted_f1": weighted_f1,
    }
    return summary, per_label


def _load_lr_data(dictionary_path: Path):
    predictions = pd.read_csv(LR_PREDICTIONS).fillna({"entities": "[]"})
    with dictionary_path.open(encoding="utf-8") as file:
        dictionary_data = json.load(file)
    if dictionary_data.get("preproc_id") != PREPROC_ID:
        raise ValueError("O dicionario weak labels usa uma versao diferente do tokenizador.")

    from project02.models.classificar_nos.step2_weak_labels import WeakDictionary

    dictionary = WeakDictionary.load(dictionary_path)
    gold_by_case, pred_by_case = {}, {}
    for row in predictions.to_dict("records"):
        case_id = str(row["case_id"])
        case_text = str(row["case_text"])
        article_id = str(row["article_id"])
        weak_record = label_case(process_case(case_id, article_id, case_text), dictionary)
        gold_by_case[case_id] = {
            _entity_key(case_id, span["token_start"], span["token_end"] + 1, span["label"])
            for span in weak_record["spans"] if span["label"] != "IGNORE"
        }
        entities = row["entities"]
        if isinstance(entities, str):
            entities = json.loads(entities)
        pred_by_case[case_id] = {
            _entity_key(case_id, entity["token_start"], entity["token_end"] + 1,
                        entity["label"])
            for entity in entities
        }
    return predictions, gold_by_case, pred_by_case


def _load_ngni_data():
    train_df = ngni_tuning.load_labeled(NGNI_TRAIN)
    test_df = pd.read_csv(NGNI_TEST).dropna(
        subset=["case_text", "case_text_tagged"]
    ).reset_index(drop=True)
    golds = []
    for row in test_df.itertuples():
        gold = ngni_tuning.build_gold(str(row.case_text), str(row.case_text_tagged))
        if gold is None:
            tagged_text = re.sub(
                r"\b(TAG-(?:PATIENT|DISEASE|SYMPTOM|DRUG|EXAME|ANATOMY|EFFECT|RESULT))/[A-Za-z]+",
                r"\1",
                str(row.case_text_tagged),
                flags=re.IGNORECASE,
            )
            gold = ngni_tuning.build_gold(str(row.case_text), tagged_text)
        if gold is None:
            raise ValueError(f"Falha ao alinhar o texto tagueado de {row.case_id}.")
        golds.append(gold)
    test_df["gold"] = golds
    params = dict(ngni_tuning.BASELINE)
    if NGNI_PARAMS.exists():
        params.update(json.loads(NGNI_PARAMS.read_text(encoding="utf-8")))

    background = ngni_tuning.ngni.brown_background()
    train_model, reverse_model = ngni_tuning.train_models(
        train_df, background=background
    )
    with ngni_tuning._PosTagCache():
        identifier = ngni_tuning.ngni.NGramsNodeIdentifier(
            train_model, model_rev=reverse_model, quiet=True, **cast(Any, params)
        )
        pairs = []
        for text, gold in zip(test_df.case_text, test_df.gold):
            predicted = {
                (node.pos_in_case_text, len(node.label.split()), node.node_type)
                for node in identifier.identify_nodes(str(text))
                if node.node_type != "TAG-PATIENT"
            }
            gold_nodes = {node for node in gold if node[2] != "TAG-PATIENT"}
            pairs.append((predicted, gold_nodes))

    gold_by_case, pred_by_case = {}, {}
    for case_id, (predicted, gold) in zip(test_df.case_id.astype(str), pairs):
        gold_by_case[case_id] = {
            _entity_key(case_id, start, start + length, tag.removeprefix("TAG-"))
            for start, length, tag in gold
        }
        pred_by_case[case_id] = {
            _entity_key(case_id, start, start + length, tag.removeprefix("TAG-"))
            for start, length, tag in predicted
        }
    return test_df, gold_by_case, pred_by_case


def _validate_test_sets(lr_df: pd.DataFrame, ngni_df: pd.DataFrame):
    lr_ids = set(lr_df.case_id.astype(str))
    ngni_ids = set(ngni_df.case_id.astype(str))
    if lr_ids != ngni_ids:
        raise ValueError(
            "Os testes do LR e NGNI nao contem os mesmos case_id. "
            f"Somente LR: {sorted(lr_ids - ngni_ids)}; somente NGNI: {sorted(ngni_ids - lr_ids)}"
        )
    lr_text = dict(zip(lr_df.case_id.astype(str), lr_df.case_text.astype(str)))
    ngni_text = dict(zip(ngni_df.case_id.astype(str), ngni_df.case_text.astype(str)))
    mismatched = sorted(case_id for case_id in lr_ids if lr_text[case_id] != ngni_text[case_id])
    if mismatched:
        raise ValueError(f"O texto de teste diverge entre os modelos: {mismatched}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dictionary", type=Path, default=WEAK_DICTIONARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    lr_df, lr_gold, lr_pred = _load_lr_data(args.dictionary)
    ngni_df, ngni_gold, ngni_pred = _load_ngni_data()
    _validate_test_sets(lr_df, ngni_df)

    summaries, details = [], []
    summary, per_label = _score(lr_gold, lr_pred, "LR", "weak_labels")
    summaries.append(summary)
    details.extend(per_label)

    common_summary, common_labels = _score(
        lr_gold, lr_pred, "LR_common_labels", "weak_labels", excluded_labels=("Patient",)
    )
    summaries.append(common_summary)
    details.extend(common_labels)

    ngni_summary, ngni_labels = _score(ngni_gold, ngni_pred, "NGNI", "test_cases_labeled")
    summaries.append(ngni_summary)
    details.extend(ngni_labels)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.csv"
    per_label_path = args.output_dir / "per_label.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    pd.DataFrame(details).to_csv(per_label_path, index=False)

    print(pd.DataFrame(summaries).to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nMetricas salvas em:\n- {summary_path}\n- {per_label_path}")


if __name__ == "__main__":
    main()