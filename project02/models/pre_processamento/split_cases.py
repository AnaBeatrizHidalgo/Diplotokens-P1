"""
split_cases.py -- separa os casos em TREINO e TESTE, agrupando por article_id.

Por que agrupar por artigo: casos do mesmo artigo sao muito parecidos. Se um fosse
para o treino e outro para o teste, a avaliacao ficaria otimista (vazamento).

Entrada : data/raw/sample/cases.csv
Saida   : data/processed/splits/train_cases.csv, test_cases.csv e split_manifest.csv
          (manifest = case_id, article_id, split -> para reproduzir/auditar)

Uso pela linha de comando (de qualquer pasta):

    python split_cases.py
    python split_cases.py --test-size 0.25 --seed 7

Uso em Python / notebook:

    from project02.models.pre_processamento.split_cases import split_cases
    train, test = split_cases()

O conjunto de TESTE e o que voce deve anotar a mao (gabarito) para medir o modelo:
a rotulacao fraca so e aplicada no treino.
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Raiz do projeto (project02), deduzida da posicao deste arquivo:
# project02/models/pre_processamento/split_cases.py -> parents[2] == project02
PROJECT_DIR = Path(__file__).resolve().parents[2]

# Valores padrao (antes vinham do load_config)
DEFAULT_CASES_CSV = PROJECT_DIR / "data" / "raw" / "sample" / "cases.csv"
DEFAULT_OUT_DIR = PROJECT_DIR / "data" / "processed" / "splits"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_SEED = 42
DEFAULT_GROUP_COL = "article_id"
TRAIN_FILE = "train_cases.csv"
TEST_FILE = "test_cases.csv"
MANIFEST_FILE = "split_manifest.csv"


def split_cases(
    cases_csv=DEFAULT_CASES_CSV,
    out_dir=DEFAULT_OUT_DIR,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SEED,
    group_col: str = DEFAULT_GROUP_COL,
    save: bool = True,
):
    """
        Separa os casos em treino e teste sem repartir nenhum artigo entre os dois.
        Retorna (train, test) como DataFrames. Se save=True, grava os CSVs em out_dir.
    """
    df = pd.read_csv(cases_csv)
    if group_col not in df.columns:
        raise KeyError(f"coluna '{group_col}' nao existe em {cases_csv}. Colunas: {list(df.columns)}")

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=test_size,
                                    random_state=seed).split(df, groups=df[group_col]))
    train, test = df.iloc[tr], df.iloc[te]
    assert not set(train[group_col]) & set(test[group_col]), "artigo aparece em treino e teste!"

    if save:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        train.to_csv(out / TRAIN_FILE, index=False)
        test.to_csv(out / TEST_FILE, index=False)
        manifest_cols = [c for c in ("case_id", group_col) if c in df.columns] + ["split"]
        pd.concat([train.assign(split="train"), test.assign(split="test")])[
            manifest_cols].to_csv(out / MANIFEST_FILE, index=False)

    return train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(DEFAULT_CASES_CSV))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--group-col", default=DEFAULT_GROUP_COL)
    a = ap.parse_args()

    train, test = split_cases(a.cases, a.out_dir, a.test_size, a.seed, a.group_col)
    g = a.group_col
    print(f"treino: {len(train)} casos / {train[g].nunique()} artigos")
    print(f"teste : {len(test)} casos / {test[g].nunique()} artigos")
    print(f"-> {a.out_dir}")


if __name__ == "__main__":
    main()
