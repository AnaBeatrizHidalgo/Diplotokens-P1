"""
features.py -- PASSO 3: features por token (janela de contexto + dicionario MeSH) para o classificador.

Modelo: features por token -> DictVectorizer -> LogisticRegression multinomial (sem embeddings).

Uso (ver notebook, secoes 6 e 7):

    from project02.models.classificar_nos.features import (
        build_dataset, make_model, predict_case, fix_bio, bio_to_entities)

    X, y, ids = build_dataset(train_df, records, text_col="case_text")
    model = make_model().fit(X, y)

COMO O DICIONARIO ENTRA (e por que nao vira simples copia dele)
    * Os rotulos fracos vem do dicionario MeSH. Se o modelo visse "este token esta no dicionario como
      Disease" o tempo todo, ele so copiaria o dicionario e nao aprenderia a reconhecer termos fora dele.
      Por isso, no treino cada termo do dicionario e ESCONDIDO com probabilidade `dict_dropout`
      (como se nao estivesse no MeSH): o modelo aprende tambem pela forma da palavra e pelo contexto.
    * Termos AMBIGUOS (mais de um tipo no dicionario, ex.: Disease|Result) tem rotulo fraco decidido por
      uma ordem de prioridade, nao pelo contexto. Por isso eles NAO entram no treino. O modelo aprende o
      contexto de cada tipo com os termos nao ambiguos e, na predicao, escolhe o TIPO entre os candidatos
      do dicionario olhando so o contexto. O B/I vem da posicao do termo no dicionario.
    * Palavras vizinhas que estao no dicionario (e o tipo delas) tambem sao features de contexto.
"""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

# Raiz do projeto (project02), deduzida da posicao deste arquivo:
# project02/models/classificar_nos/features.py -> parents[2] == project02
PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR.parent) not in sys.path:          # permite `import project02...` de qualquer pasta
    sys.path.insert(0, str(PROJECT_DIR.parent))

from sklearn.feature_extraction import DictVectorizer          # noqa: E402
from sklearn.linear_model import LogisticRegression            # noqa: E402
from sklearn.pipeline import make_pipeline                     # noqa: E402

from project02.models.pre_processamento.tokenizer import tokenize_with_spans   # noqa: E402

UNIT_WORDS = {"mg", "mcg", "g", "kg", "ml", "l", "mmhg", "mmol", "iu", "cm", "mm", "bpm", "%"}


# ------------------------------------------------------------------ features do token
def shape(w: str) -> str:
    """Formato da palavra: 'Hemoglobin' -> 'Xx', 'COVID-19' -> 'X-d'."""
    s = re.sub(r"[A-Z]", "X", w)
    s = re.sub(r"[a-z]", "x", s)
    s = re.sub(r"\d", "d", s)
    return re.sub(r"(.)\1+", r"\1", s)


def token_features(tokens, i: int, window: int = 2, bag: int = 5) -> dict:
    """
        Features do token i: forma da palavra, vizinhos (+-window), bigrama com o anterior e
        "saco de palavras" de contexto (+-bag). O contexto nao atravessa o limite da sentenca.
    """
    t, w = tokens[i], tokens[i].lower
    f = {"bias": 1, "lower": w, "norm": t.norm, "shape": shape(t.text),
         "pre3": w[:3], "suf2": w[-2:], "suf3": w[-3:], "suf4": w[-4:], "len": min(len(w), 12)}
    if t.text[:1].isupper():
        f["is_title"] = 1
    if len(t.text) > 1 and t.text.isupper():
        f["is_upper"] = 1
    if any(c.isdigit() for c in w):
        f["has_digit"] = 1
    if "-" in w:
        f["has_hyphen"] = 1

    for d in range(-window, window + 1):                     # janela, so dentro da mesma sentenca
        if d == 0:
            continue
        j = i + d
        if 0 <= j < len(tokens) and tokens[j].sent_id == t.sent_id:
            n = tokens[j]
            f[f"{d:+d}:norm"] = n.norm
            f[f"{d:+d}:shape"] = shape(n.text)
            if abs(d) == 1:
                f[f"{d:+d}:suf3"] = n.lower[-3:]
            if n.lower in UNIT_WORDS:
                f[f"{d:+d}:is_unit"] = 1
        else:
            f[f"{d:+d}:edge"] = 1

    if i > 0 and tokens[i - 1].sent_id == t.sent_id:         # bigrama com o anterior
        f["prev+cur"] = tokens[i - 1].norm + "|" + t.norm

    for j in range(max(0, i - bag), min(len(tokens), i + bag + 1)):   # palavras de contexto
        if j != i and tokens[j].sent_id == t.sent_id and not tokens[j].is_punct:
            f["ctx:" + tokens[j].norm] = 1
    return f


# ------------------------------------------------------------------ dicionario como features
def span_cands(record: dict, n: int):
    """
        A partir de um registro de label_case: cands[i] = tipos candidatos do dicionario para o
        token i (None se nao esta em nenhum termo); first[i] = True se i inicia um termo.
    """
    cands, first = [None] * n, [False] * n
    for s in record["spans"]:
        if s["source"] != "dict":                         # Patient (regra) fica de fora
            continue
        for k in range(s["token_start"], s["token_end"] + 1):
            cands[k] = tuple(s["candidates"])
        first[s["token_start"]] = True
    return cands, first


def hide_spans(record: dict, cands, first, p: float, rng: random.Random):
    """'Dropout' do dicionario: esconde cada termo com probabilidade p (como se nao estivesse no MeSH)."""
    for s in record["spans"]:
        if s["source"] == "dict" and rng.random() < p:
            for k in range(s["token_start"], s["token_end"] + 1):
                cands[k], first[k] = None, False


def add_dict_features(f: dict, tokens, i: int, cands, first, own: bool = True, window: int = 2):
    """Acrescenta ao dict `f` o tipo do proprio token (se own) e o tipo dos termos vizinhos."""
    if own and cands[i]:
        f["dict:" + "|".join(cands[i])] = 1               # ex.: dict:Disease  ou  dict:Disease|Result
        f["dict_first" if first[i] else "dict_inside"] = 1
    for d in range(-window, window + 1):
        j = i + d
        if d and 0 <= j < len(tokens) and tokens[j].sent_id == tokens[i].sent_id and cands[j]:
            for c in cands[j]:
                f[f"{d:+d}:dict:{c}"] = 1


# ------------------------------------------------------------------ conjunto de treino
def build_dataset(df, records, text_col: str = "case_text", id_col: str = "case_id",
                  dict_dropout: float = 0.5, seed: int = 0, window: int = 2, bag: int = 5,
                  skip=("IGNORE",)):
    """
        Monta (X, y, ids) a partir dos casos e dos rotulos fracos (`records`, saida do passo 2).
        X = lista de dicts de features; y = rotulos BIO; ids = (case_id, indice do token).
        Ficam fora: tokens IGNORE e tokens de termos ambiguos (rotulo fraco nao e contextual).
        dict_dropout = 0 mostra o dicionario inteiro; 1 esconde todo (mede o desempenho sem dicionario).
        A mesma lista de tokens/ids sai para qualquer valor de dict_dropout.
    """
    texts = dict(zip(df[id_col].astype(str), df[text_col].astype(str)))
    rng = random.Random(seed)
    X, y, ids = [], [], []
    for r in records:
        tokens = tokenize_with_spans(texts[r["case_id"]])    # mesma tokenizacao usada nos rotulos
        assert len(tokens) == len(r["labels"]), f"tokens e rotulos desalinhados em {r['case_id']}"
        cands, first = span_cands(r, len(tokens))
        ambiguous = [bool(c) and len(c) > 1 for c in cands]
        hide_spans(r, cands, first, dict_dropout, rng)
        for i, lab in enumerate(r["labels"]):
            if lab in skip or ambiguous[i]:
                continue
            f = token_features(tokens, i, window, bag)
            add_dict_features(f, tokens, i, cands, first, window=window)
            X.append(f)
            y.append(lab)
            ids.append((r["case_id"], i))
    return X, y, ids


def make_model(C: float = 1.0, max_iter: int = 2000):
    """DictVectorizer + LogisticRegression multinomial. class_weight balanceia o excesso de 'O'."""
    return make_pipeline(
        DictVectorizer(sparse=True),
        LogisticRegression(C=C, max_iter=max_iter, class_weight="balanced"),
    )


# ------------------------------------------------------------------ predicao
def predict_case(model, case, record: dict, window: int = 2, bag: int = 5):
    """
        Prediz as etiquetas BIO de um caso. `case` = process_case(...); `record` = label_case(case, dic)
        (so os spans do dicionario sao usados; nenhum rotulo de treino). Retorna (tags, cands).
    """
    tokens = case.tokens
    cands, first = span_cands(record, len(tokens))
    X = []
    for i in range(len(tokens)):
        f = token_features(tokens, i, window, bag)
        ambiguous = bool(cands[i]) and len(cands[i]) > 1
        add_dict_features(f, tokens, i, cands, first, own=not ambiguous, window=window)   # ambiguo: so contexto
        X.append(f)
    proba, classes, tags = model.predict_proba(X), list(model.classes_), []
    for i, (p, c) in enumerate(zip(proba, cands)):
        if c == ("IGNORE",):                              # Procedure: fora dos nos
            tags.append("O")
        elif c and len(c) > 1:                            # o contexto escolhe o TIPO entre os candidatos
            score = {t: sum(p[k] for k, lab in enumerate(classes) if lab[2:] == t) for t in c}
            tags.append(("B-" if first[i] else "I-") + max(score, key=score.get))   # B/I vem do termo
        else:
            tags.append(classes[int(p.argmax())])
    return tags, cands


# ------------------------------------------------------------------ pos-processamento
def fix_bio(tags: list[str]) -> list[str]:
    """Corrige BIO invalido: um I-X que nao continua um B-X/I-X do mesmo tipo vira B-X."""
    out = list(tags)
    for i, tag in enumerate(out):
        if tag.startswith("I-"):
            prev = out[i - 1] if i > 0 else "O"
            if prev == "O" or prev[2:] != tag[2:]:
                out[i] = "B-" + tag[2:]
    return out


def bio_to_entities(case, tags: list[str], cands=None) -> list[dict]:
    """
        Converte etiquetas BIO em entidades: texto, tipo, posicao no texto e (se `cands` for dado)
        os tipos candidatos do dicionario, para o modelo de relacoes desambiguar depois.
    """
    toks, ents, k, n = case.tokens, [], 0, len(case.tokens)
    while k < n:
        tag = tags[k]
        if tag in ("O", "IGNORE"):
            k += 1
            continue
        typ = tag[2:]
        j = k
        while j + 1 < n and tags[j + 1] == "I-" + typ:
            j += 1
        ents.append({
            "text": case.text[toks[k].char_start:toks[j].char_end], "label": typ,
            "char_start": toks[k].char_start, "char_end": toks[j].char_end,
            "token_start": k, "token_end": j,
            "candidates": list(cands[k]) if cands and cands[k] else [],
        })
        k = j + 1
    return ents
