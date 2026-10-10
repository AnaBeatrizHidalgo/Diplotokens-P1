"""
step2_weak_labels.py -- PASSO 2: rotulacao fraca (dicionario MeSH, match mais longo -> BIO).

Entrada : casos do TREINO (train_cases.csv)  +  data/external/MeSH/mesh_*.csv
Saida   : weak_labels.jsonl  (um registro por caso: labels BIO alinhados aos tokens + spans)
          dictionary.json    (dicionario normalizado; reutilizado na inferencia para
                              anexar `candidates` aos spans -- "opcao B")

Uso em Python / notebook:

    from project02.models.classificar_nos.step2_weak_labels import (
        build_dictionary, label_dataframe, save_weak_labels, summarize)
    dic = build_dictionary()
    records, metas, n_tok = label_dataframe(train_df, dic)

Uso pela linha de comando (de qualquer pasta; so vale para o TREINO):

    python step2_weak_labels.py
    python step2_weak_labels.py --train-csv outro.csv --mesh-dir outra/pasta

COMO O TEXTO E O DICIONARIO SE ENCONTRAM
    Tanto os casos quanto os termos do MeSH passam pela MESMA normalizacao
    (tokenizer.normalize_phrases / tokenizer.tokenize_with_spans: minusculas, sem
    pontuacao, radical). Se voce mudar o tokenizador, mude tokenizer.PREPROC_ID e gere
    os rotulos de novo.

DECISOES (configuraveis nas constantes abaixo)
    * Opcao B p/ ambiguidade: cada span recebe um rotulo PRINCIPAL (usado no
      BIO/treino) e uma lista `candidates` (todos os tipos possiveis, p/ o
      modelo de relacoes desambiguar depois).
    * Procedure nao e no: termo so de Procedure -> rotulo IGNORE (fora do treino).
    * Anatomy perde para qualquer outra categoria em conflito.
    * Efeito nao e classe aqui (vem do dicionario de Symptom/Disease + relacao).
    * Patient: regra lexical (todas as mencoes); idade e genero extraidos por regex.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

# Raiz do projeto (project02), deduzida da posicao deste arquivo:
# project02/models/classificar_nos/step2_weak_labels.py -> parents[2] == project02
PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR.parent) not in sys.path:          # permite `import project02...` de qualquer pasta
    sys.path.insert(0, str(PROJECT_DIR.parent))

from project02.models.pre_processamento.tokenizer import (   # noqa: E402
    PREPROC_ID, ProcessedCase, Token, normalize_phrases, process_case,
)

# ------------------------------------------------------------------ config
DEFAULT_MESH_DIR = PROJECT_DIR / "data" / "external" / "MeSH"
DEFAULT_STOPLIST = PROJECT_DIR / "models" / "classificar_nos" / "stoplist.txt"   # opcional
DEFAULT_TRAIN_CSV = PROJECT_DIR / "data" / "processed" / "splits" / "train_cases.csv"
DEFAULT_OUT_DIR = PROJECT_DIR / "data" / "processed" / "weak_labels"
WEAK_LABELS_FILE = "weak_labels.jsonl"
DICTIONARY_FILE = "dictionary.json"

TYPE_FILES = {
    "Anatomy": "mesh_anatomy.csv", "Disease": "mesh_diseases.csv",
    "Exam": "mesh_exams.csv", "Medication": "mesh_medications.csv",
    "Result": "mesh_results.csv", "Symptom": "mesh_symptoms.csv",
    "Procedure": "mesh_procedures.csv",
}
# rotulo principal de termos ambiguos = primeiro desta lista presente nos candidatos
PRIORITY = ["Symptom", "Disease", "Result", "Exam", "Medication", "Anatomy"]
IGNORE = "IGNORE"
MIN_CHARS_SINGLE = 3            # termos de 1 token com menos caracteres sao descartados
CELL_RE = re.compile(r"\b(cells?|cell lines?)$")   # linhagens celulares em Anatomy

PATIENT = "Patient"
PATIENT_HEADS = {"patient", "man", "woman", "male", "female", "boy", "girl", "infant",
                 "newborn", "neonate", "baby", "toddler", "gentleman", "lady", "adolescent",
                 "child", "men", "women", "patients", "children"}
MALE_WORDS = {"male", "man", "men", "boy", "gentleman"}
FEMALE_WORDS = {"female", "woman", "women", "girl", "lady"}
MALE_PRON = {"he", "him", "his", "himself"}
FEMALE_PRON = {"she", "her", "hers", "herself"}

UNIT_TO_YEARS = {"year": 1, "yr": 1, "y": 1, "": 1, "month": 1 / 12, "week": 1 / 52, "day": 1 / 365}
NUM_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}
_NUM = r"(\d+(?:\.\d+)?|" + "|".join(k for k in NUM_WORDS if k != "zero") + ")"
AGE_PATTERNS = [
    re.compile(_NUM + r"[\s-]*(year|yr|month|week|day)s?[\s-]*old", re.I),
    re.compile(_NUM + r"[\s-]*(y)/?o\b", re.I),
    re.compile(r"\baged?\s+(?:of\s+)?(\d+(?:\.\d+)?)()", re.I),
]


# ------------------------------------------------------------------ dicionario
@dataclass
class Entry:
    label: str                   # rotulo principal (ou IGNORE)
    candidates: tuple[str, ...]  # todos os tipos possiveis


@dataclass
class WeakDictionary:
    entries: dict[tuple[str, ...], Entry]
    preproc_id: str

    @property
    def max_len(self) -> int:
        return max((len(k) for k in self.entries), default=1)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"preproc_id": self.preproc_id, "entries": {
                " ".join(k): [e.label, list(e.candidates)] for k, e in self.entries.items()}},
                f, ensure_ascii=False)

    @classmethod
    def load(cls, path) -> "WeakDictionary":
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls({tuple(k.split(" ")): Entry(v[0], tuple(v[1])) for k, v in d["entries"].items()},
                   d["preproc_id"])


def term_variants(term: str) -> set[str]:
    """Variantes de escrita: sem parenteses; 'tomography, x-ray computed' -> 'x-ray computed tomography'."""
    t = " ".join(term.lower().split())
    out = {t}
    no_par = re.sub(r"\s*\([^)]*\)", "", t).strip()
    if no_par:
        out.add(no_par)
    for x in list(out):
        parts = x.split(", ")                       # virgula+espaco (evita '1,2-dimetil...')
        if len(parts) == 2 and all(parts):
            out.add(f"{parts[1]} {parts[0]}")
    return out


def resolve_types(types: set[str]) -> tuple[str, ...]:
    t = set(types)
    if len(t) > 1:
        t.discard("Procedure")      # Procedure nao e no
    if len(t) > 1:
        t.discard("Anatomy")        # anatomia e a mais ruidosa
    if t == {"Procedure"}:
        return (IGNORE,)
    return tuple(sorted(t, key=PRIORITY.index))


def principal_label(candidates: tuple[str, ...]) -> str:
    return IGNORE if candidates == (IGNORE,) else candidates[0]   # ja ordenado por PRIORITY


def build_dictionary(mesh_dir=DEFAULT_MESH_DIR, normalize=normalize_phrases, stoplist=None,
                     type_files: dict[str, str] | None = None, preproc_id: str = PREPROC_ID,
                     verbose: bool = True) -> WeakDictionary:
    """
        Le os mesh_*.csv (coluna `term`) e monta o dicionario normalizado.
        Arquivos ausentes sao avisados e ignorados (o tipo correspondente fica sem rotulos).
    """
    type_files = type_files or TYPE_FILES
    raw: list[tuple[str, str]] = []                  # (variante, tipo)
    loaded, missing = [], []
    for typ, fname in type_files.items():
        path = Path(mesh_dir) / fname
        if not path.exists():
            missing.append(fname)
            continue
        loaded.append(typ)
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                term = row["term"].strip()
                if not term or (typ == "Anatomy" and CELL_RE.search(term.lower())):
                    continue
                raw.extend((v, typ) for v in term_variants(term))
    if not loaded:
        raise FileNotFoundError(f"nenhum mesh_*.csv encontrado em {mesh_dir}")
    if verbose:
        print("Dicionarios carregados:", ", ".join(loaded))
        if missing:
            print("AVISO - arquivos ausentes (esses tipos nao serao rotulados):", ", ".join(missing))

    keys = normalize([v for v, _ in raw])                    # em lote (rapido)

    stop: set[tuple[str, ...]] = set()
    if stoplist and Path(stoplist).exists():
        with open(stoplist, encoding="utf-8") as f:
            lines = [l.strip().lower() for l in f if l.strip() and not l.startswith("#")]
        stop = {k for k in normalize(lines) if k}

    grouped: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for (_, typ), key in zip(raw, keys):
        if not key or key in stop:
            continue
        if len(key) == 1 and (len(key[0]) < MIN_CHARS_SINGLE or key[0].isdigit()):
            continue
        grouped[key].add(typ)

    entries = {}
    for key, types in grouped.items():
        cands = resolve_types(types)
        entries[key] = Entry(principal_label(cands), cands)
    dic = WeakDictionary(entries, preproc_id)
    if verbose:
        print(f"Dicionario: {len(dic.entries)} termos normalizados "
              f"({sum(len(e.candidates) > 1 for e in dic.entries.values())} ambiguos, "
              f"{sum(e.label == IGNORE for e in dic.entries.values())} IGNORE)")
    return dic


# ------------------------------------------------------------------ Patient (regras)
def extract_age_years(text: str, lo: int = 0, hi: int | None = None, anchor: int | None = None):
    """Idade (em anos) no trecho text[lo:hi]; com `anchor`, pega a mais proxima dele
    (evita pegar, p.ex., a idade da mae)."""
    hi = len(text) if hi is None else hi
    best = None
    for pat in AGE_PATTERNS:
        for m in pat.finditer(text, lo, hi):
            raw = m.group(1).lower()
            val = float(NUM_WORDS[raw]) if raw in NUM_WORDS else float(raw)
            unit = m.group(2).lower() if m.group(2) else ""
            dist = abs(m.start() - anchor) if anchor is not None else m.start()
            cand = (dist, round(val * UNIT_TO_YEARS.get(unit, 1), 3))
            best = cand if best is None or cand < best else best
    return None if best is None else best[1]


def extract_gender(tokens: list[Token], head_idx: list[int]):
    for i in head_idx:                                     # 1) palavra-nucleo ("woman")
        w = tokens[i].lower
        if w in MALE_WORDS:
            return "male", "head"
        if w in FEMALE_WORDS:
            return "female", "head"
    first = tokens[head_idx[0]].sent_id if head_idx else 0  # 2) pronomes, perto da 1a mencao
    near = [t for t in tokens if first <= t.sent_id <= first + 1]
    m = sum(t.lower in MALE_PRON for t in near)
    f = sum(t.lower in FEMALE_PRON for t in near)
    if m == f:                                              # 3) reserva: texto todo
        m = sum(t.lower in MALE_PRON for t in tokens)
        f = sum(t.lower in FEMALE_PRON for t in tokens)
    if m != f:
        return ("male" if m > f else "female"), "pronoun"
    return None, None


def patient_spans(tokens: list[Token]) -> list[tuple[int, int]]:
    """Intervalos [a, b] de tokens consecutivos do lexico ("male patient" = 1 span)."""
    spans, a = [], None
    for t in tokens:
        hit = t.lower in PATIENT_HEADS or t.norm in PATIENT_HEADS
        if hit and a is None:
            a = t.i
        elif not hit and a is not None:
            spans.append((a, t.i - 1))
            a = None
    if a is not None:
        spans.append((a, tokens[-1].i))
    return spans


# ------------------------------------------------------------------ rotulacao
def _punct_inside(toks: list[Token], window: list[Token]) -> int:
    """Tokens de pontuacao entre o 1o e o ultimo token da janela (ex.: o hifen de
    'x - ray'). Sao permitidos DENTRO de um span e recebem I-<tipo> em add_span."""
    a, b = window[0].i, window[-1].i
    return sum(1 for k in range(a, b + 1) if toks[k].is_punct)


def label_case(case: ProcessedCase, dic: WeakDictionary) -> dict:
    toks = case.tokens
    n = len(toks)
    labels = ["O"] * n
    occupied = [False] * n
    spans: list[dict] = []

    def add_span(a: int, b: int, label: str, cands: tuple[str, ...], source: str):
        for k in range(a, b + 1):
            occupied[k] = True
            labels[k] = IGNORE if label == IGNORE else (f"B-{label}" if k == a else f"I-{label}")
        spans.append({
            "text": case.text[toks[a].char_start:toks[b].char_end], "label": label,
            "candidates": list(cands), "char_start": toks[a].char_start,
            "char_end": toks[b].char_end, "sent_id": toks[a].sent_id,
            "token_start": a, "token_end": b, "source": source})

    # 1) Patient (tem precedencia sobre o dicionario)
    heads = []
    for a, b in patient_spans(toks):
        add_span(a, b, PATIENT, (PATIENT,), "rule")
        heads.extend(range(a, b + 1))

    # 2) dicionario: match mais longo, a esquerda, dentro de cada sentenca
    by_sent: dict[int, list[Token]] = defaultdict(list)
    for t in toks:
        if not t.is_punct and not occupied[t.i]:
            by_sent[t.sent_id].append(t)
    max_len = dic.max_len
    for sent_toks in by_sent.values():
        i = 0
        while i < len(sent_toks):
            hit = None
            for ln in range(min(max_len, len(sent_toks) - i), 0, -1):
                window = sent_toks[i:i + ln]
                # janela precisa ser contigua (so pontuacao pode ficar no meio)
                if window[-1].i - window[0].i + 1 - ln > _punct_inside(toks, window):
                    continue
                e = dic.entries.get(tuple(t.norm for t in window))
                if e:
                    hit = (window, e)
                    break
            if hit:
                w, e = hit
                add_span(w[0].i, w[-1].i, e.label, e.candidates, "dict")
                i += len(w)
            else:
                i += 1
    spans.sort(key=lambda s: s["token_start"])

    gender, g_src = extract_gender(toks, heads)
    age = None
    if heads:                                   # idade na frase da 1a mencao, perto dela
        h0 = toks[heads[0]]
        sent = [t for t in toks if t.sent_id == h0.sent_id]
        age = extract_age_years(case.text, sent[0].char_start, sent[-1].char_end, h0.char_start)
    if age is None:                             # reserva: inicio do texto
        age = extract_age_years(case.text, 0, min(len(case.text), 300))
    return {
        "case_id": case.case_id, "article_id": case.article_id,
        "labels": labels, "spans": spans,
        "patient": {"age_years": age, "gender": gender, "gender_source": g_src,
                    "has_mention": bool(heads)},
    }


# ------------------------------------------------------------------ DataFrame -> rotulos
def label_dataframe(df, dic: WeakDictionary, text_col: str = "case_text",
                    id_col: str = "case_id", group_col: str = "article_id"):
    """
        Tokeniza e rotula cada linha do DataFrame de casos.
        Retorna (records, metas, n_tokens): records = saida de label_case por caso;
        metas = {case_id: demais colunas do CSV (age, gender, ...)}.
    """
    records, metas, n_tok = [], {}, 0
    skip = {id_col, group_col, text_col}
    for row in df.to_dict("records"):
        meta = {k: v for k, v in row.items() if k not in skip}
        case = process_case(str(row[id_col]), str(row[group_col]), str(row[text_col]), meta)
        records.append(label_case(case, dic))
        metas[case.case_id] = case.meta
        n_tok += len(case.tokens)
    return records, metas, n_tok


def save_weak_labels(records: list[dict], dic: WeakDictionary, out_dir=DEFAULT_OUT_DIR):
    """Grava weak_labels.jsonl e dictionary.json em out_dir. Retorna o caminho do jsonl."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dic.save(out_dir / DICTIONARY_FILE)
    out = out_dir / WEAK_LABELS_FILE
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": {"kind": "weak_labels", "preproc_id": dic.preproc_id,
                                      "dictionary": DICTIONARY_FILE}}) + "\n")
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


# ------------------------------------------------------------------ relatorio
def _valid_number(x) -> bool:
    try:
        return x is not None and float(x) == float(x)      # False para NaN
    except (TypeError, ValueError):
        return False


def summarize(records: list[dict], metas: dict[str, dict], n_tokens: int):
    c_lab, c_cand = Counter(), Counter()
    covered = sum(l != "O" for r in records for l in r["labels"])
    for r in records:
        for s in r["spans"]:
            c_lab[s["label"]] += 1
            if len(s["candidates"]) > 1:
                c_cand["|".join(s["candidates"])] += 1
    print(f"\nCasos: {len(records)} | tokens: {n_tokens} | tokens em span/IGNORE: {covered} "
          f"({100 * covered / max(n_tokens, 1):.1f}%)")
    print("Spans por rotulo principal:", dict(c_lab.most_common()))
    print("Spans ambiguos (candidates):", dict(c_cand.most_common()))
    # validacao de Patient contra as colunas do CSV
    ok_g = bad_g = ok_a = bad_a = no_g = no_a = 0
    for r in records:
        m, p = metas[r["case_id"]], r["patient"]
        mg = str(m.get("gender") or "").lower()
        if mg in ("male", "female"):
            if p["gender"] is None:
                no_g += 1
            elif p["gender"] == mg:
                ok_g += 1
            else:
                bad_g += 1
        if _valid_number(m.get("age")):
            if p["age_years"] is None:
                no_a += 1
            elif abs(p["age_years"] - float(m["age"])) < 1:
                ok_a += 1
            else:
                bad_a += 1
    print(f"Patient/genero vs CSV: ok={ok_g} divergente={bad_g} nao_extraido={no_g}")
    print(f"Patient/idade  vs CSV: ok={ok_a} divergente={bad_a} nao_extraido={no_a}")


# ------------------------------------------------------------------ CLI
def main():
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default=str(DEFAULT_TRAIN_CSV), help="casos de TREINO")
    ap.add_argument("--mesh-dir", default=str(DEFAULT_MESH_DIR))
    ap.add_argument("--stoplist", default=str(DEFAULT_STOPLIST))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--text-col", default="case_text")
    ap.add_argument("--id-col", default="case_id")
    ap.add_argument("--group-col", default="article_id")
    a = ap.parse_args()

    df = pd.read_csv(a.train_csv)
    dic = build_dictionary(a.mesh_dir, stoplist=a.stoplist)
    records, metas, n_tok = label_dataframe(df, dic, a.text_col, a.id_col, a.group_col)
    out = save_weak_labels(records, dic, a.out_dir)
    print(f"-> {out}")
    summarize(records, metas, n_tok)


if __name__ == "__main__":
    main()