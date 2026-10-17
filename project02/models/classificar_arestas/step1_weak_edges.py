"""
Passo 1 da Classificacao de Arestas (Extracao de Relacoes):
Segmentacao de sentencas, geracao de pares candidatos (E1, E2) + texto intermediario,
e rotulacao fraca (weak supervision) para o conjunto de treino.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

# Tokenizador compativel com termos clinicos (preserva hifens e apostrofos internos)
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")

# Quebra de sentencas protegendo abreviacoes clinicas comuns (Dr., mg., fig., e.g., etc.) e decimais (3.5)
SENT_SPLIT_RE = re.compile(
    r"(?<!\bDr)(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<!\bvs)(?<!\bfig)(?<!\bFig)"
    r"(?<!\be\.g)(?<!\bi\.e)(?<!\bapprox)(?<!\bno)(?<!\bNo)"
    r"(?<=[.!?])\s+(?=[A-Z0-9])|\n+"
)

# =====================================================================
# 1. ESQUEMA DO GRAFO: PARES DE TIPOS VALIDOS E ARESTAS POSSIVEIS
# =====================================================================
VALID_SCHEMA_PAIRS: Dict[Tuple[str, str], List[str]] = {
    ("Patient", "Disease"):    ["FAMILIAR_HISTORY", "HISTORY", "DIAGNOSED_WITH"],
    ("Patient", "Symptom"):    ["PRESENTS_WITH"],
    ("Patient", "Medication"): ["TAKES"],
    ("Patient", "Exam"):       ["UNDERWENT_EXAM"],
    ("Patient", "Procedure"):  ["UNDERWENT_EXAM"],
    ("Disease", "Anatomy"):    ["LOCATED_IN"],
    ("Symptom", "Anatomy"):    ["LOCATED_IN"],
    ("Exam", "Result"):        ["REVEALS"],
    ("Medication", "Effect"):  ["HAS_EFFECT"],
    # No MeSH atual, efeitos colaterais/desfechos de medicacao frequentemente
    # sao rotulados como Symptom ou Result pelo classificador de nos:
    ("Medication", "Symptom"): ["HAS_EFFECT"],
    ("Medication", "Result"):  ["HAS_EFFECT"],
}

ALL_EDGE_LABELS = [
    "FAMILIAR_HISTORY",
    "HISTORY",
    "DIAGNOSED_WITH",
    "PRESENTS_WITH",
    "TAKES",
    "UNDERWENT_EXAM",
    "LOCATED_IN",
    "REVEALS",
    "HAS_EFFECT",
    "NONE",
]

# =====================================================================
# 2. GATILHOS LEXICAIS (HEARST PATTERNS / PISTAS DE ROTULACAO FRACA)
# =====================================================================
NEGATION_CUES = re.compile(
    r"\b(?:no|not|without|denies|denied|negative\s+for|free\s+of|unremarkable|"
    r"non-contributory|ruled\s+out|absence\s+of|neither|nor|declined|refused)\b",
    re.I,
)

CLAUSE_BARRIER_RE = re.compile(
    r"\b(?:but|however|although|though|whereas|except|yet)\b|[;:]",
    re.I,
)

FAMILY_CUES = re.compile(
    r"\b(?:family\s+history|familial|familiar|mother|father|sister|brother|"
    r"maternal|paternal|parents|grandmother|grandfather|son|daughter|hereditary)\b",
    re.I,
)

HISTORY_CUES = re.compile(
    r"\b(?:history\s+of|past\s+medical|medical\s+history|previously|prior|"
    r"former|childhood|chronic|known|long-standing|years\s+ago|months\s+ago|"
    r"had\s+a\s+history|underlying|pre-existing)\b",
    re.I,
)

DIAGNOSED_CUES = re.compile(
    r"\b(?:diagnosed|diagnosis|consistent\s+with|confirmed|confirming|"
    r"final\s+diagnosis|pathology\s+revealed|biopsy\s+confirmed|suffered\s+from)\b",
    re.I,
)

LOCATED_CUES = re.compile(
    r"\b(?:in|of|on|at|within|from|to|into|throughout|involving|affecting|"
    r"located|arising|adherent|extending|region|wall|lobe|tract|cavity)\b",
    re.I,
)

REVEALS_CUES = re.compile(
    r"\b(?:revealed|reveals|showed|shows|demonstrated|demonstrates|confirmed|"
    r"found|detected|identified|indicated|noted|seen|visualized|yielded|"
    r"positive\s+for|consistent\s+with|evidence\s+of)\b",
    re.I,
)

EFFECT_CUES = re.compile(
    r"\b(?:caused|causes|induced|induces|led\s+to|resulted\s+in|relieved|"
    r"improved|resolved|alleviated|controlled|reduced|worsened|adverse|"
    r"side\s+effect|reaction\s+to|secondary\s+to|following\s+administration|"
    r"response\s+to|after\s+receiving|after\s+taking)\b",
    re.I,
)

# Palavras específicas das pistas (usadas no Passo 2 para Cue Dropout no treino)
ALL_CUE_WORDS = {
    "history", "familial", "familiar", "mother", "father", "sister", "brother",
    "maternal", "paternal", "hereditary", "previously", "prior", "former",
    "childhood", "chronic", "known", "diagnosed", "diagnosis", "confirmed",
    "revealed", "reveals", "showed", "shows", "demonstrated", "detected",
    "identified", "caused", "induced", "relieved", "improved", "resolved",
    "adverse", "denies", "denied", "without", "unremarkable",
}


# =====================================================================
# 3. TOKENIZACAO COM POSICAO E SEGMENTACAO DE SENTENCAS
# =====================================================================
def tokenize_with_offsets(text: str) -> List[Dict[str, Any]]:
    """Retorna lista de tokens com indices e offsets de caractere exatos."""
    tokens = []
    for idx, m in enumerate(TOKEN_RE.finditer(text)):
        raw = m.group(0)
        tokens.append({
            "tok_idx": idx,
            "text": raw,
            "norm": raw.lower(),
            "char_start": m.start(),
            "char_end": m.end(),
        })
    return tokens


def split_sentences(text: str, tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Divide o texto do caso em sentencas e associa a faixa de tokens [tok_start, tok_end)
    de cada sentenca.
    """
    if not text.strip() or not tokens:
        return []

    spans_char: List[Tuple[int, int]] = []
    last = 0
    for m in SENT_SPLIT_RE.finditer(text):
        spans_char.append((last, m.start() + (1 if text[m.start()] in ".!?" else 0)))
        last = m.end()
    if last < len(text):
        spans_char.append((last, len(text)))

    sentences = []
    for sent_id, (c_start, c_end) in enumerate(spans_char):
        sent_tokens = [
            t for t in tokens
            if t["char_start"] >= c_start and t["char_end"] <= c_end
        ]
        if not sent_tokens:
            continue
        sentences.append({
            "sent_id": sent_id,
            "char_start": sent_tokens[0]["char_start"],
            "char_end": sent_tokens[-1]["char_end"],
            "tok_start": sent_tokens[0]["tok_idx"],
            "tok_end": sent_tokens[-1]["tok_idx"] + 1,
            "text": text[sent_tokens[0]["char_start"]:sent_tokens[-1]["char_end"]],
            "tokens": sent_tokens,
        })
    return sentences


# =====================================================================
# 4. ALINHAMENTO DE ENTIDADES (COMPATIVEL COM TREINO E TESTE)
# =====================================================================
def align_entities_to_tokens(
    raw_entities: Sequence[Dict[str, Any]],
    text: str,
    tokens: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Normaliza a lista de entidades (vinda de `step2_weak_labels` no treino
    ou de `bio_to_entities` no teste) garantindo `tok_start`, `tok_end`,
    `char_start`, `char_end`, `text` e `label`. Filtra rotulos `IGNORE` ou `O`.
    """
    aligned = []
    n_toks = len(tokens)
    search_cursor = 0

    for ent in raw_entities:
        label = ent.get("label", "O")
        if not label or label in ("IGNORE", "O"):
            continue

        ent_text = str(ent.get("text", "")).strip()
        if not ent_text:
            continue

        candidates = list(ent.get("candidates", [label]))
        tok_start, tok_end = None, None
        char_start, char_end = None, None

        # Caso 1: a entidade ja possui tok_start / tok_end explicitos
        if "tok_start" in ent and "tok_end" in ent:
            tok_start, tok_end = int(ent["tok_start"]), int(ent["tok_end"])

        # Caso 2: a entidade possui 'start' e 'end' (podem ser indices de token ou char)
        elif "start" in ent and "end" in ent:
            s, e = int(ent["start"]), int(ent["end"])
            # Verifica se s, e sao offsets de caractere exatos
            if 0 <= s < e <= len(text) and text[s:e].lower() == ent_text.lower():
                char_start, char_end = s, e
            # Verifica se s, e sao indices de tokens
            elif 0 <= s < e <= n_toks:
                span_str = text[tokens[s]["char_start"]:tokens[e - 1]["char_end"]]
                if span_str.lower() == ent_text.lower() or tokens[s]["norm"] in ent_text.lower():
                    tok_start, tok_end = s, e
                elif 0 <= s < e <= len(text):
                    char_start, char_end = s, e

        # Caso 3: se ainda nao temos tok_start/tok_end, localizamos via char_start/char_end ou busca sequencial
        if tok_start is None:
            if char_start is None:
                idx = text.lower().find(ent_text.lower(), search_cursor)
                if idx == -1:
                    idx = text.lower().find(ent_text.lower())
                if idx != -1:
                    char_start = idx
                    char_end = idx + len(ent_text)
                    search_cursor = char_end

            if char_start is not None and char_end is not None:
                overlapping = [
                    t["tok_idx"] for t in tokens
                    if t["char_end"] > char_start and t["char_start"] < char_end
                ]
                if overlapping:
                    tok_start = overlapping[0]
                    tok_end = overlapping[-1] + 1

        if tok_start is None or tok_end is None or not (0 <= tok_start < tok_end <= n_toks):
            continue

        char_start = tokens[tok_start]["char_start"]
        char_end = tokens[tok_end - 1]["char_end"]
        norm_text = " ".join(t["norm"] for t in tokens[tok_start:tok_end])

        aligned.append({
            "text": text[char_start:char_end],
            "norm": norm_text,
            "label": label,
            "candidates": candidates,
            "tok_start": tok_start,
            "tok_end": tok_end,
            "char_start": char_start,
            "char_end": char_end,
        })

    # Ordena pela posicao no texto e remove duplicatas exatas de mesma posicao
    aligned.sort(key=lambda x: (x["tok_start"], x["tok_end"]))
    dedup = []
    seen = set()
    for a in aligned:
        key = (a["tok_start"], a["tok_end"], a["label"])
        if key not in seen:
            seen.add(key)
            dedup.append(a)
    return dedup


# =====================================================================
# 5. HEURISTICAS DE ROTULACAO FRACA (WEAK SUPERVISION) POR PAR
# =====================================================================
def _has_negation(left_text: str, span_between: str, is_patient_edge: bool) -> bool:
    """
    Detecta se a entidade-alvo esta sob o escopo de uma negacao clinica
    (ex.: 'denies fever', 'no evidence of cyst', 'CT was not performed').
    Corta o escopo se houver barreira de oracao ('but', ';').
    """
    # Se houver barreira adversativa no span_between, olhamos apenas apos a barreira
    parts = CLAUSE_BARRIER_RE.split(span_between)
    effective_between = parts[-1] if parts else span_between

    if NEGATION_CUES.search(effective_between):
        return True
    # Olha tambem a janela imediatamente a esquerda quando as entidades estao muito proximas
    if is_patient_edge or len(span_between.split()) <= 3:
        left_parts = CLAUSE_BARRIER_RE.split(left_text)
        effective_left = left_parts[-1] if left_parts else left_text
        if NEGATION_CUES.search(effective_left):
            return True
    return False


def assign_weak_edge_label(
    e1: Dict[str, Any],
    e2: Dict[str, Any],
    span_between: str,
    left_context: str,
    right_context: str,
    sentence_text: str,
    token_dist: int,
    same_clause: bool,
    intervening_types: Sequence[str],
) -> Tuple[str, List[str]]:
    """
    Atribui o rotulo fraco (y) para o par direcionado (E1, E2) com base nas pistas
    lexicais, distancia em tokens e restricoes de esquema.
    Retorna (label, cues_matched).
    """
    pair_type = (e1["label"], e2["label"])
    cues_matched: List[str] = []

    # 1. Se o par de tipos nao existe no esquema do grafo (ex.: Exam -> Anatomy), e sempre NONE
    if pair_type not in VALID_SCHEMA_PAIRS:
        return "NONE", cues_matched

    is_patient_edge = (e1["label"] == "Patient")
    full_local_context = f"{left_context} {span_between} {right_context}".strip()

    # Coleta quais pistas disparam no contexto para passar como metadado
    if NEGATION_CUES.search(span_between) or NEGATION_CUES.search(left_context):
        cues_matched.append("NEGATION")
    if FAMILY_CUES.search(full_local_context) or FAMILY_CUES.search(sentence_text):
        cues_matched.append("FAMILY")
    if HISTORY_CUES.search(full_local_context) or HISTORY_CUES.search(sentence_text):
        cues_matched.append("HISTORY")
    if DIAGNOSED_CUES.search(full_local_context):
        cues_matched.append("DIAGNOSED")
    if LOCATED_CUES.search(span_between):
        cues_matched.append("LOCATED")
    if REVEALS_CUES.search(span_between) or REVEALS_CUES.search(left_context):
        cues_matched.append("REVEALS")
    if EFFECT_CUES.search(full_local_context):
        cues_matched.append("EFFECT")

    # 2. Filtro de Negacao: se a entidade estiver negada, nao criamos aresta (NONE)
    if _has_negation(left_context, span_between, is_patient_edge):
        return "NONE", cues_matched

    # 3. Regras especificas para (Patient -> Disease): 3 classes positivas possiveis
    if pair_type == ("Patient", "Disease"):
        # Prioridade 1: Historico familiar na sentenca/span
        if "FAMILY" in cues_matched:
            return "FAMILIAR_HISTORY", cues_matched
        # Prioridade 2: Historico pessoal pregresso
        if "HISTORY" in cues_matched:
            return "HISTORY", cues_matched
        # Caso contrario: Doenca atual diagnosticada
        return "DIAGNOSED_WITH", cues_matched

    # 4. Regras para demais arestas centradas no Paciente
    if pair_type == ("Patient", "Symptom"):
        # Se a sentenca descreve historico familiar (ex: 'mother had chest pain'), nao e sintoma do paciente
        if "FAMILY" in cues_matched:
            return "NONE", cues_matched
        return "PRESENTS_WITH", cues_matched

    if pair_type == ("Patient", "Medication"):
        return "TAKES", cues_matched

    if pair_type in (("Patient", "Exam"), ("Patient", "Procedure")):
        return "UNDERWENT_EXAM", cues_matched

    # 5. Regras para (Disease -> Anatomy) e (Symptom -> Anatomy) -> LOCATED_IN
    if pair_type in (("Disease", "Anatomy"), ("Symptom", "Anatomy")):
        # Evita cruzar entidades se houver outra Disease/Symptom ou outra Anatomy exatamente no meio
        if e1["label"] in intervening_types or "Anatomy" in intervening_types:
            return "NONE", cues_matched
        # Se estao muito proximas (ex: 'abdominal pain', 'cyst in the stomach') ou ligadas por preposicao espacial
        if (token_dist <= 4 and same_clause) or ("LOCATED" in cues_matched and token_dist <= 10 and same_clause):
            return "LOCATED_IN", cues_matched
        return "NONE", cues_matched

    # 6. Regras para (Exam -> Result) -> REVEALS
    if pair_type == ("Exam", "Result"):
        # Se houver outro Exam entre este Exam e o Result, liga apenas ao Exam mais proximo
        if "Exam" in intervening_types:
            return "NONE", cues_matched
        if ("REVEALS" in cues_matched and token_dist <= 15) or (token_dist <= 8 and same_clause):
            return "REVEALS", cues_matched
        return "NONE", cues_matched

    # 7. Regras para (Medication -> Effect / Symptom / Result) -> HAS_EFFECT
    if pair_type in (("Medication", "Effect"), ("Medication", "Symptom"), ("Medication", "Result")):
        if "Medication" in intervening_types:
            return "NONE", cues_matched
        if e2["label"] == "Effect" and token_dist <= 10 and same_clause:
            return "HAS_EFFECT", cues_matched
        if "EFFECT" in cues_matched and token_dist <= 12:
            return "HAS_EFFECT", cues_matched
        return "NONE", cues_matched

    return "NONE", cues_matched


# =====================================================================
# 6. GERACAO DE PARES CANDIDATOS POR SENTENCA E POR CASO
# =====================================================================
def generate_case_edge_candidates(
    case_id: str,
    article_id: str,
    text: str,
    raw_entities: Sequence[Dict[str, Any]],
    include_virtual_patient: bool = True,
) -> List[Dict[str, Any]]:
    """
    Gera todos os pares candidatos (E1, E2) dentro de cada sentenca de um caso clinico.
    - Para pares entre entidades secundarias (Exam, Result, Disease, Symptom, Anatomy, Medication),
      orienta (E1, E2) na direcao canonica do esquema quando aplicavel, preservando `reading_order`
      e o `span_between` exato entre elas.
    - Para entidades ligadas ao Paciente, associa cada entidade ao no `Patient` mais proximo na
      sentenca (ou a um no `Patient` virtual no inicio da sentenca caso o sujeito esteja implicito).
    """
    tokens = tokenize_with_offsets(text)
    if not tokens:
        return []

    sentences = split_sentences(text, tokens)
    entities = align_entities_to_tokens(raw_entities, text, tokens)

    # Indexa tokens por tok_idx para acesso O(1)
    candidates_out: List[Dict[str, Any]] = []

    for sent in sentences:
        s_start, s_end = sent["tok_start"], sent["tok_end"]
        sent_ents = [
            e for e in entities
            if e["tok_start"] >= s_start and e["tok_end"] <= s_end
        ]
        if not sent_ents:
            continue

        patient_ents = [e for e in sent_ents if e["label"] == "Patient"]
        non_patient_ents = [e for e in sent_ents if e["label"] != "Patient"]

        # Se a sentenca tem entidades clinicas mas nao menciona "patient/man/woman" explicitamente,
        # criamos um ancora Patient virtual no inicio da sentenca para capturar o contexto inicial
        # (ex.: "History of hypertension." -> span_between = "History of")
        if not patient_ents and include_virtual_patient and non_patient_ents:
            first_tok = tokens[s_start]
            patient_ents = [{
                "text": "Patient",
                "norm": "patient",
                "label": "Patient",
                "candidates": ["Patient"],
                "tok_start": s_start,
                "tok_end": s_start,  # span de comprimento zero no inicio da sentenca
                "char_start": first_tok["char_start"],
                "char_end": first_tok["char_start"],
                "is_virtual": True,
            }]

        # -----------------------------------------------------------------
        # A) Pares centrados no Paciente: (Patient -> Entidade)
        # -----------------------------------------------------------------
        for target in non_patient_ents:
            if ("Patient", target["label"]) not in VALID_SCHEMA_PAIRS:
                continue

            # Escolhe a mencao a Patient mais proxima (preferencialmente anterior ao target)
            preceding_patients = [p for p in patient_ents if p["tok_end"] <= target["tok_start"]]
            if preceding_patients:
                pat = min(preceding_patients, key=lambda p: target["tok_start"] - p["tok_end"])
            else:
                pat = min(patient_ents, key=lambda p: abs(p["tok_start"] - target["tok_start"]))

            pair_rec = _build_pair_record(
                case_id=case_id,
                article_id=article_id,
                sent=sent,
                e1=pat,
                e2=target,
                all_tokens=tokens,
                sent_ents=non_patient_ents,
            )
            candidates_out.append(pair_rec)

        # -----------------------------------------------------------------
        # B) Pares Entidade-Entidade na mesma sentenca (ex: Exam-Result,
        #    Disease-Anatomy, Symptom-Anatomy, Exam-Anatomy [NONE], etc.)
        # -----------------------------------------------------------------
        n_np = len(non_patient_ents)
        for i in range(n_np):
            for j in range(i + 1, n_np):
                ent_a = non_patient_ents[i]
                ent_b = non_patient_ents[j]

                # Ignora se forem exatamente a mesma mencao sobreposta
                if ent_a["tok_end"] > ent_b["tok_start"]:
                    continue

                # Define a orientacao canonica (E1 -> E2):
                # Se (B -> A) e um par valido no esquema (ex: "abdominal [Anatomy] pain [Symptom]"
                # ou "CT scan of the abdomen [Anatomy] revealed a cystic lesion [Disease]"),
                # orientamos E1 = cystic lesion e E2 = abdomen!
                if (ent_a["label"], ent_b["label"]) in VALID_SCHEMA_PAIRS:
                    e1, e2 = ent_a, ent_b
                elif (ent_b["label"], ent_a["label"]) in VALID_SCHEMA_PAIRS:
                    e1, e2 = ent_b, ent_a
                else:
                    # Par incompativel (ex: Pair 3 ("CT scan", "abdomen") -> Exam -> Anatomy):
                    # mantem na ordem de leitura para treinar a classe NONE
                    e1, e2 = ent_a, ent_b

                pair_rec = _build_pair_record(
                    case_id=case_id,
                    article_id=article_id,
                    sent=sent,
                    e1=e1,
                    e2=e2,
                    all_tokens=tokens,
                    sent_ents=non_patient_ents,
                )
                candidates_out.append(pair_rec)

    return candidates_out


def _build_pair_record(
    case_id: str,
    article_id: str,
    sent: Dict[str, Any],
    e1: Dict[str, Any],
    e2: Dict[str, Any],
    all_tokens: List[Dict[str, Any]],
    sent_ents: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Constroi o dicionario completo de um par (E1, E2) + texto entre eles e seu rotulo fraco."""
    s_start, s_end = sent["tok_start"], sent["tok_end"]

    # Determina qual entidade vem primeiro na ordem de leitura da sentenca
    if e1["tok_start"] <= e2["tok_start"]:
        first_ent, second_ent = e1, e2
        reading_order = "E1_before_E2"
    else:
        first_ent, second_ent = e2, e1
        reading_order = "E2_before_E1"

    between_start = first_ent["tok_end"]
    between_end = second_ent["tok_start"]
    between_toks = all_tokens[between_start:between_end] if between_end > between_start else []

    left_start = max(s_start, first_ent["tok_start"] - 4)
    left_toks = all_tokens[left_start:first_ent["tok_start"]]

    right_end = min(s_end, second_ent["tok_end"] + 4)
    right_toks = all_tokens[second_ent["tok_end"]:right_end]

    span_between = " ".join(t["text"] for t in between_toks)
    tokens_between = [t["norm"] for t in between_toks]
    left_context = " ".join(t["norm"] for t in left_toks)
    right_context = " ".join(t["norm"] for t in right_toks)

    token_dist = len(between_toks)
    same_clause = not bool(CLAUSE_BARRIER_RE.search(span_between))

    # Quais tipos de entidades aparecem estritamente entre first_ent e second_ent?
    intervening_types = [
        ent["label"] for ent in sent_ents
        if ent["tok_start"] >= between_start and ent["tok_end"] <= between_end
    ]

    weak_label, cues_matched = assign_weak_edge_label(
        e1=e1,
        e2=e2,
        span_between=span_between,
        left_context=left_context,
        right_context=right_context,
        sentence_text=sent["text"],
        token_dist=token_dist,
        same_clause=same_clause,
        intervening_types=intervening_types,
    )

    return {
        "case_id": str(case_id),
        "article_id": str(article_id),
        "sent_id": sent["sent_id"],
        "sentence_text": sent["text"],
        "e1_text": e1["text"],
        "e1_norm": e1["norm"],
        "type_E1": e1["label"],
        "e1_tok_start": e1["tok_start"],
        "e1_tok_end": e1["tok_end"],
        "e1_is_virtual": bool(e1.get("is_virtual", False)),
        "e2_text": e2["text"],
        "e2_norm": e2["norm"],
        "type_E2": e2["label"],
        "e2_tok_start": e2["tok_start"],
        "e2_tok_end": e2["tok_end"],
        "reading_order": reading_order,
        "span_between": span_between,
        "tokens_between": tokens_between,
        "left_context": left_context,
        "right_context": right_context,
        "token_dist": token_dist,
        "same_clause": same_clause,
        "intervening_types": intervening_types,
        "cues_matched": cues_matched,
        "label": weak_label,
    }


# =====================================================================
# 7. PROCESSAMENTO EM LOTE (DATAFRAME), SALVAMENTO E RESUMO
# =====================================================================
def label_edges_dataframe(
    df: pd.DataFrame,
    node_records_or_entities: Sequence[Any],
    text_col: str = "case_text",
    id_col: str = "case_id",
    group_col: str = "article_id",
    include_virtual_patient: bool = True,
) -> List[Dict[str, Any]]:
    """
    Gera pares candidatos e rotulos fracos de arestas para todos os casos de `df`.
    Aceita tanto `records` (saida de `step2_weak_labels.label_dataframe` no treino)
    quanto `test['entities']` (saida de `bio_to_entities` no teste).
    """
    # Mapeia case_id -> lista de spans/entidades
    spans_by_case: Dict[str, List[Dict[str, Any]]] = {}
    rows = df.to_dict("records")

    for idx, item in enumerate(node_records_or_entities):
        if isinstance(item, dict) and "spans" in item:
            cid = str(item.get("case_id", rows[idx][id_col]))
            spans_by_case[cid] = item["spans"]
        elif isinstance(item, list):
            cid = str(rows[idx][id_col])
            spans_by_case[cid] = item
        else:
            cid = str(rows[idx][id_col])
            spans_by_case[cid] = []

    all_edge_records: List[Dict[str, Any]] = []
    for row in rows:
        cid = str(row[id_col])
        aid = str(row[group_col])
        text = str(row.get(text_col, "") or "")
        raw_ents = spans_by_case.get(cid, [])

        case_edges = generate_case_edge_candidates(
            case_id=cid,
            article_id=aid,
            text=text,
            raw_entities=raw_ents,
            include_virtual_patient=include_virtual_patient,
        )
        all_edge_records.extend(case_edges)

    return all_edge_records


def save_weak_edges(edge_records: Sequence[Dict[str, Any]], out_dir: Union[str, Path]) -> Path:
    """Salva os pares candidatos e seus rotulos fracos em formato JSONL."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_path / "weak_edges.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in edge_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return jsonl_path


def summarize_edges(edge_records: Sequence[Dict[str, Any]]) -> None:
    """Exibe estatisticas detalhadas dos pares gerados e dos rotulos fracos de arestas."""
    total = len(edge_records)
    if total == 0:
        print("Nenhum par candidato gerado.")
        return

    n_cases = len({r["case_id"] for r in edge_records})
    by_label = Counter(r["label"] for r in edge_records)
    n_pos = total - by_label.get("NONE", 0)
    by_pair_type = Counter(
        f"{r['type_E1']}->{r['type_E2']}"
        for r in edge_records if r["label"] != "NONE"
    )

    print(f"Casos com pares: {n_cases} | Total de pares candidatos: {total}")
    print(f"Arestas positivas: {n_pos} ({100.0 * n_pos / total:.1f}%) | NONE (sem relacao): {by_label.get('NONE', 0)} ({100.0 * by_label.get('NONE', 0) / total:.1f}%)")
    print("Pares por rotulo de aresta:", dict(by_label.most_common()))
    print("Arestas positivas por par de tipos:", dict(by_pair_type.most_common()))