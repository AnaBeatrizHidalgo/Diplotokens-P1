"""
Passo 3: Extracao de Quantidades por Regex (Passo 0 do diagrama),
Associacao de Quantidades as Arestas por Proximidade (Passo 4 do diagrama),
e Construcao do Objeto `Graph` final (`graph.py`).
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from models.classificar_arestas.step1_weak_edges import (
    align_entities_to_tokens,
    split_sentences,
    tokenize_with_offsets,
)

# Importa Node, Edge, Graph de graph.py

from diplo_graph2.graph import (Edge, Graph, Node)
    


# =====================================================================
# 1. REGEX DE UNIDADES E QUANTIDADES
# =====================================================================
PRESSURE_UNIT = r"mmHg"
WEIGHT_UNIT = r"(?:kg|mg|mcg|ng|hg|g)"
VOLUME_UNIT = r"(?:ml|dl|l)"
TIME_UNIT_EXT = r"(?:min|hr|hrs|h|s|sec|m|days?|weeks?|months?|years?|y/o|yo|year-old|years-old|month-old)"
LENGTH_UNIT = r"(?:cm|mm)"
SPECIAL_UNIT = r"(?:iu|U|mmol)"

ANY_DENOM_UNIT = rf"(?:{WEIGHT_UNIT}|{VOLUME_UNIT}|{TIME_UNIT_EXT}|{LENGTH_UNIT}|{SPECIAL_UNIT})"

TOKEN_SEP_REGEX = re.compile(
    rf"""
        \d+(?:,\d{{3}})*(?:\.\d+)?\s?{LENGTH_UNIT}\s*[xX]\s*\d+(?:,\d{{3}})*(?:\.\d+)?\s?{LENGTH_UNIT}(?:\s*[xX]\s*\d+(?:,\d{{3}})*(?:\.\d+)?\s?{LENGTH_UNIT})?(?![A-Za-z])
        | \d+(?:,\d{{3}})*(?:\.\d+)?
        (?:
            /\d+(?:,\d{{3}})*(?:\.\d+)?(?:\s?{WEIGHT_UNIT}|\s?{VOLUME_UNIT}|\s?{PRESSURE_UNIT})?(?![A-Za-z])
            | /{TIME_UNIT_EXT}(?![A-Za-z])
            | (?:\s|-)?(?:{PRESSURE_UNIT}|{WEIGHT_UNIT}|{VOLUME_UNIT}|{TIME_UNIT_EXT}|{LENGTH_UNIT}|{SPECIAL_UNIT})(?:/{ANY_DENOM_UNIT})?(?![A-Za-z])
        )?
        | [A-Za-z]+(?:-[A-Za-z]+)*
        | [.,;:()!?]+
    """,
    re.VERBOSE | re.IGNORECASE,
)

QUANTITY_REGEX = re.compile(
    rf"""
        ^
        (?:
              (?P<dimension>\d+(?:,\d{{3}})*(?:\.\d+)?\s?{LENGTH_UNIT}\s*[xX]\s*\d+(?:,\d{{3}})*(?:\.\d+)?\s?{LENGTH_UNIT}(?:\s*[xX]\s*\d+(?:,\d{{3}})*(?:\.\d+)?\s?{LENGTH_UNIT})?)
            | (?P<num_val>\d+(?:,\d{{3}})*(?:\.\d+)?)
              (?:
                    (?P<pressure>/(?:\d+(?:,\d{{3}})*(?:\.\d+)?)\s?{PRESSURE_UNIT}(?![A-Za-z]) | \s?{PRESSURE_UNIT}(?![A-Za-z]))
                  | (?P<frequency>/{TIME_UNIT_EXT}(?![A-Za-z]))
                  | (?P<weight>\s?{WEIGHT_UNIT}(?:/{ANY_DENOM_UNIT})?(?![A-Za-z]))
                  | (?P<volume>\s?{VOLUME_UNIT}(?:/{ANY_DENOM_UNIT})?(?![A-Za-z]))
                  | (?P<time>(?:\s|-)?{TIME_UNIT_EXT}(?:/{ANY_DENOM_UNIT})?(?![A-Za-z]))
                  | (?P<lenght>\s?{LENGTH_UNIT}(?:/{ANY_DENOM_UNIT})?(?![A-Za-z]))
                  | (?P<especial>\s?{SPECIAL_UNIT}(?:/{ANY_DENOM_UNIT})?(?![A-Za-z]))
              )?
        )
        $
    """,
    re.VERBOSE | re.IGNORECASE,
)

IGNORE_PREV_WORDS = {"fig", "figs", "figure", "figures", "table", "ca"}

RULE_TO_COMPATIBLE_EDGES: Dict[str, set[str]] = {
    "dimension":   {"REVEALS", "DIAGNOSED_WITH", "LOCATED_IN"},
    "lenght":      {"REVEALS", "DIAGNOSED_WITH", "LOCATED_IN"},
    "weight":      {"TAKES", "REVEALS", "HAS_EFFECT", "DIAGNOSED_WITH"},
    "volume":      {"TAKES", "REVEALS", "HAS_EFFECT"},
    "especial":    {"REVEALS", "TAKES", "HAS_EFFECT"},
    "pressure":    {"REVEALS", "PRESENTS_WITH"},
    "frequency":   {"TAKES", "PRESENTS_WITH", "HAS_EFFECT"},
    "time":        {"PRESENTS_WITH", "HISTORY", "FAMILIAR_HISTORY", "UNDERWENT_EXAM", "TAKES", "HAS_EFFECT"},
    "numero_puro": {"UNDERWENT_EXAM", "PRESENTS_WITH", "TAKES", "REVEALS"},
}


# =====================================================================
# 2. EXTRACAO DE QUANTIDADES COM POSICAO EXATA NA SENTENCA
# =====================================================================
def _split_value_and_unit(token_str: str, matched_rule: str, match_obj: re.Match) -> Tuple[str, str]:
    """Separa o valor numerico e a unidade de medida da quantidade capturada."""
    if matched_rule == "dimension":
        unit_m = re.search(LENGTH_UNIT, token_str, re.IGNORECASE)
        unit = unit_m.group(0).lower() if unit_m else "cm"
        return token_str.strip(), unit

    if matched_rule == "numero_puro":
        return token_str.strip(), ""

    num_val = match_obj.group("num_val") or ""
    unit_raw = match_obj.group(matched_rule) or ""
    unit_clean = unit_raw.strip().lstrip("-")

    if matched_rule == "pressure" and unit_clean.startswith("/"):
        m_bp = re.match(r"^(/\d+(?:,\d{3})*(?:\.\d+)?)\s*(mmHg)$", unit_clean, re.I)
        if m_bp:
            return f"{num_val}{m_bp.group(1)}", m_bp.group(2)

    return num_val.strip(), unit_clean


def extract_sentence_quantities(sentence_text: str, sent_char_offset: int = 0) -> List[Dict[str, Any]]:
    """
    Executa o `TOKEN_SEP_REGEX` e `QUANTITY_REGEX` sobre uma sentenca,
    retornando cada quantidade com seu valor, unidade, regra e offsets de caractere.
    """
    matches = list(TOKEN_SEP_REGEX.finditer(sentence_text))
    tokens = [m.group(0) for m in matches]
    quantities: List[Dict[str, Any]] = []

    for idx, m_tok in enumerate(matches):
        token = m_tok.group(0)
        q_match = QUANTITY_REGEX.match(token)
        if not q_match:
            continue

        group_dict = q_match.groupdict()
        rule_names = ["dimension", "pressure", "frequency", "weight", "volume", "time", "lenght", "especial"]
        matched_rule = next((r for r in rule_names if group_dict.get(r) is not None), "numero_puro")

        prev_tok = tokens[idx - 1].lower() if idx > 0 else ""
        prev2_tok = tokens[idx - 2].lower() if idx > 1 else ""
        if prev_tok in IGNORE_PREV_WORDS or prev2_tok in IGNORE_PREV_WORDS:
            continue
        if "year-old" in token.lower() or "years-old" in token.lower() or "month-old" in token.lower():
            continue
        if matched_rule == "numero_puro" and prev_tok not in {"day", "pod", "stage", "grade", "type", "level"}:
            continue

        val_str, unit_str = _split_value_and_unit(token, matched_rule, q_match)
        if matched_rule == "numero_puro" and prev_tok in {"day", "pod"}:
            matched_rule = "time"
            unit_str = prev_tok

        quantities.append({
            "token_index": idx,
            "quantity": token,
            "value": val_str,
            "unit": unit_str,
            "matched_rule": matched_rule,
            "char_start": sent_char_offset + m_tok.start(),
            "char_end": sent_char_offset + m_tok.end(),
        })

    return quantities


# =====================================================================
# 3. ASSOCIADOR DE QUANTIDADES AS ARESTAS POR PROXIMIDADE (PASSO 4)
# =====================================================================
def _hydrate_edge_positions(
    edge: Dict[str, Any],
    sentences: Sequence[Dict[str, Any]],
    all_tokens: Sequence[Dict[str, Any]],
    case_text: str,
) -> Dict[str, Any]:
    """
    Garante que o dicionario da aresta possua `sent_id`, `e1_tok_start`, `e1_tok_end`,
    `e2_tok_start`, `e2_tok_end`, `e1_norm` e `e2_norm`, mesmo quando a aresta
    foi filtrada/resumida antes de ser passada para `build_case_graph`.
    """
    ecopy = dict(edge)
    if "e1_norm" not in ecopy:
        ecopy["e1_norm"] = " ".join(
            t["norm"] for t in tokenize_with_offsets(str(ecopy.get("e1_text", "")))
        )
    if "e2_norm" not in ecopy:
        ecopy["e2_norm"] = " ".join(
            t["norm"] for t in tokenize_with_offsets(str(ecopy.get("e2_text", "")))
        )

    if "e2_tok_start" in ecopy and "e2_tok_end" in ecopy and "sent_id" in ecopy:
        return ecopy

    e2_text = str(ecopy.get("e2_text", "")).strip()
    e1_text = str(ecopy.get("e1_text", "")).strip()
    span_between = str(ecopy.get("span_between", "")).strip()

    # Procura em qual sentenca estão e2_text (e span_between / e1_text)
    best_sent = None
    for sent in sentences:
        stext_low = sent["text"].lower()
        if e2_text and e2_text.lower() in stext_low:
            if not span_between or span_between.lower() in stext_low:
                best_sent = sent
                break
            if best_sent is None:
                best_sent = sent

    if best_sent is None and sentences:
        best_sent = sentences[0]

    if best_sent is not None:
        ecopy.setdefault("sent_id", best_sent["sent_id"])
        s_char = best_sent["char_start"]
        stext_low = best_sent["text"].lower()

        # Localiza tokens de E2 dentro da sentenca
        idx2 = stext_low.find(e2_text.lower()) if e2_text else -1
        if idx2 != -1:
            c2_s = s_char + idx2
            c2_e = c2_s + len(e2_text)
            toks2 = [
                t["tok_idx"] for t in best_sent["tokens"]
                if t["char_end"] > c2_s and t["char_start"] < c2_e
            ]
            if toks2:
                ecopy.setdefault("e2_tok_start", toks2[0])
                ecopy.setdefault("e2_tok_end", toks2[-1] + 1)

        # Localiza tokens de E1 dentro da sentenca
        if ecopy.get("type_E1") == "Patient" and e1_text.lower() == "patient":
            ecopy.setdefault("e1_tok_start", best_sent["tok_start"])
            ecopy.setdefault("e1_tok_end", best_sent["tok_start"])
            ecopy.setdefault("e1_is_virtual", True)
        else:
            idx1 = stext_low.find(e1_text.lower()) if e1_text else -1
            if idx1 != -1:
                c1_s = s_char + idx1
                c1_e = c1_s + len(e1_text)
                toks1 = [
                    t["tok_idx"] for t in best_sent["tokens"]
                    if t["char_end"] > c1_s and t["char_start"] < c1_e
                ]
                if toks1:
                    ecopy.setdefault("e1_tok_start", toks1[0])
                    ecopy.setdefault("e1_tok_end", toks1[-1] + 1)

    ecopy.setdefault("e1_tok_start", 0)
    ecopy.setdefault("e1_tok_end", 0)
    ecopy.setdefault("e2_tok_start", 0)
    ecopy.setdefault("e2_tok_end", min(1, len(all_tokens)))
    return ecopy


def _char_distance_to_edge(
    q_start: int,
    q_end: int,
    edge: Dict[str, Any],
    all_tokens: Sequence[Dict[str, Any]],
) -> int:
    """
    Calcula a menor distancia em caracteres entre a quantidade (q_start, q_end)
    e os nos (E1, E2) da aresta.
    """
    n_toks = len(all_tokens)
    if n_toks == 0:
        return 9999

    e2_start_idx = max(0, min(int(edge.get("e2_tok_start", 0)), n_toks - 1))
    e2_end_idx = max(0, min(int(edge.get("e2_tok_end", e2_start_idx + 1)) - 1, n_toks - 1))

    e2_s = all_tokens[e2_start_idx]["char_start"]
    e2_e = all_tokens[e2_end_idx]["char_end"]

    if q_end <= e2_s:
        dist_e2 = e2_s - q_end
    elif e2_e <= q_start:
        dist_e2 = q_start - e2_e
    else:
        dist_e2 = 0

    if edge.get("e1_is_virtual", False) or edge.get("e1_tok_start") == edge.get("e1_tok_end"):
        return dist_e2

    e1_start_idx = max(0, min(int(edge.get("e1_tok_start", 0)), n_toks - 1))
    e1_end_idx = max(0, min(int(edge.get("e1_tok_end", e1_start_idx + 1)) - 1, n_toks - 1))
    e1_s = all_tokens[e1_start_idx]["char_start"]
    e1_e = all_tokens[e1_end_idx]["char_end"]

    if q_end <= e1_s:
        dist_e1 = e1_s - q_end
    elif e1_e <= q_start:
        dist_e1 = q_start - e1_e
    else:
        dist_e1 = 0

    return min(dist_e2, dist_e1)


def associate_quantities_to_edges(
    case_text: str,
    positive_edges: Sequence[Dict[str, Any]],
    max_char_dist: int = 110,
) -> List[Dict[str, Any]]:
    """
    Para cada sentenca do caso, extrai as quantidades por Regex e associa cada
    quantidade a aresta positiva semanticamente compativel mais proxima no texto.
    """
    tokens = tokenize_with_offsets(case_text)
    sentences = split_sentences(case_text, tokens)

    enriched_edges = []
    for edge in positive_edges:
        ecopy = _hydrate_edge_positions(edge, sentences, tokens, case_text)
        ecopy["attributes"] = dict(edge.get("attributes", {}))
        enriched_edges.append(ecopy)

    for sent in sentences:
        sent_id = sent["sent_id"]
        sent_edges = [e for e in enriched_edges if e.get("sent_id") == sent_id]
        if not sent_edges:
            continue

        quantities = extract_sentence_quantities(sent["text"], sent_char_offset=sent["char_start"])
        for q in quantities:
            rule = q["matched_rule"]
            compatible_rels = RULE_TO_COMPATIBLE_EDGES.get(rule, set())

            candidates = [
                e for e in sent_edges
                if e.get("pred_label", e.get("edge_label", e.get("label"))) in compatible_rels
            ]
            if not candidates:
                candidates = sent_edges

            best_edge = min(
                candidates,
                key=lambda e: _char_distance_to_edge(q["char_start"], q["char_end"], e, tokens),
            )
            dist = _char_distance_to_edge(q["char_start"], q["char_end"], best_edge, tokens)
            if dist <= max_char_dist:
                attrs = best_edge["attributes"]
                if "value" not in attrs:
                    attrs["value"] = q["value"]
                    attrs["unit"] = q["unit"]
                    attrs["type"] = rule
                    attrs["quantity"] = q["quantity"]
                else:
                    idx_extra = sum(1 for k in attrs if k.startswith("value")) + 1
                    attrs[f"value_{idx_extra}"] = q["value"]
                    attrs[f"unit_{idx_extra}"] = q["unit"]
                    attrs[f"type_{idx_extra}"] = rule

    return enriched_edges





def build_case_graph(
    case_row: Dict[str, Any],
    raw_entities: Sequence[Dict[str, Any]],
    positive_edges: Sequence[Dict[str, Any]],
    text_col: str = "case_text",
    deduplicate_nodes: bool = True,
) -> Graph:
    """
    Constroi e retorna uma instancia de `Graph` (definida em `graph.py`) para um caso clinico.
    """
    case_id = str(case_row.get("case_id", ""))
    text = str(case_row.get(text_col, "") or "")
    tokens = tokenize_with_offsets(text)
    aligned_ents = align_entities_to_tokens(raw_entities, text, tokens)

    graph = Graph()

    # 1. No raiz: Paciente (indice 0)
    patient_attrs: Dict[str, str] = {"case_id": case_id}
    if case_row.get("age") is not None and str(case_row.get("age")) != "nan":
        patient_attrs["age"] = str(case_row["age"])
    if case_row.get("gender") is not None and str(case_row.get("gender")) != "nan":
        patient_attrs["gender"] = str(case_row["gender"])

    patient_node_idx = len(graph.nodes)
    graph.add_node(
        Node(
            node_type="Patient",
            label=f"Patient ({case_id})",
            attributes=patient_attrs,
            coord_in_case=(0, 0),
            pos_in_case_text=0,
        )
    )

    # 2. Registra nos clinicos e mapeia tanto por span quanto por texto/norm
    span_to_node_idx: Dict[Tuple[int, int, str], int] = {}
    canonical_key_to_idx: Dict[Tuple[str, str], int] = {}
    label_key_to_idx: Dict[Tuple[str, str], int] = {}

    for ent in aligned_ents:
        etype = ent["label"]
        elabel = ent["text"]
        tok_s, tok_e = ent["tok_start"], ent["tok_end"]
        char_s, char_e = ent["char_start"], ent["char_end"]

        if etype == "Patient":
            span_to_node_idx[(tok_s, tok_e, "Patient")] = patient_node_idx
            continue

        canon_key = (ent["norm"], etype)
        label_key = (elabel.lower().strip(), etype)
        if deduplicate_nodes and canon_key in canonical_key_to_idx:
            node_idx = canonical_key_to_idx[canon_key]
        else:
            node_idx = len(graph.nodes)
            graph.add_node(
                Node(
                    node_type=etype,
                    label=elabel,
                    attributes={},
                    coord_in_case=(char_s, char_e),
                    pos_in_case_text=tok_s,
                )
            )
            canonical_key_to_idx[canon_key] = node_idx
            label_key_to_idx[label_key] = node_idx

        span_to_node_idx[(tok_s, tok_e, etype)] = node_idx

    # 3. Associa quantidades (valores e unidades) as arestas por proximidade
    edges_with_attrs = associate_quantities_to_edges(text, positive_edges)

    # 4. Insere as arestas no objeto Graph
    seen_edges = set()
    for e_rec in edges_with_attrs:
        rel = e_rec.get("pred_label", e_rec.get("edge_label", e_rec.get("label", "NONE")))
        if rel == "NONE":
            continue

        t1 = e_rec["type_E1"]
        t2 = e_rec["type_E2"]

        # Resolve indice do no de origem (from_node)
        if t1 == "Patient":
            u_idx = patient_node_idx
        else:
            u_key = (e_rec.get("e1_tok_start", -1), e_rec.get("e1_tok_end", -1), t1)
            u_idx = (
                span_to_node_idx.get(u_key)
                or canonical_key_to_idx.get((e_rec.get("e1_norm", ""), t1))
                or label_key_to_idx.get((str(e_rec.get("e1_text", "")).lower().strip(), t1))
            )

        # Resolve indice do no de destino (to_node)
        if t2 == "Patient":
            v_idx = patient_node_idx
        else:
            v_key = (e_rec.get("e2_tok_start", -1), e_rec.get("e2_tok_end", -1), t2)
            v_idx = (
                span_to_node_idx.get(v_key)
                or canonical_key_to_idx.get((e_rec.get("e2_norm", ""), t2))
                or label_key_to_idx.get((str(e_rec.get("e2_text", "")).lower().strip(), t2))
            )

        if u_idx is None or v_idx is None:
            continue

        edge_attrs: Dict[str, str] = {
            k: str(v) for k, v in e_rec.get("attributes", {}).items() if v != ""
        }
        conf_val = e_rec.get("pred_confidence", e_rec.get("confidence"))
        if conf_val is not None:
            edge_attrs["confidence"] = str(conf_val)

        dedup_sig = (u_idx, v_idx, rel, edge_attrs.get("quantity", ""))
        if dedup_sig in seen_edges:
            continue
        seen_edges.add(dedup_sig)

        graph.add_edge_c(u_idx, v_idx, rel, edge_attrs)

    return graph