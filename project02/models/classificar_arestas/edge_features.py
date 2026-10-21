"""
Passo 2 da Classificacao de Arestas (Box 3 - Regressao Logistica Multinomial):
Extracao de features para cada par candidato (E1, E2) + texto intermediario:
  1. Tipos das Entidades (type_E1, type_E2, par direcionado e ordem de leitura)
  2. N-gramas / TF-IDF do texto entre as entidades (span_between) e contexto local
  3. Metrica PMI (Pointwise Mutual Information) entre (E1, E2)
  4. Distancia em tokens (token_dist, log_dist, faixas de distancia e barreiras de oracao)
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from project02.models.classificar_arestas.step1_weak_edges import (
    VALID_SCHEMA_PAIRS,
    generate_case_edge_candidates,
)


def _headword(norm_text: str) -> str:
    """
    Retorna a palavra-cabeca (headword) da entidade (ultimo token em ingles,
    ex: 'cystic lesion' -> 'lesion', 'ct scan' -> 'scan').
    """
    parts = norm_text.strip().split()
    return parts[-1] if parts else norm_text


# =====================================================================
# 1. EXTRACAO DE FEATURES ESTRUTURADAS POR PAR (E1, E2)
# =====================================================================
def extract_pair_struct_features(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Constroi o dicionario de features estruturadas de um par (E1, E2):
      1. Tipos de E1 e E2 e sua interacao (pair_type)
      2. Distancia em tokens (continua, logaritmica e categorizada)
      3. Ordem de leitura e estrutura superficial da oracao
    """
    t1 = rec["type_E1"]
    t2 = rec["type_E2"]
    pair_type = f"{t1}->{t2}"
    order = rec.get("reading_order", "E1_before_E2")
    dist = int(rec.get("token_dist", 0))
    same_clause = bool(rec.get("same_clause", True))
    is_valid_pair = (t1, t2) in VALID_SCHEMA_PAIRS

    # Faixa de distancia em tokens
    if dist == 0:
        dist_bin = "adj_0"
    elif dist <= 3:
        dist_bin = "close_1_3"
    elif dist <= 8:
        dist_bin = "med_4_8"
    elif dist <= 15:
        dist_bin = "far_9_15"
    else:
        dist_bin = "very_far_16+"

    log_dist = math.log1p(dist)
    scaled_dist = dist / 10.0

    e1_head = _headword(rec.get("e1_norm", ""))
    e2_head = _headword(rec.get("e2_norm", ""))

    feats: Dict[str, Any] = {
        # 1. Tipos das Entidades no Par
        f"type_E1={t1}": 1.0,
        f"type_E2={t2}": 1.0,
        f"pair={pair_type}": 1.0,
        f"order={order}": 1.0,
        f"pair_order={pair_type}|{order}": 1.0,
        "valid_schema_pair": 1.0 if is_valid_pair else 0.0,
        "e1_is_virtual": 1.0 if rec.get("e1_is_virtual", False) else 0.0,

        # 2. Features de Distancia
        "token_dist_scaled": scaled_dist,
        "log_dist": log_dist,
        f"dist_bin={dist_bin}": 1.0,
        f"pair_dist={pair_type}|{dist_bin}": 1.0,
        f"pair_log_dist={pair_type}": log_dist,

        # 3. Mesma oracao e palavras-cabeca das entidades
        "same_clause": 1.0 if same_clause else 0.0,
        f"pair_clause={pair_type}|same={same_clause}": 1.0,
        f"head_E1={e1_head}": 1.0,
        f"head_E2={e2_head}": 1.0,
    }

    intervening = rec.get("intervening_types", [])
    feats["num_intervening"] = float(len(intervening))
    if t1 in intervening or t2 in intervening:
        feats["has_same_type_intervening"] = 1.0
        feats[f"pair_intervening={pair_type}"] = 1.0

    return feats


# =====================================================================
# 2. CONSTRUCAO DO DATASET DE PARES CANDIDATOS
# =====================================================================
def build_edge_dataset(
    edge_records: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], np.ndarray, List[Tuple[str, int]]]:
    """
    Transforma a lista de pares candidatos em (X, y, ids) para treino, validacao ou teste.
    Cada item de X contem:
      - 'struct': dicionario de features estruturadas (tipos, distancia, etc.)
      - 'span_between': texto intermediario entre E1 e E2 (para TF-IDF de n-gramas)
      - 'context_text': janela esquerda + direita ao redor do par (para TF-IDF de contexto)
      - metadados normalizados de E1 e E2 (para calculo do PMI)
    """
    X: List[Dict[str, Any]] = []
    y_list: List[str] = []
    ids: List[Tuple[str, int]] = []

    for idx, rec in enumerate(edge_records):
        struct_feats = extract_pair_struct_features(rec)
        span_text = str(rec.get("span_between", ""))
        left_ctx = str(rec.get("left_context", ""))
        right_ctx = str(rec.get("right_context", ""))

        X.append({
            "struct": struct_feats,
            "span_between": span_text,
            "context_text": f"{left_ctx} {right_ctx}".strip(),
            "e1_norm": rec.get("e1_norm", ""),
            "e2_norm": rec.get("e2_norm", ""),
            "e1_head": _headword(rec.get("e1_norm", "")),
            "e2_head": _headword(rec.get("e2_norm", "")),
            "type_E1": rec["type_E1"],
            "type_E2": rec["type_E2"],
            "case_id": rec["case_id"],
            "sent_id": rec["sent_id"],
        })
        y_list.append(rec.get("label", "NONE"))
        ids.append((rec["case_id"], idx))

    return X, np.array(y_list, dtype=object), ids


# =====================================================================
# 3. TRANSFORMADOR UNIFICADO: PMI + TF-IDF (SPAN) + DICT VECTORIZER
# =====================================================================
class EdgeFeatureVectorizer(BaseEstimator, TransformerMixin):
    """
    Transformador compativel com scikit-learn que combina as 4 familias de features da Box 3:
      1. PMI (Pointwise Mutual Information) ajustado apenas nos dados de treino:
         PMI(E1, E2) = log2( P(E1, E2) / (P(E1) * P(E2)) )
      2. Features estruturadas (type_E1, type_E2, token_dist, log_dist, etc.) via DictVectorizer
      3. TF-IDF de N-gramas (1, 2) do texto intermediario (`span_between`)
      4. TF-IDF de N-gramas (1, 2) da janela de contexto imediato (`left_context` + `right_context`)
    """

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        max_span_features: int = 1500,
        max_ctx_features: int = 800,
        pmi_smoothing: float = 0.5,
    ):
        self.ngram_range = ngram_range
        self.max_span_features = max_span_features
        self.max_ctx_features = max_ctx_features
        self.pmi_smoothing = pmi_smoothing

    def fit(self, X: Sequence[Dict[str, Any]], y: Optional[Sequence[Any]] = None) -> "EdgeFeatureVectorizer":
        # 1. Calcula contagens para PMI exclusivamente no conjunto de treino recebido
        self._fit_pmi_tables(X)

        # 2. Enriquece o dicionario 'struct' de cada amostra com os scores de PMI
        struct_dicts: Any = [self._augment_with_pmi(item) for item in X]
        self.dict_vec_ = DictVectorizer(sparse=True)
        self.dict_vec_.fit(struct_dicts)

        # 3. Ajusta TF-IDF para o texto ENTRE as entidades (span_between)
        span_texts = [str(item.get("span_between", "")) for item in X]
        self.span_tfidf_ = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_span_features,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w[\w'-]*\b",
            lowercase=True,
        )
        if any(t.strip() for t in span_texts):
            self.span_tfidf_.fit(span_texts)
            self._has_span_vocab_ = len(self.span_tfidf_.vocabulary_) > 0
        else:
            self._has_span_vocab_ = False

        # 4. Ajusta TF-IDF para o contexto local ao redor do par
        ctx_texts = [str(item.get("context_text", "")) for item in X]
        self.ctx_tfidf_ = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_ctx_features,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w[\w'-]*\b",
            lowercase=True,
        )
        if any(t.strip() for t in ctx_texts):
            self.ctx_tfidf_.fit(ctx_texts)
            self._has_ctx_vocab_ = len(self.ctx_tfidf_.vocabulary_) > 0
        else:
            self._has_ctx_vocab_ = False

        return self

    def transform(self, X: Sequence[Dict[str, Any]]) -> sparse.csr_matrix:
        struct_dicts: Any = [self._augment_with_pmi(item) for item in X]
        X_struct = sparse.csr_matrix(self.dict_vec_.transform(struct_dicts))

        matrices: List[sparse.csr_matrix] = [X_struct]

        if getattr(self, "_has_span_vocab_", False):
            span_texts = [str(item.get("span_between", "")) for item in X]
            X_span = sparse.csr_matrix(self.span_tfidf_.transform(span_texts))
            matrices.append(X_span)

        if getattr(self, "_has_ctx_vocab_", False):
            ctx_texts = [str(item.get("context_text", "")) for item in X]
            X_ctx = sparse.csr_matrix(self.ctx_tfidf_.transform(ctx_texts))
            matrices.append(X_ctx)

        return sparse.csr_matrix(sparse.hstack(matrices, format="csr"))

    def get_feature_names_out(self) -> np.ndarray:
        """Retorna os nomes de todas as colunas (estruturadas + TF-IDF span + TF-IDF contexto)."""
        names = list(self.dict_vec_.get_feature_names_out())
        if getattr(self, "_has_span_vocab_", False):
            names.extend([f"span_tfidf:{w}" for w in self.span_tfidf_.get_feature_names_out()])
        if getattr(self, "_has_ctx_vocab_", False):
            names.extend([f"ctx_tfidf:{w}" for w in self.ctx_tfidf_.get_feature_names_out()])
        return np.array(names, dtype=object)

    def _fit_pmi_tables(self, X: Sequence[Dict[str, Any]]) -> None:
        self.n_total_ = max(1, len(X))
        self.e1_counts_: Counter = Counter()
        self.e2_counts_: Counter = Counter()
        self.pair_lex_counts_: Counter = Counter()

        self.h1_counts_: Counter = Counter()
        self.h2_counts_: Counter = Counter()
        self.pair_head_counts_: Counter = Counter()

        self.t1_counts_: Counter = Counter()
        self.t2_counts_: Counter = Counter()
        self.pair_type_counts_: Counter = Counter()

        for item in X:
            e1, e2 = item["e1_norm"], item["e2_norm"]
            h1, h2 = item["e1_head"], item["e2_head"]
            t1, t2 = item["type_E1"], item["type_E2"]

            self.e1_counts_[e1] += 1
            self.e2_counts_[e2] += 1
            self.pair_lex_counts_[(e1, e2)] += 1

            self.h1_counts_[h1] += 1
            self.h2_counts_[h2] += 1
            self.pair_head_counts_[(h1, h2)] += 1

            self.t1_counts_[t1] += 1
            self.t2_counts_[t2] += 1
            dist_scaled = item["struct"].get("token_dist_scaled", 0.0)
            if dist_scaled <= 1.0:
                self.pair_type_counts_[(t1, t2)] += 1

    def _compute_single_pmi(self, c_pair: float, c1: float, c2: float, n_total: float) -> float:
        alpha = self.pmi_smoothing
        p_xy = (c_pair + alpha) / (n_total + alpha)
        p_x = (c1 + alpha) / (n_total + alpha)
        p_y = (c2 + alpha) / (n_total + alpha)
        return math.log2(p_xy / (p_x * p_y))

    def _augment_with_pmi(self, item: Dict[str, Any]) -> Dict[str, Any]:
        feats = dict(item["struct"])
        e1, e2 = item["e1_norm"], item["e2_norm"]
        h1, h2 = item["e1_head"], item["e2_head"]
        t1, t2 = item["type_E1"], item["type_E2"]
        pair_type = f"{t1}->{t2}"

        pmi_lex = self._compute_single_pmi(
            self.pair_lex_counts_[(e1, e2)], self.e1_counts_[e1], self.e2_counts_[e2], self.n_total_
        )
        pmi_head = self._compute_single_pmi(
            self.pair_head_counts_[(h1, h2)], self.h1_counts_[h1], self.h2_counts_[h2], self.n_total_
        )
        pmi_type = self._compute_single_pmi(
            self.pair_type_counts_[(t1, t2)], self.t1_counts_[t1], self.t2_counts_[t2], self.n_total_
        )

        ppmi_lex = max(0.0, pmi_lex)
        ppmi_head = max(0.0, pmi_head)

        feats["pmi_lex"] = pmi_lex
        feats["ppmi_lex"] = ppmi_lex
        feats["pmi_head"] = pmi_head
        feats["ppmi_head"] = ppmi_head
        feats["pmi_type"] = pmi_type
        feats[f"pair_ppmi={pair_type}"] = ppmi_head

        return feats


# =====================================================================
# 4. MODELO DE REGRESSAO LOGISTICA MULTINOMIAL E INFERENCIA (SOFTMAX)
# =====================================================================
def make_edge_model(C: float = 1.0, seed: int = 42) -> Pipeline:
    """
    Cria o pipeline completo da Box 3:
      EdgeFeatureVectorizer (Tipos + TF-IDF Span + PMI + Distancia)
      -> LogisticRegression Multinomial (com saida Softmax via predict_proba).
    """
    return Pipeline([
        ("vec", EdgeFeatureVectorizer()),
        ("clf", LogisticRegression(
            C=C,
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        )),
    ])


def predict_edges_with_proba(
    model: Pipeline,
    edge_records: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Calcula o vetor de probabilidades Softmax para cada classe de aresta
    e retorna os pares enriquecidos com `pred_label`, `pred_confidence` e `probabilities`.
    """
    if not edge_records:
        return []

    X_eval, _, _ = build_edge_dataset(edge_records)
    proba_matrix = model.predict_proba(X_eval)
    classes = list(model.classes_)

    results: List[Dict[str, Any]] = []
    for rec, probs in zip(edge_records, proba_matrix):
        prob_dict = {cls_name: round(float(p), 4) for cls_name, p in zip(classes, probs)}

        pair_tuple = (rec["type_E1"], rec["type_E2"])
        allowed_positive = set(VALID_SCHEMA_PAIRS.get(pair_tuple, []))
        allowed_classes = allowed_positive | {"NONE"}

        best_label = "NONE"
        best_prob = -1.0
        for cls_name, p in zip(classes, probs):
            if cls_name in allowed_classes and float(p) > best_prob:
                best_label = cls_name
                best_prob = float(p)

        out_rec = dict(rec)
        out_rec["pred_label"] = best_label
        out_rec["pred_confidence"] = round(best_prob, 4)
        out_rec["probabilities"] = prob_dict
        out_rec["pmi_head"] = round(
            float(model.steps[0][1]._augment_with_pmi(X_eval[len(results)])["pmi_head"]),
            3,
        )
        results.append(out_rec)

    return results


def predict_case_edges(
    model: Pipeline,
    case_id: str,
    article_id: str,
    text: str,
    entities: Sequence[Dict[str, Any]],
    only_positive: bool = True,
) -> List[Dict[str, Any]]:
    """
    Gera todos os pares candidatos de um caso de teste, aplica a Regressao
    Logistica Multinomial e retorna as arestas do grafo.
    """
    candidates = generate_case_edge_candidates(
        case_id=case_id,
        article_id=article_id,
        text=text,
        raw_entities=entities,
    )
    predicted = predict_edges_with_proba(model, candidates)
    if only_positive:
        return [r for r in predicted if r["pred_label"] != "NONE"]
    return predicted