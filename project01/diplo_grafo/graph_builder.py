from __future__ import annotations

import re
from typing import NamedTuple

import pandas as pd

from diplo_grafo.clinical_vocabulary import (
    ENTITY_CATEGORIES,
    RE_DIMENSION_3D,
    RE_DIMENSION_2D,
    RE_DIMENSION_1D,
    RE_LAB_MEASUREMENT,
    RE_DURATION_HISTORY,
    RE_DURATION_GENERIC,
    RE_DOSAGE,
    RE_TIMELINE_POD,
    RE_LATERALITY,
    RE_ASPECT,
    RE_NEGATION,
)

# Aliases privados — mantém os nomes internos do módulo inalterados,
# evitando alterações em cascata no restante do arquivo.
_RE_DIMENSION_3D      = RE_DIMENSION_3D
_RE_DIMENSION_2D      = RE_DIMENSION_2D
_RE_DIMENSION_1D      = RE_DIMENSION_1D
_RE_LAB_MEASUREMENT   = RE_LAB_MEASUREMENT
_RE_DURATION_HISTORY  = RE_DURATION_HISTORY
_RE_DURATION_GENERIC  = RE_DURATION_GENERIC
_RE_DOSAGE            = RE_DOSAGE
_RE_TIMELINE_POD      = RE_TIMELINE_POD
_RE_LATERALITY        = RE_LATERALITY
_RE_ASPECT            = RE_ASPECT
_RE_NEGATION          = RE_NEGATION


# ==============================================================================
# TIPOS AUXILIARES
# ==============================================================================

class _Span(NamedTuple):
    """Posição (start, end) de uma ocorrência no texto + conteúdo."""
    start: int
    end: int
    text: str


class _SentenceContext(NamedTuple):
    """Dados extraídos de uma única sentença antes de gerar nós/arestas."""
    symptoms: list[str]
    diseases: list[str]
    exams: list[str]
    results: list[str]
    anatomy: list[str]
    meds: list[str]
    assigned_sizes: dict[str, str]
    assigned_labs: dict[str, list[str]]
    dosages: list[_Span]
    duration: str
    timeline: str
    laterality: str
    aspect: str


# ==============================================================================
# CLASSE PRINCIPAL
# ==============================================================================

class KnowledgeGraphBuilder:
    """Constrói um grafo de conhecimento clínico a partir de sentenças tokenizadas.

    O ciclo de vida típico é:
    1. Instanciar (ou reutilizar) o builder.
    2. Chamar ``build(case_row, sentences)`` para obter os DataFrames.

    O builder é stateful: cada chamada a ``build`` reseta o estado interno,
    portanto a mesma instância pode ser reutilizada para casos diferentes.

    Attributes de classe
    --------------------
    NODE_PREFIXES : dict
        Mapeamento de tipo de nó para prefixo de ID (ex: "Symptom" → "S").
        Definido aqui pois é uma convenção de representação do grafo, não
        vocabulário clínico.
    ENTITY_CATEGORIES : dict
        Vocabulário clínico categorizado, importado de ``clinical_vocabulary``.
    """

    # ------------------------------------------------------------------
    # Atributos de classe
    # NODE_PREFIXES: convenção de IDs do grafo — pertence ao builder.
    # ENTITY_CATEGORIES: vocabulário clínico — vem de clinical_vocabulary.py
    # e é referenciado aqui por clareza.
    # ------------------------------------------------------------------

    NODE_PREFIXES: dict[str, str] = {
        "Patient":    "P",
        "Symptom":    "S",
        "Disease":    "D",
        "Exam":       "E",
        "Result":     "R",
        "Anatomy":    "A",
        "Procedure":  "Pr",
        "Medication": "M",
        "Effect":     "Ef",
    }

    ENTITY_CATEGORIES = ENTITY_CATEGORIES          # fonte: clinical_vocabulary.py

    # ------------------------------------------------------------------
    # Inicialização e estado de instância
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        # Estado mutável do grafo — resetado a cada chamada a build()
        self._nodes: list[dict]         = []
        self._edges: list[dict]         = []
        self._node_ids: dict[tuple, str] = {}

        # Índice invertido term→tipo construído uma vez a partir de ENTITY_CATEGORIES
        self._term_to_type: dict[str, str] = {
            term.lower(): category
            for category, terms in self.ENTITY_CATEGORIES.items()
            for term in terms
        }

    def _reset(self) -> None:
        """Limpa o estado interno para permitir reuso da instância."""
        self._nodes.clear()
        self._edges.clear()
        self._node_ids.clear()

    # ------------------------------------------------------------------
    # Propriedades de inspeção (somente leitura)
    # ------------------------------------------------------------------

    @property
    def nodes_df(self) -> pd.DataFrame:
        """DataFrame dos nós do grafo construído mais recentemente."""
        return pd.DataFrame(self._nodes)

    @property
    def edges_df(self) -> pd.DataFrame:
        """DataFrame das arestas do grafo construído mais recentemente."""
        return pd.DataFrame(self._edges)

    # ------------------------------------------------------------------
    # Métodos públicos de manipulação do grafo
    # ------------------------------------------------------------------

    def add_node(self, label: str, node_type: str, attributes: str = "") -> str:
        """Insere ou recupera um nó, atualizando atributos se necessário.

        Parameters
        ----------
        label : str
            Rótulo legível do nó (ex: "gastric duplication cyst").
        node_type : str
            Tipo semântico do nó (ex: "Disease"). Deve ser uma chave de NODE_PREFIXES.
        attributes : str
            String de atributos no formato "chave=valor; chave=valor".

        Returns
        -------
        str
            ID único do nó (ex: "D3").
        """
        key = (label.lower(), node_type)

        if key in self._node_ids:
            # Atualiza atributos se o registro existente ainda não os possui
            existing_id = self._node_ids[key]
            for node in self._nodes:
                if node["node_id"] == existing_id:
                    if not node["attributes"] and attributes:
                        node["attributes"] = attributes
                    elif attributes and attributes not in node["attributes"]:
                        node["attributes"] += f"; {attributes}"
            return existing_id

        prefix = self.NODE_PREFIXES.get(node_type, "N")
        count  = sum(1 for k in self._node_ids if k[1] == node_type)
        nid    = f"{prefix}{count + 1}"

        self._node_ids[key] = nid
        self._nodes.append({
            "node_id":    nid,
            "type":       node_type,
            "label":      label,
            "attributes": attributes,
        })
        return nid

    def add_edge(
        self,
        source_id:  str,
        target_id:  str,
        relation:   str,
        attributes: str = "",
    ) -> str:
        """Insere uma aresta direcionada entre dois nós existentes.

        Parameters
        ----------
        source_id : str
            ID do nó de origem (ex: "P1").
        target_id : str
            ID do nó de destino (ex: "S1").
        relation : str
            Tipo de relação semântica (ex: "PRESENTS_WITH").
        attributes : str
            Atributos da aresta no formato "chave=valor; chave=valor".

        Returns
        -------
        str
            ID único da aresta (ex: "e7").
        """
        eid = f"e{len(self._edges) + 1}"
        self._edges.append({
            "edge_id":   eid,
            "source_id": source_id,
            "target_id": target_id,
            "relation":  relation,
            "attributes": attributes,
        })
        return eid

    # ------------------------------------------------------------------
    # Método público principal
    # ------------------------------------------------------------------

    def build(self, case_row: pd.Series, sentences: list) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Constrói o grafo de conhecimento para um caso clínico.

        Percorre as sentenças tokenizadas, extrai atributos por regex e aplica
        as regras semânticas para gerar nós e arestas.

        Parameters
        ----------
        case_row : pd.Series
            Linha do DataFrame de casos com colunas ``case_id``, ``age``, ``gender``.
        sentences : list[Sentence]
            Lista de objetos ``Sentence`` já tokenizados (com ``s.tokens`` preenchido).

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Par ``(nodes_df, edges_df)`` compatível com o visualizador Diploghraph.
        """
        self._reset()

        # Nó do paciente com atributos demográficos
        patient_attrs = f"age={case_row['age']}; gender={case_row['gender']}"
        patient_id = self.add_node(
            f"Patient {case_row['case_id']}",
            "Patient",
            patient_attrs,
        )

        for sentence in sentences:
            ctx = self._extract_sentence_context(sentence)

            self._apply_symptoms(patient_id, ctx)
            self._apply_medications(patient_id, ctx, sentence.text)
            self._apply_exams(patient_id, ctx, sentence.text)
            self._apply_diseases(patient_id, ctx, sentence.text)

        return self.nodes_df, self.edges_df


    # ------------------------------------------------------------------
    # Métodos privados — extração de contexto por sentença
    # ------------------------------------------------------------------

    def _extract_sentence_context(self, sentence) -> _SentenceContext:
        """Orquestra toda a extração de uma sentença e retorna um contexto estruturado."""
        text = sentence.text

        # 1. Classificação de tokens em entidades ativas (não negadas)
        entity_groups = self._classify_tokens(sentence.tokens, text)

        results = entity_groups.get("Result", [])

        # Spans textuais dos resultados (usados para atribuição de medidas)
        res_spans = self._locate_spans(results, text)

        # 2. Extração de medidas e modificadores
        assigned_sizes = self._extract_sizes(text, res_spans)
        assigned_labs  = self._extract_labs(text, res_spans, results)
        modifiers      = self._extract_modifiers(text)

        return _SentenceContext(
            symptoms       = entity_groups.get("Symptom",    []),
            diseases       = entity_groups.get("Disease",    []),
            exams          = entity_groups.get("Exam",       []),
            results        = results,
            anatomy        = entity_groups.get("Anatomy",    []),
            meds           = entity_groups.get("Medication", []),
            assigned_sizes = assigned_sizes,
            assigned_labs  = assigned_labs,
            dosages        = modifiers["dosages"],
            duration       = modifiers["duration"],
            timeline       = modifiers["timeline"],
            laterality     = modifiers["laterality"],
            aspect         = modifiers["aspect"],
        )

    def _classify_tokens(self, tokens: list[str], text: str) -> dict[str, list[str]]:
        """Mapeia tokens reconhecidos para seus tipos, descartando os negados.

        Returns
        -------
        dict
            Chaves são tipos de entidade; valores são listas de tokens ativos.
        """
        found = [
            (tok, self._term_to_type[tok.lower()])
            for tok in tokens
            if tok.lower() in self._term_to_type
        ]

        active = [
            (tok, etype)
            for tok, etype in found
            if not self.is_negated(tok, text)
        ]

        groups: dict[str, list[str]] = {}
        for tok, etype in active:
            groups.setdefault(etype, []).append(tok)
        return groups

    def _locate_spans(self, entities: list[str], text: str) -> list[_Span]:
        """Localiza as posições textuais de uma lista de entidades no texto."""
        spans = []
        for entity in entities:
            m = re.search(re.escape(entity), text, re.IGNORECASE)
            if m:
                spans.append(_Span(m.start(), m.end(), entity))
        return spans

    def _extract_sizes(
        self,
        text: str,
        res_spans: list[_Span],
    ) -> dict[str, str]:
        """Extrai dimensões físicas (3D > 2D > 1D) e as associa ao resultado mais próximo.

        A prioridade 3D > 2D > 1D evita que "6cm" (componente de "6cm x 9cm")
        seja atribuído separadamente, sobrescrevendo a medida mais completa.
        """
        # Coleta todas as dimensões em ordem de prioridade
        raw_sizes: list[_Span] = []
        covered: set[int] = set()

        for pattern in (_RE_DIMENSION_3D, _RE_DIMENSION_2D, _RE_DIMENSION_1D):
            for m in pattern.finditer(text):
                span_range = range(m.start(), m.end())
                if not any(i in covered for i in span_range):
                    raw_sizes.append(_Span(m.start(), m.end(), m.group(0).strip()))
                    covered.update(span_range)

        assigned: dict[str, str] = {}
        for size_span in raw_sizes:
            if not res_spans:
                break
            closest = min(res_spans, key=lambda e: self._span_distance(e, size_span))
            if self._span_distance(closest, size_span) < 30:
                assigned[closest.text] = size_span.text

        return assigned

    def _extract_labs(
        self,
        text: str,
        res_spans: list[_Span],
        results: list[str],
    ) -> dict[str, list[str]]:
        """Associa medições laboratoriais (valor + unidade) ao resultado precedente mais próximo."""
        assigned: dict[str, list[str]] = {r: [] for r in results}

        for m in _RE_LAB_MEASUREMENT.finditer(text):
            lab_start = m.start()
            # Considera apenas entidades que aparecem ANTES da medição no texto
            preceding = [span for span in res_spans if span.end <= lab_start]
            if not preceding:
                continue
            closest = min(preceding, key=lambda e: lab_start - e.end)
            if (lab_start - closest.end) < 60:
                assigned[closest.text].append(f"value={m.group(1)}; unit={m.group(2)}")

        return assigned

    def _extract_modifiers(self, text: str) -> dict:
        """Extrai modificadores temporais, espaciais e quantitativos de uma sentença.

        Returns
        -------
        dict com chaves: ``duration``, ``timeline``, ``laterality``, ``aspect``, ``dosages``.
        """
        dur_match  = _RE_DURATION_HISTORY.search(text) or _RE_DURATION_GENERIC.search(text)
        pod_match  = _RE_TIMELINE_POD.search(text)
        lat_match  = _RE_LATERALITY.search(text)
        asp_match  = _RE_ASPECT.search(text)

        dosages = [
            _Span(m.start(), m.end(), m.group(0).strip())
            for m in _RE_DOSAGE.finditer(text)
        ]

        return {
            "duration":   dur_match.group(0) if dur_match  else "",
            "timeline":   pod_match.group(0) if pod_match  else "",
            "laterality": lat_match.group(0).lower() if lat_match else "",
            "aspect":     asp_match.group(0).lower() if asp_match else "",
            "dosages":    dosages,
        }

    # ------------------------------------------------------------------
    # Métodos privados — aplicação das regras semânticas
    # ------------------------------------------------------------------

    def _apply_symptoms(self, patient_id: str, ctx: _SentenceContext) -> None:
        """Cria nós de Symptom e arestas PRESENTS_WITH + LOCATED_IN quando aplicável."""
        if not ctx.symptoms:
            return

        pres_attr = f"duration={ctx.duration}" if ctx.duration else ""

        for sym in ctx.symptoms:
            sid = self.add_node(sym, "Symptom")
            self.add_edge(patient_id, sid, "PRESENTS_WITH", pres_attr)

            # Dores são localizadas anatomicamente com lateralidade na aresta
            if "pain" in sym.lower():
                for anat in ctx.anatomy:
                    loc_attr = f"laterality={ctx.laterality}" if ctx.laterality else ""
                    aid = self.add_node(anat, "Anatomy")
                    self.add_edge(sid, aid, "LOCATED_IN", loc_attr)

    def _apply_medications(
        self,
        patient_id: str,
        ctx: _SentenceContext,
        orig_text: str,
    ) -> None:
        """Cria nós de Medication e arestas TAKES com dosagem quando disponível."""
        if not ctx.meds:
            return

        for med in ctx.meds:
            mid = self.add_node(med, "Medication")

            dose_attr = ""
            if ctx.dosages:
                m_match = re.search(re.escape(med), orig_text, re.IGNORECASE)
                if m_match:
                    # Dosagem mais próxima textualmente do medicamento
                    closest_dose = min(ctx.dosages, key=lambda d: abs(d.start - m_match.end()))
                    dose_attr = f"dosage={closest_dose.text}"
                else:
                    dose_attr = f"dosage={ctx.dosages[0].text}"

            self.add_edge(patient_id, mid, "TAKES", dose_attr)

    def _apply_exams(
        self,
        patient_id: str,
        ctx: _SentenceContext,
        orig_text: str,
    ) -> None:
        """Cria nós de Exam, arestas UNDERWENT_EXAM, REVEALS e LOCATED_IN de resultados."""
        if not ctx.exams:
            return

        txt_lower = orig_text.lower()
        res_spans = self._locate_spans(ctx.results, orig_text)

        for ex in ctx.exams:
            ex_attr = f"timeline={ctx.timeline}" if ctx.timeline else ""
            eid = self.add_node(ex, "Exam")
            self.add_edge(patient_id, eid, "UNDERWENT_EXAM", ex_attr)

            for res in ctx.results:
                res_attrs: list[str] = []
                if res in ctx.assigned_sizes:
                    res_attrs.append(f"size={ctx.assigned_sizes[res]}")
                if ctx.assigned_labs.get(res):
                    res_attrs.extend(ctx.assigned_labs[res])
                if "normal" in txt_lower and "echotexture" in res.lower():
                    res_attrs.append("status=normal")

                # Cria o nó de Result limpo, sem atributos
                rid = self.add_node(res, "Result")
                
                # Aplica os atributos numéricos diretamente na aresta REVEALS
                self.add_edge(eid, rid, "REVEALS", "; ".join(res_attrs))

                for anat in ctx.anatomy:
                    loc_attr = f"aspect={ctx.aspect}" if ctx.aspect else ""
                    aid = self.add_node(anat, "Anatomy")
                    self.add_edge(rid, aid, "LOCATED_IN", loc_attr)

    def _apply_diseases(
        self,
        patient_id: str,
        ctx: _SentenceContext,
        orig_text: str,
    ) -> None:
        """Cria nós de Disease e arestas DIAGNOSED_WITH (ou HAS_*_HISTORY) com status/método."""
        if not ctx.diseases:
            return

        txt_lower = orig_text.lower()

        for dis in ctx.diseases:
            did = self.add_node(dis, "Disease")

            # Histórico familiar ou pessoal recebe relação distinta
            if "history" in txt_lower and ("medical" in txt_lower or "past" in txt_lower):
                rel = "HAS_FAMILY_HISTORY" if "family" in txt_lower else "HAS_PERSONAL_HISTORY"
                edge_attr = ""
            else:
                rel = "DIAGNOSED_WITH"
                edge_attr = self._diagnosis_attributes(txt_lower)

            self.add_edge(patient_id, did, rel, edge_attr)

            for anat in ctx.anatomy:
                loc_attr = f"aspect={ctx.aspect}" if ctx.aspect else ""
                aid = self.add_node(anat, "Anatomy")
                self.add_edge(did, aid, "LOCATED_IN", loc_attr)

    # ------------------------------------------------------------------
    # Métodos privados — utilitários
    # ------------------------------------------------------------------

    @staticmethod
    def is_negated(entity: str, text: str) -> bool:
        """Verifica se uma entidade está sob escopo de negação ou resolução clínica.

        Divide o texto em orações adversativas ("but", "however", "although")
        antes de aplicar o padrão de negação, para não propagar a negação além
        da cláusula correta.

        Parameters
        ----------
        entity : str
            Token ou expressão a verificar.
        text : str
            Sentença completa onde a entidade ocorre.

        Returns
        -------
        bool
            True se a entidade estiver negada ou descrita como resolvida.
        """
        clauses = re.split(r"\s+(?:but|however|although)\s+", text, flags=re.IGNORECASE)
        for clause in clauses:
            m = _RE_NEGATION.search(clause)
            if m and entity.lower() in m.group(1).lower():
                return True
        return False

    @staticmethod
    def _span_distance(span: _Span, other: _Span) -> int:
        """Distância em caracteres entre dois spans não sobrepostos."""
        if span.end <= other.start:
            return other.start - span.end
        if other.end <= span.start:
            return span.start - other.end
        return 0  # sobrepostos

    @staticmethod
    def _diagnosis_attributes(txt_lower: str) -> str:
        """Determina os atributos de status e método de um diagnóstico."""
        attrs: list[str] = []
        if "pathology" in txt_lower or "final pathology" in txt_lower:
            attrs.extend(["status=confirmed", "method=pathology"])
        elif "suggesting" in txt_lower or "suspected" in txt_lower:
            attrs.extend(["status=suspected", "method=clinical/imaging"])
        else:
            attrs.append("status=confirmed")
        return "; ".join(attrs)