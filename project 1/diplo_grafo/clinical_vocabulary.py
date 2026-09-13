"""Vocabulário Clínico e Padrões de Extração (fonte única de verdade).

Este módulo centraliza dois tipos de dados estáticos que antes estavam
espalhados entre ``clinical_terms.py`` e ``graph_builder.py``:

1. **Expressões regulares de extração** — padrões que identificam dimensões,
   medições laboratoriais, duração, dosagem, lateralidade e negação no texto
   clínico. Compilados uma única vez na importação.

2. **Vocabulário clínico categorizado** — termos médicos organizados por tipo
   de entidade (Symptom, Disease, Exam, Result, Anatomy, Medication). É a
   fonte única consumida tanto pelo tokenizador (``clinical_terms.py``) quanto
   pelo construtor do grafo (``graph_builder.py``).

Regra de manutenção
-------------------
- Para **adicionar um novo termo** ao pipeline: edite ``ENTITY_CATEGORIES``.
  O tokenizador e o grafo recebem a atualização automaticamente.
- Para **ajustar um padrão de extração**: edite a constante ``RE_*`` relevante.
- Nenhum outro arquivo deve definir listas de termos clínicos ou padrões
  de extração — eles devem importar daqui.
"""

from __future__ import annotations

import re

# ==============================================================================
# SEÇÃO 1 — EXPRESSÕES REGULARES DE EXTRAÇÃO DE ATRIBUTOS
#
# Prefixo RE_ (público) porque são importadas por graph_builder.py.
# Compiladas no nível de módulo: instanciadas uma vez, reutilizadas sempre.
# ==============================================================================

#: Dimensões físicas em três eixos, ex: "9.5cm x 4.5cm x 2.0cm"
RE_DIMENSION_3D = re.compile(
    r"(\d+(?:\.\d+)?)\s*(cm|mm)\s*x\s*(\d+(?:\.\d+)?)\s*(cm|mm)\s*x\s*(\d+(?:\.\d+)?)\s*(cm|mm)",
    re.IGNORECASE,
)

#: Dimensões em dois eixos, ex: "6cm x 9cm"
RE_DIMENSION_2D = re.compile(
    r"(\d+(?:\.\d+)?)\s*(cm|mm)\s*x\s*(\d+(?:\.\d+)?)\s*(cm|mm)",
    re.IGNORECASE,
)

#: Dimensão linear isolada, ex: "6cm"
RE_DIMENSION_1D = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(cm|mm)\b",
    re.IGNORECASE,
)

#: Medições laboratoriais com valor e unidade, ex: "12,476.5ng/ml", "6iu/ml"
RE_LAB_MEASUREMENT = re.compile(
    r"(\d+(?:,\d+)*(?:\.\d+)?)\s*"
    r"(mg/dL|g/dL|ng/mL|U/L|iu/ml|mm/h|mmol/L|g%|fL|/uL|x10\^9/L|K/uL|%)\b",
    re.IGNORECASE,
)

#: Duração com a palavra "history", ex: "3-day history", "2-week history"
RE_DURATION_HISTORY = re.compile(
    r"\b(\d+)-(day|week|month|year)\s+history\b",
    re.IGNORECASE,
)

#: Duração genérica sem "history", ex: "3-day", "2-week"
RE_DURATION_GENERIC = re.compile(
    r"\b(\d+-(?:day|week|month|year))\b",
    re.IGNORECASE,
)

#: Dosagem de medicamentos, ex: "40mg", "300 mcg", "1.5g"
RE_DOSAGE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(mg|mcg|g|IU|units)\b",
    re.IGNORECASE,
)

#: Dia pós-operatório, ex: "postoperative day 4", "POD 4"
RE_TIMELINE_POD = re.compile(
    r"\b(?:POD|postoperative day)\s*(\d+)\b",
    re.IGNORECASE,
)

#: Lateralidade anatômica, ex: "right", "left", "bilateral"
RE_LATERALITY = re.compile(
    r"\b(right|left|bilateral)\b",
    re.IGNORECASE,
)

#: Aspecto anatômico, ex: "posterior", "anterior", "distal", "proximal"
RE_ASPECT = re.compile(
    r"\b(posterior|anterior|superior|inferior|distal|proximal)\b",
    re.IGNORECASE,
)

#: Negação ou resolução clínica, ex: "no evidence of malignancy",
#: "free of internal septations", "resolved", "uninvolved"
RE_NEGATION = re.compile(
    r"(?:no\s+evidence\s+of|negative\s+for|free\s+of|denies|denied|without"
    r"|resolution\s+of|resolved|uninvolved)\s+([^.,;]+)",
    re.IGNORECASE,
)


# ==============================================================================
# SEÇÃO 2 — VOCABULÁRIO CLÍNICO CATEGORIZADO
#
# Fonte única consumida por:
#   - graph_builder.py  → classifica tokens em tipos de entidade do grafo
#   - clinical_terms.py → constrói o léxico do tokenizador (união de todos os sets)
#
# Para adicionar um termo ao pipeline completo, basta incluí-lo aqui
# na categoria correta. Nenhum outro arquivo precisa ser editado.
# ==============================================================================

ENTITY_CATEGORIES: dict[str, set[str]] = {
    "Symptom": {
        "lower quadrant abdominal pain",
        "right flank and lower quadrant abdominal pain",
        "abdominal pain",
        "nausea",
        "constipation",
        "pain",
        "dyspnea",
        "fever",
        "chest pain",
        "shortness of breath",
        "headache",
    },
    "Disease": {
        "mucinous pancreatic cystic neoplasm",
        "mucinous pancreatic cyst",
        "gastric duplication cyst",
        "GDC [gastric duplication cyst]",
        "systemic lupus erythematosus",
        "ventricular septal defect",
        "infective endocarditis",
        "malignancy",
        "adenocarcinoma",
        "heart failure",
        "tuberculosis",
    },
    "Exam": {
        "contrast enhanced computed tomography",
        "computed tomography",
        "EUS-FNA [endoscopic ultrasound-guided fine needle aspiration]",
        "FNA [fine needle aspiration]",
        "intraoperative endoscopy",
        "oesophagogastroduodenoscopy",
        "final pathology",
        "physical examination",
        "biopsy",
        "ultrasound",
    },
    "Result": {
        "cystic lesion",
        "pancreatic echotexture",
        "cyst",
        "extracellular mucin",
        "carcinoembryonic antigen",
        "carbohydrate antigen",
        "internal septations",
        "associated masses",
    },
    "Anatomy": {
        "right flank",
        "stomach",
        "pancreas",
        "posterior wall",
        "lesser sac",
        "coeliac axis",
        "coeliac vessels",
        "abdomen",
    },
    "Procedure": {
        "laparoscopic distal pancreatectomy",
        "surgical resection",
        "surgical stapler",
        "open resection",
    },
    "Medication": {
        "antibiotics",
        "chemotherapy",
        "isoniazid",
        "rifampin",
        "ethambutol",
        "pyrazinamide",
        "aspirin",
        "omeprazole",
    },
}

# ==============================================================================
# SEÇÃO 3 — FRAGMENTOS DE SENTENÇAS IRRELEVANTES PARA O GRAFO
#
# Sentenças que contenham qualquer um desses fragmentos são ignoradas
# durante a construção do grafo (histórico não-contributório, exames normais).
# ==============================================================================

SKIP_SENTENCE_FRAGMENTS: tuple[str, ...] = (
    "non-contributory",
    "were normal",
)


# ==============================================================================
# UTILITÁRIO — visão plana do vocabulário (para o tokenizador)
# ==============================================================================

def all_terms() -> set[str]:
    """Retorna a união de todos os termos de todas as categorias.

    Usada por ``clinical_terms.py`` para construir o léxico do tokenizador
    sem precisar conhecer a estrutura de categorias.

    Returns
    -------
    set[str]
        Todos os termos clínicos cadastrados, em letras minúsculas.
    """
    return {
        term.lower()
        for terms in ENTITY_CATEGORIES.values()
        for term in terms
    }