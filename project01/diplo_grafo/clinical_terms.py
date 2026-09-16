"""Módulo de Normalização e Reconstrução de Termos Clínicos (Diplotokens).

Este módulo implementa a fusão de expressões multipalavras contíguas (ex: "lower quadrant
abdominal pain", "mucinous pancreatic cystic neoplasm") e a normalização de siglas
médicas (ex: "GDC" -> "GDC [gastric duplication cyst]"), preservando medições escalares
e pontuação intactas.

========================================================================================
DOCUMENTAÇÃO DA CRIAÇÃO DOS DICIONÁRIOS
========================================================================================

1. ACRONYM_MAP (Dicionário de Siglas e Acrônimos):
   - Origem: Extraído diretamente dos 56 relatos de caso da amostra original,
     onde os autores definiram as siglas no próprio texto pelo padrão canônico
     `termo por extenso (SIGLA)`.
   - Complementado com acrônimos médicos de alta frequência.

2. IRREGULAR_LEMMAS (Dicionário Morfológico de Plurais Médicos Irregulares):
   - Origem: Vocabulário cirúrgico de exceções da língua inglesa e termos biomédicos
     de raiz greco-latina recorrentes nos casos.
   - Finalidade: Evitar o custo de redes neurais pesadas para lematização superficial.

3. LÉXICO MÉDICO:
   - Fonte única consumida a partir de `clinical_vocabulary.ENTITY_CATEGORIES` via
     `clinical_vocabulary.all_terms()`.
"""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path

from diplo_grafo.clinical_vocabulary import all_terms
from diplo_grafo.config import EXTERNAL_DATA_DIR


# ==============================================================================
# 1. TABELA DE SIGLAS E ACRÔNIMOS CLÍNICOS
# ==============================================================================
ACRONYM_MAP: dict[str, str] = {
    "GDC":     "gastric duplication cyst",
    "CEA":     "carcinoembryonic antigen",
    "CA":      "carbohydrate antigen",
    "EUS-FNA": "endoscopic ultrasound-guided fine needle aspiration",
    "FNA":     "fine needle aspiration",
    "CT":      "computed tomography",
    "MRI":     "magnetic resonance imaging",
    "VSD":     "ventricular septal defect",
    "SLE":     "systemic lupus erythematosus",
    "TTP":     "thrombotic thrombocytopenic purpura",
    "HTN":     "hypertension",
    "DM2":     "diabetes mellitus type 2",
    "IM":      "internal medicine",
    "ER":      "emergency room",
    "UC":      "ulcerative colitis",
}


# ==============================================================================
# 2. DICIONÁRIO DE LEMATIZAÇÃO IRREGULAR
# ==============================================================================
IRREGULAR_LEMMAS: dict[str, str] = {
    "cysts":        "cyst",
    "lesions":      "lesion",
    "masses":       "mass",
    "examinations": "examination",
    "rates":        "rate",
    "antibodies":   "antibody",
    "defects":      "defect",
    "infections":   "infection",
    "seizures":     "seizure",
    "stenoses":     "stenosis",
    "thrombi":      "thrombus",
    "diverticula":  "diverticulum",
}


def lemmatize_token(token: str) -> str:
    """Lematiza substantivos médicos em inglês preservando siglas em caixa alta.

    Regras:
    1. Se for sigla em caixa alta (len >= 2), mantém intacta.
    2. Se constar na tabela de irregulares, retorna a forma canônica singular.
    3. Aplica regras morfológicas regulares para plurais em inglês (-ies, -es, -s).
    """
    if token.isupper() and len(token) >= 2:
        return token

    lower = token.lower()
    if lower in IRREGULAR_LEMMAS:
        return IRREGULAR_LEMMAS[lower]

    if lower.endswith("ies") and len(lower) > 4:
        return lower[:-3] + "y"
    if lower.endswith("es") and len(lower) > 3 and lower[-3] in "shxz":
        return lower[:-2]
    if lower.endswith("s") and not lower.endswith("ss") and len(lower) > 3:
        return lower[:-1]

    return lower


# ==============================================================================
# 3. CONSTRUÇÃO E CARREGAMENTO DO LÉXICO MÉDICO
# ==============================================================================

def build_medical_lexicon() -> set[str]:
    """Extrai termos médicos canônicos.

    Os termos centrais vêm de ``clinical_vocabulary.all_terms()``, que agrega
    todos os sets de ``ENTITY_CATEGORIES``.
    """
    return all_terms()


def load_medical_lexicon(path: Path | str | None = None) -> set[str]:
    """Carrega o léxico médico de data/external/medical_lexicon.json ou gera dinamicamente."""
    if path is None:
        path = EXTERNAL_DATA_DIR / "medical_lexicon.json"

    file_path = Path(path)
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return set(json.load(f))

    return build_medical_lexicon()


# ==============================================================================
# 4. FUNÇÃO FINAL DE RECONSTRUÇÃO DE TOKENS
# ==============================================================================
def reconstruct_clinical_tokens(
    tokens: list[str],
    vocabulary: set[str] | None = None,
    acronym_map: dict[str, str] | None = None,
    lemmatizer: Callable[[str], str] | None = None,
    expand_acronyms: bool = True,
    max_n: int = 5,
) -> list[str]:
    """FUNÇÃO FINAL DO PIPELINE CLÍNICO.

    Recebe a lista de tokens fatiada pelo tokenizador e produz a lista com correções:
    - Agrupa sequências multipalavras contíguas que representam conceitos médicos (longest-match).
    - Lematiza plurais para casar com a ontologia canônica (suporta lematizador customizado).
    - Resolve e expande siglas médicas (ex: "GDC" -> "GDC [gastric duplication cyst]").
    - Preserva números, pontuação e medições escalares intactos.

    Args:
        tokens: Lista de tokens brutos produzidos pelo regex.
        vocabulary: Conjunto de expressões médicas válidas. Se None, carrega o léxico padrão.
        acronym_map: Mapeamento de siglas. Se None, usa ACRONYM_MAP.
        lemmatizer: Função opcional para lematizar cada token. Se None, usa lemmatize_token.
        expand_acronyms: Se True, gera formato "SIGLA [forma por extenso]".
        max_n: Tamanho máximo da janela de n-gramas a inspecionar (default: 5).

    Returns:
        Lista de tokens com termos médicos compostos reunidos e siglas tratadas.
    """
    if vocabulary is None:
        vocabulary = load_medical_lexicon()

    if acronym_map is None:
        acronym_map = ACRONYM_MAP

    lemma_func = lemmatizer if lemmatizer is not None else lemmatize_token

    n = len(tokens)
    result: list[str] = []
    i = 0

    while i < n:
        tok = tokens[i]

        # 1. Tratamento de Siglas / Acrônimos
        if tok in acronym_map:
            expanded = f"{tok} [{acronym_map[tok]}]" if expand_acronyms else acronym_map[tok]
            result.append(expanded)
            i += 1
            continue

        # 2. Janela Deslizante Gulosa para Termos Multipalavras (de max_n até 2 tokens)
        matched = False
        for k in range(min(max_n, n - i), 1, -1):
            window = tokens[i : i + k]

            # Não atravessar pontuação forte ou parênteses
            if any(p in window for p in [".", ";", "!", "?", "(", ")"]):
                continue

            # Testa correspondência literal e lematizada
            raw_phrase = " ".join(window).lower()
            lemma_phrase = " ".join(lemma_func(t) for t in window)

            if raw_phrase in vocabulary or lemma_phrase in vocabulary:
                result.append(" ".join(window))
                i += k
                matched = True
                break

        if matched:
            continue

        # 3. Termo isolado (preserva intacto)
        result.append(tok)
        i += 1

    return result


if __name__ == "__main__":
    from diplo_grafo.sentence import Sentence

    test_sentence = (
        "A 44-year-old woman presented with a 3-day history of right flank "
        "and lower quadrant abdominal pain associated with nausea and constipation."
    )

    sent_obj = Sentence(test_sentence, 0)
    raw_toks = sent_obj.tokenize()

    lex = load_medical_lexicon()
    merged = reconstruct_clinical_tokens(raw_toks, vocabulary=lex)

    print("=== TESTE DO MÓDULO diplo_grafo.clinical_terms ===")
    print(f"Tokens brutos: {raw_toks}\n")
    print(f"Tokens reconstruídos: {merged}\n")

    assert "lower quadrant abdominal pain" in merged
    assert "right flank" in merged
    assert "nausea" in merged
    assert "constipation" in merged
    print("✓ Todos os testes do módulo passaram com sucesso!")