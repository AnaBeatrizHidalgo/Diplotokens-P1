"""
    Função para separar o texto em tokens
    E sistema de lematização do texto a partir dos tokens já separados
"""

import re

UNIT = r"(?:kg|mg|mcg|ng|iu|U|L|mmHg|mmol|ml|dl|l|cm|mm|g|hg|min|y/o|yo|yrs?|years?|months?|weeks?|days?)"
TIME_UNIT = r"(?:min|hr|hrs|h|s|sec|m)"
TOKEN_SEP_REGEX = re.compile(
    rf"""
        \d+(?:,\d{{3}})*(?:\.\d+)?
        (?:
            /\d+(?:,\d{{3}})*(?:\.\d+)?(?:\s?{UNIT}(?:\s?{UNIT})?)?(?![A-Za-z])
            | /{TIME_UNIT}(?![A-Za-z])
            | \s?{UNIT}(?:/{UNIT})?(?![A-Za-z])
        )?
        | [A-Za-z]+(?:-[A-Za-z]+)*
        | [.,;:()!?]+
    """,
    # antes a última linha do regex era: | [.,;:()]
    re.VERBOSE | re.IGNORECASE,
)


def tokenize(text: str) -> list[str]:
    """
        Recebe um texto e retorna uma lista de tokens.
    """
    return [match.group(0) for match in TOKEN_SEP_REGEX.finditer(text)]

def tokenize_to_lower(text: str) -> list[str]:
    """
        Recebe um texto e retorna uma lista de tokens em letras minúsculas.
    """
    return [token.lower() for token in tokenize(text)]

def lemmatize(tokens: list[str]) -> list[str]:
    """
        Recebe uma lista de tokens e retorna uma lista de lemas correspondentes.
    """
    pass


def unite_mesh_terms(tokens: list[str]) -> list[str]:
    """
        Recebe uma lista de tokens e retorna uma lista de tokens com termos do mesh unidos.
    """
    pass


def unite_special_terms(tokens: list[str]) -> list[str]:
    """
        Recebe uma lista de tokens e retorna uma lista de tokens com termos especiais unidos.
    """
    # aqui dá pra usar alguma lib bacana de lematização com ml, treinando com os textos ou sla
    pass