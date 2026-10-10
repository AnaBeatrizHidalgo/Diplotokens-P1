"""
    Função para separar o texto em tokens
    E sistema de lematização do texto a partir dos tokens já separados
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache

import nltk
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import MWETokenizer, NLTKWordTokenizer

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

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

# Um token é considerado pontuação se não tiver nenhuma letra ou dígito.
_HAS_ALNUM = re.compile(r"[^\W_]", re.UNICODE)

# Stemmer (radical). Os casos clínicos estão em inglês; para outro idioma,
# troque por SnowballStemmer("portuguese"), por exemplo.
_STEMMER = SnowballStemmer("english")
_LEMMATIZER = WordNetLemmatizer()

# Tokenizador de palavras / sentenças que também informa a posição (offsets) no texto.
_WORD_TOK = NLTKWordTokenizer()
try:
    from nltk.tokenize.punkt import PunktTokenizer
    _SENT_TOK = PunktTokenizer("english")
except ImportError:  # versões antigas do nltk
    _SENT_TOK = nltk.data.load("tokenizers/punkt/english.pickle")


def separate_sentences(text: str) -> list[str]:
    """
        Recebe um texto e retorna uma lista de sentenças.
    """
    return nltk.tokenize.sent_tokenize(text)

def tokenize(text: str) -> list[str]:
    """
        Recebe um texto e retorna uma lista de tokens.
    """
    #return [match.group(0) for match in TOKEN_SEP_REGEX.finditer(text)]
    return nltk.tokenize.word_tokenize(text)

def tokenize_to_lower(text: str, blacklist: list[str] = []) -> list[str]:
    """
        Recebe um texto e retorna uma lista de tokens em letras minúsculas.
    """
    return [token if token in blacklist else token.lower() for token in tokenize(text)]

def remove_punctuation(tokens: list[str]) -> list[str]:
    """
        Recebe uma lista de tokens e remove os que são apenas pontuação.
    """
    return [token for token in tokens if _HAS_ALNUM.search(token)]

def stem(tokens: list[str], keep: list[str] = []) -> list[str]:
    """
        Recebe uma lista de tokens e retorna apenas o radical de cada um.
        Tokens numéricos e os presentes em `keep` não são alterados.
    """
    return [
        token if (token in keep or not token.isalpha()) else _STEMMER.stem(token)
        for token in tokens
    ]

def lemmatize(tokens: list[str]) -> list[str]:
    """
        Recebe uma lista de tokens e retorna uma lista de lemas correspondentes.
        Mais conservador que o stemming (ex.: "lungs" -> "lung", "was" -> "wa" não ocorre).
    """
    return [_LEMMATIZER.lemmatize(token) for token in tokens]

def preprocess(
    text: str,
    use_stem: bool = True,
    drop_punctuation: bool = True,
    keep_numbers: bool = True,
    blacklist: list[str] = [],
) -> list[str]:
    """
        Pipeline completo: tokeniza -> minúsculas -> remove pontuação -> radical.
        - blacklist: tokens que não devem ser colocados em minúsculas nem reduzidos ao radical
          (ex.: siglas como "COVID", "HIV").
        - keep_numbers: se False, remove tokens puramente numéricos.
    """
    tokens = tokenize_to_lower(text, blacklist)
    if drop_punctuation:
        tokens = remove_punctuation(tokens)
    if not keep_numbers:
        tokens = [t for t in tokens if not re.fullmatch(r"[\d.,/]+", t)]
    if use_stem:
        tokens = stem(tokens, keep=blacklist)
    return tokens

def preprocess_corpus(texts: list[str], **kwargs) -> list[list[str]]:
    """
        Aplica `preprocess` a uma lista de casos clínicos.
    """
    return [preprocess(text, **kwargs) for text in texts]


# ---------------------------------------------------------------------------
# Tokens com posição no texto (usado pela rotulação fraca, step2_weak_labels.py)
# ---------------------------------------------------------------------------
# Identifica COMO o texto foi normalizado. O dicionário MeSH é normalizado do mesmo jeito,
# então se você mudar o stemmer/tokenizador, mude este id e gere os rótulos de novo.
PREPROC_ID = "nltk-treebank+snowball-english-v1"


@dataclass
class Token:
    i: int            # posição do token no caso
    text: str         # texto original do token
    lower: str        # em minúsculas
    norm: str         # forma normalizada (radical) usada para casar com o dicionário
    sent_id: int      # índice da sentença
    is_punct: bool    # True se for só pontuação
    char_start: int   # posição inicial no texto original
    char_end: int     # posição final (exclusiva) no texto original


@dataclass
class ProcessedCase:
    case_id: str
    article_id: str
    text: str
    tokens: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)   # demais colunas do CSV (age, gender, ...)


@lru_cache(maxsize=None)
def _stem_word(word: str) -> str:
    return _STEMMER.stem(word)


def normalize_token(lower: str) -> str:
    """
        Forma normalizada de um token em minúsculas: radical se for só letras;
        números e palavras com hífen (ex.: "x-ray") ficam como estão.
    """
    return _stem_word(lower) if lower.isalpha() else lower


def tokenize_with_spans(text: str) -> list[Token]:
    """
        Recebe um texto e retorna Tokens com posição no texto original, índice da sentença,
        marcação de pontuação e forma normalizada.
    """
    tokens: list[Token] = []
    for sent_id, (s0, s1) in enumerate(_SENT_TOK.span_tokenize(text)): # pyright: ignore[reportAttributeAccessIssue]
        sentence = text[s0:s1]
        for a, b in _WORD_TOK.span_tokenize(sentence):
            surface = text[s0 + a:s0 + b]
            lower = surface.lower()
            tokens.append(Token(
                i=len(tokens), text=surface, lower=lower, norm=normalize_token(lower),
                sent_id=sent_id, is_punct=not _HAS_ALNUM.search(surface),
                char_start=s0 + a, char_end=s0 + b,
            ))
    return tokens


def process_case(case_id: str, article_id: str, text: str, meta: dict | None = None) -> ProcessedCase:
    """
        Tokeniza um caso clínico e devolve um ProcessedCase.
    """
    return ProcessedCase(case_id, article_id, text, tokenize_with_spans(text), meta or {})


def normalize_phrases(phrases: list[str]) -> list[tuple[str, ...]]:
    """
        Normaliza frases (ex.: termos do MeSH) do mesmo jeito que o texto dos casos:
        tokeniza, tira pontuação e reduz cada palavra ao radical. Retorna uma tupla por frase.
    """
    out = []
    for phrase in phrases:
        toks = [t.lower() for t in _WORD_TOK.tokenize(phrase) if _HAS_ALNUM.search(t)]
        out.append(tuple(normalize_token(t) for t in toks))
    return out


def _unite_terms(tokens: list[str], terms: list[str]) -> list[str]:
    """
        Une, em um único token (com "_"), sequências de tokens que formam um termo composto.
    """
    mwe = MWETokenizer(
        [tuple(term.lower().split()) for term in terms], separator="_"
    )
    return mwe.tokenize(tokens)

def unite_mesh_terms(tokens: list[str], mesh_terms: list[str] = []) -> list[str]:
    """
        Recebe uma lista de tokens e retorna uma lista de tokens com termos do mesh unidos.
        Ex.: ["myocardial", "infarction"] -> ["myocardial_infarction"]
        Use ANTES do stemming, passando os termos do MeSH em `mesh_terms`.
    """
    # não necessariamente precisamos implementar e usar no projeto 2
    return _unite_terms(tokens, mesh_terms)


def unite_special_terms(tokens: list[str], special_terms: list[str] = []) -> list[str]:
    """
        Recebe uma lista de tokens e retorna uma lista de tokens com termos especiais unidos.
        Ex.: nomes compostos de remédios, partes do corpo ("left", "ventricle"), etc.
    """
    # aqui dá pra usar alguma lib bacana de lematização com ml, treinando com os textos ou sla
    # não necessariamente precisamos implementar e usar no projeto 2
    return _unite_terms(tokens, special_terms)