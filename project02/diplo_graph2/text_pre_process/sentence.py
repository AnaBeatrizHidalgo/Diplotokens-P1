import re
import pandas as pd

class Sentence:
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
          | [.,;:()]
        """,
        re.VERBOSE | re.IGNORECASE,
    )

    def __init__(self, text, pos):
        self.text = text
        self.pos = pos
        self.tokens = []
        self.denotations = []

    def tokenize(self):
        self.tokens = [match.group(0) for match in self.TOKEN_SEP_REGEX.finditer(self.text)]
        return self.tokens

    def reconstruct_terms(self, vocabulary=None, acronym_map=None, expand_acronyms=True):
        from .clinical_terms import reconstruct_clinical_tokens

        if not self.tokens:
            self.tokenize()
        self.tokens = reconstruct_clinical_tokens(
            self.tokens,
            vocabulary=vocabulary,
            acronym_map=acronym_map,
            expand_acronyms=expand_acronyms,
        )
        return self.tokens

    def __str__(self):
        return self.text

# Sentence separator
SENTENCE_SEP_REGEX = r"((?<!\b[Ff]ig)(?<!\b[Dd]r)(?<!\b[Aa]rt)[\.\!](\s+)[A-Z])"
re_sentence_sep = re.compile(SENTENCE_SEP_REGEX)

SENTENCE_QUESTION_SEP_REGEX = r"((?<!\b[Ff]ig)(?<!\b[Dd]r)(?<!\b[Aa]rt)\?(\s+)[A-Z])"
re_question_sep = re.compile(SENTENCE_QUESTION_SEP_REGEX)

# utilizado pelo get_sentences
def find_separators(text):
    seps = []

    for match in re_sentence_sep.finditer(text):
        seps.append({
            "texto": match.group(0),
            "pos": match.start(),
            "interrogation": False
        })

    for match in re_question_sep.finditer(text):
        seps.append({
            "texto": match.group(0),
            "pos": match.start(),
            "interrogation": True
        })

    return sorted(seps, key=lambda sep: sep["pos"])

def get_sentences(case):
    text = case['case_text']
    seps = find_separators(text)
    current_pos = 0
    sentences = []

    for sep in seps:
        if sep['interrogation']:
            current_pos = sep['pos'] + len(sep['texto'])
            continue

        sentence = Sentence(text[current_pos:sep['pos']], current_pos)
        sentences.append(sentence)
        current_pos = sep['pos'] + len(sep['texto']) -1

    if current_pos < len(text):
        sentences.append(Sentence(text[current_pos:], current_pos))

    return sentences