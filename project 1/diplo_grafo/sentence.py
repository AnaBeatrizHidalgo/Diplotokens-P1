import re


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
