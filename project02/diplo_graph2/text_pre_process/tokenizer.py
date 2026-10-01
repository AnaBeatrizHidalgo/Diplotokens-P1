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
        | [.,;:()]
    """,
    re.VERBOSE | re.IGNORECASE,
)

def tokenize(text):
    tokens = [match.group(0) for match in TOKEN_SEP_REGEX.finditer(text)]