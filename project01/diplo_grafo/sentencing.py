import pandas as pd
import re
from diplo_grafo.sentence import Sentence

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