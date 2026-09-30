

def blind_generate_word_ids(texts) -> dict[str, int]:
    """
        Lista todas as palavras presentes nos textos associando um índice (partindo de 0) para cada uma]
        Útil para criação da bag of words e possivelmente alguma outra representação numérica de palavras.
        Retorna um dicionário de palavras associadas a números inteiros 0, 1,..., n.
    """
    pass

def smart_generate_word_ids(texts) -> dict[str, float]:
    """
        associar números ás palavras de forma inteligente a partir de frequência e contextos?
        Retorna um dicionário de palavras associadas a números (podem ser negativos ou decimais)
    """
    pass

def bag_of_words(context, word_ids) -> list[float]:
    """
        Retorna vetor (len == len(word_ids)) representando o contexto.
    """
    pass

def ideia_ousada_e_ruim(context, word_ids) -> list[float]:
    """
        kkkk
    """
    pass