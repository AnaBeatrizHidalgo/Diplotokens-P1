import nltk.lm as lm
from nltk.lm.api import LanguageModel
from nltk.lm.preprocessing import padded_everygram_pipeline

def train_ngram_model(tokens: list[list[str]], n = 4) -> LanguageModel:
    # Preparar os dados para o modelo n-grama
    train_data, padded_vocab = padded_everygram_pipeline(n, tokens)

    # Treinar o modelo com Laplace Smoothing
    #model = lm.Laplace(n)
    model = lm.Lidstone(gamma=0.1, order=n)
    model.fit(train_data, padded_vocab)

    return model