from nltk.lm import Laplace
from nltk.lm.preprocessing import padded_everygram_pipeline

def train_ngram_model(tokens: list[list[str]], n = 4) -> Laplace:
    # Preparar os dados para o modelo n-grama
    train_data, padded_vocab = padded_everygram_pipeline(n, tokens)

    # Treinar o modelo com Laplace Smoothing
    model = Laplace(n)
    model.fit(train_data, padded_vocab)

    return model