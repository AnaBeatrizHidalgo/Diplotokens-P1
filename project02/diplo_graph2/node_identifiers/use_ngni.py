import diplo_graph2.node_identifiers.ngni as ngni
import diplo_graph2.process_text.tokenizer as t
from diplo_graph2.process_text.n_grams import train_ngram_model
from diplo_graph2.config import PROCESSED_DATA_DIR
import pandas as pd

import nltk
nltk.download('brown')


def get_default_ngni_ngrams_model():
    train_data_file = PROCESSED_DATA_DIR / "ngni_train-data" / "train_cases_labeled.csv"

    train_df = pd.read_csv(train_data_file)

    train_data = train_df["case_text_tagged"].dropna().astype(str).tolist()

    tokens = []
    for train_text in train_data:
        tokens.append(t.tokenize_to_lower(train_text, blacklist=ngni.NODE_TAGS))

    tokens += ngni.brown_background(n_tokens=10_000)

    model = train_ngram_model(tokens, n=5)

    tokens_rev = [tk[::-1] for tk in tokens]
    model_rev = train_ngram_model(tokens_rev, n=5)

    return model, model_rev

def get_ngni_results(model, model_rev, text):
    ni = ngni.NGramsNodeIdentifier(model, model_rev=model_rev, cross=True, quiet=True, max_node_len=10, coef_a=1, coef_b=1, nodecity_treshold=1)

    nodes = ni.identify_nodes(text)
    return nodes

if __name__ == "__main__":
    # exemplo de uso:
    text = "The boy has pain on the knee."

    model, model_rev = get_default_ngni_ngrams_model()
    nodes = get_ngni_results(model, model_rev, text)
    print("\n".join([str(node) for node in nodes]))
