from diplo_graph2.graph import Node
from diplo_graph2.process_text.n_grams import train_ngram_model
import diplo_graph2.process_text.tokenizer as t
from nltk.lm.api import LanguageModel

# ideia rafa pra caixa preta 1

NODE_TAGS = ['TAG-PATIENT', 'TAG-DISEASE', 'TAG-SYMPTOM', 'TAG-DRUG', 'TAG-EXAME', 'TAG-ANATOMY', 'TAG-EFFECT', 'TAG-RESULT']
NUM_TAGS = len(NODE_TAGS)


def get_nodes_by_ngrams(model: LanguageModel, model_n: int, case_text: str) -> list[Node]:
    sentences = t.separate_sentences(case_text)
    nodes: list[Node] = []

    sentence_pos = 0

    # para cada sentença
    for sentence in sentences:
        tokens = t.tokenize_to_lower(sentence)

        i_target = 2
        l_tokens = len(tokens)

        # para cada target
        while i_target < l_tokens:
            target = tokens[i_target]

            context_max_len = min(i_target, model_n - 1)

            sum_probs: list[int] = [0 for _ in NODE_TAGS]

            print(f"\n----> Para o target '{target}' as avg_probs são:")
            print("** Contexts utilizados:", end="")

            # para cada context len
            for cl in range(2, context_max_len + 1):
                context = tokens[i_target-cl: i_target]
                print(f" {context}", end="")
                for i in range(NUM_TAGS):
                    sum_probs[i] += model.score(NODE_TAGS[i], context)

            avg_probs = [s / (context_max_len - 1) for s in sum_probs]

            print("")
            print("\n".join([f"{NODE_TAGS[i]}: {avg_probs[i]}" for i in range(NUM_TAGS)]))

            i_target += 1;

        sentence_pos += l_tokens;
        

    return nodes

