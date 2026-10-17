from diplo_graph2.graph import Node
import diplo_graph2.process_text.tokenizer as t
from nltk.lm.api import LanguageModel
import numpy as np

# ideia rafa pra caixa preta 1

NODE_TYPES = ['PATIENT', 'DISEASE', 'SYMPTOM', 'DRUG', 'EXAME', 'ANATOMY', 'EFFECT', 'RESULT']
NODE_TAGS = [f"TAG-{s}" for s in NODE_TYPES]
NUM_TAGS = len(NODE_TAGS)

class NGramsNodeIdentifier:
    def __init__(self, model: LanguageModel, coef_a: float = 1, coef_b: float = 2, quiet=True, max_node_len: int =7, nodecity_treshold=2) -> None:
        self.model = model
        self.coef_a = coef_a
        self.coef_b = coef_b
        self.quiet = quiet
        self.max_node_len = max_node_len
        self.nodecity_treshold = nodecity_treshold

    def log(self, s: object, end="\n"):
        if not self.quiet:
            print(s, end=end)
        
    def identify_nodes(self, case_text: str) -> list[Node]:
        sentences = t.separate_sentences(case_text)
        nodes: list[Node] = []

        sentence_pos = 0

        self.log("------------------------------")
        # Para cada sentença
        for sentence in sentences:
            sentence_spreading = 0;
            tokens = t.tokenize_to_lower(sentence)

            i_target = 2
            initial_len_tokens = len(tokens)


            # Para cada target
            while i_target < len(tokens):
                target = tokens[i_target]

                self.log(f"\n\n----------------> Para o target '{target}':")

                if target in ['.', '!', '?']:
                    self.log("Pontuação, skip!")
                    i_target += 1;
                    continue

                # tags' probs
                tagps = self._get_nfollows_avg_probs(i_target, tokens, NODE_TAGS)
                #self.log(" || ".join([f"{NODE_TAGS[i]}: {tags_probs[i]}" for i in range(NUM_TAGS)]))

                i_max_tagp = int(np.argmax(tagps))
                max_tagp = tagps[i_max_tagp]
                best_tag = NODE_TAGS[i_max_tagp]

                mean_tagp = np.mean(tagps)
                std_tagp = np.std(tagps)
                not_nodep = self._get_1follow_avg_prob(i_target, tokens, target)
                nodecity = max_tagp / not_nodep

                print(f"--- (max_tagp={max_tagp}, mean_tagp={mean_tagp}, tagp_std={std_tagp}, NOT_nodep={not_nodep}, nodecity={nodecity})")

                #if (max_tagp > (self.coef_a * mean_tagp) + (self.coef_b * std_tagp)) \
                #                    and nodecity > self.nodecity_treshold:
                    
                node_len, node_label = self._try_fit_node(i_target, tokens, best_tag)

                if node_len <= 0:
                    i_target += 1
                    continue

                new_node = Node(
                    node_type = best_tag,
                    label = node_label,
                    pos_in_case_text = sentence_pos + i_target + sentence_spreading
                )
                sentence_spreading += node_len - 1

                self.log(f":::NODE ENCONTRADO (label={node_label}, type={new_node.node_type}, pos_in_case={new_node.pos_in_case_text})")
                nodes.append(new_node)

                i_target += 1

            sentence_pos += initial_len_tokens;

        self.log("\n\n------------------------------")
        self.log(f"Tokens processados: {tokens}")
        self.log("------------------------------")

        return nodes

    def _get_nfollows_avg_probs(self, i_target: int, sentence_tokens: list[str], follow_options: list[str]) -> list[float]:
        context_max_len = min(i_target, self.model.order - 1)
        
        sum_probs: list[float] = [0.0 for _ in follow_options]

        
        #self.log(f"** Contexts: (follows: LISTA[0]={follow_options[0]})", end="")

        # Para cada context len
        for cl in range(2, context_max_len + 1):
            context = sentence_tokens[i_target-cl: i_target]
            #self.log(f" {context}", end="")
            for i in range(len(follow_options)):
                sum_probs[i] += self.model.score(follow_options[i], context)

        #self.log("")

        return [s / (context_max_len - 1) for s in sum_probs]

    def _get_1follow_avg_prob(self, i_target: int, sentence_tokens: list[str], follow: str) -> float:
        context_max_len = min(i_target, self.model.order - 1)
        prob_sum: float = 0.0

        #self.log(f"** Contexts (follow:{follow}):", end="")

        for cl in range(2, context_max_len + 1):
            context = sentence_tokens[i_target-cl: i_target]
            #self.log(f" {context}", end="")
            prob_sum += self.model.score(follow, context)

        #self.log("")

        return prob_sum / (context_max_len - 1)

    def _spread_to_node_end(self, node_start: int, sentence_tokens: list[str], node: Node, node_tag):
        """
            Retorna o quanto avançou
        """
        # "The child has TAG-SYMPTOM on the TAG-ANATOMY."
        # "The child has pain on the right knee."
        # "The child has pain on the TAG-ANATOMY [knee .]"

        max_node_len = min(self.max_node_len, len(sentence_tokens) - 1 - node_start)
        self.log(f"max_node_len={max_node_len}")

        # Obtém a lista das possíveis palavras seguintes do node
        follow_options = []
        for node_len in range(1, max_node_len + 1):
            follow = sentence_tokens[node_start + node_len]
            follow_options.append(follow)

        self.log(f"Follow options: {follow_options}")

        if len(follow_options) == 0:
            return 0

        # --------------------------
        # Probs com a sequência (follows_follow) de um follow original sem consumi-lo
        # ou seja, prob contra incluir no nó
        # "The child has pain on the right knee and should not run."
        # "The child has pain on the TAG-ANATOMY knee and should not run."

        against_probs = []
        test_sentence = sentence_tokens[0:node_start+1]

        self.log("_________________")

        follow = sentence_tokens[node_start + 1]
        self.log(f"test i={0}: {test_sentence+[follow]}")
        p = self._get_1follow_avg_prob(len(test_sentence), test_sentence, follow)
        against_probs.append((follow, p))

        sentence_tokens[node_start] = node_tag
        test_sentence = sentence_tokens[0:node_start+1]
        
        possible_inclusions = follow_options[:-1]

        
        for i in range(len(possible_inclusions)):
            self.log("_________________")
            test_sentence.append(possible_inclusions[i])
            follow = sentence_tokens[node_start + i + 2]
            self.log(f"test i={i+1}: {test_sentence+[follow]}")
            p = self._get_1follow_avg_prob(len(test_sentence), test_sentence, follow)
            against_probs.append((follow, p))
            test_sentence.pop();

        self.log("")
        self.log(f"Against probs: {against_probs}")
        
        # --------------------------

        
            
        q = self.quiet
        self.quiet = True
        avg_probs = self._get_nfollows_avg_probs(node_start+1, sentence_tokens, follow_options)
        self.quiet = q

        self.log(f"Probs follow: {[(follow_options[i], avg_probs[i]) for i in range(len(follow_options))]}")

        i_max_prob = int(np.argmax(avg_probs))

        self.log(f"Best next token = {follow_options[i_max_prob]}")

        node_end_in = node_start + i_max_prob

        spreading = 0

        for i in range(node_start + 1, node_end_in + 1):
            node.label += f" {sentence_tokens.pop(node_start + 1)}"
            spreading += 1;

        self.log(f"Label final node = {node.label}")

        return spreading

    def _try_fit_node(self, node_start: int, sentence_ref: list[str], node_tag) -> tuple[int, str]:
            """
                Retorna o node_len e a label. Se node_len=0 é porque o node não deve existir
            """
            # "The child has TAG-SYMPTOM on the TAG-ANATOMY."
            # "The child has pain on the right knee."
            # "The child has pain on the TAG-ANATOMY [knee .]"

            sentence_tokens = sentence_ref.copy()

            self.log(f'Fitting node {node_tag} = "{sentence_tokens[node_start]}..." ?):')

            # define o tamanho máximo deste nó de acordo com os limites da frase (excluindo o último token que deve ser pontuação)
            max_node_len = min(self.max_node_len, len(sentence_tokens) - 1 - node_start)
            self.log(f".max_node_len={max_node_len}")

            # palavras que podem ser inclusas no node (em ordem) e seus follows (pode ser ".") correspondentes
            possible_inclusions = []
            possible_follows = []
            for node_len in range(max_node_len):
                inclusion = sentence_tokens[node_start + node_len]
                possible_inclusions.append(inclusion)
                follow = sentence_tokens[node_start + node_len + 1]
                possible_follows.append(follow)

            self.log(f".possible_inclusions={possible_inclusions}")
            self.log(f".possible_follows={possible_follows}")

            test_include_sentence = sentence_tokens[0:node_start] + [node_tag]
            test_exclude_sentence = sentence_tokens[0:node_start]

            include_probs: list[float] = []
            against_include_probs: list[float] = []

            for i in range(max_node_len):
                self.log("...................")

                inclusion_i = node_start + i
                follow_i = inclusion_i + 1
                inclusion = sentence_tokens[inclusion_i]
                follow = sentence_tokens[follow_i]

                self.log(f"> '{inclusion}' deve ser incluído?")

                # prob a favor da inclusão. Ex: "... TAG-NODE follow"  (onde inclusion está dentro de TAG-NODE)
                inc_p = self._get_1follow_avg_prob(follow_i, test_include_sentence, follow)
                include_probs.append(inc_p)

                self.log(f'> SIM = {inc_p}  >  "{test_include_sentence+[follow]}"')

                # prob contra a inclusão. Ex "... TAG-NODE inclusion follow"  (inclusion está fora de TAG-NODE)
                test_exclude_sentence.append(inclusion)
                exc_p = self._get_1follow_avg_prob(follow_i, test_exclude_sentence, follow)
                against_include_probs.append(exc_p)
                self.log(f'> NÃO = {exc_p}  >  "{test_exclude_sentence+[follow]}"')
                test_exclude_sentence.pop()

                self.log(f'> COEF = {inc_p / exc_p}')

                if i == 0:
                    test_exclude_sentence.append(node_tag)

            return -1, ""

