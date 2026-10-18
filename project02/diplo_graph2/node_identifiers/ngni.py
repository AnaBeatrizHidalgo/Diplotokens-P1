from diplo_graph2.graph import Node
import diplo_graph2.process_text.tokenizer as t
from nltk.lm.api import LanguageModel
import numpy as np
import nltk
import random

from nltk.corpus import brown

def brown_background(n_tokens=10_000, doc_len=300, seed=7,
                     categories=('editorial', 'government')):
                        # 'learned' 'news'
    """Devolve listas de tokens (minúsculos, sem tags), em blocos parecidos com um 'caso'."""
    sents = [[w.lower() for w in s if w not in ("``", "''")]
             for s in brown.sents(categories=list(categories))]
    random.Random(seed).shuffle(sents)
    docs, cur, total = [], [], 0
    for s in sents:
        cur += s
        total += len(s)
        if len(cur) >= doc_len:
            docs.append(cur)
            cur = []
        if total >= n_tokens:
            break
    return docs

def _is_punct(tag: str) -> bool:
    return not any(c.isalpha() for c in tag)   # ',', '.', ':', '(', ')', '``'...

# ideia rafa pra caixa preta 1

NODE_TYPES = ['DISEASE', 'SYMPTOM', 'DRUG', 'EXAME', 'ANATOMY', 'EFFECT', 'RESULT']  # 'PATIENT', 
NODE_TAGS = [f"TAG-{s}" for s in NODE_TYPES]
NUM_TAGS = len(NODE_TAGS)


# nunca começam um nó
SKIP_START = {'VB', 'VBD', 'VBP', 'VBZ', 'MD',            # verbos finitos
              'PRP', 'PRP$', 'WP', 'WP$', 'WDT', 'WRB',   # pronomes / wh-
              'DT', 'PDT', 'IN', 'CC', 'TO', 'EX', 'RP', 'POS'}
# nó não pode atravessar (IN e DT ficam de fora: "carcinoma of the pancreas")
STOP_EXTEND = {'VB', 'VBD', 'VBP', 'VBZ', 'MD', 'PRP', 'WP', 'WDT', 'WRB',
               'CC', 'TO', 'EX', 'RP', 'PRP$', 'WP$'}
TRIM_END = {'IN', 'DT', 'CC', 'TO', 'PRP$'}

def _is_tag(tok: str) -> bool:
    return tok.startswith("TAG-")


class NGramsNodeIdentifier:
    def __init__(self, model: LanguageModel, model_rev: LanguageModel | None = None, cross=True, coef_a: float = 1, coef_b: float = 2, include_base: float = 1, quiet=True, max_node_len: int =7, nodecity_treshold=1.01) -> None:
        self.model = model
        self.model_rev = model_rev
        self.coef_a = coef_a
        self.coef_b = coef_b
        self.quiet = quiet
        self.max_node_len = max_node_len
        self.nodecity_treshold = nodecity_treshold
        self.include_base = include_base
        self.cross = cross

        self._endless_fit_count = 0
        self._prior_cache = {}

    def log(self, s: object, end="\n"):
        if not self.quiet:
            print(s, end=end)

    def _tag_priors(self, model):
        k = id(model)
        if k not in self._prior_cache:
            self._prior_cache[k] = np.array([model.score(tg) for tg in NODE_TAGS])
        return self._prior_cache[k]
        
    def identify_nodes(self, case_text: str) -> list[Node]:
        sentences = t.separate_sentences(case_text)
        nodes: list[Node] = []
        sentence_pos = 0
        do_cross = self.cross and self.model_rev is not None

        for sentence in sentences:
            tokens = t.tokenize_to_lower(sentence)
            pos = [tag for _, tag in nltk.pos_tag(tokens)]
            orig = list(range(len(tokens)))
            n_tok = len(tokens)

            fwd = self._scan(tokens, pos, orig, self.model)        # tokens/pos/orig ficam colapsados
            bwd = []
            if self.model_rev is not None:
                bwd = self._scan(tokens[::-1] + ['.'], pos[::-1] + ['.'], orig[::-1] + [-1], self.model_rev)

            rev_tokens = tokens[::-1]    # frase colapsada pela ida, invertida (contexto da volta)

            for tag, span, sorig, f_tagps, f_not in fwd:
                if do_cross:
                    jr = len(tokens) - 1 - orig.index(sorig[0])           # posição do nó na lista invertida
                    other = self._scores_at(self.model_rev, rev_tokens, jr, span[-1])
                    if other is not None:
                        tag = self._cross_decide((f_tagps, f_not), other)
                if tag is not None:
                    nodes.append(self._make_node(tag, span, sorig[0], sentence_pos))

            for tag, span, sorig, b_tagps, b_not in bwd:
                span, start = span[::-1], sorig[-1]                        # desfaz a inversão
                if do_cross:
                    s_c = orig.index(start)                                # posição na lista colapsada pela ida
                    other = self._scores_at(self.model, tokens, s_c, span[0])
                    if other is not None:
                        tag = self._cross_decide((b_tagps, b_not), other)
                if tag is not None:
                    nodes.append(self._make_node(tag, span, start, sentence_pos))

            sentence_pos += n_tok

        nodes.sort(key=lambda n: n.pos_in_case_text)
        return nodes

    def _make_node(self, tag, span_tokens, orig_start, sentence_pos) -> Node:
        return Node(
            node_type=tag,
            label=" ".join(span_tokens),
            coord_in_case=(sentence_pos, orig_start),
            pos_in_case_text=sentence_pos + orig_start,
        )

    def _scan(self, tokens, pos, orig, model):
        """Varre a lista (in-place), colapsa cada nó achado em uma tag e devolve
        [(tag, tokens_do_span, índices_originais_do_span), ...]."""
        found = []
        saved_model, self.model = self.model, model   # _try_fit_node e os _get_* usam self.model
        try:
            i_target = 2
            while i_target < len(tokens):
                target = tokens[i_target]
                if target in ['.', '!', '?', ','] or _is_tag(target):
                    i_target += 1
                    continue
                if any(c.isdigit() for c in target):
                    i_target += 1
                    continue
                if pos[i_target] in SKIP_START or _is_punct(pos[i_target]):
                    i_target += 1
                    continue


                tagps = np.array(self._get_nfollows_avg_probs(i_target, tokens, NODE_TAGS))
                not_nodep = self._get_1follow_avg_prob(i_target, tokens, target)
                i_best = self._decide(tagps, not_nodep, self._tag_priors(model))
                if i_best is None:
                    i_target += 1
                    continue
                best_tag = NODE_TAGS[i_best]

                node_len = self._try_fit_node(i_target, tokens, best_tag, pos)
                if node_len <= 0:
                    i_target += 1
                    continue
                while node_len > 1 and pos[i_target + node_len - 1] in TRIM_END:
                    node_len -= 1

                end = i_target + node_len
                found.append((best_tag, tokens[i_target:end], orig[i_target:end], tagps, not_nodep))
                tokens[i_target:end] = [best_tag]
                pos[i_target:end] = ['NN']
                orig[i_target:end] = [orig[i_target]]
                i_target += 1          # o span virou 1 token; o próximo não processado está em i_target+1
        finally:
            self.model = saved_model
        return found

    def _get_nfollows_avg_probs(self, i_target, sentence_tokens, follow_options, model=None):
        model = self.model if model is None else model
        context_max_len = min(i_target, model.order - 1)
        sum_probs = [0.0 for _ in follow_options]
        for cl in range(2, context_max_len + 1):
            context = sentence_tokens[i_target-cl: i_target]
            for i in range(len(follow_options)):
                sum_probs[i] += model.score(follow_options[i], context)
        return [s / (context_max_len - 1) for s in sum_probs]

    def _get_1follow_avg_prob(self, i_target, sentence_tokens, follow, model=None):
        model = self.model if model is None else model
        context_max_len = min(i_target, model.order - 1)
        prob_sum = 0.0
        for cl in range(2, context_max_len + 1):
            context = sentence_tokens[i_target-cl: i_target]
            prob_sum += model.score(follow, context)
        return prob_sum / (context_max_len - 1)

    def _scores_at(self, model, tokens, i, word):
        """(tagps, not_nodep) de `model` na posição i, usando só o contexto tokens[:i].
        None se não há contexto suficiente."""
        if i < 2:
            return None
        tagps = np.array(self._get_nfollows_avg_probs(i, tokens, NODE_TAGS, model))
        return tagps, self._get_1follow_avg_prob(i, tokens, word, model)

    def _decide(self, tagps, not_nodep, priors):
        """A sua regra (candidatas > a*mean + b*std, nodecity, desempate pelo prior).
        Devolve o índice da tag ou None."""
        mean_p, std_p = np.mean(tagps), np.std(tagps)
        cands = [i for i in range(len(tagps)) if tagps[i] > self.coef_a * mean_p + self.coef_b * std_p]
        if not cands or max(tagps) / not_nodep <= self.nodecity_treshold:
            return None
        return max(cands, key=lambda i: tagps[i] / priors[i])

    def _cross_decide(self, a, b):
        """a, b = (tagps, not_nodep) das duas direções. Média geométrica; devolve a tag ou None."""
        eps = 1e-12
        comb = np.sqrt(np.maximum(a[0], eps) * np.maximum(b[0], eps))
        comb_not = np.sqrt(max(a[1], eps) * max(b[1], eps))
        priors = np.sqrt(np.maximum(self._tag_priors(self.model), eps) *
                        np.maximum(self._tag_priors(self.model_rev), eps))
        i = self._decide(comb, comb_not, priors)
        return None if i is None else NODE_TAGS[i]
    

    def _try_fit_node(self, node_start: int, sentence_ref: list[str], node_tag: str, pos: list[str]) -> int:
            """
                Retorna o node_len. Se node_len=0 é porque o node não deve existir.
            """
            # "The child has TAG-SYMPTOM on the TAG-ANATOMY."
            # "The child has pain on the right knee."
            # "The child has pain on the TAG-ANATOMY [knee .]"

            sentence_tokens = sentence_ref.copy()

            self.log(f'Fitting node {node_tag} = "{sentence_tokens[node_start]}..." ?):')

            # define o tamanho máximo deste nó de acordo com os limites da frase (excluindo o último token que deve ser pontuação)
            max_node_len = min(self.max_node_len, len(sentence_tokens) - 1 - node_start)

            # capa mais ainda o limite para pontuações / termos proibidos / tags já "oficializadas"
            for k in range(max_node_len):
                p = pos[node_start + k]
                if p in STOP_EXTEND or _is_punct(p) or _is_tag(sentence_tokens[node_start + k]):
                    max_node_len = k
                    break


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
            includecity = []

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

                includecity.append(inc_p / exc_p)
                self.log(f'> COEF = {inc_p / exc_p}')

                if i == 0:
                    test_exclude_sentence.append(node_tag)

            std_against_inc_p = np.std(against_include_probs)
            mean_against_inc_p = np.mean(against_include_probs)
            pipipi = mean_against_inc_p + std_against_inc_p

            for i in range(max_node_len):
                if includecity[i] < self.include_base:
                    return i

            self._endless_fit_count += 1
            self.log(f"ENDLESS FIT += 1   -->   {possible_inclusions}")
            return max_node_len

