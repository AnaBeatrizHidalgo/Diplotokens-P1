"""
Tuning dos hiperparametros do NGramsNodeIdentifier (ngni) usando validacao.

Fluxo
-----
1. Le train_cases_labeled.csv (colunas: article_id, case_text, case_text_tagged, ...).
2. Monta o GABARITO de cada caso alinhando o texto original com o texto rotulado
   (cada TAG-* no rotulado corresponde a um span de tokens do original).
3. Separa treino/validacao POR ARTIGO (casos do mesmo artigo nunca ficam em lados
   diferentes). n_folds=1 -> um holdout de val_frac; n_folds>1 -> k-fold por artigo.
4. Treina os n-gramas (frente e reverso) so com a parte de treino de cada fold.
5. Busca aleatoria nos hiperparametros (a configuracao atual entra como trial 0, de
   baseline) e escolhe o de maior F1 na validacao.
6. Retreina com TODO o treino e devolve modelos + melhores parametros.
O split de TESTE nao e tocado, a nao ser que voce chame evaluate_on_test().

Diagnostico (onde esta o gargalo?): diagnose() ou `--diagnose` -> teto de recall imposto pelos
filtros, matriz de confusao de tags, e as palavras que mais geram falsos positivos / sao perdidas.

Uso (a partir de project02/):
    python -m diplo_graph2.node_identifiers.ngni_tuning --trials 60
    python -m diplo_graph2.node_identifiers.ngni_tuning --trials 60 --folds 3 --eval-test
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import nltk
import numpy as np
import pandas as pd

import diplo_graph2.node_identifiers.ngni as ngni
import diplo_graph2.process_text.tokenizer as t
from diplo_graph2.process_text.n_grams import train_ngram_model

# todas as tags possiveis nos CSVs (mesmo as que voce tirou de NODE_TYPES), para o
# tokenizer nao colocar em minusculas uma tag que ainda esteja no texto rotulado
ALL_TAGS = [f"TAG-{s}" for s in
            ['PATIENT', 'DISEASE', 'SYMPTOM', 'DRUG', 'EXAME', 'ANATOMY', 'EFFECT', 'RESULT']]
MAX_SPAN = 15

DEFAULT_SPACE = {
    "coef_a": (0.5, 3.0, "lin"),
    "coef_b": (0.0, 3.0, "lin"),
    "nodecity_treshold": (1.0, 8.0, "log"),
    "include_base": (0.6, 1.6, "lin"),
    "max_node_len": [4, 6, 8, 10],
    "cross": [True, False],
}
# configuracao que voce vinha usando no notebook
BASELINE = dict(coef_a=1.0, coef_b=1.0, nodecity_treshold=1.01,
                include_base=1.0, max_node_len=10, cross=True)


# --------------------------------------------------------------------------
# Gabarito: alinhamento texto original <-> texto rotulado
# --------------------------------------------------------------------------
def _is_tag(tok: str) -> bool:
    return tok.startswith("TAG-")


def _align(orig: list[str], tagged: list[str]):
    """[(start, length, tag), ...] ou None se nao alinhar. Em ambiguidade, o span mais curto vence."""
    n_o, n_t = len(orig), len(tagged)
    memo: dict = {}
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10_000))

    def rec(i, j):
        key = (i, j)
        if key in memo:
            return memo[key]
        if j == n_t:
            res = [] if i == n_o else None
        elif _is_tag(tagged[j]):
            res = None
            for length in range(1, min(MAX_SPAN, n_o - i) + 1):
                rest = rec(i + length, j + 1)
                if rest is not None:
                    res = [(i, length, tagged[j])] + rest
                    break
        elif i < n_o and orig[i] == tagged[j]:
            res = rec(i + 1, j + 1)
        else:
            res = None
        memo[key] = res
        return res

    return rec(0, 0)


def build_gold(case_text: str, tagged_text: str):
    """Gabarito em posicoes globais (mesma convencao de pos_in_case_text do identificador).
    Devolve None se o alinhamento falhar (o caso e descartado)."""
    sents_o = t.separate_sentences(case_text)
    sents_t = t.separate_sentences(tagged_text)
    if len(sents_o) != len(sents_t):
        return None
    gold, pos = [], 0
    for so, st in zip(sents_o, sents_t):
        to = t.tokenize_to_lower(so)
        tt = t.tokenize_to_lower(st, blacklist=ALL_TAGS)
        spans = _align(to, tt)
        if spans is None:
            return None
        gold += [(pos + s, length, tag) for s, length, tag in spans]
        pos += len(to)
    return gold


# --------------------------------------------------------------------------
# Dados, folds e treino
# --------------------------------------------------------------------------
def load_labeled(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).dropna(subset=["case_text", "case_text_tagged"]).reset_index(drop=True)
    golds = [build_gold(str(a), str(b)) for a, b in zip(df.case_text, df.case_text_tagged)]
    bad = [i for i, g in enumerate(golds) if g is None]
    if bad:
        print(f"[aviso] {len(bad)} caso(s) descartado(s) por falha de alinhamento: "
              f"{', '.join(df.case_id.iloc[bad].astype(str))}")
    df["gold"] = golds
    return df[df.gold.notna()].reset_index(drop=True)


def make_folds(df: pd.DataFrame, n_folds: int = 1, val_frac: float = 0.2, seed: int = 0):
    arts = sorted(df.article_id.unique())
    random.Random(seed).shuffle(arts)
    if n_folds <= 1:
        k = max(1, round(len(arts) * val_frac))
        groups = [set(arts[:k])]
    else:
        groups = [set(arts[i::n_folds]) for i in range(n_folds)]
    return [(df[~df.article_id.isin(g)], df[df.article_id.isin(g)]) for g in groups]


def train_models(train_df: pd.DataFrame, order: int = 5, background=None):
    toks = [t.tokenize_to_lower(str(x), blacklist=ALL_TAGS) for x in train_df.case_text_tagged]
    if background:
        toks = toks + list(background)           # ex.: brown_background(...)
    return train_ngram_model(toks, n=order), train_ngram_model([x[::-1] for x in toks], n=order)


# --------------------------------------------------------------------------
# Metricas
# --------------------------------------------------------------------------
MODES = {
    "strict": lambda x: x,                 # (inicio, tamanho, tag)
    "start_tag": lambda x: (x[0], x[2]),   # (inicio, tag)  <- padrao
    "start": lambda x: x[0],               # so o inicio
}
ALL_MODES = list(MODES) + ["overlap"]      # "overlap": span previsto cruza span do gabarito, mesma tag


def _to_sets(nodes, gold, ignore_tags):
    pred = {(n.pos_in_case_text, len(n.label.split()), n.node_type)
            for n in nodes if n.node_type not in ignore_tags}
    gld = {g for g in gold if g[2] not in ignore_tags}
    return pred, gld


def _overlaps(a, b) -> bool:
    """a, b = (inicio, tamanho, tag): mesma tag e intervalos de tokens que se cruzam."""
    return a[2] == b[2] and a[0] < b[0] + b[1] and b[0] < a[0] + a[1]


def metrics(pairs, mode="overlap") -> dict:
    """pairs = [(previstos, gabarito), ...] (conjuntos de (inicio, tamanho, tag)).
    Modos exatos: strict / start_tag / start. Modo "overlap": um previsto acerta se cruza algum span
    do gabarito com a mesma tag (precisao = previstos que acertam; recall = nos do gabarito cobertos).
    Obs.: no overlap um previsto largo que cobre 2 nos do gabarito conta como cobertura para os dois."""
    if mode == "overlap":
        n_pred = n_gold = hit_pred = hit_gold = 0
        for pred, gld in pairs:
            n_pred += len(pred); n_gold += len(gld)
            hit_pred += sum(any(_overlaps(p, g) for g in gld) for p in pred)
            hit_gold += sum(any(_overlaps(p, g) for p in pred) for g in gld)
        prec = hit_pred / n_pred if n_pred else 0.0
        rec = hit_gold / n_gold if n_gold else 0.0
        tp, fp, fn = hit_gold, n_pred - hit_pred, n_gold - hit_gold
    else:
        f = MODES[mode]
        tp = fp = fn = 0
        for pred, gld in pairs:
            p, g = {f(x) for x in pred}, {f(x) for x in gld}
            tp += len(p & g); fp += len(p - g); fn += len(g - p)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(precision=prec, recall=rec, f1=f1, tp=tp, fp=fp, fn=fn)


def per_tag_report(pairs, mode="overlap") -> pd.DataFrame:
    rows = []
    tags = sorted({x[2] for p, g in pairs for x in (p | g)})
    for tag in tags:
        sub = [({x for x in p if x[2] == tag}, {x for x in g if x[2] == tag}) for p, g in pairs]
        m = metrics(sub, mode)
        rows.append(dict(tag=tag, **{k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()}))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Execucao do identificador
# --------------------------------------------------------------------------
class _PosTagCache:
    """pos_tag e deterministico e lento; durante a busca ele roda nas mesmas frases em todo trial."""
    def __enter__(self):
        self._orig = nltk.pos_tag
        cache, orig = {}, self._orig

        def cached(tokens, *a, **k):
            key = tuple(tokens)
            if key not in cache:
                cache[key] = orig(list(tokens), *a, **k)
            return cache[key]

        nltk.pos_tag = cached
        return self

    def __exit__(self, *exc):
        nltk.pos_tag = self._orig


def run_identifier(model, model_rev, params: dict, df: pd.DataFrame, ignore_tags=("TAG-PATIENT",)):
    ni = ngni.NGramsNodeIdentifier(model, model_rev=model_rev, quiet=True, **params)
    return [_to_sets(ni.identify_nodes(str(text)), gold, set(ignore_tags))
            for text, gold in zip(df.case_text, df.gold)]


# --------------------------------------------------------------------------
# Busca
# --------------------------------------------------------------------------
def _sample(space: dict, rng: random.Random) -> dict:
    out = {}
    for name, spec in space.items():
        if isinstance(spec, tuple):
            lo, hi, kind = spec
            v = math.exp(rng.uniform(math.log(lo), math.log(hi))) if kind == "log" else rng.uniform(lo, hi)
            out[name] = round(v, 3)
        else:
            out[name] = rng.choice(spec)
    return out


@dataclass
class TuneResult:
    best_params: dict
    best_metrics: dict
    baseline_metrics: dict
    trials: pd.DataFrame
    model: object
    model_rev: object
    objective_mode: str


def tune(train_csv=None, *, order: int = 5, n_trials: int = 60, n_folds: int = 1,
         val_frac: float = 0.2, seed: int = 0, objective_mode: str = "overlap",
         space: dict | None = None, background=None, ignore_tags=("TAG-PATIENT",),
         out_json=None, trials_csv=None, verbose: bool = True) -> TuneResult:
    if train_csv is None:
        from diplo_graph2.config import PROCESSED_DATA_DIR
        train_csv = Path(PROCESSED_DATA_DIR) / "ngni_train-data" / "train_cases_labeled.csv"
    space = space or DEFAULT_SPACE
    rng = random.Random(seed)

    df = load_labeled(train_csv)
    folds = make_folds(df, n_folds, val_frac, seed)
    if verbose:
        print(f"{len(df)} casos / {df.article_id.nunique()} artigos | "
              f"{len(folds)} fold(s): " + ", ".join(
                  f"treino={len(a)} val={len(b)}" for a, b in folds))

    fold_models = [train_models(tr, order, background) for tr, _ in folds]

    def evaluate_params(params):
        pairs = []
        for (m, mr), (_, va) in zip(fold_models, folds):
            pairs += run_identifier(m, mr, params, va, ignore_tags)
        return pairs

    trials = [("baseline", dict(BASELINE))] + [(f"rand{i}", _sample(space, rng)) for i in range(n_trials)]
    rows, best, baseline_m = [], None, None
    with _PosTagCache():
        for k, (name, params) in enumerate(trials):
            t0 = time.time()
            pairs = evaluate_params(params)
            m = metrics(pairs, objective_mode)
            rows.append(dict(trial=name, **params, **{key: m[key] for key in ("precision", "recall", "f1")},
                             strict_f1=metrics(pairs, "strict")["f1"],
                             start_tag_f1=metrics(pairs, "start_tag")["f1"],
                             overlap_f1=metrics(pairs, "overlap")["f1"],
                             secs=round(time.time() - t0, 1)))
            if name == "baseline":
                baseline_m = m
            if best is None or (m["f1"], m["precision"]) > (best[1]["f1"], best[1]["precision"]):
                best = (params, m, pairs)
            if verbose:
                print(f"[{k + 1}/{len(trials)}] {name}: F1={m['f1']:.3f} P={m['precision']:.3f} "
                      f"R={m['recall']:.3f} ({rows[-1]['secs']}s)  melhor={best[1]['f1']:.3f}")

    trials_df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    best_params, best_m, best_pairs = best
    if verbose:
        print(f"\nBaseline  F1={baseline_m['f1']:.3f}  ->  Melhor  F1={best_m['f1']:.3f}")
        print("Melhores parametros:", best_params)
        print("\nPor tipo (validacao, melhor config):")
        print(per_tag_report(best_pairs, objective_mode).to_string(index=False))
        print("\nTop 5 trials:")
        print(trials_df.head(5).to_string(index=False))

    if trials_csv:
        trials_df.to_csv(trials_csv, index=False)
    if out_json:
        Path(out_json).write_text(json.dumps(best_params, indent=2))

    model, model_rev = train_models(df, order, background)   # retreina com todo o treino
    return TuneResult(best_params, best_m, baseline_m, trials_df, model, model_rev, objective_mode)


def evaluate_on_test(result: TuneResult, test_csv, ignore_tags=("TAG-PATIENT",)) -> dict:
    """Use UMA vez, no fim, com os melhores parametros ja escolhidos."""
    df = load_labeled(test_csv)
    with _PosTagCache():
        pairs = run_identifier(result.model, result.model_rev, result.best_params, df, ignore_tags)
    out = {mode: metrics(pairs, mode) for mode in ALL_MODES}
    print("\n=== TESTE ===")
    for mode, m in out.items():
        print(f"{mode:>9}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    print(per_tag_report(pairs).to_string(index=False))
    return out


# --------------------------------------------------------------------------
# Diagnostico: onde esta o gargalo?
# --------------------------------------------------------------------------
def _case_tokens(text: str):
    """Tokens e POS globais do caso, na mesma convencao de posicoes do identificador.
    Devolve (tokens, pos, [(inicio_da_frase, n_tokens), ...])."""
    toks, poss, sents, p0 = [], [], [], 0
    for s in t.separate_sentences(text):
        tk = t.tokenize_to_lower(s)
        sents.append((p0, len(tk)))
        toks += tk
        poss += [tag for _, tag in nltk.pos_tag(tk)]
        p0 += len(tk)
    return toks, poss, sents


def _start_block_reason(tok: str, pos: str):
    """Por que `tok` nao pode iniciar um candidato (None = pode)."""
    if pos in ngni.SKIP_START:
        return f"pos:{pos}"
    if ngni._is_punct(pos) or ngni._is_punct(tok):
        return "pontuacao"
    if any(c.isdigit() for c in tok):
        return "digito"
    return None


def _span_ceiling(toks, pos, ls, length, n, max_node_len):
    """O identificador PODERIA propor este span? (so filtros: posicao, POS, digito, tamanho;
    nada de probabilidades). Vale para a passada de ida (comeca no 1o token do span) ou a de
    volta (comeca no ultimo). Devolve (ok, motivo)."""
    le = ls + length - 1
    fwd = "inicio_da_frase" if ls < 2 else _start_block_reason(toks[ls], pos[ls])
    bwd = "fim_da_frase" if le > n - 3 else _start_block_reason(toks[le], pos[le])
    ext = None
    if length > max_node_len:
        ext = "span_longo"
    else:
        for k in range(length):
            p = pos[ls + k]
            if p in ngni.STOP_EXTEND or ngni._is_punct(p):
                ext = f"dentro:{p}"
                break
    ok = ext is None and (fwd is None or bwd is None)
    return ok, (None if ok else (ext or f"{fwd} | {bwd}"))


ORACLE_ORDER = ["inicio_da_frase", "filtro", "regra_mean_std", "nodecity", "tag_certa_fora_dos_candidatos",
                "prior_trocou_a_tag", "outra_tag_ganhou_no_contexto", "fit_zero", "fit_curto", "fit_longo", "ok"]


def _oracle_stage(ni, toks, pos, ls, length, gold_tag, before):
    """Reproduz a decisao da passada de IDA em `ls` com CONTEXTO ORACLE: os nos do gabarito anteriores
    ao alvo (`before` = [(inicio_local, tamanho, tag)]) ja aparecem colapsados em suas tags, como
    estariam se a ida tivesse acertado tudo ate ali. Diz em que etapa o no do gabarito morre.
    (Replica a logica do _scan; se voce mudar a regra de aceitacao la, mude aqui tambem.)"""
    res = dict(stage=None, reason=None, top1_is_gold=None, nodecity=None,
               chosen_cur=None, chosen_cands=None, chosen_all=None, len_pred=None,
               n_seen=None, n_ctx=None)
    tags = ngni.NODE_TAGS
    if gold_tag not in tags:
        return None
    g_idx = tags.index(gold_tag)

    otoks, opos, i = [], [], 0
    for s, length2, tg in before:                       # colapsa os spans anteriores
        otoks += toks[i:s] + [tg]
        opos += pos[i:s] + ["NN"]
        i = s + length2
    otoks += toks[i:]
    opos += pos[i:]
    it = ls - sum(length2 - 1 for _, length2, _ in before)
    target = otoks[it]

    if it < 2:
        res["stage"] = "inicio_da_frase"
        return res
    why = "pontuacao" if target in [".", "!", "?", ","] else _start_block_reason(target, opos[it])
    if why:
        res.update(stage="filtro", reason=why)
        return res

    try:      # quantos dos contextos testados (tamanhos 2..order-1) o modelo ja viu no treino?
        mdl = ni.model
        cmax = min(it, mdl.order - 1)
        res["n_ctx"] = cmax - 1
        res["n_seen"] = sum(mdl.context_counts(mdl.vocab.lookup(otoks[it - cl:it])).N() > 0
                            for cl in range(2, cmax + 1))
    except Exception:
        pass

    tagps = np.array(ni._get_nfollows_avg_probs(it, otoks, tags))
    not_nodep = ni._get_1follow_avg_prob(it, otoks, target)
    top1 = int(np.argmax(tagps))
    thr = ni.coef_a * tagps.mean() + ni.coef_b * tagps.std()
    cands = [k for k in range(len(tags)) if tagps[k] > thr]
    res.update(top1_is_gold=(top1 == g_idx), nodecity=float(tagps.max() / not_nodep))

    if not cands:
        res["stage"] = "regra_mean_std"
        return res
    if tagps.max() / not_nodep <= ni.nodecity_treshold:
        res["stage"] = "nodecity"
        return res

    priors = ni._tag_priors(ni.model)
    cur = max(cands, key=lambda k: tagps[k] / priors[k])
    by_ctx = max(cands, key=lambda k: tagps[k])
    res.update(chosen_cur=(cur == g_idx), chosen_cands=(by_ctx == g_idx), chosen_all=(top1 == g_idx))
    if cur != g_idx:
        if g_idx not in cands:
            res["stage"] = "tag_certa_fora_dos_candidatos"
        elif by_ctx == g_idx:
            res["stage"] = "prior_trocou_a_tag"            # sem a divisao pelo prior acertaria
        else:
            res["stage"] = "outra_tag_ganhou_no_contexto"
        return res

    n_len = ni._try_fit_node(it, otoks, tags[cur], opos)
    while n_len > 1 and opos[it + n_len - 1] in ngni.TRIM_END:
        n_len -= 1
    res["len_pred"] = n_len
    res["stage"] = ("fit_zero" if n_len <= 0 else "ok" if n_len == length
                    else "fit_curto" if n_len < length else "fit_longo")
    return res


def _print_oracle(out):
    o = out["oracle"]
    if o.empty:
        return
    n = len(o)
    print("\n=== 4) ONDE CADA NO DO GABARITO MORRE (contexto oracle; so passada de ida) ===")
    print(f"Com contexto perfeito a ida recuperaria {(o.stage == 'ok').mean():.1%} dos nos "
          f"(posicao + tag + tamanho) | recall real so de posicao: {out['start_recall']:.1%}")
    cnt = o.stage.value_counts().reindex(ORACLE_ORDER).dropna().astype(int)
    print(pd.DataFrame({"n": cnt, "pct": (cnt / n * 100).round(1)}).to_string())
    if "n_seen" in o and o.n_seen.notna().any():
        sub = o[o.n_seen.notna()]
        print("\nContextos ja vistos no treino (media por etapa; n_ctx = quantos foram testados):")
        print(sub.groupby("stage")[["n_seen", "n_ctx"]].mean().round(2).to_string())
        nd = sub[sub.stage == "nodecity"]
        if len(nd):
            print(f"Em 'nodecity': {(nd.n_seen == 0).mean():.1%} dos nos nao tinham NENHUM contexto visto "
                  f"(a probabilidade cai no fallback e nodecity = 1.0)")
    cols = [c for c in ORACLE_ORDER if c in set(o.stage)]
    print("\nPor tipo (% dos nos de cada tag):")
    print((pd.crosstab(o.tag, o.stage, normalize="index")[cols] * 100).round(0).astype(int).to_string())
    for st in ("regra_mean_std", "nodecity"):
        sub = o[o.stage == st]
        if len(sub):
            print(f"\nMorreram em {st}: {len(sub)} nos; a tag certa era a top-1 em "
                  f"{sub.top1_is_gold.astype(float).mean():.1%} (nodecity mediana {sub.nodecity.median():.2f})")
    pol = o.dropna(subset=["chosen_cur"])
    if len(pol):
        print(f"\nEscolha da tag entre os {len(pol)} nos que passaram nas regras:")
        print(f"  atual (tagps/prior entre candidatas): {pol.chosen_cur.astype(float).mean():.1%}")
        print(f"  argmax entre candidatas:              {pol.chosen_cands.astype(float).mean():.1%}")
        print(f"  argmax geral:                         {pol.chosen_all.astype(float).mean():.1%}")
    fit = o[o.stage.isin(["fit_curto", "fit_longo"])]
    if len(fit):
        print(f"\nFit com tamanho errado: {(fit.stage == 'fit_curto').sum()} curtos, "
              f"{(fit.stage == 'fit_longo').sum()} longos")


def diagnose(train_csv=None, params=None, *, order: int = 5, n_folds: int = 3,
             val_frac: float = 0.2, seed: int = 0, background=None,
             ignore_tags=("TAG-PATIENT",), top: int = 15, oracle: bool = True,
             verbose: bool = True) -> dict:
    """Mesmos folds/modelos do tune(), mas em vez de buscar parametros responde:
      1. TETO: que fracao dos nos do gabarito os filtros (POS, digito, posicao, tamanho) deixam
         o identificador sequer propor?
      2. CONFUSAO: nos nos achados na posicao certa, qual tag saiu? (e o que foi perdido / inventado)
      3. PALAVRAS: quais palavras mais geram falso positivo e quais nos mais se perdem.
    `params`: dict, TuneResult, ou None (usa best_params.json se existir, senao o baseline)."""
    from diplo_graph2.config import PROCESSED_DATA_DIR
    d = Path(PROCESSED_DATA_DIR) / "ngni_train-data"
    if train_csv is None:
        train_csv = d / "train_cases_labeled.csv"
    if params is None:
        pj = d / "best_params.json"
        params = json.loads(pj.read_text()) if pj.exists() else dict(BASELINE)
    elif hasattr(params, "best_params"):
        params = params.best_params
    params = {**BASELINE, **params}
    ign = set(ignore_tags)

    df = load_labeled(train_csv)
    folds = make_folds(df, n_folds, val_frac, seed)
    fold_models = [train_models(tr, order, background) for tr, _ in folds]

    ceiling_rows = []
    conf: Counter = Counter()
    fp_labels: Counter = Counter()
    fn_labels: Counter = Counter()
    len_total = len_ok = 0
    gold_dist: Counter = Counter()
    pred_dist: Counter = Counter()
    pairs_all = []
    oracle_rows = []

    with _PosTagCache():
        for (m, mr), (_, va) in zip(fold_models, folds):
            ni = ngni.NGramsNodeIdentifier(m, model_rev=mr, quiet=True, **params)
            for text, gold in zip(va.case_text, va.gold):
                text = str(text)
                toks, poss, sents = _case_tokens(text)
                nodes = [nd for nd in ni.identify_nodes(text) if nd.node_type not in ign]
                gold = [g for g in gold if g[2] not in ign]
                pairs_all.append(_to_sets(nodes, gold, ign))

                for s, length, tag in gold:                      # 1) teto
                    ss, n = next((a, b) for a, b in sents if a <= s < a + b)
                    ok, why = _span_ceiling(toks[ss:ss + n], poss[ss:ss + n], s - ss, length, n,
                                            params["max_node_len"])
                    ceiling_rows.append(dict(tag=tag, ok=ok, reason=why, word=toks[s], pos=poss[s]))
                    if oracle:
                        before = [(a - ss, l2, tg2) for a, l2, tg2 in gold if ss <= a and a + l2 <= s]
                        r = _oracle_stage(ni, toks[ss:ss + n], poss[ss:ss + n], s - ss, length, tag, before)
                        if r:
                            oracle_rows.append(dict(tag=tag, word=toks[s], len_gold=length, **r))

                pred = {}
                for nd in nodes:
                    pred.setdefault(nd.pos_in_case_text, nd)
                gstart = {s: (length, tag) for s, length, tag in gold}

                for s, (length, tag) in gstart.items():          # 2) confusao + 3) perdidos
                    gold_dist[tag] += 1
                    if s in pred:
                        nd = pred[s]
                        conf[(tag, nd.node_type)] += 1
                        if nd.node_type == tag:
                            len_total += 1
                            len_ok += int(len(nd.label.split()) == length)
                    else:
                        conf[(tag, "MISSED")] += 1
                        fn_labels[(" ".join(toks[s:s + length]), tag)] += 1
                for s, nd in pred.items():                       # inventados
                    pred_dist[nd.node_type] += 1
                    if s not in gstart:
                        conf[("SPURIOUS", nd.node_type)] += 1
                        fp_labels[(nd.label, nd.node_type)] += 1

    # ---- agregacao ----
    ceil = pd.DataFrame(ceiling_rows)
    by_tag = ceil.groupby("tag").ok.agg(teto="mean", nos="count").round(3)
    reasons = ceil[~ceil.ok].reason.value_counts().head(top)
    blocked = (ceil[~ceil.ok].groupby(["word", "pos"]).size()
               .sort_values(ascending=False).head(top).rename("n"))

    gold_tags = sorted(gold_dist)
    pred_tags = sorted({p for (_, p) in conf if p != "MISSED"})
    confusion = pd.DataFrame(0, index=gold_tags + ["SPURIOUS"], columns=pred_tags + ["MISSED"])
    for (g, p), c in conf.items():
        confusion.loc[g, p] = c

    matched = sum(c for (g, p), c in conf.items() if g != "SPURIOUS" and p != "MISSED")
    same_tag = sum(c for (g, p), c in conf.items() if g == p)
    n_gold = sum(gold_dist.values())
    adj = (df.case_text_tagged.str.count(r"TAG-\w+\s+TAG-\w+").sum()
           / max(1, df.case_text_tagged.str.count(r"TAG-\w+").sum()))

    out = dict(ceiling_overall=float(ceil.ok.mean()), ceiling_by_tag=by_tag, unreachable_reasons=reasons,
               unreachable_words=blocked, confusion=confusion,
               start_recall=matched / max(1, n_gold),
               tag_accuracy_given_start=same_tag / max(1, matched),
               length_accuracy_given_start_tag=len_ok / max(1, len_total),
               top_false_positives=fp_labels.most_common(top), top_missed=fn_labels.most_common(top),
               adjacent_tag_ratio=float(adj), gold_dist=dict(gold_dist), pred_dist=dict(pred_dist))
    out["overlap"] = metrics(pairs_all, "overlap")
    out["oracle"] = pd.DataFrame(oracle_rows)

    if verbose:
        print(f"\nParametros: {params}")
        o = out["overlap"]
        print(f"\nMetrica por sobreposicao (span cruza span do gabarito, mesma tag): "
              f"P={o['precision']:.3f} R={o['recall']:.3f} F1={o['f1']:.3f}")
        print(f"\n=== 1) TETO DE RECALL IMPOSTO PELOS FILTROS ===")
        print(f"{out['ceiling_overall']:.1%} dos nos do gabarito podem ser propostos "
              f"(recall atual so de posicao: {out['start_recall']:.1%})")
        print(by_tag.to_string())
        print("\nMotivos mais comuns dos nos inalcancaveis (ida | volta):")
        print(reasons.to_string())
        print("\nPalavras iniciais mais bloqueadas (palavra, POS):")
        print(blocked.to_string())
        print(f"\n=== 2) CONFUSAO DE TAGS (linhas = gabarito, colunas = previsto) ===")
        print(confusion.to_string())
        print(f"\nTag correta dado inicio correto: {out['tag_accuracy_given_start']:.1%} | "
              f"tamanho correto dado inicio+tag: {out['length_accuracy_given_start_tag']:.1%}")
        print(f"Distribuicao gabarito: {dict(gold_dist)}")
        print(f"Distribuicao prevista: {dict(pred_dist)}")
        print(f"\n=== 3) PALAVRAS ===")
        print("Falsos positivos mais comuns (palavra, tag):")
        for (w, tg), c in out["top_false_positives"]:
            print(f"  {c:4d}  {w!r}  {tg}")
        print("Nos do gabarito mais perdidos (palavra, tag):")
        for (w, tg), c in out["top_missed"]:
            print(f"  {c:4d}  {w!r}  {tg}")
        _print_oracle(out)
        print(f"\n=== GABARITO ===\nTags adjacentes (fronteira ambigua no alinhamento): {adj:.1%} das tags")
    return out


def main():
    from diplo_graph2.config import PROCESSED_DATA_DIR
    d = Path(PROCESSED_DATA_DIR) / "ngni_train-data"
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default=str(d / "train_cases_labeled.csv"))
    ap.add_argument("--test-csv", default=str(d / "test_cases_labeled.csv"))
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--folds", type=int, default=1)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--order", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="overlap", choices=ALL_MODES)
    ap.add_argument("--eval-test", action="store_true")
    ap.add_argument("--diagnose", action="store_true",
                    help="roda diagnose() (usa best_params.json se existir) em vez de tune()")
    a = ap.parse_args()
    if a.diagnose:
        diagnose(a.train_csv, order=a.order, n_folds=max(a.folds, 3), seed=a.seed)
        return
    res = tune(a.train_csv, order=a.order, n_trials=a.trials, n_folds=a.folds, val_frac=a.val_frac,
               seed=a.seed, objective_mode=a.mode, background=ngni.brown_background(),
               out_json=d / "best_params.json", trials_csv=d / "tuning_trials.csv")
    if a.eval_test:
        evaluate_on_test(res, a.test_csv)


if __name__ == "__main__":
    main()