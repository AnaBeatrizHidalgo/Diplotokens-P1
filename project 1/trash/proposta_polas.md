# Pipeline de Extração de Informação de Casos Clínicos → Grafo de Conhecimento

## Objetivo

Extrair, a partir de casos clínicos em texto livre (case reports em inglês), as entidades e relações necessárias para popular um grafo de conhecimento com o seguinte esquema:

```
Artigo --tem--> MeSH-terms
Artigo --tem--> Ano
Artigo --tem--> Casos --tem--> Paciente
Paciente --tem--> Idade
Paciente --tem--> Gênero
Paciente --possui--> Histórico
Paciente --toma--> Medicação --faz--> Efeito
Paciente --faz--> Exame --tem--> Resultado
Paciente --sente--> Sintoma
Paciente --tem--> Partes do corpo afetadas
Paciente --diagnosticado--> Doença --incorre em--> Partes do corpo afetadas
```

## Restrição do projeto

Apenas **estratégias clássicas de extração de informação** (regras, dicionários, expressões regulares, POS tagging, parsing raso). Nenhuma abordagem baseada em LLM/redes neurais.

---

## Etapa 1 — Pré-processar e segmentar

1. **Segmentação de sentenças adaptada ao domínio.** Não usar split ingênuo por `.`, pois ele quebra em:
   - Abreviações (`Fig.`, `e.g.`, `Dr.`)
   - Números decimais (`12,476.5ng/ml`, `6.0cm`)

   Usar ferramentas que já conhecem essas exceções para texto biomédico: **scispaCy** ou **cTAKES** (sentence splitter). O NLTK `punkt` padrão erra bastante nesse domínio.

2. **Tokenização adaptada**, preservando maiúsculas/minúsculas e números intactos por enquanto (a normalização de caixa vem depois da extração case-sensitive).

3. **Detecção de negação (NegEx)**, aplicada logo aqui, sobre o texto quase cru:
   - Lista de **gatilhos de pré-negação**: `no`, `no evidence of`, `denies`, `without`, `free of`, `absence of`
   - Lista de **gatilhos de pós-negação**: `was ruled out`, `was negative`
   - Ao encontrar um gatilho, abre-se uma **janela de escopo** (até o fim da sentença ou até pontuação forte / `and` / `but`)
   - Toda entidade dentro da janela é marcada como **negada**

   Versão estendida: **ConText**, que também classifica hipotético vs. real, e histórico/família vs. paciente atual.

   Exemplos do corpus:
   | Trecho | Gatilho | Entidade(s) negada(s) |
   |---|---|---|
   | "no evidence of malignancy" | `no evidence of` | malignancy |
   | "free of internal septations or associated masses" | `free of` | internal septations, associated masses |
   | "no mucosal abnormalities of the stomach" | `no` | mucosal abnormalities |
   | "no evidence of bleeding or leak" | `no evidence of` | bleeding, leak |

---

## Etapa 2 — Extrair metadados do Artigo

- **Ano** e **MeSH-terms** normalmente vêm de campos estruturados (XML/RIS do PubMed), não do corpo do texto.
- Extrair via parsing dos metadados da fonte, não via NLP sobre texto livre — mais confiável.

---

## Etapa 3 — Identificar Casos e dados demográficos do Paciente

- Casos clínicos seguem abertura quase formulaica. Regex útil:
  ```
  A \d+-year-old (wo)?man
  ```
  → extrai **Idade** e **Gênero** com alta precisão.
- Se o artigo descreve múltiplos casos, usar esse padrão como marcador para segmentar o texto em blocos de "Caso".

---

## Etapa 4 — Reconhecer entidades por dicionário (gazetteer / lookup)

**Conceito:** casar substrings do texto contra vocabulários controlados (listas fechadas e padronizadas, cada termo com um ID de conceito e sinônimos).

**Vocabulários por tipo de entidade:**
| Entidade do grafo | Vocabulário |
|---|---|
| Medicação | RxNorm |
| Exame / Resultado | LOINC |
| Sintoma, Doença, Partes do corpo | SNOMED CT |
| MeSH-terms do Artigo | MeSH |
| Unificador de todos acima | UMLS Metathesaurus (CUI) |

**Como funciona:**
1. Construir/baixar a tabela `termo → ID do conceito` (várias entradas de sinônimo apontam pro mesmo ID).
2. Indexar os termos com estrutura eficiente para múltiplos padrões (trie / **Aho-Corasick**).
3. Casar contra o texto sentença por sentença (após a negação já detectada), priorizando o **match mais longo** em caso de sobreposição.
4. Resolver para o **ID do conceito**, não o texto literal — isso normaliza sinônimos (`CT`, `computed tomography`, `CAT scan` → mesmo nó).

**Na prática:** rodar o texto pré-processado pelo **MetaMap** / **MetaMapLite** (que já fazem esse lookup contra o UMLS, incluindo desambiguação de siglas pelo contexto), em vez de reimplementar o índice do zero. Filtrar depois só os tipos semânticos relevantes (medicamento, sintoma, doença, procedimento, anatomia).

---

## Etapa 5 — Filtrar achados negados/incertos

- Cruzar as entidades reconhecidas na Etapa 4 com os spans negados marcados na Etapa 1.
- Entidade dentro de span negado → não vira aresta no grafo, ou vira com atributo `negado=true` (dependendo da modelagem desejada).

---

## Etapa 6 — Extrair relações via padrões sintáticos

1. **POS tagging**: etiquetar cada token com sua função gramatical (substantivo, verbo, etc.), usando padrão Penn Treebank. Preferir o **GENIA tagger** (treinado em abstracts biomédicos) a taggers genéricos, que erram em termos médicos.

   ```
   She/PRP underwent/VBD contrast/JJ enhanced/VBN computed/JJ tomography/NN
   ```

2. **Chunking (parsing raso)**: agrupar tokens etiquetados em blocos não sobrepostos (NP, VP, PP) via gramática de regex sobre as tags POS, sem construir árvore sintática completa:

   ```
   NP: {<DT>?<JJ>*<CD>?<JJ>*<NN.*>+}
   ```

   Resultado: `[NP She] [VP underwent] [NP contrast enhanced computed tomography]`

3. **Léxico de verbos → tipo de relação do esquema:**
   | Verbo/expressão no texto | Relação no grafo |
   |---|---|
   | underwent, performed | faz |
   | presented with, complained of | sente |
   | diagnosed with, consistent with, suggesting | diagnosticado |
   | history of | possui |
   | revealed, demonstrated | (resultado do Exame) |

4. **Heurística de ligação sujeito-relação-objeto:** NP imediatamente antes do VP = sujeito; NP imediatamente depois = objeto; sempre dentro da mesma sentença.

   Exemplo: `[NP She] — [VP underwent] — [NP computed tomography]` → `Paciente —faz→ Exame(computed tomography)`

---

## Etapa 7 — Resolver correferência do Paciente

- O texto usa pronomes (`she`, `her`) e `the patient` ao longo do caso.
- Resolução simples baseada em regra: mapear pronomes com o mesmo gênero identificado na Etapa 3 de volta ao nó Paciente do caso atual.
- Evita criar nós de paciente duplicados no grafo.

---

## Etapa 8 — Montar e validar o grafo

1. Converter as triplas `(entidade, relação, entidade)` em nós e arestas, seguindo o esquema Artigo → Casos → Paciente → atributos.
2. Validar manualmente uma amostra de casos contra o grafo gerado, medindo precisão/recall antes de rodar no corpus inteiro — pipelines baseados em regra tendem a ter boa precisão mas recall mais baixo.

---

## Resumo do fluxo

```
Texto bruto
   │
   ▼
[1] Segmentação + tokenização + detecção de negação
   │
   ▼
[2] Metadados do Artigo (Ano, MeSH-terms) — via campos estruturados
   │
   ▼
[3] Regex demográfico → Idade, Gênero, blocos de Caso
   │
   ▼
[4] Lookup em vocabulários controlados (UMLS/SNOMED/RxNorm/LOINC) → entidades candidatas
   │
   ▼
[5] Filtro de negação (cruza [1] com [4])
   │
   ▼
[6] POS tagging + chunking + léxico de verbos → relações (tem, toma, faz, sente, diagnosticado, possui, incorre em)
   │
   ▼
[7] Correferência do Paciente
   │
   ▼
[8] Triplas → grafo → validação manual
```