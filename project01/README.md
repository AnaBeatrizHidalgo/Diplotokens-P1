# Projeto 1
# Project `Diplotokens`

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         diplo_grafo and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── diplo_grafo   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes diplo_grafo a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

## Slides

> Coloque aqui o link para o PDF da apresentação da parte 3.

## Metodologia

A metodologia desenvolvida para a extração e estruturação do grafo de conhecimento baseia-se em um *pipeline* sequencial e modular de Processamento de Língua Natural (PLN) focado em textos biomédicos não estruturados. O sistema isola a base de conhecimento estática das regras heurísticas de extração e da rotulação semântica. O fluxo de processamento é composto pelos seguintes estágios:

1. **Segmentação Textual (*Sentence Splitting*)**: O texto bruto de cada caso clínico é inicialmente decomposto em sentenças independentes. Utiliza-se um conjunto de expressões regulares desenhado para identificar os limites das frases, preservando abreviações médicas comuns (e.g., "Dr.", "Fig.") e pontuações interrogativas sem causar divisões espúrias.


2. **Tokenização Preservando Unidades**: As sentenças são segmentadas em unidades lexicais (*tokens*). A expressão regular empregada garante que grandezas numéricas permaneçam anexadas às suas respectivas unidades de medida clínicas e temporais (e.g., "mmHg", "mg", "months"), além de preservar intactas as palavras hifenizadas.


3. **Normalização e Reconstrução de Termos Multipalavra**: Emprega-se um algoritmo de janela deslizante gulosa (*longest-match*) que inspeciona de 2 a 5 *tokens* simultâneos para agrupar palavras contíguas em conceitos médicos complexos (e.g., "lower quadrant abdominal pain"). Esta etapa executa o tratamento de acrônimos médicos a partir de um mapeamento estático e a lematização de plurais irregulares para compatibilizar os termos com a base de conhecimento.


4. **Mapeamento Semântico via Vocabulário Clínico**: Os termos reconstruídos são comparados a uma ontologia central, ancorada em terminologias do *Medical Subject Headings* (MeSH), que atua como fonte única de verdade do projeto. Os termos detectados são mapeados para categorias semânticas estritas: *Symptom*, *Disease*, *Exam*, *Result*, *Anatomy*, *Procedure* e *Medication*.


5. **Extração de Atributos e Modificadores**: Através de expressões regulares otimizadas, o sistema extrai parâmetros contextuais e quantitativos da sentença, tais como dimensões físicas (priorizando o reconhecimento espacial 3D sobre 2D e 1D), medições laboratoriais atreladas a unidades, dosagens farmacológicas, durações sintomáticas e marcos temporais pós-operatórios. Estes atributos são associados às entidades clínicas correspondentes com base na vizinhança e distância de caracteres no texto.


6. **Resolução de Escopo de Negação**: Para evitar a extração de falsos positivos — como patologias ativamente descartadas no relato —, o sistema analisa a oração em busca de operadores de negação ou de resolução clínica (e.g., "no evidence of", "resolved"). O texto é previamente dividido em orações adversativas (usando "but", "however") para restringir com precisão o escopo da negação apenas à cláusula correta.


7. **Construção Lógica do Grafo de Propriedades**: O grafo é instanciado alocando o paciente como o nó central de origem. As entidades médicas extraídas são transformadas em nós canônicos compartilhados e conectadas ao paciente (ou entre si) por meio de arestas com semântica bem definida (e.g., `PRESENTS_WITH`, `TAKES`, `UNDERWENT_EXAM`, `DIAGNOSED_WITH`, `REVEALS`). Atributos quantitativos e cronológicos são incorporados diretamente como propriedades das arestas relacionais, minimizando a redundância estrutural da rede.



A orquestração central deste processo pode ser observada no núcleo da classe `KnowledgeGraphBuilder`, cujo método `build` ilustra a instanciação do nó raiz do paciente e a aplicação sequencial e contextual das regras extrativas para cada sentença:

```python
def build(self, case_row: pd.Series, sentences: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    self._reset()
    patient_attrs = f"age={case_row['age']}; gender={case_row['gender']}"
    patient_id = self.add_node(f"Patient {case_row['case_id']}", "Patient", patient_attrs)

    for sentence in sentences:
        ctx = self._extract_sentence_context(sentence)
        self._apply_symptoms(patient_id, ctx)
        self._apply_medications(patient_id, ctx, sentence.text)
        self._apply_exams(patient_id, ctx, sentence.text)
        self._apply_diseases(patient_id, ctx, sentence.text)

    return self.nodes_df, self.edges_df

```


---

### Metodologia da Linha de Base (*Baseline* com Vocabulário Sintético)

Como contraponto experimental ao pipeline determinístico canônico, implementou-se uma arquitetura de linha de base (*baseline*) focada em modularidade por arquivos externos e casamento léxico direto (*dictionary lookup*). Nessa abordagem, utilizou-se um conjunto de vocabulários sintéticos tabulares (`simulated_*.csv`), gerados com auxílio de Modelos de Linguagem (LLMs) a partir da indução de termos observados na amostra clínica inicial, servindo para aferir o comportamento do grafo sob expansão léxica automatizada.

O fluxo metodológico desse *baseline* opera sob os seguintes pilares:

1. **Inferência de Categorias por Metadados de Arquivo**: Em vez de concentrar o léxico em um módulo central único, o sistema varre o diretório de dados externos (`data/external/*.csv`) e infere a classe ontológica da entidade diretamente a partir da nomenclatura do arquivo (e.g., arquivos contendo `disease`, `icd` ou `mesh` geram nós do tipo `Disease`, enquanto prefixos como `symptom`, `anatomy` e `exam` alimentam as respectivas classes).


2. **Reconhecimento de Entidades por Casamento Guloso (*Longest-Match First*)**: Os termos de todos os arquivos externos são normalizados e ordenados de forma decrescente pelo número de palavras. A identificação de entidades no texto ocorre por meio de uma janela deslizante contígua de *tokens*, garantindo que expressões compostas mais longas tenham precedência sobre unigramas e evitando sobreposição de *spans* textuais (`occupied`).


3. **Bloqueio Heurístico de Escopo por Janela de Caracteres**: A detecção de negações e incertezas diagnósticas baseia-se em janelas de contexto rígidas em torno da entidade. O sistema inspeciona uma faixa de 60 caracteres precedentes em busca de marcadores de negação clínica (`NEGATION_RE`) e um intervalo assimétrico de caracteres (−35 a +90 posições) para marcadores de incerteza diagnóstica (`UNCERTAINTY_RE`). Caso qualquer padrão seja satisfeito, a entidade é descartada do grafo.


4. **Atribuição Relacional por Proximidade de Pistas (*Cue Words*)**: As arestas semânticas são inferidas identificando a pista textual (`RELATION_CUES`, tais como marcadores de histórico, queixa ou procedimento) permitida para a categoria e calculando a menor distância absoluta em caracteres até a entidade. Na ausência de pistas explícitas na frase, recorre-se a uma relação padrão por categoria (e.g., `Symptom` $\rightarrow$ `PRESENTS_WITH`).


5. **Topologia Estelar Centrada no Paciente**: Diferentemente da modelagem relacional encadeada (onde exames revelam resultados e sintomas localizam-se em anatomias), o *baseline* ancora quase a totalidade das arestas geradas diretamente no nó central do paciente (`source_id: patient_id`), estruturando o caso como uma topologia em estrela de atributos globais por sentença.



O trecho a seguir sintetiza a lógica de inferência de relações e o filtro de escopo por distância de caracteres aplicado no *baseline*:

```python
def relation_for(entity: dict, sentence: Sentence) -> str:
    allowed = {"Disease": {"HAS_HISTORY", "DIAGNOSED_WITH", "LOCATED_IN"},
               "Symptom": {"HAS_HISTORY", "PRESENTS_WITH", "LOCATED_IN"},
               "Exam": {"UNDERWENT_EXAM"}, "Medication": {"TAKES"}}.get(entity["type"])
    
    position = word_position(sentence, entity["start"])
    cues = [(abs(m.start() - position), rel)
            for rel, pattern in RELATION_CUES.items()
            for m in pattern.finditer(sentence.text) if rel in allowed]
    
    return min(cues)[1] if cues else DEFAULT_RELATIONS.get(entity["type"], "MENTIONS")

```

---


## Modelo Lógico

O modelo de dados foi estruturado seguindo o paradigma de Grafo de Propriedades (*Property Graph*), onde as informações clínicas são representadas como uma rede de nós semânticos interconectados por arestas direcionadas tipadas. Essa abordagem permite associar metadados diretamente aos nós e atributos relacionais (como dosagens ou duração) diretamente às arestas, reduzindo a redundância estrutural da rede e capturando com precisão o raciocínio médico.



Conforme ilustrado no diagrama lógico abaixo, a topologia foi desenhada de forma paciente-cêntrica. O nó `Paciente` atua como o vértice raiz da rede, encapsulando metadados demográficos (idade e gênero) como propriedades internas da própria entidade. A partir deste nó central, irradiam-se as arestas que representam os eventos clínicos e o histórico de saúde do indivíduo:

![Modelo Lógico do Grafo de Conhecimento](project1/docs/modelo_logico.jpeg)

1. **Relações Diretas do Paciente**:
* `Paciente` $\xrightarrow{\text{possui}}$ `Histórico Familiar` e `Histórico Individual`

* `Paciente` $\xrightarrow{\text{toma}}$ `Medicação`

* `Paciente` $\xrightarrow{\text{faz}}$ `Exame`

* `Paciente` $\xrightarrow{\text{sente}}$ `Sintoma`

* `Paciente` $\xrightarrow{\text{diagnosticado com}}$ `Doença`

* `Paciente` $\xrightarrow{\text{tem}}$ `Partes do corpo afetadas`



2. **Relações de Desdobramento Clínico (Cascata)**:
Entidades clínicas secundárias não se conectam diretamente ao paciente, mas derivam dos eventos primários, espelhando a relação de causa, efeito e investigação do raciocínio médico:


* `Medicação` $\xrightarrow{\text{faz}}$ `Efeito`

* `Exame` $\xrightarrow{\text{revela}}$ `Resultado`

* `Sintoma` $\xrightarrow{\text{pode ocorrer em}}$ `Partes do corpo afetadas`

* `Doença` $\xrightarrow{\text{incorre em}}$ `Partes do corpo afetadas`




Essa modelagem favorece o reaproveitamento de nós no nível global da base de dados. Por exemplo, a entidade `Doença` (e.g., "gastric duplication cyst") é instanciada apenas uma vez como nó canônico; se múltiplos pacientes possuírem o mesmo diagnóstico, múltiplas arestas `diagnosticado com` apontarão para esse mesmo vértice compartilhado.

## Análises que podem ser realizadas

> Apresente aqui uma análise  uma discussão de análises que podem ser realizadas com o seu grafo.

## Ferramentas

> Panorama das ferramentas utilizadas incluindo discussão sobre o uso das mesmas.
Cookiecutter: Ferramenta de criação organização de arquivos recomendada pelo professor, que após vermos como funciona, decidimos aderir à ela.

Python e Nootebook: Linguagem com alto suporte para auxiliar na tarefa apresentada e o uso de Nootebook pela praticidade de poder alterar partes do código sem precisar rodar o código inteiro novamente.

Diplograph: Ferramenta de visualização do nosso grafo desenvolvida pelo modelo de linguagem Gemini.

## Resultados

> Descrição e discussão dos resultados mais importantes obtidos.
>
> Você pode apresentar imagens apresentando o grafo e discutir o que obteve.

## Como Modelos de Linguagem foram Usados

> Descreva aqui em que tarefas os modelos de linguagem foram usados.
Foram utilizados modelos de linguagem para criar um dicionário de teste com os termos médicos para compararmos com os resultados obtidos utilizando um dicionário real

Também foi utilizado um modelo de linguagem para criar a ferramenta de visualização do grafo: Diplograph

## Referências Bibliográficas

> Lista de artigos, links e referências bibliográficas.
>
> Fiquem à vontade para escolher o padrão de referenciamento preferido pelo grupo.
