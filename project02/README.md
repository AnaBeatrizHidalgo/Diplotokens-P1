# project02 Diplotokens

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Projeto 2 da equipe Diplotokens para a disciplina NLP da Unicamp (2026.2)

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
│                         diplo_graph2 and configuration for tools like black
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
└── diplo_graph2   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes diplo_graph2 a Python module
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

--------

## Slides

> Coloque aqui o link para o PDF da apresentação da parte 2.

## Demonstração Executável

> Coloque aqui um link para uma demonstração executável do seu código. Pode ser um Jupyter notebook (rodando no Binder ou no Colab) e/ou uma página Web que apresente visualmente seus resultados.

## Metodologia
> Descreva aqui a metodologia que foi usada para a extração do grafo de conhecimento. Você pode apresentar um diagrama, com estágios e descrever cada um deles.

> Nesta seção podem aparecer destaques de código como indicado a seguir. Note que foi usada uma técnica de highlight de código, que envolve colocar o nome da linguagem na abertura de um trecho com `~~~`, tal como `~~~python`.
>
> Os destaques de código devem ser trechos pequenos de poucas linhas, que estejam diretamente ligados a alguma explicação. Não utilize trechos extensos de código. Se algum código funcionar online (tal como um Jupyter Notebook), aqui pode haver links. No caso do Jupyter, preferencialmente para o Binder abrindo diretamente o notebook em questão.

### Arquitetura do Projeto
![Arquitetura do modelo](docs/NLPproj2Arquitetura.png)

0. **Pre-processamento**:

Nesta etapa, separamos os casos em conjuntos de treinamento e teste e aplicamos a tokenização para padronizar o texto nas abordagens testadas. O tokenizador divide o texto em palavras usando o NLTK, converte os tokens para minúsculas e remove pontuações; por padrão, mantém os números e reduz as palavras ao radical em inglês (stemming com Snowball).

Os arquivos responsáveis pela separação dos casos e pela tokenização estão em `models/pre_processamento`.

1. **Caixa preta 1 - Classificação de nos**:
- Abordagem 1: **NGNI**

- Abordagem 2: **Rotulação fraca + Regressão Logistica Multinomial**

Nesta abordagem, a rotulação fraca usa um dicionário de termos MeSH (doenças, sintomas, exames, medicamentos, resultados e anatomia), normalizado da mesma forma que os casos. A busca prioriza correspondências mais longas e marca os tokens com rótulos BIO (`B-` para o início e `I-` para a continuação da entidade). Menções ao paciente são identificadas por regras; termos ambíguos guardam seus tipos candidatos, e casos sem um rótulo confiável são ignorados no treinamento.

Para cada token, extraímos características da própria palavra (forma, prefixos, sufixos e radical) e do contexto próximo, incluindo palavras vizinhas e tipos encontrados no dicionário. Durante o treinamento, parte das indicações do dicionário é ocultada para que o modelo também aprenda padrões linguísticos. Essas características são vetorizadas e usadas por uma regressão logística multinomial, com pesos balanceados entre classes. Na predição, o modelo classifica os tokens e usa o contexto para escolher entre os tipos candidatos de termos ambíguos.


2. **Caixa preta 2 - Classificação de arestas/relações**

3. **Identificação de quantidades**

### Confronto com Trabalho Anterior

> Apresente dentro desta subseção um texto confrontando esta versão com a anterior. O confronto pode ser qualitativo ou baseado em comparações experimentais.

## Trabalhos Estudados

> Se foram feitas pesquisas de outros trabalhos, debata brevemente as referências.

## Modelo Lógico

> Modelo de grafo que a equipe criou. Para o modelo de grafos de propriedades, utilize este
> [modelo de base](https://docs.google.com/presentation/d/10RN7bDKUka_Ro2_41WyEE76Wxm4AioiJOrsh6BRY3Kk/edit?usp=sharing) para construir o seu.
> Coloque a imagem do PNG do seu modelo lógico como ilustrado abaixo (a imagem estará na pasta `image`):
>
> ![Modelo Lógico de Grafos](images/modelo-logico-grafos.png)
O modelo do grado permaneceu o que foi usado no projeto 1, pois durante discusões não se notou necessidade de alterar o modelo lógico para conseguir realizar a arquitetura que propusemos

![Modelo Lógico de Grafos](docs/modelo_grafo.jpeg)


## Análises que podem ser realizadas

> Apresente aqui uma análise  uma discussão de análises que podem ser realizadas com o seu grafo.
Com as caixas pretas estruturadas conseguimos realizar as seguinte comparações
1. Diferença entre soluções de classificação de nos

-> Aqui analisar os graficos que apresentamos
2. Comparação entre grafo projeto 1 e projeto 2

## Ferramentas

> Panorama das ferramentas utilizadas incluindo discussão sobre o uso das mesmas.

## Resultados

> Descrição e discussão dos resultados mais importantes obtidos.
>
> Você pode apresentar imagens apresentando o grafo e discutir o que obteve.

-> Indicação de metricas e mostra exemplo de grafo e as medidas coletadas

## Como Modelos de Linguagem foram Usados

> Descreva aqui em que tarefas os modelos de linguagem foram usados.

## Referências Bibliográficas

> Lista de artigos, links e referências bibliográficas.
>
> Fiquem à vontade para escolher o padrão de referenciamento preferido pelo grupo.