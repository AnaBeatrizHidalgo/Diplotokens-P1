# Guia de desenvolvimento

Este documento explica os comandos disponíveis no `Makefile`, a organização do
projeto e o papel de cada categoria de dados. Os comandos abaixo devem ser
executados a partir da pasta `project 1`.

## Utilidades do Makefile

O `Makefile` reúne comandos frequentes em alvos (targets). Assim, tarefas que
exigiriam vários comandos podem ser executadas com uma chamada curta, como
`make requirements` ou `make lint`.

### Comandos principais

| Comando | Finalidade |
| --- | --- |
| `make` ou `make help` | Lista os alvos disponíveis. `help` e o alvo padrao. |
| `make requirements` | Instala o projeto e as dependencias do `pyproject.toml` com `pip install -e .`. A instalacao editavel faz com que o pacote local acompanhe as alteracoes no codigo. |
| `make clean` | Remove arquivos compilados do Python (`.pyc`, `.pyo`) e pastas `__pycache__`. |
| `make lint` | Verifica formatacao e problemas de estilo usando o Ruff, sem alterar os arquivos. |
| `make format` | Corrige problemas automaticos do Ruff e formata os arquivos Python. |
| `make create_environment` | Tenta criar um ambiente virtual usando `virtualenvwrapper`. |
| `make data` | Instala as dependencias e executa `diplo_grafo/dataset.py`. |

### Adicionando dependencias

Para adicionar uma nova dependencia, inclua-a manualmente na lista
`dependencies` do `pyproject.toml` e depois execute `make requirements`. Instalar
um pacote diretamente com `pip install pacote` nao atualiza o arquivo TOML.

Os alvos marcados com `.PHONY` representam tarefas, e nao arquivos. Isso faz
com que, por exemplo, `make clean` execute a limpeza mesmo que exista um
arquivo chamado `clean`.

O alvo `data` depende de `requirements`, por isso a instalacao e executada
antes do script de dados. Atualmente, `dataset.py` possui um processamento
demonstrativo com `tqdm` e `loguru`; ele ainda precisa ser implementado para
ler, transformar e gravar os datasets do projeto.

## Organizacao do codigo

| Caminho | Responsabilidade |
| --- | --- |
| `diplo_grafo/config.py` | Define os caminhos principais do projeto, como `data/raw`, `data/processed`, `models` e `reports`. |
| `diplo_grafo/dataset.py` | Baixa, carrega ou transforma dados. E o script chamado por `make data`. |
| `diplo_grafo/features.py` | Deve concentrar a criacao e a transformacao de atributos para analise ou modelagem. |
| `diplo_grafo/plots.py` | Funcoes para gerar visualizacoes. |
| `diplo_grafo/modeling/train.py` | Codigo de treinamento dos modelos. |
| `diplo_grafo/modeling/predict.py` | Codigo de inferencia usando modelos treinados. |
| `data/` | Arquivos de entrada, intermediarios e processados. |
| `models/` | Modelos treinados, previsoes serializadas ou resumos dos modelos. |
| `notebooks/` | Exploracao, experimentos e analises interativas. |
| `references/` | Enunciados, artigos, manuais, dicionarios e outros materiais de consulta. |
| `reports/` | Relatorios e resultados gerados. |
| `reports/figures/` | Graficos e figuras usados nos relatorios. |
| `docs/` | Documentacao de desenvolvimento e orientacoes de uso do projeto. |
| `pyproject.toml` | Metadados do pacote, dependencias e configuracao do Ruff. |
| `Makefile` | Atalhos para instalacao, limpeza, verificacao, formatacao e processamento. |

Embora `models/` faca parte da estrutura padrao do Cookiecutter Data Science,
provavelmente ela nao sera utilizada neste projeto. O enunciado prioriza NLP
tradicional, com regras, expressoes regulares, dicionarios, tesauros e
ontologias, sem modelos de linguagem. A pasta pode permanecer com o
`.gitkeep` para preservar a estrutura e acostumarmos com ela.

O fluxo recomendado e manter a logica reutilizavel nos modulos de
`diplo_grafo/` e usar notebooks principalmente para exploracao e registro de
experimentos. Isso evita que regras importantes existam somente em uma celula
de notebook.

## Classificacao dos dados

A pasta `data/` segue a convencao do Cookiecutter Data Science. As categorias
indicam a origem e o grau de processamento de cada arquivo.

| Pasta | Tipo de dado | Uso esperado |
| --- | --- | --- |
| `data/external/` | Dados externos | Arquivos obtidos de fontes de terceiros, APIs, repositorios ou bases publicas. |
| `data/raw/` | Dados brutos | Dados originais, preservados como foram recebidos ou coletados. Devem ser tratados como imutaveis. |
| `data/interim/` | Dados intermediarios | Resultados parciais de limpeza, conversao, filtragem ou combinacao. |
| `data/processed/` | Dados processados | Versao final e canonica, pronta para analise, extracao de caracteristicas ou modelagem. |

Um fluxo comum e:

```text
external/raw -> interim -> processed -> features/modelos/relatorios
```

Nem todo projeto precisa usar todas as etapas. Por exemplo, uma fonte externa
pode ser copiada diretamente para `raw`, enquanto uma tabela intermediaria pode
ser criada em `interim` antes da geracao dos nos e arestas do grafo.

Os caminhos usados pelo codigo nao devem ser espalhados em varios modulos.
Eles ficam centralizados em `diplo_grafo/config.py`, por exemplo:

```python
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
```

## Versionamento dos dados

Como os dados deste projeto sao pequenos, `data/` permanece versionada no
repositorio. Por isso, a regra `/data/` esta comentada no `.gitignore`.

Se os dados crescerem muito ou contiverem informacoes que nao devam ser
publicadas, a regra pode ser reativada. Nesse caso, deve-se avaliar o uso de
Git LFS, armazenamento externo ou arquivos de configuracao que documentem como
obter os dados novamente.

O arquivo `.gitkeep` e apenas uma convencao para manter uma pasta vazia no Git.
Quando uma pasta de `data/` ja possui arquivos versionados, ele deixa de ser
necessario, mas pode permanecer sem afetar o funcionamento do projeto.
