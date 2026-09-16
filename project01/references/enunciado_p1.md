# Processamento de Linguagem Natural 2026

## Projeto 1

Este projeto envolve a extração de dados de um repositório que cataloga milhares de casos clínicos, conhecido como MultiCaRe, e cada caso em um grafo de conhecimento. O MultiCaRe está descrito no artigo:

> Nievas Offidani, M., Roffet, F., González Galtier, M. C., Massiris, M., & Delrieux, C. (2025). An Open-Source Clinical Case Dataset for Medical Image Classification and Multimodal AI Applications. Data 2025, Vol. 10, Page 123, 10(8), 123. https://doi.org/10.3390/DATA10080123

O banco de dados extraiu os casos clínicos disponíveis de artigos publicados no PubMed.

Na pasta [sample/](sample/) estão disponíveis três arquivos de amostra, extraídos pelo notebook [`02a_csv_data_exploraton_sample.ipynb`](../../../to-kg/notebooks/02a_csv_data_exploraton_sample.ipynb) a partir de uma amostra fixa de 50 artigos do MultiCaRe (mais detalhes em [`to-kg/docs/data_source.md`](../../../to-kg/docs/data_source.md)):

- **[cases.csv](sample/cases.csv)** — um caso clínico por linha (um paciente). Colunas: `article_id` (PMCID do artigo, chave estrangeira para `metadata.csv`), `case_id` (`article_id` + sufixo sequencial, ex. `PMC5137649_01`), `case_text` (texto integral do relato de caso — a principal fonte para a extração de entidades e relações), `age` (idade; valores menores que 1 ano aparecem como 0) e `gender` (`Female`, `Male`, `Transgender` ou `Unknown`). 246 casos nesta amostra.

- **[metadata.csv](sample/metadata.csv)** — um artigo por linha. Colunas: `article_id` (PMCID, chave primária), `title`, `authors`, `journal`, `journal_detail`, `year`, `doi`, `pmid`, `pmcid`, `mesh_terms`/`major_mesh_terms` (termos MeSH do artigo), `keywords`, `link`, `license` e `case_amount` (nº de casos daquele artigo). Útil para contextualizar cada caso — por exemplo, cruzar as entidades extraídas do texto com os termos MeSH do artigo de origem, ou verificar a licença antes de reusar um trecho. 50 artigos nesta amostra. Atenção: como CSV não tem tipo array nativo, colunas de lista (`authors`, `mesh_terms`, `major_mesh_terms`, `keywords`) aparecem como texto entre colchetes (ex. `[Female]`), e `year` é texto, não número.

- **[data_dictionary.csv](sample/data_dictionary.csv)** — dicionário de dados original do Zenodo (colunas `file`, `field`, `explanation`), copiado sem filtro. Ele documenta os campos de **todos** os arquivos do dataset publicado, incluindo `captions_and_labels.csv`, `case_images.parquet` e `abstracts.parquet` — que **não** fazem parte desta amostra nem são usados neste projeto (o dataset original é focado em classificação de imagens médicas, e este projeto usa apenas o texto dos casos). Use as linhas com `file` igual a `cases.parquet` ou `metadata.parquet` como referência para os dois arquivos acima.

A equipe deve refletir e propor uma abordagem para os seguintes pontos:
1. Quais são os dados interessantes para serem extraídos e como:
  * entidades: sintomas, doenças, exames, tratamentos, medicamentos, regiões anatômicas, etc.;
  * valores associados a resultados de exames e dosagem de medicamentos, bem como suas unidades de medida;
2. Como é interessante organizar estes dados em um grafo de conhecimento.
3. Estratégias clássicas para extrair os dados desejados do texto, por exemplo: tokenização, normalização, remoção de stop-words, associação a dicionários/tesauros/ontologias

Para este estágio de trabalho, não poderão ser usados modelos de linguagem para a extração dos dados. A equipe deve explorar as técnicas tradicionais de modo que entenda cada algoritmo que foi usado e seu impacto na geração final do grafo.

Algumas estratégias interessantes que podem ser investigadas:
* Unificação de formas diferentes de escrever um mesmo conceito usando, por exemplo, dicionários, tesauros ou ontologias.
* Expressões regulares e modelos de regras para decompor o texto em elementos do grafo e relacionar estes componentes.

Recomendo a leitura deste artigo para a compreensão sobre conceitos de base de grafos de conhecimento:

> Ji, S., Pan, S., Cambria, E., Marttinen, P., & Yu, P. S. (2022). A Survey on Knowledge Graphs: Representation, Acquisition, and Applications. IEEE Transactions on Neural Networks and Learning Systems, 33(2), 494–514. https://doi.org/10.1109/TNNLS.2021.3070843

## Formato de extração do grafo de conhecimento

O grafo extraído de cada caso deve ser representado em duas tabelas simples:

* **nós**, com a identificação e os atributos de cada entidade extraída;
* **arestas**, com o nó de origem, o nó de destino e os atributos da relação entre eles.

Sugestão de esquema (livre para a equipe ajustar, mas mantendo a ideia de duas tabelas):

* **nós**: `node_id`, `type` (ex. `Symptom`, `Diagnosis`, `Exam`, `ExamResult`, `Treatment`, ...), `label` (nome/forma normalizada da entidade) e `attributes` (demais atributos da entidade, como `value`, `unit`, `reference_range`, `status`, `onset`, etc.).
* **arestas**: `edge_id`, `source_id`, `target_id`, `relation` (ex. `HAS_SYMPTOM`, `UNDERWENT_EXAM`, `SUPPORTS`, `TREATED_BY`, ...) e `attributes` (ex. trecho do texto que evidencia a relação, confiança, etc.).

Dois exemplos ilustrativos (em inglês, no estilo dos casos do MultiCaRe), cada um mostrando um texto de caso clínico e o grafo de conhecimento que poderia ser extraído dele, foram colocados em arquivos separados para não sobrecarregar este README:

- **[example1.md](example1.md)** — grafo "básico": cada entidade extraída (sintoma, histórico clínico, exame, resultado de exame, achado de imagem, diagnóstico, tratamento) é um nó, e seus atributos (valor, unidade, faixa de referência, status etc.) ficam agrupados na coluna `attributes` desse nó.
- **[example2.md](example2.md)** — mesmo caso clínico do Exemplo 1, mas levando a decomposição mais longe, ao estilo RDF: atributos que no Exemplo 1 eram texto dentro de `attributes` (por exemplo, o valor `850` e a unidade `U/L` de um resultado de exame) viram nós próprios, ligados por arestas (`HAS_VALUE`, `HAS_UNIT`, ...). Também mostra como ligar sintomas, tipos de exame e diagnósticos a vocabulários controlados/ontologias (SNOMED CT, LOINC, ICD-10, MeSH).

Os dois exemplos são ilustrativos — a equipe não precisa seguir exatamente estes tipos de nó/aresta, nem replicar o nível de decomposição de nenhum dos dois; pode escolher (ou combinar) o que fizer mais sentido para o projeto.

## Apresentação do Grafo

O grafo resultante pode ser apresentado visualmente (em uma interface em Python ou Web) e isto será considerado como ponto positivo do projeto. Por não ser objeto específico da disciplina, a construção da apresentação visual pode ser feita inteiramente com assistência de modelos de linguagem.

## Diferenciais do Projeto

Serão valorizados diferenciais do projeto como:
* criatividade
* originalidade
* audácia em propostas desafiadoras;
* recursos de visualização de dados do grafo
* busca por soluções diferenciais na literatura

É muito importante considerar que serão valorizadas equipes que arrisquem em análises ousadas, nas quais não se saiba se se alcançará o resultado esperado. Equipes podem obter nota máxima, mesmo que não alcancem o resultado esperado, considerando que apresentem um trabalho bem fundamentado, audacioso, que demonstre integração entre os componentes.

## Diretrizes para montagem dos slides

* Explore diagramas e ilustrações na apresentação.
* Não coloque muito texto, utilize apenas tópicos indicativos para guiar a apresentação
* Não há espaço para mostrar todos os detalhes; selecione aquilo que você acha mais importante e enfatize. Se vai mostrar um código, não mostre inteiro, apenas a parte que se destaca.

## Parte Escrita

O projeto tem uma parte escrita seguindo o [template](../template/project1/README.md). Cada equipe deve criar um projeto no GitHub do qual todos os membros participam e devem colocar a parte escrita lá. Nos slides de apresentação deve ser colocado o endereço do GitHub.