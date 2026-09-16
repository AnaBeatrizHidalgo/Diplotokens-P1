# Construção do Grafo de Conhecimento Clínico (Pipeline de Extração)

Este documento detalha a arquitetura e as 6 etapas determinísticas empregadas para converter o texto clínico livre dos relatos de caso em um Grafo de Conhecimento estruturado em nós e arestas com atributos ricos, pronto para consumo pelo visualizador D3/React e pelo Mermaid.

A implementação de referência deste pipeline pode ser consultada no notebook [`notebooks/complete_case_analysis.ipynb`](file:///home/raphael/Desktop/faculdade/MC896_NLP/Diplotokens-P1/project%201/notebooks/complete_case_analysis.ipynb) e nos módulos [`diplo_grafo/sentence.py`](file:///home/raphael/Desktop/faculdade/MC896_NLP/Diplotokens-P1/project%201/diplo_grafo/sentence.py) e [`diplo_grafo/clinical_terms.py`](file:///home/raphael/Desktop/faculdade/MC896_NLP/Diplotokens-P1/project%201/diplo_grafo/clinical_terms.py).

---

## O Ponto de Partida

Para cada sentença do relato clínico, o extrator recebe dois insumos já preparados pelas camadas anteriores do Diplotokens:

1. **O texto original completo (`orig_text`)**: Essencial para preservar pontuação, calcular spans absolutos e computar distâncias lineares de caracteres entre termos e medidas.
2. **A lista de tokens limpos e reconstruídos**: Onde termos multipalavras clínicos e siglas foram unificados em tokens únicos:
   ```python
   [
       "A", "44", "year-old", "woman", "presented", "with", "a", "3", "day",
       "history", "of", "right flank", "and", "lower quadrant abdominal pain",
       "associated", "with", "nausea", "and", "constipation", "."
   ]
   ```

A partir dessa estrutura, executamos **6 passos determinísticos** para montar o grafo.

---

## Diagrama do Pipeline de Construção (6 Etapas)

```mermaid
flowchart TD
    %% Entradas
    subgraph INPUTS ["Entradas do Pipeline"]
        IN1["Sentenças Segmentadas (orig_text)"]
        IN2["Tokens Clínicos (multipalavras unificadas)"]
        IN3["Metadados do Caso (case_id, age, gender)"]
    end

    %% Etapa 1
    subgraph STEP1 ["Etapa 1: Tipagem das Entidades (Matching Ontológico)"]
        S1_PROC["Cruzamento com ENTITY_CATEGORIES<br/>Classificação dos tokens em categorias"]
        S1_OUT["Entidades Tipadas na Sentença:<br/>• Patient, Symptom, Exam<br/>• Result, Disease, Anatomy, Medication"]
    end

    %% Etapa 2
    subgraph STEP2 ["Etapa 2: Filtro de Negação com Escopo Adversativo"]
        S2_PROC["Quebra da frase por orações ('but', 'however')<br/>Busca por gatilhos ('no evidence of', 'free of')"]
        S2_COND{"Entidade sob<br/>escopo de negação?"}
        S2_DROP["Descarta / Marca negada<br/>(ex: 'no evidence of malignancy')"]
        S2_KEEP["Mantém Entidade Ativa<br/>(ex: 'presence of extracellular mucin')"]
    end

    %% Etapa 3
    subgraph STEP3 ["Etapa 3: Extração de Atributos e Vinculação por Proximidade"]
        S3_REGEX["Regex no texto original (orig_text):<br/>• Medidas 1D/2D/3D (cm, mm)<br/>• Laboratório (ng/ml, iu/ml)<br/>• Linha do tempo (POD, duration)<br/>• Lateralidade & Aspecto"]
        S3_SPAN["Cálculo de Distância Linear de Spans:<br/>dist = start_medida - end_entidade<br/>Vincula ao nó Result mais próximo (limiar < 30)"]
    end

    %% Etapa 4
    subgraph STEP4 ["Etapa 4: Decisão de Modelagem (Nó vs Aresta)"]
        S4_NODE["Atributos Intrínsecos -> No NÓ<br/>• Patient: age, gender<br/>• Result: size, value, unit, status"]
        S4_EDGE["Atributos Relacionais -> Na ARESTA<br/>• PRESENTS_WITH: duration<br/>• LOCATED_IN: laterality, aspect<br/>• TAKES: dosage<br/>• DIAGNOSED_WITH: status, method"]
    end

    %% Etapa 5
    subgraph STEP5 ["Etapa 5: Aplicação das Regras Semânticas (Triplas)"]
        S5_R1["Sintomas & Queixas:<br/>Patient --PRESENTS_WITH--> Symptom<br/>Symptom --LOCATED_IN--> Anatomy"]
        S5_R2["Investigação & Procedimentos:<br/>Patient --UNDERWENT_EXAM--> Exam<br/>Exam --REVEALS--> Result<br/>Result --LOCATED_IN--> Anatomy"]
        S5_R3["Diagnósticos & Patologia:<br/>Patient --DIAGNOSED_WITH--> Disease<br/>Disease --LOCATED_IN--> Anatomy"]
        S5_R4["Farmacoterapia:<br/>Patient --TAKES--> Medication"]
    end

    %% Etapa 6
    subgraph STEP6 ["Etapa 6: Deduplicação e Exportação"]
        S6_DEDUP["get_or_create_node(label, type):<br/>• Reutiliza IDs de nós já existentes (P1, A2...)<br/>• Concatena novos atributos de sentenças futuras"]
        S6_CSV1["nodesCase1.csv (node_id, type, label, attributes)"]
        S6_CSV2["edgesCase1.csv (edge_id, source_id, target_id, relation, attributes)"]
    end

    %% Conexões
    INPUTS --> S1_PROC
    S1_PROC --> S1_OUT
    S1_OUT --> S2_PROC
    S2_PROC --> S2_COND
    S2_COND -- "Sim" --> S2_DROP
    S2_COND -- "Não" --> S2_KEEP

    S2_KEEP --> S3_REGEX
    IN1 -.-> S3_REGEX
    S3_REGEX --> S3_SPAN

    S3_SPAN --> STEP4
    STEP4 --> STEP5

    STEP5 --> S6_DEDUP
    S6_DEDUP --> S6_CSV1
    S6_DEDUP --> S6_CSV2
```

---

## As 6 Etapas Determinísticas

### 1. Tipagem das Entidades (Matching Ontológico)
Cruzamos os tokens da sentença contra um dicionário léxico estruturado (`ENTITY_CATEGORIES`), que classifica cada termo em uma das classes ontológicas do grafo:

* **`Patient`**: Criado na largada usando os dados estruturados do CSV (`case_id`, `age`, `gender`).
* **`Symptom`**: `lower quadrant abdominal pain`, `nausea`, `constipation`, etc.
* **`Exam`**: `contrast enhanced computed tomography`, `EUS-FNA`, `FNA`, etc.
* **`Result`**: `cystic lesion`, `pancreatic echotexture`, `carcinoembryonic antigen`, etc.
* **`Disease`**: `mucinous pancreatic cystic neoplasm`, `gastric duplication cyst`, etc.
* **`Anatomy`**: `right flank`, `stomach`, `pancreas`, `posterior wall`, etc.
* **`Medication`**: `chemotherapy`, `isoniazid`, `rifampin`, etc.

Se a sentença tem tokens pertencentes a essas classes, ela se torna candidata a gerar nós e arestas.

---

### 2. Filtro de Negação com Respeito a Orações Adversativas (`is_entity_negated`)
Antes de criar qualquer aresta, precisamos garantir que o médico não estava negando a condição:

* Buscamos gatilhos como: `no evidence of`, `negative for`, `free of`, `denies`, `without`, `uninvolved`.
* **Cuidado crucial com conjunções**: Se aplicássemos a negação na frase inteira, quebraríamos trechos como:
  > *"FNA of the cyst demonstrated **no evidence of malignancy** BUT did show the **presence of extracellular mucin**..."*
* **Solução**: O extrator fatia a sentença em orações separadas por `but`, `however` ou `although`. Assim, `malignancy` é descartada (ou marcada como negada), mas `extracellular mucin` continua ativa para o grafo!

---

### 3. Extração de Atributos por Regex e Vinculação por Proximidade
Os tokens nos dão os conceitos médicos, mas os números e qualificadores precisam ser amarrados a eles:

* **Dimensões (1D, 2D e 3D)**: `6cm`, `6cm x 9cm`, `9.5cm x 4.5cm x 2.0cm`.
* **Medições laboratoriais**: `12,476.5 ng/ml`, `6 iu/ml`.
* **Duração / Histórico**: `3-day history`.
* **Linha do tempo pós-operatória**: `POD 4` (*postoperative day 4*).
* **Lateralidade e Aspecto**: `right`, `left`, `posterior`, `anterior`.

#### Como associamos a medida ao nó correto?
Usamos **proximidade de caracteres (spans)** no texto original. Por exemplo, quando o regex encontra `6cm x 9cm`, calculamos a distância linear de caracteres entre o início da medida e o fim de todos os nós `Result` encontrados na mesma frase:

$$\text{distância} = \text{start\_medida} - \text{end\_entidade}$$

No texto:
> `"...revealed normal pancreatic echotexture and a cyst measuring 6cm x 9cm that was free..."`

* Distância de `6cm x 9cm` até `pancreatic echotexture`: **22 caracteres**.
* Distância de `6cm x 9cm` até `cyst`: **11 caracteres** (as letras de `" measuring "`).

O algoritmo seleciona o menor valor (11) e vincula a dimensão `size=6cm x 9cm` diretamente ao nó de `cyst`. Aplicamos um limiar de sanidade (`dist < 30`) para evitar associações indevidas a longa distância.

---

### 4. Decisão de Modelagem: O que vai no Nó vs O que vai na Aresta
Para manter o grafo limpo e semanticamente rico, separamos propriedades intrínsecas de modificadores relacionais:

| Onde colocar? | Tipo de Informação | Exemplos no nosso Grafo |
|---|---|---|
| **No NÓ (`attributes`)** | Propriedades intrínsecas da entidade que não dependem da relação. | • `Patient`: `age=44.0; gender=Female`<br>• `Result`: `size=6cm x 9cm`<br>• `Result`: `value=12,476.5; unit=ng/ml`<br>• `Result`: `status=normal` |
| **Na ARESTA (`attributes`)** | Modificadores do evento, tempo, dosagem ou certeza. | • `PRESENTS_WITH`: `duration=3-day history`<br>• `LOCATED_IN`: `laterality=right` ou `aspect=posterior`<br>• `TAKES`: `dosage=300 mg`<br>• `DIAGNOSED_WITH`: `status=confirmed; method=pathology` |

---

### 5. Aplicação das Regras Semânticas de Relação (Geração das Triplas)
Dentro do escopo de cada sentença ativa, disparadores sintático-semânticos ligam as entidades:

#### A. Apresentação Clínica (`Patient --PRESENTS_WITH--> Symptom`)
* **Gatilho**: Presença de `Symptom` associada a verbos como *presented with*, *complained of*, *history of*, *pain*.
* **Ação**:
  * Cria aresta `Patient --PRESENTS_WITH [duration=3-day history]--> Symptom`.
  * Se for dor e houver anatomia na frase: liga `Symptom --LOCATED_IN [laterality=right]--> Anatomy`.

#### B. Investigação por Imagem e Procedimentos (`Patient --UNDERWENT_EXAM--> Exam`)
* **Gatilho**: Presença de `Exam` (*computed tomography*, *EUS-FNA*, *endoscopy*).
* **Ação**:
  * Cria `Patient --UNDERWENT_EXAM [timeline]--> Exam`.
  * Todos os `Result` daquela sentença são ligados via `Exam --REVEALS--> Result` (já com suas dimensões e valores laboratoriais embutidos no nó de resultado).
  * Liga os achados à anatomia descrita: `Result --LOCATED_IN [aspect]--> Anatomy`.

#### C. Diagnóstico e Histórico Patológico (`Patient --DIAGNOSED_WITH--> Disease`)
* **Gatilho**: Presença de `Disease` (*mucinous pancreatic cystic neoplasm*, *GDC*).
* **Ação**:
  * Se o contexto indicar histórico prévio (*past medical history*), cria `HAS_PERSONAL_HISTORY` ou `HAS_FAMILY_HISTORY`.
  * Se for diagnóstico da internação: cria `DIAGNOSED_WITH`.
  * **Inferência de certeza**:
    * Se na frase tem *pathology* ou *consistent with*: anota na aresta `status=confirmed; method=pathology`.
    * Se tem *suggesting* ou *suspected*: anota `status=suspected; method=clinical/imaging`.
  * Conecta a doença ao órgão afetado: `Disease --LOCATED_IN--> Anatomy`.

---

### 6. Deduplicação e Exportação dos CSVs
Para evitar que o nó `stomach` ou o exame `CT` sejam duplicados a cada frase em que aparecem:

1. Mantemos um registro central via `get_or_create_node(label, type)`.
2. Se o nó já existe, reutilizamos seu `node_id` (ex: `A2`). Se uma sentença futura trouxer um detalhe novo (ex: um `status=normal`), o atributo é concatenado no nó existente.
3. Por fim, geramos os DataFrames e salvamos em:
   * [`data/processed/nodesCase1.csv`](file:///home/raphael/Desktop/faculdade/MC896_NLP/Diplotokens-P1/project%201/data/processed/nodesCase1.csv)
   * [`data/processed/edgesCase1.csv`](file:///home/raphael/Desktop/faculdade/MC896_NLP/Diplotokens-P1/project%201/data/processed/edgesCase1.csv)

---

## Visualização do Grafo Gerado (Exemplo: Caso 1)

O processamento completo do relato do Caso 1 (`PMC5137649_01`) resulta no seguinte Grafo de Conhecimento:

```mermaid
graph LR
    P1(("Patient PMC5137649_01<br/><small>age=44.0; gender=Female</small>"))
    S1["lower quadrant abdominal pain"]
    S2["nausea"]
    S3["constipation"]
    A1["right flank"]
    E1["contrast enhanced computed tomography"]
    R1["cystic lesion <br/><small>size=6cm</small>"]
    A2["stomach"]
    A3["pancreas"]
    E2["EUS-FNA"]
    R3["cyst <br/><small>size=6cm x 9cm</small>"]
    E3["FNA"]
    R5["carcinoembryonic antigen <br/><small>12,476.5 ng/ml</small>"]
    D1["mucinous pancreatic cystic neoplasm"]
    D2["gastric duplication cyst"]

    P1 -- "PRESENTS_WITH [duration=3-day history]" --> S1
    S1 -- "LOCATED_IN [laterality=right]" --> A1
    P1 -- "PRESENTS_WITH [duration=3-day history]" --> S2
    P1 -- "PRESENTS_WITH [duration=3-day history]" --> S3

    P1 -- "UNDERWENT_EXAM" --> E1
    E1 -- "REVEALS" --> R1
    R1 -- "LOCATED_IN" --> A2
    R1 -- "LOCATED_IN" --> A3

    P1 -- "UNDERWENT_EXAM" --> E2
    E2 -- "REVEALS" --> R3

    P1 -- "UNDERWENT_EXAM" --> E3
    E3 -- "REVEALS" --> R5
    P1 -- "DIAGNOSED_WITH [status=suspected]" --> D1

    P1 -- "DIAGNOSED_WITH [status=confirmed]" --> D2
    D2 -- "LOCATED_IN [aspect=posterior]" --> A2
```
