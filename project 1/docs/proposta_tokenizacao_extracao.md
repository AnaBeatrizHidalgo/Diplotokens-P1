# Proposta: Tokenização, Extração e Geração do Grafo

A ideia aqui é amarrar o que foi apresentado nas aulas do professor Santanchè com a estrutura que já montamos no repositório, respondendo também aos pontos que levantamos em `ideias.md`.

---

## 1. O que vimos nas aulas e como isso se aplica ao projeto

O conteúdo das aulas nos dá exatamente as ferramentas necessárias:

| O que precisamos fazer | Conceito da aula | Onde está nos slides | Como vamos usar no código |
| --- | --- | --- | --- |
| **Separar sentenças sem quebrar siglas** | Granularidade e fronteiras de sentença | *Aula: Words and Tokens*, slides 4–5 e 22–26 | Quebrar o texto em frases, mas protegendo abreviações médicas comuns (`Fig. 1`, `POD 4`, `Dr.`). |
| **Preservar termos médicos compostos** | Tokenização baseada em regras e pontuação | *Aula: Information Extraction*, slide 36; *Aula: Words and Tokens*, slides 28–29 | Não remover pontuação indiscriminadamente. Manter hífens (`3-day`, `CA 19-9`), barras (`120/80`, `ng/ml`) e decimais (`12,476.5`). |
| **Pegar dosagens, medidas e tempos** | Regex com grupos de captura `()` | *Aula: Information Extraction*, slides 20–30; *Aula: Words and Tokens*, slide 29 | No slide 30 da aula 1, o professor dá o exemplo de pressão com `([0-9]+)/([0-9]+)`. Vamos usar essa mesma lógica para pegar exames, unidades e tamanhos de lesões. |
| **Resolver siglas médicas e sinônimos** | Normalização e classes de equivalência | *Aula: Words and Tokens*, slides 49–52 | No slide 51, há o exemplo real de prontuário com siglas (`HTN`, `DM2`, `Cr`, `Hb`, `BUN`). Criamos um dicionário para mapear isso para os termos canônicos. |
| **Padronizar maiúsculas e minúsculas** | *Case folding* seletivo e tradeoff precisão/revocação | *Aula: Words and Tokens*, slides 55–57 | O professor mostra que jogar tudo para minúsculas aumenta revocação mas derruba precisão. Se fizermos isso, `US` (ultrassom) vira `us` (nós) e `BUN` vira pão. Por isso, identificamos siglas antes de normalizar. |
| **Padronizar plurais e tempos verbais** | Lematização vs. *Stemming* | *Aula: Words and Tokens*, slides 61–65 | Usar lematização em vez de Porter Stemmer. O stemmer corta palavras de forma bruta (ex.: *pancreatitis* viraria *pancreatit*). A lematização preserva a palavra médica (`cysts` ➔ `cyst`). |
| **Pegar termos médicos longos** | *POS Tagging* e *NP Chunking* (Sintagmas Nominais) | *Aula: Information Extraction*, slides 44–50 | A maioria dos termos médicos são sintagmas nominais (`ADJ + NOUN`). Com regras gramaticais simples sobre POS tags, pegamos expressões como *cystic lesion* e *lower quadrant abdominal pain*. |
| **Classificar os nós do grafo** | Ontologia clínica de 4 classes | *Aula: Information Extraction*, slides 39–43 | No slide 40 e 43, o professor mostra o grafo dele no Hugging Face: nó central `Patient` ligado a `Problem`, `Test`, `Treatment` e `Anatomy`. |
| **Identificar negações** | Padrões de negação | *Aula: Information Extraction*, slides 19 e 25 | Detectar termos como `denied`, `negative for`, `without` para marcar que o sintoma não estava presente. |

---

## 2. Como vamos tokenizar e normalizar

O texto clínico tem muitas armadilhas. Se fizermos apenas um `split()` ou removermos pontuação, quebramos os dados mais importantes:

1. **Tokenização de Sentenças:**
   * O texto de cada caso é quebrado em sentenças. As relações entre sintomas, exames e tratamentos geralmente acontecem dentro da mesma frase. 
   * Precisamos de uma lista de exceções para pontos que não encerram frase (`Fig.`, `vs.`, `POD`, `mg/dL`, `y.o.`).

2. **Case Folding Seletivo:**
   * Primeiro passamos uma checagem de siglas conhecidas em caixa alta (`CT`, `MRI`, `EUS`, `FNA`, `CEA`, `BUN`, `Hb`, `HTN`).
   * Depois, para casamento com o vocabulário geral de sintomas e anatomia, usamos minúsculas.

3. **Lematização:**
   * Reduzimos plurais e variações para a forma canônica (`cysts` ➔ `cyst`, `edemas` ➔ `edema`, `complaints` ➔ `complaint`). Isso ajuda a bater diretamente com os termos MeSH do arquivo `metadata.csv`.

---

## 3. Extração com Regex e Dicionários

Nosso motor em `diplo_grafo/` pode ser dividido em componentes simples e bem definidos:

### 3.1 Padrões de Regex principais
Inspirados no slide 30 da aula 1, podemos usar expressões regulares com grupos de captura para extrair dados escalares:

```python
import re

PATTERNS = {
    # Idade e genero (ex: "A 44-year-old woman", "65-year-old male", "20-month-old boy")
    "patient_demographics": re.compile(
        r"(?:A|An)\s+(\d{1,3})-(year|month|day)-old\s+(male|female|man|woman|boy|girl)",
        re.IGNORECASE,
    ),
    
    # Pressao arterial (ex: "195/110 mmHg", "120/80")
    "blood_pressure": re.compile(
        r"(\d{2,3})/(\d{2,3})\s*(?:mmHg)?",
        re.IGNORECASE,
    ),
    
    # Valores de laboratorio com unidades (ex: "12,476.5ng/ml", "1.9 mg/dL", "66 mg/dL")
    "lab_measurement": re.compile(
        r"(\d+(?:,\d+)*(?:\.\d+)?)\s*(mg/dL|g/dL|ng/mL|U/L|iu/ml|mm/h|mmol/L|g%|fL)",
        re.IGNORECASE,
    ),
    
    # Tamanho de lesoes e medidas (ex: "6cm x 9cm", "9.5cm x 4.5cm x 2.0cm", "4 cm")
    "dimensions": re.compile(
        r"(\d+(?:\.\d+)?)\s*(cm|mm)(?:\s*x\s*(\d+(?:\.\d+)?)\s*(cm|mm))?",
        re.IGNORECASE,
    ),
    
    # Duracao e historico temporal (ex: "3-day history", "3-year history")
    "duration": re.compile(
        r"(\d+)-(day|week|month|year)\s+history",
        re.IGNORECASE,
    ),
    
    # Negacoes (ex: "denied fever", "no evidence of malignancy", "without rebound")
    "negation": re.compile(
        r"\b(denied|denies|no evidence of|negative for|free of|unremarkable|without)\b",
        re.IGNORECASE,
    ),
}
```

### 3.2 Dicionário médico local (`data/external/`)
Em vez de depender de chamadas externas ou bibliotecas pesadas, mantemos um arquivo JSON ou tabela simples em `data/external/medical_lexicon.json` com termos comuns dos casos:
* **Anatomia:** `abdomen`, `stomach`, `liver`, `pancreas`, `kidney`, `lung`, `spleen`, `flank`.
* **Exames:** `computed tomography`, `CT`, `ultrasound`, `EUS-FNA`, `biopsy`, `blood workup`, `endoscopy`, `X-ray`.
* **Tratamentos:** `resection`, `pancreatectomy`, `stent`, `chemotherapy`, `antibiotics`, `drainage`, `laparoscopy`.
* **Abreviações:** Mapeando `HTN` ➔ `Hypertension`, `DM2` ➔ `Type 2 Diabetes`, `Hb` ➔ `Hemoglobin`, etc.

Também aproveitamos a coluna `mesh_terms` de `metadata.csv`, que já traz os termos MeSH associados ao artigo para enriquecer os diagnósticos.

---

## 4. Respondendo às dúvidas que anotamos em `ideias.md`

No documento `ideias.md` levantamos algumas perguntas importantes. Com o que vimos nas aulas, podemos fechar essas decisões:

* **"Uso de regex? Criar dicionário? Como tratar termos diferentes para mesma terminologia?"**
  * Usamos regex para números, unidades e padrões fixos.
  * Usamos um dicionário de sinônimos/MeSH para unificar termos equivalentes na forma canônica (ex.: *heart attack* e *myocardial infarction* apontam para o mesmo nó).
* **"Seria interessante algumas infos estarem guardadas dentro do nó?"**
  * Sim. No modelo de tabelas pedido pelo professor e aceito pelo nosso visualizador, valores como dosagem (`40 mg`), unidade (`mg/dL`) e duração (`3 days`) ficam na coluna `attributes` do próprio nó (ex: `value=40; unit=mg`).
* **"Existem perguntas nos casos clínicos, devemos ignorar?"**
  * Sim. Se uma sentença terminar com `?` ou contiver marcadores hipotéticos como *"could this be"*, podemos simplesmente descartar essa sentença no pré-processamento para evitar falsos positivos no grafo.

---

## 5. Como vamos ligar os nós (Arestas)

Para não complicar demais e manter regras claras e reproduzíveis, definimos as seguintes regras de ligação:

1. **Paciente (`P1`) ➔ Entidades:**
   * Se o caso cita sintomas no início: `P1 ➔ PRESENTS_WITH ➔ Symptom`
   * Se o caso cita histórico prévio: `P1 ➔ HAS_HISTORY ➔ Problem`
   * Se o paciente realizou exames: `P1 ➔ UNDERWENT_EXAM ➔ Exam`
   * Se houve diagnóstico final confirmado: `P1 ➔ DIAGNOSED_WITH ➔ Diagnosis`
   * Se o paciente passou por conduta/remédio: `P1 ➔ TREATED_BY ➔ Treatment`

2. **Exame ➔ Resultado / Achado:**
   * Se na mesma frase do exame houver uma medição ou achado: `Exam ➔ HAS_RESULT ➔ ExamResult` (ou `Exam ➔ REVEALS ➔ Finding`).

3. **Entidade ➔ Região Anatômica:**
   * Se um sintoma, lesão ou exame especificar uma parte do corpo (ex.: *cystic lesion of the pancreas*): `Finding ➔ LOCATED_IN ➔ Anatomy`.

---

## 6. Formato das tabelas de saída

As tabelas geradas em `data/processed/` seguem exatamente o formato exigido na entrega e já aceito pelo [Diploghraph-Viewer](https://github.com/TochaFh/Diploghraph-Viewer):

### `nodes.csv`
| node_id | type | label | attributes |
| --- | --- | --- | --- |
| P1 | Patient | Patient PMC5137649_01 | age=44; gender=Female |
| S1 | Problem | Abdominal pain | duration=3 days; location=right flank |
| E1 | Test | Contrast enhanced CT | modality=imaging |
| F1 | Finding | Cystic lesion | size=6cm; location=pancreas |
| E2 | Test | EUS-FNA | modality=imaging |
| R1 | ExamResult | CEA level | value=12476.5; unit=ng/ml |
| D1 | Diagnosis | Mucinous pancreatic cystic neoplasm | status=confirmed |
| T1 | Treatment | Laparoscopic distal pancreatectomy | type=surgery |

### `edges.csv`
| edge_id | source_id | target_id | relation | attributes |
| --- | --- | --- | --- | --- |
| e1 | P1 | S1 | PRESENTS_WITH | |
| e2 | P1 | E1 | UNDERWENT_EXAM | |
| e3 | E1 | F1 | REVEALS | |
| e4 | P1 | E2 | UNDERWENT_EXAM | |
| e5 | E2 | R1 | HAS_RESULT | |
| e6 | P1 | D1 | DIAGNOSED_WITH | |
| e7 | D1 | T1 | TREATED_BY | |

---

## 7. Próximos passos práticos

Para colocarmos essa proposta em prática no repositório:

1. **Adicionar biblioteca básica de NLP:** Incluir `nltk` ou `spacy` no `pyproject.toml` para termos sentencizador e lematizador confiáveis, rodando `make requirements`.
2. **Criar o módulo `diplo_grafo/clinical_regex.py`:** Colocar os padrões de regex organizados e testados.
3. **Criar o dicionário inicial:** Montar `data/external/medical_lexicon.json` com as siglas e termos MeSH da nossa amostra.
4. **Fazer o primeiro teste em notebook:** Criar `notebooks/01_extracao_sample.ipynb` rodando esse fluxo no caso `PMC5137649_01` e gerar os primeiros `nodes.csv` e `edges.csv` em `data/processed/`.
5. **Carregar no Diploghraph-Viewer:** Fazer o upload dos CSVs no visualizador web para ver o primeiro caso renderizado.
