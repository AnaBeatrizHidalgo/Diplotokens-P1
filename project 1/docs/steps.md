# passo a passo poggers

1 - separar em frase  
2 - tokenizar?  
3 - lematizar usando LVG
4 - lidar com frases de perguntas  
5 - lidar com frases com negação  
6 - fazer as relações de entidades.

---
## Passo a Passo poggerers (mais poggers)

1. Para cada case:

    2. Estruturar das informações básicas do csv
        - ID, age, gender..

    3. Usar regex sep para organizar o texto em uma lista de sentenças. Uma sentença pode ter um marcador de "has interrogation".

    4. Para cada sentença:

        - - Não sei se a lematização entra antes ou depois da tokenização

        5. Tokenização: organizar cada sentença em uma lista de tokens

        6. Classificação:
            - Para cada token:
                - Checar se é um termo classificável como: doença, quantidade. Já dá pra "marcar" aqui se o token representa uma DENOTAÇÃO (diagnosticar, ter histórico, realizar procedimento etc)
                - Descartar tokens inúteis?
        
        7. Identificar e marcar se a sentença é "duvidosa" (contém negação, pergunta, possibilidade tipo "could have"...). Para uma primeira versão podemos simplesmente descartar e anular essas sentenças. Para sentenças com interrogação, pode haver uma lógica para anular apenas a última denotação.

    8. Para cada sentença válida:

        9. Transformar em arestas ligando nós:

            - Exemplo: se há um token "diagnosticar" e 3 doenças, serão criadas arestas "diagnosticado -> doenças".

            - Tem ter um script para resolver sentenças com mais de um verbo/ação que ligue um mesmo tipo de nó, tipo se houver "diagnosticar" e "histórico" na mesma frase com n doenças, o algoritmo tem que decidir quais doenças vão pra cada aresta.



```mermaid
graph LR
    %% Nós Centrais e Estruturais
    
    Paciente((Paciente))

    %% Atributos e Relações Diretas do Paciente

    Paciente -- "possui histórico familiar" --> Doenca["Doença"]
    Paciente -- "possui histórico individual" --> Doenca["Doença"]
    Paciente -- "toma" --> Medicacao["Medicação"]
    Paciente -- "faz" --> Exame["Exame"]
    Paciente -- "sente" --> Sintoma["Sintoma"]
    Paciente -- "diagnosticado" --> Doenca["Doença"]
    

    %% Desdobramentos Clínicos Secundários
    Medicacao -- "faz" --> Efeito["Efeito"]
    Exame -- "revela" --> Resultado["Resultado"]
    Doenca -- "ocorre em" --> PartesCorpo
    Sintoma -- "ocorre em" --> PartesCorpo
    

    %% Estilização visual (Níveis de Hierarquia)
    classDef main fill:#1E293B,stroke:#0F172A,stroke-width:2px,color:#ffffff;
    classDef entity fill:#F1F5F9,stroke:#64748B,stroke-width:1.5px,color:#0F172A;
    classDef target fill:#E0E7FF,stroke:#4338CA,stroke-width:1.5px,color:#1E1B4B;

    class Artigo,Casos,Paciente main;
    class MeSH,Ano,Idade,Genero,Historico,Medicacao,Sintoma,Exame,Doenca,Efeito,Resultado entity;
    class PartesCorpo target;
```

