Estruturação do grafo


os nós:

* Paciente -> guarda info da idade e gênero
* Médico
* Remédios -> pode ter tempo tb, se for um remédio já tomado a um tempo
* Sintomas -> guardar o tempo que sente
* Doenças -> se for crônica, indicar a quanto tempo a pessoa tem
* Parte do corpo -> pode ser que se relacione tanto com alergia como sintoma
* Alergias/efeitos colaterais
* Dosagem?
* Exames
* Cirurgia
* Histórico (dor/doença crônica, uso de drogas...)? -> pode ser interessante guardar essas infos no paciente


Relações

* Diagnosticar: Medico diagnostica Doenças
* Sentir: Paciente sente sintomas
* Prescreve: Medico prescreve remédio
* Toma: Paciente toma remédio
* Possui: Paciente possui Doença
* Dose: Remédio dose Dosagem
* Dor: Sintoma dói parte do corpo
* Ocorre: Remédio ocasiona Alergia/efeito colateral
* Realiza: Paciente realiza exame/cirurgia


Exemplos

Medico -(diagnostica)- Doença -(Possui)- Paciente

Paciente -(toma)- Remédio -(Prescreve)- Médico


Como extrair do texto

* Uso de regex? -> criar um dicionário com os termos? -> existe algum docs em que  agente possa extrair esses termos médicos?

  * Tentar utilizar uma gramatica para identificar?
  * Como tratar termos diferentes para mesma termologia? ex: AVC e ataque do coração
  * Seria interesante relaizar contagem das palavras?
* Seria interessante algumas infos estarem guardadas dentro do nó?

  * Exemplo: dosagem do medicamento, o tempo do sintoma, o resultado do exame
* Existem perguntas nos casos clínicos, então deveríamos ignorar sintomas e doenças sitados nesses casos?

  * ex: PMC4630775\_01 (id do caso)
