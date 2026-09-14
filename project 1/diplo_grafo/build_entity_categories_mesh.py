"""
build_entity_categories_mesh.py

Constroi o dicionario ENTITY_CATEGORIES (Symptom, Disease, Exam, Result,
Anatomy, Procedure, Medication) a partir do arquivo de descritores do MeSH
(desc<ano>.xml), usando os "Tree Numbers" (hierarquia oficial do MeSH) para
classificar cada termo.

COMO CONSEGUIR O ARQUIVO DE ENTRADA
------------------------------------
1. Acesse a pagina oficial de downloads do MeSH da NLM:
   https://www.nlm.nih.gov/databases/download/mesh.html
2. Baixe o "MeSH descriptor data" em formato XML (desc<ano>.xml, ex: desc2026.xml).
   Nao precisa de login nem licenca -- e dominio publico.
3. Rode este script apontando para o arquivo baixado.

USO
---
    python build_entity_categories_mesh.py --mesh-xml desc2026.xml --out entity_categories.py

    # incluindo sinonimos/termos de entrada (dicionario fica bem maior):
    python build_entity_categories_mesh.py --mesh-xml desc2026.xml --out entity_categories.py --all-synonyms

    # validar a logica do script sem precisar do arquivo real do MeSH:
    python build_entity_categories_mesh.py --self-test

COMO FUNCIONA (resumo da classificacao)
----------------------------------------
Cada descritor do MeSH tem um ou mais "Tree Numbers" que indicam onde ele
fica na hierarquia oficial. Usamos esses prefixos para mapear para as
categorias do projeto:

    Anatomy    -> A            (Anatomy)
    Symptom    -> C23.888      (Signs and Symptoms)
    Result     -> C23.149, C23.300, C23.550
                  (Morphological/Microscopic Findings; Pathological
                  Conditions, Anatomical; Pathologic Processes -- achados
                  de exame/patologia, em vez de doencas em si)
    Disease    -> C            (Diseases) -- exceto os 3 ramos de "Result"
                  e o de "Symptom" acima
    Exam       -> E01, E05     (Diagnosis; Investigative Techniques)
    Procedure  -> E02, E03, E04 (Therapeutics; Anesthesia/Analgesia;
                  Surgical Procedures, Operative)
    Medication -> D26          (Pharmaceutical Preparations), OU qualquer
                  descritor em D (Chemicals and Drugs) que tenha pelo menos
                  uma "Pharmacological Action" associada -- e assim que se
                  distingue "e um farmaco" de "e so um composto quimico"
                  dentro do MeSH.

Um mesmo descritor pode cair em mais de uma categoria (ele pode ter varios
Tree Numbers em ramos diferentes) -- isso e esperado e reflete a propria
estrutura do MeSH.

LIMITACOES CONHECIDAS (vale revisar manualmente depois)
---------------------------------------------------------
- O MeSH classifica "Biopsy" dentro de Surgical Procedures (E04), entao ele
  cai em "Procedure" aqui, mesmo que voces prefiram tratar biopsia como
  "Exam". Use MANUAL_OVERRIDES abaixo para casos assim.
- Termos genericos de tratamento como "chemotherapy" ficam em Therapeutics
  (E02, ou seja, "Procedure"), nao em "Medication" -- o MeSH separa "o
  tratamento" do "o farmaco em si".
- "Result" no MeSH cobre achados morfologicos/patologicos, mas biomarcadores
  quimicos (ex: antigenos) ficam classificados em D (Chemicals), nao em C23;
  eles vao cair em Medication (se tiverem Pharmacological Action) ou ficar
  de fora. Se precisarem desses termos em "Result", adicionem manualmente.
"""

import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict

# Termos especificos cuja categoria voces querem forcar, independente do
# que o MeSH diz (chave: termo em minusculas exatamente como aparece no
# MeSH; valor: categoria de destino). Adicione quantos quiserem.
MANUAL_OVERRIDES = {
    "biopsy": "Exam",
}


def matches_prefix(tree_number: str, prefix: str) -> bool:
    """True se tree_number for exatamente o prefixo ou um descendente dele
    (ex: prefix='C23.888' casa com 'C23.888' e com 'C23.888.512')."""
    return tree_number == prefix or tree_number.startswith(prefix + ".")


def classify_tree_numbers(tree_numbers, has_pharm_action):
    categories = set()
    for tree in tree_numbers:
        if matches_prefix(tree, "C23.888"):
            categories.add("Symptom")
        elif any(matches_prefix(tree, p) for p in ("C23.149", "C23.300", "C23.550")):
            categories.add("Result")
        elif tree.startswith("C"):
            categories.add("Disease")
        elif tree.startswith("A"):
            categories.add("Anatomy")
        elif any(matches_prefix(tree, p) for p in ("E01", "E05")):
            categories.add("Exam")
        elif any(matches_prefix(tree, p) for p in ("E02", "E03", "E04")):
            categories.add("Procedure")
        elif tree.startswith("D"):
            if has_pharm_action or matches_prefix(tree, "D26"):
                categories.add("Medication")
    return categories


def iter_descriptor_records(xml_path):
    """Faz streaming do XML (arquivo pode ter 100+ MB) e retorna, por
    descritor: (lista de tree numbers, tem_pharm_action, lista de termos)."""
    context = ET.iterparse(xml_path, events=("end",))
    for event, elem in context:
        if elem.tag != "DescriptorRecord":
            continue

        tree_numbers = [
            tn.text for tn in elem.findall("./TreeNumberList/TreeNumber")
            if tn.text
        ]

        has_pharm_action = elem.find("./PharmacologicalActionList") is not None

        terms = []
        name_el = elem.find("./DescriptorName/String")
        if name_el is not None and name_el.text:
            terms.append(name_el.text)

        yield tree_numbers, has_pharm_action, terms, elem

        elem.clear()


def iter_all_synonyms(elem):
    for term_el in elem.findall("./ConceptList/Concept/TermList/Term/String"):
        if term_el.text:
            yield term_el.text


def build_categories(xml_path, all_synonyms=False):
    categories = defaultdict(set)

    for tree_numbers, has_pharm_action, terms, elem in iter_descriptor_records(xml_path):
        if all_synonyms:
            terms = list(iter_all_synonyms(elem)) or terms

        matched = classify_tree_numbers(tree_numbers, has_pharm_action)
        if not matched:
            continue

        for term in terms:
            term_lower = term.strip().lower()
            override = MANUAL_OVERRIDES.get(term_lower)
            target_categories = {override} if override else matched
            for cat in target_categories:
                categories[cat].add(term_lower)

    return categories


def write_python_dict(categories, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("ENTITY_CATEGORIES: dict[str, set[str]] = {\n")
        for cat in sorted(categories):
            f.write(f'    "{cat}": {{\n')
            for term in sorted(categories[cat]):
                escaped = term.replace('"', '\\"')
                f.write(f'        "{escaped}",\n')
            f.write("    },\n")
        f.write("}\n")


# ---------------------------------------------------------------------------
# Self-test: valida a logica de classificacao sem precisar do arquivo real
# do MeSH (util pra conferir que o script esta correto antes de rodar com
# o desc<ano>.xml de verdade, que tem dezenas de milhares de descritores).
# ---------------------------------------------------------------------------

_SELF_TEST_XML = """<?xml version="1.0" encoding="UTF-8"?>
<DescriptorRecordSet>
  <DescriptorRecord DescriptorClass="1">
    <DescriptorUI>D000001</DescriptorUI>
    <DescriptorName><String>Stomach</String></DescriptorName>
    <TreeNumberList><TreeNumber>A03.556.875</TreeNumber></TreeNumberList>
    <ConceptList>
      <Concept PreferredConceptYN="Y">
        <TermList><Term><String>Stomach</String></Term></TermList>
      </Concept>
    </ConceptList>
  </DescriptorRecord>
  <DescriptorRecord DescriptorClass="1">
    <DescriptorUI>D000002</DescriptorUI>
    <DescriptorName><String>Fever</String></DescriptorName>
    <TreeNumberList><TreeNumber>C23.888.646.421</TreeNumber></TreeNumberList>
    <ConceptList>
      <Concept PreferredConceptYN="Y">
        <TermList><Term><String>Fever</String></Term><Term><String>Pyrexia</String></Term></TermList>
      </Concept>
    </ConceptList>
  </DescriptorRecord>
  <DescriptorRecord DescriptorClass="1">
    <DescriptorUI>D000003</DescriptorUI>
    <DescriptorName><String>Tuberculosis</String></DescriptorName>
    <TreeNumberList><TreeNumber>C01.252.410.040.552.846</TreeNumber></TreeNumberList>
    <ConceptList>
      <Concept PreferredConceptYN="Y">
        <TermList><Term><String>Tuberculosis</String></Term></TermList>
      </Concept>
    </ConceptList>
  </DescriptorRecord>
  <DescriptorRecord DescriptorClass="1">
    <DescriptorUI>D000004</DescriptorUI>
    <DescriptorName><String>Cysts</String></DescriptorName>
    <TreeNumberList><TreeNumber>C23.300.707.164</TreeNumber></TreeNumberList>
    <ConceptList>
      <Concept PreferredConceptYN="Y">
        <TermList><Term><String>Cyst</String></Term></TermList>
      </Concept>
    </ConceptList>
  </DescriptorRecord>
  <DescriptorRecord DescriptorClass="1">
    <DescriptorUI>D000005</DescriptorUI>
    <DescriptorName><String>Tomography, X-Ray Computed</String></DescriptorName>
    <TreeNumberList><TreeNumber>E01.370.350.825</TreeNumber></TreeNumberList>
    <ConceptList>
      <Concept PreferredConceptYN="Y">
        <TermList><Term><String>Computed Tomography</String></Term></TermList>
      </Concept>
    </ConceptList>
  </DescriptorRecord>
  <DescriptorRecord DescriptorClass="1">
    <DescriptorUI>D000006</DescriptorUI>
    <DescriptorName><String>Biopsy</String></DescriptorName>
    <TreeNumberList><TreeNumber>E04.074</TreeNumber></TreeNumberList>
    <ConceptList>
      <Concept PreferredConceptYN="Y">
        <TermList><Term><String>Biopsy</String></Term></TermList>
      </Concept>
    </ConceptList>
  </DescriptorRecord>
  <DescriptorRecord DescriptorClass="1">
    <DescriptorUI>D000007</DescriptorUI>
    <DescriptorName><String>Pancreatectomy</String></DescriptorName>
    <TreeNumberList><TreeNumber>E04.210.696</TreeNumber></TreeNumberList>
    <ConceptList>
      <Concept PreferredConceptYN="Y">
        <TermList><Term><String>Pancreatectomy</String></Term></TermList>
      </Concept>
    </ConceptList>
  </DescriptorRecord>
  <DescriptorRecord DescriptorClass="1">
    <DescriptorUI>D000008</DescriptorUI>
    <DescriptorName><String>Aspirin</String></DescriptorName>
    <TreeNumberList><TreeNumber>D02.455.426.559.389.657.410.595.176</TreeNumber></TreeNumberList>
    <ConceptList>
      <Concept PreferredConceptYN="Y">
        <TermList><Term><String>Aspirin</String></Term></TermList>
      </Concept>
    </ConceptList>
    <PharmacologicalActionList>
      <PharmacologicalAction>
        <DescriptorReferredTo><DescriptorUI>D000900</DescriptorUI></DescriptorReferredTo>
      </PharmacologicalAction>
    </PharmacologicalActionList>
  </DescriptorRecord>
  <DescriptorRecord DescriptorClass="1">
    <DescriptorUI>D000009</DescriptorUI>
    <DescriptorName><String>Ethanol</String></DescriptorName>
    <TreeNumberList><TreeNumber>D02.033.375</TreeNumber></TreeNumberList>
    <ConceptList>
      <Concept PreferredConceptYN="Y">
        <TermList><Term><String>Ethanol</String></Term></TermList>
      </Concept>
    </ConceptList>
  </DescriptorRecord>
</DescriptorRecordSet>
"""


def run_self_test():
    import io

    path = io.BytesIO(_SELF_TEST_XML.encode("utf-8"))
    categories = build_categories(path, all_synonyms=True)

    checks = [
        ("Anatomy", "stomach", True),
        ("Symptom", "fever", True),
        ("Symptom", "pyrexia", True),
        ("Disease", "tuberculosis", True),
        ("Result", "cyst", True),
        ("Exam", "computed tomography", True),
        ("Exam", "biopsy", True),        # override em ação
        ("Procedure", "biopsy", False),  # não deve sobrar na categoria original
        ("Procedure", "pancreatectomy", True),
        ("Medication", "aspirin", True),
        ("Medication", "ethanol", False),  # sem Pharmacological Action -> fica de fora
    ]

    print("Categorias geradas no self-test:")
    for cat, terms in sorted(categories.items()):
        print(f"  {cat}: {sorted(terms)}")

    failures = []
    for cat, term, should_be_present in checks:
        present = term in categories.get(cat, set())
        ok = present == should_be_present
        status = "OK" if ok else "FALHOU"
        print(f"[{status}] '{term}' em '{cat}': esperado={should_be_present}, obtido={present}")
        if not ok:
            failures.append((cat, term, should_be_present, present))

    if failures:
        raise SystemExit(f"Self-test falhou em {len(failures)} verificacao(oes).")
    print("\nSelf-test passou em todas as verificacoes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mesh-xml", help="Caminho para o desc<ano>.xml baixado da NLM")
    parser.add_argument("--out", default="entity_categories.py", help="Arquivo Python de saida")
    parser.add_argument("--all-synonyms", action="store_true",
                         help="Inclui todos os termos de entrada (sinonimos) de cada descritor, nao so o termo preferencial")
    parser.add_argument("--self-test", action="store_true",
                         help="Roda uma verificacao interna com dados sinteticos, sem precisar do arquivo real do MeSH")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()
    elif args.mesh_xml:
        cats = build_categories(args.mesh_xml, all_synonyms=args.all_synonyms)
        write_python_dict(cats, args.out)
        print("Termos por categoria:")
        for cat, terms in sorted(cats.items()):
            print(f"  {cat}: {len(terms)}")
    else:
        parser.error("Use --mesh-xml <arquivo> ou --self-test")