from pathlib import Path
import json
import re
import pandas as pd


def _normalizar_coluna(nome):
    return re.sub(r"[^a-z0-9]", "", str(nome).lower())


def _como_token(valor):
    """Converte uma nomenclatura médica em um único token."""
    valor = str(valor).strip()

    if not valor or valor.lower() == "nan":
        return ""

    # Remove pontuação e troca espaços por underscore.
    valor = re.sub(r"[^\w\s-]", "", valor, flags=re.UNICODE)
    valor = re.sub(r"[\s-]+", "_", valor)

    return valor.lower().strip("_")


def carregar_dicionario_medico(data_dir):
    """
    Carrega CSVs/JSONs de data/external.

    O dataset deve possuir uma coluna com o termo original e outra
    com a nomenclatura preferida/canônica.
    """
    data_dir = Path(data_dir)
    pares = []

    arquivos = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.json"))

    for arquivo in arquivos:
        try:
            if arquivo.suffix == ".csv":
                tabela = pd.read_csv(arquivo)
            else:
                with open(arquivo, encoding="utf-8") as f:
                    dados = json.load(f)
                tabela = pd.DataFrame(dados)

            colunas = {
                _normalizar_coluna(c): c
                for c in tabela.columns
            }

            coluna_original = next(
                (
                    colunas[c] for c in (
                        "term", "termo", "synonym", "sinonimo",
                        "entry", "original", "label"
                    ) if c in colunas
                ),
                None
            )

            coluna_canonica = next(
                (
                    colunas[c] for c in (
                        "preferredterm", "preferredname", "canonical",
                        "canonico", "nomenclature", "nome",
                        "name", "normalized"
                    ) if c in colunas
                ),
                None
            )

            if coluna_original and coluna_canonica:
                for _, linha in tabela.iterrows():
                    original = str(linha[coluna_original]).strip()
                    canonica = _como_token(linha[coluna_canonica])

                    if (
                        original
                        and original.lower() != "nan"
                        and canonica
                    ):
                        pares.append((original, canonica))

        except Exception as erro:
            print(f"Arquivo ignorado: {arquivo.name} ({erro})")

    # Termos maiores primeiro, evitando substituir parcialmente expressões.
    return sorted(
        dict(pares).items(),
        key=lambda par: len(par[0]),
        reverse=True
    )


def substituir_termos_medicos(texto, data_dir=None, dicionario=None):
    """
    Substitui termos médicos por sua nomenclatura canônica.

    Exemplo:
        "O paciente teve enfarte agudo do miocárdio"
        -> "O paciente teve infarto agudo do miocárdio"
    """
    if not texto:
        return texto

    if dicionario is None:
        if data_dir is None:
            data_dir = Path(__file__).resolve().parents[1] / "data" / "external"

        dicionario = carregar_dicionario_medico(data_dir)

    resultado = texto

    for termo, nomenclatura in dicionario:
        padrao = re.compile(
            rf"(?<!\w){re.escape(termo)}(?!\w)",
            flags=re.IGNORECASE
        )

        resultado = padrao.sub(nomenclatura, resultado)

    return resultado