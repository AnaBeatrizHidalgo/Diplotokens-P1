"""
    Este script deve conter a função de extrair quantidades do texto (ex: "56 ml") e a de atrelá-las às devidas arestas
"""

class Quantity:
    def __init__(self, value: str, position: int):
        self.value = value
        self.pos = position

    def __repr__(self):
        return f'({self.pos}: "{self.value}")'

def extract_quantities(text: str) -> list[Quantity]:
    """
        Recebe um texto e retorna uma lista de objetos Quantity
        Ex: [Quantity("56 ml", 21), Quantity("500 mg", 67)]
    """
    pass

def associate_quantities_to_edges(quantities: list[Quantity], grafo):
    pass