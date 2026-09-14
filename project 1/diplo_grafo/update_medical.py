import json
from diplo_grafo.config import EXTERNAL_DATA_DIR
from diplo_grafo.clinical_vocabulary import all_terms

def update_lexicon_file():
    # 1. Coleta todos os termos atualizados do clinical_vocabulary.py
    lexicon_set = all_terms()
    
    # 2. Converte para lista para poder salvar em JSON e ordena para ficar organizado
    lexicon_list = sorted(list(lexicon_set))
    
    # 3. Define o caminho exato do arquivo
    file_path = EXTERNAL_DATA_DIR / "medical_lexicon.json"
    
    # Garante que o diretório existe
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 4. Salva (sobrescreve) o arquivo
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(lexicon_list, f, indent=4, ensure_ascii=False)
        
    print(f"Sucesso! Arquivo '{file_path.name}' atualizado com {len(lexicon_list)} termos.")

if __name__ == "__main__":
    update_lexicon_file()