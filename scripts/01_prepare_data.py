import os
import pandas as pd
from tqdm import tqdm

def load_texts_from_folder(folder_path, label):
    texts = []
    # Itera sobre os arquivos TXT
    for filename in tqdm(os.listdir(folder_path), desc=f"Loading {label}"):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                content = f.read().strip()
                texts.append({'text': content, 'label': label})
    return texts

if __name__ == "__main__":
    base_path = "data/fake_br/size_normalized_texts"
    
    # Carrega notícias falsas (label 1) e verdadeiras (label 0)
    fake_data = load_texts_from_folder(os.path.join(base_path, "fake"), 1)
    true_data = load_texts_from_folder(os.path.join(base_path, "true"), 0)
    
    # Consolida, embaralha e salva
    df = pd.DataFrame(fake_data + true_data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    output_path = "data/fake_br_consolidated.csv"
    df.to_csv(output_path, index=False)
    print(f"Dataset salvo em {output_path} com {len(df)} registros.")