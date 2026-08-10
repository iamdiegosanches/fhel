import pandas as pd
import spacy
from tqdm import tqdm
import os

# 1. Carrega o modelo de PLN do spaCy
print("Carregando modelo pt_core_news_lg...")
nlp = spacy.load("pt_core_news_lg")

# 2. Carrega a amostra processada anteriormente (ou o dataset completo)
input_file = "data/fake_br_consolidated.csv"
df = pd.read_csv(input_file).head(40) # Usando a amostra de 40 para o MVP
textos = df['text'].tolist()

# Estrutura para armazenar as features de cada texto
features_list = []

print("Extraindo atributos linguísticos em paralelo...")
# nlp.pipe processa os documentos em lotes, paralelizando a carga na CPU
for doc in tqdm(nlp.pipe(textos, batch_size=10, n_process=-1), total=len(textos)):
    
    # --- NÍVEL RETÓRICO ---
    tamanho_texto = len(doc)
    num_sentencas = len(list(doc.sents))
    comprimento_medio_sentenca = tamanho_texto / num_sentencas if num_sentencas > 0 else 0
    
    # Densidade informacional (tipicamente a razão de palavras únicas / total de palavras)
    tokens_unicos = set([token.text.lower() for token in doc if token.is_alpha])
    densidade_info = len(tokens_unicos) / tamanho_texto if tamanho_texto > 0 else 0

    # --- NÍVEL MORFOSSINTÁTICO ---
    # Contagem de classes gramaticais (POS Tags)
    pos_counts = {"VERB": 0, "ADJ": 0, "PROPN": 0, "NOUN": 0}
    for token in doc:
        if token.pos_ in pos_counts:
            pos_counts[token.pos_] += 1
            
    densidade_verbal = pos_counts["VERB"] / tamanho_texto if tamanho_texto > 0 else 0
    razao_proprio_comum = pos_counts["PROPN"] / pos_counts["NOUN"] if pos_counts["NOUN"] > 0 else pos_counts["PROPN"]

    # --- NÍVEL SEMÂNTICO ---
    # Reconhecimento de Entidades Nomeadas (NER)
    ner_counts = {"PER": 0, "ORG": 0, "LOC": 0, "MISC": 0}
    for ent in doc.ents:
        if ent.label_ in ner_counts:
            ner_counts[ent.label_] += 1
            
    # Montando o vetor do documento
    doc_features = {
        "text_snippet": doc.text[:50] + "...",
        # Retórica
        "ret_tamanho_texto": tamanho_texto,
        "ret_comprimento_medio_sentenca": comprimento_medio_sentenca,
        "ret_densidade_info": densidade_info,
        # Morfossintaxe
        "mor_densidade_verbal": densidade_verbal,
        "mor_proprio_vs_comum": razao_proprio_comum,
        # Semântica
        "sem_entidades_Pessoa": ner_counts["PER"],
        "sem_entidades_Organizacao": ner_counts["ORG"]
    }
    features_list.append(doc_features)

# 3. Salva a matriz de características
df_features = pd.DataFrame(features_list)
df_final = pd.concat([df, df_features], axis=1)

os.makedirs("outputs", exist_ok=True)
output_path = "outputs/fhel_extracted_features.csv"
df_final.to_csv(output_path, index=False)
print(f"\nExtração concluída! Matriz hierárquica salva em {output_path}")