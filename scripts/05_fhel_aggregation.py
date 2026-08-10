import pandas as pd
import shap
import torch
from transformers import pipeline
import spacy
import numpy as np
import os

# 1. Inicialização de Modelos
print("Carregando BERTimbau e spaCy...")
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("text-classification", model="neuralmind/bert-base-portuguese-cased", device=device, truncation=True, max_length=512)
nlp = spacy.load("pt_core_news_lg")
explainer = shap.Explainer(classifier)

# 2. Carrega uma pequena subamostra
df = pd.read_csv("outputs/fhel_extracted_features.csv")
textos_teste = df['text'].iloc[[0, 1, 20, 21]].tolist() 
labels_teste = df['label'].iloc[[0, 1, 20, 21]].tolist()

print("\nExtraindo tensores SHAP...")
shap_values = explainer(textos_teste)

resultados_fhel = []

# 3. O Motor de Agregação Blindado
print("Processando Agregação Hierárquica...")
for i, texto in enumerate(textos_teste):
    tokens_shap = shap_values[i].data
    valores_shap = shap_values[i].values
    
    # --- CORREÇÃO DE ESTRUTURA DO SHAP ---
    # Capturamos o impacto máximo daquela palavra, independentemente de qual classe o SHAP focou
    if len(valores_shap.shape) > 1:
        importancia_matriz = np.abs(valores_shap).max(axis=1)
    else:
        importancia_matriz = np.abs(valores_shap)
        
    # Converte para float puro (evita comportamento NaN do numpy)
    importancias = [float(v) for v in importancia_matriz]
    importancia_total = sum(importancias)
    
    # Trava de segurança contra divisão por zero
    if importancia_total == 0.0:
        importancia_total = 1e-9
        
    impacto_morfossintatico = 0.0
    impacto_semantico = 0.0
    impacto_lexical_descartado = 0.0 
    
    for token_str, valor in zip(tokens_shap, importancias):
        clean_token = token_str.replace("##", "").strip()
        
        if not clean_token:
            impacto_lexical_descartado += valor
            continue
            
        doc_spacy = nlp(clean_token)
        
        if len(doc_spacy) > 0:
            token_analisado = doc_spacy[0]
            
            if token_analisado.ent_type_ in ['PER', 'ORG', 'LOC', 'MISC']:
                impacto_semantico += valor
            elif token_analisado.pos_ in ['VERB', 'ADJ', 'PROPN']:
                impacto_morfossintatico += valor
            else:
                impacto_lexical_descartado += valor
        else:
            impacto_lexical_descartado += valor
                
    # Cálculo das porcentagens
    percentual_morfo = (impacto_morfossintatico / importancia_total) * 100
    percentual_seman = (impacto_semantico / importancia_total) * 100
    percentual_outros = (impacto_lexical_descartado / importancia_total) * 100
    
    resultados_fhel.append({
        "label": labels_teste[i],
        "texto": texto[:45] + "...",
        "% Semântico (Entidades)": round(percentual_seman, 2),
        "% Morfossintático": round(percentual_morfo, 2),
        "% Resto (Outros/Pontuação)": round(percentual_outros, 2),
    })

# 4. Exibe os resultados
df_resultados = pd.DataFrame(resultados_fhel)
print("\n--- RESULTADOS DA AGREGAÇÃO FHEL ---")
print(df_resultados.to_string())

os.makedirs("outputs", exist_ok=True)
df_resultados.to_csv("outputs/fhel_aggregation_test.csv", index=False)