import pandas as pd
import shap
import torch
from transformers import pipeline
import os

# 1. Configuração do modelo e dispositivo
device = 0 if torch.cuda.is_available() else -1
print("Carregando o modelo BERTimbau...")
classifier = pipeline(
    "text-classification", 
    model="neuralmind/bert-base-portuguese-cased", 
    device=device,
    truncation=True,
    max_length=512,
    padding=True
)

# 2. Carregamento e Amostragem Estratificada
print("Carregando os dados consolidados...")
df = pd.read_csv("data/fake_br_consolidated.csv")

# Seleciona 20 notícias falsas (label 1) e 20 verdadeiras (label 0)
df_fake = df[df['label'] == 1].head(20)
df_true = df[df['label'] == 0].head(20)

# Une e embaralha a amostra
df_amostra = pd.concat([df_fake, df_true]).sample(frac=1, random_state=42).reset_index(drop=True)
textos = df_amostra['text'].tolist()
labels_reais = df_amostra['label'].tolist()

# 3. Configuração do SHAP Explainer
print(f"Iniciando o cálculo SHAP para {len(textos)} documentos. Isso levará alguns minutos...")
explainer = shap.Explainer(classifier)
shap_values = explainer(textos)

# 4. Geração do Relatório Interativo
# O método shap.plots.text aceita a lista inteira e cria uma interface de navegação
print("Gerando o relatório visual...")
html_content = shap.plots.text(shap_values, display=False)

# Garante que a pasta existe
os.makedirs("outputs", exist_ok=True)

output_file = "outputs/shap_baseline_completo.html"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Sucesso! Abra o arquivo {output_file} no seu navegador.")
print("Você poderá usar a barra de rolagem para alternar entre as 40 notícias analisadas.")