import torch
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

def setup_pipeline():
    # Verifica disponibilidade de aceleração de hardware
    device = 0 if torch.cuda.is_available() else -1
    print(f"Executando no dispositivo: {'GPU' if device == 0 else 'CPU'}")
    
    # Utilizando um modelo base BERTimbau pré-treinado 
    # (Nota: em um cenário real, você pode apontar para um checkpoint fine-tunado no Fake.br)
    model_name = "neuralmind/bert-base-portuguese-cased" 
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)
    return classifier

if __name__ == "__main__":
    classifier = setup_pipeline()
    
    # Teste rápido
    texto_teste = "O governo aprovou hoje uma nova lei que reduz impostos."
    resultado = classifier(texto_teste)
    print(f"Resultado do teste: {resultado}")