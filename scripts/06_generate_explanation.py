import pandas as pd

def gerar_explicacao_fhel(row):
    """
    Transforma os percentuais linguísticos agregados em uma explicação textual fluida.
    """
    classe_str = "Desinformação (Fake)" if row['label'] == 1 else "Notícia Legítima (True)"
    sem_pct = row['% Semântico (Entidades)']
    morfo_pct = row['% Morfossintático']
    
    # Template Base
    explicacao = f"O texto foi classificado como {classe_str}. "
    
    # Lógica de prioridade de explicação
    if morfo_pct > sem_pct:
        explicacao += (
            f"A interpretabilidade do FHEL revela que a decisão da rede neural foi impulsionada "
            f"principalmente pelo nível Morfossintático ({morfo_pct}%), indicando uma forte dependência "
            f"do modelo em verbos, adjetivos e na estrutura de ação para inferir o tom da mensagem. "
        )
        if sem_pct > 5.0:
            explicacao += f"Secundariamente, o nível Semântico contribuiu com {sem_pct}% através de menções diretas a entidades."
            
    else:
        explicacao += (
            f"A interpretabilidade do FHEL revela que a decisão da rede neural foi impulsionada "
            f"principalmente pelo nível Semântico ({sem_pct}%), indicando que o foco central da predição "
            f"recaiu sobre as entidades específicas (Pessoas, Organizações ou Locais) citadas no texto. "
        )
        if morfo_pct > 5.0:
            explicacao += f"O nível Morfossintático atuou como fator de suporte estilístico, contribuindo com {morfo_pct}% da decisão."
            
    return explicacao

if __name__ == "__main__":
    print("Carregando resultados da agregação FHEL...")
    try:
        df_resultados = pd.read_csv("outputs/fhel_aggregation_test.csv")
        
        print("\n=== EXPLICAÇÕES GERADAS (HUMAN-READABLE) ===\n")
        
        for index, row in df_resultados.iterrows():
            texto_curto = row['texto']
            texto_explicativo = gerar_explicacao_fhel(row)
            
            print(f"DOCUMENTO {index}: {texto_curto}")
            print(f"EXPLICAÇÃO: {texto_explicativo}")
            print("-" * 80)
            
    except FileNotFoundError:
        print("Erro: O arquivo 'fhel_aggregation_test.csv' não foi encontrado. Execute o script 05 primeiro.")