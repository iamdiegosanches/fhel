# FHEL — Framework Hierárquico de Explicabilidade Linguística

Este repositório contém uma implementação reprodutível do FHEL (Representation-mapping Explainer)
projetado para explicar predições de modelos BERT (BERTimbau) no contexto de detecção de Fake News em português.

## Metodologia

O FHEL resolve o "Paradoxo do Modelo Substituto" mantendo o modelo BERT original e mapeando as explicações geradas pelo SHAP diretamente para estruturas linguísticas anotadas pelo spaCy. O fluxo é:

1. SHAP avalia o modelo BERT no texto bruto, gerando pesos para cada *subtoken* (WordPiece).
2. O spaCy (`pt_core_news_lg`) anota o texto produzindo tokens com POS e NER.
3. O FHEL alinha subtokens (usando offsets do tokenizer rápido) com tokens do spaCy, agrega os valores absolutos do SHAP correspondentes a cada token e distribui esse peso entre três níveis linguísticos:
   - Morfossintático (POS)
   - Semântico (NER)
   - Retórico (heurísticas: pontuação, interjeições, all-caps, sinais de ênfase)

Essa agregação preserva fielmente os sinais do SHAP sobre o modelo BERT original e oferece uma explicação no nível linguístico superior, resolvendo o paradoxo causado pela explicação de modelos substitutos.

## Métricas

- AOPC (Area Over the Perturbation Curve): medimos quão fiel é a explicação perturbando (mascarando)
  tokens sugeridos como importantes pelo FHEL e observando a queda média na probabilidade do modelo
  para a classe originalmente prevista. Implementação aproximada: máscaras iterativas com `tokenizer.mask_token`.
- Sparsity: medimos a redução de complexidade da explicação pelo índice de Gini. Comparamos a distribuição de massa de SHAP no nível de subtokens versus a distribuição condensada nos 3 níveis linguísticos. Uma redução no índice de Gini indica explicações mais concentradas (mais esparsas) após a agregação.

## Instalação

Recomenda-se criar um ambiente virtual Python 3.9+ e instalar dependências:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download pt_core_news_lg
```

## Uso

Exemplo de execução do pipeline completo:

```bash
python main.py --text "Exemplo de notícia em português a ser explicada pelo FHEL."
```

Parâmetros:

- `--text`: texto a explicar.
- `--model`: nome do modelo transformers (opcional). Padrão: `neuralmind/bert-base-portuguese-cased`.

## Estrutura dos módulos

- `fhel/explainer.py`: executa SHAP sobre o modelo BERT e retorna subtokens, offsets e valores SHAP.
- `fhel/annotator.py`: conecta ao spaCy e produz anotações token-level (POS, NER, spans).
- `fhel/fhel_core.py`: faz o alinhamento subtokens->tokens e agrega as contribuições nos níveis Morfossintático, Semântico e Retórico.
- `fhel/metrics.py`: implementa AOPC e Sparsity.
- `fhel/generator.py`: cria explicações em linguagem natural.
- `main.py`: pipeline orquestrador e CLI.

## Notas sobre reprodutibilidade

- Use tokenizers rápidos (FastTokenizer) para garantir offsets corretos.
- SHAP pode ser custoso; reduza `nsamples` no `explain_shap` para testes rápidos.
- Documente e fixe as versões de `transformers`, `spacy` e `shap` usadas nas suas experiências.
