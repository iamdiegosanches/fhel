"""Entrypoint para execução do pipeline FHEL.

Exemplo:
    python main.py --mode batch --n_per_class 50 --n_examples 20 --nsamples 20
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import argparse
import json
import datetime

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from fhel.explainer import explain_shap
from fhel.annotator import Annotator
from fhel.fhel_core import FHELCore
from fhel.metrics import compute_aopc, compute_sparsity
from fhel.generator import generate_explanation


DATA_PATH = Path("data/fake_br_consolidated.csv")
OUTPUT_DIR = Path("outputs")


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found in {path}")
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")
    return df


def sample_balanced(df: pd.DataFrame, n_per_class: int = 50, seed: int = 42) -> pd.DataFrame:
    groups = df.groupby("label")
    parts: List[pd.DataFrame] = []
    for label, group in groups:
        parts.append(group.sample(n=min(n_per_class, len(group)), random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def ensure_outputs_dir(path: Path = OUTPUT_DIR) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_pipeline(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        device=device,
    )
    return model, tokenizer, pipe


def process_example(
    text: str,
    model: Any,
    tokenizer: Any,
    pipe: Any,
    ann: Annotator,
    core: FHELCore,
    nsamples: int = 20,
) -> Dict[str, Any]:
    spacy_tokens = ann.annotate(text)
    shap_out = explain_shap(text=text, model=model, tokenizer=tokenizer, nsamples=nsamples)
    agg = core.aggregate(
        tokens_offsets=shap_out.get("offsets", []),
        subtokens=shap_out.get("tokens", []),
        subtoken_shap=shap_out.get("shap_values", []),
        spacy_tokens=spacy_tokens,
    )
    winner = max(agg["levels_percent"].items(), key=lambda x: x[1])[0]
    aopc = compute_aopc(
        original_text=text,
        model_pipeline=pipe,
        fhel_output={**agg, "predicted_label": shap_out.get("predicted_label")},
        target_level=winner,
        tokenizer=tokenizer,
    )
    sparsity = compute_sparsity(
        original_shap=shap_out.get("shap_values", []),
        aggregated_levels=list(agg.get("level_sums", {}).values()),
    )
    explanation = generate_explanation(agg)

    return {
        "text": text,
        "shap": shap_out,
        "agg": agg,
        "aopc": aopc,
        "sparsity": sparsity,
        "explanation": explanation,
    }


def save_results(results: List[Dict[str, Any]], output_dir: Path) -> None:
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"fhel_results_{timestamp}.jsonl"
    csv_path = output_dir / f"fhel_summary_{timestamp}.csv"

    with json_path.open("w", encoding="utf-8") as fh:
        for result in results:
            fh.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")

    summary_rows: List[Dict[str, Any]] = []
    for result in results:
        summary_rows.append(
            {
                "text": result["text"],
                "winner_level": max(result["agg"]["levels_percent"].items(), key=lambda x: x[1])[0],
                "winner_pct": max(result["agg"]["levels_percent"].items(), key=lambda x: x[1])[1],
                "aopc": result["aopc"].get("aopc"),
                "sparsity_reduction": result["sparsity"].get("reduction_ratio"),
                "compression_ratio": result["sparsity"].get("compression_ratio"),
            }
        )

    pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
    print(f"Saved detailed results to: {json_path}")
    print(f"Saved summary metrics to: {csv_path}")


def run_batch(
    df: pd.DataFrame,
    model_name: str = "neuralmind/bert-base-portuguese-cased",
    nsamples: int = 20,
    n_examples: int = 100,
    n_per_class: int = 50,
) -> None:
    ensure_outputs_dir(OUTPUT_DIR)
    model, tokenizer, pipe = build_pipeline(model_name)
    ann = Annotator()
    core = FHELCore()

    sampled_df = sample_balanced(df, n_per_class=n_per_class)
    results: List[Dict[str, Any]] = []
    error_path = OUTPUT_DIR / "error_examples.jsonl"
    with error_path.open("w", encoding="utf-8") as errf:
        for idx, row in sampled_df.head(n_examples).iterrows():
            text = str(row["text"])
            print(f"Processing {idx+1}/{min(n_examples, len(sampled_df))}")
            try:
                result = process_example(text=text, model=model, tokenizer=tokenizer, pipe=pipe, ann=ann, core=core, nsamples=nsamples)
                result["label"] = int(row["label"])
                results.append(result)
            except Exception as exc:
                errf.write(json.dumps({"index": int(idx), "text": text, "error": str(exc)}, ensure_ascii=False) + "\n")
                print(f"Error processing example {idx}: {exc}")

    if results:
        save_results(results, OUTPUT_DIR)
        for result in results[:3]:
            print("\n--- EXPLICAÇÃO ---\n")
            print(result["explanation"])
            print("\n--- METRICS ---\n")
            print(f"AOPC: {result['aopc'].get('aopc')}")
            print(f"Sparsity reduction ratio: {result['sparsity'].get('reduction_ratio')}")
            print(f"Compression ratio: {result['sparsity'].get('compression_ratio')}")
    else:
        print("Nenhum resultado foi gerado. Verifique o arquivo de erros e os dados.")


def cli() -> None:
    parser = argparse.ArgumentParser(description="FHEL pipeline runner")
    parser.add_argument("--text", type=str, default=None, help="Texto para explicar (requerido no modo single)")
    parser.add_argument("--model", type=str, default="neuralmind/bert-base-portuguese-cased", help="Nome do modelo BERTimbau (transformers)")
    parser.add_argument("--nsamples", type=int, default=20, help="Número de amostras SHAP (reduza para testes rápidos)")
    parser.add_argument("--mode", type=str, choices=["single", "batch"], default="single", help="Modo de execução: single para um texto, batch para dataset")
    parser.add_argument("--n_per_class", type=int, default=50, help="Número por classe ao amostrar o dataset")
    parser.add_argument("--n_examples", type=int, default=100, help="Número máximo de exemplos a processar no modo batch")
    args = parser.parse_args()

    if args.mode == "single":
        if not args.text:
            parser.error("--text is required in single mode")
        model, tokenizer, pipe = build_pipeline(args.model)
        ann = Annotator()
        core = FHELCore()
        result = process_example(text=args.text, model=model, tokenizer=tokenizer, pipe=pipe, ann=ann, core=core, nsamples=args.nsamples)
        print("\n--- RESULTADO FHEL ---\n")
        print(result["explanation"])
        print("\nMétricas:")
        print(f"- AOPC: {result['aopc'].get('aopc'):.4f}")
        print(f"- Sparsity reduction ratio: {result['sparsity'].get('reduction_ratio'):.4f}")
        print(f"- Compression ratio: {result['sparsity'].get('compression_ratio'):.4f}")
    else:
        df = load_dataset(DATA_PATH)
        run_batch(df, model_name=args.model, nsamples=args.nsamples, n_examples=args.n_examples, n_per_class=args.n_per_class)


if __name__ == "__main__":
    cli()
