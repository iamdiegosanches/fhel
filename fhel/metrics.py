from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _predict_prob(texts: Sequence[str], model_pipeline: Any, target_label: Optional[str] = None) -> List[float]:
    try:
        outs = model_pipeline(list(texts), truncation=True, max_length=512)
    except TypeError:
        outs = model_pipeline(list(texts))

    probs: List[float] = []
    for out in outs:
        score = 0.0
        if isinstance(out, (list, tuple)):
            if target_label is not None:
                for entry in out:
                    if isinstance(entry, dict) and entry.get("label") == target_label:
                        score = float(entry.get("score", 0.0))
                        break
            if score == 0.0:
                for entry in out:
                    if isinstance(entry, dict) and "score" in entry:
                        score = max(score, float(entry.get("score", 0.0)))
        elif isinstance(out, dict):
            if "label" in out and "score" in out:
                if target_label is not None and out.get("label") == target_label:
                    score = float(out.get("score", 0.0))
                elif target_label is None:
                    score = float(out.get("score", 0.0))
                else:
                    score = 0.0
            elif target_label is not None and target_label in out:
                score = float(out.get(target_label, 0.0))
            else:
                numeric_values = []
                for value in out.values():
                    try:
                        numeric_values.append(float(value))
                    except Exception:
                        continue
                score = float(max(numeric_values, default=0.0))
        else:
            try:
                score = float(out)
            except Exception:
                score = 0.0
        probs.append(score)
    return probs


def compute_aopc(
    original_text: str,
    model_pipeline: Any,
    fhel_output: Dict[str, Any],
    target_level: str,
    tokenizer: Any,
    steps: int = 5,
) -> Dict[str, Any]:
    """Compute AOPC (Area Over the Perturbation Curve) for a chosen linguistic level.

    Methodology:
    1. Identify tokens contributing to the specified FHEL linguistic level.
    2. Rank those tokens by contribution magnitude.
    3. Iteratively mask the top-k tokens in the original text using the
       tokenizer's mask token.
    4. Recompute the model probability for the original predicted class.
    5. Record the drop in probability after each masking step.
    6. Approximate AOPC as the mean drop across the perturbation curve.

    The implementation uses a text-level perturbation that is consistent with
    BERT-family mask semantics, while maintaining an interpretable, reproducible
    evaluation procedure for linguistic-level explanations.

    Returns:
        Dict with keys: 'aopc', 'drops', 'original_prob', 'target_label'.
    """
    per_token = fhel_output.get("per_token", [])
    candidates = [p for p in per_token if p.get("contributions", {}).get(target_level, 0.0) > 0.0]
    candidates = sorted(candidates, key=lambda x: x["contributions"][target_level], reverse=True)

    if not candidates:
        return {"aopc": 0.0, "drops": [], "original_prob": 0.0, "target_label": None}

    target_label = fhel_output.get("predicted_label")
    orig_prob = _predict_prob([original_text], model_pipeline, target_label=target_label)[0]
    drops: List[float] = []

    for k in range(1, min(len(candidates), steps) + 1):
        toks_to_mask = candidates[:k]
        mask_token = tokenizer.mask_token if tokenizer.mask_token is not None else "[MASK]"
        pieces = []
        cursor = 0
        for token in toks_to_mask:
            start, end = int(token["start"]), int(token["end"])
            if cursor < start:
                pieces.append(original_text[cursor:start])
            pieces.append(mask_token)
            cursor = end
        pieces.append(original_text[cursor:])
        masked_text = "".join(pieces)
        prob = _predict_prob([masked_text], model_pipeline, target_label=target_label)[0]
        drops.append(orig_prob - prob)

    aopc = float(np.mean(drops)) if drops else 0.0
    return {
        "aopc": aopc,
        "drops": [float(d) for d in drops],
        "original_prob": float(orig_prob),
        "target_label": target_label,
    }


def _gini(values: List[float]) -> float:
    arr = np.array(values, dtype=float)
    if arr.size == 0 or np.all(arr == 0):
        return 0.0
    arr = np.abs(arr) + 1e-12
    arr = np.sort(arr)
    n = arr.size
    index = np.arange(1, n + 1)
    return float(((np.sum((2 * index - n - 1) * arr)) / (n * np.sum(arr))))


def compute_sparsity(original_shap: List[float], aggregated_levels: List[float]) -> Dict[str, float]:
    """Compute sparsity reduction comparing raw SHAP features to aggregated FHEL levels."""
    original_abs = [abs(float(x)) for x in original_shap]
    aggregated_abs = [abs(float(x)) for x in aggregated_levels]
    original_gini = _gini(original_abs)
    aggregated_gini = _gini(aggregated_abs)
    reduction = float(original_gini - aggregated_gini) if original_gini > 0 else 0.0
    reduction_ratio = float(reduction / original_gini) if original_gini > 0 else 0.0
    compression_ratio = float(len(original_abs)) / float(len(aggregated_abs)) if aggregated_abs and original_abs else 0.0
    return {
        "original_gini": original_gini,
        "aggregated_gini": aggregated_gini,
        "reduction": reduction,
        "reduction_ratio": reduction_ratio,
        "compression_ratio": compression_ratio,
    }
