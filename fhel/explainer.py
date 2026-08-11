from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import shap
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def explain_shap(
    text: str,
    model: Any = None,
    tokenizer: Optional[Any] = None,
    model_name: Optional[str] = None,
    nsamples: int = 100,
) -> Dict[str, Any]:
    """Run SHAP on a text input using a transformers pipeline.

    Args:
        text: Input document (Portuguese).
        model: A transformers model instance (optional).
        tokenizer: A transformers tokenizer (recommended fast tokenizer).
        model_name: If `model`/`tokenizer` are not provided, a model name like
            `neuralmind/bert-base-portuguese-cased` can be passed and will be
            loaded automatically.
        nsamples: Number of SHAP evaluation samples (affects runtime).

    Returns:
        Dict with keys: `tokens` (subtokens), `offsets` (char spans), `base_values`,
        `shap_values` (per-subtoken absolute contributions), `predicted_class`, `preds`.

    Raises:
        ValueError: when tokenizer is not a fast tokenizer (offsets required).
    """
    if tokenizer is None and model_name is None:
        raise ValueError("Provide either a tokenizer or a model_name to load one.")

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    if model is not None:
        model_instance = model
    else:
        model_instance = AutoModelForSequenceClassification.from_pretrained(model_name)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_instance.to(device)
    model_instance.eval()

    def predict(texts: List[Any]) -> np.ndarray:
        processed: List[str] = []
        for t in texts:
            if isinstance(t, str):
                processed.append(t)
                continue
            try:
                import numpy as _np

                if isinstance(t, _np.generic):
                    processed.append(str(t))
                    continue
            except Exception:
                pass

            if isinstance(t, (list, tuple)):
                if len(t) == 0:
                    processed.append("")
                    continue
                if all(isinstance(x, (int,)) or hasattr(x, "__int__") for x in t):
                    ids: List[int] = []
                    vocab_size = getattr(tokenizer, "vocab_size", None)
                    for x in t:
                        try:
                            xi = int(x)
                            if vocab_size is not None and (xi < 0 or xi >= vocab_size):
                                xi = 0
                            ids.append(xi)
                        except (OverflowError, ValueError, TypeError):
                            ids.append(0)
                    processed.append(tokenizer.decode(ids, skip_special_tokens=True))
                    continue
                if all(isinstance(x, str) for x in t):
                    try:
                        processed.append(tokenizer.convert_tokens_to_string(list(t)))
                    except Exception:
                        processed.append(" ".join(t))
                    continue

            processed.append(str(t))

        max_len = getattr(tokenizer, "model_max_length", None)
        if not isinstance(max_len, int) or max_len <= 0 or max_len > 4096:
            max_len = 512

        encoding = tokenizer(
            processed,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}
        with torch.no_grad():
            outputs = model_instance(**encoding)

        logits = outputs.logits
        probs = logits.softmax(dim=-1).cpu().numpy()
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
        return probs

    # attempt to obtain offset mapping for subtokens
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Tokenizer must be a fast tokenizer with offsets (tokenizer.is_fast=True).")

    # Tokenize with truncation to the model's max length so offsets align with
    # inputs that will actually be fed to the model (avoids >512 token errors).
    max_len = getattr(tokenizer, "model_max_length", None)
    if not isinstance(max_len, int) or max_len <= 0 or max_len > 4096:
        max_len = getattr(tokenizer, "max_len", None) or 512
    if max_len <= 0 or max_len > 4096:
        max_len = 512
    try:
        encoding = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=True,
            max_length=max_len,
        )
    except OverflowError:
        encoding = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=True,
            max_length=512,
        )
    offsets: List[Tuple[int, int]] = encoding.get("offset_mapping", [])
    subtokens: List[str] = tokenizer.convert_ids_to_tokens(encoding["input_ids"]) if "input_ids" in encoding else tokenizer.tokenize(text)

    # Determine class ordering from an initial pipeline call to ensure
    # `predict` returns arrays with a stable class order (needed by SHAP).
    sample_out = predict([text])
    if hasattr(model_instance.config, "id2label") and isinstance(model_instance.config.id2label, dict):
        labels_order = [model_instance.config.id2label[i] for i in range(len(model_instance.config.id2label))]
    else:
        labels_order = [f"LABEL_{i}" for i in range(sample_out.shape[-1])] if sample_out.ndim > 1 else ["LABEL_0"]

    n_classes = max(1, len(labels_order))

    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(predict, masker)

    # compute explanation (may be slow)
    try:
        shap_exp = explainer([text], max_evals=nsamples)
    except TypeError:
        shap_exp = explainer([text])

    # shap_exp.values shape: (1, n_subtokens, n_classes) or (1, n_subtokens)
    vals = shap_exp.values[0]
    base_values = shap_exp.base_values[0]

    # If multi-class, pick class with highest base value for interpretation
    if vals.ndim == 2:
        # (n_subtokens, n_classes) -> choose class index
        class_idx = int(np.argmax(base_values)) if np.ndim(base_values) > 0 else 0
        shap_per_token = np.abs(vals[:, class_idx]).tolist()
        preds = predict([text])[0].tolist()
        predicted_class = int(np.argmax(preds))
    else:
        shap_per_token = np.abs(vals).tolist()
        preds = predict([text])[0].tolist()
        predicted_class = int(np.argmax(preds)) if len(preds) > 1 else 0

    predicted_label = labels_order[predicted_class] if 0 <= predicted_class < len(labels_order) else None

    return {
        "tokens": subtokens,
        "offsets": offsets,
        "base_values": base_values.tolist() if hasattr(base_values, "tolist") else base_values,
        "shap_values": shap_per_token,
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "preds": preds,
        "labels_order": labels_order,
    }
