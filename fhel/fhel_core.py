from typing import Any, Dict, List, Tuple

from dataclasses import dataclass


@dataclass
class TokenContribution:
    text: str
    start: int
    end: int
    shap: float
    pos: str
    ent_type: str
    is_rhetorical: bool
    contributions: Dict[str, float]


class FHELCore:
    """Core engine that aligns BERT subtokens with spaCy tokens and aggregates SHAP.

    The algorithm:
    - use offsets produced by a fast tokenizer to map subtokens -> character spans
    - overlap spans with spaCy tokens to aggregate subtoken SHAP values per word
    - distribute each word's SHAP absolute mass across three linguistic levels
      (Morfossintático, Semântico, Retórico) by simple, reproducible heuristics
    """

    LEVELS = ["Morfossintatico", "Semantico", "Retorico"]

    def __init__(self) -> None:
        pass

    @staticmethod
    def _is_rhetorical(token_text: str, token_tag: str) -> bool:
        # Heuristics for rhetorical cues
        if any(ch in token_text for ch in ["!", "?", '"', "'", "—"]):
            return True
        if token_text.isupper() and len(token_text) > 1:
            return True
        if token_tag in {"INTJ", "PUNCT"}:
            return True
        return False

    def aggregate(
        self,
        tokens_offsets: List[Tuple[int, int]],
        subtokens: List[str],
        subtoken_shap: List[float],
        spacy_tokens: List[Dict[str, object]],
        top_k: int = 5,
    ) -> Dict[str, object]:
        """Align subtokens with spaCy tokens and aggregate into linguistic levels.

        Args:
            tokens_offsets: list of (start,end) char offsets for each subtoken.
            subtokens: list of subtoken strings (BERT tokenization)
            subtoken_shap: list of absolute shap values per subtoken
            spacy_tokens: output of Annotator.annotate(text)
            top_k: number of top features to return per level

        Returns:
            Dict with aggregated percentages and top features per level, and per-token contributions.
        """
        if len(tokens_offsets) != len(subtokens) or len(subtokens) != len(subtoken_shap):
            # lengths mismatch can happen if SHAP removed special tokens; attempt best-effort
            pass

        # map each spacy token to accumulated shap
        per_word_abs = [0.0 for _ in range(len(spacy_tokens))]
        per_word_details: List[TokenContribution] = []

        # Build list of subtoken spans
        subtoken_spans = tokens_offsets

        for wi, w in enumerate(spacy_tokens):
            wstart, wend = w["start"], w["end"]
            matched_idxs = []
            for si, (sstart, send) in enumerate(subtoken_spans):
                # overlap condition
                if not (send <= wstart or sstart >= wend):
                    matched_idxs.append(si)
            # sum absolute shap of matched subtokens
            shap_sum = 0.0
            for si in matched_idxs:
                if si < len(subtoken_shap):
                    shap_sum += abs(subtoken_shap[si])
            per_word_abs[wi] = shap_sum
            is_rhet = self._is_rhetorical(w["text"], w["tag"])
            per_word_details.append(
                TokenContribution(
                    text=w["text"],
                    start=wstart,
                    end=wend,
                    shap=shap_sum,
                    pos=w["pos"],
                    ent_type=w.get("ent_type", ""),
                    is_rhetorical=is_rhet,
                    contributions={"Morfossintatico": 0.0, "Semantico": 0.0, "Retorico": 0.0},
                )
            )

        total_mass = sum(per_word_abs) if sum(per_word_abs) > 0 else 1.0

        # Distribute each token's mass across levels using reproducible heuristics
        level_sums = {lvl: 0.0 for lvl in self.LEVELS}

        for i, comp in enumerate(per_word_details):
            mass = comp.shap
            s_pos = 1.0
            s_ner = 1.0 if comp.ent_type else 0.0
            s_ret = 1.0 if comp.is_rhetorical else 0.0
            denom = s_pos + s_ner + s_ret
            if denom == 0:
                s_pos = 1.0
                denom = 1.0
            share_pos = (s_pos / denom) * mass
            share_ner = (s_ner / denom) * mass
            share_ret = (s_ret / denom) * mass
            comp.contributions["Morfossintatico"] = share_pos
            comp.contributions["Semantico"] = share_ner
            comp.contributions["Retorico"] = share_ret
            level_sums["Morfossintatico"] += share_pos
            level_sums["Semantico"] += share_ner
            level_sums["Retorico"] += share_ret

        # Normalize to percentages
        total = sum(level_sums.values()) if sum(level_sums.values()) > 0 else 1.0
        level_percent = {k: 100.0 * v / total for k, v in level_sums.items()}

        # Top features per level
        top_features = {}
        for level in self.LEVELS:
            ranked = sorted(
                [(comp.text, comp.contributions[level]) for comp in per_word_details], key=lambda x: x[1], reverse=True
            )
            top_features[level] = [ {"token": t, "contribution": float(c)} for t, c in ranked[:top_k] if c > 0 ]

        return {
            "levels_percent": level_percent,
            "level_sums": {k: float(v) for k, v in level_sums.items()},
            "top_features": top_features,
            "per_token": [
                {
                    "text": comp.text,
                    "start": comp.start,
                    "end": comp.end,
                    "pos": comp.pos,
                    "ent_type": comp.ent_type,
                    "shap": comp.shap,
                    "contributions": comp.contributions,
                }
                for comp in per_word_details
            ],
        }
