from typing import Any, Dict, List, Tuple, Literal

from dataclasses import dataclass

TokenLevel = Literal["Morfossintatico", "Semantico", "Retorico", "Outros"]


@dataclass
class TokenContribution:
    text: str
    start: int
    end: int
    shap: float
    pos: str
    ent_type: str
    level: TokenLevel
    contributions: Dict[str, float]


class FHELCore:
    """Core engine that aligns BERT subtokens with spaCy tokens and aggregates SHAP.

    The FHEL aggregation aligns WordPiece SHAP scores to spaCy tokens using character
    offsets and distributes each token's absolute mass into a linguistic taxonomy:
    Semantico, Retorico, Morfossintatico, and Outros.
    """

    LEVELS: List[str] = ["Morfossintatico", "Semantico", "Retorico"]
    ALL_LEVELS: List[str] = [*LEVELS, "Outros"]

    def __init__(self) -> None:
        pass

    @staticmethod
    def _categorize_token(token: Dict[str, object]) -> TokenLevel:
        """Map a spaCy token annotation to one of the FHEL linguistic levels.

        The categorization follows the requested taxonomy:
        - Semantico: ent_type preenchido OU pos em {NOUN, ADJ, PROPN}
        - Retorico: pos em {CCONJ, SCONJ, ADV, PUNCT, INTJ} OU is_sent_start == True
        - Morfossintatico: pos em {VERB, AUX, PRON, DET, ADP, NUM}
        - Outros: qualquer outro token
        """
        ent_type = str(token.get("ent_type", "") or "").strip()
        pos = str(token.get("pos", "") or "").upper()
        if ent_type or pos in {"NOUN", "ADJ", "PROPN"}:
            return "Semantico"
        if pos in {"CCONJ", "SCONJ", "ADV", "PUNCT", "INTJ"} or token.get("is_sent_start") is True:
            return "Retorico"
        if pos in {"VERB", "AUX", "PRON", "DET", "ADP", "NUM"}:
            return "Morfossintatico"
        return "Outros"

    def _align_subtokens_to_spacy(
        self,
        tokens_offsets: List[Tuple[int, int]],
        subtokens: List[str],
        subtoken_shap: List[float],
        spacy_tokens: List[Dict[str, object]],
    ) -> List[TokenContribution]:
        """Align BERT subtokens with spaCy tokens and aggregate absolute SHAP values.

        Returns a list of per-token contributions with the assigned linguistic level.
        """
        effective_len = min(len(tokens_offsets), len(subtokens), len(subtoken_shap))
        subtoken_spans = tokens_offsets[:effective_len]
        subtoken_values = [abs(float(v)) for v in subtoken_shap[:effective_len]]

        contributions: List[TokenContribution] = []
        for token in spacy_tokens:
            start = int(token["start"])
            end = int(token["end"])
            matched_sum = 0.0
            for span_idx, (span_start, span_end) in enumerate(subtoken_spans):
                if not (span_end <= start or span_start >= end):
                    matched_sum += subtoken_values[span_idx]
            level = self._categorize_token(token)
            contributions.append(
                TokenContribution(
                    text=str(token.get("text", "")),
                    start=start,
                    end=end,
                    shap=matched_sum,
                    pos=str(token.get("pos", "")),
                    ent_type=str(token.get("ent_type", "")),
                    level=level,
                    contributions={level: matched_sum} if level in self.LEVELS else {"Outros": matched_sum},
                )
            )
        return contributions

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
        token_contributions = self._align_subtokens_to_spacy(
            tokens_offsets=tokens_offsets,
            subtokens=subtokens,
            subtoken_shap=subtoken_shap,
            spacy_tokens=spacy_tokens,
        )

        level_sums: Dict[str, float] = {level: 0.0 for level in self.ALL_LEVELS}
        for contribution in token_contributions:
            for level, value in contribution.contributions.items():
                level_sums[level] = level_sums.get(level, 0.0) + float(value)

        primary_total = sum(level_sums[level] for level in self.LEVELS)
        primary_total = primary_total if primary_total > 0 else 1.0
        level_percent = {level: 100.0 * level_sums[level] / primary_total for level in self.LEVELS}

        top_features: Dict[str, List[Dict[str, float]]] = {}
        for level in self.LEVELS:
            ranked = sorted(
                [
                    (contribution.text, contribution.contributions.get(level, 0.0))
                    for contribution in token_contributions
                ],
                key=lambda pair: pair[1],
                reverse=True,
            )
            top_features[level] = [
                {"token": text, "contribution": float(weight)}
                for text, weight in ranked[:top_k]
                if weight > 0.0
            ]

        return {
            "levels_percent": level_percent,
            "level_sums": {level: float(value) for level, value in level_sums.items()},
            "top_features": top_features,
            "per_token": [
                {
                    "text": contribution.text,
                    "start": contribution.start,
                    "end": contribution.end,
                    "pos": contribution.pos,
                    "ent_type": contribution.ent_type,
                    "level": contribution.level,
                    "shap": float(contribution.shap),
                    "contributions": {k: float(v) for k, v in contribution.contributions.items()},
                }
                for contribution in token_contributions
            ],
        }
