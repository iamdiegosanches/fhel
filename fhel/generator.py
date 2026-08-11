from typing import Dict, Any


def generate_explanation(fhel_output: Dict[str, Any]) -> str:
    """Generate a Portuguese natural language explanation from FHEL output.

    The template reports the dominant level, its percentage, and top contributing tokens.
    """
    levels = fhel_output.get("levels_percent", {})
    top = fhel_output.get("top_features", {})

    # determine winner
    winner = max(levels.items(), key=lambda x: x[1]) if levels else (None, 0.0)
    winner_name = winner[0]
    winner_pct = winner[1]

    lines = []
    lines.append(f"Explicação FHEL — nível dominante: {winner_name} ({winner_pct:.1f}% do impacto).")
    lines.append("")
    lines.append("Contribuições por nível:")
    for lvl, pct in levels.items():
        lines.append(f"- {lvl}: {pct:.1f}%")

    lines.append("")
    lines.append("Top features por nível:")
    for lvl, feats in top.items():
        if not feats:
            continue
        names = ", ".join([f"{f['token']} ({f['contribution']:.3f})" for f in feats[:5]])
        lines.append(f"- {lvl}: {names}")

    lines.append("")
    lines.append("Observação: valores são baseados na agregação dos pesos absolutos do SHAP por subtokens alinhados aos tokens do spaCy.")
    return "\n".join(lines)
