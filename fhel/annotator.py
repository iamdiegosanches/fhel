from typing import Dict, List, Optional

import spacy


class Annotator:
    """Linguistic annotator using spaCy Portuguese models.

    Loads `pt_core_news_lg` by default but allows custom model names.
    """

    def __init__(self, model: Optional[str] = "pt_core_news_lg") -> None:
        try:
            self.nlp = spacy.load(model)
        except Exception as e:
            raise RuntimeError(f"Failed loading spaCy model '{model}': {e}")

    def annotate(self, text: str) -> List[Dict[str, object]]:
        """Annotate `text` returning a list of token-level dicts.

        Each dict contains: `text`, `start`, `end`, `pos`, `tag`, `lemma`, `ent_type`, `ent_iob`, `is_sent_start`.
        """
        doc = self.nlp(text)
        tokens = []
        for token in doc:
            tokens.append(
                {
                    "text": token.text,
                    "start": token.idx,
                    "end": token.idx + len(token.text),
                    "pos": token.pos_,
                    "tag": token.tag_,
                    "lemma": token.lemma_,
                    "ent_type": token.ent_type_ if token.ent_type_ else "",
                    "ent_iob": token.ent_iob_,
                    "is_sent_start": bool(token.is_sent_start),
                }
            )
        return tokens
