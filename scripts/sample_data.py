"""Utility to sample a balanced subset from the Fake.Br consolidated CSV.

Usage:
    python scripts/sample_data.py --n_per_class 50 --input data/fake_br_consolidated.csv --output outputs/sample_balanced.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import argparse

import pandas as pd


DATA_PATH = Path("data/fake_br_consolidated.csv")


def sample_balanced(df: pd.DataFrame, n_per_class: int = 50, seed: int = 42) -> pd.DataFrame:
    parts = []
    for label, group in df.groupby("label"):
        parts.append(group.sample(n=min(n_per_class, len(group)), random_state=seed))
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample balanced Fake.Br examples")
    parser.add_argument("--n_per_class", type=int, default=50)
    parser.add_argument("--input", dest="infile", default=str(DATA_PATH))
    parser.add_argument("--output", dest="outfile", default="outputs/sample_balanced.csv")
    args = parser.parse_args()

    infile = Path(args.infile)
    outfile = Path(args.outfile)
    if not infile.exists():
        raise FileNotFoundError(f"Input file not found: {infile}")

    outfile.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(infile)
    sampled = sample_balanced(df, n_per_class=args.n_per_class)
    sampled.to_csv(outfile, index=False)
    print(f"Wrote {len(sampled)} rows to {outfile}")


if __name__ == "__main__":
    main()
