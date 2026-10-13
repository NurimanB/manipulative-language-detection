"""
Builds dark_patterns_ecommerce_dataset.csv from two public sources:

1. Mathur et al. (2019), "Dark Patterns at Scale" — 1,818 dark-pattern texts
   from shopping websites, each with a Pattern Category and Pattern Type.
   https://github.com/aruneshmathur/dark-patterns  (GPL-3.0)
2. Yada et al. (2022), "Dark patterns in e-commerce: a dataset and its baseline
   evaluations" — the deduplicated Mathur texts (1,178) plus 1,178 non-dark-pattern
   texts scraped from the same websites.
   https://github.com/yamanalab/ec-darkpattern  (Apache-2.0)

Yada provides the balanced binary dataset; Mathur provides the fine-grained
Pattern Type, which we map onto our technique labels.
"""
import pandas as pd

MATHUR_URL = "https://raw.githubusercontent.com/aruneshmathur/dark-patterns/master/data/final-dark-patterns/dark-patterns.csv"
YADA_URL = "https://raw.githubusercontent.com/yamanalab/ec-darkpattern/master/dataset/dataset.tsv"

# Mathur Pattern Type -> our technique label
TYPE_TO_LABEL = {
    "Countdown Timer": "urgency",
    "Limited-time Message": "urgency",
    "Low-stock Message": "scarcity",
    "High-demand Message": "scarcity",
    "Activity Notification": "social_proof",
    "Testimonials of Uncertain Origin": "social_proof",
    "Confirmshaming": "emotional_pressure",
}
LABELS = ["urgency", "scarcity", "social_proof", "emotional_pressure"]


def build(mathur_path=MATHUR_URL, yada_path=YADA_URL):
    yada = pd.read_csv(yada_path, sep="\t")
    mathur = (
        pd.read_csv(mathur_path)
        .dropna(subset=["Pattern String"])
        .drop_duplicates("Pattern String")
        .rename(columns={"Pattern String": "text", "Pattern Type": "pattern_type"})
    )[["text", "pattern_type"]]

    df = yada.merge(mathur, on="text", how="left")
    df["pattern_type"] = df["pattern_type"].fillna("Not Dark Pattern")
    df = df.rename(columns={"label": "manipulative", "Pattern Category": "pattern_category"})

    for label in LABELS:
        df[label] = (df["pattern_type"].map(TYPE_TO_LABEL) == label).astype(int)

    # light cleaning: collapse whitespace and literal "\n" left over from scraping
    df["text"] = (
        df["text"].astype(str)
        .str.replace(r"\\n", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df = df[df["text"].str.len() > 0].drop_duplicates("text").reset_index(drop=True)
    df.insert(0, "id", range(1, len(df) + 1))
    df["source"] = df["manipulative"].map({1: "mathur_2019_dark_patterns", 0: "yada_2022_non_dark_patterns"})

    return df[["id", "text", "manipulative", *LABELS, "pattern_category", "pattern_type", "page_id", "source"]]


if __name__ == "__main__":
    import sys
    out = build(*(sys.argv[1:3])) if len(sys.argv) >= 3 else build()
    out.to_csv("dark_patterns_ecommerce_dataset.csv", index=False)
    print(out.shape)
    print(out["manipulative"].value_counts())
    print(out[LABELS].sum())
