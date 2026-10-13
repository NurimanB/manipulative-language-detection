"""Shared data loading, template groups, and the train/test split.

Every experiment notebook imports from here, so all of them use exactly the
same data, the same near-duplicate groups, and the same held-out test set.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

MAIN_FILE = "dark_patterns_ecommerce_dataset.csv"
EXTERNAL_FILE = "manipulative_language_dataset_300.csv"


def find_file(name):
    for p in [DATA_DIR / name, ROOT / name, Path(name), Path("/mnt/data") / name]:
        if p.exists():
            return p
    raise FileNotFoundError(f"{name} not found. Put it in {DATA_DIR}.")


def load_datasets():
    """Return (main dataset, original project dataset) as raw DataFrames."""
    df = pd.read_csv(find_file(MAIN_FILE))
    external_df = pd.read_csv(find_file(EXTERNAL_FILE))
    return df, external_df


def add_template_groups(df):
    """Add a `template` column (lower-cased, digits -> '#') and return (main_df, groups).

    Texts with the same template are near-duplicates and must stay on one side of every split.
    """
    main_df = df.reset_index(drop=True).copy()
    main_df["template"] = (
        main_df["text"].str.lower().str.replace(r"\d+", "#", regex=True).str.strip()
    )
    groups = main_df["template"].factorize()[0]
    return main_df, groups


def split_external(external_df):
    """Return (real_df, external_synthetic_df) from the original project data."""
    source_type = external_df["source_type"].str.lower()
    real_df = external_df[source_type == "real"].copy()
    external_synthetic_df = external_df[source_type == "synthetic"].copy()
    return real_df, external_synthetic_df


def get_split(main_df, groups):
    """The shared 80/20 stratified + grouped split (first fold of StratifiedGroupKFold)."""
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(main_df["text"], main_df["manipulative"], groups))
    return train_idx, test_idx


def load_all():
    """Convenience loader for experiments 2-4: everything needed in one call."""
    df, external_df = load_datasets()
    main_df, groups = add_template_groups(df)
    real_df, external_synthetic_df = split_external(external_df)
    train_idx, test_idx = get_split(main_df, groups)
    return main_df, groups, real_df, external_synthetic_df, train_idx, test_idx


def save_table(table, name):
    """Save a results table to results/<name>.csv so later notebooks can compare against it."""
    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / f"{name}.csv", index=False)


def load_table(name):
    path = RESULTS_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the earlier experiment notebook first.")
    return pd.read_csv(path)
