import pandas as pd
import numpy as np
from utils.io import setup_logger

log = setup_logger(__name__)

# Define CEFR levels
LEVELS = ["A1", "A2", "B1", "B2", "C1"]
LEVEL_TO_NUM = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5}


def load_efllex(path: str) -> pd.DataFrame:
    """
    Load EFLLex-like data with fixed columns:
      - word: string (lemma)
      - tag: POS tag
      - level_freq@a1 ... level_freq@c1: numeric frequencies
    """
    # Read as tab- or comma-separated
    log.info(f"Loading EFLLex dataset from {path} ...")
    df = pd.read_csv(path, sep="\t" if path.endswith(".tsv") else ",", engine="python")

    expected_cols = ["word", "tag"] + [f"level_freq@{lvl.lower()}" for lvl in LEVELS]
    for c in expected_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    # intersection ensures we ignore any weird extras like The_Signalman@a1
    keep = [c for c in expected_cols if c in df.columns]
    df = df[keep].copy()

    # Standardize column names
    df = df.rename(
        columns={
            "word": "lemma",
            "tag": "POS",
            "level_freq@a1": "f_A1",
            "level_freq@a2": "f_A2",
            "level_freq@b1": "f_B1",
            "level_freq@b2": "f_B2",
            "level_freq@c1": "f_C1",
        }
    )

    # Ensure numeric
    for col in ["f_A1", "f_A2", "f_B1", "f_B2", "f_C1"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    log.info("[OK] Renamed columns and validated EFLLex structure.")
    return df


def compute_cefr_targets(df: pd.DataFrame):
    """
    Compute CEFR numeric target and label, but return them separately.
    The returned dataframe no longer contains any CEFR-related columns.
    """
    log.info("Computing CEFR continuous and discrete targets ...")

    fcols = [f"f_{L}" for L in LEVELS]
    df["_f_sum"] = df[fcols].sum(axis=1).replace(0, np.nan)

    # Weighted continuous score (A1=1 ... C1=5)
    weighted_sum = np.zeros(len(df))
    for L in LEVELS:
        weighted_sum += df[f"f_{L}"].values * LEVEL_TO_NUM[L]
    cefr_score = weighted_sum / df["_f_sum"]

    # Discrete label (not used for training, just metadata)
    cefr_label = df[fcols].idxmax(axis=1).str.replace("f_", "")

    # Drop missing rows
    mask = df["_f_sum"].notna()
    df = df.loc[mask].copy()
    cefr_score = cefr_score[mask].values
    cefr_label = cefr_label[mask].values

    # Drop intermediate + CEFR columns (avoid leakage)
    df.drop(columns=fcols + ["_f_sum"], inplace=True, errors="ignore")

    log.info(f"[OK] Computed CEFR targets for {len(df)} words; cleaned dataframe returned without CEFR columns.")
    return df, cefr_score, cefr_label


def merge_subtlex(df, subtlex_path: str) -> pd.DataFrame:
    """
    Merge SUBTLEX-US frequency info (Excel or CSV) into EFLLex dataframe.
    Expected columns in Excel:
        Word, Lg10WF, Zipf-value
    """
    # Detect format and load appropriately
    log.info(f"Merging SUBTLEX frequency data from {subtlex_path} ...")
    sub = pd.read_excel(subtlex_path, engine="openpyxl")

    # Normalize word column
    sub["Word"] = sub["Word"].astype(str).str.lower()

    # Keep only relevant columns
    cols = []
    for cand in ["Lg10WF", "Zipf-value"]:
        if cand not in sub.columns:
            raise ValueError(f"Expected column '{cand}' not found in SUBTLEX file.")
        cols.append(cand)
    sub_small = sub[["Word"] + cols].copy()
    sub_small = sub_small.rename(columns={"Word": "subtlex_word", "Lg10WF": "log_freq", "Zipf-value": "zipf_value"})

    # Normalize EFLLex words and merge
    df["lemma_lowercase"] = df["lemma"].astype(str).str.lower()
    merged = df.merge(sub_small, left_on="lemma_lowercase", right_on="subtlex_word", how="left")

    # Clean and fill missing values
    merged.drop(columns=["lemma_lowercase", "subtlex_word"], inplace=True, errors="ignore")
    merged["log_freq"] = pd.to_numeric(merged["log_freq"], errors="coerce").fillna(0.0)
    merged["zipf_value"] = pd.to_numeric(merged["zipf_value"], errors="coerce").fillna(0.0)

    log.info(f"[OK] SUBTLEX merge complete — matched {(merged['log_freq'] > 0).sum()} of {len(merged)} words.")
    return merged
