import os

import textstat
from gensim.models import KeyedVectors
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import numpy as np
import time
from utils.io import setup_logger

log = setup_logger(__name__)

# Optional affix lists — you can keep or remove
AFFIX_PREFIXES = ["un", "re", "in", "im", "dis", "non", "pre", "mis"]
AFFIX_SUFFIXES = ["tion", "ness", "ment", "ity", "able", "less", "ous", "ive"]


def add_surface_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute basic surface-level and morphological features from 'lemma'.
    Assumes columns:
      - lemma : the word itself
      - POS   : part-of-speech tag
    """
    log.info(f"Extracting surface features for {len(df)} words ...")
    start = time.time()
    # Word length
    df["len_chars"] = df["lemma"].astype(str).str.len()

    # Syllable count
    df["syllables"] = df["lemma"].apply(textstat.syllable_count)

    # Prefix/suffix flags
    for p in AFFIX_PREFIXES:
        df[f"pref_{p}"] = df["lemma"].str.lower().str.startswith(p).astype(int)
    for s in AFFIX_SUFFIXES:
        df[f"suf_{s}"] = df["lemma"].str.lower().str.endswith(s).astype(int)

    # Multiword / hyphen detection
    df["is_multiword"] = df["lemma"].str.contains(r"[\s-]").astype(int)

    log.info(f"[OK] Surface features extracted in {time.time() - start:.2f}s.")
    return df


def load_fasttext_model(path="data/cc.en.300.vec", cache_path="data_cache/fasttext_model.kv"):
    """
    Load pretrained word vectors for embeddings.
    """
    if os.path.exists(cache_path):
        log.info(f"[OK] Loading cached fastText vectors from {cache_path}")
        return KeyedVectors.load(cache_path, mmap='r')
    log.info(f"Loading vectors from {path} (this may take several minutes)...")
    start = time.time()
    model = KeyedVectors.load_word2vec_format(path, binary=False)
    model.save(cache_path)
    log.info(
        f"[OK] word2vec-format vectors loaded in {time.time() - start:.2f}s. Cached fastText model saved to {cache_path}")
    return model


def add_embeddings(df, model):
    """Return a NumPy matrix of shape (n_words, vector_dim)."""
    log.info(f"Generating embeddings for {len(df)} words (dim={model.vector_size}) ...")
    start = time.time()
    vectors = []
    oov = 0
    for w in df["lemma"].astype(str):
        w_low = w.lower()
        if w_low in model:
            vectors.append(model[w_low])
        else:
            oov += 1
            vectors.append(np.zeros(model.vector_size, dtype=float))
    log.info(
        f"[OK] Embeddings computed in {time.time() - start:.2f}s. OOV words: {oov}/{len(df)} ({oov / len(df) * 100:.1f}%)")
    return np.vstack(vectors)


def add_embedding_columns(df, model, prefix: str = "emb_", reduce_dim: bool = True,
                          n_components: int = 50, random_state: int = 42):
    """
    Add embedding columns to the DataFrame for sklearn ColumnTransformer pipelines.
    Optionally applies TruncatedSVD to reduce embedding dimensionality.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with lemma column.
    model : gensim KeyedVectors
        Preloaded embedding model.
    prefix : str
        Prefix for embedding column names (default: "emb_").
    reduce_dim : bool
        Whether to apply TruncatedSVD for dimensionality reduction.
    n_components : int
        Number of SVD components if reduce_dim=True.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    (pd.DataFrame, list[str])
        Updated DataFrame with new embedding columns and their names.
    """
    start = time.time()

    # Generate embedding matrix
    M = add_embeddings(df, model)

    log.info(f"[OK] Generated raw embeddings matrix of shape {M.shape}")

    # Optionally apply SVD reduction
    if reduce_dim:
        log.info(f"Applying TruncatedSVD → {n_components} dimensions ...")
        svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        M = svd.fit_transform(M)
        log.info(f"[OK] Reduced embeddings from {model.vector_size} → {M.shape[1]} dims")

    # Create DataFrame and attach to df
    emb_cols = [f"{prefix}{i}" for i in range(M.shape[1])]
    emb_df = pd.DataFrame(M, columns=emb_cols, index=df.index)
    df = pd.concat([df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)

    log.info(f"[OK] Added {len(emb_cols)} embedding columns in {time.time() - start:.2f}s.")
    return df, emb_cols


def build_feature_matrix(df):
    """
    Select and return numeric + categorical features for modeling.
    Includes:
      - surface features
      - SUBTLEX features (log_freq, zipf_value) if present
      - embedding columns (emb_*) if previously added via add_embedding_columns
    """
    numeric_cols = [
        "len_chars", "syllables", "is_multiword",
    ]
    # include affix flags
    numeric_cols += [c for c in df.columns if c.startswith("pref_") or c.startswith("suf_")]
    # include SUBTLEX features if present
    for c in ("log_freq", "zipf_value"):
        if c in df.columns:
            numeric_cols.append(c)
    # include embedding columns if present
    numeric_cols += [c for c in df.columns if c.startswith("emb_")]

    categorical_cols = ["POS"] if "POS" in df.columns else []
    X = df[numeric_cols + categorical_cols].copy()
    log.info(f"Final features used: {[c for c in df.columns if not c.startswith('emb_')][:10]}...")
    log.info(f"Embedding dims: {len([c for c in df.columns if c.startswith('emb_')])}")
    return X, numeric_cols, categorical_cols
