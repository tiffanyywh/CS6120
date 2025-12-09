import os
import numpy as np
import pandas as pd

from data.loader import load_efllex, compute_cefr_targets, merge_subtlex
from data.features import add_surface_features, load_fasttext_model, add_embedding_columns, build_feature_matrix
from models.regressors import get_models
from models.train import train_and_evaluate, cross_validate_models
from eval.report import summarize_results
from utils.io import setup_logger
from sklearn.model_selection import train_test_split
from eval.predictions import save_predictions
from models.train import tune_xgboost

log = setup_logger("run_pipeline")


def main(use_cross_validation=True):
    # STEP 1: Loading EFLLex and SUBTLEX data
    df = load_efllex("data/EFLLex_NLP4J.tsv")
    df, y, cefr_labels = compute_cefr_targets(df)
    df = merge_subtlex(df, "data/SUBTLEX-US frequency list with PoS and Zipf information.xlsx")

    # STEP 2: Adding surface features
    df = add_surface_features(df)

    # STEP 3: Loading fastText embeddings
    vec = load_fasttext_model("data/cc.en.300.vec")
    df, emb_cols = add_embedding_columns(df, vec)

    os.makedirs("data_cache", exist_ok=True)

    # STEP 4: Building feature matrix
    X, num_cols, cat_cols = build_feature_matrix(df)

    # STEP 5: Building models
    models = get_models(num_cols, cat_cols)
    xgb_pipeline = models["xgboost"]
    best_xgb, best_params, best_rmse = tune_xgboost(X, y, xgb_pipeline, outdir="results", n_iter=30, cv_folds=3)

if __name__ == "__main__":
    main(use_cross_validation=True)
