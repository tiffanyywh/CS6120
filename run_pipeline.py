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

log = setup_logger("run_pipeline")


def main(use_cross_validation=True):
    # STEP 1: Loading EFLLex and SUBTLEX data
    df = load_efllex("data/EFLLex_NLP4J.tsv")
    df, y, _ = compute_cefr_targets(df)
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

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    if use_cross_validation:
        # STEP 6: Cross-validation training & evaluation
        cv_results = cross_validate_models(models, X, y, cv_folds=5)
        summarize_results(cv_results)
        # Select best model (lowest RMSE mean)
        best_model_name = min(cv_results, key=lambda r: r["RMSE_mean"])["model"]
        best_model = models[best_model_name]
        log.info(f"Selected best model: {best_model_name}")
    else:
        # STEP 6: Train/test split training & evaluation
        results = train_and_evaluate(models, X, y)
        summarize_results(results)
        # Select best model (lowest RMSE)
        best_model_name = min(results, key=lambda r: r["RMSE"])["model"]
        best_model = models[best_model_name]
        log.info(f"Selected best model: {best_model_name}")

        # debugging purpose - save predictions on test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
        best_model.fit(X_train, y_train)
        save_predictions(
            df=df,
            model=best_model,
            X=X_train, y=y_train,
            X_test=X_test, y_test=y_test,
            df_test=df.iloc[X_test.index] if hasattr(X_test, 'index') else None,
            outdir="results",
            model_name=best_model_name,
            mode="test"
        )


if __name__ == "__main__":
    main(use_cross_validation=True)
