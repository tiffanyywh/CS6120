import os
import json
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from eval.metrics import evaluate_regression, cefr_accuracy_scorer, cefr_accuracy_within1
from utils.io import setup_logger
from sklearn.metrics import make_scorer

log = setup_logger(__name__)


def train_and_evaluate(models, X, y, test_size=0.15, random_state=42):
    log.info(f"Splitting {len(X)} samples into train/test ...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    results = []
    for name, model in models.items():
        log.info(f"Training model: {name}")
        start = time.time()
        model.fit(X_train, y_train)
        log.info(f"{name} training done in {time.time() - start:.2f}s. Evaluating ...")
        preds = model.predict(X_test)
        metrics = evaluate_regression(y_test, preds)
        log.info(f"{name} metrics: {metrics}")
        results.append({"model": name, **metrics})
    log.info("All models trained and evaluated.")
    return results


def cross_validate_models(models, X, y, cv_folds=5, outdir="results"):
    log.info(f"Starting {cv_folds}-fold cross-validation ...")
    results = []
    for name, model in models.items():
        log.info(f"Training model: {name}")
        start = time.time()

        # --- cross-validation ---
        try:
            # Wrap into sklearn scorer
            cefr_accuracy = make_scorer(cefr_accuracy_scorer, greater_is_better=True)
            cefr_cefr_accuracy_within1 = make_scorer(cefr_accuracy_within1, greater_is_better=True)
            scores = cross_validate(
                model, X, y,
                cv=cv_folds,
                scoring={
                    "rmse": "neg_root_mean_squared_error",
                    "acc": cefr_accuracy,
                    "acc_within1": cefr_cefr_accuracy_within1,
                },
                n_jobs=-1
            )
            rmse_mean = -np.mean(scores["test_rmse"])
            rmse_std = np.std(scores["test_rmse"])
            acc_mean = np.mean(scores["test_acc"])
            acc_std = np.std(scores["test_acc"])
            acc_within1_mean = np.mean(scores["test_acc_within1"])
            acc_within1_std = np.std(scores["test_acc_within1"])

            results.append({
                "model": name,
                "RMSE_mean": rmse_mean,
                "RMSE_std": rmse_std,
                "Acc_mean": acc_mean,
                "Acc_std": acc_std,
                "Acc_within1_mean": acc_within1_mean,
                "Acc_within1_std": acc_within1_std,
            })

            log.info(f"{name} CV done in {time.time() - start:.2f}s → RMSE={rmse_mean:.3f} ± {rmse_std:.3f}, Accuracy={acc_mean:.3f} ± {acc_std:.3f}, Accuracy_Within1={acc_within1_mean:.3f} ± {acc_within1_std:.3f}")
        except Exception as e:
            log.warning(f"[WARN] Cross-validation failed for {name}: {e}")
            continue

        if hasattr(model, "named_steps") and "model" in model.named_steps:
            base_model = model.named_steps["model"]
            if isinstance(base_model, XGBRegressor):
                xgboost_feature_importance(model, base_model, X, y, outdir)

    return results

    # --- Feature importance for XGBoost ---


def xgboost_feature_importance(model, base_model, X, y, outdir="results"):
    log.info(f"Computing feature importance for xgboost ...")

    # Apply preprocessing pipeline to get numeric features
    pre = model.named_steps["pre"]
    X_preprocessed = pre.fit_transform(X)

    # Fit once on the whole data for interpretability
    base_model.fit(X_preprocessed, y)

    # Retrieve feature importances from XGBoost
    feature_names = pre.get_feature_names_out()
    importance = base_model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False)

    csv_path = os.path.join(outdir, f"xgboost_feature_importance.csv")
    importance_df.to_csv(csv_path, index=False)
    log.info(f"Saved feature importances → {csv_path}")

    # Plot top 20 features
    topn = 20
    plt.figure(figsize=(8, 6))
    importance_df.head(topn).plot(
        x="feature", y="importance", kind="barh",
        legend=False, color="skyblue"
    )
    plt.gca().invert_yaxis()
    plt.title(f"Top {topn} Feature Importances (xgboost)")
    plt.tight_layout()
    plot_path = os.path.join(outdir, f"xgboost_feature_importance.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    log.info(f"Saved feature importance plot → {plot_path}")


def tune_xgboost(X, y, xgb_pipeline, outdir="results", n_iter=25, cv_folds=3, random_state=42):
    """
    Tune the XGBoost model inside a sklearn pipeline using RandomizedSearchCV.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Training features
    y : np.ndarray
        Target vector
    xgb_pipeline : sklearn.pipeline.Pipeline
        Your full pipeline (preprocessor + model)
    outdir : str
        Folder to save tuning results (default: 'results')
    n_iter : int
        Number of random parameter combinations to try
    cv_folds : int
        Number of folds for cross-validation
    random_state : int
        Random seed for reproducibility
    """

    log.info("=== Starting XGBoost hyperparameter tuning ===")

    param_dist = {
        "model__n_estimators": [200, 400, 600, 800],
        "model__max_depth": [4, 6, 8, 10],
        "model__learning_rate": np.linspace(0.01, 0.1, 10),
        "model__subsample": np.linspace(0.7, 1.0, 4),
        "model__colsample_bytree": np.linspace(0.7, 1.0, 4),
        "model__reg_lambda": [0.5, 1.0, 2.0],
        "model__reg_alpha": [0.0, 0.1, 0.5],
    }

    search = RandomizedSearchCV(
        estimator=xgb_pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv_folds,
        scoring="neg_root_mean_squared_error",
        random_state=random_state,
        n_jobs=-1,
        verbose=2
    )

    search.fit(X, y)

    best_rmse = -search.best_score_
    best_params = search.best_params_

    os.makedirs(outdir, exist_ok=True)
    params_path = os.path.join(outdir, "best_xgboost_params.json")

    with open(params_path, "w") as f:
        json.dump({
            "best_rmse": best_rmse,
            "best_params": best_params
        }, f, indent=2)

    log.info(f"[OK] Best RMSE: {best_rmse:.4f}")
    log.info(f"[OK] Saved best parameters → {params_path}")

    return search.best_estimator_, best_params, best_rmse
