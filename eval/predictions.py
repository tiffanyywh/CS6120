import pandas as pd
import numpy as np
import os
from utils.io import setup_logger

log = setup_logger(__name__)

def save_predictions(df, model, X, y, outdir, model_name="best_model",
                     X_test=None, y_test=None, df_test=None, mode="test"):
    """
    Save model predictions to CSV.
    mode = "test"  → save predictions only on test set
    mode = "all"   → fit on all data and save full predictions
    """
    os.makedirs(outdir, exist_ok=True)

    if mode == "test" and X_test is not None and y_test is not None:
        log.info(f"Generating and saving predictions on TEST SET for {model_name} ...")
        preds = model.predict(X_test)
        out_df = df_test.copy() if df_test is not None else pd.DataFrame()
        out_df["true_cefr_score"] = y_test
        out_df["pred_cefr_score"] = preds
    else:
        log.info(f"Fitting model on ALL data and saving full predictions for {model_name} ...")
        model.fit(X, y)
        preds = model.predict(X)
        out_df = df.copy()
        out_df["pred_cefr_score"] = preds
        out_df["pred_cefr_label"] = np.round(preds).clip(1, 5)
        out_df["pred_cefr_label"] = out_df["pred_cefr_label"].map(
            {1:"A1",2:"A2",3:"B1",4:"B2",5:"C1"}
        )

    csv_path = os.path.join(outdir, f"predicted_{model_name}_{mode}.csv")
    out_df.to_csv(csv_path, index=False, encoding="utf-8")
    log.info(f"[OK] Saved {mode.upper()} predictions → {csv_path}")
    return csv_path
