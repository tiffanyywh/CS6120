# Dataset
https://cental.uclouvain.be/cefrlex/efllex/

https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus

dataset with zipf values: https://www.ugent.be/pp/experimentele-psychologie/en/research/documents/subtlexus/subtlexus1.zip

embedding model: 
https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz
https://fasttext.cc/docs/en/crawl-vectors.html

# Experiments
- Setup1 
  - dataset: efllex
  - features: surface-feature-only 
  - model: regressor (elasticnet, random_forest)
  - result:
    ```
    (python3) ➜  CS6120Project python -m run_pipeline
               model       MAE      RMSE  Pearson_r  Spearman_rho
    0     elasticnet  0.815970  1.005712   0.184297      0.185457
    1  random_forest  0.820514  1.017699   0.173965      0.178673
    Best model: elasticnet (RMSE=1.006)
    ```

- Setup2 add subtlexus features and embedding
  - dataset: efllex, subtlexus
  - features: surface-feature-only, subtlex features like log_freq and zipf, embedding
  - model: regressor (elasticnet, xgboost) (tree-based model like random forest scale very poorly with high-dimensional dense features)
  - result:
    ```
            model       MAE      RMSE  Pearson_r  Spearman_rho
    0  elasticnet  0.764115  0.962181   0.339400      0.348336
    1     xgboost  0.745343  0.945629   0.383759      0.395223
    
    Best model: xgboost (RMSE=0.946)
    ```
    
    | Metric         | ElasticNet |   XGBoost   | Interpretation                                                                    |
    | :------------- | :--------: | :---------: | :-------------------------------------------------------------------------------- |
    | **MAE**        |    0.764   | **0.745 ↓** | On average, XGBoost predictions are closer to true CEFR scores (by ~0.02 levels). |
    | **RMSE**       |    0.962   | **0.946 ↓** | Consistent improvement; fewer large errors.                                       |
    | **Pearson r**  |    0.339   | **0.384 ↑** | Stronger linear correlation with ground truth CEFR levels.                        |
    | **Spearman ρ** |    0.348   | **0.395 ↑** | Better rank-ordering: XGBoost better preserves lexical difficulty ordering.       |

- Setup3 cross-validation
  - dataset: efllex, subtlexus
  - features: surface-feature-only, subtlex features like log_freq and zipf, embedding
  - model: xgboost
  - cross-validation: 5 folds
  - result:
    ```aiignore
    [18:25:24] [INFO] Generating embeddings for 15281 words (dim=300) ...
    [18:25:24] [INFO] [OK] Embeddings computed in 0.13s. OOV words: 3991/15281 (26.1%)
    [18:25:24] [INFO] [OK] Added 300 embedding columns in 0.16s.
    [18:25:24] [INFO] Final features used: ['lemma', 'POS', 'log_freq', 'zipf_value', 'len_chars', 'syllables', 'pref_un', 'pref_re', 'pref_in', 'pref_im']...
    [18:25:24] [INFO] Embedding dims: 300
    [18:25:24] [INFO] Starting 5-fold cross-validation ...
    [18:25:24] [INFO] Training model: xgboost
    [18:25:35] [INFO] xgboost CV done in 10.45s → RMSE=0.942 ± 0.018
    
    ===== RESULTS SUMMARY =====
         model  RMSE_mean  RMSE_std
    0  xgboost   0.941928  0.018045
    ===========================
    
    Best model: xgboost (RMSE_mean=0.942 ± 0.018)
    ```

- Setup4
  - Setup 3 + dimension reduction
  - To get Faster training, less overfitting, clearer feature importance
    ```aiignore
    [18:41:34] [INFO] [OK] Generated raw embeddings matrix of shape (15281, 300)
    [18:41:34] [INFO] Applying TruncatedSVD → 50 dimensions ...
    [18:41:37] [INFO] [OK] Reduced embeddings from 300 → 50 dims
    [18:41:37] [INFO] [OK] Added 50 embedding columns in 3.06s.
    [18:41:37] [INFO] Final features used: ['lemma', 'POS', 'log_freq', 'zipf_value', 'len_chars', 'syllables', 'pref_un', 'pref_re', 'pref_in', 'pref_im']...
    [18:41:37] [INFO] Embedding dims: 50
    [18:41:37] [INFO] Starting 5-fold cross-validation ...
    [18:41:37] [INFO] Training model: xgboost
    [18:41:40] [INFO] xgboost CV done in 2.96s → RMSE=0.945 ± 0.017
    
    ===== RESULTS SUMMARY =====
         model  RMSE_mean  RMSE_std
    0  xgboost   0.945035  0.016702
    ===========================
    
    Best model: xgboost (RMSE_mean=0.945 ± 0.017)
    ```

- Setup 5: Tuning XGBoost parameters 

[//]: # (- TODO: use validation set for tuning instead of test set)
  - The current XGBoost model uses fixed defaults like:
  `max_depth=6, learning_rate=0.05, n_estimators=400`
  But depending on your dataset size and feature setup (especially after embedding SVD), the optimal combination may differ.
  Grid search will systematically test combinations and pick the best based on cross-validation RMSE.
  Randomized search for faster exploration (RandomizedSearchCV), it samples combinations randomly.
    ```aiignore
    (python3) ➜  CS6120Project python -m run_pipeline_tune_xgboost
    ...
    [19:05:27] [INFO] [OK] Best RMSE: 0.9409
    [19:05:27] [INFO] [OK] Saved best parameters → results/best_xgboost_params.json
    ```
    
    best params:
    ```aiignore
    {
      "best_rmse": 0.940882883802724,
      "best_params": {
        "model__subsample": 0.7999999999999999,
        "model__reg_lambda": 0.5,
        "model__reg_alpha": 0.1,
        "model__n_estimators": 400,
        "model__max_depth": 6,
        "model__learning_rate": 0.020000000000000004,
        "model__colsample_bytree": 0.7
      }
    }
    ```
  - comparison between initial xgboost before parameter tuning v.s. xgboost after parameter tuning:
    ```aiignore
    ===== RESULTS SUMMARY =====
                 model  RMSE_mean  RMSE_std
    0  xgboost-initial   0.945035  0.016702
    1          xgboost   0.939404  0.018031
    ===========================
    Best model: xgboost (RMSE_mean=0.939 ± 0.018)
    ````

[//]: # (  - ✅ RMSE improved slightly &#40;≈ 0.6 %&#41;, while std remained stable — so your tuned model is:)

[//]: # (    - slightly more accurate, )

[//]: # (    - equally consistent across folds, )

[//]: # (    - and not overfitting &#40;no increase in variance&#41;.)

  - Setup 6: compute accuracy and accuracy within 1 on discrete labels
      - 
```aiignore
===== RESULTS SUMMARY =====
     model  RMSE_mean  RMSE_std  Acc_mean   Acc_std  Acc_within1_mean  Acc_within1_std
0  xgboost   0.939404  0.018031  0.388522  0.007583          0.894576         0.008355
===========================

Best model: xgboost (RMSE_mean=0.939 ± 0.018)
```

