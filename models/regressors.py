from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor


def build_preprocessor(num_cols, cat_cols):
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])
    return pre


def get_models(num_cols, cat_cols, random_state=42):
    pre = build_preprocessor(num_cols, cat_cols)

    models = {
        # "elasticnet": Pipeline([
        #     ("pre", pre),
        #     ("model", ElasticNet(alpha=0.05, l1_ratio=0.2, random_state=random_state))
        # ]),
        # "xgboost-initial": Pipeline([ # initial xgboost without parameter tuning
        #     ("pre", pre),
        #     ("model", XGBRegressor(
        #         n_estimators=400,      # number of boosted trees
        #         max_depth=6,           # depth of each tree
        #         learning_rate=0.05,    # step size shrinkage
        #         subsample=0.8,         # row sampling
        #         colsample_bytree=0.8,  # column sampling
        #         n_jobs=-1,             # use all CPU cores
        #         random_state=random_state,
        #         reg_lambda=1.0,        # L2 regularization
        #         reg_alpha=0.0,         # L1 regularization
        #         tree_method="hist",    # fast histogram-based algorithm
        #         objective="reg:squarederror",
        #         verbosity=1
        #     ))
        # ]),
        "xgboost": Pipeline([ # best xgboost after parameter tuning
            ("pre", pre),
            ("model", XGBRegressor(
                n_estimators=400,      # number of boosted trees
                max_depth=6,           # depth of each tree
                learning_rate=0.02,    # step size shrinkage
                subsample=0.8,         # row sampling
                colsample_bytree=0.7,  # column sampling
                n_jobs=-1,             # use all CPU cores
                random_state=random_state,
                reg_lambda=0.5,        # L2 regularization
                reg_alpha=0.1,         # L1 regularization
                tree_method="hist",    # fast histogram-based algorithm
                objective="reg:squarederror",
                verbosity=1
            ))
        ])
    }
    return models
