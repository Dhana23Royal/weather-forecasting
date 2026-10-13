"""LightGBM models: point forecast + quantile regression (P10/P50/P90)."""
import os
import lightgbm as lgb
import pandas as pd

IN = os.path.join(os.path.dirname(__file__), "..", "data", "features.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

TARGET = "load_mw"
DROP = {"timestamp", TARGET}


def train(df, quantiles=(0.1, 0.5, 0.9)):
    X = df.drop(columns=list(DROP))
    y = df[TARGET]
    split = int(len(df) * 0.8)
    Xtr, Xte, ytr, yte = X[:split], X[split:], y[:split], y[split:]
    os.makedirs(MODEL_DIR, exist_ok=True)
    models = {}
    # Point forecast
    m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42)
    m.fit(Xtr, ytr)
    models["point"] = m
    # Quantile models
    for q in quantiles:
        mq = lgb.LGBMRegressor(objective="quantile", alpha=q,
                               n_estimators=500, learning_rate=0.05,
                               random_state=42)
        mq.fit(Xtr, ytr)
        models[f"q{int(q*100)}"] = mq
    import pickle
    with open(os.path.join(MODEL_DIR, "models.pkl"), "wb") as f:
        pickle.dump((models, list(X.columns)), f)
    return models, (Xte, yte)


if __name__ == "__main__":
    df = pd.read_csv(IN)
    models, _ = train(df)
    print("Trained:", sorted(models))
