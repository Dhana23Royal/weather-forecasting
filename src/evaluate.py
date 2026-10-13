"""Walk-forward backtesting with MAE, RMSE, normalized RMSE, bias,
and forecast skill vs. a persistence baseline.
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

IN = os.path.join(os.path.dirname(__file__), "..", "data", "features.csv")
TARGET = "load_mw"
DROP = {"timestamp", TARGET}


def metrics(y, p):
    mae = mean_absolute_error(y, p)
    rmse = mean_squared_error(y, p) ** 0.5
    return {
        "MAE": mae,
        "RMSE": rmse,
        "nRMSE": rmse / y.mean(),
        "bias": float(np.mean(p - y)),
    }


def backtest(df, n_folds=4, horizon_h=24 * 7):
    X = df.drop(columns=list(DROP))
    y = df[TARGET].values
    fold = len(df) // (n_folds + 1)
    rows = []
    for i in range(n_folds):
        tr_end = fold * (i + 1)
        te_end = min(tr_end + horizon_h, len(df))
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                              random_state=42)
        m.fit(X[:tr_end], y[:tr_end])
        pred = m.predict(X[tr_end:te_end])
        pers = y[tr_end - 24:te_end - 24][: len(pred)]  # 24h persistence
        r = metrics(y[tr_end:te_end], pred)
        rb = metrics(y[tr_end:te_end], pers)
        r["fold"] = i
        r["skill"] = 1 - r["RMSE"] / rb["RMSE"]  # skill vs persistence
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_csv(IN)
    res = backtest(df)
    print(res.to_string(index=False))
    print("\nMean skill vs persistence: %.3f" % res["skill"].mean())
