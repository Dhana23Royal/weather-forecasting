"""Feature engineering for load forecasting.

Builds a synthetic temperature-driven load series (heating/cooling degree
effects) and engineers lag, rolling, calendar, and weather features.
Swap `make_load` with real SCADA/telemetry load data when available.
"""
import os
import numpy as np
import pandas as pd

IN = os.path.join(os.path.dirname(__file__), "..", "data", "weather.csv")
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "features.csv")


def make_load(df):
    """Synthetic hourly load (MW) driven by temperature + calendar + noise."""
    temp = df["temp_c"].values
    # Heating below 18C, cooling above 22C (degree-hour effects)
    heating = np.clip(18 - temp, 0, None)
    cooling = np.clip(temp - 22, 0, None)
    hour = df["timestamp"].dt.hour.values
    dow = df["timestamp"].dt.dayofweek.values
    daily = 1 + 0.15 * np.sin(2 * np.pi * (hour - 14) / 24)      # afternoon peak
    weekly = 1 - 0.12 * (dow >= 5)                               # weekend dip
    base = 1000.0
    load = base * daily * weekly + 25 * heating + 35 * cooling
    rng = np.random.default_rng(42)
    load = load * (1 + rng.normal(0, 0.02, len(load)))
    return pd.Series(load, index=df.index, name="load_mw")


def engineer(df):
    df = df.copy()
    df["load_mw"] = make_load(df)
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["heat_deg_hr"] = (18 - df["temp_c"]).clip(lower=0)
    df["cool_deg_hr"] = (df["temp_c"] - 22).clip(lower=0)
    df["heat_index"] = df["temp_c"] + 0.1 * df["humidity"]
    for lag in (1, 24, 168):
        df[f"load_lag_{lag}"] = df["load_mw"].shift(lag)
    df["load_roll24"] = df["load_mw"].shift(1).rolling(24).mean()
    df["temp_roll24"] = df["temp_c"].shift(1).rolling(24).mean()
    return df.dropna().reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_csv(IN, parse_dates=["timestamp"])
    feats = engineer(df)
    feats.to_csv(OUT, index=False)
    print(f"Saved {len(feats)} rows x {feats.shape[1]} cols -> {OUT}")
