# Weather-Based Load Forecasting

A personal project exploring short-term electricity load forecasting using
weather data — built out of interest in renewable energy analytics.

## Approach

- **Data**: Hourly weather variables (temperature, humidity, wind speed, solar
  irradiance) from the [Open-Meteo](https://open-meteo.com/) API (free, no key
  required), plus a synthetic load series modeled on temperature-driven demand
  patterns (heating/cooling degree effects).
- **Features**: Lagged load, rolling means, calendar features (hour, weekday,
  month), heating/cooling degree hours, and weather variables.
- **Models**: LightGBM regressors — point forecasts plus quantile regression
  (P10/P50/P90) for probabilistic forecasting.
- **Validation**: Walk-forward backtesting across seasons; metrics include MAE,
  RMSE, normalized RMSE, bias, and forecast skill vs. a persistence baseline.

## Project structure

```
├── README.md
├── requirements.txt
├── src/
│   ├── data_loader.py   # Fetch weather data from Open-Meteo
│   ├── features.py      # Feature engineering
│   ├── models.py        # LightGBM point + quantile models
│   └── evaluate.py      # Backtesting and metrics
├── notebooks/
│   └── exploration.ipynb  # EDA starter (optional)
└── data/                # Cached downloads (gitignored)
```

## Quickstart

```bash
pip install -r requirements.txt
python -m src.data_loader     # downloads weather data -> data/weather.csv
python -m src.features        # builds features -> data/features.csv
python -m src.models          # trains LightGBM + quantile models
python -m src.evaluate        # walk-forward backtest, prints MAE/RMSE/skill
```

## Results (sample run)

| Model | MAE | RMSE | Skill vs persistence |
|-------|-----|------|----------------------|
| LightGBM (P50) | — | — | — |
| Persistence baseline | — | — | 0.00 |

Run `src/evaluate.py` to populate with your own numbers.

## Notes

- The load series here is synthetic (temperature-driven) for demonstration;
  swap in real SCADA/telemetry load data where available.
- Extendable to wind/solar generation forecasting by adding NWP provider
  feeds and turbine/panel physics features.
