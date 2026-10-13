"""Fetch hourly weather data from the Open-Meteo API (free, no key required).

Covers temperature, humidity, wind speed, and solar irradiance — the core
weather drivers for load forecasting.
"""
import os
import requests
import pandas as pd

LAT, LON = 41.88, -87.63  # Chicago, IL
START, END = "2023-01-01", "2024-12-31"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "weather.csv")


def fetch_weather(lat=LAT, lon=LON, start=START, end=END):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,shortwave_radiation",
        "timezone": "America/Chicago",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    hourly = r.json()["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    return df.rename(columns={
        "time": "timestamp",
        "temperature_2m": "temp_c",
        "relative_humidity_2m": "humidity",
        "wind_speed_10m": "wind_ms",
        "shortwave_radiation": "irradiance_wm2",
    })


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df = fetch_weather()
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} hourly rows -> {OUT}")
