"""
Task1.py

Loads a monthly CSV of month-end natural gas prices (Dates, Prices),
fits Holt-Winters (additive trend + additive seasonality), forecasts 12 months,
and provides an estimate_price(date) function to get a price for any date
between first observed month-end and last forecast month-end.

Dependencies:
    pip install pandas numpy matplotlib statsmodels

CSV format expected:
    - Two columns: a date column (e.g. "Dates") and a numeric price column (e.g. "Prices").
    - Dates should correspond to month-end purchase prices (but the script will coerce to month-end).
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
from datetime import datetime
import sys

CSV_PATH = '/Users/arinayare/QR JPMC /Nat_Gas_Example(task 1&2).csv'   # <-- change this path to your CSV file
FORECAST_MONTHS = 12                 # extend one year into the future
SEASONAL_PERIODS = 12                # monthly series seasonality (12 months)

def load_monthly_csv(csv_path):
    """Load CSV and try to detect date/price columns automatically."""
    df = pd.read_csv(csv_path)
    # detect likely columns
    cols = df.columns.tolist()
    date_col = None
    price_col = None
    for c in cols:
        low = c.lower()
        if any(k in low for k in ("date", "time", "day", "month")):
            date_col = c
        if any(k in low for k in ("price", "value", "close", "usd")):
            price_col = c
    if date_col is None:
        date_col = cols[0]
    if price_col is None:
        # pick a numeric column that's not the date
        numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric) > 0:
            # pick first numeric that is not date
            for c in numeric:
                if c != date_col:
                    price_col = c
                    break
        else:
            # fallback to second column
            price_col = cols[1] if len(cols) > 1 else cols[0]

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df = df[[date_col, price_col]].rename(columns={date_col: "date", price_col: "price"})
    # coerce to numeric
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    # coerce dates to month-end
    df['month_end'] = df['date'] + pd.offsets.MonthEnd(0)
    df = df.set_index('month_end').asfreq('M')  # monthly freq with month-end index
    return df[['price']]

def fill_missing(series):
    """Interpolate missing monthly values (time interpolation)."""
    if series['price'].isna().sum() > 0:
        series['price'] = series['price'].interpolate(method='time')
    return series

def fit_and_forecast(series, seasonal_periods=12, forecast_months=12):
    """Fit Holt-Winters additive model and forecast forward forecast_months."""
    y = series['price']
    # In case series has only a few points, handle gracefully
    if len(y.dropna()) < seasonal_periods + 2:
        raise ValueError("Not enough data points to fit seasonal model. Need at least seasonal_periods+2 points.")
    hw = ExponentialSmoothing(y, trend="add", seasonal="add", seasonal_periods=seasonal_periods).fit(optimized=True)
    fitted = hw.fittedvalues
    # forecast month-end steps
    forecast_index = pd.date_range(start=series.index[-1] + pd.offsets.MonthBegin(1),
                                   periods=forecast_months, freq='M')
    forecast_vals = hw.forecast(forecast_months)
    forecast = pd.Series(forecast_vals.values, index=forecast_index)
    # Combine observed month-end series with forecast month-end values
    combined = pd.concat([y, forecast])
    combined.name = 'price'
    return hw, fitted, forecast, combined

def plot_results(series, fitted, forecast):
    """Plot observed, fitted, forecast, monthly seasonality, and residuals."""
    # Observed series
    plt.figure(figsize=(10,4))
    plt.plot(series.index, series['price'], label='Observed', marker='o')
    plt.plot(fitted.index, fitted.values, label='Fitted (in-sample)', linestyle='-', alpha=0.8)
    plt.plot(forecast.index, forecast.values, label='Forecast', linestyle='--', marker='o')
    plt.title("Monthly Natural Gas Price: Observed, Fitted and Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Seasonal profile
    tmp = series.copy()
    tmp['month'] = tmp.index.month
    monthly_avg = tmp.groupby('month')['price'].mean()
    plt.figure(figsize=(8,3))
    plt.plot(monthly_avg.index, monthly_avg.values, marker='o')
    plt.title("Average Price by Month-of-Year (Seasonal Profile)")
    plt.xlabel("Month")
    plt.xticks(range(1,13))
    plt.ylabel("Average Price")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Residuals
    resid = series['price'] - fitted
    plt.figure(figsize=(10,3))
    plt.plot(resid.index, resid.values, marker='.', linestyle='-')
    plt.axhline(0, linestyle='--', color='k', alpha=0.6)
    plt.title("Residuals (Observed - Fitted)")
    plt.xlabel("Date")
    plt.ylabel("Residual")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def make_estimator(combined_monthend_series):
    """
    Returns a function estimate_price(date_input)
    - date_input: date str or datetime
    - linearly interpolates between month-end prices (observed and forecasted)
    - raises ValueError if date outside [first_month_end, last_month_end].
    """
    combined = combined_monthend_series.copy()
    combined.index.name = 'month_end'
    def estimate_price(date_input):
        dt = pd.to_datetime(date_input).normalize()
        first = combined.index[0]
        last = combined.index[-1]
        if dt < first:
            raise ValueError(f"Date {dt.date()} is before first available month-end {first.date()}.")
        if dt > last:
            raise ValueError(f"Date {dt.date()} is after forecast horizon {last.date()}.")
        # If exact month-end
        if dt in combined.index:
            return float(combined.loc[dt])
        # find previous and next month-end
        prev_me = combined.index[combined.index <= dt].max()
        next_me = combined.index[combined.index >= dt].min()
        if pd.isna(prev_me) or pd.isna(next_me):
            raise ValueError("Unable to interpolate for date: " + str(dt))
        if prev_me == next_me:
            return float(combined.loc[prev_me])
        # linear interpolation in time
        v0 = combined.loc[prev_me]
        v1 = combined.loc[next_me]
        frac = (dt - prev_me) / (next_me - prev_me)
        return float(v0 + frac * (v1 - v0))
    return estimate_price

def save_combined_csv(combined_series, out_path):
    df_out = combined_series.rename_axis('month_end').reset_index().rename(columns={'price': 'price_month_end'})
    df_out.to_csv(out_path, index=False)
    print(f"Saved combined month-end (observed + forecast) to: {out_path}")

def main(csv_path=CSV_PATH, forecast_months=FORECAST_MONTHS):
    # 1) load
    series = load_monthly_csv(csv_path)
    print("Loaded series from:", csv_path)
    print("First / Last observed rows:")
    print(series.dropna().head(3))
    print(series.dropna().tail(3))

    # 2) fill small gaps (linear time interpolation)
    series = fill_missing(series)

    # 3) fit model & forecast
    hw_model, fitted, forecast, combined = fit_and_forecast(series, seasonal_periods=SEASONAL_PERIODS,
                                                           forecast_months=forecast_months)
    print("\nForecast horizon end:", combined.index[-1].date())

    # 4) plots
    plot_results(series, fitted, forecast)

    # 5) create estimator
    estimator = make_estimator(combined)

    # Example usage
    examples = ["2022-01-15", str(combined.index[-1].date()), "2024-10-15"]
    print("\nExample price estimates:")
    for ex in examples:
        try:
            p = estimator(ex)
            print(f"  {ex} -> {p:.4f}")
        except Exception as e:
            print(f"  {ex} -> ERROR: {e}")

    # 6) save combined series
    out_path = "Nat_Gas_forecast_combined.csv"
    save_combined_csv(combined, out_path)

    # return estimator for programmatic use
    return estimator, combined

if __name__ == "__main__":
    try:
        estimator_func, combined_series = main()
        # Example: use estimator_func programmatically
        # price_on_date = estimator_func("2025-03-15")
    except Exception as exc:
        print("Error:", exc)
        sys.exit(1)
