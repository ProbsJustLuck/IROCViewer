from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DerivativeData:
    display: pd.DataFrame
    raw_price_column: str
    analyzed_price_column: str
    roc_dollars_column: str
    roc_percent_column: str
    iroc_dollars_column: str
    iroc_percent_column: str
    unit_label: str
    step_size: float


def flatten_yfinance_columns(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return data

    result = data.copy()
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = [
            next((str(part) for part in column if str(part).lower() != "nan"), "")
            for column in result.columns.to_flat_index()
        ]
    result.index = pd.to_datetime(result.index)
    return result


def select_price_column(data: pd.DataFrame) -> str:
    for column in ("Adj Close", "Close"):
        if column in data.columns and data[column].dropna().shape[0] > 0:
            return column
    raise ValueError("No adjusted close or close price column was returned.")


def moving_average(series: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def compute_derivatives(
    data: pd.DataFrame,
    smoothing_enabled: bool,
    smoothing_window: int,
    unit_label: str = "day",
    step_size: float = 1.0,
) -> DerivativeData:
    if step_size <= 0:
        raise ValueError("Derivative step size must be greater than zero.")

    price_column = select_price_column(data)
    display = data.copy()
    display["Price"] = pd.to_numeric(display[price_column], errors="coerce")
    display = display.dropna(subset=["Price"]).sort_index()

    if smoothing_enabled:
        display["Analyzed Price"] = moving_average(display["Price"], smoothing_window)
    else:
        display["Analyzed Price"] = display["Price"]

    roc_dollars_column = f"ROC dollars/{unit_label}"
    roc_percent_column = f"ROC percent/{unit_label}"
    iroc_dollars_column = f"IROC dollars/{unit_label}^2"
    iroc_percent_column = f"IROC percent/{unit_label}^2"

    display[roc_dollars_column] = display["Analyzed Price"].diff() / step_size
    display[roc_percent_column] = (
        display["Analyzed Price"].pct_change(fill_method=None) * 100 / step_size
    )
    display[iroc_dollars_column] = display[roc_dollars_column].diff() / step_size
    display[iroc_percent_column] = display[roc_percent_column].diff() / step_size

    return DerivativeData(
        display=display,
        raw_price_column=price_column,
        analyzed_price_column="Analyzed Price",
        roc_dollars_column=roc_dollars_column,
        roc_percent_column=roc_percent_column,
        iroc_dollars_column=iroc_dollars_column,
        iroc_percent_column=iroc_percent_column,
        unit_label=unit_label,
        step_size=step_size,
    )


def normalized_performance(data_by_ticker: dict[str, pd.DataFrame]) -> pd.DataFrame:
    prices = {}
    for ticker, data in data_by_ticker.items():
        if data.empty:
            continue
        try:
            column = select_price_column(data)
        except ValueError:
            continue
        price = pd.to_numeric(data[column], errors="coerce").dropna()
        if price.empty:
            continue
        prices[ticker] = price

    aligned = pd.DataFrame(prices).dropna(how="any")
    if aligned.empty:
        return aligned
    return (aligned / aligned.iloc[0] - 1) * 100


def zero_crossings(series: pd.Series) -> pd.Series:
    clean = series.dropna()
    if clean.empty:
        return clean
    signs = np.sign(clean)
    crossings = signs.ne(signs.shift(1)) & signs.ne(0) & signs.shift(1).notna()
    return clean[crossings]


def local_extrema(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    clean = series.dropna()
    if len(clean) < 3:
        return clean.iloc[0:0], clean.iloc[0:0]

    previous_values = clean.shift(1)
    next_values = clean.shift(-1)
    peaks = clean[(clean > previous_values) & (clean > next_values)]
    troughs = clean[(clean < previous_values) & (clean < next_values)]
    return peaks, troughs
