# IROC Stock Viewer

A local Streamlit app for graphing a stock price curve, its ROC / slope, and IROC.

In this project:

- `a(t)` is the adjusted close stock price.
- `ROC / slope` is the first numerical derivative.
- `IROC` is the second numerical derivative.

The app uses Yahoo Finance data through `yfinance`. Calculations are educational and should not be treated as financial advice.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```powershell
streamlit run app.py
```

Then open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

## Ticker examples

- `AAPL`
- `MSFT`
- `BMO.TO`
- `RY.TO`

Canadian tickers generally need the `.TO` suffix for Yahoo Finance.

## Features

- Price graph using adjusted close data.
- Daily and intraday intervals, including 1 minute, 5 minute, 15 minute, 30 minute, 1 hour, and 1 day bars.
- Optional candlestick chart when OHLC data is available.
- ROC / slope graph for the first derivative.
- IROC graph for the second derivative.
- IROC shown in dollar or percent units for the selected time scale.
- Optional derivative axis scaler, such as `x1,000`, for easier reading of very small ROC/IROC values.
- Moving-average smoothing before derivatives.
- Optional smooth visual curve interpolation.
- Optional zero-crossing markers.
- Optional local peak/trough markers.
- Normalized comparison chart for other tickers.
- CSV export for calculated data.
- HTML and PNG chart exports.

## Manual math check

For prices:

```text
[100, 105, 103]
```

The dollar ROC is:

```text
[NaN, 5, -2]
```

The dollar IROC is:

```text
[NaN, NaN, -7]
```

## Intraday and continuous curves

Yahoo Finance data is still sampled market data. A 1-minute interval gives one bar per minute, not a truly continuous function.

The app treats the selected interval as the derivative step:

- `1m` interval: ROC is dollars/minute and IROC is dollars/minute^2.
- `5m` interval: price change is divided by 5 to estimate dollars/minute.
- `1h` interval: ROC is dollars/hour and IROC is dollars/hour^2.
- `1d` interval: ROC is dollars/trading day and IROC is dollars/trading day^2.

The smooth visual curve option makes the graph look continuous with interpolation, but the math is still calculated from the real sampled bars.

The derivative axis scaler is display-only. For example, with `x1,000` enabled, a raw IROC value of `0.00183997` is displayed on the chart as `1.83997`. CSV exports still use the raw calculated values.
