from __future__ import annotations

from typing import Iterable

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from iroc_math import (
    DerivativeData,
    compute_derivatives,
    flatten_yfinance_columns,
    local_extrema,
    normalized_performance,
    zero_crossings,
)


WATCHLISTS = {
    "None": "",
    "Canadian banks": "BMO.TO, RY.TO, TD.TO, CM.TO, BNS.TO, NA.TO",
    "US mega caps": "AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA",
    "Indexes / ETFs": "SPY, QQQ, DIA, XIU.TO, VFV.TO",
}

PERIOD_PRESETS = {
    "1 day": "1d",
    "5 days": "5d",
    "1 month": "1mo",
    "3 months": "3mo",
    "6 months": "6mo",
    "1 year": "1y",
    "5 years": "5y",
    "Max": "max",
}

INTERVAL_PRESETS = {
    "1 minute": {"interval": "1m", "unit": "minute", "step": 1.0, "periods": ["1d", "5d"]},
    "5 minutes": {"interval": "5m", "unit": "minute", "step": 5.0, "periods": ["1d", "5d", "1mo"]},
    "15 minutes": {"interval": "15m", "unit": "minute", "step": 15.0, "periods": ["1d", "5d", "1mo"]},
    "30 minutes": {"interval": "30m", "unit": "minute", "step": 30.0, "periods": ["1d", "5d", "1mo"]},
    "1 hour": {
        "interval": "1h",
        "unit": "hour",
        "step": 1.0,
        "periods": ["5d", "1mo", "3mo", "6mo", "1y"],
    },
    "1 day": {
        "interval": "1d",
        "unit": "trading day",
        "step": 1.0,
        "periods": ["1mo", "3mo", "6mo", "1y", "5y", "max"],
    },
}

AXIS_SCALE_PRESETS = {
    "Raw values": 1.0,
    "x1,000": 1_000.0,
    "x10,000": 10_000.0,
    "x100,000": 100_000.0,
    "x1,000,000": 1_000_000.0,
}


def normalize_ticker_list(value: str) -> list[str]:
    tickers = [
        item.strip().upper()
        for chunk in value.splitlines()
        for item in chunk.split(",")
        if item.strip()
    ]
    return list(dict.fromkeys(tickers))


def market_rangebreaks(interval: str, hide_market_gaps: bool) -> list[dict[str, object]]:
    if not hide_market_gaps:
        return []

    breaks: list[dict[str, object]] = [{"bounds": ["sat", "mon"]}]
    if interval != "1d":
        breaks.append({"pattern": "hour", "bounds": [16, 9.5]})
    return breaks


def chart_x_values(index: pd.Index, compress_x_axis: bool) -> pd.Index | list[str]:
    if not compress_x_axis:
        return index
    return pd.to_datetime(index).strftime("%b %d, %Y %H:%M").tolist()


@st.cache_data(ttl=60 * 30, show_spinner=False)
def download_history(ticker: str, period: str, interval: str) -> pd.DataFrame:
    data = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=False,
    )
    return flatten_yfinance_columns(data)


@st.cache_data(ttl=60 * 30, show_spinner=False)
def download_multiple(tickers: tuple[str, ...], period: str, interval: str) -> dict[str, pd.DataFrame]:
    return {ticker: download_history(ticker, period, interval) for ticker in tickers}


def make_price_chart(
    ticker: str,
    data: pd.DataFrame,
    smoothing_enabled: bool,
    smoothing_window: int,
    show_candles: bool,
    show_extrema: bool,
    curve_shape: str,
    rangebreaks: list[dict[str, object]],
    compress_x_axis: bool,
) -> go.Figure:
    fig = go.Figure()
    x_values = chart_x_values(data.index, compress_x_axis)

    if show_candles and {"Open", "High", "Low", "Close"}.issubset(data.columns):
        fig.add_trace(
            go.Candlestick(
                x=x_values,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="OHLC",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=data["Price"],
                mode="lines",
                name="Price",
                line={"color": "#2563eb", "width": 2},
                line_shape=curve_shape,
            )
        )

    if smoothing_enabled:
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=data["Analyzed Price"],
                mode="lines",
                name=f"{smoothing_window}-bar analyzed price",
                line={"color": "#f97316", "width": 2},
                line_shape=curve_shape,
            )
        )

    if show_extrema:
        peaks, troughs = local_extrema(data["Analyzed Price"])
        fig.add_trace(
            go.Scatter(
                x=chart_x_values(peaks.index, compress_x_axis),
                y=peaks,
                mode="markers",
                name="Local peaks",
                marker={"color": "#dc2626", "size": 7, "symbol": "triangle-up"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=chart_x_values(troughs.index, compress_x_axis),
                y=troughs,
                mode="markers",
                name="Local troughs",
                marker={"color": "#16a34a", "size": 7, "symbol": "triangle-down"},
            )
        )

    fig.update_layout(
        title=f"{ticker} price - f(x)",
        xaxis_title="Trading bars" if compress_x_axis else "Date/time",
        yaxis_title="Price",
        hovermode="x unified",
        template="plotly_white",
        legend={"orientation": "h", "y": 1.08},
        margin={"l": 8, "r": 8, "t": 64, "b": 8},
    )
    fig.update_xaxes(rangebreaks=rangebreaks, type="category" if compress_x_axis else None)
    return fig


def make_derivative_chart(
    data: pd.DataFrame,
    title: str,
    columns: Iterable[str],
    y_title: str,
    show_zero_crossings: bool,
    curve_shape: str,
    rangebreaks: list[dict[str, object]],
    scale_factor: float,
    scale_label: str,
    compress_x_axis: bool,
) -> go.Figure:
    colors = ["#7c3aed", "#0891b2", "#ea580c"]
    fig = go.Figure()
    x_values = chart_x_values(data.index, compress_x_axis)
    for index, column in enumerate(columns):
        if column not in data.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=data[column] * scale_factor,
                mode="lines",
                name=column,
                line={"color": colors[index % len(colors)], "width": 2},
                line_shape=curve_shape,
                hovertemplate=f"{column}: %{{y:.6f}}<extra></extra>",
            )
        )

        if show_zero_crossings:
            crossings = zero_crossings(data[column]) * scale_factor
            fig.add_trace(
                go.Scatter(
                    x=chart_x_values(crossings.index, compress_x_axis),
                    y=crossings,
                    mode="markers",
                    name=f"{column} zero crossings",
                    marker={"color": "#111827", "size": 6, "symbol": "x"},
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#6b7280")
    fig.update_layout(
        title=title,
        xaxis_title="Trading bars" if compress_x_axis else "Date/time",
        yaxis_title=y_title if scale_factor == 1 else f"{y_title} ({scale_label})",
        hovermode="x unified",
        template="plotly_white",
        legend={"orientation": "h", "y": 1.08},
        margin={"l": 8, "r": 8, "t": 64, "b": 8},
    )
    fig.update_xaxes(rangebreaks=rangebreaks, type="category" if compress_x_axis else None)
    fig.update_yaxes(tickformat=".6f")
    return fig


def make_comparison_chart(
    normalized: pd.DataFrame,
    rangebreaks: list[dict[str, object]],
    compress_x_axis: bool,
) -> go.Figure:
    fig = go.Figure()
    x_values = chart_x_values(normalized.index, compress_x_axis)
    palette = ["#2563eb", "#dc2626", "#16a34a", "#7c3aed", "#ea580c", "#0891b2"]
    for index, column in enumerate(normalized.columns):
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=normalized[column],
                mode="lines",
                name=column,
                line={"color": palette[index % len(palette)], "width": 2},
            )
        )

    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#6b7280")
    fig.update_layout(
        title="Normalized comparison",
        xaxis_title="Trading bars" if compress_x_axis else "Date/time",
        yaxis_title="Performance since start (%)",
        hovermode="x unified",
        template="plotly_white",
        legend={"orientation": "h", "y": 1.08},
        margin={"l": 8, "r": 8, "t": 64, "b": 8},
    )
    fig.update_xaxes(rangebreaks=rangebreaks, type="category" if compress_x_axis else None)
    return fig


def figure_downloads(fig: go.Figure, filename_base: str) -> None:
    html = fig.to_html(include_plotlyjs="cdn")
    st.download_button(
        "Download chart HTML",
        data=html,
        file_name=f"{filename_base}.html",
        mime="text/html",
        use_container_width=True,
        key=f"{filename_base}-html",
    )

    try:
        png_bytes = fig.to_image(format="png", scale=2)
    except Exception:
        st.caption("PNG export needs the kaleido package available in the active Python environment.")
    else:
        st.download_button(
            "Download chart PNG",
            data=png_bytes,
            file_name=f"{filename_base}.png",
            mime="image/png",
            use_container_width=True,
            key=f"{filename_base}-png",
        )


def csv_download(data: pd.DataFrame, ticker: str) -> None:
    export = data.reset_index().rename(columns={"index": "Date"})
    csv = export.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download calculated CSV",
        data=csv,
        file_name=f"{ticker.lower()}_iroc_data.csv",
        mime="text/csv",
        use_container_width=True,
        key=f"{ticker.lower()}-csv",
    )


def metric_value(value: float, suffix: str = "") -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:,.3f}{suffix}"


def render_summary(ticker: str, derivative_data: DerivativeData) -> None:
    data = derivative_data.display
    latest = data.iloc[-1]
    st.subheader(f"{ticker} snapshot")

    cols = st.columns(4)
    cols[0].metric("Latest price", metric_value(latest["Price"]))
    cols[1].metric(
        "ROC",
        metric_value(latest[derivative_data.roc_dollars_column], f" / {derivative_data.unit_label}"),
    )
    cols[2].metric(
        "IROC $",
        metric_value(
            latest[derivative_data.iroc_dollars_column],
            f" / {derivative_data.unit_label}^2",
        ),
    )
    cols[3].metric(
        "IROC %",
        metric_value(
            latest[derivative_data.iroc_percent_column],
            f"% / {derivative_data.unit_label}^2",
        ),
    )

    st.caption(
        f"Price source: {derivative_data.raw_price_column}. "
        "ROC is the first numerical derivative. IROC is the second numerical derivative."
    )


def main() -> None:
    st.set_page_config(page_title="IROC Stock Viewer", layout="wide")
    st.title("IROC Stock Viewer")
    st.caption("Educational stock-curve analysis using Yahoo Finance market data.")

    with st.sidebar:
        st.header("Controls")
        ticker = st.text_input("Primary ticker", value="BMO.TO").strip().upper()
        interval_label = st.selectbox("Data interval", list(INTERVAL_PRESETS.keys()), index=5)
        interval_config = INTERVAL_PRESETS[interval_label]
        allowed_period_values = interval_config["periods"]
        allowed_period_labels = [
            label for label, value in PERIOD_PRESETS.items() if value in allowed_period_values
        ]
        period_label = st.selectbox("Time range", allowed_period_labels, index=0)
        period = PERIOD_PRESETS[period_label]
        interval = str(interval_config["interval"])
        unit_label = str(interval_config["unit"])
        step_size = float(interval_config["step"])

        st.divider()
        smoothing_enabled = st.toggle("Smooth price before derivatives", value=True)
        smoothing_window = st.slider("Smoothing window", min_value=2, max_value=120, value=5)
        axis_scale_label = st.selectbox(
            "Derivative axis scale",
            list(AXIS_SCALE_PRESETS.keys()),
            index=0,
            help="Display-only multiplier for ROC/IROC charts. CSV exports keep raw values.",
        )
        axis_scale_factor = AXIS_SCALE_PRESETS[axis_scale_label]
        iroc_mode = st.radio(
            "IROC display",
            ["Both", f"Dollars/{unit_label}^2", f"Percent/{unit_label}^2"],
            horizontal=False,
        )
        continuous_display = st.toggle("Smooth visual curve", value=False)
        curve_shape = "spline" if continuous_display else "linear"
        hide_market_gaps = st.toggle("Compress non-trading gaps", value=interval != "1d")
        compress_x_axis = hide_market_gaps and interval != "1d"
        rangebreaks = [] if compress_x_axis else market_rangebreaks(interval, hide_market_gaps)

        st.divider()
        show_candles = st.toggle("Candlestick price chart", value=False)
        show_zero_markers = st.toggle("Show derivative zero crossings", value=True)
        show_extrema = st.toggle("Show local peaks/troughs", value=False)

        st.divider()
        watchlist_name = st.selectbox("Comparison preset", list(WATCHLISTS.keys()))
        default_compare = WATCHLISTS[watchlist_name]
        comparison_value = st.text_area(
            "Comparison tickers",
            value=default_compare,
            placeholder="AAPL, MSFT, BMO.TO",
            height=84,
        )

    if not ticker:
        st.warning("Enter a primary ticker to begin.")
        return

    with st.spinner(f"Loading {ticker} from Yahoo Finance..."):
        try:
            history = download_history(ticker, period, interval)
        except Exception as exc:
            st.error(f"Yahoo Finance request failed for {ticker}: {exc}")
            return

    if history.empty:
        st.error(f"No {interval_label.lower()} price data was returned for {ticker}. Check the ticker symbol or choose a shorter time range.")
        return

    try:
        derivative_data = compute_derivatives(
            history,
            smoothing_enabled,
            smoothing_window,
            unit_label=unit_label,
            step_size=step_size,
        )
    except ValueError as exc:
        st.error(str(exc))
        return

    if len(derivative_data.display) < 3:
        st.warning("At least three price points are needed to calculate IROC.")
        return

    render_summary(ticker, derivative_data)

    price_fig = make_price_chart(
        ticker=ticker,
        data=derivative_data.display,
        smoothing_enabled=smoothing_enabled,
        smoothing_window=smoothing_window,
        show_candles=show_candles,
        show_extrema=show_extrema,
        curve_shape=curve_shape,
        rangebreaks=rangebreaks,
        compress_x_axis=compress_x_axis,
    )
    st.plotly_chart(price_fig, use_container_width=True)

    with st.expander("Export price chart"):
        figure_downloads(price_fig, f"{ticker.lower()}_price")

    roc_fig = make_derivative_chart(
        derivative_data.display,
        title="ROC / slope - f'(x)",
        columns=[derivative_data.roc_dollars_column, derivative_data.roc_percent_column],
        y_title="ROC",
        show_zero_crossings=show_zero_markers,
        curve_shape=curve_shape,
        rangebreaks=rangebreaks,
        scale_factor=axis_scale_factor,
        scale_label=axis_scale_label,
        compress_x_axis=compress_x_axis,
    )
    st.plotly_chart(roc_fig, use_container_width=True)

    iroc_columns = {
        "Both": [derivative_data.iroc_dollars_column, derivative_data.iroc_percent_column],
        f"Dollars/{unit_label}^2": [derivative_data.iroc_dollars_column],
        f"Percent/{unit_label}^2": [derivative_data.iroc_percent_column],
    }[iroc_mode]
    iroc_fig = make_derivative_chart(
        derivative_data.display,
        title="IROC - f''(x)",
        columns=iroc_columns,
        y_title="IROC",
        show_zero_crossings=show_zero_markers,
        curve_shape=curve_shape,
        rangebreaks=rangebreaks,
        scale_factor=axis_scale_factor,
        scale_label=axis_scale_label,
        compress_x_axis=compress_x_axis,
    )
    st.plotly_chart(iroc_fig, use_container_width=True)

    export_cols = [
        "Price",
        "Analyzed Price",
        derivative_data.roc_dollars_column,
        derivative_data.roc_percent_column,
        derivative_data.iroc_dollars_column,
        derivative_data.iroc_percent_column,
    ]
    with st.expander("Export calculated data and IROC chart"):
        csv_download(derivative_data.display[export_cols], ticker)
        figure_downloads(iroc_fig, f"{ticker.lower()}_iroc")

    comparison_tickers = [
        item for item in normalize_ticker_list(comparison_value) if item != ticker
    ]
    if comparison_tickers:
        with st.spinner("Loading comparison tickers..."):
            comparison_data = download_multiple(tuple(comparison_tickers), period, interval)

        normalized = normalized_performance({ticker: history, **comparison_data})
        if normalized.empty or normalized.shape[1] < 2:
            st.info("Comparison tickers did not return enough data to draw a comparison chart.")
        else:
            comparison_fig = make_comparison_chart(normalized, rangebreaks, compress_x_axis)
            st.plotly_chart(comparison_fig, use_container_width=True)
            with st.expander("Export comparison chart"):
                figure_downloads(comparison_fig, f"{ticker.lower()}_comparison")


if __name__ == "__main__":
    main()
