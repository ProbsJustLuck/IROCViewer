import pandas as pd

from iroc_math import compute_derivatives, normalized_performance


def test_manual_derivative_math() -> None:
    data = pd.DataFrame(
        {"Adj Close": [100.0, 105.0, 103.0]},
        index=pd.date_range("2026-01-01", periods=3, freq="D"),
    )

    result = compute_derivatives(
        data,
        smoothing_enabled=False,
        smoothing_window=5,
        unit_label="day",
        step_size=1,
    ).display

    assert pd.isna(result["ROC dollars/day"].iloc[0])
    assert result["ROC dollars/day"].iloc[1] == 5
    assert result["ROC dollars/day"].iloc[2] == -2
    assert pd.isna(result["IROC dollars/day^2"].iloc[0])
    assert pd.isna(result["IROC dollars/day^2"].iloc[1])
    assert result["IROC dollars/day^2"].iloc[2] == -7


def test_intraday_derivative_scales_by_step_size() -> None:
    data = pd.DataFrame(
        {"Adj Close": [100.0, 105.0, 103.0]},
        index=pd.date_range("2026-01-01 09:30", periods=3, freq="5min"),
    )

    computed = compute_derivatives(
        data,
        smoothing_enabled=False,
        smoothing_window=5,
        unit_label="minute",
        step_size=5,
    )
    result = computed.display

    assert computed.roc_dollars_column == "ROC dollars/minute"
    assert computed.iroc_dollars_column == "IROC dollars/minute^2"
    assert result["ROC dollars/minute"].iloc[1] == 1
    assert result["ROC dollars/minute"].iloc[2] == -0.4
    assert round(result["IROC dollars/minute^2"].iloc[2], 4) == -0.28


def test_comparison_uses_first_shared_date() -> None:
    primary = pd.DataFrame(
        {"Adj Close": [100.0, 110.0, 120.0]},
        index=pd.date_range("2026-01-01", periods=3, freq="D"),
    )
    comparison = pd.DataFrame(
        {"Adj Close": [200.0, 220.0]},
        index=pd.date_range("2026-01-02", periods=2, freq="D"),
    )

    result = normalized_performance({"AAA": primary, "BBB": comparison})

    assert result.index[0] == pd.Timestamp("2026-01-02")
    assert result["AAA"].iloc[0] == 0
    assert result["BBB"].iloc[0] == 0
    assert round(result["AAA"].iloc[1], 4) == 9.0909
    assert round(result["BBB"].iloc[1], 4) == 10


if __name__ == "__main__":
    test_manual_derivative_math()
    test_intraday_derivative_scales_by_step_size()
    test_comparison_uses_first_shared_date()
    print("Derivative math test passed.")
