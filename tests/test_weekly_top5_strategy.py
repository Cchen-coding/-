from __future__ import annotations

import pandas as pd

from src.weekly_top5_strategy import (
    StrategyConfig,
    WeeklyTop5Strategy,
    normalize_columns,
    parse_weight_overrides,
)


def make_sample_data() -> pd.DataFrame:
    dates = pd.bdate_range("2026-02-02", periods=30)
    rows = []
    weekly_jump = {
        "000001": 1.01,
        "000002": 1.02,
        "000003": 1.03,
        "000004": 1.04,
        "000005": 1.05,
        "000006": 1.06,
    }
    for stock_idx, code in enumerate(weekly_jump, start=1):
        price = 10.0 + stock_idx
        for date in dates:
            if date >= pd.Timestamp("2026-03-09"):
                price *= weekly_jump[code]
            else:
                price *= 1.0 + stock_idx * 0.001
            rows.append(
                {
                    "股票代码": code,
                    "日期": date.strftime("%Y-%m-%d"),
                    "开盘": price * 0.99,
                    "收盘": price,
                    "最高": price * 1.01,
                    "最低": price * 0.98,
                    "成交量": 1_000_000 + stock_idx * 10_000,
                }
            )
    return pd.DataFrame(rows)


def test_normalize_columns_supports_chinese_schema() -> None:
    df = normalize_columns(make_sample_data())

    assert {"stock_code", "date", "open", "close", "volume"}.issubset(df.columns)
    assert df["stock_code"].str.len().eq(6).all()
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_oracle_mode_selects_best_realised_weekly_returns() -> None:
    strategy = WeeklyTop5Strategy(
        StrategyConfig(mode="oracle", target_week_start="2026-03-09", top_k=5)
    )

    selected = strategy.select(make_sample_data())

    assert list(selected["stock_code"]) == ["000006", "000005", "000004", "000003", "000002"]
    assert round(selected["weight"].sum(), 12) == 1.0


def test_predictive_mode_respects_custom_feature_weights() -> None:
    strategy = WeeklyTop5Strategy(
        StrategyConfig(
            mode="predictive",
            target_week_start="2026-03-09",
            top_k=3,
            weights={"momentum_5": 1.0, "momentum_20": 0.0, "volatility_20": 0.0},
        )
    )

    selected = strategy.select(make_sample_data())

    assert len(selected) == 3
    assert selected.iloc[0]["stock_code"] == "000006"
    assert selected["weight"].between(0, 1).all()


def test_parse_weight_overrides_accepts_json_and_csv() -> None:
    assert parse_weight_overrides('{"momentum_5": 0.7}') == {"momentum_5": 0.7}
    assert parse_weight_overrides("momentum_5=0.7, volatility_20=-0.1") == {
        "momentum_5": 0.7,
        "volatility_20": -0.1,
    }
