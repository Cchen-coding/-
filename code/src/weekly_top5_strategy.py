"""Weekly top-5 HS300 stock selection strategy.

The module works with the Chinese-column CSV files used by the THU-BDC2026
baseline screenshots (``股票代码``, ``日期``, ``开盘``, ``收盘``, ``最高``, ``最低``,
``成交量``).  It can also read equivalent English column names.

Two ranking modes are available:

* ``predictive`` (default): rank stocks for the target week using information
  available before that week.  This is the mode to use for a real strategy.
* ``oracle``: rank by the realised return inside the target week when those
  prices already exist in the CSV.  This is useful for local evaluation and for
  answering "which five stocks had the best return this week?" retrospectively.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

COLUMN_ALIASES: Mapping[str, Sequence[str]] = {
    "stock_code": ("股票代码", "stock_code", "code", "symbol", "ticker"),
    "date": ("日期", "date", "trade_date", "datetime"),
    "open": ("开盘", "open", "Open"),
    "close": ("收盘", "close", "Close"),
    "high": ("最高", "high", "High"),
    "low": ("最低", "low", "Low"),
    "volume": ("成交量", "volume", "vol", "Volume"),
    "amount": ("成交额", "amount", "turnover"),
    "pct_change": ("涨跌幅", "pct_change", "return"),
}

DEFAULT_WEIGHTS: Mapping[str, float] = {
    # Recent trend is the main signal for one-week holding periods.
    "momentum_5": 0.34,
    "momentum_20": 0.22,
    # Prefer stocks with improving trading activity.
    "volume_trend_5": 0.14,
    # Penalise unstable names and those too extended above the moving average.
    "volatility_20": -0.18,
    "ma_gap_20": -0.08,
    # Mild reward for positive last-day close-to-close return.
    "last_return": 0.04,
}


@dataclass(frozen=True)
class StrategyConfig:
    """Configuration for :class:`WeeklyTop5Strategy`."""

    top_k: int = 5
    weights: Mapping[str, float] | None = None
    mode: str = "predictive"
    target_week_start: str | None = None
    rebalance_weekday: str = "W-MON"


class WeeklyTop5Strategy:
    """Select the top ``k`` HS300 stocks for a weekly portfolio.

    Parameters
    ----------
    config:
        Strategy configuration.  ``weights`` may include any feature returned by
        :meth:`build_features`; missing weights default to zero.
    """

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()
        if self.config.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.config.mode not in {"predictive", "oracle"}:
            raise ValueError("mode must be 'predictive' or 'oracle'")
        self.weights = dict(DEFAULT_WEIGHTS)
        if self.config.weights:
            self.weights.update(self.config.weights)

    def select(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Return selected stocks with scores, weights, and expected returns."""

        df = normalize_columns(raw)
        week_start = resolve_target_week_start(df, self.config.target_week_start)
        features = self.build_features(df, week_start)
        if features.empty:
            raise ValueError("No stock has enough history before the target week")

        if self.config.mode == "oracle":
            features = attach_realised_week_return(df, features, week_start)
            features["score"] = features["realised_week_return"]
            features["expected_return"] = features["realised_week_return"]
        else:
            features["score"] = weighted_zscore(features, self.weights)
            features["expected_return"] = estimate_expected_return(features)

        selected = (
            features.sort_values(["score", "stock_code"], ascending=[False, True])
            .head(self.config.top_k)
            .reset_index(drop=True)
        )
        selected["weight"] = normalise_portfolio_weights(selected["score"].to_numpy())
        selected.insert(1, "target_week_start", week_start.date().isoformat())
        return selected[
            [
                "stock_code",
                "target_week_start",
                "score",
                "weight",
                "expected_return",
                "momentum_5",
                "momentum_20",
                "volatility_20",
                "volume_trend_5",
                "ma_gap_20",
                "last_return",
            ]
        ]

    def build_features(self, df: pd.DataFrame, week_start: pd.Timestamp) -> pd.DataFrame:
        """Build stock-level features using only data before ``week_start``."""

        history = df[df["date"] < week_start].copy()
        rows: list[dict[str, float | str]] = []
        for stock_code, group in history.groupby("stock_code", sort=True):
            group = group.sort_values("date")
            if len(group) < 21:
                continue
            close = group["close"].astype(float)
            volume = group.get("volume", pd.Series(np.nan, index=group.index)).astype(float)
            returns = close.pct_change()
            ma20 = close.rolling(20).mean().iloc[-1]
            last_close = close.iloc[-1]
            rows.append(
                {
                    "stock_code": str(stock_code),
                    "last_date": group["date"].iloc[-1],
                    "last_close": last_close,
                    "momentum_5": close.iloc[-1] / close.iloc[-6] - 1.0,
                    "momentum_20": close.iloc[-1] / close.iloc[-21] - 1.0,
                    "volatility_20": returns.tail(20).std(ddof=0),
                    "volume_trend_5": safe_ratio(volume.tail(5).mean(), volume.tail(20).mean()) - 1.0,
                    "ma_gap_20": safe_ratio(last_close, ma20) - 1.0,
                    "last_return": returns.iloc[-1],
                }
            )
        return pd.DataFrame(rows)


def normalize_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename known Chinese/English columns and validate the required schema."""

    rename: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in raw.columns:
                rename[alias] = canonical
                break
    df = raw.rename(columns=rename).copy()
    required = {"stock_code", "date", "open", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["stock_code"] = df["stock_code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "close", "high", "low", "volume", "amount", "pct_change"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["stock_code", "date"]).reset_index(drop=True)


def resolve_target_week_start(df: pd.DataFrame, requested: str | None) -> pd.Timestamp:
    """Resolve the Monday-like target week start used for ranking."""

    if requested:
        return pd.Timestamp(requested).normalize()
    last_date = df["date"].max().normalize()
    return (last_date - pd.Timedelta(days=int(last_date.weekday()))).normalize()


def attach_realised_week_return(
    df: pd.DataFrame, features: pd.DataFrame, week_start: pd.Timestamp
) -> pd.DataFrame:
    """Add realised open-to-close return for the target trading week."""

    week_end = week_start + pd.Timedelta(days=7)
    week = df[(df["date"] >= week_start) & (df["date"] < week_end)].copy()
    realised: list[dict[str, float | str]] = []
    for stock_code, group in week.groupby("stock_code", sort=True):
        group = group.sort_values("date")
        if group.empty:
            continue
        first_open = group["open"].iloc[0]
        last_close = group["close"].iloc[-1]
        realised.append(
            {
                "stock_code": str(stock_code),
                "realised_week_return": safe_ratio(last_close, first_open) - 1.0,
            }
        )
    realised_df = pd.DataFrame(realised)
    if realised_df.empty:
        raise ValueError("No realised prices are available inside the target week")
    return features.merge(realised_df, on="stock_code", how="inner")


def weighted_zscore(features: pd.DataFrame, weights: Mapping[str, float]) -> pd.Series:
    """Calculate a weighted cross-sectional z-score."""

    score = pd.Series(0.0, index=features.index)
    for feature_name, weight in weights.items():
        if feature_name not in features.columns or weight == 0:
            continue
        values = features[feature_name].replace([np.inf, -np.inf], np.nan).astype(float)
        std = values.std(ddof=0)
        if pd.isna(std) or std == 0:
            z = values.fillna(values.median()).sub(values.median())
        else:
            z = (values - values.mean()) / std
        score = score.add(weight * z.fillna(0.0), fill_value=0.0)
    return score


def estimate_expected_return(features: pd.DataFrame) -> pd.Series:
    """Conservative expected weekly return proxy from historical features."""

    signal = (
        0.55 * features["momentum_5"].astype(float)
        + 0.30 * features["momentum_20"].astype(float) / 4.0
        + 0.15 * features["last_return"].astype(float)
        - 0.20 * features["volatility_20"].astype(float)
    )
    return signal.clip(lower=-0.20, upper=0.20)


def normalise_portfolio_weights(scores: np.ndarray) -> np.ndarray:
    """Convert selection scores into long-only portfolio weights."""

    shifted = scores - np.nanmin(scores)
    if not np.isfinite(shifted).all() or shifted.sum() <= 1e-12:
        return np.repeat(1.0 / len(scores), len(scores))
    weights = shifted + 1e-6
    return weights / weights.sum()


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return a finite ratio or ``nan`` when the denominator is unusable."""

    if pd.isna(denominator) or denominator == 0:
        return np.nan
    return numerator / denominator


def parse_weight_overrides(value: str | None) -> dict[str, float] | None:
    """Parse weights from JSON (``{"momentum_5": 0.5}``) or ``a=1,b=2``."""

    if not value:
        return None
    value = value.strip()
    if value.startswith("{"):
        parsed = json.loads(value)
        return {str(k): float(v) for k, v in parsed.items()}
    overrides: dict[str, float] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        key, raw_weight = item.split("=", 1)
        overrides[key.strip()] = float(raw_weight)
    return overrides


def save_selection(selection: pd.DataFrame, output_path: Path) -> None:
    """Persist both detailed picks and a competition-style score file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    selection.to_csv(output_path, index=False)
    final_score = float((selection["weight"] * selection["expected_return"]).sum())
    score_path = output_path.with_name("result.csv")
    pd.DataFrame({"Team Name": [1], "Final Score": [final_score]}).to_csv(score_path, index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select weekly top-5 HS300 stocks")
    parser.add_argument("--input", default="data/test.csv", help="Input OHLCV CSV path")
    parser.add_argument("--output", default="output/weekly_top5.csv", help="Detailed output CSV path")
    parser.add_argument("--top-k", type=int, default=5, help="Number of stocks to select")
    parser.add_argument("--mode", choices=["predictive", "oracle"], default="predictive")
    parser.add_argument("--target-week-start", help="Target week start date, e.g. 2026-03-09")
    parser.add_argument(
        "--weights",
        help="Custom feature weights as JSON or comma list, e.g. momentum_5=0.5,volatility_20=-0.2",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    raw = pd.read_csv(args.input)
    strategy = WeeklyTop5Strategy(
        StrategyConfig(
            top_k=args.top_k,
            weights=parse_weight_overrides(args.weights),
            mode=args.mode,
            target_week_start=args.target_week_start,
        )
    )
    selection = strategy.select(raw)
    save_selection(selection, Path(args.output))
    print(selection.to_string(index=False))
    print(f"Detailed picks written to {args.output}")
    print(f"Competition score written to {Path(args.output).with_name('result.csv')}")


if __name__ == "__main__":
    main()
