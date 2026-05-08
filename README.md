# THU-BDC2026 HS300 Weekly Top-5 Strategy

本仓库实现了一个可配置的沪深 300 周度选股策略：每周从 300 只股票中挑选预期收益最高的 5 只股票，并输出可自定义权重下的组合结果。

## 策略思路

策略入口为 `code/run_strategy.py`，核心实现为 `code/src/weekly_top5_strategy.py`。

默认使用 `predictive` 模式，只使用目标周之前的数据构造信号，避免未来函数：

- `momentum_5`：最近 5 个交易日动量；
- `momentum_20`：最近 20 个交易日动量；
- `volume_trend_5`：最近 5 日成交量相对 20 日均量的变化；
- `volatility_20`：最近 20 日收益率波动率，默认负权重惩罚高波动；
- `ma_gap_20`：收盘价相对 20 日均线的偏离，默认负权重避免过度追高；
- `last_return`：上一交易日涨跌幅。

如果已经拥有目标周真实行情，也可以使用 `oracle` 模式按该周真实开盘到收盘收益排序，用于复盘“本周最优收益的五支股票”。

## 快速运行

```bash
python code/run_strategy.py \
  --input data/test.csv \
  --output output/weekly_top5.csv \
  --target-week-start 2026-03-09
```

输出文件：

- `output/weekly_top5.csv`：5 只股票、目标周、得分、组合权重、预期收益与核心特征；
- `output/result.csv`：兼容截图中 `Team Name,Final Score` 形式的汇总分数文件。

## 自定义权重

可以通过 JSON 或 `key=value` 形式覆盖特征权重：

```bash
python code/run_strategy.py \
  --input data/test.csv \
  --weights 'momentum_5=0.50,momentum_20=0.25,volume_trend_5=0.10,volatility_20=-0.20,ma_gap_20=-0.05'
```

也可以复盘真实最优周收益：

```bash
python code/run_strategy.py \
  --input data/test.csv \
  --mode oracle \
  --target-week-start 2026-03-09
```

## 数据格式

支持基线截图里的中文列名，也支持常见英文别名。必需字段：

| 中文列名 | 英文字段 | 说明 |
| --- | --- | --- |
| 股票代码 | `stock_code` | 股票代码，会自动补齐 6 位 |
| 日期 | `date` | 交易日期 |
| 开盘 | `open` | 开盘价 |
| 收盘 | `close` | 收盘价 |

可选字段包括 `最高/最低/成交量/成交额/涨跌幅`。

## 本地检查

```bash
python -m pytest
```
