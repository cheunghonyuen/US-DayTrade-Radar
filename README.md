# US DayTrade MVP v0.1

目标：先验证“盘前异动 / Stocks-in-Play + Opening Range Breakout + RVOL + 当天平仓”在真实交易成本后是否仍有正期望。

## 当前版本
- 输入：1分钟美股 OHLCV CSV
- 盘前/昨日收盘 gap 过滤
- Opening Range: 默认 20 分钟（09:30-09:50 ET）
- 入场窗口：默认 10:00-11:30 ET
- 突破确认：价格突破 Opening Range High
- RVOL 过滤：默认 >= 1.2
- 风险控制：固定每笔账户风险 0.5%
- 止损：Opening Range 下方 / range-based stop
- 止盈：默认 1.5R
- EOD 强制平仓：15:50 ET
- 成本：佣金 + 每股滑点
- 输出：trades.csv + metrics.json

## CSV 格式
至少包含：
timestamp,symbol,open,high,low,close,volume

timestamp 建议是美国东部时间；如果不是，请先转换。

## 运行
```bash
python run_backtest.py --data data/your_minute_bars.csv
```

## 注意
这是研究 / Paper Trading 版本，不会自动连接 IBKR，不会发送真实订单。
下一步只有在大样本 OOS / walk-forward 通过后，才接 IBKR Paper。
