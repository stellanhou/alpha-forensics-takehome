# Alpha Forensics｜候选人数据与原始研究交接包（合成数据）

> 本文件仅解释数据及已有研究产物；正式任务范围、时间、提交规则，以实验室另行发布的 Take-home 试题为准。
> 所有资产、行业、行情、因子、交易条件均为**仿真构造**，不代表真实 A 股或任何金融市场。

## 包内内容

- `data/market_daily.csv.gz`：日度 OHLCV，未模拟拆分复权（`adj_factor` 恒为 1）。
- `data/factors.csv.gz`：20 个匿名候选因子的供应商研究快照；**不保证所有值在其 reference date 当天可用**。
- `data/risk_exposure.csv.gz`：行业、市值、风险代理与流动性代理。
- `data/universe.csv.gz`：历史上市状态、当期候选投资范围标识。
- `data/trading_constraints.csv.gz`：每日交易约束及当期交易成本代理。
- `data/factor_availability.csv.gz`：特殊发布时间字段。参考日期为 `date`，对应列为该值在**所列日期收盘后**最早可以观察到的日期；未单独列出的因子默认在 `date` 收盘后可观察。
- `data/data_dictionary.md`：所有字段、时点约定及约束。
- `baseline/original_research_report.md`：前任分析师交接研究报告，尚未完成独立复核。
- `baseline/baseline_results.csv`、`baseline/baseline_backtest.csv`、`baseline/baseline_predictions.csv.gz`：前任流程保存的结果。
- `baseline/baseline_code/`：前任流程源代码（仅作为交接材料；不存在正确性保证）。

## 统一交易与数据约定

`date` 表示某一**合成交易日**；日历是所有工作日，不考虑真实交易所节假日。盘后可得的信号最早用于**下一交易日开盘**。例如，观测 t 日收盘后的信号，5 日 forward target 可定义为 `open(t+6)/open(t+1)-1`；目标、建模和验证方式仍由研究者决定。日期内当日实际最终成交额、停牌与涨跌停状态属于盘后已知事实，不应该当成该日开盘前已知输入。交易成本代理也是 t 日收盘后可获得，若用于 t+1 交易需要明确说明假设。

动态上市与股票池可能造成样本不完整。某些股票在局部期间才上市或退出。若计算 t+6 标签，应确认标签终点确实存在，不得跨退市日期填补价格。

## 交接项目如何运行

建议 Python 3.10+；仅需 `numpy`、`pandas`、`scikit-learn`。从本包根目录运行：

```bash
python baseline/baseline_code/train_model.py --root . --out baseline
python baseline/baseline_code/backtest.py --root . --out baseline
```

以上命令会重建前任流程的研究结果（不代表方法可信）。CSV.gz 可被 pandas 原生直接读取，无需额外 Parquet 引擎。本包另有 `tools/optional_csv_to_parquet.py` 仅用于有 `pyarrow` 的环境自选格式转换。

## 权利与用途

全部数据为实验室内部招聘用途的合成示例。请勿将报告中的指标解释为真实金融投资表现。
