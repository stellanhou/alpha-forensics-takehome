# 数据字典｜Alpha Forensics v1

## 关键约定

- 所有数值、证券及行业均为合成。不复刻实际交易所节假日：所有周一至周五视为合成交易日。
- `date` 与 `asset_id` 联合主键（适用于除另有说明的表外）。`asset_id` 为匿名合成资产 ID。
- `market_daily` 的 OHLCV 是**该日结束后**才完整可见的最终行情。`adj_factor=1`，无拆分、分红、复权问题。
- `factors` 是按**参考日期**索引的研究供应商最新快照，不是天然的逐时点数据。`factor_availability` 单独记录部分数据的有效发布时间；其他因子在参考日期收盘后可观察。所有 `available_on` 也都在**当日收盘后**生效。
- 若在 t 日收盘后形成信号，默认最早在 t+1 日开盘执行。严禁使用 t+1 日收盘后才可得的交易约束来决定 t+1 日开盘能否买入；应解释采用滞后代理或执行时状态假设。
- 20 个匿名因子仅指具有不同统计性质的候选研究字段；不承诺具有预测力或投资收益。
- 部分样本有真实意义上的列表动态（新上市、退市、投资池筛选），相关时点验证由研究者自行处理。
- 缺失数值和极端值可能出现；不保证所有资产每天适合交易。

## 表与字段

| 文件 | 字段 | 含义及单位 |
|---|---|---|
| `market_daily` | `date`, `asset_id` | 合成工作日；匿名资产 |
| `market_daily` | `open`, `high`, `low`, `close` | 合成单位价格；`high >= max(open, close)`，`low <= min(open, close)` |
| `market_daily` | `volume` | 合成股数；非真实交易所成交统计 |
| `market_daily` | `amount` | 合成单位成交额（约为当日收盘价×成交量） |
| `market_daily` | `adj_factor` | 复权占位，恒等于 1 |
| `factors` | `factor_01` ... `factor_20` | 匿名标准化研究快照，可能相关、缺失或发生修订 |
| `risk_exposure` | `industry` | 匿名行业 IND00 ... IND09 |
| `risk_exposure` | `market_cap` | 合成市值（价格×模拟发行股数） |
| `risk_exposure` | `size_z` | 合成市值相关静态风格代理（约标准化） |
| `risk_exposure` | `beta_proxy` | 合成市场 beta 的代理值 |
| `risk_exposure` | `volatility_20d` | 过去最多 20 个合成工作日的收盘收益波动率代理，日度小数 |
| `risk_exposure` | `liquidity_z` | 流动性截面代理值 |
| `universe` | `is_listed`, `is_member` | 是否上市、是否在当期研究候选股票池 |
| `trading_constraints` | `suspended`, `limit_up`, `limit_down` | 当日事后最终状态；不得当作当日开盘前预知条件 |
| `trading_constraints` | `can_buy`, `can_sell` | 当日事后实际交易方向标记；可用于执行后审计；实时交易需要另设信息假设 |
| `trading_constraints` | `fee_bps_one_way` | 当日成本代理：单方向交易手续费（基点） |
| `trading_constraints` | `spread_bps` | 当日完整买卖价差代理（基点） |
| `trading_constraints` | `impact_bps_proxy` | 标准订单规模下的冲击成本代理（基点），非完整冲击函数 |
| `factor_availability` | `date`, `asset_id` | 对应因子快照的**参考日**及资产 |
| `factor_availability` | `factor_03_available_on` | 对应该行快照值的最早可观察日期，盘后 |
| `factor_availability` | `factor_12_available_on` | 对应该行快照值的最早可观察日期，盘后 |
| `factor_availability` | `published_at` | 固定 `after_close`，即当日收盘后 |

## 如需构建独立回测

**不是要求一定采用某个回测框架。** 应对收益标签、特征可用时点、样本切分、换手与成本做明确约定；尤其不能把后来的修订值合并到更早的参考日，并声称那些值当时已知。请保留可验证的资产、时间索引。

## 文件格式

所有 `.csv.gz` 均为 UTF-8 的逗号分隔压缩文本，可用 `pd.read_csv(path)` 读取；日期列为 ISO YYYY-MM-DD。
