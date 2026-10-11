# 时间验证与资金回测：运行说明

此说明保留此前季度 HGB 资金账本扩展的完整运行方法。年度模型的完整换仓对照见[组合回测报告](portfolio_backtest_report.md)，最新固定调仓比较见[调仓研究报告](rebalance_study_report.md)；当前入口与命令见[README](README.md)。

本次升级解决随机跨期验证、评价混入训练样本，以及用重叠五日收益差代替日净值的问题。原交接包和此前因子时点修正、原模型重跑结果全部保留。研究结论见 [升级研究报告](research_upgrade_report.md)。

## 已运行的阶段

| 阶段 | 源码 | 默认结果位置 |
|---|---|---|
| 时间验证与季度样本外预测 | `temporal_model.py` | `outputs/temporal_oos/` |
| 三种预测模型、八个资金账本 | `evaluate_portfolios.py`、`portfolio_ledger.py` | `outputs/portfolio_upgrade/` |
| 从订单独立重算现金、持仓与净值 | `audit_upgrade.py` | `outputs/upgrade_audit.json` |

保留的历史版本：题包 `baseline/`；`outputs/factor_timing/`；`outputs/model_rerun/original/`；`outputs/model_rerun/pit/`。不得将旧随机切分模型在新账本上的结果标成时间样本外策略。

## 数据和环境

数据包为 `Alpha_Forensics_Candidate_v1`，默认相对工作区的位置：

```text
Lab404-CUFEL-E/03-资产配置策略方向/Alpha_Forensics_Candidate_v1/
```

实测 Python 3.12.13、NumPy 2.5.4、pandas 2.3.3、scikit-learn 1.9.1、joblib 1.6.0、matplotlib 3.11.2，使用现有 `.venv`。精确实际版本和线程参数也记录在每次运行清单中。无需 GPU、外部市场数据或模型 API。

## 核对当前保存结果

以下命令从工作区根目录运行。测试只指定本次相关模块，避免混入其他并行研究。

```bash
PYTHONPATH=asset_allocation OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m unittest test_factor_timing test_model_rerun test_temporal_model test_portfolio_ledger -v
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B asset_allocation/audit_upgrade.py
```

审计复核原始文件哈希、旧结果哈希、时间边界、订单时点、成本与成交价、库存、现金、净值、最终状态冲突和绩效指标。它不重新训练模型。

## 在新目录完整重跑升级

前置的原模型和仅修正因子时点模型已保存在 `outputs/model_rerun/`。若这些前置文件缺失，按主 `README.md` 的原模型与 PIT 重跑命令先生成。

将 `AF_REPLAY` 设为一个尚不存在的目录。时间模型和资金回测遇到非空输出目录会报错，防止覆盖已有研究证据。

```bash
AF_REPLAY='asset_allocation/outputs/upgrade_replay'
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B asset_allocation/temporal_model.py --out "$AF_REPLAY/temporal_oos"
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B asset_allocation/evaluate_portfolios.py --temporal "$AF_REPLAY/temporal_oos" --out "$AF_REPLAY/portfolio_upgrade"
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B asset_allocation/audit_upgrade.py --temporal "$AF_REPLAY/temporal_oos" --portfolio "$AF_REPLAY/portfolio_upgrade" --output "$AF_REPLAY/upgrade_audit.json"
```

数据包放在其他位置时，三个脚本都可传 `--root /path/to/Alpha_Forensics_Candidate_v1`；前置旧模型目录可通过前两个脚本的 `--old-root` 指定。默认路径的实际运行日志分别为 `outputs/temporal_oos_run.log`、`outputs/portfolio_upgrade_run.log`、`outputs/upgrade_audit_run.log`。

## 时间模型约定

- 2021 年训练、2022 年时间验证；2023—2024 年每季度开始前重新训练一次扩展窗口模型。
- 训练冻结于预测期首个合成交易日前一日盘后。仅使用 `exit_date <= train_cutoff` 且已经实现的收益标签。
- 保留原 20 个因子及 HGB 的 55 轮、学习率 0.065、17 叶、L2=0.2、种子 1021；关闭随机内部早停，使用全部历史成熟样本，不调参。
- 对信号日所有已上市成员预测。未来标签缺失只影响诊断评价，不影响是否生成交易信号。
- 旧版与新版诊断只在相同日期、相同资产和相同标签的交集中比较，并披露排除数量。

## 资金回测约定

- 做多预测最高十分位，基准为当日成员等权；五份初始等额资金逐日轮换，各自独立再投资。持有五个交易日，t 日盘后决定名单和预算，t+1 开盘入场，正常 t+6 开盘退出。
- 费用采用前一日已知的单边手续费、半价差和冲击代理，按实际买卖金额双边扣除。不存在自动融资，允许分数股，现金无息。
- 同资产旧仓未能卖出时保留原到期日并每日重试，仅跳过该资产的新买单；其他名字使用统一缩额后的原名单分配，未成交份额留现金。
- 各资金份按到期卖出和新周期买入分别记账，不将同名买卖净额合并，成本含这项完整换仓约定。没有进行成交容量估计。
- 可买卖标记采用前一日代理，当日最终标记只用于事后核查。报告单列假设成交与当日最终禁买卖状态冲突的订单。
- 资产首次缺报价时按零回收核销并单列损失；不跨退出填价，不利用未来退出日期提前卖出。
- 预先固定最后入场信号为 2024-12-23、回测终点为 2024-12-31。期末无法卖出的仓位按真实收盘价计值，不虚构平仓。
- 每日收盘净值计算日收益、回撤、波动率和 Sharpe，年化按 252 个合成交易日；初始建仓成本纳入净值。

## 关键结果入口

- `outputs/temporal_oos/comparison.csv`：旧原版、仅 PIT 版、时间 OOS 版的同样本诊断。
- `outputs/temporal_oos/fold_manifest.csv`：每折训练日期、最大标签退出日和预测期的边界证明。
- `outputs/portfolio_upgrade/portfolio_comparison.csv`：八个资金账本的净值指标与匹配基准。
- `outputs/portfolio_upgrade/temporal_oos_net/daily.csv`：主要样本外策略日账本。
- 同目录 `orders.csv.gz`、`positions.csv.gz`：逐笔委托与每日逐仓记录。
- `outputs/portfolio_upgrade/oos_nav_drawdown.png`：样本外净值和回撤图。
- `outputs/upgrade_audit.json`：独立重算及版本保留证明。

本轮为已经看过的历史数据上的逐时点模型样本外回放，不是全新未观察数据的盲测。日度数据无法证明每笔开盘成交可行，净值需连同执行代理和退出估值假设阅读。
