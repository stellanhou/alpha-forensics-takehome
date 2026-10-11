# Alpha Forensics：预测审计与统一组合回测

> 提交版说明：本目录保留关键结果与日账户序列，省略模型、全量预测和逐笔大文件；完整新路径重建请以根目录 README.md、REPRODUCE.md 为入口。以下原阶段说明与清单用于研究溯源。

本轮交付包括**因子时点审计、时间预测验证、模型与因子筛选比较、行业市值检验、统一组合回测、固定调仓方法及联合模型增量比较**。正式结论见[研究报告](research_report.md)，最新结果见[联合模型增量报告](joint_factor_increment_report.md)。原版、仅 PIT 版及各阶段结果均保留。

此前季度 HGB 的资金账本试验保留为历史参考，见[扩展研究记录](research_upgrade_report.md)和[扩展运行说明](UPGRADE_README.md)。本轮新增年度模型组合比较使用独立输出目录；独立新样本验证仍需额外数据，列为未来研究。

私有研究仓库：[stellanhou/alpha-forensics-takehome](https://github.com/stellanhou/alpha-forensics-takehome)，其中保留季度研究报告、源码与各阶段关键结果。

## 报告阅读与最小复现入口（2026-10-11 整理）

- [正式研究报告：六页结构](research_report.md)；[排版 PDF](output/pdf/资产配置策略研究报告-陈星宇-20261011.pdf)。
- [完整实验索引、证据路径、样本口径与已知复现缺口](experiment_inventory_20261011.md)。
- [整理前的扩展主报告](research_report_extended_20261011.md)保留原文；各专题报告记录其对应阶段的方法和当时判断。当前完成范围与下一步以正式研究报告、实验索引为准。

以下所有命令从工作区根目录（包含 `asset_allocation/` 与 `Lab404-CUFEL-E/`）执行。已有本机环境可直接运行；新环境可使用 Python 3.12，并按根目录 `requirements.txt` 安装核心依赖：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -B -m unittest discover -v -s asset_allocation -p test_factor_timing.py
.venv/bin/python -B asset_allocation/factor_timing.py --out asset_allocation/outputs/factor_timing_submission_replay
```

复跑时使用尚不存在的新输出路径。需要完整模型比较时，另按 `asset_allocation/requirements-models.txt` 创建下文使用的 `.venv_models` 环境；macOS 的 LightGBM 需要 `libomp`。核心时点复现无需安装额外树模型库。

本次已在新进程实际运行时点核心命令和7项测试，三个结果表与原输出解压内容完全一致，详见 `outputs/report_review_20261011/verification.json`。本次未重新训练全部模型；最新完整增量实验的既有核验见 `outputs/joint_factor_increment/verification.json`。

提交整理时补齐了年度组合的独立统计核验入口 `submission_tools/verify_portfolio_statistics.py`，用于在新结果目录生成后续调仓所需的 `statistical_verification.json`。已对15账户、45行分期和126条配对区间实际复算通过，日志位于 `outputs/submission_validation_20261011/`。完整新路径重建步骤见工作区 `submission_tools/REPRODUCE.md`。

提交前另在新 `/bin/zsh -f` PTY 终端以及实际解压包的新终端各运行README核心命令，7项测试通过、复现结果一致。记录见 `outputs/submission_validation_20261011/verification.json` 与 `unpacked_verification.json`。

## 本轮核心运行与结果

最新直接增量实验比较“行业＋市值”与“行业＋市值＋20因子”，新增4次联合模型拟合、36个账户。43项相关测试、独立模型复拟合、36个账户资金执行及72条组合配对区间核验均通过。2024年固定收盘计划预算下，Ridge基准与联合模型净收益为25.49%、16.97%；HGB为16.64%、16.77%，差值区间跨零。加入因子改善了部分预测指标，但没有形成稳定净收益增量。

```bash
PYTHONPATH=asset_allocation OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/asset-allocation-mpl asset_allocation/.venv_models/bin/python -B -m unittest test_joint_factor_model test_joint_factor_increment test_rebalance_ledger test_portfolio_ledger test_portfolio_model_comparison -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/asset-allocation-mpl asset_allocation/.venv_models/bin/python -B asset_allocation/joint_factor_increment.py --out asset_allocation/outputs/joint_factor_increment_replay --workers 2
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 asset_allocation/.venv_models/bin/python -B asset_allocation/audit_joint_factor_increment.py --out asset_allocation/outputs/joint_factor_increment_replay
```

本轮结果位于 `outputs/joint_factor_increment/`：预测统计在 `prediction_evaluation/`，账户分期和配对结果在 `period_metrics.csv`、`paired_portfolio.csv`，逐笔记录在 `runs/`，核验结果在 `verification.json`。预算限制收盘计划而非开盘实际成交；风险敞口和实际换手仍需一起看，详见[完整报告](joint_factor_increment_report.md)。

`rebalance_ledger.py` 和 `audit_rebalance_study.py` 增加了可选预算参数，无预算行为经回归核验保持不变。两份修改前源码保存在 `outputs/joint_increment_legacy_sources/` 且匹配旧哈希；相关 `test_rebalance_ledger.py` 也已变化，旧测试源码快照未找到。直接使用这三个当前文件校验旧冻结哈希会不一致；当前版本可在新目录重新运行，不能称为还原旧测试版本。原输入和历史结果保留，详细缺口见实验索引。

### 前一轮固定调仓比较

此前固定调仓研究已完成 108 个实际新账户，引用上一轮 15 个 FULL 账户，等权不适用的规则明确使用别名且不重复运行。当时33项测试、152行统计及405行对照引用核验通过，108个账户的逐笔资金执行核验全部通过。输入复用 `outputs/portfolio_model_comparison/` 的完整预测，不重新训练模型。

```bash
PYTHONPATH=asset_allocation OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/asset-allocation-mpl asset_allocation/.venv_models/bin/python -B -m unittest test_rebalance_ledger test_portfolio_ledger test_portfolio_model_comparison -v
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/asset-allocation-mpl asset_allocation/.venv_models/bin/python -B asset_allocation/rebalance_study.py --out asset_allocation/outputs/rebalance_study_replay --workers 2
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 asset_allocation/.venv_models/bin/python -B asset_allocation/audit_rebalance_study.py --out asset_allocation/outputs/rebalance_study_replay
```

输出目录必须尚不存在。`outputs/rebalance_study/period_metrics.csv` 是新账户的全期及年度结果，`comparison.csv` 另包含旧 FULL 和明确标识的别名；`paired_comparisons.csv` 是主期配对区间。`runs/` 保存计划、开盘预算、净额委托、每日持仓、净值和暴露，核验记录在 `verification.json` 与 `statistical_verification.json`。具体参数、成本压力和适用边界见[调仓研究报告](rebalance_study_report.md)。

### 前置年度模型与完整换仓对照


前置年度模型组合比较已完成 8 次固定模型拟合及 15 个资金情景，并通过独立审计。如需重建此前的完整换仓对照，使用下列命令；默认数据包、年度模型比较和行业市值检验产物均须位于本 README 指定位置。使用新的输出目录以保留已有证据。

```bash
PYTHONPATH=asset_allocation OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/asset-allocation-mpl asset_allocation/.venv_models/bin/python -B -m unittest test_portfolio_model_comparison test_portfolio_ledger -v
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/private/tmp/asset-allocation-mpl asset_allocation/.venv_models/bin/python -B asset_allocation/portfolio_model_comparison.py --out asset_allocation/outputs/portfolio_model_comparison_replay
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 asset_allocation/.venv_models/bin/python -B asset_allocation/audit_portfolio_model_comparison.py --out asset_allocation/outputs/portfolio_model_comparison_replay
asset_allocation/.venv_models/bin/python -B submission_tools/verify_portfolio_statistics.py --out asset_allocation/outputs/portfolio_model_comparison_replay
```

已保存结果在 `outputs/portfolio_model_comparison/`：`period_metrics.csv` 为全期与年度净收益、回撤和换手，`paired_comparisons.csv` 为配对区间，`runs/` 包含每日净值、逐笔订单和持仓，`verification.json` 与 `statistical_verification.json` 为独立核验。详细规则及图表见[组合回测报告](portfolio_backtest_report.md)。本轮基于已查看的合成历史，使用前日交易许可和标准订单成本代理，五日完整轮换且不抵消同名买卖。

### 前置季度时间模型


保留既有结果，在一个尚不存在的新目录复跑时间模型：

```bash
PYTHONPATH=asset_allocation OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m unittest test_factor_timing test_model_rerun test_temporal_model -v
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B asset_allocation/temporal_model.py --out asset_allocation/outputs/temporal_oos_replay
```

默认数据包位置和原模型/PIT 前置结果见下文。现有时间结果在 `outputs/temporal_oos/`：`comparison.csv` 为同样本三版本对照，`metrics.csv` 为年度诊断，`predictions.csv.gz` 为完整预测，`fold_manifest.csv` 为训练时间边界，`run_manifest.json` 为运行溯源。上述文件对应前置季度时间模型；最新年度组合回测使用本节前列命令。

## 前期审计与原流程重跑记录

核验 `factor_03`、`factor_12` 按参考日期提前使用的问题，生成按实际发布时间对齐的研究面板；比较单因子 RankIC，并以相同样本、相同参数重训全部 20 因子的组合模型，重跑原有分组收益回测。

## 实际结果（2026-10-11）

在成员股票池内，以每天相同资产组成的共同样本比较；数值为日度截面 RankIC 的等权均值。

| 因子 | 原始快照 | 按实际发布时间修正 | 修正减原始 | 共同样本行数 | 有效日期数 |
|---|---:|---:|---:|---:|---:|
| factor_03 | 0.395527 | 0.054292 | -0.341235 | 393,287 | 1,033 |
| factor_12 | 0.044809 | 0.043859 | -0.000950 | 394,067 | 1,035 |

`factor_03` 的平均 RankIC 下降约 86.3%，原先较高的相关性高度依赖提前使用尚未发布的快照。修正后的年度均值依次为 2021 年 0.0692、2022 年 0.0823、2023 年 0.0376、2024 年 0.0278。`factor_12` 对本次时点修正较不敏感，其 2024 年修正后均值为 -0.0035。

修正面板保留 394,457 行、390 个成员资产、1,036 个信号日期；其中 03/12 的首发前缺失分别为 1,170/390 行。全部非缺失取值的发布日期均不晚于信号日，其余 18 个因子保持一致。原表全资产 IC 已单独复现，不能与此处成员池共同样本值混作一个口径。

7 项针对性测试、环境依赖检查、官方数据读取检查、原有练习 3 项测试和最小模型拟合均通过。结果可追溯到下方列出的实际运行产物。后续时间验证及账本比较已完成，见本 README 的对应实验段落。

## 全部 20 因子组合模型的实际重跑结果（2026-10-11）

原始控制组和修正时点组已分别完成训练及原回测。比较以本轮同一环境、同样本的两组结果为准。

| 指标 | 原样重跑控制组 | 修正时点后重跑 |
|---|---:|---:|
| 随机验证 Pearson | 0.336486 | 0.092549 |
| 随机验证 RMSE | 0.044856 | 0.047422 |
| 全期平均五日多空收益差 | 5.6203% | 1.5858% |
| 原口径年化代理 Sharpe | 100.5881 | 24.1367 |
| 训练 / 验证样本 | 88,000 / 22,000 | 88,000 / 22,000 |
| 全量预测 / 回测日期 | 394,457 / 1,036 | 394,457 / 1,036 |

预测相关性下降约 72.5%，平均五日多空收益差下降约 71.8%。这证明提前使用因子快照也显著影响了组合模型和原分组收益指标。修正组实际完成 55 次迭代；三项新增测试、训练/验证有序键一致性、全量预测覆盖和结果文件校验均通过。

交接文件原保存的 Pearson 为 0.337179、代理 Sharpe 为 99.7739，与本轮原样重跑有小幅差异。原包未记录依赖版本，不能据此确定差异来源。`model_comparison.csv` 同时保留交接值和两组新结果，不将复跑差异计入时点修正效果。

24.1367 仍是原脚本定义的重叠五日收益差代理指标。由于随机切分、全期包含训练样本、未扣交易成本等口径保持不变，该阶段尚不证明修正后的模型具有时间外预测能力或可交易净收益；后续验证见下文。

## 输入与环境

使用原始 `Alpha_Forensics_Candidate_v1` 数据包。默认位置为工作区下的：

```text
Lab404-CUFEL-E/03-资产配置策略方向/Alpha_Forensics_Candidate_v1/
```

读取行情、因子、股票池、因子发布时间四张表，以及原始研究结果。原数据包和交接代码保留原样。当前虚拟环境使用 Python 3.12.13、NumPy 2.5.4、pandas 2.3.3、scikit-learn 1.9.1。固定依赖记录在工作区根目录的 `requirements.txt`；组合模型另使用 scikit-learn 及其依赖 joblib。

## 运行

在工作区根目录执行：

```bash
.venv/bin/python -B -m unittest discover -s asset_allocation -p 'test_*.py' -v
.venv/bin/python -B asset_allocation/factor_timing.py
```

数据包或输出目录不同，可以指定：

```bash
.venv/bin/python -B asset_allocation/factor_timing.py --root /path/to/Alpha_Forensics_Candidate_v1 --out /path/to/output
```

重复运行会重建所选输出目录内的四个同名结果文件。

### 组合模型与原回测

因子时点步骤完成后，在工作区根目录按顺序执行：

```bash
AF_DATA='Lab404-CUFEL-E/03-资产配置策略方向/Alpha_Forensics_Candidate_v1'
AF_ORIGINAL='asset_allocation/outputs/model_rerun/original'
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B "$AF_DATA/baseline/baseline_code/train_model.py" --root "$AF_DATA" --out "$AF_ORIGINAL"
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B "$AF_DATA/baseline/baseline_code/backtest.py" --root "$AF_DATA" --out "$AF_ORIGINAL"
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B asset_allocation/retrain_pit_model.py
```

前两步原样运行交接脚本，建立当前环境的控制组。第三步重训修正时点后的模型，并自动调用原 `backtest.py`。训练部分依据交接脚本保留原参数与抽样逻辑，新增样本一致性和预测覆盖检查；无调参、无更换模型或因子列。

两组均使用 20 个因子、随机种子 1021、110,000 条抽样记录、随机 80/20 行级划分；训练 88,000 行、验证 22,000 行。原 HGB 参数和默认早停规则不变。按 `asset_id,date` 恢复相同记录顺序，独立抽样和切分后核对训练/验证有序键哈希。其余 18 个因子与标签在模型实际使用的 float32 精度上完全一致。PIT 首发前缺失交给模型原生处理，不删除样本。

两组均对完整 394,457 行预测，再按原口径计算 1,036 日的最高/最低十分位五日收益差。保留随机留出和含训练样本的全期评价，是为了隔离因子时点修正的影响。该控制变量阶段未引入交易账本、成本、执行约束或按时间划分的验证；这些环节已在后续实验中完成。

## 计算约定

1. 在信号日 `t` 收盘后观察信息。对每个资产，选择满足 `available_on <= t` 的最新参考日快照，交易最早在下一合成交易日开盘发生。读取可用日期记录，不将固定滞后天数写入取值逻辑。
2. 本包中同一资产的发布时间随参考日不递减，代码先验证这一条件，再按发布时间向后匹配。如果以后使用了发布时间乱序的数据，程序会报错，不能直接沿用这一取值逻辑。
3. 首次发布前保留缺失值；最新已发布快照本身缺失时也保留缺失，不沿用更旧的非缺失值。其余 18 个因子按题目约定于参考日盘后可得，保持原值。
4. 使用全部快照完成对齐，然后按信号日的 `is_listed`、`is_member` 及有效收益标签筛选。此前尚未入池的快照，只要已经发布，仍可用于入池后的信号。
5. 五日收益标签为 `open(t+6)/open(t+1)-1`。日期按周一至周五的合成交易日历计算，两个价格必须实际存在，不填补退出后的行情，也不以“下几行”替代目标日期。
6. 对每个因子、每个日期，仅在原始值、修正值和标签共同非缺失的资产集合中，分别重新排名并计算 Spearman 相关。汇总时每天等权；任一侧 IC 未定义的日期同时退出两侧均值，单独保留有效日期计数。
7. 交接逐因子表与建模代码使用不同股票池：交接表的 03/12 数值可由全部资产有标签样本复现（404,147 行），建模代码则筛选当日成员（394,457 行）。分别计算并记录这两个口径，修正前后比较只使用成员池内的共同样本。

## 输出

默认输出到 `asset_allocation/outputs/factor_timing/`：

| 文件 | 内容 |
|---|---|
| `pit_research_panel.csv.gz` | 全部 20 个因子、交易标签日期与数值；03/12 额外保留所用快照的参考日和发布日期。首发前缺失仍保留，后续模型须明确其缺失值处理方式。 |
| `daily_rank_ic.csv` | 两因子每天共同样本数量、原始/修正 RankIC 及配对差值。 |
| `timing_summary.csv` | 全期和逐年均值、共同样本/有效日期数、原始成员池全部有效样本 IC。 |
| `run_manifest.json` | 运行命令、环境版本、输入/代码/输出 SHA256、发布时间审计、原表全资产口径复现与计算约定。 |

单元测试覆盖发布日边界、资产隔离、缺失快照、同日发布顺序、乱序发布拒绝、标签端点与周末、共同样本排名，以及先对齐后筛选股票池。

组合模型结果位于 `asset_allocation/outputs/model_rerun/`：

| 路径 | 内容 |
|---|---|
| `original/` | 原脚本重新训练和回测的指标、全量预测、逐日五日分组收益及原运行清单。 |
| `pit/` | 修正时点后的对应结果，另含训练模型 `model.joblib`、抽样分组记录和扩展运行清单。 |
| `model_comparison.csv` | 交接保存值、当前环境原样重跑值、PIT 重跑值和同环境两组差值。 |

PIT 清单保存环境、模型参数、实际迭代数、同样本证明、输入/源码/输出校验值和原回测命令。新增 3 项测试检查缺失不改变样本、输入次序错配被拒绝、预测缺行在原回测内连接前被发现。

## 结论范围

上述时点修正阶段回答提前使用快照对单因子相关性、原组合模型和原分组收益指标有多大影响。原脚本的代理 Sharpe 仅保留用于同口径复跑比较；它基于重叠五日收益差，不是可执行策略的净值夏普。该阶段未包含时间外验证、持仓资金账本、交易成本与成交约束，不据此判断未来可交易收益或给出独立日观测假设下的显著性。新增 CPU 模型时间外比较见下节。

## CPU 备选模型比较

已下载并验证 HGB、Ridge、Elastic Net、Extra Trees、LightGBM、XGBoost、CatBoost 七种固定参数方法，另保留训练均值基准。先用 2021–2022 年训练、2023 年选择，再扩展训练期并确认 2024 年；训练及验证末端均清除跨边界标签。七个模型均完成两阶段真实运行。

按 2023 年平均日 RankIC 预选的 Ridge 在 2024 年取得 0.051048，HGB 为 0.045993，配对差值参考区间为 [0.002186, 0.008649]。但 Ridge 的五日毛分组差低于 HGB，所有模型的 RMSE 都略高于训练均值基准。本轮支持继续研究排序信号，未比较这些模型在成本后持仓账本中的表现。

- [完整结果、取舍、边界和复现命令](model_comparison_report.md)
- [模型方法、官方文档和开源许可](method_sources.md)
- [固定 Python 依赖](requirements-models.txt)
- 结果：`outputs/model_benchmark/`；安装、测试、下载来源与许可：`outputs/model_environment/`。

本机使用独立环境 `asset_allocation/.venv_models/`；macOS 另需 OpenMP 运行库 `libomp`。以下命令使用新目录复现，保留已完成结果：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 asset_allocation/.venv_models/bin/python -B asset_allocation/compare_models.py --stage validation --out asset_allocation/outputs/model_benchmark_reproduction
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 asset_allocation/.venv_models/bin/python -B asset_allocation/compare_models.py --stage confirmation --out asset_allocation/outputs/model_benchmark_reproduction
asset_allocation/.venv_models/bin/python -B asset_allocation/plot_model_comparison.py --out asset_allocation/outputs/model_benchmark_reproduction
```

源代码与输入哈希在训练前冻结，确认阶段核验验证结果和选模记录。更改代码、参数或环境后须使用新输出目录；不得在观察 2024 年结果后覆盖本次选择。本轮 10 项针对性测试和两个阶段的完整运行均已通过。

## HAC 与 Elastic Net 因子筛选

新增研究已完成：固定 Ridge/HGB，比较全部 20 因子、RankIC 筛选、HAC 每簇选代表和 Elastic Net 筛选，并保留同数量的 RankIC 对照。参数只在 2021—2022 年内部四个季度选择，然后按冻结规则比较 2023、2024 年。共完成 104 次下游模型拟合，16 项新增测试及独立结果复核通过。

结论：筛选流程没有稳定超越全 20 因子基准；同样保留 10 因子时，Elastic Net 优于 IC Top 10。HAC 没有显示明确增益。汇总结论见 [主研究报告第 3 节](research_report.md)，完整方法、因子名单、图表和复现命令见 [因子筛选报告](factor_selection_report.md)。

代码为 `factor_selection_study.py`，依次运行 `--stage tune`、`--stage validation`、`--stage confirmation`；默认结果目录 `outputs/factor_selection_study/` 已有完成证据，复现请另用新目录。专用环境及依赖沿用上一节，无新增库。所有阶段均保留完整协议、代码和文件哈希；2024 年结果属于已查看数据上的追加历史研究。

## 行业和市值控制检验

已完成年度全 20 因子 Ridge/HGB 的行业与市值暴露、部分秩相关，以及仅用行业和市值训练的基准比较。2024 年 Ridge 原始 RankIC 为 0.051048，同时控制行业与市值后为 0.047585；共同特征 Ridge 基准为 0.038976，全因子方案相对它的增量区间跨零。详细解释见[行业和市值检验报告](industry_size_attribution_report.md)，汇总已加入[阶段报告第六节](stage_report_20261011.md)和[主报告第 4 节](research_report.md)。

本轮复用原年度模型预测，只新增四次共同特征基准拟合，12 项针对性测试通过。风险表采用对应信号日盘后可知的假设；检验控制行业截距及规模排名的线性可加关系，属于追加历史诊断。沿用上一节环境及依赖，从工作区根目录运行：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 asset_allocation/.venv_models/bin/python -B -m unittest discover -s asset_allocation -p 'test_industry_size_attribution.py' -v
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 asset_allocation/.venv_models/bin/python -B asset_allocation/industry_size_attribution.py --out asset_allocation/outputs/industry_size_attribution_reproduction
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 asset_allocation/.venv_models/bin/python -B asset_allocation/audit_industry_size_attribution.py --out asset_allocation/outputs/industry_size_attribution_reproduction
```

现有完成结果位于 `outputs/industry_size_attribution/`，正式运行和测试日志位于 `outputs/industry_size_attribution_logs/`。程序拒绝已有输出目录，校验原预测来源、样本键和标签，并保存冻结协议及输入、源码、结果哈希。

独立审计已通过：四个共同特征基准经单独编码重新拟合，全部 3,060 条部分秩相关经不同计算路径复算，年度区间、样本及暴露统计一致。核验范围和数值误差记录在 `outputs/industry_size_attribution/verification.json`。审计拒绝覆盖已有核验文件，复现时对新生成的研究目录执行。
