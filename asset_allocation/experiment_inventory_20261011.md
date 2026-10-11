# 资产配置实验全景索引与报告证据清单

整理日期：2026-10-11。范围仅为本项目 `asset_allocation/` 及随题 Alpha_Forensics_Candidate_v1 数据；原始数据、源码和已有实验输出均未因本次报告整理而修改。

正式阅读入口为 [六页研究报告](research_report.md) 和 [PDF](output/pdf/资产配置策略研究报告-陈星宇-20261011.pdf)。原主报告完整保留为 [整理前扩展稿](research_report_extended_20261011.md)；既有专题报告保留各阶段的详细方法与当时判断。本索引按最新完成状态解释它们。

## 1. 题目要求与完成情况

| 题目要求 | 当前对应材料 | 状态 |
|---|---|---|
| 可实际运行的研究过程 | 原题包、研究源码、固定依赖、README 命令、各阶段真实输出 | 已完成；本次另复跑时点核心入口 |
| 至少核验一项关键结论 | 原高指标的发布时间审计与同样本 HGB 重训 | 已完成 |
| 至少一项自主后续研究 | 时间验证、模型/筛选、行业市值、资金执行和联合增量 | 已完成 |
| 数值、图表可追踪 | 下述十条实验线及输入/源码/输出校验、预测和逐笔账本 | 已保存；旧测试源码快照有一处缺口，见第 5 节 |
| 明确下一步建议 | 主报告第 6 节：风险匹配、执行敏感性、独立新增样本 | 已给出；这些后续工作尚未实施 |
| 报告 3–6 页等效篇幅 | 主报告采用六页结构，全文细节留在专题报告 | 已整理 |

题目原文：[Take-home 试题](../Lab404-CUFEL-E/03-资产配置策略方向/Take-home试题（资产配置策略方向）.md)。提交目录还要求 README、源码依赖和 outputs；提交包已完成整理，新终端核心复现及解压包复现均通过，记录在 `outputs/submission_validation_20261011/`；尚未发送邮件。

## 2. 十条实验线：目的、实际规模与当前结论

规模按该阶段实际执行统计。后续重拟合同一模型、重放相同账户和同数量别名不代表新增独立证据；测试集包含重复项目，不能把各阶段测试数量相加。

| 编号 | 实验与研究问题 | 已运行规模 / 主要产物 | 当前结论与证据地位 |
|---|---|---|---|
| 01 | 因子可用时点：原快照是否当时已知？ | 两个延迟因子；394,457 行成员面板，390 资产、1,036 日；`outputs/factor_timing/` | factor_03 共同样本 RankIC 0.395527→0.054292；factor_12 0.044809→0.043859。主结论证据 |
| 02 | 控制变量重训：时点修正能否解释原模型高分？ | 原样 HGB 与 PIT HGB 各一次，训练/验证 88,000/22,000 行；`outputs/model_rerun/` | Pearson 0.336486→0.092549；五日毛分组差5.6203%→1.5858%。同样本诊断，保留原随机切分 |
| 03 | 季度时间模型：只用已成熟标签后是否还有信号？ | 1 次 2022 验证＋8 次季度拟合；201,662 条完整预测；`outputs/temporal_oos/` | 2023—2024 RankIC 0.059525，毛分组差0.7658%。模型时间外回放；研究层面并非盲测 |
| 04 | 季度账本原型：毛信号如何进入资金账户？ | 8 个账本情景；`outputs/portfolio_upgrade/`、`outputs/upgrade_audit.json` | 时间 HGB 两年净收益34.12%，2024年-5.30%；成本侵蚀较大。保留为历史阶段，不与年度方案混用 |
| 05 | 年度模型比较：复杂模型是否必要？ | 7 候选＋训练均值基准，两阶段共16次拟合；`outputs/model_benchmark/` | 2023规则选Ridge，2024 RankIC 0.051048；排序优于HGB，RMSE未优于均值。确立后续主/对照模型 |
| 06 | 因子筛选：去冗余是否改善结果？ | 80 次内部＋24 次外层下游拟合；另有筛选器拟合；`outputs/factor_selection_study/` | HAC/IC/EN没有稳定优于ALL20；EN10优于IC10。适合作为压缩候选 |
| 07 | 行业市值检验：信号是否仅是常见暴露？ | 复用ALL20预测，新增4次基准拟合；3,060条部分秩相关；`outputs/industry_size_attribution/` | Ridge 2024部分秩相关0.047585；ALL20对简单Ridge基准的增量区间跨零。不是嵌套输入增量检验 |
| 08 | 年度完整换仓：统一组合评价 | 8次年度固定拟合，5预测器/基准×3成本=15账户；`outputs/portfolio_model_comparison/` | 2024题包成本下5类组合均亏损，Ridge/HGB ALL20为-5.94%/-5.44%。建立完整成员预测和统一账本 |
| 09 | 固定调仓：减少无效交易是否改善净收益？ | 108新账户，引用15个FULL账户；等权别名不重复运行；`outputs/rebalance_study/` | NET/缓冲有改善，但简单基准改善更大，ALL20交易增量未成立 |
| 10 | 联合模型：知道行业市值后再加因子有何价值？ | 新增4次JOINT拟合，36次账户运行；其中12控制账户重放；`outputs/joint_factor_increment/` | 主比较Ridge预算组16.97%对25.49%；HGB16.77%对16.64%、区间跨零。当前最终判断依据 |

第08—10轮共159次账户运行、147个不同配置，不等于159个独立策略。第04轮8账本另计。内部独立复算属于实现验证，不是独立市场样本。

## 3. 各阶段的样本和方法边界

### 信息时点与成员池

- 03/12 各有406,547条发布记录，延迟分别为3/1个合成工作日。按发布日期精确对齐，而不是把固定延迟硬编码成通用规则。
- 原逐因子表使用全部资产有标签的404,147行，factor_03 原IC为0.396281；共同样本比较使用成员池，不能直接拿这两个口径相减。
- 原标签使用行位移，新流程使用精确交易日期端点。本题同样本重训已证明标签一致；未观察到本数据中行位移导致的标签差错。

### 季度模型与年度模型

| 口径 | 时间设计 | 实际评价样本 |
|---|---|---|
| 季度HGB | 2021训练→2022验证；2023/2024每季度扩展训练 | 2023/2024合计199,322行、516日；完整信号201,662行、522日 |
| 年度模型/筛选/行业市值/联合增量 | 2021—2022训练→2023；2021—2023训练→2024 | 2023为99,057行、254日；2024为97,929行、256日 |
| 年度账本使用的完整交易信号 | 当日上市及成员身份决定信号覆盖，不按未来标签过滤 | 2023为101,397行、260日；2024为100,265行、262日 |

年度成熟训练行分别为192,806与294,192。2023年度标签诊断排除跨入2024的收益终点，故其评价行数不能简单由完整信号减“缺标签行”得到。2023/2024缺完整标签信号为4/2,336行。

HGB季度方案同时改变时间窗口、训练量和内部早停；年度模型则采用独立冻结的实验口径。跨表比较必须先核对口径。所有追加研究均使用此前已查看的2021—2024历史，2024的“confirmation”目录名不表示研究者从未看过该年。

### ALL20、CONTROLS 和 JOINT

- ALL20：仅20个PIT因子。
- CONTROLS：行业哑变量＋log市值。
- JOINT：CONTROLS＋20个PIT因子。

行业市值诊断比较ALL20与CONTROLS，并计算指定投影下的残差相关；直接增量实验比较JOINT与CONTROLS。最新预算并未严格匹配持仓行业或规模敞口。20%限制每份资金每次收盘计划双边变化，不是账户实际成交上限，也不是两组实际换手相等。

### 实际交易的假设与限制

成本为前日单边手续费、半价差和标准订单冲击代理，许可也使用前日标记。当日最终状态仅事后审计。缺报价零回收核销及期末受阻仓位估值属于显式假设。旧季度主净账户39,609笔假设成交中，291笔与次日最终标记冲突，说明需要进一步检验执行。

收益标签诊断、算术日收益差、年度复利收益差各有不同含义。HGB预算组年度净收益略高、平均日差略负是已核实的复利路径差异。区间采用20日移动区块、2,000次配对重采样，未作多项比较校正。

## 4. 证据到源码和运行命令的映射

以下从工作区根目录执行，采用README记录的环境和线程设置。示例中的新输出路径须尚不存在；使用新路径保留原运行证据。原样对照使用未修改的题包脚本。后续步骤依赖前置产物，不能跳过输入依赖直接在空目录运行。

| 实验 | 主计算入口 | 关键核验 / 结果 |
|---|---|---|
| 时点审计 | `factor_timing.py --out 新目录` | `timing_summary.csv`、`daily_rank_ic.csv`、`run_manifest.json` |
| 同样本重训 | 原题包 `train_model.py`、`backtest.py`；`retrain_pit_model.py` | `model_comparison.csv`、PIT组 `baseline_run_manifest.json` |
| 季度模型 | `temporal_model.py --out 新目录` | `comparison.csv`、`metrics.csv`、`fold_manifest.csv` |
| 季度账本 | `evaluate_portfolios.py`；`audit_upgrade.py` | 详见 `UPGRADE_README.md`、`outputs/upgrade_audit.json` |
| 年度模型 | `compare_models.py --stage validation/confirmation --out 新目录` | `selection.json`、两阶段 `summary.csv`、`verification.json` |
| 筛选 | `factor_selection_study.py --stage tune/validation/confirmation --out 新目录` | `protocol.json`、`chosen_pipeline.json`、各阶段汇总、`verification.json` |
| 行业市值 | `industry_size_attribution.py --out 新目录`；对应 `audit_industry_size_attribution.py` | `partial_rank_summary.csv`、`paired_baseline_comparison.csv`、`verification.json` |
| 年度账本 | `portfolio_model_comparison.py --out 新目录`；对应 `audit_portfolio_model_comparison.py` | `period_metrics.csv`、`paired_comparisons.csv`、两个verification |
| 固定调仓 | `rebalance_study.py --out 新目录 --workers 2`；对应 `audit_rebalance_study.py` | 同上；重建历史源码哈希注意第5节 |
| 直接增量 | `joint_factor_increment.py --out 新目录 --workers 2`；对应 `audit_joint_factor_increment.py` | `prediction_evaluation/`、`period_metrics.csv`、`paired_portfolio.csv`、`verification.json` |

上述文件名均在 `asset_allocation/` 下。完整参数和环境命令以 [README](README.md)、[季度扩展说明](UPGRADE_README.md) 及相应专题报告为准。所有拟合的参数、训练期预处理和运行命令均保存在各阶段清单中；README列有新目录复跑示例。

本次实际新进程核心验证命令：

```bash
.venv/bin/python -B -m unittest discover -v -s asset_allocation -p test_factor_timing.py
.venv/bin/python -B asset_allocation/factor_timing.py --out asset_allocation/outputs/report_review_20261011/factor_timing_replay
```

两条命令退出码均为0，7项测试通过。三份CSV与原输出解压内容完全一致：汇总10行，逐日统计2,068行，面板394,457行。日志和哈希详见 [本次核心复现记录](outputs/report_review_20261011/verification.json)。复跑本节请更换目录名，避免覆盖这次记录。

## 5. 本次核对结果及材料缺口

本次报告整理除上述最小复现外，没有重新训练模型或重跑全部组合，检查基于原始CSV、协议、源文件和已有独立审计记录。

- 模型比较两阶段48条输出哈希、筛选三阶段267条输出哈希、行业诊断95条源码/输出哈希记录一致。从Ridge/HGB四份预测直接重算RankIC，与汇总一致；筛选ALL20四份逐行预测与年度模型对照一致。
- 组合三轮核对1,309个唯一文件的SHA256，原输入和结果均一致；从159份日净值复算477行分期指标，最大误差约1.05×10^-14；34行主要配对区间重算最大误差约1.1×10^-16。
- 最新JOINT原有独立审计覆盖4个模型重拟合、36个账户、72条组合配对；风险敞口及预算诊断汇总表不在逐表独立复算范围，核心持仓和每次预算计划在范围内。
- 早期7/3/5项测试通过状态原先只见于报告，未找到每组单独的成功日志。本次补有时点7项的实际日志；其余未在本次重新运行。最新43项有既有运行日志，数量包含复用测试。

**旧源码版本缺口：** `outputs/rebalance_study/protocol.json` 中有3个源码哈希与现版不同。旧 `rebalance_ledger.py`、`audit_rebalance_study.py` 在 `outputs/joint_increment_legacy_sources/` 中有匹配快照；旧 `test_rebalance_ledger.py` 未找到匹配快照。旧测试哈希为 `4ea5cc1d83ec506a86e8665ade2fc5ed9937de4bc31c2e104e4b174b740011ac`，现版为 `2e733e103aa5d09b3e3294a58dab873924764f5aebf3d705379142205eb1ca3d`。已有测试日志与结果仍保留，但无法完整还原当时测试文件。当前新目录运行形成的是当前版本证据，不能声称通过旧源码哈希校验。最新JOINT的完整源码快照匹配其冻结记录。

主报告已修正两个叙述问题：明确区分季度与年度指标；把“后续拟做”的阶段性说法按当前完成状态更新。专题历史报告原样保留，引用其未来计划时以本索引为准。

## 6. 完成与未完成边界

已完成的是现有合成数据上的研究审计、模型比较、指定风险诊断、固定执行及联合增量研究。独立新样本、严格风险敞口匹配、因子组消融、容量、实际开盘执行与退出估值敏感性尚未完成。其他五种预测模型和筛选后组合未接入最新账本，不能将Ridge/HGB组合结论推广到它们。

主报告没有将模型数量或累计账户运行数作为质量证明；这些规模仅用于定位实验。下一轮优先解决新问题，避免在同一历史上继续遍历方案并把最优结果当成验证。
