# Alpha Forensics：资产配置策略方向提交项目

陈星宇｜金融人工智能北京市重点实验室 2026 年秋季实习生笔试｜2026-10-11

## 阅读入口与研究结论

先阅读根目录 [research_report.pdf](research_report.pdf)（6页，英文与数字统一为 Times New Roman）。可编辑文本在 [asset_allocation/research_report.md](asset_allocation/research_report.md)，完整实验梳理见 [实验索引](asset_allocation/experiment_inventory_20261011.md)。

研究结论：原高指标明显依赖提前使用因子快照。修正时点后仍有历史排序信息，但在行业、市值基础上加入20个匿名因子，尚未形成稳定的成本后组合增量。所有行情与资产均为题包合成数据。

## 研究仓库与季度报告

公开 GitHub 仓库：[stellanhou/alpha-forensics-takehome](https://github.com/stellanhou/alpha-forensics-takehome)。仓库归档本题报告、实验源码和关键证据；仓库可直接访问，提交包也保留相同的阅读材料。

- [季度模型与季度账本专题报告](asset_allocation/research_upgrade_report.md)
- [季度研究的运行说明](asset_allocation/UPGRADE_README.md)
- [季度时间验证指标](asset_allocation/outputs/temporal_oos/metrics.csv)与[训练边界](asset_allocation/outputs/temporal_oos/fold_manifest.csv)
- [季度账本比较结果](asset_allocation/outputs/portfolio_upgrade/portfolio_comparison.csv)
- [全部实验索引](asset_allocation/experiment_inventory_20261011.md)

## 本包内容

- `asset_allocation/*.py`：研究、独立审计和测试源码，保持实验源码原样。
- `asset_allocation/outputs/`：各阶段汇总、逐日统计、模型参数、协议、运行清单、既有审计记录、图表及全部组合的日账户序列。
- `submission_tools/verify_portfolio_statistics.py`：补齐年度组合至固定调仓之间的统计核验入口；实际验证记录随包保留。
- `requirements.txt`：原核心运行环境的固定Python依赖；`asset_allocation/requirements-models.txt`：七模型比较的额外完整环境。
- `Lab404-CUFEL-E/`：题目与原数据包的小型说明、交接代码和汇总参考材料；六张大型原始数据表未重复附带。
- `PACKAGE_CONTENTS.csv`、`SHA256SUMS.txt`：提交文件清单与校验值。`REPRODUCE.md`：在新路径重建完整研究的命令。

这是包含源码和关键证据的轻量提交包。未附虚拟环境、下载缓存、已训练模型、全量预测、PIT全量面板、逐笔订单/持仓/执行计划及排版预览。原项目中的完整产物保留在本地。原清单记录这些完整文件的历史哈希，故包内保留的 `verification.json` 表示当时完整目录的核验结果，不表示裁剪目录可以直接完成逐笔审计。

## 1. 准备题目数据

使用 **Alpha_Forensics_Candidate_v1（v1.0）**。将原题包补齐到下列相对位置，保留其数据和交接文件结构：

```text
本项目根目录/
  README.md
  requirements.txt
  asset_allocation/
  Lab404-CUFEL-E/
    03-资产配置策略方向/
      Alpha_Forensics_Candidate_v1/
        data/
          market_daily.csv.gz
          factors.csv.gz
          universe.csv.gz
          factor_availability.csv.gz
          risk_exposure.csv.gz
          trading_constraints.csv.gz
        baseline/
        data/data_dictionary.md
        版本说明.md
```

最小核心运行读取行情、因子、成员池、发布日期四张表，以及原 `baseline_factor_analysis.csv`、数据字典、版本说明；后续实验使用其余两张表。原题包 `SHA256_PUBLIC.txt` 随包保留，可核验原始输入。原始数据文件不应改写。

## 2. 在新终端运行核心命令

打开新终端并进入本README所在的目录。参考环境为 Python 3.12.13；已有同依赖环境可直接使用。首次运行需创建虚拟环境并安装依赖：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

随后按顺序执行核心命令：

```bash
.venv/bin/python -B -m unittest discover -v -s asset_allocation -p test_factor_timing.py
.venv/bin/python -B asset_allocation/factor_timing.py --out asset_allocation/outputs/factor_timing_submission_replay
```

预期：7项测试通过；输出394,457行成员面板、1,036个信号日。`timing_summary.csv` 中 `factor_03` 的共同样本RankIC应为0.395527→0.054292，`factor_12` 为0.044809→0.043859。新输出目录不随包附带；再次运行请更换 `--out` 路径以保留此前结果。

提交前已在新的 `/bin/zsh -f` PTY 终端实际执行上述两条命令，退出码均为0。三份结果文件与原实验的文件及解压内容哈希一致，行数为10、2,068、394,457。证据在 [submission_validation_20261011](asset_allocation/outputs/submission_validation_20261011/verification.json)，该目录另含测试和运行日志。

另对压缩包实际解压后的源码，在另一个新终端按同样命令复现，7项测试与核心运行均通过，结果仍完全一致。该核验复用已有Python环境，并在解压副本中连接外置官方数据；没有将环境或数据链接打入压缩包。记录见 [解压包核心核验](asset_allocation/outputs/submission_validation_20261011/unpacked_verification.json)。

## 3. 后续实验与完整重建

精简包可直接查看报告对应的汇总结果和日账户序列；完整模型训练、逐笔执行审计需要重新生成已省略的大文件。请按 [REPRODUCE.md](REPRODUCE.md) 在独立新目录重建前置链：

**时点面板 → 年度模型 → 行业市值 → 年度组合与资金审计、统计核验 → 固定调仓与审计 → 联合增量与审计。**

季度模型、季度账本及因子筛选是另外的支线，命令也列在该文件中。安装七模型环境时，macOS另需OpenMP运行库 `libomp`；标准CPU足以运行研究。

旧协议中含原运行机器的绝对路径，保留原样用于溯源。不要修改旧协议后继续宣称旧哈希有效，也不要对这个裁剪后的旧目录直接执行全量审计。新实验会记录本次实际路径及新清单。

## 4. 已知限制与核验范围

- 2023—2024历史此前已被观察。逐时点模型验证与重采样不等于未见样本盲测。
- 行业、市值采用信号日盘后可知的假设；交易采用前日许可与标准订单成本代理；缺报价零回收是明确估值假设。
- 旧固定调仓研究的一份测试源码版本未找到；两份旧账本/审计源码已保留。详见实验索引第5节。新重建使用当前源码，并产生新版本证据。
- 年度组合的历史统计审计原先缺少独立运行入口，本次在 `submission_tools/` 补齐，并以已有15账户、45行分期、126行配对结果实际核验；未修改研究规则或旧结果。它核验日序列与统计，不冒充逐笔订单审计。
- 本次提交前实际重跑的是核心时点实验、解压包核心复现及补充统计核验；未在本次重新训练全部模型、重跑所有账户。完整重建命令的范围与前置依赖在 `REPRODUCE.md` 列明。

开源实现与许可来源见 [method_sources.md](asset_allocation/method_sources.md)，相关许可全文保留在 `asset_allocation/outputs/model_environment/licenses/`。
