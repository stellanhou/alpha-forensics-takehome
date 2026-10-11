# 完整研究重建说明

本文件使用提交包中的原研究源码，在新目录生成新的完整产物及本机清单。提交版 `asset_allocation/outputs/` 是实际历史输出的关键证据集合，省略了模型、全量预测与逐笔大文件，不是完整续跑目录。

本次提交前已执行根README核心复现及补充统计核验；以下整条链未在本次打包时再次全部运行。各阶段原正式执行和审计结果已保存。命令按源码的实际前置依赖给出，运行时不复制旧核验结果冒充新核验。

## 环境与全新工作目录

先按根README补齐官方v1题包。在提交项目根目录执行以下命令，`reproduction/` 必须尚不存在：

```bash
python3.12 -m venv .venv_models
.venv_models/bin/python -m pip install -r asset_allocation/requirements-models.txt
```

macOS需要OpenMP运行库时，另安装 `libomp`，详见模型来源文档。下面的环境变量仅在当前终端生效：

```bash
set -e
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/alpha-forensics-mpl
AF_PY="$(pwd)/.venv_models/bin/python"
AF_DATA="$(pwd)/Lab404-CUFEL-E/03-资产配置策略方向/Alpha_Forensics_Candidate_v1"
AF_STATS="$(pwd)/submission_tools/verify_portfolio_statistics.py"
test ! -e reproduction
mkdir -p reproduction/asset_allocation
cp asset_allocation/*.py reproduction/asset_allocation/
AF_CODE="$(pwd)/reproduction/asset_allocation"
AF_OUT="$AF_CODE/outputs"
```

请逐条执行；任一命令失败应停止，不应越过失败继续。新目录只复制代码，未复制裁剪的历史输出。后续每次完整重建使用另一个新目录名。所有路径均为本次实际路径，因此新审计无需访问原作者机器。

## A. 主链：时点、年度模型、行业市值、组合与联合增量

### A1. 生成时点修正面板

```bash
"$AF_PY" -B "$AF_CODE/factor_timing.py" --root "$AF_DATA" --out "$AF_OUT/factor_timing"
```

### A2. 年度模型比较

```bash
"$AF_PY" -B "$AF_CODE/compare_models.py" --stage validation --panel "$AF_OUT/factor_timing/pit_research_panel.csv.gz" --out "$AF_OUT/model_benchmark"
"$AF_PY" -B "$AF_CODE/compare_models.py" --stage confirmation --panel "$AF_OUT/factor_timing/pit_research_panel.csv.gz" --out "$AF_OUT/model_benchmark"
"$AF_PY" -B "$AF_CODE/plot_model_comparison.py" --out "$AF_OUT/model_benchmark"
```

两阶段使用同一新目录；确认阶段核验验证输出及选模记录。

### A3. 行业、市值诊断及独立核验

```bash
"$AF_PY" -B "$AF_CODE/industry_size_attribution.py" --panel "$AF_OUT/factor_timing/pit_research_panel.csv.gz" --risk "$AF_DATA/data/risk_exposure.csv.gz" --benchmark "$AF_OUT/model_benchmark" --out "$AF_OUT/industry_size_attribution"
"$AF_PY" -B "$AF_CODE/audit_industry_size_attribution.py" --out "$AF_OUT/industry_size_attribution"
```

### A4. 年度完整换仓、资金审计与统计核验

```bash
"$AF_PY" -B "$AF_CODE/portfolio_model_comparison.py" --root "$AF_DATA" --panel "$AF_OUT/factor_timing/pit_research_panel.csv.gz" --benchmark "$AF_OUT/model_benchmark" --attribution "$AF_OUT/industry_size_attribution" --out "$AF_OUT/portfolio_model_comparison"
"$AF_PY" -B "$AF_CODE/audit_portfolio_model_comparison.py" --root "$AF_DATA" --out "$AF_OUT/portfolio_model_comparison"
"$AF_PY" -B "$AF_STATS" --out "$AF_OUT/portfolio_model_comparison"
```

最后一条会通过真实统计复算生成该新目录的 `statistical_verification.json`；下一阶段同时要求资金核验和统计核验通过。不要把提交包中的旧核验JSON复制到新结果目录。

### A5. 固定调仓方法及审计

```bash
"$AF_PY" -B "$AF_CODE/rebalance_study.py" --root "$AF_DATA" --source "$AF_OUT/portfolio_model_comparison" --out "$AF_OUT/rebalance_study" --workers 2
"$AF_PY" -B "$AF_CODE/audit_rebalance_study.py" --root "$AF_DATA" --out "$AF_OUT/rebalance_study"
```

### A6. 联合模型直接增量及审计

```bash
"$AF_PY" -B "$AF_CODE/joint_factor_increment.py" --root "$AF_DATA" --panel "$AF_OUT/factor_timing/pit_research_panel.csv.gz" --portfolio-source "$AF_OUT/portfolio_model_comparison" --rebalance-source "$AF_OUT/rebalance_study" --out "$AF_OUT/joint_factor_increment" --workers 2
"$AF_PY" -B "$AF_CODE/audit_joint_factor_increment.py" --root "$AF_DATA" --out "$AF_OUT/joint_factor_increment"
```

最后目录保存4个新增联合模型、完整预测、36账户逐笔记录、分期统计、配对区间及核验结果。新生成图表使用研究脚本默认样式，字体差异不改变数值。

## B. 支线：原流程重跑、季度模型与季度账本

先完成A1；此支线不依赖A2—A6。原始题包代码读取原数据，输出到新工作区：

```bash
"$AF_PY" -B "$AF_DATA/baseline/baseline_code/train_model.py" --root "$AF_DATA" --out "$AF_OUT/model_rerun/original"
"$AF_PY" -B "$AF_DATA/baseline/baseline_code/backtest.py" --root "$AF_DATA" --out "$AF_OUT/model_rerun/original"
"$AF_PY" -B "$AF_CODE/retrain_pit_model.py" --root "$AF_DATA" --original "$AF_OUT/model_rerun/original" --panel "$AF_OUT/factor_timing/pit_research_panel.csv.gz" --out "$AF_OUT/model_rerun/pit"
"$AF_PY" -B "$AF_CODE/temporal_model.py" --root "$AF_DATA" --old-root "$AF_OUT/model_rerun" --out "$AF_OUT/temporal_oos"
"$AF_PY" -B "$AF_CODE/evaluate_portfolios.py" --root "$AF_DATA" --old-root "$AF_OUT/model_rerun" --temporal "$AF_OUT/temporal_oos" --out "$AF_OUT/portfolio_upgrade"
```

季度账本独立核验：

```bash
"$AF_PY" -B "$AF_CODE/audit_upgrade.py" --root "$AF_DATA" --portfolio "$AF_OUT/portfolio_upgrade" --temporal "$AF_OUT/temporal_oos" --output "$AF_OUT/upgrade_audit.json"
```

实际历史核验另保留在提交版 `asset_allocation/outputs/upgrade_audit.json`。

## C. 支线：因子筛选

先完成A1。三阶段共用一个新输出目录：

```bash
"$AF_PY" -B "$AF_CODE/factor_selection_study.py" --stage tune --panel "$AF_OUT/factor_timing/pit_research_panel.csv.gz" --out "$AF_OUT/factor_selection_study"
"$AF_PY" -B "$AF_CODE/factor_selection_study.py" --stage validation --panel "$AF_OUT/factor_timing/pit_research_panel.csv.gz" --out "$AF_OUT/factor_selection_study"
"$AF_PY" -B "$AF_CODE/factor_selection_study.py" --stage confirmation --panel "$AF_OUT/factor_timing/pit_research_panel.csv.gz" --out "$AF_OUT/factor_selection_study"
"$AF_PY" -B "$AF_CODE/plot_factor_selection.py" --out "$AF_OUT/factor_selection_study"
```

## 输出对照与限制

核对主报告的核心点估计及各阶段样本数，同时保留新环境、命令、输入/代码哈希和运行日志。不同依赖、线程设置或机器可能产生微小浮点差异；不能直接要求含绝对路径和时间戳的新旧manifest字节相同。原始因子时点表可在相同环境下逐字节复现。

主报告中的独立核验状态来自本次研究保存的实际执行记录。因子筛选等部分历史统计核验为当时独立复算记录，其独立验证程序未全部固化成单独CLI；原研究计算源码、逐日统计及核验JSON均保留，可据相同定义复算。完整目录重建不改变“已观察历史”的研究属性，也不补足独立新样本验证。
