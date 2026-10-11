# 资产配置题目备选模型及来源

本题现有实现用 20 个数值因子预测未来五日收益，模型为 scikit-learn 的 HistGradientBoostingRegressor。此次选择六种替代方法，与重新按时间切分的 HGB 一起验证。候选覆盖不同的函数形式与正则化方式，全部使用 CPU；库的实际版本、下载文件和哈希以 `outputs/model_environment/` 为准。

## 本轮验证的方法

| 方法 | 本次研究要回答的问题 | 官方文档 |
|---|---|---|
| HGB | 修正可用时点、改用时间切分后，原模型家族还保留多少预测力 | [HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html) |
| Ridge | 简单的线性因子组合是否已经足够 | [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) |
| Elastic Net | 同时施加 L1 与 L2 约束的线性组合表现如何 | [ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html) |
| Extra Trees | 随机切分后平均多棵树，是否比逐轮提升更稳健 | [ExtraTreesRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html) |
| LightGBM | 另一种高效直方图提升树实现能否改善预测 | [LGBMRegressor](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html) |
| XGBoost | 带正则化的直方图树提升是否带来增量 | [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html) |
| CatBoost | 对称树结构的提升方法能否改善跨年表现 | [CatBoostRegressor](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor) |

具体参数在运行前写入 `outputs/model_benchmark/protocol.json`。此次比较的是固定参数方案，不是各算法经过充分调优后的性能上限。HGB 保留 55 轮、学习率 0.065、17 叶和 L2=0.2，关闭随机内部验证早停；所有方法使用同一训练期中位数填补。线性方法另外使用训练期特征标准化，Elastic Net 还标准化训练目标，再把预测还原到收益单位。Ridge 与 Elastic Net 的目标函数对样本量的归一化方式不同，不能直接比较二者的 alpha 数值。

每个模型只读取 20 个数值因子；不把资产 ID、行业或未来日期作为输入。CatBoost 在这里没有专门利用类别特征编码。

## 下载实现和许可

通过 PyPI 下载并安装官方项目的 Python 发行包，未复制第三方策略仓库，也未引入外部金融数据。下载的 wheel 是本地运行的实现；无需克隆完整源码仓库才能验证库中的模型。

| 实现 | 官方代码 | 许可 |
|---|---|---|
| scikit-learn | [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) | [BSD 3-Clause](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) |
| LightGBM | [lightgbm-org/LightGBM](https://github.com/lightgbm-org/LightGBM) | [MIT](https://github.com/lightgbm-org/LightGBM/blob/main/LICENSE) |
| XGBoost | [dmlc/xgboost](https://github.com/dmlc/xgboost) | [Apache 2.0](https://github.com/dmlc/xgboost/blob/master/LICENSE) |
| CatBoost | [catboost/catboost](https://github.com/catboost/catboost) | [Apache 2.0](https://github.com/catboost/catboost/blob/master/LICENSE) |

许可全文和下载记录保留在本地项目；报告中的官方文档用于说明方法，性能结论只来自随题数据上的实际运行。

macOS 上 LightGBM 与 XGBoost 的本轮 wheel 还依赖 OpenMP 原生运行库。首次导入实际报出缺少 `libomp.dylib`，按 [LightGBM 官方 Python 安装说明](https://github.com/lightgbm-org/LightGBM/blob/main/python-package/README.rst) 使用 Homebrew 补齐 `libomp`；Python 依赖清单不能代替这一系统依赖。首次失败和修复后的检查分别保留日志。

## 尚未纳入本轮的方法

学习排序可以直接针对同一天资产的排序建立目标。[LGBMRanker](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html) 支持分组信息；若进一步研究，需要明确收益到相关性等级的映射及排序损失，作为另一项实验。样条加线性正则化也能检查可解释的非线性结构，见 [SplineTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.SplineTransformer.html)。两者本轮未训练。

风险平价、均值方差和层次风险平价涉及组合权重构建。当前比较的对象是收益预测器；若以后研究组合构建，需要另外定义持仓、风险估计、换手、成本和成交假设。

## 因子筛选扩展研究

扩展实验固定下游模型为 Ridge 和 HGB，比较全 20 因子、单因子 RankIC 排序、HAC 分簇选代表和 Elastic Net 非零系数筛选。所有筛选统计均在各次训练窗口重新估计，参数选择使用训练期内部的按季度时间验证。

HAC 指层次凝聚聚类。先计算每天截面的因子 Spearman 相关矩阵，跨日等权平均，再构造距离 `1 - abs(mean_daily_correlation)`；使用平均连接法和指定簇数，每簇按训练期绝对平均 RankIC 选择一个代表，季度方向一致率作为平局规则和稳定性诊断。该方法检验冗余控制，聚类本身不使用收益标签判断预测能力。参考 [scikit-learn 的相关特征聚类示例](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html)、[层次聚类参数说明](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)和 [SciPy cut_tree](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.cut_tree.html)。本研究使用平均连接法，不将相关性距离直接套入要求欧氏结构的 Ward 方法。

Elastic Net 在训练窗口内完成缺失值填补、特征及目标标准化，再联合估计 20 个因子系数；保留系数非零的原因子，交给下游模型重新训练。正则化参数根据下游模型在内部时间验证中的日均 RankIC 选择，筛选器自身的拟合分数不作为最终选择依据。空因子集或未收敛情况会明确记录为不可用，不自动替换为其他方案。参考 [ElasticNet 的目标函数和参数](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)与 [模型驱动的特征筛选](https://scikit-learn.org/stable/modules/feature_selection.html#feature-selection-using-selectfrommodel)。

HAC 和 Elastic Net 另分别配置同数量的 RankIC 筛选对照，以区分筛选规则与特征数量的影响。辅助对照不参与最终流程选择。2024 年此前已经被研究过，扩展实验记录为追加历史研究。

## 行业和市值控制下的预测能力检验

本轮以年度全 20 因子 Ridge 为主要对象、HGB 为对照，沿用 2023 与 2024 年的训练和评价划分。共同特征基准仅使用行业和 `log(market_cap)`，行业编码与特征缩放在训练集拟合；方法参考 [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)和[预处理中的数据泄漏约束](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)。Ridge 保留全部行业哑变量并对整个设计矩阵做训练期标准化，参数与前轮相同。基准之间比较预测表现，不将其差值解释为联合模型中某组变量的净贡献。

逐日的控制后排序指标采用部分秩相关：先对预测、未来五日收益和市值取平均秩，再分别将预测秩与收益秩投影到相同控制变量上，最后计算两组残差的 Pearson 相关。控制变量分别为行业、规模排名、行业加规模排名，均包含截距；行业采用去掉一个基准列的哑变量编码。方法含义参考[部分相关的残差定义](https://pingouin-stats.org/generated/pingouin.partial_corr.html)，实际投影使用 [NumPy 最小二乘](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html)，未新增统计库依赖。收益秩投影仅用于事后评价，不进入训练或预测。

该指标控制的是行业截距及规模排名的线性可加关系；不将相关变化解释为收益归因比例。原始分数对行业和对数市值投影的解释度另作暴露诊断。残差数值退化时指标记为未定义，配对比较使用共同有效日期，区间沿用 20 个交易日移动区块、2,000 次重采样。风险表未单独提供发布时间，本轮明确采用行业和市值在对应信号日盘后可知的研究假设。

## 固定调仓方法比较

本轮借鉴个人既有多因子项目的调仓思想：`multifactor_account.py` 中的 `RebalancePolicy`（排名与仓位缓冲），以及 `multifactor_portfolio.py` 中的 `smooth_target_weights`、`generate_tapered_targets`、`generate_score_weighted_targets`。这些是个人研究代码的规则来源，本题的 `rebalance_ledger.py` 按日频、做多现金账户重新实现；运行不依赖原项目、原行情或原永续合约账户。

迁移时固定前 10% 入场、前 20% 续持，仓位缓冲改为目标权重的上下 10% 相对范围；保留临界同分，不按资产编号取舍。EMA 使用固定 0.5，按每个资金份的复核时点更新。梯形权重和分数权重使用与 NET 相同的入选范围，后者使用正的截面去均值分数，并设 `2/入选数` 的单资产上限。各规则在读取新账户结果前固定，完整协议与解释见[调仓研究报告](rebalance_study_report.md)。

净额执行保留当期成员、盘后计划、次日开盘、前日权限与成本以及退出估值约定，并将连续入选的仓位保留，仅交易目标与实际数量差额。排名缓冲使用真实已成交仓位，避免把未成交目标当成既有持仓。权重缓冲在信号收盘判断，目标退出及期末零目标优先于缓冲和平滑。

[Boyd 等的多期交易框架](https://web.stanford.edu/~boyd/papers/cvx_portfolio.html)提供了将预测与组合构建分开、共同考虑收益、风险和交易成本的方法背景。本轮只比较上述固定规则，没有实现该论文的凸优化求解器；分数比例配置也不代表已完成收益预测校准。

## 在行业、市值基础上加入因子的直接增量实验

本轮固定比较 `CONTROLS = 行业哑变量 + log(市值)` 与 `JOINT = 行业哑变量 + log(市值) + 20 个 PIT 因子`。Ridge 为主模型，HGB 为对照；年度训练样本、标签成熟边界和模型参数沿用既有方案。控制模型和完整成员预测直接复用并核验，新增四个联合模型。因子缺失值使用训练期中位数，Ridge 对包括行业哑变量在内的整个设计矩阵使用训练期标准化；不搜索新参数或因子名单。

交易使用相同 NET、BOTH 规则，并固定一个 `BUDGET20` 收盘计划预算：每份资金每五日检查时，设实际收盘资产权重为 `w`、NET 理想权重为 `u`，对全部目标与实际持仓的并集计算 `D = sum(abs(u - w))`。非终末时采用 `lambda = min(1, 0.20 / D)`，有效目标为 `w + lambda * (u - w)`；`D = 0` 时不缩放。初始建仓受预算限制，期末清仓明确豁免。这是一个预先固定的确定性规则，没有引入组合优化器。

该限制作用于收盘计划的双边权重变化。次日开盘仍沿用冻结权重、开盘预算和股数换算，开盘跳空、费用缩放、受阻卖单重试都可能使实际换手不同。相同计划预算不代表相同实际成交量；共同的做多、不借款和选股规则也不代表行业、市值或波动完全匹配，因此另报告实际换手、现金、集中度和风险敞口，不把收益差直接解释成风险中性的独立 alpha。

主交易比较固定为 2024 年题包成本下的 BUDGET20 Ridge 联合模型减控制模型；NET/BOTH、HGB、2023 年与其他成本情景为对照。日收益及日对数相对增长的差值采用相同日期配对、20 日移动区块、2,000 次重采样。结果属于已查看过的历史合成数据，方法及结果见[联合模型增量报告](joint_factor_increment_report.md)。
